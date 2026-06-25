
# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib
import cumm.core_cc
import argparse
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch, resource
import torch.distributed as dist
torch.multiprocessing.set_sharing_strategy('file_system')
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
# Prefer the current checkout over any editable install left in the environment.
sys.path.insert(0, str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils, train_utils
from opencood.tools.temporal_utils import (
    get_temporal_training_cfg,
    is_mambafusion_model,
    populate_temporal_fields,
    update_temporal_state,
)

import torch.utils.checkpoint as _cp
import traceback

_old_ckpt = _cp.checkpoint

def checkpoint_trace(function, *args, **kwargs):
    # 只要没显式传 use_reentrant，就说明是你现在 warning 的来源
    if "use_reentrant" not in kwargs:
        print("\n[CKPT WARNING SOURCE] checkpoint called WITHOUT use_reentrant. stack:")
        traceback.print_stack(limit=30)
        # 强制变成 False（避免默认 reentrant）
        kwargs["use_reentrant"] = False
    return _old_ckpt(function, *args, **kwargs)

_cp.checkpoint = checkpoint_trace


def autocast_context(device: torch.device, enabled: bool):
    """Return a non-deprecated autocast context for the active device."""
    use_amp = enabled and device.type == "cuda"
    return torch.amp.autocast(device_type=device.type, enabled=use_amp)

def train_parser():
    """
    Configure command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="OpenCOOD Training")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                      help="Path to training configuration yaml file")
    parser.add_argument("--model_dir", default="",
                      help="Path to continue training from a checkpoint")
    parser.add_argument("--dist_url", default="env://",
                      help="URL used to set up distributed training")
    parser.add_argument("--fusion_method", "-f", default="intermediate",
                      help="Fusion method to use during inference")
    parser.add_argument("--rank", default=0, type=int,
                      help="Node rank for distributed training")
    parser.add_argument("--tag", default="default",
                      help="Tag for the training session")
    parser.add_argument("--worker", default=0, type=int,
                      help="Number of workers for data loading")
    parser.add_argument("--amp", action="store_true",
                      help="Enable automatic mixed precision training")
    parser.add_argument("--gpu_id",default=0)
    return parser.parse_args()


def find_latest_checkpoint(model_dir: str) -> str:
    """Find the latest checkpoint by epoch number in a directory."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {model_dir}"
        )

    checkpoint_pattern = re.compile(r"^net_epoch(\d+)\.pth$")
    checkpoint_candidates = []

    for file_name in os.listdir(model_dir):
        match = checkpoint_pattern.fullmatch(file_name)
        if match:
            checkpoint_candidates.append(
                (int(match.group(1)), os.path.join(model_dir, file_name))
            )

    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No checkpoint matching 'net_epoch*.pth' found in directory: {model_dir}"
        )

    checkpoint_candidates.sort(key=lambda item: item[0])
    return checkpoint_candidates[-1][1]


def _get_checkpoint_value(checkpoint: Dict[str, Any],
                          keys: List[str],
                          error_message: str) -> Any:
    """Get a checkpoint field by trying multiple compatible key names."""
    for key in keys:
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError(error_message)


def _adapt_state_dict_for_model(model: torch.nn.Module,
                                state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Align checkpoint keys with the current model's module prefix layout."""
    if not state_dict:
        raise KeyError("Checkpoint model state_dict is empty.")

    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())
    if not model_keys or not state_keys:
        return state_dict

    model_has_module_prefix = model_keys[0].startswith("module.")
    state_has_module_prefix = state_keys[0].startswith("module.")

    if model_has_module_prefix and not state_has_module_prefix:
        return {
            f"module.{key}" if not key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    if state_has_module_prefix and not model_has_module_prefix:
        return {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    return state_dict


def _get_model_state_dict_for_save(model: torch.nn.Module) -> Dict[str, Any]:
    """Save unwrapped model weights so single-card and DDP resumes share format."""
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def resume_training_from_checkpoint(saved_path: str,
                                    model: torch.nn.Module,
                                    optimizer: torch.optim.Optimizer,
                                    scheduler: Optional[Any] = None,
                                    scaler: Optional[Any] = None,
                                    device: torch.device = torch.device("cpu")) -> Tuple[int, str]:
    """Restore full training state from the latest checkpoint in a directory."""
    checkpoint_path = find_latest_checkpoint(saved_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' must be a dict, got {type(checkpoint)}."
        )

    epoch = _get_checkpoint_value(
        checkpoint,
        ["epoch"],
        f"Checkpoint '{checkpoint_path}' is missing required field 'epoch'.",
    )
    model_state_dict = _get_checkpoint_value(
        checkpoint,
        ["model_state_dict", "state_dict", "model"],
        f"Checkpoint '{checkpoint_path}' is missing required field 'model_state_dict'.",
    )
    optimizer_state_dict = _get_checkpoint_value(
        checkpoint,
        ["optimizer_state_dict", "optimizer"],
        (
            f"Checkpoint '{checkpoint_path}' is missing required field "
            "'optimizer_state_dict', cannot resume full training state."
        ),
    )

    if scheduler is not None:
        scheduler_state_dict = _get_checkpoint_value(
            checkpoint,
            ["scheduler_state_dict", "scheduler"],
            (
                f"Checkpoint '{checkpoint_path}' is missing required field "
                "'scheduler_state_dict', cannot resume scheduler state."
            ),
        )
    else:
        scheduler_state_dict = None

    # 对齐 DP/DDP 前缀，并只加载当前模型里存在且 shape 匹配的权重
    ckpt_state = _adapt_state_dict_for_model(model, model_state_dict)
    model_state = model.state_dict()
    filtered_state = {
        k: v for k, v in ckpt_state.items()
        if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
    }

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)

    if missing_keys:
        print("Warning: missing keys when loading checkpoint (ignored):")
        print("  " + "\n  ".join(missing_keys))
    if unexpected_keys:
        print("Warning: unexpected keys in checkpoint (ignored):")
        print("  " + "\n  ".join(unexpected_keys))

    try:
        optimizer.load_state_dict(optimizer_state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to restore optimizer_state_dict from '{checkpoint_path}': {exc}"
        ) from exc

    if scheduler is not None:
        try:
            scheduler.load_state_dict(scheduler_state_dict)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore scheduler_state_dict from '{checkpoint_path}': {exc}"
            ) from exc

    scaler_state_dict = checkpoint.get("scaler_state_dict", checkpoint.get("scaler"))
    if scaler is not None and scaler_state_dict is not None:
        try:
            scaler.load_state_dict(scaler_state_dict)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore scaler_state_dict from '{checkpoint_path}': {exc}"
            ) from exc

    init_epoch = int(epoch) + 1
    print(f"Resumed training from checkpoint: {checkpoint_path}")
    print(f"Resuming at epoch: {init_epoch}")
    return init_epoch, checkpoint_path

def setup_dataloader(dataset, hypes, opt, is_train=True):
    """
    Set up data loader for training or validation.
    
    Args:
        dataset: Dataset instance
        hypes (dict): Configuration parameters
        opt: Command line arguments
        is_train (bool): Whether this is for training or validation
        
    Returns:
        DataLoader: Configured data loader
    """
    temporal_cfg = get_temporal_training_cfg(hypes)
    batch_size = hypes["train_params"]["batch_size"]
    shuffle = is_train
    if temporal_cfg["enable"]:
        if batch_size != 1:
            raise ValueError(
                "temporal_training requires train_params.batch_size=1 because "
                "SparseTemporalFusionBlock keeps a single history cache."
            )
        # queue 可以打乱当前帧 anchor；streaming 必须保持顺序。
        shuffle = is_train and temporal_cfg["mode"] == "queue"

    if opt.distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        if is_train:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last=True)
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=opt.worker,
                collate_fn=dataset.collate_batch_train,
                timeout=5200,
                pin_memory=True,
                prefetch_factor=1
            )
        else:
            loader = DataLoader(
                dataset,
                sampler=sampler,
                num_workers=opt.worker,
                collate_fn=dataset.collate_batch_train,
                timeout=5200,
                pin_memory=True,
                prefetch_factor=1
            )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=opt.worker,
            collate_fn=dataset.collate_batch_train,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=True
            #prefetch_factor=None
        )
    return loader


def is_main_process(opt: argparse.Namespace) -> bool:
    return (not getattr(opt, "distributed", False)) or opt.rank == 0


def _temporal_checkpoint_path_and_ckpt_state(
    model: torch.nn.Module,
    load_from: str,
    device: torch.device,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve baseline path and return adapted checkpoint tensors (shared by load vs. mask-only)."""
    checkpoint_path = find_latest_checkpoint(load_from) if os.path.isdir(load_from) else load_from
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"temporal_training.load_from not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint '{checkpoint_path}' must be a dict.")
    try:
        state_dict = _get_checkpoint_value(
            checkpoint,
            ["model_state_dict", "state_dict", "model", "model_state"],
            f"Checkpoint '{checkpoint_path}' does not contain model weights.",
        )
    except KeyError:
        state_dict = checkpoint
    ckpt_state = _adapt_state_dict_for_model(model, state_dict)
    return checkpoint_path, ckpt_state


def collect_temporal_loaded_parameter_names(
    model: torch.nn.Module,
    load_from: str,
    device: torch.device,
) -> Set[str]:
    """Match baseline keys/shapes like ``load_temporal_pretrain`` but do not copy weights."""
    checkpoint_path, ckpt_state = _temporal_checkpoint_path_and_ckpt_state(
        model, load_from, device
    )
    model_state = model.state_dict()
    named_parameters = dict(model.named_parameters())
    loaded_parameter_names: Set[str] = set()
    matched_tensors = 0
    with torch.no_grad():
        for key, value in ckpt_state.items():
            if (
                key not in model_state
                or not hasattr(value, "shape")
                or value.shape != model_state[key].shape
            ):
                continue
            matched_tensors += 1
            if key in named_parameters:
                loaded_parameter_names.add(key)
    missing_keys = [key for key in model_state if key not in ckpt_state]
    unexpected_keys = [key for key in ckpt_state if key not in model_state]
    print(
        f"Temporal baseline match (no weight copy) from {checkpoint_path}: "
        f"{matched_tensors} tensors shape-matched, "
        f"{len(loaded_parameter_names)} parameters masked for freeze."
    )
    if missing_keys:
        print(f"Temporal baseline missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Temporal baseline unexpected keys: {len(unexpected_keys)}")
    return loaded_parameter_names


def load_temporal_pretrain(model: torch.nn.Module,
                           load_from: str,
                           device: torch.device) -> Set[str]:
    """Load shape-compatible pretraining weights and return matched parameter names."""
    checkpoint_path, ckpt_state = _temporal_checkpoint_path_and_ckpt_state(
        model, load_from, device
    )
    model_state = model.state_dict()
    named_parameters = dict(model.named_parameters())
    loaded_parameter_names: Set[str] = set()
    matched_tensors = 0

    with torch.no_grad():
        for key, value in ckpt_state.items():
            if (
                key not in model_state
                or not hasattr(value, "shape")
                or value.shape != model_state[key].shape
            ):
                continue

            target = model_state[key]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            matched_tensors += 1
            if key in named_parameters:
                loaded_parameter_names.add(key)

    missing_keys = [key for key in model_state if key not in ckpt_state]
    unexpected_keys = [key for key in ckpt_state if key not in model_state]
    print(
        f"Loaded temporal pretrain from {checkpoint_path}: "
        f"{matched_tensors} tensors matched, "
        f"{len(loaded_parameter_names)} parameters will be frozen."
    )
    if missing_keys:
        print(f"Temporal pretrain missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Temporal pretrain unexpected keys: {len(unexpected_keys)}")
    return loaded_parameter_names


def _is_detection_head_parameter(name: str) -> bool:
    """Detection heads on ``Airv2xMambafusion``: ``cls_head`` / ``reg_head`` / ``obj_head``."""
    if name.startswith("module."):
        name = name[len("module.") :]
    return name.startswith(("cls_head", "reg_head", "obj_head"))


def freeze_loaded_parameters(model: torch.nn.Module,
                             loaded_parameter_names: Set[str]) -> None:
    """Freeze parameters that were successfully loaded from a checkpoint.

    Detection heads remain trainable so they can adapt if BEV distributions shift.
    """
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if (
            name in loaded_parameter_names
            and not _is_detection_head_parameter(name)
        ):
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters remain after freezing loaded checkpoint weights."
        )
    print(
        "Checkpoint freeze summary: "
        f"trainable={trainable:,}, frozen={frozen:,}"
    )


def forward_mambafusion_temporal_queue(model: torch.nn.Module,
                                       batch_data: Dict[str, Any],
                                       criterion: torch.nn.Module,
                                       device: torch.device,
                                       epoch: int,
                                       scaler: Optional[Any]) -> torch.Tensor:
    """Run prev/current queue; only the current frame contributes loss."""
    queue = batch_data["ego"]["temporal_queue"]
    if len(queue) != 2:
        raise ValueError(f"Expected two-frame temporal queue, got {len(queue)} frames")

    cur_frame = train_utils.to_device(queue[1], device)
    cur_frame["epoch"] = epoch

    prev_valid_list = batch_data["ego"].get("temporal_prev_valid", [True])
    prev_valid = bool(prev_valid_list[0])
    prev_frame = None
    
    if prev_valid:
        prev_frame = train_utils.to_device(queue[0], device)
        prev_frame["epoch"] = epoch
        populate_temporal_fields(prev_frame, device, prev_frame=None, force_reset=True)
        with torch.no_grad():
            with autocast_context(device, enabled=scaler is not None):
                model(prev_frame)

    populate_temporal_fields(
        cur_frame,
        device,
        prev_frame=prev_frame,
        force_reset=not prev_valid,
    )
    with autocast_context(device, enabled=scaler is not None):
        output_dict = model(cur_frame)
        loss = criterion(output_dict, cur_frame["label_dict"])
    return loss


def forward_mambafusion_streaming(model: torch.nn.Module,
                                  batch_data: Dict[str, Any],
                                  criterion: torch.nn.Module,
                                  device: torch.device,
                                  epoch: int,
                                  scaler: Optional[Any],
                                  temporal_state: Dict[str, Any]) -> torch.Tensor:
    """Run one streaming frame and update temporal metadata state."""
    batch_data = train_utils.to_device(batch_data, device)
    frame = batch_data["ego"]
    frame["epoch"] = epoch
    prev_frame = temporal_state.get("prev_frame")
    populate_temporal_fields(frame, device, prev_frame=prev_frame)

    with autocast_context(device, enabled=scaler is not None):
        output_dict = model(frame)
        loss = criterion(output_dict, frame["label_dict"])

    update_temporal_state(temporal_state, frame)
    return loss

def validate_model(model, val_loader, criterion, epoch,
                   device, hypes, scaler=None):
    model.eval()
    total_loss, n_sample = 0.0, 0
    show_progress = multi_gpu_utils.get_dist_info()[0] == 0
    temporal_cfg = get_temporal_training_cfg(hypes)
    temporal_state: Dict[str, Any] = {}
    with torch.no_grad():
        for _, batch_data in tqdm(enumerate(val_loader),
                                  total=len(val_loader),
                                  desc="Validation", leave=False,
                                  disable=not show_progress):
            if batch_data is None:
                continue
            if "scope" in hypes["name"] or "how2comm" in hypes["name"]:
                _batch_data = train_utils.to_device(batch_data[0], device)
                batch_data  = train_utils.to_device(batch_data,   device)
                with autocast_context(device, enabled=scaler is not None):
                    out = model(batch_data)
                    loss = criterion(out, _batch_data["ego"]["label_dict"])
            elif is_mambafusion_model(hypes):
                # MambaFusion 特殊处理 - 使用AirV2X的检测loss
                if temporal_cfg["enable"]:
                    loss = forward_mambafusion_streaming(
                        model, batch_data, criterion, device, epoch, scaler, temporal_state
                    )
                else:
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data["ego"]["epoch"] = epoch
                    with autocast_context(device, enabled=scaler is not None):
                        out = model(batch_data["ego"])
                        # 使用AirV2X标准的检测loss计算
                        loss = criterion(out, batch_data["ego"]["label_dict"])
            else:
                batch_data = train_utils.to_device(batch_data, device)
                batch_data["ego"]["epoch"] = epoch
                with autocast_context(device, enabled=scaler is not None):
                    out  = model(batch_data["ego"])
                    loss = criterion(out, batch_data["ego"]["label_dict"])

            bsz = 1 if isinstance(loss, torch.Tensor) else len(loss)
            total_loss += loss.item() * bsz   
            n_sample   += bsz

    return total_loss, n_sample

def main():
    """Main training function."""
    # Setup
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    temporal_cfg = get_temporal_training_cfg(hypes)
    multi_gpu_utils.init_distributed_mode(opt)
    hypes["tag"] = opt.tag
    main_process = is_main_process(opt)
    print("load from yaml file: ", opt.hypes_yaml)
    # Build datasets
    print("Building datasets...")
    # 对于MambaFusion模型，需要启用visualize模式以包含origin_lidar等字段
    visualize_mode = hypes.get("visualize", False)
   
    train_dataset = build_dataset(hypes, visualize=visualize_mode, train=True)
    val_dataset = build_dataset(hypes, visualize=visualize_mode, train=False)
    
    # Create dataloaders
    train_loader = setup_dataloader(train_dataset, hypes, opt, is_train=True)
    val_loader = setup_dataloader(val_dataset, hypes, opt, is_train=False)
    
    # Create model
    print("Creating model...")
    # 检查是否是 MambaFusion 模型
    if is_mambafusion_model(hypes):
        # MambaFusion 需要特殊的模型创建方式，传入训练数据集
        from opencood.models.airv2x_mambafusion import Airv2xMambafusion
        model = Airv2xMambafusion(hypes["model"]["args"], dataset=train_dataset)
        print("Created MambaFusion model")
    else:
        model = train_utils.create_model(hypes)
    total_params = sum(p.nelement() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    if opt.distributed:
        # init_distributed_mode 已经根据 LOCAL_RANK 设置好了 opt.gpu / opt.rank / world_size
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        device = torch.device(f"cuda:{opt.gpu}")
        torch.cuda.set_device(opt.gpu)
    else:
        if torch.cuda.is_available():
            gpu_id = int(opt.gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device("cpu")

    model.to(device)
    if temporal_cfg["enable"]:
        if temporal_cfg["load_from"] and not opt.model_dir:
            loaded_parameter_names = load_temporal_pretrain(
                model, temporal_cfg["load_from"], device
            )
            freeze_loaded_parameters(model, loaded_parameter_names)
        elif temporal_cfg["load_from"] and opt.model_dir:
            print(
                "Skipping temporal weight copy from load_from; after --model_dir resume "
                "the baseline freeze mask from temporal_training.load_from will be applied."
            )

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            find_unused_parameters=True
        )
       

    
    # Setup training components
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    
    # Initialize mixed precision training if enabled
    scaler = torch.amp.GradScaler("cuda") if opt.amp and torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training")

    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, n_iter_per_epoch=len(train_loader))
    
    # Load checkpoint if continuing training
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, _ = resume_training_from_checkpoint(
            saved_path=saved_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if temporal_cfg["enable"] and temporal_cfg["load_from"].strip():
            mask_names = collect_temporal_loaded_parameter_names(
                model, temporal_cfg["load_from"].strip(), device
            )
            freeze_loaded_parameters(model, mask_names)
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes) if main_process else None
        if opt.distributed:
            saved_path_holder = [saved_path]
            dist.broadcast_object_list(saved_path_holder, src=0)
            saved_path = saved_path_holder[0]
        if main_process:
            print(f"Results will be saved to: {saved_path}")
    assert saved_path is not None

    # Training loop
    writer = SummaryWriter(saved_path) if main_process else None
    print("Starting training...")
    epochs = hypes["train_params"]["epoches"]
    
    for epoch in range(init_epoch, epochs):
        if opt.distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr}")
        
        # Training epoch
        model.train()
        temporal_state: Dict[str, Any] = {}
        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    disable=not main_process)
        for i, batch_data in pbar:
            if batch_data is None:
                continue
            '''
            print("batch_data: ", batch_data.keys())
            print("batch_data['ego']: ", batch_data["ego"].keys())
            print("batch_data['ego']['origin_lidar']: ", batch_data["ego"]["origin_lidar"].shape)
            # 打印batch_data['ego']['origin_lidar']xyz坐标分别的最大最小值
            print("batch_data['ego']['origin_lidar']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar"][..., 0]).item(), torch.min(batch_data["ego"]["origin_lidar"][..., 0]).item())
            print("batch_data['ego']['origin_lidar']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar"][..., 1]).item(), torch.min(batch_data["ego"]["origin_lidar"][..., 1]).item())
            print("batch_data['ego']['origin_lidar']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar"][..., 2]).item(), torch.min(batch_data["ego"]["origin_lidar"][..., 2]).item())
            print("batch_data['ego']['origin_lidar_rsu']: ", batch_data["ego"]["origin_lidar_rsu"].shape)
            print("batch_data['ego']['origin_lidar_drone']: ", batch_data["ego"]["origin_lidar_drone"].shape)
            print("batch_data['ego']['origin_lidar_rsu']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar_rsu"][..., 0]).item(), torch.min(batch_data["ego"]["origin_lidar_rsu"][..., 0]).item())
            print("batch_data['ego']['origin_lidar_rsu']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar_rsu"][..., 1]).item(), torch.min(batch_data["ego"]["origin_lidar_rsu"][..., 1]).item())
            print("batch_data['ego']['origin_lidar_rsu']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar_rsu"][..., 2]).item(), torch.min(batch_data["ego"]["origin_lidar_rsu"][..., 2]).item())
            print("batch_data['ego']['origin_lidar_drone']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar_drone"][..., 0]).item(), torch.min(batch_data["ego"]["origin_lidar_drone"][..., 0]).item())
            print("batch_data['ego']['origin_lidar_drone']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar_drone"][..., 1]).item(), torch.min(batch_data["ego"]["origin_lidar_drone"][..., 1]).item())
            print("batch_data['ego']['origin_lidar_drone']xyz坐标分别的最大最小值: ", torch.max(batch_data["ego"]["origin_lidar_drone"][..., 2]).item(), torch.min(batch_data["ego"]["origin_lidar_drone"][..., 2]).item())
            print("batch_data['ego']['label_dict']: ", batch_data["ego"]["label_dict"].keys())
            print("batch_data['ego']['label_dict']['dynamic_seg_label']: ", batch_data["ego"]["label_dict"]["dynamic_seg_label"].shape)
            print("batch_data['ego']['label_dict']['static_seg_label']: ", batch_data["ego"]["label_dict"]["static_seg_label"].shape)
            print("batch_data['ego']['label_dict']['dynamic_seg_label_bev']: ", batch_data["ego"]["label_dict"]["dynamic_seg_label_bev"].shape)
            print("batch_data['ego']['label_dict']['static_seg_label_bev']: ", batch_data["ego"]["label_dict"]["static_seg_label_bev"].shape)
            '''
            # Forward pass
            model.zero_grad()

            optimizer.zero_grad()
            if "scope" in hypes["name"] : #or "how2comm" in hypes["name"]
                _batch_data = batch_data[0]
                batch_data = train_utils.to_device(batch_data, device)
                _batch_data = train_utils.to_device(_batch_data, device)
                
                with autocast_context(device, enabled=scaler is not None):
                    output_dict = model(batch_data)
                    loss = criterion(output_dict, _batch_data["ego"]["label_dict"])
            elif is_mambafusion_model(hypes):
                # MambaFusion 特殊处理 - 使用AirV2X的检测loss
                if temporal_cfg["enable"] and temporal_cfg["mode"] == "queue":
                    loss = forward_mambafusion_temporal_queue(
                        model, batch_data, criterion, device, epoch, scaler
                    )
                elif temporal_cfg["enable"] and temporal_cfg["mode"] == "streaming":
                    loss = forward_mambafusion_streaming(
                        model, batch_data, criterion, device, epoch, scaler, temporal_state
                    )
                else:
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data["ego"]["epoch"] = epoch
                    
                    with autocast_context(device, enabled=scaler is not None):
                        output_dict = model(batch_data["ego"])
                        # 使用AirV2X标准的检测loss计算
                        loss = criterion(output_dict, batch_data["ego"]["label_dict"])
            else:
                batch_data = train_utils.to_device(batch_data, device)
                batch_data["ego"]["epoch"] = epoch
                
                with autocast_context(device, enabled=scaler is not None):
                    output_dict = model(batch_data["ego"])
                    loss = criterion(output_dict, batch_data["ego"]["label_dict"])
                
                # 前向传播后也清理一下显存
                torch.cuda.empty_cache()
            torch.autograd.set_detect_anomaly(True)
            # Backward pass with mixed precision support
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update progress bar
            print_msg = None
            if main_process:
                assert writer is not None
                try:
                    print_msg = criterion.logging(epoch, i, len(train_loader), writer, pbar)
                except Exception:
                    print_msg = criterion.logging(epoch, i, len(train_loader), writer)
                if print_msg:
                    pbar.set_description(print_msg)
            
            # Log training loss
            if main_process:
                with open(os.path.join(saved_path, "train_loss.txt"), "a+") as f:
                    f.write(f"Epoch[{epoch}], iter[{i}/{len(train_loader)}], loss[{loss.item():.4f}]\n")
                    
        if opt.distributed:
            torch.cuda.synchronize()             
            torch.distributed.barrier()  
        
        # Save checkpoint
        if opt.rank == 0 and epoch % hypes["train_params"]["save_freq"] == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': _get_model_state_dict_for_save(model),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(save_dict, os.path.join(saved_path, f"net_epoch{epoch + 1}.pth"))
        
        # Validation
        need_val = (epoch % hypes["train_params"]["eval_freq"] == 0)

        if need_val:
            if opt.distributed and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            local_sum, local_cnt = validate_model(model, val_loader,
                                                criterion, epoch,
                                                device, hypes, scaler)

            if opt.distributed:
                stats = torch.tensor([local_sum, local_cnt],
                                    dtype=torch.float32, device=device)
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                global_sum, global_cnt = stats.tolist()
            else:
                global_sum, global_cnt = local_sum, local_cnt

            if main_process:
                assert writer is not None
                val_loss = global_sum / max(global_cnt, 1)
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
                writer.add_scalar("Validate_Loss", val_loss, epoch)
                with open(os.path.join(saved_path, "validation_loss.txt"), "a+") as f:
                    f.write(f"Epoch[{epoch}], loss[{val_loss:.4f}]\n")

        
        scheduler.step(epoch)
                
        if opt.distributed:
            torch.distributed.barrier(device_ids=[opt.gpu])
        
    
    torch.cuda.empty_cache()
    if writer is not None:
        writer.close()

    if opt.distributed:
        dist.barrier()
        dist.destroy_process_group()

    if main_process:
        print(f"Training finished. Checkpoints saved to {saved_path}")
        # Run inference after training only once after DDP teardown.
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running inference: {cmd}")
        os.system(cmd)

if __name__ == "__main__":
    main()
