# -*- coding: utf-8 -*-
"""Runtime audit for learned-depth training path (use_gt_depth=False).

Checks on REAL training data + REAL model:
1. depth_head created & requires_grad & in optimizer
2. FINAL OUTPUT KEYS reaching the criterion (depth_items*)
3. Per-component loss values (cls/reg/obj/depth)
4. depth grad flowing back after backward()
5. depth GT index distribution (OOB clamp ratio per agent)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from torch.utils.data import DataLoader


def main():
    model_dir = sys.argv[1]
    device = torch.device("cuda:0")

    hypes = yaml_utils.load_yaml(None, __import__("argparse").Namespace(model_dir=model_dir))
    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=dataset.collate_batch_train, shuffle=True)

    model = train_utils.create_model(hypes).to(device)
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    model.train()

    # ---- load trained checkpoint if any ----
    import glob, re
    ckpts = sorted(glob.glob(os.path.join(model_dir, "net_epoch*.pth")),
                   key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    if ckpts:
        sd = torch.load(ckpts[-1], map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[0] loaded {ckpts[-1]} | missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]: print("  missing sample:", missing[:3])
        if unexpected[:3]: print("  unexpected sample:", unexpected[:3])

    # ---- 1. depth params audit ----
    print("=" * 70)
    print("[1] depth params: created / requires_grad / in optimizer")
    opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    n_depth = 0
    for name, p in model.named_parameters():
        if "depth" in name.lower():
            in_opt = any(p is pp for pp in opt_params)
            print(f"  {name}: req_grad={p.requires_grad} in_opt={in_opt} shape={tuple(p.shape)}")
            n_depth += 1
    print(f"  total depth params: {n_depth}")

    # ---- fetch one batch ----
    batch = next(iter(loader))
    batch = train_utils.to_device(batch, device)
    label_dict = batch["ego"]["label_dict"]

    # ---- 2. forward & output keys ----
    output_dict = model(batch["ego"])
    print("=" * 70)
    print("[2] FINAL OUTPUT KEYS (reaching criterion):")
    for k, v in output_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v)}")

    # ---- 3. per-component loss ----
    loss = criterion(output_dict, label_dict)
    print("=" * 70)
    print("[3] loss components:")
    for k, v in criterion.loss_dict.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL: {loss.item():.4f}")

    # ---- 3b. depth GT index distribution (OOB stats) ----
    print("=" * 70)
    print("[3b] depth GT index distribution per agent:")
    for key in ("depth_items", "depth_items_rsu", "depth_items_drone"):
        if key not in output_dict:
            print(f"  {key}: MISSING")
            continue
        depth_logit, depth_gt_idx = output_dict[key][0], output_dict[key][1]
        D = depth_logit.shape[1]
        hist = torch.bincount(depth_gt_idx.flatten().cpu(), minlength=D).float()
        pct = (hist / hist.sum() * 100).round().tolist()
        print(f"  {key}: logit{tuple(depth_logit.shape)} gt_idx{tuple(depth_gt_idx.shape)} D={D}")
        print(f"    bin%%: {pct}")
        print(f"    last-bin(=OOB clamp)%%: {pct[-1]:.1f} | first-bin%%: {pct[0]:.1f}")

    # ---- 4. backward & depth grads ----
    optimizer.zero_grad()
    loss.backward()
    print("=" * 70)
    print("[4] depth param grads after backward:")
    for name, p in model.named_parameters():
        if "depth" in name.lower():
            g = None if p.grad is None else p.grad.abs().mean().item()
            print(f"  {name}: grad={g}")

    print("=" * 70)
    print("[5] done")


if __name__ == "__main__":
    main()
