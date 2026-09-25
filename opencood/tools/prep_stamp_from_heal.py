#!/usr/bin/env python3
"""Prepare STAMP collab run from a HEAL collab checkpoint.

Creates a model_dir with:
  - config.yaml (from heal_adapter hypes)
  - net_epoch20.pth: HEAL weights + near-identity ConvNeXt adapters (train start)
  - anchor_heal_identity.pth: HEAL weights + Identity adapters (AP floor / rollback)

Usage:
  python opencood/tools/prep_stamp_from_heal.py \\
    --heal_ckpt opencood/logs/airv2x_HEAL_collab_cam_lidar/heal_collab_fixed_2026_07_22_16_47_05/net_epoch20.pth \\
    --hypes_yaml opencood/hypes_yaml/airv2x/camera_lidar/det/airv2x_stamp/airv2x_stamp_collab_cam_lidar_heal_adapter.yaml \\
    --out_dir opencood/logs/airv2x_stamp_collab_cam_lidar/stamp_heal_adapter_2026_07_27
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from typing import Any, Dict

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-3])
sys.path.insert(0, root_path)

import torch
import torch.nn as nn

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils


def _init_conv_identity(conv: nn.Conv2d) -> None:
    """Initialize 1x1/3x3 conv to (approximate) identity."""
    with torch.no_grad():
        conv.weight.zero_()
        out_c, in_c, _, _ = conv.weight.shape
        k = min(out_c, in_c)
        center = conv.weight.shape[-1] // 2
        for i in range(k):
            conv.weight[i, i, center, center] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def init_adapters_near_identity(model: nn.Module) -> None:
    """Make ConvNeXt adapters start close to identity (preserve HEAL AP).

    ConvNeXt blocks already use tiny gamma residual scale; we force gamma→0 and
    set channel 1x1 convs to identity so adapter(x) ≈ x at step 0.
    """
    for name in ("adapter_rsu", "adapter_drone", "adapter_vehicle"):
        if not hasattr(model, name):
            continue
        adapter_wrap = getattr(model, name)
        inner = getattr(adapter_wrap, "adapter", None)
        if inner is None:
            continue
        # Identity adapter has no learnable map beyond upsample.
        if inner.__class__.__name__ == "AdapterIdentity":
            print(f"[prep] {name}: Identity (no init needed)")
            continue
        if hasattr(inner, "channel_convert1"):
            _init_conv_identity(inner.channel_convert1)
        if hasattr(inner, "channel_convert2"):
            _init_conv_identity(inner.channel_convert2)
        if hasattr(inner, "smoothing"):
            _init_conv_identity(inner.smoothing)
        # Zero residual scales inside ConvNeXt blocks.
        n_gamma = 0
        for mod in inner.modules():
            if hasattr(mod, "gamma") and isinstance(mod.gamma, nn.Parameter):
                with torch.no_grad():
                    mod.gamma.zero_()
                n_gamma += 1
        print(f"[prep] {name}: near-identity init (gamma_zeroed={n_gamma})")


def load_heal_into_stamp(model: nn.Module, heal_ckpt: str) -> None:
    """Load HEAL state_dict into STAMP; leave STAMP-only keys (adapters) intact."""
    state = torch.load(heal_ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # strip possible module. prefix
    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    miss_adapt = [k for k in missing if "adapter" in k]
    miss_other = [k for k in missing if "adapter" not in k]
    print(
        f"[prep] loaded HEAL → STAMP | missing={len(missing)} "
        f"(adapter={len(miss_adapt)}, other={len(miss_other)}) "
        f"unexpected={len(unexpected)}"
    )
    if miss_other:
        print("[prep] WARNING non-adapter missing keys:", miss_other[:10])


def count_trainable(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    groups: Dict[str, int] = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            g = n.split(".")[0]
            groups[g] = groups.get(g, 0) + p.numel()
    print(
        f"[prep] trainable {trainable}/{total} "
        f"({100.0 * trainable / max(total, 1):.2f}%) groups={groups}"
    )


def build_identity_hypes(hypes: Dict[str, Any]) -> Dict[str, Any]:
    """Clone hypes and force all agent adapters to Identity."""
    h = copy.deepcopy(hypes)
    for agent in ("vehicle", "rsu", "drone"):
        if agent not in h["model"]["args"]:
            continue
        ad = h["model"]["args"][agent].get("adapter")
        if ad is None:
            continue
        ad["core_method"] = "identity"
        # Identity only needs base args; drop submodule_args if present.
        ad.get("args", {}).pop("submodule_args", None)
    # Skip freeze during build for cleaner save; freeze happens at train time.
    h["model"]["args"]["backbone_fix"] = False
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep STAMP from HEAL collab ckpt")
    parser.add_argument(
        "--heal_ckpt",
        type=str,
        required=True,
        help="Path to HEAL net_epochXX.pth",
    )
    parser.add_argument(
        "--hypes_yaml",
        type=str,
        required=True,
        help="STAMP heal_adapter yaml",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output model_dir for train_stamp.py --model_dir",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=20,
        help="Saved as net_epoch{N}.pth so resume continues from N",
    )
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    cfg_dst = os.path.join(opt.out_dir, "config.yaml")
    shutil.copy2(opt.hypes_yaml, cfg_dst)
    print(f"[prep] wrote {cfg_dst}")

    # ---- Train start ckpt: ConvNeXt adapters, near-identity ----
    hypes = load_yaml(opt.hypes_yaml)
    # Build without freeze first (cleaner init), then apply freeze for sanity check.
    fix_cfg = hypes["model"]["args"].get("backbone_fix", True)
    freeze_ad = bool(hypes["model"]["args"].get("freeze_adapters", False))
    hypes["model"]["args"]["backbone_fix"] = False
    model = train_utils.create_model(hypes)
    load_heal_into_stamp(model, opt.heal_ckpt)
    init_adapters_near_identity(model)
    model.args["freeze_adapters"] = freeze_ad
    if fix_cfg:
        model.backbone_fix(fix_cfg)
    count_trainable(model)

    train_ckpt = os.path.join(opt.out_dir, f"net_epoch{opt.start_epoch}.pth")
    torch.save(model.state_dict(), train_ckpt)
    print(f"[prep] saved train start → {train_ckpt}")

    # ---- Anchor: Identity adapters (rollback / floor) ----
    id_hypes = build_identity_hypes(load_yaml(opt.hypes_yaml))
    id_model = train_utils.create_model(id_hypes)
    load_heal_into_stamp(id_model, opt.heal_ckpt)
    anchor_path = os.path.join(opt.out_dir, "anchor_heal_identity.pth")
    torch.save(id_model.state_dict(), anchor_path)
    id_cfg_path = os.path.join(opt.out_dir, "config_identity_anchor.yaml")
    id_yaml_src = opt.hypes_yaml.replace(
        "heal_adapter.yaml", "heal_identity.yaml"
    )
    if os.path.isfile(id_yaml_src):
        shutil.copy2(id_yaml_src, id_cfg_path)
        print(f"[prep] saved identity config → {id_cfg_path}")
    else:
        print(
            f"[prep] NOTE: create {id_yaml_src} for identity eval; "
            f"anchor weights still at {anchor_path}"
        )
    print(f"[prep] saved identity anchor → {anchor_path}")
    print("[prep] done.")


if __name__ == "__main__":
    main()
