"""Save per-agent transmission maps, corrupt maps, overlays and region stats."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_SAVE_EVERY = 50


def _corrupt_tag(corrupt_cfg: Any) -> str:
    if corrupt_cfg is None or not getattr(corrupt_cfg, "enabled", False):
        return "clean"
    return (
        f"{corrupt_cfg.scenario}_{corrupt_cfg.corrupt_type}_{corrupt_cfg.level}"
    )


def _to_hw(x: torch.Tensor) -> torch.Tensor:
    t = x.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        if t.shape[0] == 1:
            t = t[0]
        else:
            t = t.mean(dim=0)
    return t


def _resize_hw_to(src: torch.Tensor, target_hw: torch.Size) -> torch.Tensor:
    """Nearest-resize a 2D map to ``target_hw`` (H, W)."""
    if tuple(src.shape) == tuple(target_hw):
        return src
    return torch.nn.functional.interpolate(
        src.view(1, 1, *src.shape).float(),
        size=tuple(target_hw),
        mode="nearest",
    )[0, 0]


def _map_stats(tx: torch.Tensor) -> Dict[str, float]:
    g = _to_hw(tx)
    return {
        "mean": float(g.mean().item()),
        "std": float(g.std().item()),
        "min": float(g.min().item()),
        "max": float(g.max().item()),
        "ratio_gt_0.5": float((g > 0.5).float().mean().item()),
        "ratio_gt_0.7": float((g > 0.7).float().mean().item()),
        "ratio_gt_0.9": float((g > 0.9).float().mean().item()),
    }


def _region_stats(
    tx: torch.Tensor,
    corrupt_map: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Partition transmission by corrupt region when a binary/intensity map exists."""
    out = {
        "mean_corrupt_region": float("nan"),
        "mean_clean_region": float("nan"),
        "delta_clean_minus_corrupt": float("nan"),
        "corrupt_coverage": float("nan"),
    }
    if corrupt_map is None:
        return out
    g = _to_hw(tx)
    c = _resize_hw_to(_to_hw(corrupt_map), g.shape)
    # Treat >0.5 as corrupted for mask; for gaussian intensity use > median.
    if float(c.max().item()) <= 1.0 + 1e-3 and float((c == 0).float().mean()) > 0.05:
        binary = c > 0.5
    else:
        binary = c > float(c.median().item())
    if binary.any():
        out["mean_corrupt_region"] = float(g[binary].mean().item())
        out["corrupt_coverage"] = float(binary.float().mean().item())
    if (~binary).any():
        out["mean_clean_region"] = float(g[~binary].mean().item())
    if not np.isnan(out["mean_clean_region"]) and not np.isnan(
        out["mean_corrupt_region"]
    ):
        out["delta_clean_minus_corrupt"] = (
            out["mean_clean_region"] - out["mean_corrupt_region"]
        )
    return out


def _save_triptych(
    corrupt_hw: Optional[torch.Tensor],
    tx_hw: torch.Tensor,
    save_path: str,
    title: str,
) -> None:
    n_panels = 2 if corrupt_hw is None else 3
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels, 4.0), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]
    panel_i = 0
    # Align corrupt map to transmission resolution (Where2comm often differs).
    if corrupt_hw is not None:
        corrupt_hw = _resize_hw_to(corrupt_hw, tx_hw.shape)

    if corrupt_hw is not None:
        im0 = axes[panel_i].imshow(corrupt_hw.numpy(), vmin=0.0, vmax=1.0, cmap="magma")
        fig.colorbar(im0, ax=axes[panel_i], shrink=0.75)
        axes[panel_i].set_title("corrupt map", fontsize=9)
        axes[panel_i].set_xticks([])
        axes[panel_i].set_yticks([])
        panel_i += 1

    im1 = axes[panel_i].imshow(tx_hw.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im1, ax=axes[panel_i], shrink=0.75)
    axes[panel_i].set_title("transmission", fontsize=9)
    axes[panel_i].set_xticks([])
    axes[panel_i].set_yticks([])
    panel_i += 1

    if corrupt_hw is not None:
        axes[panel_i].imshow(tx_hw.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        # Red overlay where corrupted
        overlay = np.zeros((*tx_hw.shape, 4), dtype=np.float32)
        c_np = corrupt_hw.numpy()
        overlay[..., 0] = 1.0
        overlay[..., 3] = np.clip(c_np, 0.0, 1.0) * 0.45
        axes[panel_i].imshow(overlay)
        axes[panel_i].set_title("overlay", fontsize=9)
        axes[panel_i].set_xticks([])
        axes[panel_i].set_yticks([])

    fig.suptitle(title, fontsize=9)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def should_save_transmission(idx: int, every: int = DEFAULT_SAVE_EVERY) -> bool:
    """Return True every ``every`` samples (including idx=0)."""
    if every <= 0:
        return True
    return (idx % every) == 0


def save_transmission_statistics(
    transmission_maps: Optional[Dict[str, torch.Tensor]],
    save_root: str,
    idx: int,
    sample_id: str = "unknown",
    corrupt_cfg: Any = None,
    corrupt_maps: Optional[Dict[str, torch.Tensor]] = None,
    model_tag: str = "model",
    every: int = DEFAULT_SAVE_EVERY,
) -> None:
    """Save per-agent transmission / corrupt / overlay panels and CSV stats.

    Layout::

        {save_root}/gate_statistics/{corrupt_tag}/
            {model_tag}_{agent_key}_{idx:05d}.png
            stats.csv
            summary.txt

    Args:
        transmission_maps: Named maps, e.g. ``gate_rsu``, ``drone_0``, ``vehicle_0``.
        save_root: Model dir (same parent as vis_bev).
        idx: Global sample index.
        sample_id: Stable sample id.
        corrupt_cfg: Optional corruption config for subdir naming.
        corrupt_maps: Optional named corrupt spatial maps.
        model_tag: ``mambafusion`` / ``where2comm``.
        every: Save cadence (default 50).
    """
    if not transmission_maps:
        return
    if not should_save_transmission(idx, every=every):
        return

    tag = _corrupt_tag(corrupt_cfg)
    save_dir = os.path.join(save_root, "gate_statistics", tag)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "stats.csv")
    write_header = not os.path.exists(csv_path)

    fieldnames = [
        "idx",
        "sample_id",
        "model",
        "agent",
        "mean",
        "std",
        "min",
        "max",
        "ratio_gt_0.5",
        "ratio_gt_0.7",
        "ratio_gt_0.9",
        "mean_corrupt_region",
        "mean_clean_region",
        "delta_clean_minus_corrupt",
        "corrupt_coverage",
    ]
    rows = []
    corrupt_maps = corrupt_maps or {}

    for agent_key, tx in transmission_maps.items():
        if tx is None or not torch.is_tensor(tx):
            continue
        tx_hw = _to_hw(tx)
        stats = _map_stats(tx_hw)

        # Match corrupt map: exact key, or type prefix (rsu / drone / vehicle)
        c_map = corrupt_maps.get(agent_key)
        if c_map is None:
            agent_type = agent_key.split("_")[0]
            c_map = corrupt_maps.get(agent_type)
        region = _region_stats(tx_hw, c_map)
        rows.append(
            {
                "idx": idx,
                "sample_id": sample_id,
                "model": model_tag,
                "agent": agent_key,
                **stats,
                **region,
            }
        )

        c_hw = _to_hw(c_map) if c_map is not None else None
        title = (
            f"{model_tag} | {agent_key} | idx={idx:05d} | {sample_id}\n"
            f"mean={stats['mean']:.4f}"
        )
        if not np.isnan(region["delta_clean_minus_corrupt"]):
            title += (
                f" | clean={region['mean_clean_region']:.4f} "
                f"corrupt={region['mean_corrupt_region']:.4f} "
                f"Δ={region['delta_clean_minus_corrupt']:.4f}"
            )
        out_png = os.path.join(
            save_dir, f"{model_tag}_{agent_key}_{idx:05d}.png"
        )
        _save_triptych(c_hw, tx_hw, out_png, title)

    if not rows:
        return

    with open(csv_path, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    _update_summary(save_dir, csv_path)


def _update_summary(save_dir: str, csv_path: str) -> None:
    from collections import defaultdict

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('model', 'model')}/{row['agent']}"
            sums[key] += float(row["mean"])
            counts[key] += 1

    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("model_agent,num_frames,mean_of_transmission_mean\n")
        for key in sorted(counts.keys()):
            mean_val = sums[key] / max(counts[key], 1)
            f.write(f"{key},{counts[key]},{mean_val:.6f}\n")


# Backward-compatible alias
def save_gate_statistics(
    gate_outputs: Optional[Dict[str, Optional[torch.Tensor]]],
    save_root: str,
    idx: int,
    sample_id: str = "unknown",
    corrupt_cfg: Any = None,
    corrupt_maps: Optional[Dict[str, torch.Tensor]] = None,
    every: int = DEFAULT_SAVE_EVERY,
) -> None:
    """Legacy wrapper for MambaFusion gate dicts."""
    if not gate_outputs:
        return
    tx: Dict[str, torch.Tensor] = {}
    # Ego vehicle is always fully kept in MambaFusion gating design.
    first = next((v for v in gate_outputs.values() if isinstance(v, torch.Tensor)), None)
    if first is not None:
        ones = torch.ones_like(first)
        tx["vehicle_0"] = ones
    mapping = (("gate_rsu", "rsu_0"), ("gate_drone", "drone_0"))
    for src, dst in mapping:
        g = gate_outputs.get(src)
        if isinstance(g, torch.Tensor):
            tx[dst] = g
    # Also keep original keys if present
    for k, v in gate_outputs.items():
        if isinstance(v, torch.Tensor) and k not in tx:
            tx[k] = v
    save_transmission_statistics(
        transmission_maps=tx,
        save_root=save_root,
        idx=idx,
        sample_id=sample_id,
        corrupt_cfg=corrupt_cfg,
        corrupt_maps=corrupt_maps,
        model_tag="mambafusion",
        every=every,
    )
