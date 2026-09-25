"""Paired visualization for gating corruption and temporal ablation.

Outputs are written under ``/home/dell/suyi/visualization`` following plan.md.
"""

from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.tools.temporal_utils import (
    get_temporal_training_cfg,
    is_mambafusion_model,
    populate_temporal_fields,
    update_temporal_state,
)
from opencood.utils import common_utils
from opencood.utils.bev_corruption import build_corrupt_config, extract_sample_id
from opencood.visualization import simple_vis

DEFAULT_OUT = Path("/home/dell/suyi/visualization")
DEFAULT_TEST_DIR = "/home/dell/suyi/AirV2X-Perception/test/test"
LOG_ROOT = ROOT / "opencood/logs/airv2x_intermediate_mambafusion"
PC_RANGE = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gating/Temporal visualization")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["gating", "temporal", "both"],
        help="Which experiment group to run",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--shortlist_k", type=int, default=20)
    parser.add_argument("--select_k", type=int, default=3)
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument("--corrupt_level", type=str, default="medium")
    parser.add_argument("--corrupt_seed_base", type=int, default=0)
    parser.add_argument(
        "--save_all_candidates",
        action="store_true",
        help="Also dump shortlist visualizations (not only selected)",
    )
    parser.add_argument(
        "--render_paper",
        action="store_true",
        help="Only render publication-quality figures for selected/paper keys",
    )
    parser.add_argument(
        "--paper_keys",
        type=str,
        default="",
        help="Comma-separated frame keys for paper render (e.g. scen0000_3,scen0000_1)",
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default="",
        help="Optional x crop as 'xmin,xmax' in meters (shared by all paper figures)",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default="",
        help="Optional y crop as 'ymin,ymax' in meters (shared by all paper figures)",
    )
    parser.add_argument(
        "--window_size",
        type=str,
        default="10,11",
        help="Temporal routing window size as 'H,W' (matches ROUTE_WINDOW)",
    )
    return parser.parse_args()


def parse_optional_range(text: str) -> Optional[Tuple[float, float]]:
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'a,b', got {text!r}")
    return float(parts[0]), float(parts[1])


def parse_window_size(text: str) -> Tuple[int, int]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected window 'H,W', got {text!r}")
    return int(parts[0]), int(parts[1])


def resolve_paper_keys(args: argparse.Namespace) -> Dict[str, set]:
    """Return {'gating': set(...), 'temporal': set(...)} of frame keys to render."""
    gating_keys: set = set()
    temporal_keys: set = set()
    if args.paper_keys.strip():
        keys = {k.strip() for k in args.paper_keys.split(",") if k.strip()}
        gating_keys |= keys
        temporal_keys |= keys
    else:
        g_sel = Path(args.out_root) / "gating_corruption" / "selected"
        t_sel = Path(args.out_root) / "temporal" / "selected"
        if g_sel.is_dir():
            gating_keys |= {p.name for p in g_sel.iterdir() if p.is_dir()}
        if t_sel.is_dir():
            temporal_keys |= {p.name for p in t_sel.iterdir() if p.is_dir()}
    return {"gating": gating_keys, "temporal": temporal_keys}


def _first(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return x[0]
    if isinstance(x, torch.Tensor) and x.numel() >= 1:
        return x.flatten()[0].item()
    return x


def frame_key(batch_data: Dict[str, Any]) -> str:
    ego = batch_data["ego"]
    scen = int(_first(ego["scenario_index_list"]))
    ts = str(_first(ego["timestamp_key_list"]))
    return f"scen{scen:04d}_{ts}"


def load_hypes(model_dir: Path, test_dir: str) -> Dict[str, Any]:
    opt = argparse.Namespace(model_dir=str(model_dir), config_file="config.yaml")
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["test_dir"] = test_dir
    hypes["validate_dir"] = test_dir
    return hypes


def create_and_load(
    model_dir: Path,
    epoch: int,
    hypes: Dict[str, Any],
    dataset: Any,
    device: torch.device,
) -> torch.nn.Module:
    model = train_utils.create_model(hypes, dataset)
    _, model = train_utils.load_model(
        str(model_dir), model, epoch, start_from_best=False
    )
    model.to(device)
    model.eval()
    return model


def boxes_to_numpy(boxes: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if boxes is None:
        return None
    if isinstance(boxes, torch.Tensor):
        return boxes.detach().cpu().numpy()
    return np.asarray(boxes)


def frame_match_stats(
    pred_boxes: Optional[torch.Tensor],
    pred_scores: Optional[torch.Tensor],
    gt_boxes: Optional[torch.Tensor],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """Compute TP/FP/FN and matched center/yaw error for one frame."""
    empty = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "gt": 0,
        "mean_center_err": float("nan"),
        "mean_yaw_err": float("nan"),
        "mean_fp_score": float("nan"),
    }
    if gt_boxes is None:
        return empty
    gt_np = boxes_to_numpy(gt_boxes)
    empty["gt"] = int(gt_np.shape[0])
    empty["fn"] = int(gt_np.shape[0])
    if pred_boxes is None or pred_scores is None or pred_boxes.numel() == 0:
        return empty

    det_boxes = boxes_to_numpy(pred_boxes)
    det_score = boxes_to_numpy(pred_scores).reshape(-1)
    order = np.argsort(-det_score)
    det_boxes = det_boxes[order]
    det_score = det_score[order]
    det_polys = list(common_utils.convert_format(det_boxes))
    gt_polys = list(common_utils.convert_format(gt_np))

    tp = 0
    fp = 0
    center_errs: List[float] = []
    yaw_errs: List[float] = []
    fp_scores: List[float] = []
    matched_gt = set()

    for i, det_poly in enumerate(det_polys):
        if len(gt_polys) == 0:
            fp += 1
            fp_scores.append(float(det_score[i]))
            continue
        ious = common_utils.compute_iou(det_poly, gt_polys)
        best = int(np.argmax(ious))
        if float(ious[best]) < iou_thresh:
            fp += 1
            fp_scores.append(float(det_score[i]))
            continue
        # remap best index into original gt list
        # gt_polys shrinks; track via remaining indices
        remain_idx = [j for j in range(len(gt_np)) if j not in matched_gt]
        gt_i = remain_idx[best]
        matched_gt.add(gt_i)
        gt_polys.pop(best)
        tp += 1
        # center / yaw from first 4 corners approx
        det_c = det_boxes[i, :4, :2].mean(axis=0)
        gt_c = gt_np[gt_i, :4, :2].mean(axis=0)
        center_errs.append(float(np.linalg.norm(det_c - gt_c)))
        det_yaw = np.arctan2(
            det_boxes[i, 1, 1] - det_boxes[i, 0, 1],
            det_boxes[i, 1, 0] - det_boxes[i, 0, 0],
        )
        gt_yaw = np.arctan2(
            gt_np[gt_i, 1, 1] - gt_np[gt_i, 0, 1],
            gt_np[gt_i, 1, 0] - gt_np[gt_i, 0, 0],
        )
        dyaw = abs(((det_yaw - gt_yaw + np.pi) % (2 * np.pi)) - np.pi)
        yaw_errs.append(float(dyaw))

    fn = int(gt_np.shape[0] - tp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": int(gt_np.shape[0]),
        "mean_center_err": float(np.mean(center_errs)) if center_errs else float("nan"),
        "mean_yaw_err": float(np.mean(yaw_errs)) if yaw_errs else float("nan"),
        "mean_fp_score": float(np.mean(fp_scores)) if fp_scores else float("nan"),
    }


def tensor_to_hw(x: torch.Tensor) -> np.ndarray:
    t = x.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        t = t[0] if t.shape[0] == 1 else t.abs().mean(dim=0)
    return t.numpy()


def percentile_norm(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    a = arr.astype(np.float32)
    p_lo, p_hi = np.percentile(a, [lo, hi])
    if p_hi <= p_lo + 1e-6:
        return np.zeros_like(a)
    return np.clip((a - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def feature_to_chw(feature: Any) -> np.ndarray:
    """Return float32 feature as ``[C, H, W]`` without changing values."""
    if isinstance(feature, torch.Tensor):
        t = feature.detach().float().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() != 3:
            raise ValueError(f"Expected CHW/BCHW feature, got {tuple(feature.shape)}")
        return t.numpy()
    arr = np.asarray(feature, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW/BCHW feature, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def rms_log_energy(feature: Any) -> np.ndarray:
    """Convert multi-channel BEV feature to RMS energy then ``log1p``."""
    chw = feature_to_chw(feature)
    energy = np.sqrt(np.mean(np.square(chw), axis=0)).astype(np.float32)
    return np.log1p(energy)


def shared_percentile_bounds(
    arrays: List[np.ndarray],
    lo: float = 2.0,
    hi: float = 98.0,
) -> Tuple[float, float]:
    """Shared percentile bounds across all maps to be compared."""
    flat = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays])
    p_lo, p_hi = np.percentile(flat, [lo, hi])
    if float(p_hi) <= float(p_lo) + 1e-6:
        return 0.0, 1.0
    return float(p_lo), float(p_hi)


def apply_shared_norm(
    arr: np.ndarray,
    p_lo: float,
    p_hi: float,
) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return np.clip((a - p_lo) / (p_hi - p_lo + 1e-12), 0.0, 1.0)


def default_extent() -> Tuple[float, float, float, float]:
    return (PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4])


def resolve_extent(
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = default_extent()
    if xlim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
    if ylim is not None:
        ymin, ymax = float(ylim[0]), float(ylim[1])
    return xmin, xmax, ymin, ymax


def setup_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
                "serif",
            ],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _paper_figsize(
    extent: Tuple[float, float, float, float],
    width_in: float = 3.4,
) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = extent
    aspect = max((ymax - ymin) / max(xmax - xmin, 1e-6), 0.15)
    return width_in, width_in * aspect + 0.55


def _style_paper_axes(
    ax: Any,
    extent: Tuple[float, float, float, float],
    title: str,
) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=7.5)
    ax.set_ylabel("y (m)", fontsize=7.5)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _add_compact_response_colorbar(fig: Any, mappable: Any, ax: Any) -> None:
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02, shrink=0.92)
    cbar.set_label("Response", fontsize=7.5)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7, length=2.0, width=0.5)
    cbar.outline.set_linewidth(0.5)


def draw_gt_boxes(
    ax: Any,
    gt_boxes: Optional[Any],
    linewidth: float = 0.85,
) -> None:
    if gt_boxes is None:
        return
    boxes = boxes_to_numpy(gt_boxes)
    if boxes is None or boxes.size == 0:
        return
    for box in boxes:
        # Use the first 4 corners as the BEV footprint.
        pts = box[[0, 1, 2, 3, 0], :2]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="white",
            linestyle="--",
            linewidth=linewidth,
            solid_capstyle="round",
            alpha=0.95,
            zorder=5,
        )


def selected_window_rects_from_mask(
    pixel_mask: np.ndarray,
    window_size: Tuple[int, int],
    extent: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """Recover axis-aligned selected windows from a dense pixel mask.

    Returns rectangles as ``(x, y, w, h)`` in meters.
    """
    mask = np.asarray(pixel_mask, dtype=np.float32)
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    height, width = mask.shape
    win_h, win_w = int(window_size[0]), int(window_size[1])
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / float(width)
    dy = (ymax - ymin) / float(height)
    rects: List[Tuple[float, float, float, float]] = []
    for i in range(0, height, win_h):
        for j in range(0, width, win_w):
            i1 = min(i + win_h, height)
            j1 = min(j + win_w, width)
            patch = mask[i:i1, j:j1]
            if patch.size == 0:
                continue
            # Selected windows are filled solidly by WindowRouter expansion.
            if float(patch.mean()) >= 0.5:
                x0 = xmin + j * dx
                y0 = ymin + i * dy
                rects.append((x0, y0, (j1 - j) * dx, (i1 - i) * dy))
    return rects


def selected_window_rects_from_window_mask(
    window_mask: np.ndarray,
    window_size: Tuple[int, int],
    map_hw: Tuple[int, int],
    extent: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """Build rectangles directly from a coarse ``window_mask``."""
    wm = np.asarray(window_mask, dtype=np.float32)
    if wm.ndim == 4:
        wm = wm[0, 0]
    elif wm.ndim == 3:
        wm = wm[0]
    win_h, win_w = int(window_size[0]), int(window_size[1])
    map_h, map_w = int(map_hw[0]), int(map_hw[1])
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / float(map_w)
    dy = (ymax - ymin) / float(map_h)
    rects: List[Tuple[float, float, float, float]] = []
    n_h, n_w = wm.shape
    for wi in range(n_h):
        for wj in range(n_w):
            if float(wm[wi, wj]) < 0.5:
                continue
            i0 = wi * win_h
            j0 = wj * win_w
            i1 = min(i0 + win_h, map_h)
            j1 = min(j0 + win_w, map_w)
            if i0 >= map_h or j0 >= map_w:
                continue
            rects.append(
                (
                    xmin + j0 * dx,
                    ymin + i0 * dy,
                    (j1 - j0) * dx,
                    (i1 - i0) * dy,
                )
            )
    return rects


def draw_window_rectangles(
    ax: Any,
    rects: List[Tuple[float, float, float, float]],
    linewidth: float = 0.9,
    edgecolor: str = "#2ca02c",
) -> None:
    """Legacy per-window rectangles (prefer ``draw_connected_window_outlines``)."""
    from matplotlib.patches import Rectangle

    for x, y, w, h in rects:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=4,
            )
        )


def make_gate_split_cmap() -> mcolors.Colormap:
    """Colormap: gray for ``[0, 0.5)``, magma gradient for ``[0.5, 1]``."""
    n = 256
    colors = np.zeros((n, 4), dtype=np.float32)
    half = n // 2
    for i in range(half):
        t = i / max(half - 1, 1)
        g = 0.18 + 0.42 * t
        colors[i] = (g, g, g, 1.0)
    for i in range(half, n):
        t = (i - half) / max(n - half - 1, 1)
        colors[i] = plt.cm.magma(t)
    return mcolors.ListedColormap(colors, name="gate_gray_magma")


def gate_split_rgba(gate_hw: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """RGBA overlay: gray below 0.5, magma gradient above 0.5."""
    g = np.clip(np.asarray(gate_hw, dtype=np.float32), 0.0, 1.0)
    rgba = np.zeros((*g.shape, 4), dtype=np.float32)
    low = g < 0.5
    high = ~low
    if np.any(low):
        t = g[low] / 0.5
        gray = 0.18 + 0.42 * t
        rgba[low, 0] = gray
        rgba[low, 1] = gray
        rgba[low, 2] = gray
        rgba[low, 3] = alpha
    if np.any(high):
        t = (g[high] - 0.5) / 0.5
        rgba[high] = plt.cm.magma(t)
        rgba[high, 3] = alpha
    return rgba


def selected_window_binary_grid(
    window_mask: Optional[np.ndarray],
    pixel_mask: Optional[np.ndarray],
    window_size: Tuple[int, int],
    map_hw: Tuple[int, int],
) -> np.ndarray:
    """Return a coarse binary window grid ``[n_h, n_w]`` of selected windows."""
    win_h, win_w = int(window_size[0]), int(window_size[1])
    map_h, map_w = int(map_hw[0]), int(map_hw[1])
    if window_mask is not None:
        wm = np.asarray(window_mask, dtype=np.float32)
        if wm.ndim == 4:
            wm = wm[0, 0]
        elif wm.ndim == 3:
            wm = wm[0]
        return (wm >= 0.5).astype(np.uint8)
    if pixel_mask is None:
        return np.zeros((1, 1), dtype=np.uint8)
    pm = np.asarray(pixel_mask, dtype=np.float32)
    if pm.ndim == 4:
        pm = pm[0, 0]
    elif pm.ndim == 3:
        pm = pm[0]
    if pm.shape != (map_h, map_w):
        pm = _resize_hw(pm, (map_h, map_w))
    n_h = int(np.ceil(map_h / win_h))
    n_w = int(np.ceil(map_w / win_w))
    grid = np.zeros((n_h, n_w), dtype=np.uint8)
    for i in range(n_h):
        for j in range(n_w):
            i0, j0 = i * win_h, j * win_w
            i1, j1 = min(i0 + win_h, map_h), min(j0 + win_w, map_w)
            patch = pm[i0:i1, j0:j1]
            if patch.size and float(patch.mean()) >= 0.5:
                grid[i, j] = 1
    return grid


def draw_connected_window_outlines(
    ax: Any,
    window_grid: np.ndarray,
    window_size: Tuple[int, int],
    map_hw: Tuple[int, int],
    extent: Tuple[float, float, float, float],
    color: str = "#2ca02c",
    linewidth: float = 0.9,
) -> None:
    """Draw green outer outlines of 4-connected selected windows (no internal edges)."""
    grid = (np.asarray(window_grid) >= 0.5).astype(np.uint8)
    if grid.ndim != 2 or not np.any(grid):
        return
    win_h, win_w = int(window_size[0]), int(window_size[1])
    map_h, map_w = int(map_hw[0]), int(map_hw[1])
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / float(map_w)
    dy = (ymax - ymin) / float(map_h)
    n_h, n_w = grid.shape

    def selected(i: int, j: int) -> bool:
        return 0 <= i < n_h and 0 <= j < n_w and bool(grid[i, j])

    # Horizontal edges between row i-1 and row i.
    for i in range(n_h + 1):
        j = 0
        while j < n_w:
            if selected(i - 1, j) == selected(i, j):
                j += 1
                continue
            j0 = j
            while j < n_w and selected(i - 1, j) != selected(i, j):
                # keep run while the boundary type stays "crossing"
                if (selected(i - 1, j) != selected(i - 1, j0)) or (
                    selected(i, j) != selected(i, j0)
                ):
                    break
                j += 1
            y_edge = ymin + min(i * win_h, map_h) * dy
            x_left = xmin + (j0 * win_w) * dx
            x_right = xmin + min(j * win_w, map_w) * dx
            ax.plot(
                [x_left, x_right],
                [y_edge, y_edge],
                color=color,
                linewidth=linewidth,
                solid_capstyle="butt",
                zorder=4,
            )

    # Vertical edges between col j-1 and col j.
    for j in range(n_w + 1):
        i = 0
        while i < n_h:
            if selected(i, j - 1) == selected(i, j):
                i += 1
                continue
            i0 = i
            while i < n_h and selected(i, j - 1) != selected(i, j):
                if (selected(i, j - 1) != selected(i0, j - 1)) or (
                    selected(i, j) != selected(i0, j)
                ):
                    break
                i += 1
            x_edge = xmin + min(j * win_w, map_w) * dx
            y_bottom = ymin + (i0 * win_h) * dy
            y_top = ymin + min(i * win_h, map_h) * dy
            ax.plot(
                [x_edge, x_edge],
                [y_bottom, y_top],
                color=color,
                linewidth=linewidth,
                solid_capstyle="butt",
                zorder=4,
            )


def _resize_hw(src: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if tuple(src.shape) == tuple(target_hw):
        return src
    t = torch.from_numpy(np.asarray(src, dtype=np.float32))[None, None]
    t = torch.nn.functional.interpolate(t, size=target_hw, mode="nearest")
    return t[0, 0].numpy()


def save_paper_figure(
    fig: Any,
    dest_dir: Path,
    stem: str,
) -> Tuple[Path, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = dest_dir / f"{stem}.pdf"
    png_path = dest_dir / f"{stem}.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    return pdf_path, png_path


def save_gate_paper(
    corrupted_drone_bev: Any,
    gate: Any,
    dest_dir: Path,
    gt_boxes: Optional[Any] = None,
    feat_norm: Optional[np.ndarray] = None,
    shared_bounds: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[Path, Path]:
    """Publication figure: feature base + split gate (<0.5 gray, >=0.5 magma)."""
    setup_paper_style()
    extent = resolve_extent(xlim, ylim)
    energy = rms_log_energy(corrupted_drone_bev)
    if feat_norm is None:
        if shared_bounds is None:
            shared_bounds = shared_percentile_bounds([energy])
        feat_norm = apply_shared_norm(energy, shared_bounds[0], shared_bounds[1])
    gate_hw = tensor_to_hw(gate) if isinstance(gate, torch.Tensor) else np.asarray(gate)
    if gate_hw.ndim != 2:
        raise ValueError(f"gate must be HxW, got {gate_hw.shape}")
    if gate_hw.shape != feat_norm.shape:
        gate_hw = _resize_hw(gate_hw, feat_norm.shape)

    # Low-contrast gray feature base.
    gray = 0.22 + 0.50 * feat_norm
    fig, ax = plt.subplots(figsize=_paper_figsize(extent), dpi=150)
    ax.imshow(
        gray,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        interpolation="nearest",
        zorder=1,
    )
    gate_rgba = gate_split_rgba(gate_hw, alpha=0.65)
    ax.imshow(
        gate_rgba,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        interpolation="nearest",
        zorder=2,
    )
    # Colorbar uses the same split mapping, fixed to [0, 1].
    gate_cmap = make_gate_split_cmap()
    gate_im = ax.imshow(
        np.zeros((1, 1)),
        cmap=gate_cmap,
        vmin=0.0,
        vmax=1.0,
        visible=False,
    )
    draw_gt_boxes(ax, gt_boxes)
    _style_paper_axes(ax, extent, "UAV Reliability Gate")
    _add_compact_response_colorbar(fig, gate_im, ax)
    paths = save_paper_figure(fig, dest_dir, "gate_visualization_paper")
    plt.close(fig)
    return paths


def save_need_paper(
    pre_temporal_feature: Any,
    need_map: Any,
    dest_dir: Path,
    pixel_mask: Optional[Any] = None,
    window_mask: Optional[Any] = None,
    window_size: Tuple[int, int] = (10, 11),
    gt_boxes: Optional[Any] = None,
    feat_norm: Optional[np.ndarray] = None,
    shared_bounds: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[Path, Path]:
    """Publication figure: grayscale pre-temporal feature + magma need overlay."""
    setup_paper_style()
    extent = resolve_extent(xlim, ylim)
    energy = rms_log_energy(pre_temporal_feature)
    if feat_norm is None:
        if shared_bounds is None:
            shared_bounds = shared_percentile_bounds([energy])
        feat_norm = apply_shared_norm(energy, shared_bounds[0], shared_bounds[1])
    need = (
        tensor_to_hw(need_map)
        if isinstance(need_map, torch.Tensor)
        else np.asarray(need_map, dtype=np.float32)
    )
    if need.ndim != 2:
        raise ValueError(f"need_map must be HxW, got {need.shape}")
    if need.shape != feat_norm.shape:
        need = _resize_hw(need, feat_norm.shape)

    fig, ax = plt.subplots(figsize=_paper_figsize(extent), dpi=150)
    ax.imshow(
        feat_norm,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        alpha=0.25,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        interpolation="nearest",
        zorder=1,
    )
    need_im = ax.imshow(
        np.clip(need, 0.0, 1.0),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        alpha=0.70,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        interpolation="nearest",
        zorder=2,
    )

    wm_np: Optional[np.ndarray] = None
    if isinstance(window_mask, torch.Tensor):
        t = window_mask.detach().float().cpu()
        if t.dim() == 4:
            t = t[0, 0]
        elif t.dim() == 3:
            t = t[0]
        wm_np = t.numpy()
    elif window_mask is not None:
        wm_np = np.asarray(window_mask, dtype=np.float32)
        if wm_np.ndim == 4:
            wm_np = wm_np[0, 0]
        elif wm_np.ndim == 3:
            wm_np = wm_np[0]

    pm_np: Optional[np.ndarray] = None
    if isinstance(pixel_mask, torch.Tensor):
        pm_np = tensor_to_hw(pixel_mask)
    elif pixel_mask is not None:
        pm_np = np.asarray(pixel_mask, dtype=np.float32)
        if pm_np.ndim == 4:
            pm_np = pm_np[0, 0]
        elif pm_np.ndim == 3:
            pm_np = pm_np[0]

    window_grid = selected_window_binary_grid(
        wm_np, pm_np, window_size, feat_norm.shape
    )
    # Connected selected windows share one green outer outline (no internal edges).
    draw_connected_window_outlines(
        ax,
        window_grid,
        window_size,
        feat_norm.shape,
        extent,
        color="#2ca02c",
        linewidth=0.9,
    )
    draw_gt_boxes(ax, gt_boxes)
    _style_paper_axes(ax, extent, "Temporal Need Map")
    _add_compact_response_colorbar(fig, need_im, ax)
    paths = save_paper_figure(fig, dest_dir, "temporal_visualization_paper")
    plt.close(fig)
    return paths


def save_visualization_preview(
    gate_png: Path,
    temporal_png: Path,
    out_path: Path,
) -> None:
    """Side-by-side preview of the two paper-style figures."""
    setup_paper_style()
    if not gate_png.exists() or not temporal_png.exists():
        return
    img_g = plt.imread(str(gate_png))
    img_t = plt.imread(str(temporal_png))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), dpi=150)
    axes[0].imshow(img_g)
    axes[0].axis("off")
    axes[0].set_title("UAV Reliability Gate", fontsize=8.5)
    axes[1].imshow(img_t)
    axes[1].axis("off")
    axes[1].set_title("Temporal Need Map", fontsize=8.5)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_gate_overlay(
    gate: torch.Tensor,
    corrupt: Optional[torch.Tensor],
    save_path: Path,
    title: str,
) -> None:
    """Legacy debug visualization (kept for non-paper outputs)."""
    gate_hw = tensor_to_hw(gate)
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    im = ax.imshow(
        gate_hw,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=[PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4]],
        origin="lower",
        aspect="auto",
    )
    if corrupt is not None:
        c_hw = tensor_to_hw(corrupt)
        if c_hw.shape != gate_hw.shape:
            c_t = torch.from_numpy(c_hw)[None, None].float()
            c_t = torch.nn.functional.interpolate(
                c_t, size=gate_hw.shape, mode="nearest"
            )
            c_hw = c_t[0, 0].numpy()
        overlay = np.zeros((*gate_hw.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = np.clip(c_hw, 0.0, 1.0) * 0.45
        ax.imshow(
            overlay,
            extent=[PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4]],
            origin="lower",
            aspect="auto",
        )
        try:
            ax.contour(
                np.linspace(PC_RANGE[0], PC_RANGE[3], gate_hw.shape[1]),
                np.linspace(PC_RANGE[1], PC_RANGE[4], gate_hw.shape[0]),
                c_hw,
                levels=[0.5],
                colors="red",
                linewidths=0.8,
            )
        except Exception:
            pass
    fig.colorbar(im, ax=ax, shrink=0.8, label="gate")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=9)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_need_overlay(
    feature: torch.Tensor,
    need_map: torch.Tensor,
    save_path: Path,
    title: str,
) -> None:
    """Legacy debug visualization (kept for non-paper outputs)."""
    feat = feature.detach().float().cpu()
    if feat.dim() == 4:
        feat = feat[0]
    resp = percentile_norm(feat.abs().mean(dim=0).numpy())
    need = tensor_to_hw(need_map)
    if need.shape != resp.shape:
        need_t = torch.from_numpy(need)[None, None].float()
        need_t = torch.nn.functional.interpolate(
            need_t, size=resp.shape, mode="bilinear", align_corners=False
        )
        need = need_t[0, 0].numpy()

    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.imshow(
        resp,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=[PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4]],
        origin="lower",
        aspect="auto",
    )
    need_rgba = plt.cm.magma(need)
    need_rgba[..., 3] = np.clip(need, 0.0, 1.0) * 0.55
    ax.imshow(
        need_rgba,
        extent=[PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4]],
        origin="lower",
        aspect="auto",
    )
    try:
        ax.contour(
            np.linspace(PC_RANGE[0], PC_RANGE[3], need.shape[1]),
            np.linspace(PC_RANGE[1], PC_RANGE[4], need.shape[0]),
            need,
            levels=[0.5],
            colors="cyan",
            linewidths=0.9,
        )
    except Exception:
        pass
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=9)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_detection(
    pred_boxes: Optional[torch.Tensor],
    gt_boxes: Optional[torch.Tensor],
    batch_data: Dict[str, Any],
    hypes: Dict[str, Any],
    save_path: Path,
) -> None:
    if pred_boxes is None:
        # keep empty canvas with GT only
        pred_boxes = torch.zeros((0, 8, 3), device=gt_boxes.device if gt_boxes is not None else "cpu")
    simple_vis.visualize(
        pred_boxes,
        gt_boxes,
        batch_data["ego"]["origin_lidar"][0],
        hypes["preprocess"]["cav_lidar_range"],
        str(save_path),
        method="bev",
        left_hand=True,
        vis_pred_box=True,
        pcd_rsu=batch_data["ego"].get("origin_lidar_rsu", [None])[0],
        pcd_drone=batch_data["ego"].get("origin_lidar_drone", [None])[0],
        batch_data=batch_data,
    )


def stitch_panel(paths: List[Path], out_path: Path, titles: List[str]) -> None:
    imgs = [plt.imread(str(p)) for p in paths if p.exists()]
    if not imgs:
        return
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _to_cpu_tensor(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def _move_batch_for_vis(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields needed by simple_vis on CPU."""
    ego = batch_data["ego"]
    out_ego: Dict[str, Any] = {}
    for key in (
        "origin_lidar",
        "origin_lidar_rsu",
        "origin_lidar_drone",
        "scenario_index_list",
        "timestamp_key_list",
        "metadata_path_list",
    ):
        if key not in ego:
            continue
        val = ego[key]
        if isinstance(val, torch.Tensor):
            out_ego[key] = val.detach().cpu()
        elif isinstance(val, list):
            out_ego[key] = [
                v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in val
            ]
        else:
            out_ego[key] = val
    return {"ego": out_ego}


def infer_one(
    model: torch.nn.Module,
    batch_data: Dict[str, Any],
    dataset: Any,
    device: torch.device,
    temporal_state: Dict[str, Any],
    corrupt_cfg: Any = None,
    visualization_debug: bool = True,
) -> Dict[str, Any]:
    data = train_utils.to_device(copy.deepcopy(batch_data), device)
    if corrupt_cfg is not None and getattr(corrupt_cfg, "enabled", False):
        sample_id = extract_sample_id(data)
        data["ego"]["bev_corrupt_cfg"] = corrupt_cfg
        data["ego"]["bev_corrupt_sample_id"] = sample_id
    if visualization_debug:
        data["ego"]["visualization_debug"] = True
    populate_temporal_fields(
        data["ego"],
        device,
        prev_frame=temporal_state.get("prev_frame"),
    )
    with torch.no_grad():
        outputs = inference_utils.inference_intermediate_fusion(data, model, dataset)
    update_temporal_state(temporal_state, data["ego"])

    if len(outputs) == 5:
        pred_box, pred_score, gt_box, _, pred_boxes3d = outputs
    else:
        pred_box, pred_score, gt_box, _ = outputs
        pred_boxes3d = None

    aux = data["ego"].get("_saved_fusion_aux_outputs", {}) or {}
    gates = data["ego"].get("_saved_fusion_gate_outputs", {}) or {}
    corrupts = data["ego"].get("_saved_bev_corrupt_maps", {}) or {}
    pre_temp = data["ego"].get("_saved_pre_temporal_feature", None)
    corrupted_drone = data["ego"].get("_saved_corrupted_drone_bev", None)

    return {
        "pred_box": _to_cpu_tensor(pred_box),
        "pred_score": _to_cpu_tensor(pred_score),
        "gt_box": _to_cpu_tensor(gt_box),
        "pred_boxes3d": _to_cpu_tensor(pred_boxes3d),
        "gate_drone": _to_cpu_tensor(gates.get("gate_drone")),
        "gate_rsu": _to_cpu_tensor(gates.get("gate_rsu")),
        "corrupt_drone": _to_cpu_tensor(corrupts.get("drone")),
        "corrupted_drone_bev": _to_cpu_tensor(corrupted_drone),
        "need_map": _to_cpu_tensor(aux.get("need_map")),
        "pixel_mask": _to_cpu_tensor(aux.get("pixel_mask")),
        "window_mask": _to_cpu_tensor(aux.get("window_mask")),
        "history_ready": bool(aux.get("history_ready", False)),
        "temporal_applied": bool(aux.get("temporal_applied", False)),
        "pre_temporal_feature": _to_cpu_tensor(pre_temp),
        "batch_data": _move_batch_for_vis(data),
        "temporal_reset": bool(data["ego"].get("temporal_reset", False)),
    }


def nanmean_safe(x: float, default: float = 0.0) -> float:
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def score_gating(row: Dict[str, Any]) -> float:
    # Prefer better full model + worse no-gate model + gate suppression in corrupt region.
    score = 0.0
    score += 2.0 * (row["full_tp"] - row["nogate_tp"])
    score += 1.5 * (row["nogate_fp"] - row["full_fp"])
    score += 1.5 * (row["full_fn"] < row["nogate_fn"])
    score += 1.0 * nanmean_safe(row.get("nogate_center_err"), 0.0)
    score += 0.5 * nanmean_safe(row.get("nogate_yaw_err"), 0.0)
    score += 3.0 * nanmean_safe(row.get("gate_delta_clean_minus_corrupt"), 0.0)
    score += 0.5 * nanmean_safe(row.get("corrupt_coverage"), 0.0)
    return float(score)


def score_temporal(row: Dict[str, Any]) -> float:
    score = 0.0
    score += 2.0 * (row["only_tp"] - row["concat_tp"])
    score += 1.5 * (row["concat_fp"] - row["only_fp"])
    score += 1.5 * (row["only_fn"] < row["concat_fn"])
    score += 1.0 * nanmean_safe(row.get("concat_center_err"), 0.0)
    score += 0.5 * nanmean_safe(row.get("concat_yaw_err"), 0.0)
    score += 2.0 * nanmean_safe(row.get("need_mean"), 0.0)
    score += 1.0 * float(row.get("need_ratio_gt_0.5", 0.0))
    return float(score)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def gate_region_stats(
    gate: Optional[torch.Tensor], corrupt: Optional[torch.Tensor]
) -> Dict[str, float]:
    out = {
        "gate_mean": float("nan"),
        "gate_mean_corrupt": float("nan"),
        "gate_mean_clean": float("nan"),
        "gate_delta_clean_minus_corrupt": float("nan"),
        "corrupt_coverage": float("nan"),
    }
    if gate is None:
        return out
    g = tensor_to_hw(gate)
    out["gate_mean"] = float(g.mean())
    if corrupt is None:
        return out
    c = tensor_to_hw(corrupt)
    if c.shape != g.shape:
        c_t = torch.from_numpy(c)[None, None].float()
        c_t = torch.nn.functional.interpolate(c_t, size=g.shape, mode="nearest")
        c = c_t[0, 0].numpy()
    # mask_gaussian maps mix hard-dropped patches (=1) with continuous noise intensity.
    # Prefer hard-mask regions when available; otherwise fall back to median split.
    zero_ratio = float((c <= 1e-6).mean())
    hard_ratio = float((c >= 0.99).mean())
    if hard_ratio > 0.01 and hard_ratio < 0.95:
        binary = c >= 0.99
    elif zero_ratio > 0.05:
        binary = c > 0.5
    else:
        binary = c > float(np.median(c))
    out["corrupt_coverage"] = float(binary.mean())
    if binary.any():
        out["gate_mean_corrupt"] = float(g[binary].mean())
    if (~binary).any():
        out["gate_mean_clean"] = float(g[~binary].mean())
    if not np.isnan(out["gate_mean_corrupt"]) and not np.isnan(out["gate_mean_clean"]):
        out["gate_delta_clean_minus_corrupt"] = (
            out["gate_mean_clean"] - out["gate_mean_corrupt"]
        )
    return out


def run_gating(args: argparse.Namespace, device: torch.device) -> None:
    out_root = Path(args.out_root) / "gating_corruption"
    shortlist_dir = out_root / "shortlist"
    selected_dir = out_root / "selected"
    out_root.mkdir(parents=True, exist_ok=True)
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    full_dir = LOG_ROOT / "default_all_set"
    nogate_dir = LOG_ROOT / "temporal_only"
    hypes = load_hypes(full_dir, args.test_dir)
    # Keep temporal streaming for both models.
    assert get_temporal_training_cfg(hypes)["enable"] and is_mambafusion_model(hypes)

    print("[gating] Building dataset...")
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    print(f"[gating] {len(dataset)} samples")

    print("[gating] Loading models...")
    model_full = create_and_load(full_dir, 16, hypes, dataset, device)
    hypes_ng = load_hypes(nogate_dir, args.test_dir)
    model_ng = create_and_load(nogate_dir, 6, hypes_ng, dataset, device)

    corrupt_cfg = build_corrupt_config(
        scenario="uav",
        corrupt_type="mask_gaussian",
        level=args.corrupt_level,
        seed_base=args.corrupt_seed_base,
    )
    print(f"[gating] corrupt={corrupt_cfg}")

    state_full: Dict[str, Any] = {}
    state_ng: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    # Min-heap of (score, counter, payload) keeping only shortlist_k dump payloads.
    top_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    heap_counter = 0

    for i, batch_data in tqdm(enumerate(loader), total=len(loader), desc="gating"):
        if args.max_batches is not None and i >= args.max_batches:
            break
        key = frame_key(batch_data)
        sample_id = extract_sample_id(batch_data)
        # require drone presence
        drone = batch_data["ego"].get("drone", {})
        has_drone = False
        if isinstance(drone, dict):
            batch_idxs = drone.get("batch_idxs", [])
            has_drone = len(batch_idxs) > 0
        if not has_drone and "origin_lidar_drone" in batch_data["ego"]:
            pcd = batch_data["ego"]["origin_lidar_drone"][0]
            has_drone = pcd is not None and torch.as_tensor(pcd).numel() > 0

        out_f = infer_one(
            model_full, batch_data, dataset, device, state_full, corrupt_cfg, True
        )
        out_n = infer_one(
            model_ng, batch_data, dataset, device, state_ng, corrupt_cfg, True
        )

        if not has_drone or out_f["gate_drone"] is None:
            continue
        gstats = gate_region_stats(out_f["gate_drone"], out_f["corrupt_drone"])
        if np.isnan(gstats["corrupt_coverage"]) or gstats["corrupt_coverage"] <= 0:
            continue

        st_f = frame_match_stats(out_f["pred_box"], out_f["pred_score"], out_f["gt_box"])
        st_n = frame_match_stats(out_n["pred_box"], out_n["pred_score"], out_n["gt_box"])
        row = {
            "idx": i,
            "frame_key": key,
            "sample_id": sample_id,
            "full_tp": st_f["tp"],
            "full_fp": st_f["fp"],
            "full_fn": st_f["fn"],
            "full_center_err": st_f["mean_center_err"],
            "full_yaw_err": st_f["mean_yaw_err"],
            "nogate_tp": st_n["tp"],
            "nogate_fp": st_n["fp"],
            "nogate_fn": st_n["fn"],
            "nogate_center_err": st_n["mean_center_err"],
            "nogate_yaw_err": st_n["mean_yaw_err"],
            **gstats,
            "history_ready_full": int(out_f["history_ready"]),
            "history_ready_nogate": int(out_n["history_ready"]),
            "temporal_reset": int(out_f["temporal_reset"]),
        }
        # Prefer non-reset temporal-contiguous frames for cherry-pick.
        if row["temporal_reset"]:
            row["score"] = score_gating(row) - 5.0
        else:
            row["score"] = score_gating(row)
        rows.append(row)

        payload = {"row": row, "out_f": out_f, "out_n": out_n}
        item = (row["score"], heap_counter, payload)
        heap_counter += 1
        if len(top_heap) < args.shortlist_k:
            heapq.heappush(top_heap, item)
        elif row["score"] > top_heap[0][0]:
            heapq.heapreplace(top_heap, item)

    write_csv(out_root / "candidates.csv", rows)
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    shortlist = rows_sorted[: args.shortlist_k]
    selected = shortlist[: args.select_k]
    write_csv(out_root / "shortlist.csv", shortlist)
    write_csv(out_root / "selected.csv", selected)

    cache = {p["row"]["frame_key"]: p for _, _, p in top_heap}

    def dump_one(row: Dict[str, Any], dest: Path) -> None:
        key = row["frame_key"]
        if key not in cache:
            print(f"[gating] skip dump missing cache: {key}")
            return
        item = cache[key]
        out_f, out_n = item["out_f"], item["out_n"]
        dest.mkdir(parents=True, exist_ok=True)
        p1 = dest / "01_gate_drone_corruption_overlay.png"
        p2 = dest / "02_full_epoch16_detection.png"
        p3 = dest / "03_no_gating_epoch6_detection.png"
        save_gate_overlay(
            out_f["gate_drone"],
            out_f["corrupt_drone"],
            p1,
            f"{key} | gate_drone + corrupt overlay | score={row['score']:.3f}",
        )
        save_detection(out_f["pred_box"], out_f["gt_box"], out_f["batch_data"], hypes, p2)
        save_detection(out_n["pred_box"], out_n["gt_box"], out_n["batch_data"], hypes, p3)
        stitch_panel(
            [p1, p2, p3],
            dest / "panel.png",
            ["gate+corrupt", "full ep16", "no-gating ep6"],
        )
        np.savez_compressed(
            dest / "tensors.npz",
            gate_drone=tensor_to_hw(out_f["gate_drone"]),
            corrupt_drone=(
                tensor_to_hw(out_f["corrupt_drone"])
                if out_f["corrupt_drone"] is not None
                else np.zeros((1, 1))
            ),
            corrupted_drone_bev=(
                feature_to_chw(out_f["corrupted_drone_bev"])
                if out_f["corrupted_drone_bev"] is not None
                else np.zeros((1, 1, 1), dtype=np.float32)
            ),
        )
        if out_f["corrupted_drone_bev"] is not None and out_f["gate_drone"] is not None:
            save_gate_paper(
                out_f["corrupted_drone_bev"],
                out_f["gate_drone"],
                dest,
                gt_boxes=out_f["gt_box"],
                xlim=parse_optional_range(args.xlim),
                ylim=parse_optional_range(args.ylim),
            )
        meta = {
            **row,
            "model_full": str(full_dir / "net_epoch16.pth"),
            "model_nogate": str(nogate_dir / "net_epoch6.pth"),
            "corrupt": {
                "scenario": corrupt_cfg.scenario,
                "type": corrupt_cfg.corrupt_type,
                "level": corrupt_cfg.level,
                "seed_base": corrupt_cfg.seed_base,
            },
            "test_dir": args.test_dir,
            "reason": (
                "higher full TP / lower FN, more no-gate FP/drift, "
                "and lower gate in corrupt region"
            ),
        }
        with open(dest / "metrics.json", "w") as f:
            json.dump(meta, f, indent=2)

    for row in shortlist:
        dump_one(row, shortlist_dir / row["frame_key"])
    for row in selected:
        dump_one(row, selected_dir / row["frame_key"])

    print(f"[gating] candidates={len(rows)} shortlist={len(shortlist)} selected={len(selected)}")
    print(f"[gating] outputs -> {out_root}")


def run_temporal(args: argparse.Namespace, device: torch.device) -> None:
    out_root = Path(args.out_root) / "temporal"
    shortlist_dir = out_root / "shortlist"
    selected_dir = out_root / "selected"
    out_root.mkdir(parents=True, exist_ok=True)
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    only_dir = LOG_ROOT / "temporal_only"
    concat_dir = LOG_ROOT / "temporal_concat"
    hypes_only = load_hypes(only_dir, args.test_dir)
    hypes_concat = load_hypes(concat_dir, args.test_dir)

    print("[temporal] Building dataset...")
    dataset = build_dataset(hypes_only, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    print(f"[temporal] {len(dataset)} samples")

    print("[temporal] Loading models...")
    model_only = create_and_load(only_dir, 6, hypes_only, dataset, device)
    model_concat = create_and_load(concat_dir, 1, hypes_concat, dataset, device)

    state_only: Dict[str, Any] = {}
    state_concat: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    top_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    heap_counter = 0

    for i, batch_data in tqdm(enumerate(loader), total=len(loader), desc="temporal"):
        if args.max_batches is not None and i >= args.max_batches:
            break
        key = frame_key(batch_data)
        sample_id = extract_sample_id(batch_data)

        out_o = infer_one(
            model_only, batch_data, dataset, device, state_only, None, True
        )
        out_c = infer_one(
            model_concat, batch_data, dataset, device, state_concat, None, True
        )

        # Only cherry-pick frames where temporal actually applied for our method.
        if out_o["temporal_reset"] or not out_o["history_ready"] or not out_o["temporal_applied"]:
            continue
        if out_o["need_map"] is None or out_o["pre_temporal_feature"] is None:
            continue

        st_o = frame_match_stats(out_o["pred_box"], out_o["pred_score"], out_o["gt_box"])
        st_c = frame_match_stats(out_c["pred_box"], out_c["pred_score"], out_c["gt_box"])
        need = tensor_to_hw(out_o["need_map"])
        row = {
            "idx": i,
            "frame_key": key,
            "sample_id": sample_id,
            "only_tp": st_o["tp"],
            "only_fp": st_o["fp"],
            "only_fn": st_o["fn"],
            "only_center_err": st_o["mean_center_err"],
            "only_yaw_err": st_o["mean_yaw_err"],
            "concat_tp": st_c["tp"],
            "concat_fp": st_c["fp"],
            "concat_fn": st_c["fn"],
            "concat_center_err": st_c["mean_center_err"],
            "concat_yaw_err": st_c["mean_yaw_err"],
            "need_mean": float(need.mean()),
            "need_ratio_gt_0.5": float((need > 0.5).mean()),
            "history_ready_only": int(out_o["history_ready"]),
            "history_ready_concat": int(out_c["history_ready"]),
            "temporal_applied_only": int(out_o["temporal_applied"]),
            "temporal_applied_concat": int(out_c["temporal_applied"]),
        }
        row["score"] = score_temporal(row)
        rows.append(row)

        payload = {"row": row, "out_o": out_o, "out_c": out_c}
        item = (row["score"], heap_counter, payload)
        heap_counter += 1
        if len(top_heap) < args.shortlist_k:
            heapq.heappush(top_heap, item)
        elif row["score"] > top_heap[0][0]:
            heapq.heapreplace(top_heap, item)

    write_csv(out_root / "candidates.csv", rows)
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    shortlist = rows_sorted[: args.shortlist_k]
    selected = shortlist[: args.select_k]
    write_csv(out_root / "shortlist.csv", shortlist)
    write_csv(out_root / "selected.csv", selected)

    cache = {p["row"]["frame_key"]: p for _, _, p in top_heap}

    def dump_one(row: Dict[str, Any], dest: Path) -> None:
        key = row["frame_key"]
        if key not in cache:
            print(f"[temporal] skip dump missing cache: {key}")
            return
        item = cache[key]
        out_o, out_c = item["out_o"], item["out_c"]
        dest.mkdir(parents=True, exist_ok=True)
        p1 = dest / "01_pre_temporal_feature_need_overlay.png"
        p2 = dest / "02_temporal_only_epoch6_detection.png"
        p3 = dest / "03_temporal_concat_epoch1_detection.png"
        save_need_overlay(
            out_o["pre_temporal_feature"],
            out_o["need_map"],
            p1,
            f"{key} | pre-temporal feat + need_map | score={row['score']:.3f}",
        )
        save_detection(
            out_o["pred_box"], out_o["gt_box"], out_o["batch_data"], hypes_only, p2
        )
        save_detection(
            out_c["pred_box"], out_c["gt_box"], out_c["batch_data"], hypes_concat, p3
        )
        stitch_panel(
            [p1, p2, p3],
            dest / "panel.png",
            ["feat+need", "temporal_only ep6", "temporal_concat ep1"],
        )
        np.savez_compressed(
            dest / "tensors.npz",
            pre_temporal_feature=(
                feature_to_chw(out_o["pre_temporal_feature"])
                if out_o["pre_temporal_feature"] is not None
                else np.zeros((1, 1, 1), dtype=np.float32)
            ),
            pre_temporal_response=percentile_norm(
                out_o["pre_temporal_feature"].detach().float().cpu()[0].abs().mean(0).numpy()
            ),
            need_map=tensor_to_hw(out_o["need_map"]),
            pixel_mask=(
                tensor_to_hw(out_o["pixel_mask"])
                if out_o["pixel_mask"] is not None
                else np.zeros((1, 1), dtype=np.float32)
            ),
            window_mask=(
                out_o["window_mask"].detach().float().cpu().numpy()
                if isinstance(out_o["window_mask"], torch.Tensor)
                else np.zeros((1, 1), dtype=np.float32)
            ),
        )
        if out_o["pre_temporal_feature"] is not None and out_o["need_map"] is not None:
            save_need_paper(
                out_o["pre_temporal_feature"],
                out_o["need_map"],
                dest,
                pixel_mask=out_o["pixel_mask"],
                window_mask=out_o["window_mask"],
                window_size=parse_window_size(args.window_size),
                gt_boxes=out_o["gt_box"],
                xlim=parse_optional_range(args.xlim),
                ylim=parse_optional_range(args.ylim),
            )
        meta = {
            **row,
            "model_temporal_only": str(only_dir / "net_epoch6.pth"),
            "model_temporal_concat": str(concat_dir / "net_epoch1.pth"),
            "test_dir": args.test_dir,
            "reason": (
                "temporal_only better TP/FP than concat, with clear need_map response"
            ),
        }
        with open(dest / "metrics.json", "w") as f:
            json.dump(meta, f, indent=2)

    for row in shortlist:
        dump_one(row, shortlist_dir / row["frame_key"])
    for row in selected:
        dump_one(row, selected_dir / row["frame_key"])

    print(
        f"[temporal] candidates={len(rows)} shortlist={len(shortlist)} selected={len(selected)}"
    )
    print(f"[temporal] outputs -> {out_root}")


def run_render_paper(args: argparse.Namespace, device: torch.device) -> None:
    """Re-infer selected frames and write publication-quality figures only."""
    key_sets = resolve_paper_keys(args)
    gating_keys = set(key_sets["gating"])
    temporal_keys = set(key_sets["temporal"])
    if args.mode == "gating":
        temporal_keys = set()
    elif args.mode == "temporal":
        gating_keys = set()
    if not gating_keys and not temporal_keys:
        raise RuntimeError(
            "No paper keys found. Pass --paper_keys or ensure selected/ dirs exist."
        )

    xlim = parse_optional_range(args.xlim)
    ylim = parse_optional_range(args.ylim)
    window_size = parse_window_size(args.window_size)
    out_root = Path(args.out_root)
    print(f"[paper] gating_keys={sorted(gating_keys)}")
    print(f"[paper] temporal_keys={sorted(temporal_keys)}")
    print(f"[paper] xlim={xlim} ylim={ylim} window_size={window_size}")

    gate_cache: Dict[str, Dict[str, Any]] = {}
    need_cache: Dict[str, Dict[str, Any]] = {}

    if gating_keys:
        full_dir = LOG_ROOT / "default_all_set"
        hypes = load_hypes(full_dir, args.test_dir)
        dataset = build_dataset(hypes, visualize=True, train=False)
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        model = create_and_load(full_dir, 16, hypes, dataset, device)
        corrupt_cfg = build_corrupt_config(
            scenario="uav",
            corrupt_type="mask_gaussian",
            level=args.corrupt_level,
            seed_base=args.corrupt_seed_base,
        )
        state: Dict[str, Any] = {}
        remaining = set(gating_keys)
        for i, batch_data in tqdm(enumerate(loader), total=len(loader), desc="paper-gating"):
            if args.max_batches is not None and i >= args.max_batches:
                break
            key = frame_key(batch_data)
            out = infer_one(
                model, batch_data, dataset, device, state, corrupt_cfg, True
            )
            if key in remaining:
                if out["corrupted_drone_bev"] is None or out["gate_drone"] is None:
                    print(f"[paper] gating missing tensors for {key}, skip")
                else:
                    gate_cache[key] = out
                remaining.discard(key)
            if not remaining:
                break
        if remaining:
            print(f"[paper] gating keys not reached: {sorted(remaining)}")

    if temporal_keys:
        only_dir = LOG_ROOT / "temporal_only"
        hypes_only = load_hypes(only_dir, args.test_dir)
        dataset = build_dataset(hypes_only, visualize=True, train=False)
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        model = create_and_load(only_dir, 6, hypes_only, dataset, device)
        state = {}
        remaining = set(temporal_keys)
        for i, batch_data in tqdm(
            enumerate(loader), total=len(loader), desc="paper-temporal"
        ):
            if args.max_batches is not None and i >= args.max_batches:
                break
            key = frame_key(batch_data)
            out = infer_one(model, batch_data, dataset, device, state, None, True)
            if key in remaining:
                if out["pre_temporal_feature"] is None or out["need_map"] is None:
                    print(f"[paper] temporal missing tensors for {key}, skip")
                else:
                    need_cache[key] = out
                remaining.discard(key)
            if not remaining:
                break
        if remaining:
            print(f"[paper] temporal keys not reached: {sorted(remaining)}")

    # Shared RMS-log energy percentile across all paper feature maps in this run.
    energies: List[np.ndarray] = []
    for out in gate_cache.values():
        energies.append(rms_log_energy(out["corrupted_drone_bev"]))
    for out in need_cache.values():
        energies.append(rms_log_energy(out["pre_temporal_feature"]))
    if not energies:
        raise RuntimeError("No tensors collected for paper rendering.")
    bounds = shared_percentile_bounds(energies, lo=2.0, hi=98.0)
    print(f"[paper] shared feature percentile bounds (2%,98%)={bounds}")

    gate_pngs: List[Path] = []
    need_pngs: List[Path] = []

    for key, out in gate_cache.items():
        dest = out_root / "gating_corruption" / "selected" / key
        dest.mkdir(parents=True, exist_ok=True)
        energy = rms_log_energy(out["corrupted_drone_bev"])
        feat_norm = apply_shared_norm(energy, bounds[0], bounds[1])
        _, png = save_gate_paper(
            out["corrupted_drone_bev"],
            out["gate_drone"],
            dest,
            gt_boxes=out["gt_box"],
            feat_norm=feat_norm,
            shared_bounds=bounds,
            xlim=xlim,
            ylim=ylim,
        )
        gate_pngs.append(png)
        np.savez_compressed(
            dest / "tensors_paper.npz",
            corrupted_drone_bev=feature_to_chw(out["corrupted_drone_bev"]),
            gate_drone=tensor_to_hw(out["gate_drone"]),
            feature_energy_log_rms=energy,
            feature_norm_shared=feat_norm,
            shared_p2=np.array([bounds[0]], dtype=np.float32),
            shared_p98=np.array([bounds[1]], dtype=np.float32),
        )
        # Keep original debug PNG untouched.
        print(f"[paper] wrote gating paper figs -> {dest}")

    for key, out in need_cache.items():
        dest = out_root / "temporal" / "selected" / key
        dest.mkdir(parents=True, exist_ok=True)
        energy = rms_log_energy(out["pre_temporal_feature"])
        feat_norm = apply_shared_norm(energy, bounds[0], bounds[1])
        _, png = save_need_paper(
            out["pre_temporal_feature"],
            out["need_map"],
            dest,
            pixel_mask=out["pixel_mask"],
            window_mask=out["window_mask"],
            window_size=window_size,
            gt_boxes=out["gt_box"],
            feat_norm=feat_norm,
            shared_bounds=bounds,
            xlim=xlim,
            ylim=ylim,
        )
        need_pngs.append(png)
        wm_np = np.zeros((1, 1), dtype=np.float32)
        if isinstance(out["window_mask"], torch.Tensor):
            t = out["window_mask"].detach().float().cpu()
            if t.dim() == 4:
                t = t[0, 0]
            elif t.dim() == 3:
                t = t[0]
            wm_np = t.numpy()
        np.savez_compressed(
            dest / "tensors_paper.npz",
            pre_temporal_feature=feature_to_chw(out["pre_temporal_feature"]),
            need_map=tensor_to_hw(out["need_map"]),
            pixel_mask=(
                tensor_to_hw(out["pixel_mask"])
                if out["pixel_mask"] is not None
                else np.zeros((1, 1), dtype=np.float32)
            ),
            window_mask=wm_np,
            feature_energy_log_rms=energy,
            feature_norm_shared=feat_norm,
            shared_p2=np.array([bounds[0]], dtype=np.float32),
            shared_p98=np.array([bounds[1]], dtype=np.float32),
        )
        print(f"[paper] wrote temporal paper figs -> {dest}")

    preview = out_root / "visualization_preview.png"
    if gate_pngs and need_pngs:
        save_visualization_preview(gate_pngs[0], need_pngs[0], preview)
        print(f"[paper] preview -> {preview}")
    elif gate_pngs:
        # Single-panel preview fallback.
        setup_paper_style()
        fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=150)
        ax.imshow(plt.imread(str(gate_pngs[0])))
        ax.axis("off")
        fig.savefig(preview, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"[paper] preview -> {preview}")
    elif need_pngs:
        setup_paper_style()
        fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=150)
        ax.imshow(plt.imread(str(need_pngs[0])))
        ax.axis("off")
        fig.savefig(preview, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"[paper] preview -> {preview}")


def main() -> None:
    args = parse_args()
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"device={device} mode={args.mode} test_dir={args.test_dir}")

    meta = {
        "mode": args.mode,
        "test_dir": args.test_dir,
        "max_batches": args.max_batches,
        "shortlist_k": args.shortlist_k,
        "select_k": args.select_k,
        "corrupt_level": args.corrupt_level,
        "corrupt_seed_base": args.corrupt_seed_base,
        "render_paper": bool(args.render_paper),
        "paper_keys": args.paper_keys,
        "xlim": args.xlim,
        "ylim": args.ylim,
        "window_size": args.window_size,
        "device": str(device),
    }
    with open(Path(args.out_root) / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.render_paper:
        run_render_paper(args, device)
        return

    if args.mode in {"gating", "both"}:
        run_gating(args, device)
    if args.mode in {"temporal", "both"}:
        run_temporal(args, device)


if __name__ == "__main__":
    main()
