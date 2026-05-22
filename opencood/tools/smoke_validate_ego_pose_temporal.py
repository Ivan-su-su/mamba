#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline ego-pose semantic smoke test.

This script does NOT modify runtime training codepaths. It:
1) Loads two consecutive frames from IntermediateFusionDatasetAirv2x using the SAME
   yaml parsing path as training (optional root_dir override for portability).
2) Projects each frame's ego lidar to the ego-frame (same transforms as dataset).
3) Builds T(prev_ego -> cur_ego) using x1_to_x2(prev_pose, cur_pose), matching train.py.
4) Saves a simple BEV scatter PNG + prints numeric alignment stats.

Requirements: matplotlib Agg backend (writes PNG only).
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from typing import Tuple

import matplotlib


def _is_airv2x_single_scenario_dir(path: str) -> bool:
    """Whether ``path`` is one sequence folder: children include ``timestamp_*`` dirs.

    AirV2X ``BaseDataset`` expects ``params['root_dir']`` to list **scenario**
    subdirectories; each scenario is parsed via ``parse_seq`` and contains
    ``timestamp_<idx>`` folders. Passing a scenario path yields timestamp dirs
    mis-scanned as scenarios and can produce empty parses / IndexError.

    Args:
        path: Absolute path to probe.

    Returns:
        True if ``path`` looks like ``.../<scenario_date>/timestamp_*/...``.
    """
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if not name.startswith("timestamp_"):
            continue
        if os.path.isdir(os.path.join(path, name)):
            return True
    return False


def _lift_root_dir_if_scenario_leaf(
    root_dir: str, scenario_index: int
) -> Tuple[str, int]:
    """If ``root_dir`` is a single scenario folder, return parent + index of that folder.

    Args:
        root_dir: Dataset root as provided by the user (may be wrong level).
        scenario_index: Requested scenario index; ignored when lift applies
            (replaced by the index of the scenario folder under the parent).

    Returns:
        ``(effective_root, effective_scenario_index)``.
    """
    root_dir = os.path.abspath(root_dir)
    if not _is_airv2x_single_scenario_dir(root_dir):
        return root_dir, scenario_index

    parent = os.path.dirname(root_dir)
    base = os.path.basename(root_dir)
    siblings = sorted(
        x for x in os.listdir(parent) if os.path.isdir(os.path.join(parent, x))
    )
    if base not in siblings:
        return root_dir, scenario_index

    new_index = siblings.index(base)
    print(
        "[ego_pose_smoke] 检测到传入的是单个 scenario 目录（含 timestamp_* 子目录），"
        f"已自动将 root_dir 提升为训练根目录:\n"
        f"  {root_dir}\n  -> {parent}\n"
        f"  对应 scenario_index={new_index}（原参数为 {scenario_index}，已覆盖）。"
    )
    return parent, new_index


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual + numeric ego_pose smoke test")
    parser.add_argument(
        "--hypes_yaml",
        type=str,
        required=True,
        help="Training yaml (same format as train.py -y)",
    )
    parser.add_argument(
        "--scenario_index",
        type=int,
        default=0,
        help="Which scenario folder (0-based ordering used by dataset).",
    )
    parser.add_argument(
        "--cur_timestamp_index",
        type=int,
        default=1,
        help=(
            "Local timestamp index within the scenario for the CURRENT frame. "
            "PREV becomes cur_index-1. Must be >=1."
        ),
    )
    parser.add_argument(
        "--sample_points",
        type=int,
        default=8192,
        help="Random subsample size for qualitative plots / timings.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/tmp/airv2x_ego_pose_smoke",
        help="Where to save PNG plots.",
    )
    parser.add_argument(
        "--root_dir_override",
        type=str,
        default="",
        help=(
            "Optional: override param['root_dir'] after loading yaml. "
            "Must be the **training root** that contains scenario subfolders "
            "(date-named dirs), not a single scenario path. If you only pass "
            "one scenario folder (its children are timestamp_*), this script "
            "auto-lifts to the parent directory and fixes scenario_index."
        ),
    )
    return parser.parse_args()


def scene_index(dataset, scenario_index: int) -> Tuple[int, int]:
    """Return global [start,end) indices for scenario_index."""
    if scenario_index < 0 or scenario_index >= len(dataset.len_record):
        raise IndexError(f"scenario_index out of range: {scenario_index}")
    start = 0 if scenario_index == 0 else dataset.len_record[scenario_index - 1]
    end = dataset.len_record[scenario_index]
    return int(start), int(end)


def _ego_frame_points_from_processed_dataset(
    dataset, scenario_index: int, local_ts: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return xyz in ego-frame for ego vehicle lidar + ego_lidar_pose (6dof list)."""
    start, end = scene_index(dataset, scenario_index)
    global_idx = start + local_ts
    if global_idx < start or global_idx >= end:
        raise IndexError(
            "cur_timestamp_index leads to invalid global dataset index "
            f"(global_idx={global_idx}, scene_range=[{start},{end}))."
        )

    base_dict, scenario_from_retrieve, timestamp_key = dataset.retrieve_base_data(
        global_idx,
        cur_ego_pos_flag=dataset.cur_ego_pose_flag,
    )
    if int(scenario_from_retrieve) != int(scenario_index):
        raise RuntimeError(
            f"scenario mismatch: expected {scenario_index}, got {scenario_from_retrieve}"
        )

    ego_pose = []
    ego_cav_base = None
    for cav_id, cav_content in base_dict.items():
        if cav_content["ego"]:
            ego_cav_base = cav_content
            ego_pose = cav_content["params"]["delay_ego_lidar_pose"]
            break
    if ego_cav_base is None:
        raise RuntimeError("No ego vehicle found in base_data_dict.")

    from opencood.utils import box_utils  # noqa: WPS433 (runtime import mirrors training)
    from opencood.utils.pcd_utils import (
        mask_ego_points,
        mask_points_by_range,
        shuffle_points,
    )

    lidar_np = ego_cav_base["lidar_np"]
    transformation_matrix = ego_cav_base["params"]["transformation_matrix"].astype(
        np.float32
    )

    lidar_np = shuffle_points(lidar_np)
    lidar_np = mask_ego_points(lidar_np)
    if dataset.proj_first:
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(
            lidar_np[:, :3], transformation_matrix
        )
    lidar_np = mask_points_by_range(
        lidar_np, dataset.params["preprocess"]["cav_lidar_range"]
    )
    xyz = lidar_np[:, :3].astype(np.float32)

    ego_pose_np = np.asarray(ego_pose, dtype=np.float32).reshape(-1).tolist()
    if len(ego_pose_np) != 6:
        raise RuntimeError(f"Unexpected ego_pose length={len(ego_pose_np)}, expected 6.")
    return xyz, ego_pose_np


def _subsample(xyz: np.ndarray, n: int, seed: int) -> np.ndarray:
    if xyz.shape[0] == 0:
        return xyz
    if xyz.shape[0] <= n:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=n, replace=False)
    return xyz[idx]


def mean_nn_distance_xyz(a_xyz: np.ndarray, b_xyz: np.ndarray, sample_k: int) -> float:
    """Mean nearest-neighbor distance from a_xyz -> b_xyz (CPU brute force on subsample)."""
    if a_xyz.shape[0] == 0 or b_xyz.shape[0] == 0:
        return float("nan")
    a_sub = _subsample(a_xyz, sample_k, seed=123)
    b_sub = _subsample(b_xyz, sample_k, seed=456)
    diff = a_sub[:, None, :] - b_sub[None, :, :]
    d2 = np.sum(diff * diff, axis=2).min(axis=1)
    return float(np.mean(np.sqrt(d2)))


def plot_bev(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    plot_path: str,
    *,
    alpha: float,
    marker_size: float,
) -> None:
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(xyz_a[:, 0], xyz_a[:, 1], s=marker_size, c="tab:blue", alpha=alpha, label="warped_prev")
    ax.scatter(xyz_b[:, 0], xyz_b[:, 1], s=marker_size, c="tab:orange", alpha=alpha, label="curr")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3)
    ax.set_xlabel("x (ego)")
    ax.set_ylabel("y (ego)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _ensure_mambafusion_grid_size(cfg: dict) -> None:
    """Fill ``GRID_SIZE`` when yaml omits it (IntermediateFusionDatasetAirv2x mambafusion path).

    Some configs only list ``VOXEL_SIZE`` and ``POINT_CLOUD_RANGE``; grid dimensions are
    ``round((max - min) / voxel)`` per axis, matching ``load_airv2x_params`` grids.

    Args:
        cfg: Loaded hyperparameter dict (modified in place).
    """
    if cfg.get("model", {}).get("core_method") != "airv2x_mambafusion":
        return
    if cfg.get("GRID_SIZE"):
        return
    pcr = np.asarray(cfg["POINT_CLOUD_RANGE"], dtype=np.float64)
    voxel = cfg.get("VOXEL_SIZE")
    if voxel is None:
        voxel = cfg["preprocess"]["args"]["voxel_size"]
    voxel = np.asarray(voxel, dtype=np.float64)
    span = pcr[3:6] - pcr[0:3]
    grid = np.round(span / voxel).astype(np.float32)
    cfg["GRID_SIZE"] = grid.tolist()
    print(
        "[ego_pose_smoke] yaml 缺少 GRID_SIZE；已从 POINT_CLOUD_RANGE 与 voxel 推断: "
        f"{cfg['GRID_SIZE']}"
    )


def main() -> None:
    args = parse_args()

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    import sys

    sys.path.insert(0, root_path)

    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset

    from opencood.utils.transformation_utils import x1_to_x2

    hypes = load_yaml(args.hypes_yaml, None)
    cfg = copy.deepcopy(hypes)

    if args.root_dir_override:
        cfg["root_dir"] = args.root_dir_override

    root_resolved, scenario_index_eff = _lift_root_dir_if_scenario_leaf(
        cfg["root_dir"], args.scenario_index
    )
    cfg["root_dir"] = root_resolved

    _ensure_mambafusion_grid_size(cfg)

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = build_dataset(cfg, visualize=False, train=True)

    scenario_start, scenario_end = scene_index(dataset, scenario_index_eff)
    scenario_len = scenario_end - scenario_start
    if scenario_len <= 1:
        raise RuntimeError("Scenario length must be >=2 for temporal pair comparison.")

    if args.cur_timestamp_index <= 0 or args.cur_timestamp_index >= scenario_len:
        raise IndexError(
            f"--cur_timestamp_index must be in [1,{scenario_len-1}], "
            f"got {args.cur_timestamp_index}"
        )

    prev_ts = args.cur_timestamp_index - 1
    cur_ts = args.cur_timestamp_index

    prev_xyz_w, prev_pose = _ego_frame_points_from_processed_dataset(
        dataset, scenario_index_eff, prev_ts
    )
    cur_xyz_w, cur_pose = _ego_frame_points_from_processed_dataset(
        dataset, scenario_index_eff, cur_ts
    )

    T_np = x1_to_x2(prev_pose, cur_pose).astype(np.float32)
    T_inv = np.linalg.inv(T_np)
    err_inv = float(np.max(np.abs(T_inv @ T_np - np.eye(4, dtype=np.float32))))

    prev_xyz_sub = _subsample(prev_xyz_w.astype(np.float32), args.sample_points, seed=0)
    ones = np.ones((prev_xyz_sub.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([prev_xyz_sub, ones], axis=1)  # (N,4)
    warped = (homo @ T_np.T)[:, :3]

    mean_nn = mean_nn_distance_xyz(warped, cur_xyz_w, sample_k=min(4096, args.sample_points))

    warp_plot = os.path.join(
        args.out_dir,
        (
            f"bev_overlay_sc{scenario_index_eff}_prev{prev_ts}_cur{cur_ts}_warp.png"
        ),
    )
    naive_plot = os.path.join(
        args.out_dir,
        (
            f"bev_overlay_sc{scenario_index_eff}_prev{prev_ts}_cur{cur_ts}_naive_no_pose.png"
        ),
    )

    plot_bev(warped, cur_xyz_w, warp_plot, alpha=0.08, marker_size=1.5)
    plot_bev(prev_xyz_sub, cur_xyz_w, naive_plot, alpha=0.08, marker_size=1.5)

    report_path = os.path.join(args.out_dir, "metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f_handle:
        f_handle.write("ego_pose smoke test metrics\n")
        f_handle.write(f"root_dir_resolved={root_resolved}\n")
        f_handle.write(f"scenario_index_effective={scenario_index_eff}\n")
        f_handle.write(f"hypes_yaml={args.hypes_yaml}\n")
        f_handle.write(f"scenario_index_arg={args.scenario_index}\n")
        f_handle.write(f"prev_ts={prev_ts} cur_ts={cur_ts}\n")
        f_handle.write(f"prev_xyz_count={prev_xyz_w.shape[0]} cur_xyz_count={cur_xyz_w.shape[0]}\n")
        f_handle.write(f"mean_nn_xyz_meters_aligned={mean_nn}\n")
        f_handle.write(f"T_inv_consistency_max_abs_error={err_inv}\n")
        f_handle.write("T_np (prev->curr) first row=\n")
        f_handle.write(str(T_np[0].tolist()) + "\n")

    print("[ego_pose_smoke]")
    print(f"  saved png: {warp_plot}")
    print(f"  control png (wrong if pose sign flips badly): {naive_plot}")
    print(f"  mean_nn_xyz_aligned_meters ~= {mean_nn:.4f} (lower is better for static env)")
    print(f"  inv_consistency max_abs_error ~= {err_inv:.3e} (expect ~1e-6)")
    print(f"  report: {report_path}")
    print(f"  T(prev->curr) shape=(1,4,4), matching train ego_pose tensor stacking")


if __name__ == "__main__":
    main()
