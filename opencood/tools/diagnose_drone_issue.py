#!/usr/bin/env python3
"""Diagnose drone HEAL single failure: GT z, anchor Δz, camera extrinsic."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.utils.transformation_utils import get_abs_world_pose, x1_to_x2
from opencood.utils.box_utils import create_bbx, corner_to_center, mask_boxes_outside_range_numpy


DATA_ROOT = Path("/home/dell/suyi/AirV2X-Perception/train/train")
OUT = Path("/home/dell/suyi/AirV2X-Perception_copy/opencood/logs/diagnose_drone_issue.txt")

VEH_RANGE = [-140.8, -40, -3, 140.8, 40, 1]
DRONE_RANGE = [-140.8, -40, -150, 140.8, 40, -6]
ANCHOR_Z = -1.0
ANCHOR_H = 1.56


def load_meta(agent_dir: Path) -> Dict[str, Any]:
    meta_path = agent_dir / "metadata.yaml"
    if not meta_path.exists():
        # try json/pkl variants
        for name in ("metadata.pkl", "meta.yaml", "params.yaml"):
            p = agent_dir / name
            if p.exists():
                meta_path = p
                break
    if meta_path.suffix == ".pkl":
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    # yaml
    import yaml

    with open(meta_path, "r") as f:
        return yaml.safe_load(f)


def load_objects(agent_dir: Path) -> Dict[str, Any]:
    for name in ("objects.pkl", "objects.yaml"):
        p = agent_dir / name
        if p.exists():
            if p.suffix == ".pkl":
                with open(p, "rb") as f:
                    return pickle.load(f)
            import yaml

            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"no objects in {agent_dir}")


def find_agents(scenario: Path, timestamp: str, agent_type: str, limit: int = 3) -> List[Path]:
    ts_dir = scenario / timestamp
    if not ts_dir.exists():
        return []
    found = []
    for agent_dir in sorted(ts_dir.glob("agent_*")):
        try:
            meta = load_meta(agent_dir)
        except Exception:
            continue
        if meta.get("agent_type") == agent_type:
            found.append(agent_dir)
        if len(found) >= limit:
            break
    return found


def project_objects_to_lidar(
    objects: Dict[str, Any],
    lidar_pose: List[float],
    lidar_range: List[float],
    order: str = "hwl",
) -> np.ndarray:
    """Return (N, 7) boxes in lidar frame."""
    boxes = []
    for object_id, object_content in objects.items():
        if not isinstance(object_content, dict):
            continue
        if "location" not in object_content:
            continue
        location = object_content["location"][:3]
        angle = [
            object_content["location"][3],
            object_content["location"][4],
            object_content["location"][5],
        ]
        center = object_content["center"]
        extent = object_content["extent"]
        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            angle[0],
            angle[1],
            angle[2],
        ]
        object2lidar = x1_to_x2(object_pose, lidar_pose)
        bbx = create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar, _ = mask_boxes_outside_range_numpy(
            bbx_lidar, lidar_range, order, return_mask=True
        )
        if bbx_lidar.shape[0] > 0:
            boxes.append(bbx_lidar[0])
    if not boxes:
        return np.zeros((0, 7), dtype=np.float64)
    return np.stack(boxes, axis=0)


def lidar_world_pose(meta: Dict[str, Any]) -> List[float]:
    lidar_rel = meta["lidar"]["lidar_pose"]
    ego_pos = meta["odometry"]["ego_pos"]
    return get_abs_world_pose(lidar_rel, ego_pos)


def analyze_agent(agent_dir: Path, agent_type: str, lidar_range: List[float]) -> Dict[str, Any]:
    meta = load_meta(agent_dir)
    objects = load_objects(agent_dir)
    # filter like dataset if possible
    try:
        from opencood.utils.airv2x_utils import filter_objects

        objects = filter_objects(objects)
    except Exception:
        pass

    pose = lidar_world_pose(meta)
    boxes = project_objects_to_lidar(objects, pose, lidar_range, order="hwl")

    result: Dict[str, Any] = {
        "agent_dir": str(agent_dir),
        "agent_type": agent_type,
        "ego_pos_z": float(meta["odometry"]["ego_pos"][2]),
        "lidar_pose_rel": meta["lidar"]["lidar_pose"],
        "n_boxes": int(boxes.shape[0]),
    }
    if boxes.shape[0] > 0:
        z = boxes[:, 2]
        result.update(
            {
                "gt_z_mean": float(z.mean()),
                "gt_z_min": float(z.min()),
                "gt_z_max": float(z.max()),
                "gt_z_std": float(z.std()),
                "delta_z_vs_anchor": float(((z - ANCHOR_Z) / ANCHOR_H).mean()),
                "delta_z_abs_mean": float(np.abs((z - ANCHOR_Z) / ANCHOR_H).mean()),
                "delta_z_abs_max": float(np.abs((z - ANCHOR_Z) / ANCHOR_H).max()),
            }
        )
    else:
        result["note"] = "no boxes in range"

    # camera extrinsic check
    if agent_type == "drone" and "bev_camera" in meta:
        cam = meta["bev_camera"]
        ext = np.array(cam["extrinsic"], dtype=np.float64)
        cam_rel = cam["cords"]
        drone_pose = meta["odometry"]["ego_pos"]
        cam_abs = get_abs_world_pose(cam_rel, drone_pose)
        # old commented path: extrinsic = x1_to_x2(cam_abs, drone_pose)
        # Note: x1_to_x2(a,b) transforms from a to b
        computed = x1_to_x2(cam_abs, drone_pose)
        # Also try cam -> lidar
        lidar_abs = pose
        computed_cam2lidar = x1_to_x2(cam_abs, lidar_abs)

        result["bev_cam_cords"] = cam_rel
        result["bev_cam_extrinsic_from_data"] = ext.tolist()
        result["bev_cam_extrinsic_computed_cam2drone"] = computed.tolist()
        result["bev_cam_extrinsic_computed_cam2lidar"] = computed_cam2lidar.tolist()
        result["ext_diff_fro_vs_cam2drone"] = float(np.linalg.norm(ext - computed))
        result["ext_diff_fro_vs_cam2lidar"] = float(
            np.linalg.norm(ext - computed_cam2lidar)
        )
        # translation of extrinsic (last column or depending on convention)
        result["ext_data_t"] = ext[:3, 3].tolist()
        result["ext_cam2lidar_t"] = computed_cam2lidar[:3, 3].tolist()
        result["cam_abs_z"] = float(cam_abs[2])
        result["drone_pose_z"] = float(drone_pose[2])
        result["lidar_abs_z"] = float(lidar_abs[2])

    if agent_type == "vehicle" and "front_camera" in meta:
        cam = meta["front_camera"]
        ext = np.array(cam["extrinsic"], dtype=np.float64)
        result["veh_front_ext_t"] = ext[:3, 3].tolist()

    return result


def main() -> None:
    lines: List[str] = []
    lines.append("=== Drone HEAL issue diagnosis ===\n")

    scenarios = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[:4]
    vehicle_stats = []
    drone_stats = []

    for scenario in scenarios:
        timestamps = sorted([p.name for p in scenario.iterdir() if p.name.startswith("timestamp_")])
        if not timestamps:
            continue
        # sample a few timestamps
        for ts in timestamps[:: max(1, len(timestamps) // 3)][:3]:
            for agent_dir in find_agents(scenario, ts, "vehicle", limit=1):
                try:
                    r = analyze_agent(agent_dir, "vehicle", VEH_RANGE)
                    vehicle_stats.append(r)
                except Exception as e:
                    lines.append(f"[vehicle fail] {agent_dir}: {e}\n")
            for agent_dir in find_agents(scenario, ts, "drone", limit=2):
                try:
                    r = analyze_agent(agent_dir, "drone", DRONE_RANGE)
                    drone_stats.append(r)
                except Exception as e:
                    lines.append(f"[drone fail] {agent_dir}: {e}\n")

    def summarize(name: str, stats: List[Dict[str, Any]]) -> None:
        lines.append(f"\n--- {name}: {len(stats)} samples ---\n")
        zs = [s["gt_z_mean"] for s in stats if "gt_z_mean" in s]
        dz = [s["delta_z_abs_mean"] for s in stats if "delta_z_abs_mean" in s]
        dzmax = [s["delta_z_abs_max"] for s in stats if "delta_z_abs_max" in s]
        if zs:
            lines.append(
                f"GT z mean across samples: mean={np.mean(zs):.3f}, "
                f"min={np.min(zs):.3f}, max={np.max(zs):.3f}\n"
            )
        if dz:
            lines.append(
                f"|Δz| vs anchor(z=-1,h=1.56): mean={np.mean(dz):.3f}, "
                f"max_of_max={np.max(dzmax):.3f}\n"
            )
        # print a few examples
        for s in stats[:3]:
            lines.append(json.dumps({k: v for k, v in s.items() if "extrinsic_computed" not in k and "extrinsic_from_data" not in k}, indent=2))
            lines.append("\n")

    summarize("VEHICLE", vehicle_stats)
    summarize("DRONE", drone_stats)

    # extrinsic focus
    lines.append("\n--- DRONE camera extrinsic consistency ---\n")
    diffs_drone = [s["ext_diff_fro_vs_cam2drone"] for s in drone_stats if "ext_diff_fro_vs_cam2drone" in s]
    diffs_lidar = [s["ext_diff_fro_vs_cam2lidar"] for s in drone_stats if "ext_diff_fro_vs_cam2lidar" in s]
    if diffs_drone:
        lines.append(
            f"||ext_data - x1_to_x2(cam, drone_pose)||_F : "
            f"mean={np.mean(diffs_drone):.4f}, max={np.max(diffs_drone):.4f}\n"
        )
        lines.append(
            f"||ext_data - x1_to_x2(cam, lidar_pose)||_F : "
            f"mean={np.mean(diffs_lidar):.4f}, max={np.max(diffs_lidar):.4f}\n"
        )
        # show one full matrix pair
        s0 = next(s for s in drone_stats if "bev_cam_extrinsic_from_data" in s)
        lines.append("Example data extrinsic:\n")
        lines.append(np.array2string(np.array(s0["bev_cam_extrinsic_from_data"]), precision=4))
        lines.append("\nExample computed cam2lidar:\n")
        lines.append(np.array2string(np.array(s0["bev_cam_extrinsic_computed_cam2lidar"]), precision=4))
        lines.append("\nExample computed cam2drone:\n")
        lines.append(np.array2string(np.array(s0["bev_cam_extrinsic_computed_cam2drone"]), precision=4))
        lines.append("\n")
        lines.append(
            f"cam_abs_z={s0['cam_abs_z']:.2f}, drone_z={s0['drone_pose_z']:.2f}, "
            f"lidar_z={s0['lidar_abs_z']:.2f}, cords={s0['bev_cam_cords']}\n"
        )

    # verdict
    lines.append("\n=== VERDICT ===\n")
    if vehicle_stats and drone_stats:
        v_dz = np.mean([s["delta_z_abs_mean"] for s in vehicle_stats if "delta_z_abs_mean" in s])
        d_dz = np.mean([s["delta_z_abs_mean"] for s in drone_stats if "delta_z_abs_mean" in s])
        lines.append(f"Vehicle mean |Δz|={v_dz:.3f}; Drone mean |Δz|={d_dz:.3f}\n")
        if d_dz > 5 * max(v_dz, 0.1):
            lines.append(
                "FINDING-A: Anchor z=-1 is incompatible with drone-ego GT z "
                "(Δz much larger than vehicle). This alone can break drone single training.\n"
            )
        else:
            lines.append("FINDING-A: Anchor Δz gap not as severe as expected; check other causes.\n")

    if diffs_lidar:
        if np.mean(diffs_lidar) > 1.0:
            lines.append(
                "FINDING-B: Dataset bev_camera.extrinsic DOES NOT match "
                "x1_to_x2(cam_abs, lidar_abs). Camera branch may be misaligned.\n"
            )
        elif np.mean(diffs_drone) < 0.1:
            lines.append(
                "FINDING-B: extrinsic matches cam->drone_pose; check if code expects cam->lidar.\n"
            )
        else:
            lines.append(
                f"FINDING-B: extrinsic partially consistent "
                f"(cam2drone_diff={np.mean(diffs_drone):.3f}, "
                f"cam2lidar_diff={np.mean(diffs_lidar):.3f}).\n"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines))
    print("".join(lines))
    print(f"\nWrote report to {OUT}")


if __name__ == "__main__":
    main()
