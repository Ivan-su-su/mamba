"""Small helpers for temporal queue/streaming state."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np
import torch

from opencood.utils.transformation_utils import x1_to_x2


def get_temporal_training_cfg(hypes: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized temporal configuration.

    Args:
        hypes: Loaded yaml configuration.

    Returns:
        Normalized temporal configuration.

    Raises:
        ValueError: If temporal mode is unsupported.
    """
    cfg = hypes.get("temporal_training", {}) or {}
    mode = str(cfg.get("mode", "streaming")).lower()
    if mode not in {"queue", "streaming"}:
        raise ValueError(
            f"temporal_training.mode must be 'queue' or 'streaming', got {mode!r}"
        )
    return {
        "enable": bool(cfg.get("enable", False)),
        "load_from": str(cfg.get("load_from", "") or ""),
        "mode": mode,
    }


def is_mambafusion_model(hypes: Dict[str, Any]) -> bool:
    """Check whether the current config builds an AirV2X MambaFusion model."""
    name = hypes.get("name", "").lower()
    core_method = str(hypes.get("model", {}).get("core_method", "")).lower()
    return "mambafusion" in name or "mambafusion" in core_method


def _first_meta_value(frame: Dict[str, Any], key: str) -> Any:
    value = frame[key]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.flatten()[0].item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _timestamp_to_int(timestamp: Any) -> Optional[int]:
    text = str(timestamp)
    match = re.search(r"(\d+)$", text)
    if match is None:
        return None
    return int(match.group(1))


def _identity_pose_tensor(device: torch.device) -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)


def _relative_pose_tensor(
    prev_pose: Any,
    cur_pose: Any,
    device: torch.device,
) -> torch.Tensor:
    matrix = x1_to_x2(prev_pose, cur_pose).astype(np.float32)
    return torch.from_numpy(matrix).to(device=device).unsqueeze(0)


def frames_are_temporally_contiguous(
    prev_frame: Dict[str, Any],
    cur_frame: Dict[str, Any],
) -> bool:
    """Check whether two collated frames are consecutive in the same scenario."""
    prev_scenario = _first_meta_value(prev_frame, "scenario_index_list")
    cur_scenario = _first_meta_value(cur_frame, "scenario_index_list")
    if int(prev_scenario) != int(cur_scenario):
        return False

    prev_timestamp = _timestamp_to_int(
        _first_meta_value(prev_frame, "timestamp_key_list")
    )
    cur_timestamp = _timestamp_to_int(
        _first_meta_value(cur_frame, "timestamp_key_list")
    )
    if prev_timestamp is None or cur_timestamp is None:
        return False
    return cur_timestamp == prev_timestamp + 1


def populate_temporal_fields(
    cur_frame: Dict[str, Any],
    device: torch.device,
    prev_frame: Optional[Dict[str, Any]] = None,
    force_reset: bool = False,
) -> bool:
    """Attach ``ego_pose`` and ``temporal_reset`` to one collated frame."""
    valid_prev = (
        prev_frame is not None
        and not force_reset
        and frames_are_temporally_contiguous(prev_frame, cur_frame)
    )
    if valid_prev:
        prev_pose = _first_meta_value(prev_frame, "ego_lidar_pose_list")
        cur_pose = _first_meta_value(cur_frame, "ego_lidar_pose_list")
        cur_frame["ego_pose"] = _relative_pose_tensor(prev_pose, cur_pose, device)
        cur_frame["temporal_reset"] = False
    else:
        cur_frame["ego_pose"] = _identity_pose_tensor(device)
        cur_frame["temporal_reset"] = True
    return valid_prev


def update_temporal_state(
    temporal_state: Dict[str, Any],
    frame: Dict[str, Any],
) -> None:
    """Store current frame metadata needed for the next streaming step."""
    temporal_state["prev_frame"] = {
        "scenario_index_list": frame["scenario_index_list"],
        "timestamp_key_list": frame["timestamp_key_list"],
        "ego_lidar_pose_list": frame["ego_lidar_pose_list"],
    }
