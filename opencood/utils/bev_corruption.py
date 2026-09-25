"""Deterministic BEV feature corruption for reliability ablation.

Used only at inference. Does not alter training or default model behavior.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SCENARIO_TO_AGENTS: Dict[str, List[str]] = {
    "clean": [],
    "rsu": ["rsu"],
    "uav": ["drone"],
    "rsu_uav": ["rsu", "drone"],
}

GAUSSIAN_SIGMA: Dict[str, float] = {
    "light": 0.1,
    "medium": 0.3,
    "heavy": 0.5,
}

MASK_RATIO: Dict[str, float] = {
    "light": 0.1,
    "medium": 0.3,
    "heavy": 0.5,
}

CANONICAL_MASK_HW = (200, 200)
DEFAULT_PATCH_SIZE = 16
AGENT_ORDER = ("vehicle", "rsu", "drone")


@dataclass(frozen=True)
class CorruptConfig:
    """Inference-time BEV corruption settings."""

    scenario: str = "clean"
    corrupt_type: str = "none"  # none | gaussian | mask | mask_gaussian | zero
    level: str = "medium"  # light | medium | heavy (ignored for zero)
    agents: tuple = ()
    seed_base: int = 0
    patch_size: int = DEFAULT_PATCH_SIZE

    @property
    def enabled(self) -> bool:
        return (
            self.corrupt_type != "none"
            and len(self.agents) > 0
            and self.scenario != "clean"
        )


# Aliases so CLI can use mask+gauss / gaussian_mask / drop etc.
_CORRUPT_TYPE_ALIASES: Dict[str, str] = {
    "none": "none",
    "gaussian": "gaussian",
    "mask": "mask",
    "mask_gaussian": "mask_gaussian",
    "gaussian_mask": "mask_gaussian",
    "mask+gauss": "mask_gaussian",
    "mask+gaussian": "mask_gaussian",
    "mask_gauss": "mask_gaussian",
    "zero": "zero",
    "drop": "zero",
    "ablate": "zero",
}


def build_corrupt_config(
    scenario: str = "clean",
    corrupt_type: str = "none",
    level: str = "medium",
    seed_base: int = 0,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> CorruptConfig:
    """Build config from CLI-like arguments."""
    scenario = scenario.lower()
    corrupt_type = corrupt_type.lower().replace(" ", "")
    level = level.lower()
    if scenario not in SCENARIO_TO_AGENTS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if corrupt_type not in _CORRUPT_TYPE_ALIASES:
        raise ValueError(
            f"Unknown corrupt_type: {corrupt_type}. "
            f"Valid: {sorted(set(_CORRUPT_TYPE_ALIASES.values()))}"
        )
    corrupt_type = _CORRUPT_TYPE_ALIASES[corrupt_type]
    if level not in GAUSSIAN_SIGMA:
        raise ValueError(f"Unknown level: {level}")
    agents = tuple(SCENARIO_TO_AGENTS[scenario])
    if scenario == "clean" or corrupt_type == "none":
        corrupt_type = "none"
        agents = ()
    return CorruptConfig(
        scenario=scenario,
        corrupt_type=corrupt_type,
        level=level,
        agents=agents,
        seed_base=int(seed_base),
        patch_size=int(patch_size),
    )


def should_drop_agent(cfg: Optional[CorruptConfig], agent: str) -> bool:
    """Return True if ``agent`` should be excluded (``corrupt_type=zero`` ablation)."""
    if cfg is None or not getattr(cfg, "enabled", False):
        return False
    if cfg.corrupt_type != "zero":
        return False
    return agent in cfg.agents


def filter_agents_for_drop(
    available_agents: Sequence[str],
    cfg: Optional[CorruptConfig],
) -> List[str]:
    """Drop collaborators listed in cfg when ``corrupt_type`` is ``zero``.

    Used by MambaFusion ``available_agents`` so fusion receives
    ``feature_drone=None`` / ``feature_rsu=None`` instead of all-zero BEV.
    """
    if cfg is None or not getattr(cfg, "enabled", False):
        return list(available_agents)
    if cfg.corrupt_type != "zero":
        return list(available_agents)
    drop = set(cfg.agents)
    return [a for a in available_agents if a not in drop]


def extract_sample_id(batch_data: Dict[str, Any]) -> str:
    """Extract a stable sample id from batch metadata."""
    ego = batch_data.get("ego", batch_data)
    meta_list = ego.get("metadata_path_list", None)
    if meta_list is not None and len(meta_list) > 0:
        path = meta_list[0]
        if isinstance(path, (list, tuple)):
            path = path[0]
        path = str(path)
        m = re.search(r"(timestamp_\d+)", path)
        if m:
            return m.group(1)
        return path
    for key in ("scenario_name", "timestamp", "frame_id"):
        if key in ego:
            return str(ego[key])
    return "unknown_sample"


def _stable_seed(
    sample_id: str,
    agents: Sequence[str],
    corrupt_type: str,
    level: str,
    seed_base: int,
) -> int:
    payload = "|".join(
        [str(sample_id), ",".join(sorted(agents)), corrupt_type, level, str(seed_base)]
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _make_generator(
    sample_id: str,
    agents: Sequence[str],
    corrupt_type: str,
    level: str,
    seed_base: int,
    agent: str,
) -> torch.Generator:
    seed = _stable_seed(sample_id, agents, corrupt_type, level, seed_base)
    seed = (seed + int(hashlib.md5(agent.encode()).hexdigest()[:4], 16)) % (2**31 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _gaussian_noise(
    feat: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return corrupted feature and per-pixel noise intensity `[N,1,H,W]` in ~[0,1]."""
    std = feat.detach().float().std().clamp_min(1e-6)
    noise = torch.randn(
        feat.shape,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    ).to(device=feat.device, dtype=feat.dtype)
    scaled = (sigma * std) * noise
    intensity = scaled.detach().abs().mean(dim=1, keepdim=True)
    denom = intensity.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    intensity = (intensity / denom).clamp(0.0, 1.0)
    return feat + scaled, intensity


def _random_mask(
    feat: torch.Tensor,
    ratio: float,
    patch_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return masked feature and corrupt binary map `[1,1,H,W]` (1=corrupted)."""
    _, _, h, w = feat.shape
    ch, cw = CANONICAL_MASK_HW
    n_ph = max(1, (ch + patch_size - 1) // patch_size)
    n_pw = max(1, (cw + patch_size - 1) // patch_size)
    patch_rand = torch.rand((n_ph, n_pw), generator=generator)
    patch_keep = (patch_rand >= ratio).float()  # 1=keep
    keep = patch_keep.repeat_interleave(patch_size, dim=0)[:ch]
    keep = keep.repeat_interleave(patch_size, dim=1)[:, :cw]
    # Pad if patch grid does not exactly cover canonical HW.
    if keep.shape[0] < ch or keep.shape[1] < cw:
        keep = F.pad(keep, (0, cw - keep.shape[1], 0, ch - keep.shape[0]), value=1.0)
    keep = keep.view(1, 1, ch, cw)
    keep = F.interpolate(keep, size=(h, w), mode="nearest")
    keep = keep.to(device=feat.device, dtype=feat.dtype)
    corrupt_map = 1.0 - keep  # 1=corrupted/dropped
    # Broadcast to N if needed
    if feat.shape[0] > 1:
        corrupt_map = corrupt_map.expand(feat.shape[0], -1, -1, -1).clone()
    return feat * keep, corrupt_map


def corrupt_bev(
    feat: torch.Tensor,
    cfg: Optional[CorruptConfig],
    sample_id: str,
    agent: str,
    maps_out: Optional[Dict[str, torch.Tensor]] = None,
    map_key: Optional[str] = None,
) -> torch.Tensor:
    """Apply deterministic corruption to one agent BEV feature.

    Args:
        feat: BEV feature `[B,C,H,W]` or `[N,C,H,W]`.
        cfg: Corruption config; no-op if None/disabled.
        sample_id: Stable sample identifier.
        agent: Agent type name (`rsu` / `drone` / ...).
        maps_out: Optional dict to store spatial corrupt maps.
        map_key: Key used in ``maps_out`` (defaults to ``agent``).

    Returns:
        Corrupted feature (or original if not applicable).
    """
    if cfg is None or not cfg.enabled:
        return feat
    if agent not in cfg.agents:
        return feat
    if feat is None or not torch.is_tensor(feat) or feat.numel() == 0:
        return feat
    if feat.dim() != 4:
        raise ValueError(f"Expected 4D BEV feature, got shape {tuple(feat.shape)}")

    corrupt_map: Optional[torch.Tensor] = None
    if cfg.corrupt_type == "zero":
        # Agent ablation: zero collaborator BEV (w/o that agent at fusion).
        n, _, h, w = feat.shape
        corrupt_map = torch.ones(
            (n, 1, h, w), device=feat.device, dtype=feat.dtype
        )
        feat = torch.zeros_like(feat)
    else:
        generator = _make_generator(
            sample_id=sample_id,
            agents=cfg.agents,
            corrupt_type=cfg.corrupt_type,
            level=cfg.level,
            seed_base=cfg.seed_base,
            agent=agent,
        )
        if cfg.corrupt_type == "gaussian":
            feat, corrupt_map = _gaussian_noise(
                feat, GAUSSIAN_SIGMA[cfg.level], generator
            )
        elif cfg.corrupt_type == "mask":
            feat, corrupt_map = _random_mask(
                feat, MASK_RATIO[cfg.level], cfg.patch_size, generator
            )
        elif cfg.corrupt_type == "mask_gaussian":
            # Order: spatial mask first, then Gaussian noise on remaining features.
            feat, mask_map = _random_mask(
                feat, MASK_RATIO[cfg.level], cfg.patch_size, generator
            )
            feat, noise_map = _gaussian_noise(
                feat, GAUSSIAN_SIGMA[cfg.level], generator
            )
            # Overlay map: dropped patches = 1; kept patches = noise intensity.
            corrupt_map = torch.maximum(mask_map, noise_map * (1.0 - mask_map))

    if maps_out is not None and corrupt_map is not None:
        key = map_key if map_key is not None else agent
        maps_out[key] = corrupt_map.detach()
    return feat


def iter_packed_agent_slices(
    data_dict: Dict[str, Any],
    collaborators: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int, int, int]]:
    """List packed slices as ``(agent_type, local_idx, start, end)``."""
    if collaborators is None:
        collaborators = AGENT_ORDER
    present = [
        a
        for a in AGENT_ORDER
        if a in collaborators
        and a in data_dict
        and len(data_dict[a].get("batch_idxs", [])) > 0
    ]
    if not present:
        return []
    batch_size = max(len(data_dict[a]["batch_idxs"]) for a in present)
    slices: List[Tuple[str, int, int, int]] = []
    cursor = 0
    local_counters = {a: 0 for a in present}
    for idx in range(batch_size):
        for agent_type in present:
            batch_idxs = data_dict[agent_type]["batch_idxs"]
            if idx not in batch_idxs:
                continue
            n = int(data_dict[agent_type]["record_len"][idx].item())
            if n <= 0:
                continue
            for k in range(n):
                local_idx = local_counters[agent_type]
                slices.append((agent_type, local_idx, cursor + k, cursor + k + 1))
                local_counters[agent_type] += 1
            cursor += n
    return slices


def corrupt_packed_bev_pre_fusion(
    feat: torch.Tensor,
    data_dict: Dict[str, Any],
    cfg: Optional[CorruptConfig],
    sample_id: str,
    collaborators: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Corrupt packed collaborator BEV features immediately before fusion.

    Also writes per-unit corrupt maps into ``data_dict['_bev_corrupt_maps']``
    with keys like ``drone_0``, ``rsu_1``.
    """
    if cfg is None or not cfg.enabled:
        return feat
    # Agent ablation is handled by dropping agents from packing / available_agents,
    # not by injecting all-zero features into the fusion branch.
    if cfg.corrupt_type == "zero":
        return feat
    if feat is None or not torch.is_tensor(feat) or feat.numel() == 0:
        return feat
    if feat.dim() != 4:
        raise ValueError(f"Expected packed 4D BEV, got shape {tuple(feat.shape)}")

    if collaborators is None:
        collaborators = AGENT_ORDER
    present = [
        a
        for a in AGENT_ORDER
        if a in collaborators
        and a in data_dict
        and len(data_dict[a].get("batch_idxs", [])) > 0
    ]
    if not present:
        return feat

    batch_size = max(len(data_dict[a]["batch_idxs"]) for a in present)
    maps_out: Dict[str, torch.Tensor] = {}
    pieces: List[torch.Tensor] = []
    cursor = 0
    local_counters = {a: 0 for a in present}
    for idx in range(batch_size):
        for agent_type in present:
            batch_idxs = data_dict[agent_type]["batch_idxs"]
            if idx not in batch_idxs:
                continue
            n = int(data_dict[agent_type]["record_len"][idx].item())
            if n <= 0:
                continue
            for k in range(n):
                local_idx = local_counters[agent_type]
                chunk = feat[cursor + k : cursor + k + 1]
                chunk = corrupt_bev(
                    chunk,
                    cfg,
                    sample_id,
                    agent_type,
                    maps_out=maps_out,
                    map_key=f"{agent_type}_{local_idx}",
                )
                pieces.append(chunk)
                local_counters[agent_type] += 1
            cursor += n

    if cursor != feat.shape[0]:
        raise RuntimeError(
            "Packed BEV corruption index mismatch: "
            f"consumed={cursor}, feat_N={feat.shape[0]}. "
            "Agent packing order may have changed."
        )
    if maps_out:
        existing = data_dict.get("_bev_corrupt_maps", {})
        existing.update(maps_out)
        data_dict["_bev_corrupt_maps"] = existing
    if not pieces:
        return feat
    return torch.cat(pieces, dim=0)


def label_packed_transmission_maps(
    masks: torch.Tensor,
    data_dict: Dict[str, Any],
    collaborators: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Split packed ``[N,1,H,W]`` transmission masks into named agent maps."""
    labeled: Dict[str, torch.Tensor] = {}
    for agent_type, local_idx, start, end in iter_packed_agent_slices(
        data_dict, collaborators
    ):
        if end > masks.shape[0]:
            break
        labeled[f"{agent_type}_{local_idx}"] = masks[start:end].detach()
    return labeled
