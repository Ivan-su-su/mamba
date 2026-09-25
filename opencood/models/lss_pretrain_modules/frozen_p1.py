"""Load a frozen Airv2xGaussian0822 and cache F90 + depth for one batch."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from opencood.hypes_yaml import yaml_utils
from opencood.models.airv2x_gaussian_0822 import Airv2xGaussian0822
from opencood.models.gaussian_modules_0822.image_frontend import (
    flatten_camera_world_z,
    present_camera_agents,
)


class FrozenP1(nn.Module):
    """Shared P1 frontend. Parameters stay frozen and in eval."""

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        ckpt = str(args["p1_checkpoint"])
        hypes_path = str(args["p1_hypes"])
        hypes = yaml_utils.load_yaml(hypes_path)
        parser_name = hypes.get("yaml_parser")
        if parser_name:
            hypes = getattr(yaml_utils, parser_name)(hypes)
        self.p1 = Airv2xGaussian0822(hypes["model"]["args"])
        raw = torch.load(ckpt, map_location="cpu")
        state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        missing, unexpected = self.p1.load_state_dict(state, strict=True)
        del missing, unexpected
        for param in self.p1.parameters():
            param.requires_grad = False
        self.p1.eval()
        # Pin the current batch dict (identity, not id()) so veh/rsu/drone
        # share one encode. Builtin dict is not weakref-able; holding the
        # object also prevents CPython from recycling its address mid-step.
        self._cache_obj: Optional[Dict[str, Any]] = None
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def train(self, mode: bool = True) -> "FrozenP1":
        super().train(mode)
        self.p1.eval()
        return self

    @torch.no_grad()
    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._cache_obj is data_dict:
            return self._cache
        self._cache_obj = data_dict
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for agent in present_camera_agents(data_dict):
            cam = data_dict[agent]["batch_merged_cam_inputs"]
            imgs = cam["imgs"]
            r2, f45 = self.p1.frontend.extract_backbone_features(agent, imgs)
            f90 = self.p1.highres[agent](r2, f45)
            item: Dict[str, torch.Tensor] = {"f90": f90}
            if agent == "drone":
                height = flatten_camera_world_z(cam["camera_world_z"], imgs)
                height = height.to(device=f90.device, dtype=f90.dtype)
                embed = self.p1.drone_height_embed(height, (int(f90.shape[2]), int(f90.shape[3])))
                delta = self.p1.drone_delta_head(f90, embed.to(dtype=f90.dtype))
                item["z"] = height[:, None, None] + delta
            else:
                item["depth"] = F.softmax(self.p1.depth_heads[agent](f90), dim=1)
            out[agent] = item
        self._cache = out
        return out
