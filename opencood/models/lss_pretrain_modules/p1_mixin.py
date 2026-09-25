"""Replace cam encoders with P1 splat. Lidar is not used by these yaml."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from opencood.models.lss_pretrain_modules.frozen_p1 import FrozenP1
from opencood.models.lss_pretrain_modules.p1_splat_encoder import P1SplatEncoder


class P1CamMixin:
    """Overrides ``Airv2xBase.init_encoders`` for frozen-P1 camera lift."""

    def init_encoders(self, args: Dict[str, Any]) -> None:
        self.p1_core = FrozenP1(args)
        encode = self.p1_core.encode
        self.veh_models = nn.ModuleList()
        self.rsu_models = nn.ModuleList()
        self.drone_models = nn.ModuleList()
        for agent, bucket in (
            ("vehicle", self.veh_models),
            ("rsu", self.rsu_models),
            ("drone", self.drone_models),
        ):
            if agent not in self.collaborators:
                continue
            for modality in args[agent]["modalities"]:
                if modality != "cam":
                    raise NotImplementedError(
                        f"P1 cam frontend does not wrap modality={modality}"
                    )
                bucket.append(P1SplatEncoder(args[agent]["cam"], agent, encode))
