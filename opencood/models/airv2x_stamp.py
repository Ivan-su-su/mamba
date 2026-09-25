"""AirV2X STAMP: adapter (+ optional reverter) for heterogeneous collab.

Backward compatible with previous AirV2X STAMP runs:
- Default ``use_reverter: false``: only adapters, same as the old code path
  (reverter modules are not built; old checkpoints load cleanly).
- ``use_reverter: true``: min-patch mode with reverter + adapter_align feats
  for AdapterLoss (see ``*_minpatch.yaml``).
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.fuse_modules.adapter import Adapter, Reverter
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.task_heads.segmentation_head import BevSegHead


class Airv2xSTAMP(Airv2xBase):
    def __init__(self, args: Dict[str, Any]) -> None:
        """Initialize STAMP. Reverter is optional for legacy checkpoint resume."""
        super(Airv2xSTAMP, self).__init__(args)

        self.args = args
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        # Legacy default: False so old configs / ckpts keep working.
        self.use_reverter = bool(args.get("use_reverter", False))

        self.init_encoders(args)
        modality_args = args["modality_fusion"]
        self.encoder_out_channels = self._fused_encoder_channels()
        self.backbone = ResNetBEVBackbone(
            modality_args["base_bev_backbone"], self.encoder_out_channels
        )

        self.shrink_flag = False
        if "shrink_header" in modality_args and modality_args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        self.compression = False
        if modality_args["compression"] > 0:
            self.compression = True
            from opencood.models.common_modules.naive_compress import (
                NaiveCompressor,
            )

            self.naive_compressor = NaiveCompressor(256, args["compression"])

        self.build_adapter_and_reverter(args)
        self.pyramid_backbone = PyramidFusion(args["fusion_backbone"])

        if args["task"] == "det":
            self.cls_head = nn.Conv2d(
                args["in_head"],
                args["anchor_number"] * args["num_class"],
                kernel_size=1,
            )
            self.reg_head = nn.Conv2d(
                args["in_head"], 7 * args["anchor_number"], kernel_size=1
            )
            if args["obj_head"]:
                self.obj_head = nn.Conv2d(
                    args["in_head"], args["anchor_number"], kernel_size=1
                )
        elif args["task"] == "seg":
            self.seg_head = BevSegHead(
                args["seg_hw"],
                args["seg_hw"],
                args["in_head"],
                args["dynamic_class"],
                args["static_class"],
                seg_res=args["seg_res"],
                cav_range=args["cav_range"],
            )

        if args.get("backbone_fix"):
            self.backbone_fix(args["backbone_fix"])

    def build_adapter_and_reverter(self, args: Dict[str, Any]) -> None:
        """Build adapters always; build reverters only when ``use_reverter``."""
        if "vehicle" in self.collaborators:
            self.adapter_vehicle = Adapter(args["vehicle"]["adapter"])
            if self.use_reverter:
                self.reverter_vehicle = Reverter(args["vehicle"]["reverter"])
        if "rsu" in self.collaborators:
            self.adapter_rsu = Adapter(args["rsu"]["adapter"])
            if self.use_reverter:
                self.reverter_rsu = Reverter(args["rsu"]["reverter"])
        if "drone" in self.collaborators:
            self.adapter_drone = Adapter(args["drone"]["adapter"])
            if self.use_reverter:
                self.reverter_drone = Reverter(args["drone"]["reverter"])

        mode = "reverter+align" if self.use_reverter else "legacy-adapter-only"
        print(f"[STAMP] mode={mode}")

    def _fused_encoder_channels(self) -> int:
        """Return fused BEV channels after multi-modal concat (64 × num modalities)."""
        modality_counts: List[int] = []
        for models in (self.veh_models, self.rsu_models, self.drone_models):
            if models is not None and len(models) > 0:
                modality_counts.append(len(models))
        if not modality_counts:
            return 64
        return 64 * max(modality_counts)

    def fuse_bev(
        self, batch_dict_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Concat multi-modal BEV features along channel dim."""
        if len(batch_dict_list) == 1:
            return {"spatial_features": batch_dict_list[0]["spatial_features"]}
        return {
            "spatial_features": torch.cat(
                [batch_dict["spatial_features"] for batch_dict in batch_dict_list],
                dim=1,
            )
        }

    def backbone_fix(self, args: Any) -> None:
        """Freeze parameters for STAMP collab finetune.

        Args:
            args: ``True`` freezes encoders + shared backbone/pyramid/heads
                (legacy adapter-only training). A list like ``[rsu, drone]``
                freezes only those encoders. Extra flags:

                - ``freeze_adapters``: freeze adapter/reverter modules.
                - ``freeze_bev_backbone``: freeze shared ResNet BEV backbone
                  (recommended when adapters start near-identity from HEAL,
                  so adapter input distribution stays stable while pyramid /
                  heads / vehicle encoder / adapters adapt).
        """
        if isinstance(args, bool):
            if not args:
                return
            agents_to_fix = [
                agent
                for agent in ("vehicle", "rsu", "drone")
                if agent in self.collaborators
            ]
            freeze_shared = True
        elif isinstance(args, list):
            agents_to_fix = args
            freeze_shared = False
        else:
            raise ValueError("backbone_fix should be bool or list")

        for agent in agents_to_fix:
            if agent == "vehicle":
                print("[STAMP] freeze vehicle encoder")
                for p in self.veh_models.parameters():
                    p.requires_grad = False
            elif agent == "rsu":
                print("[STAMP] freeze rsu encoder")
                for p in self.rsu_models.parameters():
                    p.requires_grad = False
            elif agent == "drone":
                print("[STAMP] freeze drone encoder")
                for p in self.drone_models.parameters():
                    p.requires_grad = False
            else:
                raise ValueError(f"Unknown agent in backbone_fix: {agent}")

        if bool(self.args.get("freeze_adapters", False)):
            print("[STAMP] freeze adapters (keep pretrained alignment)")
            for name in ("adapter_vehicle", "adapter_rsu", "adapter_drone"):
                if hasattr(self, name):
                    for p in getattr(self, name).parameters():
                        p.requires_grad = False
            if self.use_reverter:
                for name in (
                    "reverter_vehicle",
                    "reverter_rsu",
                    "reverter_drone",
                ):
                    if hasattr(self, name):
                        for p in getattr(self, name).parameters():
                            p.requires_grad = False

        # List-mode optional: freeze BEV backbone only (keep pyramid/heads).
        if (not freeze_shared) and bool(
            self.args.get("freeze_bev_backbone", False)
        ):
            print("[STAMP] freeze shared BEV backbone")
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.compression:
                for p in self.naive_compressor.parameters():
                    p.requires_grad = False

        if not freeze_shared:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(
                f"[STAMP] encoder-only freeze; trainable: "
                f"{trainable}/{total} ({100.0 * trainable / max(total, 1):.2f}%)"
            )
            return

        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.pyramid_backbone.parameters():
            p.requires_grad = False

        if self.args["task"] == "det":
            for p in self.cls_head.parameters():
                p.requires_grad = False
            for p in self.reg_head.parameters():
                p.requires_grad = False
            if self.args["obj_head"]:
                for p in self.obj_head.parameters():
                    p.requires_grad = False
        elif self.args["task"] == "seg":
            for p in self.seg_head.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[STAMP] freeze backbone/heads; trainable: "
            f"{trainable}/{total} ({100.0 * trainable / max(total, 1):.2f}%)"
        )

    def _encode_agent_legacy(
        self,
        models: nn.ModuleList,
        data_dict: Dict[str, Any],
        adapter: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Legacy path: backbone + adapter only (same as previous STAMP)."""
        output_list = [m(data_dict) for m in models]
        fused = self.fuse_bev(output_list)
        # Backbone mutates ``fused`` in-place and returns it.
        backbone_out = self.backbone(fused)
        feat = backbone_out["spatial_features_2d"]
        backbone_out["spatial_features_2d"] = adapter(feat)
        return backbone_out

    def _encode_agent_with_reverter(
        self,
        models: nn.ModuleList,
        data_dict: Dict[str, Any],
        adapter: nn.Module,
        reverter: nn.Module,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Min-patch path: adapter to protocol + reverter cycle feats for loss."""
        output_list = [m(data_dict) for m in models]
        fused = self.fuse_bev(output_list)
        backbone_out = self.backbone(fused)
        local_feat = backbone_out["spatial_features_2d"]
        protocol_feat = adapter(local_feat)
        cycled_feat = reverter(protocol_feat)
        protocol_to_local = reverter(local_feat.detach())

        pack_dict = {
            "spatial_features": backbone_out["spatial_features"],
            "spatial_features_2d": protocol_feat,
        }
        align_dict = {
            "FM": local_feat,
            "FM2P": protocol_feat,
            "FM2P2M": cycled_feat,
            "FP2M": protocol_to_local,
        }
        return pack_dict, align_dict

    def extract_features(
        self, data_dict: Dict[str, Any]
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], torch.Tensor],
        Tuple[
            Dict[str, torch.Tensor],
            torch.Tensor,
            Dict[str, Dict[str, torch.Tensor]],
        ],
    ]:
        """Extract per-agent features; return align dict only in reverter mode."""
        batch_dicts: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()
        align_feats: Dict[str, Dict[str, torch.Tensor]] = {}

        if (
            "vehicle" in self.collaborators
            and len(data_dict["vehicle"]["batch_idxs"]) > 0
        ):
            assert self.veh_models is not None, "Vehicle model is not initialized."
            if self.use_reverter:
                pack_dict, align_dict = self._encode_agent_with_reverter(
                    self.veh_models,
                    data_dict,
                    self.adapter_vehicle,
                    self.reverter_vehicle,
                )
                batch_dicts["vehicle"] = pack_dict
                align_feats["vehicle"] = align_dict
            else:
                batch_dicts["vehicle"] = self._encode_agent_legacy(
                    self.veh_models, data_dict, self.adapter_vehicle
                )

        if "rsu" in self.collaborators and len(data_dict["rsu"]["batch_idxs"]) > 0:
            assert self.rsu_models is not None, "RSU model is not initialized."
            if self.use_reverter:
                pack_dict, align_dict = self._encode_agent_with_reverter(
                    self.rsu_models,
                    data_dict,
                    self.adapter_rsu,
                    self.reverter_rsu,
                )
                batch_dicts["rsu"] = pack_dict
                align_feats["rsu"] = align_dict
            else:
                batch_dicts["rsu"] = self._encode_agent_legacy(
                    self.rsu_models, data_dict, self.adapter_rsu
                )

        if (
            "drone" in self.collaborators
            and len(data_dict["drone"]["batch_idxs"]) > 0
        ):
            assert self.drone_models is not None, "Drone model is not initialized."
            if self.use_reverter:
                pack_dict, align_dict = self._encode_agent_with_reverter(
                    self.drone_models,
                    data_dict,
                    self.adapter_drone,
                    self.reverter_drone,
                )
                batch_dicts["drone"] = pack_dict
                align_feats["drone"] = align_dict
            else:
                batch_dicts["drone"] = self._encode_agent_legacy(
                    self.drone_models, data_dict, self.adapter_drone
                )

        B = max(
            len(data_dict["vehicle"]["batch_idxs"]),
            len(data_dict["rsu"]["batch_idxs"]),
            len(data_dict["drone"]["batch_idxs"]),
        )
        batch_output_dict, batch_record_len = self.repack_batch(
            batch_dicts, data_dict, B
        )
        assert (
            batch_output_dict["spatial_features"].shape[0]
            == batch_record_len.sum().item()
        ), f"{batch_output_dict['spatial_features'].shape}, {batch_record_len}"

        if self.use_reverter:
            return batch_output_dict, batch_record_len, align_feats
        return batch_output_dict, batch_record_len

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Forward collab detection/seg; attach adapter_align only in min-patch."""
        output_dict: Dict[str, Any] = {"pyramid": "single"}

        extracted = self.extract_features(data_dict)
        if self.use_reverter:
            batch_output_dict, batch_record_len, align_feats = extracted
        else:
            batch_output_dict, batch_record_len = extracted
            align_feats = None

        comm_rates = batch_output_dict["spatial_features"].count_nonzero().item()
        batch_spatial_features_2d = batch_output_dict["spatial_features_2d"]
        pairwise_t_matrix = self.get_fusion_pairwise_t_matrix(data_dict)

        fused_feature, _occ_outputs = self.pyramid_backbone.forward_collab(
            batch_spatial_features_2d,
            batch_record_len,
            pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]],
        )
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        if self.args["task"] == "det":
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)
            if self.args["obj_head"]:
                output_dict.update({"obj": self.obj_head(fused_feature)})
            output_dict.update(
                {"psm": psm, "rm": rm, "comm_rate": comm_rates}
            )
        elif self.args["task"] == "seg":
            seg_logits = self.seg_head(fused_feature)
            output_dict.update({"comm_rate": comm_rates})
            output_dict.update(seg_logits)

        if align_feats:
            output_dict["adapter_align"] = align_feats
        return output_dict
