""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.task_heads.segmentation_head import BevSegHead
import importlib
# from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

class Airv2xHEAL(Airv2xBase):
    def __init__(self, args):
        super(Airv2xHEAL, self).__init__(args)
        
        self.args = args

        # here we use image encoder LSS instead of lidar
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        
        self.init_encoders(args)
        modality_args = args["modality_fusion"]
        # Multi-modal support: backbone input channels = 64 * num_modalities.
        # Single-modal (lidar or cam) -> 64; cam+lidar concat -> 128.
        self.encoder_out_channels = self._fused_encoder_channels()
        self.backbone = ResNetBEVBackbone(
            modality_args["base_bev_backbone"], self.encoder_out_channels
        )
        
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in modality_args and modality_args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        self.compression = False

        if modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])
            
        self.pyramid_backbone = PyramidFusion(args["fusion_backbone"])

        """
        Shared Heads, Would load from pretrain base.
        """
        if args["task"] == "det":
            self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'] * args["num_class"],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                    kernel_size=1)
            if args["obj_head"]:
                self.obj_head = nn.Conv2d(
                    args['in_head'], args["anchor_number"], kernel_size=1
                )
        elif args["task"] == "seg":
            self.seg_head = BevSegHead(
                args["seg_branch"], args["seg_hw"], args["seg_hw"], args['in_head'], args["dynamic_class"], args["static_class"],
                seg_res=args["seg_res"], cav_range=args["cav_range"]
            )
        # self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
        #                           kernel_size=1) # BIN_NUM = 2
        
        if args["backbone_fix"]:
            self.backbone_fix(args["backbone_fix"])
            
    def backbone_fix(self, args):
        """Freeze selected agent encoders for collab finetune.

        When ``args`` is a list like ``[rsu, drone]``, only those agent
        encoders are frozen. Shared backbone / PyramidFusion / heads stay
        trainable so they can adapt to multi-agent features.

        When ``args`` is ``True``, freeze all agent encoders and the shared
        backend (legacy full-freeze behavior).
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
            # List mode: freeze only the listed agent encoders.
            freeze_shared = False
        else:
            raise ValueError("backbone_fix should be bool or list")

        for agent in agents_to_fix:
            if agent == "vehicle":
                print("fix vehicle encoder")
                for p in self.veh_models.parameters():
                    p.requires_grad = False
            elif agent == "rsu":
                print("fix rsu encoder")
                for p in self.rsu_models.parameters():
                    p.requires_grad = False
            elif agent == "drone":
                print("fix drone encoder")
                for p in self.drone_models.parameters():
                    p.requires_grad = False
            else:
                raise ValueError(f"Unknown agent in backbone_fix: {agent}")

        if not freeze_shared:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(
                f"[HEAL] encoder-only freeze; trainable params: "
                f"{trainable}/{total} ({100.0 * trainable / max(total, 1):.1f}%)"
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

    def _fused_encoder_channels(self):
        """返回单 agent 多模态融合后的 BEV 特征通道数。

        每个模态的 encoder 输出 64 通道；多模态时沿 channel 维 concat，
        通道数为 64 × 该 agent 的模态数。各 agent 模态数应一致，
        取已初始化 agent 中最大的模态数作为 backbone 输入通道。
        """
        modality_counts = []
        for models in (self.veh_models, self.rsu_models, self.drone_models):
            if models is not None and len(models) > 0:
                modality_counts.append(len(models))
        if not modality_counts:
            return 64
        return 64 * max(modality_counts)

    def fuse_bev(self, batch_dict_list):
        """多模态 BEV 特征沿 channel 维 concat。

        单模态直接返回；多模态时与 CoBEVT/Where2Comm 一致用 concat，
        避免 mean 融合导致不同模态特征相互稀释。
        """
        if len(batch_dict_list) == 1:
            return {"spatial_features": batch_dict_list[0]["spatial_features"]}
        return {
            "spatial_features": torch.cat(
                [batch_dict["spatial_features"] for batch_dict in batch_dict_list],
                dim=1,
            )
        }

    def forward(self, data_dict):
        output_dict = {'pyramid': 'single'}
        
        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        comm_rates = batch_output_dict["spatial_features"].count_nonzero().item()
        batch_output_dict = self.backbone(batch_output_dict)
        
        batch_spatial_features_2d = batch_output_dict["spatial_features_2d"]
        # proj_first inputs are already expressed in the ego LiDAR frame.
        pairwise_t_matrix = self.get_fusion_pairwise_t_matrix(data_dict)
        
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                        batch_spatial_features_2d,
                                        batch_record_len, 
                                        pairwise_t_matrix[:, :, :, [0, 1], :][
                                            :, :, :, :, [0, 1, 3]], 
                                    )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        if self.args["task"] == "det":
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                output_dict.update({"obj": obj})
            output_dict.update(
                {
                    "psm": psm,
                    "rm": rm,
                    "comm_rate": comm_rates,
                }
            )

        elif self.args["task"] == "seg":
            seg_logits = self.seg_head(fused_feature)
            output_dict.update(
                {
                    "comm_rate": comm_rates,
                }
            )
            output_dict.update(seg_logits)
       
        return output_dict
        
        
        
        
    