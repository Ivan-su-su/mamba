# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# Modified by: Yuheng Wu <yuhengwu@kaist.ac.kr>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.where2comm_modules.where2comm_fuse import Where2comm
from opencood.models.task_heads.segmentation_head import BevSegHead 


class Airv2xWhere2com(Airv2xBase):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

        # here we use image encoder LSS instead of lidar
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]

        # if "vehicle" in self.collaborators:
        #     self.veh_model = LiftSplatShootEncoder(args, agent_type="vehicle")

        # if "rsu" in self.collaborators:
        #     self.rsu_model = LiftSplatShootEncoder(args, agent_type="rsu")

        # if "drone" in self.collaborators:
        #     self.drone_model = LiftSplatShootEncoder(args, agent_type="drone")
        
        self.init_encoders(args)

        modality_args = args["modality_fusion"]
        self.encoder_out_channels = self._fused_encoder_channels()
        self.backbone = BaseBEVBackbone(
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

        self.fusion_net = Where2comm(args["where2com_fusion"])
        self.multi_scale = args["where2com_fusion"]["multi_scale"]
        self.proj_first = args.get("proj_first", False)

        self.outC = args["outC"]

        if args["task"] == "det":
            self.cls_head = nn.Conv2d(
                self.outC, args["anchor_number"] * args["num_class"], kernel_size=1
            )
            self.reg_head = nn.Conv2d(
                self.outC, 7 * args["anchor_number"], kernel_size=1
            )
            if args["obj_head"]:
                self.obj_head = nn.Conv2d(
                    self.outC, args["anchor_number"], kernel_size=1
                )

        elif args["task"] == "seg":
            self.seg_head = BevSegHead(
                args["seg_branch"], args["seg_hw"], args["seg_hw"], self.outC, args["dynamic_class"], args["static_class"],
                seg_res=args["seg_res"], cav_range=args["cav_range"]
            )

        if args["backbone_fix"]:
            self.backbone_fix()

    def _fused_encoder_channels(self):
        """64 channels per modality; concat yields 64 * num_modalities."""
        modality_counts = []
        for models in (self.veh_models, self.rsu_models, self.drone_models):
            if models is not None and len(models) > 0:
                modality_counts.append(len(models))
        if not modality_counts:
            return 64
        return 64 * max(modality_counts)

    def fuse_bev(self, batch_dict_list):
        """Concatenate multi-modal BEV features along the channel dimension."""
        if len(batch_dict_list) == 1:
            return {"spatial_features": batch_dict_list[0]["spatial_features"]}
        return {
            "spatial_features": torch.cat(
                [batch_dict["spatial_features"] for batch_dict in batch_dict_list],
                dim=1,
            )
        }

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        if "vehicle" in self.collaborators:
            for p in self.veh_model.parameters():
                p.requires_grad = False
        if "rsu" in self.collaborators:
            for p in self.rsu_model.parameters():
                p.requires_grad = False
        if "drone" in self.collaborators:
            for p in self.drone_model.parameters():
                p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
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

    def _get_pairwise_t_matrix(self, data_dict):
        """
        Select the pairwise transform used for BEV warping during fusion.

        When proj_first=True, LiDAR and (if enabled) camera features are already
        expressed in the ego LiDAR frame, so identity pairwise matrices are used.
        Otherwise, warp collaborators with img_pairwise_t_matrix_collab.
        """
        if self.proj_first:
            return data_dict["pairwise_t_matrix_collab"]
        return data_dict["img_pairwise_t_matrix_collab"]

    def forward(self, data_dict):
        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        batch_dict = self.backbone(batch_output_dict)

        batch_spatial_features = batch_dict["spatial_features"]
        comm_rates = batch_spatial_features.count_nonzero().item()

        batch_spatial_features_2d = batch_dict["spatial_features_2d"]
        pairwise_t_matrix = self._get_pairwise_t_matrix(data_dict)

        # downsample feature to reduce memory
        if self.shrink_flag:
            batch_spatial_features_2d = self.shrink_conv(batch_spatial_features_2d)

        output_dict = {}
        if self.args["task"] == "det":
            # determine where2comm
            psm = self.cls_head(batch_spatial_features_2d)
            # compressor
            if self.compression:
                batch_spatial_features_2d = self.naive_compressor(
                    batch_spatial_features_2d
                )

            if self.multi_scale:
                fused_feature, communication_rates = self.fusion_net(
                    batch_dict["spatial_features"],
                    psm,
                    batch_record_len,
                    pairwise_t_matrix,
                    self.backbone,
                )

                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates = self.fusion_net(
                    batch_spatial_features_2d, psm, batch_record_len, pairwise_t_matrix
                )

            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            output_dict.update({"psm": psm, "rm": rm})

            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                output_dict.update({"obj": obj})

            output_dict.update(
                {"mask": 0, "com": communication_rates, "comm_rate": comm_rates}
            )

        elif self.args["task"] == "seg":
            _, ori_x = self.seg_head(batch_spatial_features_2d, True)
            if self.compression:
                batch_spatial_features_2d = self.naive_compressor(
                    batch_spatial_features_2d
                )

            if self.multi_scale:
                fused_feature, communication_rates = self.fusion_net(
                    batch_dict["spatial_features"],
                    ori_x,
                    batch_record_len,
                    pairwise_t_matrix,
                    self.backbone,
                )

                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates = self.fusion_net(
                    batch_spatial_features_2d,
                    ori_x,
                    batch_record_len,
                    pairwise_t_matrix,
                )
            # seg_logits = self.seg_head(fused_feature)
            # import cv2
            # import numpy as np
            # dynamic = data_dict['label_dict']['dynamic_seg_label'][0]
            # dynamic = dynamic.detach().cpu().numpy()
            # cv2.imwrite("/home/xiangbog/Folder/Research/airv2x/debug/debug_image_dynamic.png", ((dynamic - dynamic.min()) / (dynamic.max() - dynamic.min()) * 255).astype(np.uint8),)
            # # output_dict.update({"seg_logits": seg_logits})
            seg_output_dict = self.seg_head(fused_feature)
            # feat = seg_output_dict['dynamic_seg'][0]
            # feat = feat.mean(0).detach().cpu().numpy()
            # cv2.imwrite("/home/xiangbog/Folder/Research/airv2x/debug/debug_image_segfeat.png", ((feat - feat.min()) / (feat.max() - feat.min()) * 255).astype(np.uint8),)
            
            # feat = batch_spatial_features_2d[0].mean(0).detach().cpu().numpy()
            # import cv2; import numpy as np
            # cv2.imwrite("/home/xiangbog/Folder/Research/airv2x/debug/debug_image_bevfeat.png", ((feat - feat.min()) / (feat.max() - feat.min()) * 255).astype(np.uint8),)
        
            # import pdb; pdb.set_trace()
            output_dict.update(seg_output_dict)
            output_dict.update(
                {"mask": 0, "com": communication_rates, "comm_rate": comm_rates}
            )
        return output_dict
