import random

import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
import math

from ...vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d
from mamba_ssm.models.mixer_seq_simple import create_block
from collections import OrderedDict
from ..base_bev_backbone import BasicBlock
from ...model_utils.voxel_mamba_utils import get_hilbert_index_2d_mamba_lite
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from ..sparse_temporal import SparseTemporalFusionBlock, TemporalFusionConcatBlock
class ConvFuser(nn.Module):
    """
    【AirV2X多agent ConvFuser模块】
    
    功能：融合多agent的图像BEV特征和激光雷达BEV特征
    
    架构对比：
    - MambaFusion: 单agent多视角 -> 统一BEV -> 双模态融合
    - AirV2X: 多agent独立 -> 多BEV -> 多agent融合 -> 双模态融合
    
    输入：
    - 多agent情况: batch_dict[agent]['spatial_features_img'] 每个 [B_i, 80, H, W]
    - 单agent情况: batch_dict['spatial_features_img'] [B, 80, H, W]
    - 激光雷达: batch_dict['spatial_features'] [B, 128, H, W]
    
    输出：
    - batch_dict['spatial_features'] [B, 128, H, W] 与MambaFusion对齐
    
    融合策略：
    - mean: 平均融合，保留所有agent信息
    - max: 最大融合，突出最强特征
    - concat: 通道拼接，保留所有原始信息
    """
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.image_channel = self.model_cfg.IMAGE_CHANNEL
        self.lidar_channel = self.model_cfg.LIDAR_CHANNEL
        self.merge_type = self.model_cfg.get('MERGE_TYPE', 'default')
        self.importance_generator = ImportanceGenerator(num_channels=out_channel, max_agents=3, use_softmax=True)
        # 使用空间相关的 BatchCompressorV2，对每个像素位置在 agent 维上做可学习加权融合
        self.batch_compressor = BatchCompressorV2(in_channels=self.image_channel,
                                                  mid_channels=self.image_channel*2,
                                                  out_channels=self.image_channel)
        # 支持只用雷达数据的情况
        self.lidar_only = self.model_cfg.get('LIDAR_ONLY', False)

        # 根据是否只用雷达数据调整输入通道数
        if self.lidar_only:
            # 只用雷达数据时，输入通道数只有雷达特征维度
            lidar_channel = 64  # 雷达特征维度
            actual_in_channel = lidar_channel
        else:
            # 使用图像+雷达数据时，输入通道数是两者之和
            actual_in_channel = in_channel
            
        if self.merge_type == 'default':
            self.conv = nn.Sequential(
                nn.Conv2d(actual_in_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
        else:
            self.conv = nn.Sequential(
                # DepthwiseSeparableConv(actual_in_channel, actual_in_channel, 3, 1, 1),
                nn.Conv2d(actual_in_channel, out_channel * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel * 2),
                nn.ReLU(),
                nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                )
        self.use_vmamba = model_cfg.get('USE_VMAMBA', False)
        self.use_checkpoint = model_cfg.get('USE_CHECKPOINT', True)
        self.use_merge_after = model_cfg.get('USE_MERGE_AFTER', False)
        self.agent_fusion_strategy = model_cfg.get('AGENT_FUSION_STRATEGY', 'mean')  # 'mean', 'concat', 'max'
        self.use_offset_guided_hierarchical_fusion = model_cfg.get('USE_OFFSET_GUIDED_HIERARCHICAL_FUSION', False)
        if self.use_offset_guided_hierarchical_fusion:
            fusion_cfg = model_cfg.get('OFFSET_GUIDED_HIERARCHICAL_FUSION', {})
            sparse_temporal_cfg = model_cfg.get('SPARSE_TEMPORAL', {})
            self.offset_guided_hierarchical_fusion = OffsetGuidedSelectiveHierarchicalMambaFusionBlock(
                channels=out_channel,
                num_points=fusion_cfg.get('NUM_POINTS', 4),
                offset_range=fusion_cfg.get('OFFSET_RANGE', 2.0),
                transmission_alpha=fusion_cfg.get('TRANSMISSION_ALPHA', 1.0),
                window_size=fusion_cfg.get('WINDOW_SIZE', 4),
                align_corners=fusion_cfg.get('ALIGN_CORNERS', False),
                padding_mode=fusion_cfg.get('PADDING_MODE', 'zeros'),
                local_mamba_depth=fusion_cfg.get('LOCAL_MAMBA_DEPTH', 1),
                global_fusion_type=fusion_cfg.get('GLOBAL_FUSION_TYPE', 'mamba'),
                global_mamba_depth=fusion_cfg.get('GLOBAL_MAMBA_DEPTH', 1),
                global_conv_depth=fusion_cfg.get('GLOBAL_CONV_DEPTH', 2),
                ssm_d_state=fusion_cfg.get('SSM_D_STATE', 1),
                ssm_ratio=fusion_cfg.get('SSM_RATIO', 1.0),
                ssm_dt_rank=fusion_cfg.get('SSM_DT_RANK', 'auto'),
                ssm_conv=fusion_cfg.get('SSM_CONV', 3),
                ssm_conv_bias=fusion_cfg.get('SSM_CONV_BIAS', False),
                mlp_ratio=fusion_cfg.get('MLP_RATIO', 4.0),
                mlp_drop_rate=fusion_cfg.get('MLP_DROP_RATE', 0.0),
                forward_type=fusion_cfg.get('FORWARD_TYPE', 'v05_noz'),
                sample_da_ffn_ratio=fusion_cfg.get('SAMPLE_DA_FFN_RATIO', 2.0),
                sample_da_drop_rate=fusion_cfg.get('SAMPLE_DA_DROP_RATE', 0.1),
                sample_da_layer_scale_init=fusion_cfg.get(
                    'SAMPLE_DA_LAYER_SCALE_INIT', 1e-2
                ),
                final_gate_hidden_dim=fusion_cfg.get('FINAL_GATE_HIDDEN_DIM', 32),
                gate_head_cfg=fusion_cfg.get('GATE_HEAD', {}),
                sparse_temporal_cfg=sparse_temporal_cfg,
                gate_topk_cfg=fusion_cfg.get('GATE_TOPK', {}),
                visualize_gate_maps=fusion_cfg.get('VISUALIZE_GATE_MAPS', False),
            )
        if self.use_merge_after:
            depths = [1]
            num_block = len(depths)
            merge_dim = 144  # 80 + 60 = 140
            self.merge_blocks = nn.ModuleList()
            # self.merge_norm = nn.ModuleList()
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
            for i_layer in range(num_block):
                self.merge_blocks.append(self._make_vmamba_layer(
                    dim=merge_dim,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=1,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
        if self.use_vmamba:
            # self.img_pos_embed_layer = PositionEmbeddingLearned(20, 128)
            # self.lidar_pos_embed_layer = PositionEmbeddingLearned(3, 128)
            self.use_dw_conv = True
            depths = [1, 1, 1] # [1, 2, 2]
            num_block = len(depths)
            image_dim = 80
            point_dim = 64 # 从128改为60
            cross_dim = 128
            ssm_conv = 3
            max_channel = 1
            use_4x = False
            self.use_cross = False
            self.use_res_merge = False
            d_state = 1
            self.image_down_blocks = nn.ModuleList()
            self.image_de_blocks = nn.ModuleList()
            self.lidar_de_blocks = nn.ModuleList()
            self.lidar_down_blocks = nn.ModuleList()

            
            if self.use_res_merge:
                self.image_norm = nn.ModuleList()
                self.point_norm = nn.ModuleList()


            self.image_vmamba_blocks = nn.ModuleList()
            self.point_vmamba_blocks = nn.ModuleList()
            num_block_cross = 0

            if self.use_cross:
                depths_cross = [1, 1, 1]
                self.use_res_merge = False
                if not self.use_res_merge:
                    self.image_cross_blocks = nn.ModuleList()
                    self.point_cross_blocks = nn.ModuleList()
                
                self.image_up_blocks = nn.ModuleList()
                
                num_block_cross = len(depths_cross)
                dpr_cross = []
                for x in torch.linspace(0, 0.1, sum(depths_cross)):
                    dpr_cross.extend([x.item(), x.item()])
                self.cross_vmamba_blocks = nn.ModuleList()
                for i_layer in range(num_block_cross):
                    self.image_up_blocks.append(
                        nn.Sequential(
                            nn.Conv2d(image_dim, cross_dim, kernel_size=1),
                            nn.BatchNorm2d(cross_dim),
                            nn.ReLU(),
                            DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                        )
                    )
                    if not self.use_res_merge:
                        self.image_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, image_dim,  3, padding=1, bias=False),
                                nn.BatchNorm2d(image_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(image_dim, image_dim, 3, 1, 1),
                            )
                        )
                        self.point_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, cross_dim, 3, padding=1, bias=False),
                                nn.BatchNorm2d(cross_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                            )
                        )
                    self.cross_vmamba_blocks.append(self._make_vmamba_layer(
                        dim=cross_dim,
                        cross_dim=cross_dim,
                        drop_path = dpr_cross[sum(depths_cross[:i_layer]):sum(depths_cross[:i_layer + 1])],
                        use_checkpoint=False,
                        norm_layer=LayerNorm2d,
                        downsample=nn.Identity(),
                        channel_first=True,
                        # =================
                        ssm_d_state=d_state,
                        ssm_ratio=1.0,
                        ssm_dt_rank='auto',
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type='cross_noz',
                        # =================
                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,
                        gmlp=False,
                        cross=True,
                    ))
            if not self.use_res_merge:
                self.image_conv = nn.Sequential(
                        nn.Conv2d(image_dim * (num_block + 1), image_dim * 2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(image_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(image_dim * 2, image_dim, 3, 1, 1),
                    )
                self.lidar_conv = nn.Sequential(
                        nn.Conv2d(point_dim * (num_block + 1), point_dim *2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(point_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(point_dim * 2, point_dim, 3, 1, 1),
                    )

            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

            for i_layer in range(num_block):
                if self.use_res_merge:
                    self.image_norm.append(nn.BatchNorm2d(image_dim))
                    self.point_norm.append(nn.BatchNorm2d(point_dim))

                # if i_layer == 0 and use_4x:
                #     point_cur_layers.append(BasicBlock(point_dim, point_dim, 2, 1, True))
                if self.use_dw_conv:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(image_dim*min(i_layer + 2, max_channel), image_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(point_dim*min(i_layer + 2, max_channel), point_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                else:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                    # if i_layer == 0 and use_4x:
                    #     image_cur_layers.append(BasicBlock(image_dim, image_dim, 2, 1, True))
                    
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                self.image_down_blocks.append(nn.Sequential(*image_cur_layers))
                self.lidar_down_blocks.append(nn.Sequential(*point_cur_layers))
                
                    
                image_cur_de_layers = []
                point_cur_de_layers = []

                for j in range(i_layer + 1):
                    # if self.use_cross:
                    #     image_cur_de_layers.append(nn.ConvTranspose2d(point_dim, image_dim, kernel_size=2, stride=2, bias=False))
                    # else:
                    image_cur_de_layers.append(nn.ConvTranspose2d(image_dim*min(i_layer + 2 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    image_cur_de_layers.append(nn.BatchNorm2d(image_dim*min(i_layer + 1 - j, max_channel)))
                    image_cur_de_layers.append(nn.ReLU())
                    point_cur_de_layers.append(nn.ConvTranspose2d(point_dim*min(i_layer + 2 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    point_cur_de_layers.append(nn.BatchNorm2d(point_dim*min(i_layer + 1 - j, max_channel)))
                    point_cur_de_layers.append(nn.ReLU())
                    if self.use_dw_conv:
                        image_cur_de_layers.append(DepthwiseSeparableConv(image_dim*min(i_layer + 1 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                        point_cur_de_layers.append(DepthwiseSeparableConv(point_dim*min(i_layer + 1 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                self.image_de_blocks.append(nn.Sequential(*image_cur_de_layers))
                self.lidar_de_blocks.append(nn.Sequential(*point_cur_de_layers))
                self.image_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = image_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))

                self.point_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = point_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
        
    @staticmethod
    def _make_vmamba_layer(
        dim=96,
        cross_dim=0,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        cross=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        
        if cross_dim != 0:
            blocks1 = []
            blocks2 = []
        else:
            blocks = []
        for d in range(depth):
            if cross_dim != 0:
                blocks1.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=cross_dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
                blocks2.append(VSSBlock(
                    hidden_dim=cross_dim, 
                    cross_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
            else:
                blocks.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=0,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
        if not cross:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))
        else:
            return nn.Sequential(OrderedDict(
                blocks1=nn.Sequential(*blocks1),
                blocks2=nn.Sequential(*blocks2),
            ))
    def forward(self,batch_dict,available_agents = None,lidar_only = False):

        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        # 【MambaFusion融合策略分析】
        # 1. 直接从batch_dict获取两种模态的BEV特征
        # 2. 这里假设spatial_features_img已经是整合后的图像BEV特征
        # 3. 没有多agent处理，直接进行双模态融合
       
        agent_spatial_features = {}
        img_bev_dict = {}
        lidar_bev_dict = {}
        cat_bev_dict = {}
        
        for agent in available_agents:
            #img_bev = self.batch_compressor(batch_dict[agent]['spatial_features_img'])  # [B, 80, H, W] - 图像BEV特征  should ignore
            img_bev = batch_dict[agent]['spatial_features_img']
            lidar_bev = batch_dict[agent]['spatial_features']   # [B, 128, H, W] - 激光雷达BEV特征
            
            # 存储用于可视化
            img_bev_dict[agent] = img_bev
            lidar_bev_dict[agent] = lidar_bev
            
            if self.use_vmamba:
                # 【VMamba融合】使用复杂的多尺度VMamba块进行融合
                if self.use_checkpoint:
                    cat_bev = checkpoint.checkpoint(self.mamba_forward, img_bev, lidar_bev)
                else:
                    cat_bev = self.mamba_forward(img_bev, lidar_bev)
            elif lidar_only:
                cat_bev = lidar_bev
            else:
                # 【简单拼接融合】直接在通道维度拼接两种模态
                cat_bev = torch.cat([img_bev, lidar_bev], dim=1)  # [B, 144, H, W]
            
            # 【后处理融合】可选的额外融合层
            if self.use_merge_after:
                for block in self.merge_blocks:
                    cat_bev = block(cat_bev)
            
            # 【最终卷积】将融合后的特征映射到目标通道数
            mm_bev = self.conv(cat_bev) # [B, 128, H, W]
            agent_spatial_features[agent] = mm_bev
            
            # 存储融合后的特征用于可视化
            cat_bev_dict[agent] = cat_bev
            
        # 多agent融合
        if self.use_offset_guided_hierarchical_fusion and 'vehicle' in agent_spatial_features:
            mm_bev = self.offset_guided_hierarchical_fusion(
                agent_spatial_features['vehicle'],
                agent_spatial_features.get('rsu'),
                agent_spatial_features.get('drone'),
                batch_dict=batch_dict,
            )
        else:
            # 首版安全策略：至少需要 vehicle 分支；若 vehicle 缺失则回退到原有融合逻辑。
            mm_bev = self.importance_generator(agent_spatial_features, available_agents)[0]
        
        # 可视化ConvFuser的特征处理过程
        # self.visualize_agent_features(
        #     img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, 
        #     available_agents, save_dir="./convfuser_visualization"
        # )
       
        batch_dict['spatial_features'] = mm_bev
        return batch_dict

    def mamba_forward(self, img_bev, lidar_bev):
        ups_img = []
        ups_img.append(img_bev)
        ups_lidar = []
        ups_lidar.append(lidar_bev)
        for i, (block_img, block_lidar) in enumerate(zip(self.image_vmamba_blocks, self.point_vmamba_blocks)):
            img_bev = self.image_down_blocks[i](img_bev) # [2, 80, 90, 90]
            img_bev = block_img(img_bev)
            lidar_bev = self.lidar_down_blocks[i](lidar_bev)
            lidar_bev = block_lidar(lidar_bev)
            if self.use_cross:
                img_bev = self.image_up_blocks[i](img_bev) # [batch_size, 128, 180, 180]
                img_bev_cross = self.cross_vmamba_blocks[i].blocks1((img_bev, lidar_bev)) # [batch_size, 128, 180, 180]
                lidar_bev_cross = self.cross_vmamba_blocks[i].blocks2((lidar_bev, img_bev))
                if not self.use_res_merge:
                    img_bev = self.image_cross_blocks[i](torch.cat([img_bev, lidar_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                    lidar_bev = self.point_cross_blocks[i](torch.cat([lidar_bev, img_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                else:
                    img_bev = img_bev_cross
                    lidar_bev = lidar_bev_cross
            if self.use_res_merge:
                img_bev = self.image_norm[i](img_bev + self.image_de_blocks[i](img_bev))
                lidar_bev = self.point_norm[i](lidar_bev + self.lidar_de_blocks[i](lidar_bev))
            else:
                ups_img.append(self.image_de_blocks[i](img_bev))
                ups_lidar.append(self.lidar_de_blocks[i](lidar_bev))
        if self.use_res_merge:
            merge_img = img_bev
            merge_lidar = lidar_bev
        else:
            merge_img = self.image_conv(torch.cat(ups_img, dim=1)) # [1, 80, 360, 360]
            merge_lidar = self.lidar_conv(torch.cat(ups_lidar, dim=1)) # [1, 64, 360, 360]
        cat_bev = torch.cat([merge_img,merge_lidar],dim=1)

        return cat_bev

    def visualize_agent_features(self, img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, agent_names, save_dir="./convfuser_visualization"):
        """
        可视化ConvFuser中不同agent的特征
        
        Args:
            img_bev_dict: 字典，包含每个agent的图像BEV特征
            lidar_bev_dict: 字典，包含每个agent的激光雷达BEV特征  
            cat_bev_dict: 字典，包含每个agent的融合后特征
            mm_bev: 最终的多模态融合特征
            agent_names: agent名称列表
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[ConvFuser Visualization] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_img = list(img_bev_dict.values())[0]
        first_lidar = list(lidar_bev_dict.values())[0]
        first_cat = list(cat_bev_dict.values())[0]
        
        batch_size = first_img.shape[0]
        img_channels = first_img.shape[1]
        lidar_channels = first_lidar.shape[1]
        cat_channels = first_cat.shape[1]
        mm_channels = mm_bev.shape[1]
        height, width = first_img.shape[2], first_img.shape[3]
        
        print(f"[ConvFuser Visualization] Feature shapes:")
        print(f"  Image BEV: {first_img.shape}")
        print(f"  Lidar BEV: {first_lidar.shape}")
        print(f"  Cat BEV: {first_cat.shape}")
        print(f"  MM BEV: {mm_bev.shape}")
        
        # 为每个batch创建可视化
        for batch_idx in range(batch_size):
            # 1. 可视化融合前的图像BEV特征
            self._visualize_pre_fusion_features(
                img_bev_dict, lidar_bev_dict, agent_names, batch_idx, 
                save_dir, "pre_fusion"
            )
            
            # 2. 可视化融合后的特征
            self._visualize_post_fusion_features(
                cat_bev_dict, mm_bev, agent_names, batch_idx,
                save_dir, "post_fusion"
            )
            
            # 3. 创建综合对比图
            self._visualize_comprehensive_comparison(
                img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, 
                agent_names, batch_idx, save_dir
            )

    def _visualize_pre_fusion_features(self, img_bev_dict, lidar_bev_dict, agent_names, batch_idx, save_dir, prefix):
        """可视化融合前的特征"""
        fig, axes = plt.subplots(2, len(agent_names), figsize=(4*len(agent_names), 8))
        if len(agent_names) == 1:
            axes = axes.reshape(2, 1)
        
        # 收集所有占用图用于统一颜色范围
        all_img_maps = []
        all_lidar_maps = []
        
        for agent_name in agent_names:
            img_features = img_bev_dict[agent_name][batch_idx]  # [C, H, W]
            lidar_features = lidar_bev_dict[agent_name][batch_idx]  # [C, H, W]
            
            # 使用L2范数计算占用强度
            img_map = torch.norm(img_features, dim=0).detach().cpu().numpy()
            lidar_map = torch.norm(lidar_features, dim=0).detach().cpu().numpy()
            
            all_img_maps.append(img_map)
            all_lidar_maps.append(lidar_map)
        
        # 计算统一的颜色范围
        all_img_values = np.concatenate([m.flatten() for m in all_img_maps])
        all_lidar_values = np.concatenate([m.flatten() for m in all_lidar_maps])
        img_vmin, img_vmax = np.min(all_img_values), np.max(all_img_values)
        lidar_vmin, lidar_vmax = np.min(all_lidar_values), np.max(all_lidar_values)
        
        for i, agent_name in enumerate(agent_names):
            img_map = all_img_maps[i]
            lidar_map = all_lidar_maps[i]
            
            # 上排：图像BEV特征
            im1 = axes[0, i].imshow(img_map, cmap='hot', aspect='equal', vmin=img_vmin, vmax=img_vmax)
            axes[0, i].set_title(f'{agent_name}\nImage BEV')
            axes[0, i].set_ylabel('Y')
            
            # 下排：激光雷达BEV特征
            im2 = axes[1, i].imshow(lidar_map, cmap='hot', aspect='equal', vmin=lidar_vmin, vmax=lidar_vmax)
            axes[1, i].set_title(f'{agent_name}\nLidar BEV')
            axes[1, i].set_ylabel('Y')
        
        # 移除颜色条，保持简洁
        
        fig.suptitle(f'Pre-Fusion Features - Batch {batch_idx}', fontsize=14)
        plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
        
        save_path = os.path.join(save_dir, f'{prefix}_batch_{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[ConvFuser Visualization] Saved pre-fusion: {save_path}")

    def _visualize_post_fusion_features(self, cat_bev_dict, mm_bev, agent_names, batch_idx, save_dir, prefix):
        """可视化融合后的特征"""
        fig, axes = plt.subplots(2, len(agent_names) + 1, figsize=(4*(len(agent_names) + 1), 8))
        if len(agent_names) == 1:
            axes = axes.reshape(2, 2)
        
        # 收集所有占用图用于统一颜色范围
        all_cat_maps = []
        
        for agent_name in agent_names:
            cat_features = cat_bev_dict[agent_name][batch_idx]  # [C, H, W]
            cat_map = torch.norm(cat_features, dim=0).detach().cpu().numpy()
            all_cat_maps.append(cat_map)
        
        # 添加最终融合特征
        mm_map = torch.norm(mm_bev[batch_idx], dim=0).detach().cpu().numpy()
        all_cat_maps.append(mm_map)
        
        # 计算统一的颜色范围
        all_values = np.concatenate([m.flatten() for m in all_cat_maps])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        for i, agent_name in enumerate(agent_names):
            cat_map = all_cat_maps[i]
            
            # 上排：融合后特征
            im1 = axes[0, i].imshow(cat_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'{agent_name}\nFused BEV')
            axes[0, i].set_ylabel('Y')
            
            # 下排：统计信息
            occupied_cells = np.count_nonzero(cat_map)
            total_cells = cat_map.size
            occupancy_ratio = occupied_cells / total_cells
            
            stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
            Ratio: {occupancy_ratio:.2%}"""
            
            axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes, 
                          fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[1, i].set_xlim(0, 1)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].axis('off')
            axes[1, i].set_title('Statistics')
        
        # 最后一列：最终多模态融合结果
        im_final = axes[0, -1].imshow(mm_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
        axes[0, -1].set_title('Final\nMulti-Modal BEV')
        axes[0, -1].set_ylabel('Y')
        
        # 最终结果统计
        occupied_cells = np.count_nonzero(mm_map)
        total_cells = mm_map.size
        occupancy_ratio = occupied_cells / total_cells
        
        stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
        Ratio: {occupancy_ratio:.2%}"""
        
        axes[1, -1].text(0.1, 0.9, stats_text, transform=axes[1, -1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, -1].set_xlim(0, 1)
        axes[1, -1].set_ylim(0, 1)
        axes[1, -1].axis('off')
        axes[1, -1].set_title('Statistics')
        
        # 移除颜色条，保持简洁
        
        fig.suptitle(f'Post-Fusion Features - Batch {batch_idx}', fontsize=14)
        plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
        
        save_path = os.path.join(save_dir, f'{prefix}_batch_{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[ConvFuser Visualization] Saved post-fusion: {save_path}")

    def _visualize_comprehensive_comparison(self, img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, agent_names, batch_idx, save_dir):
        """创建综合对比图"""
        fig, axes = plt.subplots(3, len(agent_names) + 1, figsize=(4*(len(agent_names) + 1), 12))
        if len(agent_names) == 1:
            axes = axes.reshape(3, 2)
        
        # 收集所有特征图
        all_maps = []
        for agent_name in agent_names:
            img_map = torch.norm(img_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            lidar_map = torch.norm(lidar_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            cat_map = torch.norm(cat_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            all_maps.extend([img_map, lidar_map, cat_map])
        
        # 添加最终融合结果
        mm_map = torch.norm(mm_bev[batch_idx], dim=0).detach().cpu().numpy()
        all_maps.append(mm_map)
        
        # 计算统一的颜色范围
        all_values = np.concatenate([m.flatten() for m in all_maps])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        for i, agent_name in enumerate(agent_names):
            img_map = torch.norm(img_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            lidar_map = torch.norm(lidar_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            cat_map = torch.norm(cat_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            
            # 第一行：图像BEV
            im1 = axes[0, i].imshow(img_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'{agent_name}\nImage BEV')
            axes[0, i].set_ylabel('Y')
            
            # 第二行：激光雷达BEV
            im2 = axes[1, i].imshow(lidar_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f'{agent_name}\nLidar BEV')
            axes[1, i].set_ylabel('Y')
            
            # 第三行：融合后特征
            im3 = axes[2, i].imshow(cat_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f'{agent_name}\nFused BEV')
            axes[2, i].set_ylabel('Y')
        
        # 最后一列：最终多模态融合结果
        im_final = axes[0, -1].imshow(mm_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
        axes[0, -1].set_title('Final\nMulti-Modal BEV')
        axes[0, -1].set_ylabel('Y')
        
        # 中间和底部行显示统计信息
        occupied_cells = np.count_nonzero(mm_map)
        total_cells = mm_map.size
        occupancy_ratio = occupied_cells / total_cells
        
        stats_text = f"""Final Fusion Result:
            Occupied: {occupied_cells:,}/{total_cells:,}
            Ratio: {occupancy_ratio:.2%}"""
        
        axes[1, -1].text(0.1, 0.9, stats_text, transform=axes[1, -1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, -1].set_xlim(0, 1)
        axes[1, -1].set_ylim(0, 1)
        axes[1, -1].axis('off')
        axes[1, -1].set_title('Final Statistics')
        
        # 第三行显示融合过程
        axes[2, -1].text(0.1, 0.9, 'ConvFuser Pipeline:\n1. Image BEV\n2. Lidar BEV\n3. Agent Fusion\n4. Multi-Modal Fusion', 
                        transform=axes[2, -1].transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[2, -1].set_xlim(0, 1)
        axes[2, -1].set_ylim(0, 1)
        axes[2, -1].axis('off')
        axes[2, -1].set_title('Pipeline')
        
        # 移除颜色条，保持简洁
        
        fig.suptitle(f'ConvFuser Comprehensive Comparison - Batch {batch_idx}', fontsize=16)
        plt.subplots_adjust(top=0.85, bottom=0.2, hspace=0.4)
        
        save_path = os.path.join(save_dir, f'comprehensive_comparison_batch_{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[ConvFuser Visualization] Saved comprehensive comparison: {save_path}")


def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert an int, list, or tuple into a validated 2-tuple."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-element window size, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class OffsetPredictor(nn.Module):
    """Predict multi-point local offsets for one BEV source."""

    def __init__(
        self,
        channels: int,
        num_points: int,
        offset_range: float,
        hidden_channels: Optional[int] = None,
    ) -> None:
        """Initialize the offset predictor.

        Args:
            channels: Input feature channel count.
            num_points: Number of offsets predicted at each BEV location.
            offset_range: Local offset range `r`; outputs are constrained to `[-r, r]`.
            hidden_channels: Optional hidden channel count for the predictor head.
        """
        super().__init__()
        hidden_dim = hidden_channels if hidden_channels is not None else channels
        self.num_points = num_points
        self.offset_range = float(offset_range)
        self.offset_head = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_points * 2, kernel_size=1, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_raw: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict constrained offsets.

        Args:
            x: Input tensor of shape `[B, C, H, W]`.
            return_raw: Whether to also return unconstrained raw offsets.

        Returns:
            Constrained offsets of shape `[B, K + 1, 2, H, W]`, where an
            additional center point `(0, 0)` is prepended before the learned
            offsets.
            If `return_raw=True`, also returns raw offsets with the same shape.
        """
        batch_size, _, height, width = x.shape
        raw_offsets = self.offset_head(x).reshape(batch_size, self.num_points, 2, height, width)
        offsets = torch.tanh(raw_offsets) * self.offset_range
        center_offsets = torch.zeros(
            batch_size,
            1,
            2,
            height,
            width,
            device=x.device,
            dtype=x.dtype,
        )
        offsets = torch.cat([center_offsets, offsets], dim=1)
        if return_raw:
            raw_offsets = torch.cat([center_offsets, raw_offsets], dim=1)
            return offsets, raw_offsets
        return offsets


class MultiPointSampler(nn.Module):
    """Sample K points from one BEV source using deformable bilinear sampling."""

    def __init__(
        self,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
    ) -> None:
        """Initialize the sampler.

        Args:
            align_corners: Explicit `grid_sample` alignment choice.
            padding_mode: Explicit `grid_sample` padding mode shared by all sources.
        """
        super().__init__()
        self.align_corners = align_corners
        self.padding_mode = padding_mode

    def _normalize_coordinate(self, coord: torch.Tensor, size: int) -> torch.Tensor:
        """Normalize pixel coordinates into `[-1, 1]` for `grid_sample`."""
        if self.align_corners:
            if size <= 1:
                return torch.zeros_like(coord)
            return (2.0 * coord / float(size - 1)) - 1.0
        return ((2.0 * coord + 1.0) / float(size)) - 1.0

    def forward(self, feature: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Sample K offset-guided points from the same BEV source.

        Args:
            feature: Source feature map of shape `[B, C, H, W]`.
            offsets: Constrained offsets of shape `[B, K, 2, H, W]`.

        Returns:
            Sampled features of shape `[B, K, C, H, W]`.
        """
        if feature.dim() != 4:
            raise ValueError(f"feature must be [B, C, H, W], got {feature.shape}")
        if offsets.dim() != 5:
            raise ValueError(f"offsets must be [B, K, 2, H, W], got {offsets.shape}")

        batch_size, channels, height, width = feature.shape
        _, num_points, offset_dim, offset_height, offset_width = offsets.shape
        if offset_dim != 2:
            raise ValueError(f"offset dimension must be 2, got {offset_dim}")
        if offset_height != height or offset_width != width:
            raise ValueError("offset spatial shape must match feature spatial shape")

        yy, xx = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=feature.dtype),
            torch.arange(width, device=feature.device, dtype=feature.dtype),
            indexing='ij',
        )
        base_x = xx.unsqueeze(0).unsqueeze(1).expand(batch_size, num_points, -1, -1)
        base_y = yy.unsqueeze(0).unsqueeze(1).expand(batch_size, num_points, -1, -1)
        offset_x = offsets[:, :, 0, :, :]
        offset_y = offsets[:, :, 1, :, :]
        sample_x = base_x + offset_x
        sample_y = base_y + offset_y
        grid_x = self._normalize_coordinate(sample_x, width)
        grid_y = self._normalize_coordinate(sample_y, height)
        grid = torch.stack([grid_x, grid_y], dim=-1)

        feature_bk = (
            feature.unsqueeze(1)
            .expand(-1, num_points, -1, -1, -1)
            .reshape(batch_size * num_points, channels, height, width)
        )
        grid_bk = grid.reshape(batch_size * num_points, height, width, 2)
        sampled = F.grid_sample(
            feature_bk,
            grid_bk,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return sampled.view(batch_size, num_points, channels, height, width)


class SampleAttentionAggregator(nn.Module):
    """Aggregate K sampled features into one refined feature map."""

    def __init__(
        self,
        channels: int,
        num_points: int,
        ffn_ratio: float = 2.0,
        drop_rate: float = 0.1,
        layer_scale_init: float = 1e-2,
    ) -> None:
        """Initialize the DA-like query-only sample attention head.

        Args:
            channels: Sample feature channel count.
            num_points: Number of sampled points for per-point weight prediction.
            ffn_ratio: FFN hidden ratio on top of the aggregated feature.
            drop_rate: Dropout rate used in aggregation and FFN branches.
            layer_scale_init: Initial value for residual layer scales.
        """
        super().__init__()
        hidden_dim = max(int(channels * ffn_ratio), channels)
        self.num_points = int(num_points)
        self.query_norm = LayerNorm2d(channels)
        self.value_norm = LayerNorm2d(channels)
        self.attention_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(channels, self.num_points, kernel_size=1, bias=True),
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Dropout(drop_rate),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
            nn.Dropout(drop_rate),
        )
        self.layer_scale_attn = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )
        self.layer_scale_ffn = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )

    def forward(
        self,
        sampled_features: torch.Tensor,
        query_feature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict DA-like query-only weights and aggregate sampled features.

        Args:
            sampled_features: Tensor of shape `[B, K, C, H, W]`.
            query_feature: Query tensor of shape `[B, C, H, W]`. When omitted,
                the mean of sampled features is used as the query feature.

        Returns:
            A tuple containing:
                - attention weights of shape `[B, K, H, W]`
                - aggregated feature of shape `[B, C, H, W]`
                - refined feature of shape `[B, C, H, W]`
        """
        if sampled_features.dim() != 5:
            raise ValueError(
                f"sampled_features must be [B, K, C, H, W], got {sampled_features.shape}"
            )

        batch_size, num_points, channels, height, width = sampled_features.shape
        if num_points != self.num_points:
            raise ValueError(
                f"sampled_features point count must be {self.num_points}, got {num_points}"
            )
        if query_feature.shape != (batch_size, channels, height, width):
            raise ValueError(
                "query_feature must match sampled feature shape [B, C, H, W], got "
                f"{query_feature.shape}"
            )

        residual_query = query_feature
        attention_logits = self.attention_head(self.query_norm(query_feature)) #TODO
        attention_weights = torch.softmax(attention_logits, dim=1)

        flat_samples = sampled_features.reshape(batch_size * num_points, channels, height, width)
        value = self.value_norm(flat_samples).reshape(batch_size, num_points, channels, height, width)
        aggregated_feature = (attention_weights.unsqueeze(2) * value).sum(dim=1)
        refined_feature = residual_query + self.layer_scale_attn * self.output_proj(aggregated_feature)#TODO 基本被压缩到0了，不行
        refined_feature = refined_feature + self.layer_scale_ffn * self.ffn(refined_feature) 
        return attention_weights, aggregated_feature, refined_feature


class SemanticHead(nn.Module):
    """Predict a compact semantic descriptor from refined BEV features."""

    def __init__(self, channels: int) -> None:
        """Initialize the semantic head."""
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 8, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(8, 4, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a 4-channel semantic cue map."""
        return self.head(x)


class CueBasedFinalGateHead(nn.Module):
    """Predict final ego-centric gates from explicit cues."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        drop_rate: float = 0.1,
    ) -> None:
        """Initialize the final gate head."""
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a gate map in `[0, 1]`."""
        return torch.sigmoid(self.head(x))


class ForegroundReliabilityGateHead(nn.Module):
    """Predict ego-centric gates from refined features and reliability cues."""

    def __init__(
        self,
        channels: int = 128,
        feature_mid_channels: int = 32,
        feature_out_channels: int = 16,
        reliability_in_channels: int = 3,
        reliability_out_channels: int = 4,
        gate_hidden_channels: int = 16,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_descriptor = nn.Sequential(
            nn.Conv2d(channels, feature_mid_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(feature_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                feature_mid_channels,
                feature_mid_channels,
                kernel_size=3,
                padding=1,
                groups=feature_mid_channels,
                bias=False,
            ),
            nn.Conv2d(feature_mid_channels, feature_out_channels, kernel_size=1, bias=False),
            LayerNorm2d(feature_out_channels),
            nn.ReLU(inplace=True),
        )
        self.reliability_descriptor = nn.Sequential(
            nn.Conv2d(reliability_in_channels, reliability_out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        gate_layers: List[nn.Module] = [
            nn.Conv2d(
                feature_out_channels + reliability_out_channels,
                gate_hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ]
        if drop_rate > 0.0:
            gate_layers.append(nn.Dropout2d(drop_rate))
        gate_layers.append(nn.Conv2d(gate_hidden_channels, 1, kernel_size=1, bias=True))
        self.gate_head = nn.Sequential(*gate_layers)
        nn.init.constant_(self.gate_head[-1].bias, 0.0)

    def forward(
        self,
        refined_feature: torch.Tensor,
        reliability_cues: Dict[str, torch.Tensor],
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feature_desc = self.feature_descriptor(refined_feature)
        reliability_input = torch.cat(
            [
                reliability_cues['attention_conf'],
                reliability_cues['mean_offset'],
                reliability_cues['sample_agree'],
            ],
            dim=1,
        )
        reliability_desc = self.reliability_descriptor(reliability_input)
        gate_input = torch.cat([feature_desc, reliability_desc], dim=1)
        gate_logit = self.gate_head(gate_input)
        gate = torch.sigmoid(gate_logit)
        if return_debug:
            return gate, {
                'feature_descriptor_mean': feature_desc.mean(dim=1, keepdim=True),
                'feature_descriptor_norm': torch.norm(feature_desc, dim=1, keepdim=True),
                'reliability_descriptor_mean': reliability_desc.mean(dim=1, keepdim=True),
                'attention_conf': reliability_cues['attention_conf'],
                'mean_offset': reliability_cues['mean_offset'],
                'sample_agree': reliability_cues['sample_agree'],
            }
        return gate


class GeometricTransmissionGate(nn.Module):
    """Convert offset magnitude into geometric transmission weights."""

    def __init__(self, alpha: float) -> None:
        """Initialize the geometric gate.

        Args:
            alpha: Exponential decay factor in `exp(-alpha * d)`.
        """
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, offsets: torch.Tensor, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute offset-aware transmission gates.

        Args:
            offsets: Offset tensor of shape `[B, K, 2, H, W]`.
            attention_weights: Attention weights of shape `[B, K, H, W]`.

        Returns:
            A tuple containing:
                - distance map `[B, 1, H, W]`
                - gate map `[B, 1, H, W]`
        """
        if offsets.dim() != 5:
            raise ValueError(f"offsets must be [B, K, 2, H, W], got {offsets.shape}")
        if attention_weights.dim() != 4:
            raise ValueError(
                f"attention_weights must be [B, K, H, W], got {attention_weights.shape}"
            )

        offset_distance = torch.norm(offsets, p=2, dim=2)
        distance_map = (attention_weights * offset_distance).sum(dim=1, keepdim=True)
        gate_map = torch.exp(-self.alpha * distance_map)
        return distance_map, gate_map


class EgoFusionLayer(nn.Module):
    """Fuse ego, RSU, and drone refined features into one ego-centric BEV."""

    def __init__(self, channels: int) -> None:
        """Initialize the ego-centric source fusion layer.

        Args:
            channels: Channel count of each source BEV feature.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        refined_veh: torch.Tensor,
        refined_rsu: torch.Tensor,
        refined_drone: torch.Tensor,
        gate_rsu: torch.Tensor,
        gate_drone: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse three refined BEV maps.

        Args:
            refined_veh: Vehicle feature `[B, C, H, W]`.
            refined_rsu: RSU feature `[B, C, H, W]`.
            refined_drone: Drone feature `[B, C, H, W]`.
            gate_rsu: RSU gate `[B, 1, H, W]`.
            gate_drone: Drone gate `[B, 1, H, W]`.

        Returns:
            Ego-centric fused feature of shape `[B, C, H, W]`.
        """
        fused_input = torch.cat(
            [
                refined_veh,
                gate_rsu * refined_rsu,
                gate_drone * refined_drone,
            ],
            dim=1,
        )
        return self.proj(fused_input)


class WindowPositionalEncoding(nn.Module):
    """Add learnable 2D positional embedding inside each local window."""

    def __init__(self, channels: int, window_size: Union[int, Tuple[int, int]]) -> None:
        """Initialize the local window positional encoding.

        Args:
            channels: Channel count of the local window features.
            window_size: Window size `(wh, ww)` or scalar.
        """
        super().__init__()
        window_height, window_width = _to_2tuple(window_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, window_height, window_width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to window features.

        Args:
            x: Window features of shape `[B_windows, C, wh, ww]`.

        Returns:
            Position-enhanced window features with the same shape.
        """
        return x + self.pos_embed


class LocalReorderHead(nn.Module):
    """Predict one local scan-priority score per token within each window."""

    def __init__(self, channels: int) -> None:
        """Initialize the reorder score head.

        Args:
            channels: Input channel count.
        """
        super().__init__()
        self.score_head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict local reorder scores.

        Args:
            x: Window tensor of shape `[B_windows, C, wh, ww]`.

        Returns:
            Flattened local scores of shape `[B_windows, wh * ww]`.
        """
        return self.score_head(x).flatten(start_dim=1)


class LocalMambaBlock(nn.Module):
    """Apply local window-wise reordering and local Mamba modeling."""

    def __init__(
        self,
        channels: int,
        window_size: Union[int, Tuple[int, int]],
        depth: int = 1,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        forward_type: str = 'v05_noz',
    ) -> None:
        """Initialize the local Mamba block.

        Args:
            channels: Input channel count.
            window_size: Local window size `(wh, ww)` or scalar.
            depth: Number of local VSS blocks.
            ssm_d_state: VSS state size.
            ssm_ratio: VSS inner ratio.
            ssm_dt_rank: VSS dt rank.
            ssm_conv: VSS depthwise conv kernel size.
            ssm_conv_bias: Whether VSS depthwise conv uses bias.
            mlp_ratio: VSS MLP ratio.
            mlp_drop_rate: VSS MLP dropout rate.
            forward_type: Existing VSS forward type to reuse.
        """
        super().__init__()
        self.window_size = _to_2tuple(window_size)
        self.pos_encoding = WindowPositionalEncoding(channels=channels, window_size=self.window_size)
        self.reorder_head = LocalReorderHead(channels=channels)
        self.local_blocks = nn.Sequential(
            *[
                VSSBlock(
                    hidden_dim=channels,
                    cross_dim=0,
                    drop_path=0.0,
                    norm_layer=LayerNorm2d,
                    channel_first=True,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=False,
                    use_checkpoint=False,
                )
                for _ in range(depth)
            ]
        )

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Zero-pad BEV features so that they can be partitioned into windows."""
        batch_size, channels, height, width = x.shape
        window_height, window_width = self.window_size
        pad_height = (window_height - (height % window_height)) % window_height
        pad_width = (window_width - (width % window_width)) % window_width
        if pad_height > 0 or pad_width > 0:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0.0)
        padded_height = height + pad_height
        padded_width = width + pad_width
        meta = {
            'batch_size': batch_size,
            'channels': channels,
            'height': height,
            'width': width,
            'padded_height': padded_height,
            'padded_width': padded_width,
            'pad_height': pad_height,
            'pad_width': pad_width,
        }
        return x, meta

    def _partition_windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Partition BEV features into local windows."""
        padded_x, meta = self._pad_to_window(x)
        batch_size, channels, padded_height, padded_width = padded_x.shape
        window_height, window_width = self.window_size
        num_windows_h = padded_height // window_height
        num_windows_w = padded_width // window_width

        windows = padded_x.reshape(
            batch_size,
            channels,
            num_windows_h,
            window_height,
            num_windows_w,
            window_width,
        )
        windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.reshape(
            batch_size * num_windows_h * num_windows_w,
            channels,
            window_height,
            window_width,
        )

        meta['num_windows_h'] = num_windows_h
        meta['num_windows_w'] = num_windows_w
        return windows, meta

    def _merge_windows(self, windows: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
        """Merge local windows back to the BEV grid and crop zero padding."""
        batch_size = meta['batch_size']
        channels = meta['channels']
        num_windows_h = meta['num_windows_h']
        num_windows_w = meta['num_windows_w']
        window_height, window_width = self.window_size

        merged = windows.reshape(batch_size, num_windows_h, num_windows_w, channels, window_height, window_width)
        merged = merged.permute(0, 3, 1, 4, 2, 5).contiguous()
        merged = merged.reshape(batch_size, channels, meta['padded_height'], meta['padded_width'])
        return merged[:, :, :meta['height'], :meta['width']]

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run local reordering and local Mamba.

        Args:
            x: Input feature map of shape `[B, C, H, W]`.
            return_aux: Whether to return reorder-related auxiliary tensors.

        Returns:
            Local fused feature of shape `[B, C, H, W]`.
            If `return_aux=True`, also returns an auxiliary dictionary.
        """
        windows, meta = self._partition_windows(x)
        windows = self.pos_encoding(windows)
        reorder_scores = self.reorder_head(windows)

        batch_windows, channels, window_height, window_width = windows.shape
        token_count = window_height * window_width
        tokens = windows.reshape(batch_windows, channels, token_count)

        sort_idx = torch.argsort(reorder_scores, dim=-1, descending=True)
        gather_idx = sort_idx.unsqueeze(1).expand(-1, channels, -1)
        sorted_tokens = torch.gather(tokens, dim=2, index=gather_idx)

        # 首版安全实现：按行优先将排序后的 token 序列回填成同尺寸伪 2D 网格，
        # 以便直接复用现有 2D `VSSBlock`，避免额外引入新的 1D Mamba 路径。
        sorted_windows = sorted_tokens.reshape(batch_windows, channels, window_height, window_width)
        local_windows = self.local_blocks(sorted_windows)

        local_tokens = local_windows.reshape(batch_windows, channels, token_count)
        inverse_idx = torch.argsort(sort_idx, dim=-1)
        inverse_gather_idx = inverse_idx.unsqueeze(1).expand(-1, channels, -1)
        restored_tokens = torch.gather(local_tokens, dim=2, index=inverse_gather_idx)
        restored_windows = restored_tokens.reshape(batch_windows, channels, window_height, window_width)
        restored_feature = self._merge_windows(restored_windows, meta)

        if not return_aux:
            return restored_feature

        reorder_score_map = reorder_scores.reshape(
            meta['batch_size'],
            meta['num_windows_h'],
            meta['num_windows_w'],
            window_height,
            window_width,
        )
        aux_info = {
            'reorder_scores': reorder_score_map,
            'sort_idx': sort_idx.reshape(
                meta['batch_size'],
                meta['num_windows_h'],
                meta['num_windows_w'],
                token_count,
            ),
        }
        return restored_feature, aux_info


class HilbertGlobalMambaBlock(nn.Module):
    """Apply global 1D Mamba after Hilbert-order serialization."""

    def __init__(
        self,
        channels: int,
        depth: int = 1,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        use_reverse_scan: bool = True,
        norm_epsilon: float = 1e-5,
    ) -> None:
        """Initialize the Hilbert global fusion block.

        Notes:
            This first dense-BEV version only borrows the Hilbert-order global
            serialization idea from the old sparse `GlobalMamba`. It intentionally
            does not reproduce the old sparse downsample / upsample path.

        Args:
            channels: Input channel count.
            depth: Number of 1D Mamba blocks.
            ssm_d_state: 1D Mamba state size.
            ssm_ratio: 1D Mamba expansion ratio.
            ssm_dt_rank: 1D Mamba dt rank.
            ssm_conv: 1D Mamba local conv kernel size.
            ssm_conv_bias: Whether 1D Mamba conv uses bias.
            use_reverse_scan: Whether to alternate reverse scanning across layers.
            norm_epsilon: Final sequence norm epsilon.
        """
        super().__init__()
        self.channels = channels
        self.use_reverse_scan = use_reverse_scan
        self.curve_template: Dict[str, torch.Tensor] = {}
        self.hilbert_spatial_size: Dict[str, Tuple[int, int, int]] = {}
        self._hilbert_cache: Dict[Tuple[int, int, int, str], Dict[str, torch.Tensor]] = {}

        self.pos_embed = nn.Sequential(
            nn.Linear(3, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        ssm_cfg = {
            'd_state': ssm_d_state,
            'd_conv': ssm_conv,
            'expand': ssm_ratio,
            'dt_rank': ssm_dt_rank,
            'conv_bias': ssm_conv_bias,
        }
        self.mamba_layers = nn.ModuleList(
            [
                create_block(
                    d_model=channels,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=False,
                    residual_in_fp32=True,
                    fused_add_norm=False,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(depth)
            ]
        )
        self.norm_f = nn.LayerNorm(channels, eps=norm_epsilon)

        template_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'ckpts', 'hilbert_template')
        )
        for rank in (7, 8, 9, 10):
            self._load_template(
                os.path.join(template_root, f'curve_template_3d_rank_{rank}.pth'),
                rank,
            )

    def _load_template(self, path: str, rank: int) -> None:
        """Load one Hilbert curve template."""
        template = torch.load(path, map_location='cpu')
        if isinstance(template, dict):
            curve = template['data'].reshape(-1).long()
            spatial_size = tuple(template['size'])
        else:
            curve = template.reshape(-1).long()
            side = 2 ** rank
            spatial_size = (1, side, side)
        self.curve_template[f'curve_template_rank{rank}'] = curve
        self.hilbert_spatial_size[f'curve_template_rank{rank}'] = spatial_size

    def _select_template_key(self, height: int, width: int) -> str:
        """Select a Hilbert template rank that covers the current BEV size."""
        max_dim = max(height, width)
        if max_dim > 512:
            template_key = 'curve_template_rank10'
        elif max_dim > 256:
            template_key = 'curve_template_rank9'
        elif max_dim > 128:
            template_key = 'curve_template_rank8'
        else:
            template_key = 'curve_template_rank7'

        _, hilbert_height, hilbert_width = self.hilbert_spatial_size[template_key]
        if hilbert_height < height or hilbert_width < width:
            raise ValueError(
                f"Hilbert template {template_key} cannot cover dense BEV size {(height, width)}. "
                f"Template spatial size is {(hilbert_height, hilbert_width)}."
            )
        return template_key

    def _build_hilbert_cache(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Build and cache Hilbert ordering indices for a dense BEV grid."""
        cache_key = (batch_size, height, width, str(device))
        if cache_key in self._hilbert_cache:
            return self._hilbert_cache[cache_key]

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.long),
            torch.arange(width, device=device, dtype=torch.long),
            indexing='ij',
        )
        yy_flat = yy.reshape(-1)
        xx_flat = xx.reshape(-1)
        token_count = yy_flat.numel()

        coords_all = []
        pos_coords = []
        denom_h = max(height - 1, 1)
        denom_w = max(width - 1, 1)
        base_pos = torch.stack(
            [
                torch.zeros_like(yy_flat, dtype=torch.float32),
                yy_flat.to(torch.float32) / float(denom_h),
                xx_flat.to(torch.float32) / float(denom_w),
            ],
            dim=1,
        )
        for batch_idx in range(batch_size):
            batch_col = torch.full((token_count, 1), batch_idx, device=device, dtype=torch.long)
            z_col = torch.zeros((token_count, 1), device=device, dtype=torch.long)
            coords_all.append(torch.cat([batch_col, z_col, yy_flat[:, None], xx_flat[:, None]], dim=1))
            pos_coords.append(base_pos)

        coords = torch.cat(coords_all, dim=0)
        pos_coords_tensor = torch.stack(pos_coords, dim=0)

        template_key = self._select_template_key(height, width)
        template = self.curve_template[template_key].to(device)
        hilbert_size = self.hilbert_spatial_size[template_key]
        index_info = get_hilbert_index_2d_mamba_lite(
            template=template,
            coors=coords,
            batch_size=batch_size,
            hilbert_spatial_size=hilbert_size,
            shift=(0, 0),
            debug=False,
        )
        sort_idx = torch.stack(
            [index_info['inds_curt_to_next'][batch_idx] for batch_idx in range(batch_size)],
            dim=0,
        )
        inverse_idx = torch.stack(
            [index_info['inds_next_to_curt'][batch_idx] for batch_idx in range(batch_size)],
            dim=0,
        )
        cache = {
            'sort_idx': sort_idx,
            'inverse_idx': inverse_idx,
            'pos_coords': pos_coords_tensor,
        }
        self._hilbert_cache[cache_key] = cache
        return cache

    def _run_sequence_mamba(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run stacked 1D Mamba blocks on serialized global tokens."""
        hidden = tokens
        residual = None
        for layer_idx, layer in enumerate(self.mamba_layers):
            if self.use_reverse_scan and (layer_idx % 2 == 1):
                hidden_rev = hidden.flip(1)
                residual_rev = residual.flip(1) if residual is not None else None
                hidden_rev, residual_rev = layer(hidden_rev, residual_rev)
                hidden = hidden_rev.flip(1)
                residual = residual_rev.flip(1) if residual_rev is not None else None
            else:
                hidden, residual = layer(hidden, residual)
        residual = (hidden + residual) if residual is not None else hidden
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Serialize dense BEV tokens by Hilbert order, run 1D Mamba, and restore."""
        batch_size, channels, height, width = x.shape
        cache = self._build_hilbert_cache(batch_size, height, width, x.device)
        tokens = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        pos_embed = self.pos_embed(cache['pos_coords'].to(dtype=x.dtype))
        tokens = tokens + pos_embed

        sort_idx = cache['sort_idx'].unsqueeze(-1).expand(-1, -1, channels)
        inverse_idx = cache['inverse_idx'].unsqueeze(-1).expand(-1, -1, channels)

        ordered_tokens = torch.gather(tokens, dim=1, index=sort_idx)
        global_tokens = self._run_sequence_mamba(ordered_tokens)
        restored_tokens = torch.gather(global_tokens, dim=1, index=inverse_idx)
        return restored_tokens.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


class GlobalFusionBlock(nn.Module):
    """Apply configurable global fusion after local Mamba."""

    def __init__(
        self,
        channels: int,
        global_fusion_type: str = 'mamba',
        mamba_depth: int = 1,
        conv_depth: int = 2,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        forward_type: str = 'v05_noz',
    ) -> None:
        """Initialize the global fusion backend.

        Args:
            channels: Input channel count.
            global_fusion_type: Backend type, one of `'mamba'`, `'conv'`, or `'mamba_hilbert'`.
            mamba_depth: Number of global VSS blocks.
            conv_depth: Number of conv blocks when using conv backend.
            ssm_d_state: VSS state size.
            ssm_ratio: VSS inner ratio.
            ssm_dt_rank: VSS dt rank.
            ssm_conv: VSS depthwise conv kernel size.
            ssm_conv_bias: Whether VSS depthwise conv uses bias.
            mlp_ratio: VSS MLP ratio.
            mlp_drop_rate: VSS MLP dropout rate.
            forward_type: Existing VSS forward type to reuse.
        """
        super().__init__()
        self.global_fusion_type = global_fusion_type
        if global_fusion_type == 'mamba':
            self.backend = nn.Sequential(
                *[
                    VSSBlock(
                        hidden_dim=channels,
                        cross_dim=0,
                        drop_path=0.0,
                        norm_layer=LayerNorm2d,
                        channel_first=True,
                        ssm_d_state=ssm_d_state,
                        ssm_ratio=ssm_ratio,
                        ssm_dt_rank=ssm_dt_rank,
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=ssm_conv_bias,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type=forward_type,
                        mlp_ratio=mlp_ratio,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=mlp_drop_rate,
                        gmlp=False,
                        use_checkpoint=False,
                    )
                    for _ in range(mamba_depth)
                ]
            )
        elif global_fusion_type == 'mamba_hilbert':
            self.backend = HilbertGlobalMambaBlock(
                channels=channels,
                depth=mamba_depth,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
            )
        elif global_fusion_type == 'conv':
            conv_blocks = []
            for _ in range(conv_depth):
                conv_blocks.append(DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, padding=1))
            self.backend = nn.Sequential(*conv_blocks)
        else:
            raise ValueError(
                f"Unsupported global_fusion_type: {global_fusion_type}. "
                "Expected 'mamba', 'conv', or 'mamba_hilbert'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured global fusion backend.

        Args:
            x: Input feature map of shape `[B, C, H, W]`.

        Returns:
            Output feature map of shape `[B, C, H, W]`.
        """
        return self.backend(x)


class OffsetGuidedSelectiveHierarchicalMambaFusionBlock(nn.Module):
    """Fuse vehicle, RSU, and drone BEV features with selective hierarchical Mamba."""

    def __init__(
        self,
        channels: int,
        num_points: int = 4,
        offset_range: float = 2.0,
        transmission_alpha: float = 1.0,
        window_size: Union[int, Tuple[int, int]] = 4,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
        local_mamba_depth: int = 1,
        global_fusion_type: str = 'mamba',
        global_mamba_depth: int = 1,
        global_conv_depth: int = 2,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        forward_type: str = 'v05_noz',
        sample_da_ffn_ratio: float = 2.0,
        sample_da_drop_rate: float = 0.1,
        sample_da_layer_scale_init: float = 1e-2,
        final_gate_hidden_dim: int = 32,
        gate_head_cfg: Optional[Dict[str, Any]] = None,
        sparse_temporal_cfg: Optional[Dict[str, Any]] = None,
        gate_topk_cfg: Optional[Dict[str, Any]] = None,
        visualize_gate_maps: bool = False,
    ) -> None:
        """Initialize the selective hierarchical fusion block.

        Args:
            channels: Channel count of each aligned BEV feature map.
            num_points: Number of deformable sampling points per location.
            offset_range: Local offset range `r`.
            transmission_alpha: Exponential decay factor for external source gates.
            window_size: Local window size for local reorder and local Mamba.
            align_corners: Explicit `grid_sample` alignment choice.
            padding_mode: Explicit `grid_sample` padding mode shared by all sources.
            local_mamba_depth: Number of local VSS blocks.
            global_fusion_type: Global backend type, `'mamba'` or `'conv'`.
            global_mamba_depth: Number of global VSS blocks.
            global_conv_depth: Number of conv blocks in conv backend.
            ssm_d_state: Shared VSS state size.
            ssm_ratio: Shared VSS inner ratio.
            ssm_dt_rank: Shared VSS dt rank.
            ssm_conv: Shared VSS depthwise conv kernel size.
            ssm_conv_bias: Shared VSS depthwise conv bias flag.
            mlp_ratio: Shared VSS MLP ratio.
            mlp_drop_rate: Shared VSS MLP dropout rate.
            forward_type: Existing VSS forward type to reuse.
            sample_da_ffn_ratio: DA attention FFN hidden ratio.
            sample_da_drop_rate: DA attention dropout rate.
            sample_da_layer_scale_init: DA attention layer scale init.
            final_gate_hidden_dim: Hidden channel count of the final gate head.
            sparse_temporal_cfg: Optional sparse temporal refinement config.
            gate_topk_cfg: Optional gate topk config.
            visualize_gate_maps: Whether to save gate-map debug images during training.
        """
        super().__init__()
        self.visualize_gate_maps = bool(visualize_gate_maps)
        self.num_points = num_points
        self.total_num_points = num_points + 1
        self.offset_range = float(offset_range)
        self.semantic_channels = 4
        sparse_temporal_cfg = sparse_temporal_cfg or {}
        self.use_sparse_temporal = sparse_temporal_cfg.get('ENABLE', False)
        self.temporal_fusion_type = str(
            sparse_temporal_cfg.get('TYPE', 'sparse')
        ).lower()
        self.temporal_cue_distance_channels = int(
            sparse_temporal_cfg.get('CUE_DISTANCE_CHANNELS', 3)
        )
        self.offset_predictors = nn.ModuleDict(
            {
                'vehicle': OffsetPredictor(channels, num_points, offset_range),
                'rsu': OffsetPredictor(channels, num_points, offset_range),
                'drone': OffsetPredictor(channels, num_points, offset_range),
            }
        )
        self.sample_attention = nn.ModuleDict(
            {
                'vehicle': SampleAttentionAggregator(
                    channels=channels,
                    num_points=self.total_num_points,
                    ffn_ratio=sample_da_ffn_ratio,
                    drop_rate=sample_da_drop_rate,
                    layer_scale_init=sample_da_layer_scale_init,
                ),
                'rsu': SampleAttentionAggregator(
                    channels=channels,
                    num_points=self.total_num_points,
                    ffn_ratio=sample_da_ffn_ratio,
                    drop_rate=sample_da_drop_rate,
                    layer_scale_init=sample_da_layer_scale_init,
                ),
                'drone': SampleAttentionAggregator(
                    channels=channels,
                    num_points=self.total_num_points,
                    ffn_ratio=sample_da_ffn_ratio,
                    drop_rate=sample_da_drop_rate,
                    layer_scale_init=sample_da_layer_scale_init,
                ),
            }
        )
        self.sampler = MultiPointSampler(
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        self.transmission_gate = GeometricTransmissionGate(alpha=transmission_alpha)
        gate_head_cfg = gate_head_cfg or {}
        self.final_gate_heads = nn.ModuleDict(
            {
                'rsu': ForegroundReliabilityGateHead(
                    channels=channels,
                    feature_mid_channels=gate_head_cfg.get('FEATURE_MID_CHANNELS', 32),
                    feature_out_channels=gate_head_cfg.get('FEATURE_OUT_CHANNELS', 16),
                    reliability_in_channels=3,
                    reliability_out_channels=gate_head_cfg.get('RELIABILITY_OUT_CHANNELS', 4),
                    gate_hidden_channels=gate_head_cfg.get('GATE_HIDDEN_CHANNELS', 16),
                    drop_rate=sample_da_drop_rate,
                ),
                'drone': ForegroundReliabilityGateHead(
                    channels=channels,
                    feature_mid_channels=gate_head_cfg.get('FEATURE_MID_CHANNELS', 32),
                    feature_out_channels=gate_head_cfg.get('FEATURE_OUT_CHANNELS', 16),
                    reliability_in_channels=3,
                    reliability_out_channels=gate_head_cfg.get('RELIABILITY_OUT_CHANNELS', 4),
                    gate_hidden_channels=gate_head_cfg.get('GATE_HIDDEN_CHANNELS', 16),
                    drop_rate=sample_da_drop_rate,
                ),
            }
        )
        self.ego_fusion = EgoFusionLayer(channels=channels)
        self.local_mamba = LocalMambaBlock(
            channels=channels,
            window_size=window_size,
            depth=local_mamba_depth,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            forward_type=forward_type,
        )
        self.global_fusion = GlobalFusionBlock(
            channels=channels,
            global_fusion_type=global_fusion_type,
            mamba_depth=global_mamba_depth,
            conv_depth=global_conv_depth,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            forward_type=forward_type,
        )
        if self.use_sparse_temporal:
            point_cloud_range = sparse_temporal_cfg.get('POINT_CLOUD_RANGE')
            if point_cloud_range is None:
                raise ValueError(
                    "SPARSE_TEMPORAL.ENABLE=True requires SPARSE_TEMPORAL.POINT_CLOUD_RANGE"
                )
            if self.temporal_fusion_type == 'sparse':
                self.sparse_temporal_fusion = SparseTemporalFusionBlock(
                    channels=channels,
                    num_points=sparse_temporal_cfg.get('K_SAMPLES', num_points),
                    cue_distance_channels=self.temporal_cue_distance_channels,
                    point_cloud_range=tuple(point_cloud_range),
                    window_size=sparse_temporal_cfg.get('ROUTE_WINDOW', window_size),
                    need_topk=sparse_temporal_cfg.get('TOPK'),
                    need_threshold=sparse_temporal_cfg.get('NEED_THRESHOLD', 0.5),
                    need_smoothing_kernel_size=sparse_temporal_cfg.get(
                        'NEED_SMOOTHING_KERNEL_SIZE',
                        sparse_temporal_cfg.get('SMOOTHING_KERNEL_SIZE', 1),
                    ),
                    need_init_bias=sparse_temporal_cfg.get('NEED_INIT_BIAS', 0.25),
                    need_feature_channels=sparse_temporal_cfg.get('NEED_FEATURE_CHANNELS', 16),
                    need_residual_channels=sparse_temporal_cfg.get('NEED_RESIDUAL_CHANNELS', 16),
                    need_prior_clamp_min=sparse_temporal_cfg.get('NEED_PRIOR_CLAMP_MIN', 0.01),
                    need_prior_clamp_max=sparse_temporal_cfg.get('NEED_PRIOR_CLAMP_MAX', 0.99),
                    residual_offset_range=sparse_temporal_cfg.get('RESIDUAL_OFFSET_RANGE', offset_range),
                    base_scale_multiplier=sparse_temporal_cfg.get('BASE_SCALE_MULTIPLIER', 1.0),
                    base_scale_min=sparse_temporal_cfg.get('BASE_SCALE_MIN', 1.0),
                    base_scale_max=sparse_temporal_cfg.get('BASE_SCALE_MAX', 8.0),
                    base_motion_epsilon=sparse_temporal_cfg.get('BASE_MOTION_EPSILON', 0.5),
                    base_fixed_expansion_no_motion=sparse_temporal_cfg.get(
                        'BASE_FIXED_EXPANSION_NO_MOTION', 1.0
                    ),
                    include_diagonal_base_offsets=sparse_temporal_cfg.get('INCLUDE_DIAGONAL_BASE_OFFSETS', False),
                    offset_mode=sparse_temporal_cfg.get('OFFSET_MODE', 'guided'),
                    direct_offset_range=sparse_temporal_cfg.get('DIRECT_OFFSET_RANGE'),
                    align_corners=align_corners,
                    padding_mode=padding_mode,
                    return_debug=False,
                )
            elif self.temporal_fusion_type == 'concat':
                self.sparse_temporal_fusion = TemporalFusionConcatBlock(
                    channels=channels,
                    point_cloud_range=tuple(point_cloud_range),
                    hidden_channels=sparse_temporal_cfg.get('CONCAT_HIDDEN_CHANNELS'),
                    align_corners=align_corners,
                    padding_mode=padding_mode,
                    return_debug=False,
                )
            else:
                raise ValueError(
                    f"Unsupported SPARSE_TEMPORAL.TYPE={self.temporal_fusion_type}. "
                    "Expected 'sparse' or 'concat'."
                )
        else:
            self.sparse_temporal_fusion = None
        gate_topk_cfg = gate_topk_cfg or {}
        self.gate_topk_enabled = bool(gate_topk_cfg.get("TOPK_ENABLED", True))
        self.gate_inference_threshold = float(
            gate_topk_cfg.get("INFERENCE_THRESHOLD", 0.7)
        )
        self.grad_debug_step = 0
        self.gate_vis_counter = 0
        self.gate_vis_epoch = None

    def _apply_gate_topk(self, gate: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Binarize gate for fusion: top-K pixels pass full features (where2comm-style)."""
        if gate is None or not self.gate_topk_enabled:
            return gate

        batch_size, _, height, width = gate.shape
        flat_gate = gate.reshape(batch_size, -1)

        if self.training:
            k = int(height * width * 0.05)   #TODO: 这里的k是随机生成的，需要根据实际情况调整
            if k <= 0:
                hard_mask = torch.zeros_like(gate)
            else:
                k = min(k, flat_gate.shape[-1])
                _, indices = torch.topk(flat_gate, k=k, dim=-1, sorted=False)
                binary_flat = torch.zeros_like(flat_gate)
                binary_flat.scatter_(-1, indices, 1.0)
                hard_mask = binary_flat.reshape(batch_size, 1, height, width)
            # STE keeps gate_head trainable while forward uses hard top-K mask.
            return hard_mask + gate - gate.detach()

        return (gate > self.gate_inference_threshold).to(gate.dtype)

    def _maybe_print_gate_schedule_debug(
        self,
        gate_outputs: Dict[str, Optional[torch.Tensor]],
        epoch: Optional[int] = None,
    ) -> None:
        if not self.training:
            return
        print(f"[GateSchedule] epoch={epoch}")
        for name, gate in (("rsu", gate_outputs.get("gate_rsu")), ("drone", gate_outputs.get("gate_drone"))):
            if not isinstance(gate, torch.Tensor):
                continue
            with torch.no_grad():
                g = gate.detach().float()
                print(
                    f"[GateSchedule][{name}] raw_gate mean={g.mean().item():.4f} "
                    f"max={g.max().item():.4f} "
                    f"ratio>0.5={(g > 0.5).float().mean().item():.4f} "
                    f"ratio>0.7={(g > 0.7).float().mean().item():.4f}"
                )

    def _visualize_gate_maps(
        self,
        batch_dict: Optional[Dict[str, Any]],
        gate_outputs: Dict[str, Optional[torch.Tensor]],
        gate_inputs: Optional[Dict[str, Optional[Dict[str, torch.Tensor]]]] = None,
        gate_fusion_outputs: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> None:
        """Save RSU / Drone final gate maps and selected structured gate inputs."""
        if not self.visualize_gate_maps:
            return
        if not self.training:
            return
        if batch_dict is None:
            return

        save_dir = "./map_gate_visualization"
        max_samples_per_batch = 2
        dpi = 150
        cmap = "viridis"

        epoch_raw = batch_dict.get("epoch")
        epoch_num = None
        if epoch_raw is None:
            epoch_str = "unknown"
        elif isinstance(epoch_raw, torch.Tensor):
            epoch_num = int(epoch_raw.detach().cpu().item())
            epoch_str = str(epoch_num)
        elif isinstance(epoch_raw, float) and epoch_raw.is_integer():
            epoch_num = int(epoch_raw)
            epoch_str = str(epoch_num)
        elif isinstance(epoch_raw, int):
            epoch_num = epoch_raw
            epoch_str = str(epoch_raw)
        else:
            epoch_str = str(epoch_raw)
            try:
                epoch_num = int(epoch_raw)
            except (TypeError, ValueError):
                epoch_num = None

        if epoch_num is not None and self.gate_vis_epoch != epoch_num:
            self.gate_vis_epoch = epoch_num
            self.gate_vis_counter = 0

        step_num = self.gate_vis_counter
        self.gate_vis_counter += 1

        os.makedirs(save_dir, exist_ok=True)

        pairs = (
            ("rsu", gate_outputs.get("gate_rsu")),
            ("drone", gate_outputs.get("gate_drone")),
        )

        for agent_name, gate_tensor in pairs:
            if gate_tensor is None:
                continue
            fusion_gate_tensor = None
            if self.gate_topk_enabled and gate_fusion_outputs is not None:
                fusion_gate_tensor = gate_fusion_outputs.get(f"gate_{agent_name}")
            gate_cpu = gate_tensor.detach().float().cpu()
            if gate_cpu.dim() != 4 or gate_cpu.shape[1] != 1:
                continue
            input_tensors = None
            if gate_inputs is not None:
                input_tensors = gate_inputs.get(f"gate_{agent_name}")
            batch_size = gate_cpu.shape[0]
            n_save = min(batch_size, max_samples_per_batch)
            for sample_idx in range(n_save):
                gate_hw = gate_cpu[sample_idx, 0]
                gate_np = gate_hw.numpy()
                mean_val = float(gate_hw.mean().item())
                min_val = float(gate_hw.min().item())
                max_val = float(gate_hw.max().item())
                std_val = float(gate_hw.std().item())
                ratio_05 = float((gate_hw > 0.5).float().mean().item())
                ratio_07 = float((gate_hw > 0.7).float().mean().item())
                ratio_09 = float((gate_hw > 0.9).float().mean().item())

                fname = (
                    f"gate_epoch_{epoch_str}_step_{step_num:06d}_{agent_name}_b{sample_idx}.png"
                )
                save_path = os.path.join(save_dir, fname)

                panels = []
                if isinstance(input_tensors, dict):
                    for input_name in (
                        "feature_descriptor_mean",
                        "feature_descriptor_norm",
                        "reliability_descriptor_mean",
                    ):
                        input_tensor = input_tensors.get(input_name)
                        if not isinstance(input_tensor, torch.Tensor):
                            continue
                        input_cpu = input_tensor.detach().float().cpu()
                        if input_cpu.dim() != 4 or sample_idx >= input_cpu.shape[0]:
                            continue
                        if input_cpu.shape[1] == 1:
                            input_hw = input_cpu[sample_idx, 0]
                        else:
                            input_hw = input_cpu[sample_idx].mean(dim=0)
                        panels.append((input_name, input_hw.numpy(), input_hw))
                panels.append(("gate_map", gate_np, gate_hw))
                if isinstance(fusion_gate_tensor, torch.Tensor):
                    fusion_cpu = fusion_gate_tensor.detach().float().cpu()
                    if fusion_cpu.dim() == 4 and fusion_cpu.shape[1] == 1 and sample_idx < fusion_cpu.shape[0]:
                        fusion_hw = fusion_cpu[sample_idx, 0]
                        panels.append(("g_fusion", fusion_hw.numpy(), fusion_hw))

                fig, axes = plt.subplots(
                    1,
                    len(panels),
                    figsize=(4.0 * len(panels), 4.2),
                    constrained_layout=True,
                )
                if len(panels) == 1:
                    axes = [axes]
                for ax, (panel_name, panel_np, panel_hw) in zip(axes, panels):
                    if panel_name in (
                        "gate_map",
                        "g_fusion",
                        "feature_descriptor_mean",
                        "reliability_descriptor_mean",
                    ):
                        im = ax.imshow(panel_np, vmin=0.0, vmax=1.0, cmap=cmap)
                    else:
                        im = ax.imshow(panel_np, cmap=cmap)
                    fig.colorbar(im, ax=ax, shrink=0.75)
                    ax.set_title(
                        f"{panel_name}\n"
                        f"mean={panel_hw.mean().item():.4f}, std={panel_hw.std().item():.4f}",
                        fontsize=8,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                title = (
                    f"{agent_name}: mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}, "
                    f"std={std_val:.4f}\n"
                    f"ratio>0.5={ratio_05:.4f}, ratio>0.7={ratio_07:.4f}, ratio>0.9={ratio_09:.4f}"
                )
                fig.suptitle(title, fontsize=9)
                fig.savefig(save_path, dpi=dpi)
                plt.close(fig)

    def _refine_source(
        self,
        source_name: str,
        feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run offset prediction, deformable sampling, and sample aggregation for one source."""
        offsets = self.offset_predictors[source_name](feature, return_raw=False)
        sampled_features = self.sampler(feature, offsets)
        attention_weights, aggregated_feature, refined_feature = self.sample_attention[source_name](
            sampled_features,
            feature,
        )
        return {
            'sampled_features': sampled_features,       # [B, K, C, H, W]
            'attention_weights': attention_weights,     # [B, K, H, W]
            'offsets': offsets,                         # [B, K, 2, H, W]
            'query_feature': feature,                   # [B, C, H, W]
            'aggregated_feature': aggregated_feature,   # [B, C, H, W]
            'refined_feature': refined_feature,         # [B, C, H, W]
        }

    def _build_explicit_cues(
        self,
        offsets: torch.Tensor,
        attention_weights: torch.Tensor,
        query_feature: torch.Tensor,
        sampled_features: torch.Tensor,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """Build explicit per-offset cues for the final gate and temporal aux.

        `d_k` is offset L2 norm divided by `offset_range` so gating and temporal
        inputs share a comparable scale to learned offset magnitudes.
        """
        d_k_raw = torch.norm(offsets, p=2, dim=2)  # [B, K, H, W]
        range_scale = max(self.offset_range, eps)
        d_k = d_k_raw / range_scale
        w_k = attention_weights  # [B, K, H, W]
        query_norm = F.normalize(query_feature, p=2, dim=1, eps=eps).unsqueeze(1)
        sampled_norm = F.normalize(sampled_features, p=2, dim=2, eps=eps)
        s_k = (query_norm * sampled_norm).sum(dim=2)  # [B, K, H, W]
        ws_k = w_k * d_k * s_k  # [B, K, H, W], uses normalized d_k
        return {
            'd_k': d_k,
            'w_k': w_k,
            's_k': s_k,
            'ws_k': ws_k,
        }

    def _build_reliability_cues(
        self,
        source_outputs: Dict[str, torch.Tensor],
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """Build structured reliability cues for final external-source gating."""
        w = source_outputs['attention_weights']          # [B, K, H, W]
        offsets = source_outputs['offsets']              # [B, K, 2, H, W]
        sampled = source_outputs['sampled_features']     # [B, K, C, H, W]
        query = source_outputs['query_feature']          # [B, C, H, W]
        agg = source_outputs['aggregated_feature']       # [B, C, H, W]

        num_points = w.shape[1]
        entropy = -(w * torch.log(w + eps)).sum(dim=1, keepdim=True)
        attention_conf = 1.0 - entropy / math.log(float(num_points))
        attention_conf = attention_conf.clamp(0.0, 1.0)
        # 如果attention_weights在k个采样点上非常平均，说明采样不稳定，attention_conf就低

        offset_dist = torch.norm(offsets, p=2, dim=2) / max(float(self.offset_range), eps)
        mean_offset = (w * offset_dist).sum(dim=1, keepdim=True)
        # 没有明确含义，不希望offset太大或太小，但作为上下两项的依赖条件？

        diff = sampled - agg.unsqueeze(1)
        sample_var = (w.unsqueeze(2) * diff.pow(2)).sum(dim=1).mean(dim=1, keepdim=True)
        sample_agree = torch.exp(-sample_var)
        # 如果k个采样feature差异特别大，说明在当前位置附近特征不稳定，sample_agree就低

        update_mag = torch.norm(agg - query, p=2, dim=1, keepdim=True)
        update_mag = update_mag / (torch.norm(query, p=2, dim=1, keepdim=True) + eps)
        # 更新幅度，与offset类似，也不希望太大或太小，或许可以删去？

        return {
            'attention_conf': attention_conf,
            'mean_offset': mean_offset,
            'sample_agree': sample_agree,
            'update_mag': update_mag,
        }

    def _build_structured_final_gate(
        self,
        source_name: str,
        refined_feature: torch.Tensor,
        reliability_cues: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build final gate from refined features and reliability cues."""
        final_gate, gate_input_maps = self.final_gate_heads[source_name](
            refined_feature,
            reliability_cues,
            return_debug=True,
        )
        return final_gate, gate_input_maps

    @staticmethod
    def _stack_temporal_cue_distance(cue_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack summed spatial-fusion cues for the temporal need head `distance_*` slot."""
        sum_dk = cue_dict['d_k'].sum(dim=1, keepdim=True)  # [B, 1, H, W]
        sum_sk = cue_dict['s_k'].sum(dim=1, keepdim=True)
        sum_wsk = cue_dict['ws_k'].sum(dim=1, keepdim=True)
        return torch.cat([sum_dk, sum_sk, sum_wsk], dim=1)  # [B, 3, H, W]

    def _build_missing_source_placeholders(
        self,
        feature_vehicle: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create final-fusion placeholders for a missing external source.

        Args:
            feature_vehicle: Vehicle feature used only for shape/device/dtype reference.

        Returns:
            A tuple containing:
                - zero refined feature `[B, C, H, W]`
                - zero gate `[B, 1, H, W]`
                - zero temporal cue stack `[B, C_cue, H, W]` for sparse temporal `distance_*`
        """
        batch_size, _, height, width = feature_vehicle.shape
        zero_feature = torch.zeros_like(feature_vehicle)
        zero_gate = torch.zeros(
            (batch_size, 1, height, width),
            device=feature_vehicle.device,
            dtype=feature_vehicle.dtype,
        )
        zero_temporal_cues = torch.zeros(
            (batch_size, self.temporal_cue_distance_channels, height, width),
            device=feature_vehicle.device,
            dtype=feature_vehicle.dtype,
        )
        return zero_feature, zero_gate, zero_temporal_cues

    def _process_optional_external_source(
        self,
        source_name: str,
        feature: Optional[torch.Tensor],
        feature_vehicle: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Process one optional external source or create final-fusion placeholders."""
        
        if feature is None:
            zero_feature, zero_gate, zero_temporal_cues = self._build_missing_source_placeholders(
                feature_vehicle
            )
            return zero_feature, zero_gate, zero_temporal_cues, {}
        
        source_outputs = self._refine_source(
            source_name,
            feature,
        )
        reliability_cues = self._build_reliability_cues(source_outputs)
        final_gate, gate_input_maps = self._build_structured_final_gate(
            source_name=source_name,
            refined_feature=source_outputs['refined_feature'],
            reliability_cues=reliability_cues,
        )

        cue_dict = self._build_explicit_cues(
            offsets=source_outputs['offsets'],
            attention_weights=source_outputs['attention_weights'],
            query_feature=source_outputs['query_feature'],
            sampled_features=source_outputs['sampled_features'],
        )
        temporal_cue_distance = self._stack_temporal_cue_distance(cue_dict)
        return (
            source_outputs['refined_feature'],
            final_gate,
            temporal_cue_distance,
            gate_input_maps,
        )

    def forward(
        self,
        feature_vehicle: torch.Tensor,
        feature_rsu: Optional[torch.Tensor] = None,
        feature_drone: Optional[torch.Tensor] = None,
        batch_dict: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Fuse three aligned BEV features.

        Args:
            feature_vehicle: Vehicle BEV feature of shape `[B, C, H, W]`.
            feature_rsu: Optional RSU BEV feature of shape `[B, C, H, W]`.
            feature_drone: Optional drone BEV feature of shape `[B, C, H, W]`.
            batch_dict: Optional metadata dict that provides temporal pose/reset.

        Returns:
            Final fused BEV feature of shape `[B, C, H, W]`.
        """
        if feature_vehicle.dim() != 4:
            raise ValueError(
                "OffsetGuidedSelectiveHierarchicalMambaFusionBlock expects "
                f"`feature_vehicle` to be [B, C, H, W], got {feature_vehicle.shape}"
            )
        if feature_rsu is not None and feature_rsu.shape != feature_vehicle.shape:
            raise ValueError(
                "Non-None external inputs must match vehicle shape, got "
                f"vehicle={feature_vehicle.shape}, rsu={feature_rsu.shape}"
            )
        if feature_drone is not None and feature_drone.shape != feature_vehicle.shape:
            raise ValueError(
                "Non-None external inputs must match vehicle shape, got "
                f"vehicle={feature_vehicle.shape}, drone={feature_drone.shape}"
            )

        vehicle_outputs = self._refine_source('vehicle', feature_vehicle)
        refined_vehicle = vehicle_outputs['refined_feature']
        refined_rsu, g_final_rsu, temporal_cue_rsu, gate_inputs_rsu = self._process_optional_external_source(
            'rsu',
            feature_rsu,
            feature_vehicle,
        )
        refined_drone, g_final_drone, temporal_cue_drone, gate_inputs_drone = self._process_optional_external_source(
            'drone',
            feature_drone,
            feature_vehicle,
        )
        
        current_epoch = batch_dict.get('epoch', None) if batch_dict is not None else None
        g_fusion_rsu = self._apply_gate_topk(g_final_rsu)
        g_fusion_drone = self._apply_gate_topk(g_final_drone)

        if batch_dict is not None:
            batch_dict['fusion_gate_outputs'] = {
                'gate_rsu': g_final_rsu if feature_rsu is not None else None,
                'gate_drone': g_final_drone if feature_drone is not None else None,
            }
            batch_dict['fusion_gate_inputs'] = {
                'gate_rsu': gate_inputs_rsu if feature_rsu is not None else None,
                'gate_drone': gate_inputs_drone if feature_drone is not None else None,
            }
            if self.grad_debug_step % 30 == 0:
                # ===== Gate grad debug =====================
                self._maybe_print_gate_schedule_debug(
                    gate_outputs=batch_dict['fusion_gate_outputs'],
                    epoch=current_epoch,
                )
                # ===============================================
                # ===== Gate map visualization for debugging =====
                if self.visualize_gate_maps:
                    self._visualize_gate_maps(
                        batch_dict=batch_dict,
                        gate_outputs=batch_dict['fusion_gate_outputs'],
                        gate_inputs=batch_dict['fusion_gate_inputs'],
                        gate_fusion_outputs={
                            'gate_rsu': g_fusion_rsu if feature_rsu is not None else None,
                            'gate_drone': g_fusion_drone if feature_drone is not None else None,
                        } if self.gate_topk_enabled else None,
                    )
                # ===============================================
            self.grad_debug_step += 1

        fused_feature = self.ego_fusion(
            refined_veh=refined_vehicle,
            refined_rsu=refined_rsu,
            refined_drone=refined_drone,
            gate_rsu=g_fusion_rsu,
            gate_drone=g_fusion_drone,
        )

        # Read-only visualization export: does not alter fused_feature numerics.
        if batch_dict is not None and bool(batch_dict.get("visualization_debug", False)):
            batch_dict["pre_temporal_feature"] = fused_feature.detach()

        if self.sparse_temporal_fusion is not None:
            relative_pose = None if batch_dict is None else batch_dict.get('ego_pose')
            temporal_reset = False if batch_dict is None else bool(batch_dict.get('temporal_reset', False))
            temporal_out = self.sparse_temporal_fusion(
                current_feature=fused_feature,
                relative_pose=relative_pose,
                temporal_reset=temporal_reset,
                distance_drone=temporal_cue_drone,
                gate_drone=g_final_drone,
                distance_rsu=temporal_cue_rsu,
                gate_rsu=g_final_rsu,
                return_debug=True,
                epoch=current_epoch,
            )
            if isinstance(temporal_out, tuple):
                fused_feature, temporal_debug = temporal_out
            else:
                fused_feature = temporal_out
                temporal_debug = {}

            if batch_dict is not None:
                aux_outputs = batch_dict.get("fusion_aux_outputs", {})
                aux_outputs["need_map"] = temporal_debug.get("need_map", None)
                aux_outputs["window_scores"] = temporal_debug.get("window_scores", None)
                aux_outputs["pixel_mask"] = temporal_debug.get("pixel_mask", None)
                aux_outputs["window_mask"] = temporal_debug.get("window_mask", None)
                aux_outputs["history_ready"] = temporal_debug.get("history_ready", None)
                aux_outputs["temporal_applied"] = temporal_debug.get("temporal_applied", None)
                batch_dict["fusion_aux_outputs"] = aux_outputs

        local_feature = self.local_mamba(fused_feature, return_aux=False)
        global_feature = self.global_fusion(local_feature)
        return global_feature

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ImportanceGenerator(nn.Module):
    """
    鲁棒的BEV融合权重生成器
    支持动态数量的agent输入
    """
    def __init__(self, num_channels=128, max_agents=3, use_softmax=True):
        super().__init__()
        self.num_channels = num_channels
        self.max_agents = max_agents
        self.use_softmax = use_softmax
        
        # 定义agent的embedding（支持最大数量的agent）
        self.agent_emb = nn.Embedding(max_agents, num_channels)  # max_agents个agent，每个C维
        
        # 动态融合网络：根据实际输入数量调整
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(num_channels * max_agents, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, max_agents, 1)   # 输出 (B,max_agents,H,W)
        )  
    def forward(self, bev_features_dict, agent_names=None):
        """
        鲁棒的BEV融合权重生成
        
        Args:
            bev_features_dict: Dict of BEV features, e.g., {'vehicle': Fv, 'drone': Fd, 'rsu': Fr}
            agent_names: List of agent names (optional, inferred from dict keys)
            
        Returns:
            fused_bev: 融合后的BEV特征 (B, C, H, W)
            importance_weights: 权重图 (B, num_agents, H, W)
            active_agents: List of active agent names
        """
        if agent_names is None:
            agent_names = list(bev_features_dict.keys())
        
        num_agents = len(agent_names)
        if num_agents == 0:
            raise ValueError("No agents provided")
        if num_agents > self.max_agents:
            raise ValueError(f"Too many agents: {num_agents} > {self.max_agents}")
        
        # 创建完整的特征张量（包含所有可能的agent）
        full_features = []
        agent_indices = []
        
        # 定义agent到索引的映射
        agent_to_idx = {'vehicle': 0, 'rsu': 1, 'drone': 2}
        
        for i, agent_name in enumerate(agent_names):
            if agent_name in bev_features_dict:
                # 添加身份embedding
                agent_idx = agent_to_idx.get(agent_name, i)
                agent_emb = self.agent_emb(torch.tensor(agent_idx, device=bev_features_dict[agent_name].device))
                agent_emb = agent_emb.view(1, self.num_channels, 1, 1)
                
                feat_with_emb = bev_features_dict[agent_name] + agent_emb
                full_features.append(feat_with_emb)
                agent_indices.append(agent_idx)
            else:
                print(f"Warning: Agent {agent_name} not found in input")
        
        # 如果agent数量不足，用零填充到max_agents
        while len(full_features) < self.max_agents:
            zero_feat = torch.zeros_like(full_features[0])
            full_features.append(zero_feat)
            agent_indices.append(len(full_features) - 1)
        
        # 拼接所有特征
        F_cat = torch.cat(full_features, dim=1)  # (B, max_agents*C, H, W)
        
        # 生成权重
        W = self.fuse_conv(F_cat)  # (B, max_agents, H, W)
        
        # 创建mask，只对有效的agent计算权重
        valid_mask = torch.zeros(self.max_agents, device=W.device)
        for i in range(num_agents):
            valid_mask[i] = 1.0
        
        # 应用mask
        W_masked = W * valid_mask.view(1, -1, 1, 1)
        
        # 归一化权重
        if self.use_softmax:
            W_masked = torch.softmax(W_masked, dim=1)
        else:
            W_masked = torch.sigmoid(W_masked)
            W_masked = W_masked / (W_masked.sum(dim=1, keepdim=True) + 1e-6)
        
        # 只使用有效agent的权重进行融合
        valid_features = full_features[:num_agents]  # 只取有效的特征
        F_stack = torch.stack(valid_features, dim=1)  # (B, num_agents, C, H, W)
        W_valid = W_masked[:, :num_agents, :, :]  # (B, num_agents, H, W)
        F_fused = (W_valid.unsqueeze(2) * F_stack).sum(dim=1)  # (B, C, H, W)
        
        return F_fused, W_masked, agent_names
    
    def forward_with_list(self, bev_features_list, agent_names):
        """
        使用列表输入的forward方法（向后兼容）
        
        Args:
            bev_features_list: List of BEV features
            agent_names: List of agent names
            
        Returns:
            fused_bev: 融合后的BEV特征 (B, C, H, W)
            importance_weights: 权重图 (B, num_agents, H, W)
            active_agents: List of active agent names
        """
        # 转换为字典格式
        bev_features_dict = {}
        for i, agent_name in enumerate(agent_names):
            if i < len(bev_features_list):
                bev_features_dict[agent_name] = bev_features_list[i]
        
        return self.forward(bev_features_dict, agent_names)


class BatchCompressor(nn.Module):
    def __init__(self, in_channels = 80, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),  # 1x1卷积调整通道
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 对每个batch分别处理
        features = []
        for i in range(B):
            feat = self.conv(x[i:i+1])  # [1, C', H, W]
            features.append(feat)
        
        # 堆叠并最大池化
        stacked = torch.cat(features, dim=0)  # [B, C', H, W]
        compressed = torch.max(stacked, dim=0, keepdim=True)[0]  # [1, C', H, W]
        
        return compressed


class BatchCompressorV2(nn.Module):
    """
    空间相关的 BEV 融合模块（支持动态 B）

    输入:
        x: [B, C, H, W]，这里的 B 是要融合的 BEV 数量（agent / 时刻），不固定
    输出:
        fused: [1, out_channels, H, W]

    设计:
        1. 共享特征编码: 对每个 BEV 通过同一个卷积块提取中间特征 feat
        2. 空间相关权重: 用 feat 生成每个 (b, h, w) 的标量 logits，并在 B 维做 softmax 得到 alpha
        3. 加权融合: 在 B 维按 alpha 对 feat 做加权和，得到 fused，再用 1x1 conv 调整通道
    """

    def __init__(self, in_channels: int = 80, mid_channels: int = None, out_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = mid_channels

        # Step 1: 共享特征编码 φ
        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Step 2: 生成空间相关 logits（每个 BEV、每个像素一个 score）
        self.score_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),  # 输出 [B,1,H,W]
        )

        # Step 3: 融合后通道调整
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]，B 不固定
        if x.dim() != 4:
            raise ValueError(f"BatchCompressorV2 expects 4D input [B,C,H,W], got shape {x.shape}")

        B, C, H, W = x.shape
        if B == 1:
            # 只有一个 BEV 时，直接做特征编码和投影，避免数值不稳定的 softmax
            feat_single = self.feat_conv(x)
            return self.out_proj(feat_single)

        # 1) 特征编码
        feat = self.feat_conv(x)  # [B, C_mid, H, W]

        # 2) 空间相关权重: logits -> softmax over B 维
        logits = self.score_conv(feat)              # [B, 1, H, W]
        alpha = torch.softmax(logits, dim=0)        # [B, 1, H, W]，对每个 (h,w) 在 B 上归一化

        # 3) 按权重融合
        fused = (alpha * feat).sum(dim=0, keepdim=True)  # [1, C_mid, H, W]

        # 4) 通道调整
        fused = self.out_proj(fused)  # [1, out_channels, H, W]
        return fused