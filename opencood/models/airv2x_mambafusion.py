from opencood.models.mambafusion_modules.detector3d_template import Detector3DTemplate
from opencood.models.mambafusion_modules.backbones_3d import pfe, vfe
from opencood.models.mambafusion_modules import backbones_3d, mm_backbone, view_transforms, backbones_2d
from opencood.models.mambafusion_modules.backbones_image import img_neck
from opencood.models.mambafusion_modules.backbones_2d import fuser, map_to_bev
from opencood.models.mambafusion_modules.spconv_utils import find_all_spconv_keys
from opencood.models.mambafusion_modules.vmamba import build_vssm_model
import torch.profiler
import torch.nn.functional as F
from easydict import EasyDict

class Airv2xMambafusion(Detector3DTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        self.model_cfg = EasyDict(self.model_cfg)
        self.dataset = dataset
        self.use_voxel_mamba = self.model_cfg.get('USE_VOXEL_MAMBA', False)
        self.new_order = self.model_cfg.VTRANSFORM.get('USE_MAMBA', False)
        self.agent = ['vehicle','drone','rsu']
        
        # 从配置读取BEV尺寸，用于后续的默认值设置
        self.bev_size_H = self.model_cfg.get('BEV_SIZE_H', 200)
        self.bev_size_W = self.model_cfg.get('BEV_SIZE_W', 704)

        if self.use_voxel_mamba:
            if self.new_order:
                self.module_topology = [
                    'vfe', 'backbone_3d', 'mm_backbone', 
                    'neck','vtransform', 'map_to_bev_module', 'fuser',
                    'backbone_2d'
                ]
            else:
                self.module_topology = [
                    'vfe', 'backbone_3d', 'mm_backbone', 'map_to_bev_module', 
                    'neck','vtransform', 'fuser',
                    'backbone_2d','dense_head',
                ]
            
        else:
            if self.new_order:
                self.module_topology = [
                    'vfe','mm_backbone', 
                    'neck','vtransform', 'map_to_bev_module', 'fuser',
                    'backbone_2d','dense_head',
                ]
            else:
                self.module_topology = [
                    'vfe','mm_backbone', 'map_to_bev_module', 
                    'neck','vtransform', 'fuser',
                    'backbone_2d','dense_head',
                ]
        
        num_anchors = self.model_cfg.get('num_anchors', 2)
        num_classes = self.model_cfg.get('num_classes', 7)
        C = 512   # TODO: 从配置文件中读取
        self.cls_head = torch.nn.Conv2d(C, num_anchors * num_classes, kernel_size=1)
        self.reg_head = torch.nn.Conv2d(C, 7 * num_anchors, kernel_size=1)
        self.obj_head = torch.nn.Conv2d(C, num_anchors, kernel_size=1)
        self.module_list = self.build_networks()
        self.time_list = []
        
        # 打印各模块参数量
        self.print_module_params()
    
    def print_module_params(self):
        """打印各模块的参数量"""
        print("=" * 80)
        print("AirV2X MambaFusion 模型参数量分析")
        print("=" * 80)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print()
        
        # 按模块统计参数量
        module_params = {}
        
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    # 提取模块名称（去掉具体层名）
                    module_name = name.split('.')[0] if '.' in name else name
                    if module_name not in module_params:
                        module_params[module_name] = 0
                    module_params[module_name] += param_count
        
        # 按参数量排序
        sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
        
        print("各模块参数量 (按参数量排序):")
        print("-" * 50)
        for module_name, param_count in sorted_modules:
            percentage = (param_count / total_params) * 100
            print(f"{module_name:25s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("模块拓扑结构参数量:")
        print("-" * 50)
        for i, module_name in enumerate(self.module_topology):
            param_count = module_params.get(module_name, 0)
            percentage = (param_count / total_params) * 100
            print(f"{i+1:2d}. {module_name:20s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("检测头参数量:")
        print("-" * 50)
        cls_params = sum(p.numel() for p in self.cls_head.parameters())
        reg_params = sum(p.numel() for p in self.reg_head.parameters())
        obj_params = sum(p.numel() for p in self.obj_head.parameters())
        head_total = cls_params + reg_params + obj_params
        
        print(f"cls_head: {cls_params:10,}")
        print(f"reg_head: {reg_params:10,}")
        print(f"obj_head: {obj_params:10,}")
        print(f"检测头总计: {head_total:10,} ({(head_total/total_params)*100:5.1f}%)")
        
        print("=" * 80)
    

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            # TODO: 这些不需要从数据集中读，可以从配置文件中读取
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,   #5
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,   #5
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict


    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_mm_backbone(self, model_info_dict):
        if self.model_cfg.get('MM_BACKBONE', None) is None:
            return None, model_info_dict
        mm_backbone_name = self.model_cfg.MM_BACKBONE.NAME
        del self.model_cfg.MM_BACKBONE['NAME']
        mm_backbone_module = mm_backbone.__all__[mm_backbone_name](
            model_cfg=self.model_cfg.MM_BACKBONE
            )
        model_info_dict['module_list'].append(mm_backbone_module)

        return mm_backbone_module, model_info_dict

    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features', None)
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            # adapt pretrain image backbone to mm backbone
            if 'image_backbone' in key:
                key = key.replace("image","mm")
                if 'input_layer' in key:
                    key = key.replace("input_layer","image_input_layer")


            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print("not exist",key)
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state
    
    def pre_process(self, agent_idx, available_agents, batch_dict):
        for agent in available_agents:
            camera_num = batch_dict[agent]['batch_merged_cam_inputs']['imgs'].shape[1]
            agent_to_ego_transform = batch_dict['img_pairwise_t_matrix_collab'][0,agent_idx[agent]:agent_idx[agent]+batch_dict[agent]['record_len'],0,:,:]
            agent_to_ego_transform = agent_to_ego_transform.unsqueeze(0)
            agent_to_ego_transform = agent_to_ego_transform.repeat_interleave(camera_num, dim=1)
            batch_dict[agent]['agent_to_ego_transform'] = agent_to_ego_transform
        return batch_dict
    
    def _check_agent_shapes(self, agent_dict: dict, agent: str, module_name: str):
        """检查agent数据的shape"""
        issues = []
        
        # 检查voxel数据
        if 'voxel_features' in agent_dict and 'voxel_coords' in agent_dict:
            voxel_features = agent_dict['voxel_features']
            voxel_coords = agent_dict['voxel_coords']
            if isinstance(voxel_features, torch.Tensor) and isinstance(voxel_coords, torch.Tensor):
                if voxel_features.shape[0] != voxel_coords.shape[0]:
                    issues.append(f"[{module_name}] {agent}: voxel_features数量 {voxel_features.shape[0]} != voxel_coords数量 {voxel_coords.shape[0]}")
                if len(voxel_coords.shape) != 2 or voxel_coords.shape[1] != 4:
                    issues.append(f"[{module_name}] {agent}: voxel_coords shape应为[N, 4]，实际为{voxel_coords.shape}")
        
        # 检查BEV特征
        if 'spatial_features' in agent_dict:
            spatial_features = agent_dict['spatial_features']
            if isinstance(spatial_features, torch.Tensor):
                if len(spatial_features.shape) != 4:
                    issues.append(f"[{module_name}] {agent}: spatial_features应为4D [B,C,H,W]，实际为{spatial_features.shape}")
                else:
                    # 检查BEV尺寸是否匹配配置
                    H, W = spatial_features.shape[2:]
                    expected_H = self.bev_size_H // 2  # 考虑stride=2
                    expected_W = self.bev_size_W // 2
                    if H != expected_H or W != expected_W:
                        issues.append(f"[{module_name}] {agent}: spatial_features BEV尺寸 ({H}, {W}) 与预期 ({expected_H}, {expected_W}) 不匹配")
        
        if issues:
            print(f"⚠️ Shape检查警告 [{module_name}] {agent}:")
            for issue in issues:
                print(f"    {issue}")
    
    def _check_output_shapes(self, agent_dict: dict, agent: str, module_name: str):
        """检查模块输出shape"""
        # 可以在这里添加输出shape检查
        pass
    
    def forward(self, batch_dict): 
        available_agents = []
        count = 0
        agent_idx = {}
        for agent in self.agent:
            if agent == 'vehicle' and 'origin_lidar' in batch_dict:
                # 检查vehicle的origin_lidar是否有效
                origin_lidar = batch_dict['origin_lidar']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
            elif agent != 'vehicle' and f'origin_lidar_{agent}' in batch_dict:
                # 检查RSU/Drone的origin_lidar是否有效
                origin_lidar = batch_dict[f'origin_lidar_{agent}']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
            agent_idx[agent] = count
            count += batch_dict[agent]['record_len'].item()
        
        # AirV2X需要agent循环处理，但要保持数据流一致
        print("available_agents:",available_agents)
        batch_dict = self.pre_process(agent_idx, available_agents, batch_dict)
        
        for cur_module, model_name in zip(self.module_list, self.module_topology):
            print("model_name:",model_name)
            if model_name in ['vfe', 'backbone_3d', 'mm_backbone', 'map_to_bev_module','vtransform']:
                # 这些模块需要agent参数，但只处理有效的agent
                for agent in available_agents:
                    batch_dict = cur_module(batch_dict, agent)
            elif model_name in ['fuser']:
                batch_dict = cur_module(batch_dict, available_agents)
            else:
                # 其他模块不需要agent参数
                batch_dict = cur_module(batch_dict)

        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
            # 输出BEV特征格式，让AirV2X的post_process处理
            # 获取fuser输出的BEV特征
        spatial_features = batch_dict.get('spatial_features_2d', None)  # [B, C, H, W]
        
        if spatial_features is not None:
            # 通过头部网络得到最终-output
            psm = self.cls_head(spatial_features)  # [B, A*C, H, W] = [B, 2*7, H, W]
            rm = self.reg_head(spatial_features)   # [B, A*7, H, W] = [B, 2*7, H, W]
            obj = self.obj_head(spatial_features)  # [B, A, H, W] = [B, 2, H, W]
            
        else:
            # 如果没有特征，创建空的特征
            # 使用配置中的BEV尺寸（考虑feature_stride=2，所以实际是 H/2, W/2）
            default_H, default_W = self.bev_size_H // 2, self.bev_size_W // 2
            psm = torch.zeros(1, 14, default_H, default_W, device=next(self.parameters()).device)  # 2*7=14
            rm = torch.zeros(1, 14, default_H, default_W, device=next(self.parameters()).device)   # 2*7=14
            obj = torch.zeros(1, 2, default_H, default_W, device=next(self.parameters()).device)   # 

        # 创建AirV2X兼容的输出格式
        output_dict = {
            'psm': psm,                  # [1, A*C, H, W] - 分类特征
            'rm': rm,                    # [1, A*7, H, W] - 回归特征  
            'obj': obj,                  # [1, A, H, W] - 目标特征
            'mask': 0,                   # 占位符
            'com': None,                 # 通信相关
            'comm_rate': None            # 通信率
        }
        if 'fusion_aux_outputs' in batch_dict:
            output_dict['fusion_aux_outputs'] = batch_dict['fusion_aux_outputs']
        if 'fusion_gate_outputs' in batch_dict:
            output_dict['fusion_gate_outputs'] = batch_dict['fusion_gate_outputs']
        if 'epoch' in batch_dict:
            output_dict['epoch'] = batch_dict['epoch']

        return output_dict

    def get_training_loss(self, batch_dict):
        """
        使用MambaFusion标准的loss计算方法
        """
        disp_dict = {}
        
        # 使用dense_head的loss计算方法
        if hasattr(self, 'dense_head') and self.dense_head is not None:
            loss, tb_dict = self.dense_head.get_loss()
        else:
            # 如果没有dense_head，使用默认的loss计算
            loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
            tb_dict = {
                'loss_trans': loss_trans.item(),
                **tb_dict
            }
            loss = loss_trans
            
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            # 简单退化检查：打印每帧 3D 检测框数量
            # 注意：如需关闭，只需注释或删除下一行
            print(f"[Airv2xMambafusion] batch_index={index}, num_pred_boxes={pred_boxes.shape[0]}")

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
    
