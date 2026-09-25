"""Ablation ConvFuser: no spatial gate in ego fusion + no gate prior in sparse temporal."""

from typing import Any, Dict, Tuple, Union

import torch
from torch import nn

from ..sparse_temporal.sparse_fusion_no_gate import SparseTemporalFusionBlockNoGate
from .convfuser_mamba import (
    ConvFuser,
    EgoFusionLayer,
    OffsetGuidedSelectiveHierarchicalMambaFusionBlock,
)


class EgoFusionLayerNoGate(EgoFusionLayer):
    """Fuse ego, RSU, and drone features without spatial gate weighting."""

    def forward(
        self,
        refined_veh: torch.Tensor,
        refined_rsu: torch.Tensor,
        refined_drone: torch.Tensor,
        gate_rsu: torch.Tensor,
        gate_drone: torch.Tensor,
    ) -> torch.Tensor:
        del gate_rsu, gate_drone
        fused_input = torch.cat(
            [refined_veh, refined_rsu, refined_drone],
            dim=1,
        )
        return self.proj(fused_input)


def _build_sparse_temporal_fusion_no_gate(
    channels: int,
    sparse_temporal_cfg: Dict[str, Any],
    num_points: int,
    offset_range: float,
    window_size: Union[int, Tuple[int, int]],
    align_corners: bool,
    padding_mode: str,
    cue_distance_channels: int,
) -> SparseTemporalFusionBlockNoGate:
    point_cloud_range = sparse_temporal_cfg.get('POINT_CLOUD_RANGE')
    if point_cloud_range is None:
        raise ValueError(
            "SPARSE_TEMPORAL.ENABLE=True requires SPARSE_TEMPORAL.POINT_CLOUD_RANGE"
        )
    return SparseTemporalFusionBlockNoGate(
        channels=channels,
        num_points=sparse_temporal_cfg.get('K_SAMPLES', num_points),
        cue_distance_channels=cue_distance_channels,
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
        include_diagonal_base_offsets=sparse_temporal_cfg.get(
            'INCLUDE_DIAGONAL_BASE_OFFSETS', False
        ),
        offset_mode=sparse_temporal_cfg.get('OFFSET_MODE', 'guided'),
        direct_offset_range=sparse_temporal_cfg.get('DIRECT_OFFSET_RANGE'),
        align_corners=align_corners,
        padding_mode=padding_mode,
        return_debug=False,
    )


class OffsetGuidedSelectiveHierarchicalMambaFusionBlockAblation(
    OffsetGuidedSelectiveHierarchicalMambaFusionBlock
):
    """Hierarchical fusion ablation: ego fusion without gate, temporal need without gate prior."""

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
        gate_head_cfg=None,
        sparse_temporal_cfg=None,
        gate_alpha_cfg=None,
    ) -> None:
        super().__init__(
            channels=channels,
            num_points=num_points,
            offset_range=offset_range,
            transmission_alpha=transmission_alpha,
            window_size=window_size,
            align_corners=align_corners,
            padding_mode=padding_mode,
            local_mamba_depth=local_mamba_depth,
            global_fusion_type=global_fusion_type,
            global_mamba_depth=global_mamba_depth,
            global_conv_depth=global_conv_depth,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            forward_type=forward_type,
            sample_da_ffn_ratio=sample_da_ffn_ratio,
            sample_da_drop_rate=sample_da_drop_rate,
            sample_da_layer_scale_init=sample_da_layer_scale_init,
            final_gate_hidden_dim=final_gate_hidden_dim,
            gate_head_cfg=gate_head_cfg,
            sparse_temporal_cfg=sparse_temporal_cfg,
            gate_alpha_cfg=gate_alpha_cfg,
        )
        self.ego_fusion = EgoFusionLayerNoGate(channels)
        sparse_temporal_cfg = sparse_temporal_cfg or {}
        if self.use_sparse_temporal:
            self.sparse_temporal_fusion = _build_sparse_temporal_fusion_no_gate(
                channels=channels,
                sparse_temporal_cfg=sparse_temporal_cfg,
                num_points=num_points,
                offset_range=offset_range,
                window_size=window_size,
                align_corners=align_corners,
                padding_mode=padding_mode,
                cue_distance_channels=self.temporal_cue_distance_channels,
            )


class ConvFuserAblation(ConvFuser):
    """ConvFuser with spatial/temporal gate ablations for controlled experiments."""

    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg)
        if not self.use_offset_guided_hierarchical_fusion:
            return

        fusion_cfg = model_cfg.get('OFFSET_GUIDED_HIERARCHICAL_FUSION', {})
        sparse_temporal_cfg = model_cfg.get('SPARSE_TEMPORAL', {})
        out_channel = model_cfg.OUT_CHANNEL
        self.offset_guided_hierarchical_fusion = OffsetGuidedSelectiveHierarchicalMambaFusionBlockAblation(
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
            gate_alpha_cfg=fusion_cfg.get('GATE_ALPHA', {}),
        )
