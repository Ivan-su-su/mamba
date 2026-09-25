"""Ablation variant: sparse temporal fusion without gate-map priors in TemporalNeedHead."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ...vmamba.vmamba import LayerNorm2d
from .sparse_fusion import SparseTemporalFusionBlock


class TemporalNeedHeadNoGate(nn.Module):
    """Predict temporal need maps from current BEV features only (no gate prior)."""

    def __init__(
        self,
        channels: int,
        feature_channels: int = 16,
        residual_channels: int = 16,
        smoothing_kernel_size: int = 1,
        need_init_bias: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.residual_channels = int(residual_channels)
        self.smoothing_kernel_size = int(smoothing_kernel_size)
        self.need_init_bias = float(need_init_bias)
        self.current_feature_encoder = nn.Sequential(
            nn.Conv2d(channels, self.feature_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.feature_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.residual_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.residual_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.residual_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.residual_head[-1].bias, 0.0)

    def forward(
        self,
        current_feature: torch.Tensor,
        distance_drone: Optional[torch.Tensor] = None,
        gate_drone: Optional[torch.Tensor] = None,
        distance_rsu: Optional[torch.Tensor] = None,
        gate_rsu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del distance_drone, gate_drone, distance_rsu, gate_rsu
        if current_feature.dim() != 4:
            raise ValueError(f"current_feature must be [B, C, H, W], got {current_feature.shape}")

        batch_size, _, height, width = current_feature.shape
        device = current_feature.device
        dtype = current_feature.dtype

        current_desc = self.current_feature_encoder(current_feature)
        prior = torch.full(
            (batch_size, 1, height, width),
            self.need_init_bias,
            device=device,
            dtype=dtype,
        )
        prior_logit = torch.log(prior / (1.0 - prior))
        residual_logit = self.residual_head(current_desc)
        need_logits = prior_logit + residual_logit
        need_map = torch.sigmoid(need_logits)

        if self.smoothing_kernel_size > 1:
            kernel = self.smoothing_kernel_size
            padding = kernel // 2
            need_map = F.avg_pool2d(
                need_map,
                kernel_size=kernel,
                stride=1,
                padding=padding,
            )
        need_map = need_map.clamp(0.0, 1.0)
        confidence_map = 1.0 - need_map
        return confidence_map, need_map


class SparseTemporalFusionBlockNoGate(SparseTemporalFusionBlock):
    """Sparse temporal block that routes need purely from current features."""

    def __init__(
        self,
        channels: int,
        num_points: int,
        point_cloud_range: Tuple[float, float, float, float, float, float],
        cue_distance_channels: int = 3,
        window_size: Union[int, Tuple[int, int]] = 4,
        need_topk: Optional[int] = None,
        need_threshold: Optional[float] = 0.5,
        need_smoothing_kernel_size: int = 1,
        need_init_bias: float = 0.25,
        need_feature_channels: int = 16,
        need_residual_channels: int = 16,
        need_prior_clamp_min: float = 0.01,
        need_prior_clamp_max: float = 0.99,
        residual_offset_range: float = 2.0,
        base_scale_multiplier: float = 1.0,
        base_scale_min: float = 1.0,
        base_scale_max: float = 8.0,
        base_motion_epsilon: float = 0.5,
        base_fixed_expansion_no_motion: float = 1.0,
        include_diagonal_base_offsets: bool = False,
        offset_mode: str = 'guided',
        direct_offset_range: Optional[float] = None,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
        return_debug: bool = False,
    ) -> None:
        del cue_distance_channels, need_prior_clamp_min, need_prior_clamp_max
        super().__init__(
            channels=channels,
            num_points=num_points,
            point_cloud_range=point_cloud_range,
            window_size=window_size,
            need_topk=need_topk,
            need_threshold=need_threshold,
            need_smoothing_kernel_size=need_smoothing_kernel_size,
            need_init_bias=need_init_bias,
            need_feature_channels=need_feature_channels,
            need_residual_channels=need_residual_channels,
            residual_offset_range=residual_offset_range,
            base_scale_multiplier=base_scale_multiplier,
            base_scale_min=base_scale_min,
            base_scale_max=base_scale_max,
            base_motion_epsilon=base_motion_epsilon,
            base_fixed_expansion_no_motion=base_fixed_expansion_no_motion,
            include_diagonal_base_offsets=include_diagonal_base_offsets,
            offset_mode=offset_mode,
            direct_offset_range=direct_offset_range,
            align_corners=align_corners,
            padding_mode=padding_mode,
            return_debug=return_debug,
        )
        self.need_head = TemporalNeedHeadNoGate(
            channels=channels,
            feature_channels=need_feature_channels,
            residual_channels=need_residual_channels,
            smoothing_kernel_size=need_smoothing_kernel_size,
            need_init_bias=need_init_bias,
        )
