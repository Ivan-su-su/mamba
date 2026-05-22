import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ...vmamba.vmamba import LayerNorm2d


def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert an int or pair-like value into a validated 2-tuple."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-element window size, got {value}")
        return int(value[0]), int(value[1])
    scalar = int(value)
    return scalar, scalar


class TemporalNeedHead(nn.Module):
    """Predict temporal need maps directly from fused BEV plus fusion cues."""

    def __init__(
        self,
        channels: int,
        cue_distance_channels: int = 3,
        hidden_channels: Optional[int] = None,
        smoothing_kernel_size: int = 3,
        need_init_bias: float = 0.25,
    ) -> None:
        """Initialize the temporal need prediction head.

        Args:
            channels: BEV feature channel count.
            cue_distance_channels: Channel count of each source `distance_*` tensor
                (e.g. stacked ``sum(d_k), sum(s_k), sum(ws_k)`` maps).
            hidden_channels: Optional hidden width for the prediction head.
            smoothing_kernel_size: Average smoothing kernel for need maps.
            need_init_bias: Initial bias for the final need-logit layer.
        """
        super().__init__()
        hidden_dim = hidden_channels if hidden_channels is not None else channels
        self.cue_distance_channels = int(cue_distance_channels)
        self.smoothing_kernel_size = int(smoothing_kernel_size)
        aux_channels = 2 * (self.cue_distance_channels + 1) + 2
        self.need_head = nn.Sequential(
            nn.Conv2d(channels + aux_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.need_head[-1].bias, float(need_init_bias))

    def _normalize_cue_distance(
        self,
        value: Optional[torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Pad or trim multi-channel distance/cue maps to a fixed channel width."""
        expected = self.cue_distance_channels
        if value is None:
            return torch.zeros(batch_size, expected, height, width, device=device, dtype=dtype)
        if value.dim() != 4:
            raise ValueError(f"distance/cue maps must be [B, C, H, W], got {value.shape}")
        if value.shape[0] != batch_size or value.shape[2:] != (height, width):
            raise ValueError(
                "distance/cue spatial/batch shape must match current feature, "
                f"got value={value.shape}, feature={[batch_size, height, width]}"
            )
        if value.shape[1] == expected:
            return value.to(device=device, dtype=dtype)
        if value.shape[1] > expected:
            return value[:, :expected].to(device=device, dtype=dtype)
        pad_channels = expected - value.shape[1]
        pad = torch.zeros(batch_size, pad_channels, height, width, device=device, dtype=dtype)
        return torch.cat([value.to(device=device, dtype=dtype), pad], dim=1)

    def _normalize_single_map(
        self,
        value: Optional[torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize one optional aux map into `[B, 1, H, W]`."""
        if value is None:
            return torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
        if value.dim() != 4 or value.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, H, W], got {value.shape}")
        if value.shape[0] != batch_size or value.shape[2:] != (height, width):
            raise ValueError(
                "Aux map spatial/batch shape must match current feature, "
                f"got aux={value.shape}, feature={[batch_size, height, width]}"
            )
        return value.to(device=device, dtype=dtype)

    def _build_source_aux(
        self,
        distance: Optional[torch.Tensor],
        gate: Optional[torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack one source's aux tensors into a fixed channel block."""
        normalized_distance = self._normalize_cue_distance(
            distance, batch_size, height, width, device, dtype
        )
        normalized_gate = self._normalize_single_map(gate, batch_size, height, width, device, dtype)
        return torch.cat([normalized_distance, normalized_gate], dim=1), normalized_gate

    def forward(
        self,
        current_feature: torch.Tensor,
        distance_drone: Optional[torch.Tensor] = None,
        gate_drone: Optional[torch.Tensor] = None,
        distance_rsu: Optional[torch.Tensor] = None,
        gate_rsu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict current-frame temporal need directly.

        Args:
            current_feature: Current fused BEV feature of shape `[B, C, H, W]`.
            distance_drone: Drone stacked cue maps `[B, C_cue, H, W]` (e.g. sum of
                ``d_k``, ``s_k``, ``ws_k`` along the point axis).
            gate_drone: Drone final fusion gate `[B, 1, H, W]`.
            distance_rsu: RSU stacked cue maps `[B, C_cue, H, W]`.
            gate_rsu: RSU final fusion gate `[B, 1, H, W]`.

        Returns:
            A tuple containing:
                - confidence map `[B, 1, H, W]`
                - temporal need map `[B, 1, H, W]`
        """
        if current_feature.dim() != 4:
            raise ValueError(f"current_feature must be [B, C, H, W], got {current_feature.shape}")
        batch_size, _, height, width = current_feature.shape
        device = current_feature.device
        dtype = current_feature.dtype
        aux_drone, normalized_gate_drone = self._build_source_aux(
            distance_drone,
            gate_drone,
            batch_size,
            height,
            width,
            device,
            dtype,
        )
        aux_rsu, normalized_gate_rsu = self._build_source_aux(
            distance_rsu,
            gate_rsu,
            batch_size,
            height,
            width,
            device,
            dtype,
        )
        gate_max = torch.maximum(normalized_gate_drone, normalized_gate_rsu)
        gate_weak = 1.0 - gate_max
        need_input = torch.cat(
            [current_feature, aux_drone, aux_rsu, gate_max, gate_weak],
            dim=1,
        )
        need_logits = self.need_head(need_input)
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


class WindowRouter(nn.Module):
    """Route temporal compute to high-need windows."""

    def __init__(
        self,
        window_size: Union[int, Tuple[int, int]],
        topk: Optional[int] = None,
        need_threshold: Optional[float] = 0.5,
    ) -> None:
        """Initialize the window router.

        Args:
            window_size: Pooling window size.
            topk: Optional cap on windows per sample after threshold filtering.
            need_threshold: Mean-need threshold per window; also pre-filters before ``topk``.
        """
        super().__init__()
        self.window_size = _to_2tuple(window_size)
        self.topk = None if topk is None else int(topk)
        self.need_threshold = None if need_threshold is None else float(need_threshold)

    def _pad_to_window(self, need_map: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Pad a need map so it can be evenly partitioned into windows."""
        batch_size, _, height, width = need_map.shape
        window_height, window_width = self.window_size
        pad_height = (window_height - (height % window_height)) % window_height
        pad_width = (window_width - (width % window_width)) % window_width
        padded = need_map
        if pad_height > 0 or pad_width > 0:
            padded = F.pad(need_map, (0, pad_width, 0, pad_height), mode='constant', value=0.0)
        meta = {
            'batch_size': batch_size,
            'height': height,
            'width': width,
            'padded_height': height + pad_height,
            'padded_width': width + pad_width,
        }
        return padded, meta

    def forward(self, need_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a dense need map into window-level and pixel-level masks.

        Args:
            need_map: Dense temporal need map `[B, 1, H, W]`.

        Returns:
            A tuple containing:
                - pixel mask `[B, 1, H, W]`
                - window mask `[B, 1, Wh, Ww]`
                - window scores `[B, 1, Wh, Ww]`
        """
        if need_map.dim() != 4 or need_map.shape[1] != 1:
            raise ValueError(f"need_map must be [B, 1, H, W], got {need_map.shape}")
        padded_need, meta = self._pad_to_window(need_map)
        window_height, window_width = self.window_size
        batch_size, _, padded_height, padded_width = padded_need.shape
        num_windows_h = padded_height // window_height
        num_windows_w = padded_width // window_width

        windows = padded_need.reshape(
            batch_size,
            1,
            num_windows_h,
            window_height,
            num_windows_w,
            window_width,
        )
        window_scores = windows.mean(dim=(3, 5))
        flat_scores = window_scores.flatten(start_dim=2)

        threshold = 0.5 if self.need_threshold is None else self.need_threshold
        pass_threshold = flat_scores >= threshold
        if self.topk is not None and self.topk > 0:
            num_windows = flat_scores.shape[-1]
            k = min(self.topk, num_windows)
            scores_rank = flat_scores.squeeze(1).masked_fill(
                ~pass_threshold.squeeze(1), float("-inf")
            )
            vals, topk_indices = torch.topk(scores_rank, k=k, dim=1)
            valid_pick = torch.isfinite(vals)
            weights = valid_pick.to(dtype=need_map.dtype)
            flat_mask = torch.zeros_like(flat_scores)
            flat_mask.scatter_(2, topk_indices.unsqueeze(1), weights.unsqueeze(1))
        else:
            flat_mask = pass_threshold.to(dtype=need_map.dtype)

        window_mask = flat_mask.reshape(batch_size, 1, num_windows_h, num_windows_w)
        expanded_mask = (
            window_mask.repeat_interleave(window_height, dim=2)
            .repeat_interleave(window_width, dim=3)
            .to(dtype=need_map.dtype)
        )
        pixel_mask = expanded_mask[:, :, : meta['height'], : meta['width']]
        return pixel_mask, window_mask, window_scores


class BaseOffsetGenerator(nn.Module):
    """Generate pose-driven directional base offsets for historical retrieval."""

    def __init__(
        self,
        point_cloud_range: Tuple[float, float, float, float, float, float],
        scale_multiplier: float = 1.0,
        min_expansion: float = 1.0,
        max_expansion: float = 8.0,
        motion_epsilon: float = 0.5,
        fixed_expansion_no_motion: float = 1.0,
        include_diagonal_offsets: bool = False,
    ) -> None:
        """Initialize the base offset generator.

        Args:
            point_cloud_range: BEV range `[x_min, y_min, z_min, x_max, y_max, z_max]`.
            scale_multiplier: Global multiplier applied to motion-derived expansion scale.
            min_expansion: Minimum expansion radius in pixel units when using motion-based scale.
            max_expansion: Maximum expansion radius in pixel units (applies to both branches).
            motion_epsilon: If planar translation norm in pixels is below this, use
                `fixed_expansion_no_motion` instead of motion-based scale.
            fixed_expansion_no_motion: Fixed expansion in pixels when ego motion is below epsilon
                (covers oncoming / cross traffic while ego is nearly static).
            include_diagonal_offsets: Whether to add diagonal directional hypotheses.
        """
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"point_cloud_range must contain 6 values, got {point_cloud_range}"
            )
        self.scale_multiplier = float(scale_multiplier)
        self.min_expansion = float(min_expansion)
        self.max_expansion = float(max_expansion)
        self.motion_epsilon = float(motion_epsilon)
        self.fixed_expansion_no_motion = float(fixed_expansion_no_motion)
        self.include_diagonal_offsets = bool(include_diagonal_offsets)
        self.num_base_offsets = 9 if self.include_diagonal_offsets else 5
        self.register_buffer(
            'point_cloud_range',
            torch.tensor(point_cloud_range, dtype=torch.float32),
            persistent=False,
        )

    def _build_direction_bank(
        self,
        current_to_previous: torch.Tensor,
        voxel_x: torch.Tensor,
        voxel_y: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build directional unit vectors aligned with ego motion in BEV pixel space."""
        translation_x = current_to_previous[:, 0, 3] / voxel_x
        translation_y = current_to_previous[:, 1, 3] / voxel_y
        translation = torch.stack([translation_x, translation_y], dim=1)
        translation_norm = torch.norm(translation, dim=1)

        fallback_direction = current_to_previous[:, :2, 0]
        fallback_norm = torch.norm(fallback_direction, dim=1, keepdim=True).clamp_min(1e-6)
        fallback_direction = fallback_direction / fallback_norm

        normalized_translation = translation / translation_norm.unsqueeze(1).clamp_min(1e-6)
        has_motion = translation_norm >= self.motion_epsilon
        motion_direction = torch.where(
            has_motion.unsqueeze(1),
            normalized_translation,
            fallback_direction.to(dtype=dtype),
        )
        perpendicular_direction = torch.stack(
            [-motion_direction[:, 1], motion_direction[:, 0]],
            dim=1,
        )

        base_directions = [
            torch.zeros_like(motion_direction),
            motion_direction,
            -motion_direction,
            perpendicular_direction,
            -perpendicular_direction,
        ]
        if self.include_diagonal_offsets:
            diagonal_1 = F.normalize(motion_direction + perpendicular_direction, dim=1)
            diagonal_2 = F.normalize(motion_direction - perpendicular_direction, dim=1)
            diagonal_3 = F.normalize(-motion_direction + perpendicular_direction, dim=1)
            diagonal_4 = F.normalize(-motion_direction - perpendicular_direction, dim=1)
            base_directions.extend([diagonal_1, diagonal_2, diagonal_3, diagonal_4])

        direction_bank = torch.stack(base_directions, dim=1)
        return direction_bank.to(dtype=dtype), translation_norm.to(dtype=dtype)

    def forward(
        self,
        relative_pose: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert relative pose into directional base offsets on the historical BEV.

        Args:
            relative_pose: Relative transform `[B, 4, 4]`. This implementation
                assumes the transform maps previous ego coordinates into current
                ego coordinates, so it inverts the transform once to retrieve
                history-aligned sampling coordinates.
            height: Current BEV height.
            width: Current BEV width.
            device: Output device.
            dtype: Output dtype.

        Returns:
            A tuple containing:
                - directional base offset bank `[B, N, 2, H, W]`
                - rigid offset field `[B, 2, H, W]`
                - clamped expansion scale `[B]`
                - direction bank `[B, N, 2]`
        """
        if relative_pose.dim() != 3 or relative_pose.shape[1:] != (4, 4):
            raise ValueError(f"relative_pose must be [B, 4, 4], got {relative_pose.shape}")

        batch_size = relative_pose.shape[0]
        pose = relative_pose.to(device=device, dtype=dtype)
        current_to_previous = torch.linalg.inv(pose)

        pc_range = self.point_cloud_range.to(device=device, dtype=dtype)
        x_min, y_min, _, x_max, y_max, _ = pc_range.unbind()
        voxel_x = (x_max - x_min) / float(width)
        voxel_y = (y_max - y_min) / float(height)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij',
        )
        x_current = x_min + (xx + 0.5) * voxel_x
        y_current = y_min + (yy + 0.5) * voxel_y

        zeros = torch.zeros_like(x_current)
        ones = torch.ones_like(x_current)
        homogeneous = torch.stack([x_current, y_current, zeros, ones], dim=-1)
        homogeneous = homogeneous.reshape(1, height * width, 4).expand(batch_size, -1, -1)

        previous_coords = torch.bmm(homogeneous, current_to_previous.transpose(1, 2))
        previous_coords = previous_coords.reshape(batch_size, height, width, 4)
        x_previous = previous_coords[..., 0]
        y_previous = previous_coords[..., 1]

        previous_x_index = (x_previous - x_min) / voxel_x - 0.5
        previous_y_index = (y_previous - y_min) / voxel_y - 0.5

        offset_x = previous_x_index - xx.unsqueeze(0)
        offset_y = previous_y_index - yy.unsqueeze(0)
        rigid_offset = torch.stack([offset_x, offset_y], dim=1)

        direction_bank, translation_norm = self._build_direction_bank(
            current_to_previous=current_to_previous,
            voxel_x=voxel_x,
            voxel_y=voxel_y,
            dtype=dtype,
        )
        has_motion = translation_norm >= self.motion_epsilon
        motion_expansion = (translation_norm * self.scale_multiplier).clamp(
            min=self.min_expansion,
            max=self.max_expansion,
        )
        fixed_expansion = torch.full_like(
            translation_norm,
            self.fixed_expansion_no_motion,
            dtype=dtype,
        ).clamp(min=0.0, max=self.max_expansion)
        expansion_scale = torch.where(has_motion, motion_expansion, fixed_expansion)
        base_offset_bank = rigid_offset.unsqueeze(1) + (
            expansion_scale[:, None, None, None, None] * direction_bank[:, :, :, None, None]
        )
        return base_offset_bank, rigid_offset, expansion_scale, direction_bank


class ResidualOffsetPredictor(nn.Module):
    """Predict semantic offsets from current-query semantics only."""

    def __init__(
        self,
        channels: int,
        num_points: int,
        offset_range: float,
        hidden_channels: Optional[int] = None,
    ) -> None:
        """Initialize the residual offset predictor.

        Args:
            channels: BEV channel count.
            num_points: Number of sparse sampling points.
            offset_range: Residual offset range in pixel units.
            hidden_channels: Optional hidden width.
        """
        super().__init__()
        hidden_dim = hidden_channels if hidden_channels is not None else channels
        self.num_points = int(num_points)
        self.offset_range = float(offset_range)
        pose_channels = 6
        self.offset_head = nn.Sequential(
            nn.Conv2d(channels + 1 + pose_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.num_points * 2, kernel_size=1, bias=True),
        )

    def _pose_embedding(
        self,
        relative_pose: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Extract compact planar pose features and broadcast them spatially."""
        pose = relative_pose.to(device=device, dtype=dtype)
        tx = pose[:, 0, 3]
        ty = pose[:, 1, 3]
        rot00 = pose[:, 0, 0]
        rot01 = pose[:, 0, 1]
        rot10 = pose[:, 1, 0]
        rot11 = pose[:, 1, 1]
        pose_features = torch.stack([tx, ty, rot00, rot01, rot10, rot11], dim=1)
        return pose_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

    def forward(
        self,
        current_feature: torch.Tensor,
        need_map: torch.Tensor,
        relative_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Predict semantic offsets from current-query semantics.

        Args:
            current_feature: Current fused BEV `[B, C, H, W]`.
            need_map: Temporal need `[B, 1, H, W]`.
            relative_pose: Relative transform `[B, 4, 4]`.

        Returns:
            Semantic offsets `[B, K, 2, H, W]`.
        """
        batch_size, _, height, width = current_feature.shape
        if need_map.shape != (batch_size, 1, height, width):
            raise ValueError(
                f"need_map must be [B, 1, H, W], got {need_map.shape}"
            )
        pose_embedding = self._pose_embedding(
            relative_pose,
            height=height,
            width=width,
            device=current_feature.device,
            dtype=current_feature.dtype,
        )
        predictor_input = torch.cat(
            [current_feature, need_map.to(dtype=current_feature.dtype), pose_embedding],
            dim=1,
        )
        raw_offsets = self.offset_head(predictor_input)
        semantic_offsets = raw_offsets.reshape(batch_size, self.num_points, 2, height, width)
        return torch.tanh(semantic_offsets) * self.offset_range


class TemporalSamplerAggregator(nn.Module):
    """Sample history sparsely and aggregate it with standard attention."""

    def __init__(
        self,
        channels: int,
        num_points: int,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
    ) -> None:
        """Initialize the temporal sampler and aggregation layers.

        Args:
            channels: BEV feature channel count.
            num_points: Number of sparse temporal samples per query.
            align_corners: `grid_sample` alignment mode.
            padding_mode: `grid_sample` padding mode.
        """
        super().__init__()
        self.num_points = int(num_points)
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

    def _normalize_coordinate(self, coord: torch.Tensor, size: int) -> torch.Tensor:
        """Normalize pixel coordinates for `grid_sample`."""
        if self.align_corners:
            if size <= 1:
                return torch.zeros_like(coord)
            return (2.0 * coord / float(size - 1)) - 1.0
        return ((2.0 * coord + 1.0) / float(size)) - 1.0

    def sample(
        self,
        feature: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Sample sparse points from a BEV feature map.

        Args:
            feature: Historical BEV feature `[B, C, H, W]`.
            offsets: Sampling offsets `[B, K, 2, H, W]`.

        Returns:
            Sampled historical tokens `[B, K, C, H, W]`.
        """
        if feature.dim() != 4:
            raise ValueError(f"feature must be [B, C, H, W], got {feature.shape}")
        if offsets.dim() != 5:
            raise ValueError(f"offsets must be [B, K, 2, H, W], got {offsets.shape}")
        batch_size, _, height, width = feature.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=feature.dtype),
            torch.arange(width, device=feature.device, dtype=feature.dtype),
            indexing='ij',
        )
        base_x = xx.unsqueeze(0).expand(batch_size, -1, -1)
        base_y = yy.unsqueeze(0).expand(batch_size, -1, -1)
        sampled_features = []
        for point_idx in range(offsets.shape[1]):
            offset_x = offsets[:, point_idx, 0]
            offset_y = offsets[:, point_idx, 1]
            sample_x = base_x + offset_x
            sample_y = base_y + offset_y
            grid_x = self._normalize_coordinate(sample_x, width)
            grid_y = self._normalize_coordinate(sample_y, height)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            sampled = F.grid_sample(
                feature,
                grid,
                mode='bilinear',
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            sampled_features.append(sampled)
        return torch.stack(sampled_features, dim=1)

    def forward(
        self,
        current_feature: torch.Tensor,
        history_feature: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample historical evidence and aggregate it into one residual feature.

        Args:
            current_feature: Current fused BEV `[B, C, H, W]`.
            history_feature: Historical fused BEV `[B, C, H, W]`.
            offsets: Final sampling offsets `[B, K, 2, H, W]`.

        Returns:
            A tuple containing:
                - temporal residual `[B, C, H, W]`
                - attention weights `[B, K, H, W]`
                - sampled historical tokens `[B, K, C, H, W]`
        """
        sampled_history = self.sample(history_feature, offsets)
        batch_size, num_points, channels, height, width = sampled_history.shape
        query = self.query_proj(current_feature).unsqueeze(1)
        flat_sampled_history = sampled_history.reshape(batch_size * num_points, channels, height, width)
        key = self.key_proj(flat_sampled_history).reshape(batch_size, num_points, channels, height, width)
        value = self.value_proj(flat_sampled_history).reshape(batch_size, num_points, channels, height, width)
        attention_logits = (
            (query * key).sum(dim=2) / math.sqrt(float(channels))
        )
        attention_weights = torch.softmax(attention_logits, dim=1)
        aggregated_history = (attention_weights.unsqueeze(2) * value).sum(dim=1)
        temporal_residual = self.output_proj(torch.cat([current_feature, aggregated_history], dim=1))
        return temporal_residual, attention_weights, sampled_history


class SparseTemporalFusionBlock(nn.Module):
    """Sparse temporal refinement on fused BEV before local-global fusion."""

    def __init__(
        self,
        channels: int,
        num_points: int,
        point_cloud_range: Tuple[float, float, float, float, float, float],
        cue_distance_channels: int = 3,
        window_size: Union[int, Tuple[int, int]] = 4,
        need_topk: Optional[int] = None,
        need_threshold: Optional[float] = 0.5,
        need_smoothing_kernel_size: int = 3,
        need_init_bias: float = 0.25,
        residual_offset_range: float = 2.0,
        base_scale_multiplier: float = 1.0,
        base_scale_min: float = 1.0,
        base_scale_max: float = 8.0,
        base_motion_epsilon: float = 0.5,
        base_fixed_expansion_no_motion: float = 1.0,
        include_diagonal_base_offsets: bool = False,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
        return_debug: bool = False,
    ) -> None:
        """Initialize the sparse temporal fusion block.

        Args:
            channels: BEV feature channel count.
            num_points: Number of sparse temporal retrieval points.
            cue_distance_channels: Channel count of each `distance_*` map passed to
                `TemporalNeedHead` (stacked spatial-fusion cues per source).
            point_cloud_range: BEV range `[x_min, y_min, z_min, x_max, y_max, z_max]`.
            window_size: Need routing window size.
            need_topk: Optional number of routed windows.
            need_threshold: Optional routed-window threshold.
            need_smoothing_kernel_size: Need smoothing kernel size.
            need_init_bias: Initial bias for direct need prediction.
            residual_offset_range: Learned residual offset range in pixel units.
            base_scale_multiplier: Multiplier applied to motion-derived base expansion.
            base_scale_min: Minimum base expansion in pixel units.
            base_scale_max: Maximum base expansion in pixel units.
            base_motion_epsilon: Translation norm threshold in pixels to switch between
                motion-based expansion and fixed `base_fixed_expansion_no_motion`.
            base_fixed_expansion_no_motion: Pixel expansion when below `base_motion_epsilon`.
            include_diagonal_base_offsets: Whether to include diagonal base directions.
            align_corners: `grid_sample` alignment mode.
            padding_mode: `grid_sample` padding mode.
            return_debug: Default debug return behavior.
        """
        super().__init__()
        self.return_debug = return_debug
        self.need_head = TemporalNeedHead(
            channels=channels,
            cue_distance_channels=cue_distance_channels,
            smoothing_kernel_size=need_smoothing_kernel_size,
            need_init_bias=need_init_bias,
        )
        self.window_router = WindowRouter(
            window_size=window_size,
            topk=need_topk,
            need_threshold=need_threshold,
        )
        self.base_offset_generator = BaseOffsetGenerator(
            point_cloud_range=point_cloud_range,
            scale_multiplier=base_scale_multiplier,
            min_expansion=base_scale_min,
            max_expansion=base_scale_max,
            motion_epsilon=base_motion_epsilon,
            fixed_expansion_no_motion=base_fixed_expansion_no_motion,
            include_diagonal_offsets=include_diagonal_base_offsets,
        )
        self.residual_offset_predictor = ResidualOffsetPredictor(
            channels=channels,
            num_points=num_points,
            offset_range=residual_offset_range,
        )
        self.temporal_sampler = TemporalSamplerAggregator(
            channels=channels,
            num_points=self.base_offset_generator.num_base_offsets * num_points,
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        self.register_buffer('history_feature', torch.empty(0), persistent=False)
        self.register_buffer('history_valid', torch.zeros(1, dtype=torch.bool), persistent=False)

    def _clear_history(self) -> None:
        """Clear temporal history."""
        self.history_feature = torch.empty(0, device=self.history_valid.device)
        self.history_valid.zero_()

    def _update_history(self, current_feature: torch.Tensor) -> None:
        """Store the pre-temporal fused feature for the next frame."""
        self.history_feature = current_feature.detach()
        self.history_valid.fill_(True)

    def _history_ready(self, current_feature: torch.Tensor) -> bool:
        """Check whether the cached historical feature is usable."""
        if not bool(self.history_valid.item()):
            return False
        if self.history_feature.numel() == 0:
            return False
        return tuple(self.history_feature.shape) == tuple(current_feature.shape)

    def forward(
        self,
        current_feature: torch.Tensor,
        relative_pose: Optional[torch.Tensor],
        temporal_reset: bool = False,
        distance_drone: Optional[torch.Tensor] = None,
        gate_drone: Optional[torch.Tensor] = None,
        distance_rsu: Optional[torch.Tensor] = None,
        gate_rsu: Optional[torch.Tensor] = None,
        return_debug: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Apply sparse temporal refinement to the current fused BEV.

        Args:
            current_feature: Current fused feature `[B, C, H, W]`.
            relative_pose: Relative pose `[B, 4, 4]`.
            temporal_reset: Whether the whole batch starts a new temporal chunk.
            distance_drone: Drone stacked cue maps `[B, C_cue, H, W]`.
            gate_drone: Drone final fusion gate `[B, 1, H, W]`.
            distance_rsu: RSU stacked cue maps `[B, C_cue, H, W]`.
            gate_rsu: RSU final fusion gate `[B, 1, H, W]`.
            return_debug: Override the default debug behavior.

        Returns:
            Refined BEV `[B, C, H, W]`. When debug is enabled, also returns a
            dictionary with routing and temporal sampling tensors.
        """
        debug_flag = self.return_debug if return_debug is None else return_debug
        if current_feature.dim() != 4:
            raise ValueError(f"current_feature must be [B, C, H, W], got {current_feature.shape}")

        confidence_map, need_map = self.need_head(
            current_feature=current_feature,
            distance_drone=distance_drone,
            gate_drone=gate_drone,
            distance_rsu=distance_rsu,
            gate_rsu=gate_rsu,
        )
        pixel_mask, window_mask, window_scores = self.window_router(need_map)

        if temporal_reset:
            self._clear_history()

        history_ready_before_update = self._history_ready(current_feature)
        temporal_applied = False
        delta_p_base: Optional[torch.Tensor] = None
        delta_p_res: Optional[torch.Tensor] = None
        temporal_attention: Optional[torch.Tensor] = None
        temporal_samples: Optional[torch.Tensor] = None
        delta_f = torch.zeros_like(current_feature)
        refined_feature = current_feature
        if relative_pose is not None and history_ready_before_update:
            batch_size, _, height, width = current_feature.shape
            history_feature = self.history_feature.to(
                device=current_feature.device,
                dtype=current_feature.dtype,
            )
            base_direction_bank: Optional[torch.Tensor]
            base_expansion_scale: Optional[torch.Tensor]
            rigid_offset: Optional[torch.Tensor]
            delta_p_base, rigid_offset, base_expansion_scale, base_direction_bank = self.base_offset_generator(
                relative_pose=relative_pose,
                height=height,
                width=width,
                device=current_feature.device,
                dtype=current_feature.dtype,
            )
            delta_p_res = self.residual_offset_predictor(
                current_feature=current_feature,
                need_map=need_map,
                relative_pose=relative_pose,
            )
            total_offsets = delta_p_base.unsqueeze(2) + delta_p_res.unsqueeze(1)
            total_offsets = total_offsets.reshape(batch_size, -1, 2, height, width)
            temporal_residual, temporal_attention, temporal_samples = self.temporal_sampler(
                current_feature=current_feature,
                history_feature=history_feature,
                offsets=total_offsets,
            )
            delta_f = temporal_residual * pixel_mask
            refined_feature = current_feature + delta_f
            temporal_applied = bool(pixel_mask.any().item())
        else:
            rigid_offset = None
            base_expansion_scale = None
            base_direction_bank = None

        self._update_history(current_feature)

        if not debug_flag:
            return refined_feature

        debug_info: Dict[str, Any] = {
            'temporal_applied': temporal_applied,
            'confidence_map': confidence_map,
            'need_map': need_map,
            'pixel_mask': pixel_mask,
            'window_mask': window_mask,
            'window_scores': window_scores,
            'delta_p_base': delta_p_base,
            'delta_p_res': delta_p_res,
            'rigid_offset': rigid_offset,
            'base_expansion_scale': base_expansion_scale,
            'base_direction_bank': base_direction_bank,
            'delta_f': delta_f,
            'temporal_attention': temporal_attention,
            'temporal_samples': temporal_samples,
            'history_ready': history_ready_before_update,
        }
        return refined_feature, debug_info
