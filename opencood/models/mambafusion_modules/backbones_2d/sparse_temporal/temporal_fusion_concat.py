from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ...vmamba.vmamba import LayerNorm2d


class RigidHistoryAligner(nn.Module):
    """Warp the cached historical BEV into the current BEV frame with rigid pose."""

    def __init__(
        self,
        point_cloud_range: Tuple[float, float, float, float, float, float],
        align_corners: bool = False,
        padding_mode: str = 'zeros',
    ) -> None:
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"point_cloud_range must contain 6 values, got {point_cloud_range}"
            )
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.register_buffer(
            'point_cloud_range',
            torch.tensor(point_cloud_range, dtype=torch.float32),
            persistent=False,
        )

    def _normalize_coordinate(self, coord: torch.Tensor, size: int) -> torch.Tensor:
        """Normalize pixel coordinates for `grid_sample`."""
        if self.align_corners:
            if size <= 1:
                return torch.zeros_like(coord)
            return (2.0 * coord / float(size - 1)) - 1.0
        return ((2.0 * coord + 1.0) / float(size)) - 1.0

    def build_rigid_offset(
        self,
        relative_pose: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build current-pixel to previous-pixel offsets from relative pose."""
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
        return torch.stack([offset_x, offset_y], dim=1)

    def forward(
        self,
        history_feature: torch.Tensor,
        relative_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align historical BEV feature to the current BEV frame."""
        if history_feature.dim() != 4:
            raise ValueError(f"history_feature must be [B, C, H, W], got {history_feature.shape}")
        batch_size, _, height, width = history_feature.shape
        rigid_offset = self.build_rigid_offset(
            relative_pose=relative_pose,
            height=height,
            width=width,
            device=history_feature.device,
            dtype=history_feature.dtype,
        )

        yy, xx = torch.meshgrid(
            torch.arange(height, device=history_feature.device, dtype=history_feature.dtype),
            torch.arange(width, device=history_feature.device, dtype=history_feature.dtype),
            indexing='ij',
        )
        base_x = xx.unsqueeze(0).expand(batch_size, -1, -1)
        base_y = yy.unsqueeze(0).expand(batch_size, -1, -1)
        sample_x = base_x + rigid_offset[:, 0]
        sample_y = base_y + rigid_offset[:, 1]
        grid_x = self._normalize_coordinate(sample_x, width)
        grid_y = self._normalize_coordinate(sample_y, height)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        aligned_history = F.grid_sample(
            history_feature,
            grid,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return aligned_history, rigid_offset


class TemporalFusionConcatBlock(nn.Module):
    """Temporal baseline: rigidly align history, concat with current BEV, predict residual."""

    def __init__(
        self,
        channels: int,
        point_cloud_range: Tuple[float, float, float, float, float, float],
        hidden_channels: Optional[int] = None,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
        return_debug: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_channels) if hidden_channels is not None else int(channels)
        self.return_debug = return_debug
        self.history_aligner = RigidHistoryAligner(
            point_cloud_range=point_cloud_range,
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, kernel_size=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
        )
        print("****************************************")
        print("use concat block")
        print("****************************************")
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
        epoch: Optional[Any] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Apply rigid-align concat temporal fusion to the current fused BEV."""
        del distance_drone, gate_drone, distance_rsu, gate_rsu, epoch
        debug_flag = self.return_debug if return_debug is None else return_debug
        if current_feature.dim() != 4:
            raise ValueError(f"current_feature must be [B, C, H, W], got {current_feature.shape}")

        if temporal_reset:
            self._clear_history()

        history_ready_before_update = self._history_ready(current_feature)
        temporal_applied = False
        aligned_history: Optional[torch.Tensor] = None
        rigid_offset: Optional[torch.Tensor] = None
        delta_f = torch.zeros_like(current_feature)
        refined_feature = current_feature

        if relative_pose is not None and history_ready_before_update:
            history_feature = self.history_feature.to(
                device=current_feature.device,
                dtype=current_feature.dtype,
            )
            aligned_history, rigid_offset = self.history_aligner(
                history_feature=history_feature,
                relative_pose=relative_pose,
            )
            delta_f = self.residual_head(torch.cat([current_feature, aligned_history], dim=1))
            refined_feature = current_feature + delta_f
            temporal_applied = True

        self._update_history(current_feature)

        if not debug_flag:
            return refined_feature

        debug_info: Dict[str, Any] = {
            'temporal_applied': temporal_applied,
            'aligned_history': aligned_history,
            'rigid_offset': rigid_offset,
            'delta_f': delta_f,
            'history_ready': history_ready_before_update,
        }
        return refined_feature, debug_info
