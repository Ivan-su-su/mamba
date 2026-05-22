# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""

import sys
import warnings
from typing import Sequence, Mapping, Dict

import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor


warnings.filterwarnings(
    "once",
    category=UserWarning,
    module="sp_voxel_preprocessor",
    message="Warning: empty point cloud. Add dummy points.*"
)


class EmptyPointCloudWarning(UserWarning):
    pass


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params, train)

        self.params = preprocess_params
        self.train = train

        self.lidar_range = self.params["cav_lidar_range"]
        self.voxel_size = self.params["args"]["voxel_size"]
        self.max_points_per_voxel = self.params["args"]["max_points_per_voxel"]

        if train:
            self.max_voxels = self.params["args"]["max_voxel_train"]
        else:
            self.max_voxels = self.params["args"]["max_voxel_test"]

        grid_size = (
            np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])
        ) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # Prefer new spconv 2.x pytorch API.
        # This avoids importing spconv.utils, which fails in your source-built spconv.
        try:
            import cumm.core_cc  # noqa: F401
            from spconv.pytorch.utils import PointToVoxel

            self.spconv_version = "spconv2_pytorch"

            self.voxel_generator = PointToVoxel(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                num_point_features=4,
                max_num_voxels=self.max_voxels,
                max_num_points_per_voxel=self.max_points_per_voxel,
                device=torch.device("cpu"),
            )

        except Exception as e:
            raise RuntimeError(
                "Failed to import/use spconv.pytorch.utils.PointToVoxel; "
                "do not fallback to spconv.utils in this environment."
            ) from e

            print("[SpVoxelPreprocessor] Failed to use spconv.pytorch.utils.PointToVoxel.")
            print("[SpVoxelPreprocessor] Error:", repr(e))
            print("[SpVoxelPreprocessor] Falling back to old spconv.utils voxel generator.")

            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
                self.spconv_version = "spconv1_v2"
            except Exception:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_version = "spconv2_cpu_old"

            if self.spconv_version == "spconv1_v2":
                self.voxel_generator = VoxelGenerator(
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.lidar_range,
                    max_num_points=self.max_points_per_voxel,
                    max_voxels=self.max_voxels,
                )
            else:
                from cumm import tensorview as tv  # noqa: F401

                self.voxel_generator = VoxelGenerator(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.lidar_range,
                    max_num_points_per_voxel=self.max_points_per_voxel,
                    num_point_features=4,
                    max_num_voxels=self.max_voxels,
                )

    def preprocess(self, pcd_np):
        data_dict = {}

        # Handle empty lidar points.
        if len(pcd_np) == 0:
            pcd_np1 = np.zeros((1, 4), dtype=np.float32)
            pcd_np2 = np.array(
                [-0.218277, -11.13425732, -80.05884552, 1.230595649e-38],
                dtype=np.float32
            ).reshape(1, 4)

            pcd_np = np.concatenate((pcd_np1, pcd_np2), axis=0)

            with warnings.catch_warnings():
                warnings.warn(
                    "Warning: empty point cloud. Add dummy points. Add dummy points. "
                    "This is because of some package loss during the data collection. "
                    "It will be solved in the future dataset version.",
                    EmptyPointCloudWarning
                )

        pcd_np = np.asarray(pcd_np, dtype=np.float32)

        if self.spconv_version == "spconv2_pytorch":
            points = torch.from_numpy(pcd_np).float()
            voxels, coordinates, num_points = self.voxel_generator(points)

            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        elif self.spconv_version == "spconv1_v2":
            voxel_output = self.voxel_generator.generate(pcd_np)

            if isinstance(voxel_output, dict):
                voxels = voxel_output["voxels"]
                coordinates = voxel_output["coordinates"]
                num_points = voxel_output["num_points_per_voxel"]
            else:
                voxels, coordinates, num_points = voxel_output

        else:
            from cumm import tensorview as tv

            pcd_tv = tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)

            if isinstance(voxel_output, dict):
                voxels = voxel_output["voxels"]
                coordinates = voxel_output["coordinates"]
                num_points = voxel_output["num_points_per_voxel"]
            else:
                voxels, coordinates, num_points = voxel_output

            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        data_dict["voxel_features"] = voxels
        data_dict["voxel_coords"] = coordinates
        data_dict["voxel_num_points"] = num_points

        return data_dict

    def collate_batch(self, batch):
        """
        Collate function for a list-of-dict or dict-of-list batch.
        """
        if isinstance(batch, Sequence):
            batch = self._transpose_to_dict(batch)
        elif not isinstance(batch, Mapping):
            raise TypeError("batch must be list or dict, got {}".format(type(batch)))

        vf = torch.from_numpy(np.concatenate(batch["voxel_features"]))
        vnp = torch.from_numpy(np.concatenate(batch["voxel_num_points"]))

        coords_with_idx = np.concatenate(
            [
                np.hstack((np.full((c.shape[0], 1), i, dtype=c.dtype), c))
                for i, c in enumerate(batch["voxel_coords"])
            ],
            axis=0,
        )
        vc = torch.from_numpy(coords_with_idx)

        return {
            "voxel_features": vf,
            "voxel_coords": vc,
            "voxel_num_points": vnp,
        }

    @staticmethod
    def _transpose_to_dict(batch_list: Sequence[Dict]) -> Dict[str, list]:
        """
        Turn list-of-dict into dict-of-list without extra copies.
        """
        keys = batch_list[0].keys()
        out = {k: [] for k in keys}

        for sample in batch_list:
            for k in keys:
                out[k].append(sample[k])

        return out