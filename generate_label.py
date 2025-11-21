"""
AirV2X Dynamic Segmentation Label Class Definitions

Dynamic Segmentation: 7 classes (0-6)
Based on basedataset.py:885 - assert len(seg_bev_imgs_list) == 7
File order: map_dynamic_bev_layer_0.png, ..., map_dynamic_bev_layer_6.png
Label indices: 0, 1, 2, 3, 4, 5, 6

Classes: background, cars, motorcycles, bicycles, vans, trucks, and buses

FOR DETR TRAINING:
'object_bbx_center', 'object_bbx_mask', 'class_ids' is needed for DETR training.
"""
DYNAMIC_SEG_CLASSES = [
    "background",  # class 0 (map_dynamic_bev_layer_0.png)
    "car",         # class 1 (map_dynamic_bev_layer_1.png)
    "motorcycle",  # class 2 (map_dynamic_bev_layer_2.png)
    "bicycle",     # class 3 (map_dynamic_bev_layer_3.png)
    "van",         # class 4 (map_dynamic_bev_layer_4.png)
    "truck",       # class 5 (map_dynamic_bev_layer_5.png)
    "bus",         # class 6 (map_dynamic_bev_layer_6.png)
]

# Label index mapping:
# - Dynamic: 0=background, 1-6=object classes
# - Invalid points: -1 (points outside BEV range)

import os
import os.path as osp
import sys
import re
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add current directory to path for imports
root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset


def assign_bev_labels_to_points(points, bev_label_map, lidar_range, seg_res, seg_hw=512):
    """
    Assign BEV labels to point cloud points.
    
    This function implements the reverse projection from BEV labels to point cloud,
    following AirV2X's label processing method (basedataset.py:883-934).
    
    Note: BEV label map has been processed with:
    - label_map = label_map.T  (transpose)
    - label_map = label_map[:, ::-1]  (flip y-axis)
    
    The function handles the case where the BEV label map may not be exactly 512x512:
    1. If the label map is larger than seg_hw x seg_hw, it crops to the center seg_hw x seg_hw region
    2. If the label map is smaller than seg_hw x seg_hw, it pads with zeros (background class 0)
    3. Points are projected using seg_res (0.25 m/pixel) resolution
    4. A mask is generated to indicate which points are within the valid BEV range
    
    Args:
        points: [N, 3] point cloud (x, y, z) in LiDAR coordinate system
        bev_label_map: [H, W] BEV label map (already processed with transpose and flip)
        lidar_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        seg_res: BEV resolution (meters per pixel), typically 0.25
        seg_hw: Target BEV map size (height and width), default 512
    
    Returns:
        point_labels: [N] labels for each point (0-based class indices, -1 for invalid)
        valid_mask: [N] boolean mask indicating which points are within valid BEV range
    """
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points

    if isinstance(bev_label_map, torch.Tensor):
        bev_label_map = bev_label_map.detach().cpu().numpy()

    x_min, y_min = lidar_range[0], lidar_range[1]
    H_orig, W_orig = bev_label_map.shape
    
    # Crop or pad BEV label map to target size (seg_hw x seg_hw)
    if H_orig != seg_hw or W_orig != seg_hw:
        if H_orig > seg_hw or W_orig > seg_hw:
            # Crop: take center region
            h_start = (H_orig - seg_hw) // 2
            w_start = (W_orig - seg_hw) // 2
            bev_label_map = bev_label_map[h_start:h_start+seg_hw, w_start:w_start+seg_hw]
        else:
            # Pad: pad with zeros (background class 0)
            h_pad = (seg_hw - H_orig) // 2
            w_pad = (seg_hw - W_orig) // 2
            bev_label_map = np.pad(
                bev_label_map,
                ((h_pad, seg_hw - H_orig - h_pad), (w_pad, seg_hw - W_orig - w_pad)),
                mode='constant',
                constant_values=0  # background class
            )
    
    H, W = bev_label_map.shape  # Now H == W == seg_hw
    
    # Calculate BEV pixel coordinates for each point
    # Note: BEV label map has been transposed and flipped, so we need to account for this
    # Original BEV: x -> W dimension, y -> H dimension
    # After transpose: x -> H dimension, y -> W dimension
    # After flip: y-axis is reversed
    
    # Map point (x, y) to BEV pixel coordinates
    # x coordinate maps to W dimension (columns)
    x_idx = ((points_np[:, 0] - x_min) / seg_res).astype(int)
    # y coordinate maps to H dimension (rows), but flipped
    y_idx = ((points_np[:, 1] - y_min) / seg_res).astype(int)
    
    # Boundary check: points must be within [0, seg_hw) for both x and y
    valid_mask = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
    
    # Debug: Print BEV label map statistics
    unique_labels = np.unique(bev_label_map)
    print(f"BEV label map shape: {bev_label_map.shape}")
    print(f"BEV label map unique values: {unique_labels}")
    print(f"BEV label map value range: [{bev_label_map.min()}, {bev_label_map.max()}]")
    print(f"BEV label map value counts:")
    for label_val in unique_labels:
        count = np.sum(bev_label_map == label_val)
        percentage = count / bev_label_map.size * 100
        print(f"  Label {label_val}: {count} pixels ({percentage:.2f}%)")
    # Initialize labels as invalid (-1)
    num_points = points_np.shape[0]
    point_labels = np.full(num_points, -1, dtype=np.int16)
    
    # Get labels from BEV map
    # After transpose and flip: bev_label_map[y_idx, x_idx] corresponds to point (x, y)
    point_labels[valid_mask] = bev_label_map[y_idx[valid_mask], x_idx[valid_mask]]
    
    return point_labels, valid_mask


def project_lidar_to_image(points, lidar2camera, camera_intrinsics, img_height, img_width):
    """
    Project LiDAR points to image plane.
    
    Args:
        points: [N, 3] LiDAR points
        lidar2camera: [4, 4] transformation matrix
        camera_intrinsics: [3, 3] camera intrinsic matrix
        img_height: image height
        img_width: image width
    
    Returns:
        points_2d: [N, 2] image coordinates (u, v)
        valid_mask: [N] boolean mask for valid points
    """
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # [N, 4]
    
    # Transform to camera coordinates
    points_camera = (lidar2camera @ points_homogeneous.T).T  # [N, 4]
    points_camera = points_camera[:, :3]  # [N, 3]
    
    # Filter points with negative depth
    valid_depth_mask = points_camera[:, 2] > 0
    points_camera = points_camera[valid_depth_mask]
    
    if points_camera.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(points.shape[0], dtype=bool)
    
    # Project to image plane
    points_2d_homogeneous = (camera_intrinsics @ points_camera.T).T  # [N, 3]
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]  # [N, 2]
    
    # Filter points outside image bounds
    valid_bounds_mask = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
    )
    
    # Create full valid mask
    full_valid_mask = np.zeros(points.shape[0], dtype=bool)
    full_valid_mask[valid_depth_mask] = valid_bounds_mask
    
    # Create full points_2d array
    full_points_2d = np.zeros((points.shape[0], 2))
    full_points_2d[full_valid_mask] = points_2d[valid_bounds_mask]
    
    return full_points_2d, full_valid_mask


def voxelize_points_and_labels(points, point_labels, voxel_size, grid_size, point_cloud_range):
    """
    Voxelize points and aggregate point labels to voxel labels using majority voting.
    
    This function implements the same voxelization logic as DynamicVoxelVFE:
    1. Convert point coordinates to voxel indices
    2. Filter points outside grid bounds
    3. Group points into voxels
    4. Aggregate point labels to voxel labels using majority voting
    
    Args:
        points: [N, 3] point cloud coordinates (x, y, z)
        point_labels: [N] point labels (0-6 for classes, -1 for invalid)
        voxel_size: [3] voxel size in meters [size_x, size_y, size_z]
        grid_size: [3] grid size [size_x, size_y, size_z]
        point_cloud_range: [6] point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    
    Returns:
        voxel_coords: [V, 4] voxel coordinates [batch_idx, z, y, x]
        voxel_labels: [V] voxel labels (majority vote from points in each voxel)
        voxel_point_counts: [V] number of points in each voxel
    """
    points = np.asarray(points, dtype=np.float32)
    point_labels = np.asarray(point_labels, dtype=np.int16)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    grid_size = np.asarray(grid_size, dtype=np.int32)
    point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
    
    # Extract range min values for x, y, z
    range_min = point_cloud_range[:3]  # [x_min, y_min, z_min]
    
    # Step 1: Convert point coordinates to voxel indices
    # Formula: voxel_idx = floor((point_coord - range_min) / voxel_size)
    points_coords = np.floor((points[:, [0, 1, 2]] - range_min) / voxel_size).astype(np.int32)
    
    # Step 2: Filter points outside grid bounds
    valid_mask = (
        (points_coords[:, 0] >= 0) & (points_coords[:, 0] < grid_size[0]) &
        (points_coords[:, 1] >= 0) & (points_coords[:, 1] < grid_size[1]) &
        (points_coords[:, 2] >= 0) & (points_coords[:, 2] < grid_size[2])
    )
    
    points = points[valid_mask]
    points_coords = points_coords[valid_mask]
    point_labels = point_labels[valid_mask]
    
    if points.shape[0] == 0:
        # Return empty arrays if no valid points
        return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.int32)
    
    # Step 3: Group points into voxels by converting 3D indices to 1D unique values
    # Same logic as DynamicVoxelVFE: merge_coords = x * scale_yz + y * scale_z + z
    scale_yz = grid_size[1] * grid_size[2]
    scale_z = grid_size[2]
    merge_coords = (
        points_coords[:, 0] * scale_yz +
        points_coords[:, 1] * scale_z +
        points_coords[:, 2]
    )
    
    # Find unique voxels
    unq_coords, unq_inv, unq_cnt = np.unique(merge_coords, return_inverse=True, return_counts=True)
    num_voxels = len(unq_coords)
    
    # Step 4: Aggregate point labels to voxel labels using majority voting
    # For each voxel, find the most common label among its points
    voxel_labels = np.full(num_voxels, -1, dtype=np.int16)
    voxel_point_counts = unq_cnt.astype(np.int32)
    
    # Filter out invalid labels (-1) for majority voting
    valid_label_mask = point_labels != -1
    
    for voxel_idx in range(num_voxels):
        # Get points belonging to this voxel
        voxel_point_mask = (unq_inv == voxel_idx)
        voxel_point_labels = point_labels[voxel_point_mask]
        
        # Filter valid labels
        valid_voxel_labels = voxel_point_labels[voxel_point_labels != -1]
        
        if len(valid_voxel_labels) > 0:
            # Majority voting: find the most common label
            unique_labels, counts = np.unique(valid_voxel_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            voxel_labels[voxel_idx] = majority_label
        else:
            # If all points in voxel have invalid labels, keep -1
            voxel_labels[voxel_idx] = -1
    
    # Step 5: Decode 1D coordinates back to 3D (z, y, x) format
    # Same logic as DynamicVoxelVFE
    voxel_coords_3d = np.zeros((num_voxels, 3), dtype=np.int32)
    voxel_coords_3d[:, 0] = unq_coords // scale_yz  # x
    voxel_coords_3d[:, 1] = (unq_coords % scale_yz) // scale_z  # y
    voxel_coords_3d[:, 2] = unq_coords % scale_z  # z
    
    # Reorder to [z, y, x] format (matching DynamicVoxelVFE output)
    voxel_coords_3d = voxel_coords_3d[:, [2, 1, 0]]  # [z, y, x]
    
    # Add batch_idx dimension (always 0 for single batch)
    batch_idx = np.zeros((num_voxels, 1), dtype=np.int32)
    voxel_coords = np.concatenate([batch_idx, voxel_coords_3d], axis=1)  # [batch_idx, z, y, x]
    
    return voxel_coords, voxel_labels, voxel_point_counts


def generate(hypes_yaml, save_dir):
    """
    Generate point labels for AirV2X dataset.
    
    Args:
        hypes_yaml: Path to yaml config file
        save_dir: Directory to save generated labels
    """
    print("=" * 80)
    print("Loading AirV2X Dataset")
    print("=" * 80)
    
    # Load config
    hypes = yaml_utils.load_yaml(hypes_yaml, None)
    
    # Build dataset with visualize=True to get origin_lidar
    print("Building dataset with visualize=True...")
    dataset = build_dataset(hypes, visualize=True, train=True)
    print(f"Dataset built: {len(dataset)} samples found.")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,  # Set to 0 for debugging, increase for faster processing
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )
    
    # Get lidar range and BEV resolution from config
    lidar_range = hypes["preprocess"]["args"].get("cav_lidar_range", 
        [-140.8, -40, -3, 140.8, 40, 1])
    # Get BEV segmentation resolution (meters per pixel)
    # Check in multiple possible locations in config
    seg_res = hypes.get("seg_res", None)
    if seg_res is None:
        seg_res = hypes.get("segmentation", {}).get("seg_res", 0.25)
    if seg_res is None:
        seg_res = 0.25  # Default BEV resolution
    
    # Get BEV segmentation map size (height and width)
    seg_hw = hypes.get("seg_hw", None)
    if seg_hw is None:
        seg_hw = hypes.get("segmentation", {}).get("seg_hw", 512)
    if seg_hw is None:
        seg_hw = 512  # Default BEV map size
    
    # Get voxelization parameters (for voxel-level label generation)
    # These parameters should match the model's voxelization settings
    voxel_size = hypes["preprocess"]["args"].get("voxel_size", [0.4, 0.4, 4])
    if isinstance(voxel_size, list) and len(voxel_size) == 3:
        voxel_size = voxel_size
    else:
        voxel_size = [0.4, 0.4, 4]  # Default: [x, y, z]
    
    # Calculate grid_size from lidar_range and voxel_size
    # grid_size = ceil((range_max - range_min) / voxel_size)
    grid_size_x = int(np.ceil((lidar_range[3] - lidar_range[0]) / voxel_size[0]))
    grid_size_y = int(np.ceil((lidar_range[4] - lidar_range[1]) / voxel_size[1]))
    grid_size_z = int(np.ceil((lidar_range[5] - lidar_range[2]) / voxel_size[2]))
    grid_size = [grid_size_x, grid_size_y, grid_size_z]
    
    # point_cloud_range is the same as lidar_range
    point_cloud_range = lidar_range
    
    print(f"Lidar range: {lidar_range}")
    print(f"BEV resolution: {seg_res} (meters per pixel)")
    print(f"BEV map size: {seg_hw}x{seg_hw} pixels")
    print(f"Dynamic segmentation classes: {len(DYNAMIC_SEG_CLASSES)} classes")
    print(f"  {DYNAMIC_SEG_CLASSES}")
    print(f"\nVoxelization parameters:")
    print(f"  Voxel size: {voxel_size} (meters)")
    print(f"  Grid size: {grid_size} (voxels)")
    print(f"  Point cloud range: {point_cloud_range}")
    print("=" * 80)
    
    
    # Process each sample
    for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        try:
            ego_data = batch_data['ego']
            import pdb; pdb.set_trace()
            # Get point clouds for each agent type
            for agent_type in ['veh', 'rsu', 'drone']:
                origin_lidar_key = f'origin_lidar_{agent_type}' if agent_type != 'drone' else 'origin_lidar'
                cam_inputs_key = f'cam_inputs_{agent_type}'
                if origin_lidar_key not in ego_data:
                    continue
                    
                points = ego_data[origin_lidar_key][0]  # [N, 4] (x, y, z, intensity)
                if points.shape[0] == 0:
                    continue
                
                # Get BEV labels from label_dict (following AirV2X's method)
                # Labels are stored in label_dict after post-processing
                label_dict = ego_data.get('label_dict', {})
                dynamic_seg_label = label_dict.get('dynamic_seg_label', None)
                
                # Fallback: try to get from ego_data directly (for backward compatibility)
                if dynamic_seg_label is None:
                    dynamic_seg_label = ego_data.get('dynamic_seg_label', None)
                
                # Get camera inputs
                cam_inputs = ego_data.get(cam_inputs_key, None)# TODO
                
                # Get metadata path for saving
                metadata_path = ego_data.get('metadata_path', None)
                if metadata_path is not None:
                    if isinstance(metadata_path, list):
                        metadata_path = metadata_path[0]
                
                # Process point cloud labels from BEV
                # For all three agent types (vehicle, rsu, drone), use the same BEV label map
                # The BEV label map is in ego coordinate system, so all points are already transformed
                if dynamic_seg_label is not None:
                    dynamic_label_np = dynamic_seg_label[0].numpy() if hasattr(dynamic_seg_label, 'numpy') else dynamic_seg_label
                    if isinstance(dynamic_label_np, np.ndarray):
                        point_dynamic_labels, dynamic_valid_mask = assign_bev_labels_to_points(
                            points[:, :3], 
                            dynamic_label_np,
                            lidar_range, 
                            seg_res,
                            seg_hw
                        )
                    else:
                        point_dynamic_labels = np.full(points.shape[0], -1, dtype=np.int16)
                        dynamic_valid_mask = np.zeros(points.shape[0], dtype=bool)
                else:
                    point_dynamic_labels = np.full(points.shape[0], -1, dtype=np.int16)
                    dynamic_valid_mask = np.zeros(points.shape[0], dtype=bool)
                
                # Voxelize points and aggregate labels using majority voting
                # This generates voxel-level labels that match the model's prediction format
                if points.shape[0] > 0:
                    # Convert points to numpy if needed
                    points_np = points[:, :3].cpu().numpy() if hasattr(points, 'cpu') else points[:, :3]
                    if isinstance(points_np, torch.Tensor):
                        points_np = points_np.detach().cpu().numpy()
                    
                    # Convert point_labels to numpy if needed
                    point_labels_np = point_dynamic_labels
                    if isinstance(point_labels_np, torch.Tensor):
                        point_labels_np = point_labels_np.detach().cpu().numpy()
                    
                    # Perform voxelization
                    voxel_coords, voxel_labels, voxel_point_counts = voxelize_points_and_labels(
                        points_np,
                        point_labels_np,
                        voxel_size,
                        grid_size,
                        point_cloud_range
                    )
                    
                    # Save voxel labels
                    if voxel_coords.shape[0] > 0:
                        # Generate save path for voxel labels
                        if metadata_path:
                            timestamp_match = None
                            if isinstance(metadata_path, str):
                                timestamp_match = re.search(r'(\d{4}(?:_\d{2}){5})', metadata_path)
                            
                            if timestamp_match:
                                timestamp = timestamp_match.group(1)
                                voxel_label_save_path = osp.join(
                                    save_dir,
                                    f"{timestamp}_{agent_type}_voxel_labels.npy"
                                )
                            else:
                                voxel_label_save_path = osp.join(
                                    save_dir,
                                    f"sample_{idx}_{agent_type}_voxel_labels.npy"
                                )
                        else:
                            voxel_label_save_path = osp.join(
                                save_dir,
                                f"sample_{idx}_{agent_type}_voxel_labels.npy"
                            )
                        
                        # Create directory if needed
                        os.makedirs(osp.dirname(voxel_label_save_path), exist_ok=True)
                        
                        # Save voxel data: only coords, labels, and voxel_size
                        voxel_data = {
                            'coords': voxel_coords,  # [V, 4] [batch_idx, z, y, x]
                            'labels': voxel_labels,  # [V] voxel labels (0-6, -1 for invalid)
                            'voxel_size': voxel_size  # [3] voxel size [size_x, size_y, size_z]
                        }
                        np.save(voxel_label_save_path, voxel_data, allow_pickle=True)
                
                # Project points to camera images if available
                if cam_inputs is not None and len(cam_inputs) > 0:
                    for cam_idx, cam_data in enumerate(cam_inputs):
                        if cam_data is None:
                            continue
                            
                        # Extract camera parameters
                        imgs = cam_data.get('imgs', None)
                        intrinsics = cam_data.get('intrinsics', None)
                        extrinsics = cam_data.get('extrinsics', None)
                        
                        if imgs is None or intrinsics is None or extrinsics is None:
                            continue
                        
                        # Process each camera
                        for single_cam_idx in range(imgs.shape[0]):
                            img = imgs[single_cam_idx]  # [C, H, W]
                            intrinsic = intrinsics[single_cam_idx]  # [3, 3]
                            extrinsic = extrinsics[single_cam_idx]  # [4, 4]
                            
                            # Convert to numpy if needed
                            if hasattr(extrinsic, 'numpy'):
                                extrinsic = extrinsic.numpy()
                            if hasattr(intrinsic, 'numpy'):
                                intrinsic = intrinsic.numpy()
                            
                            # Project points to image
                            points_2d, valid_mask = project_lidar_to_image(
                                points[:, :3],
                                extrinsic,
                                intrinsic,
                                img.shape[1],  # height
                                img.shape[2]   # width
                            )
                            
                            if valid_mask.sum() == 0:
                                continue
                            
                            # Create labeled points for image
                            valid_points_2d = points_2d[valid_mask]
                            valid_labels_dynamic = point_dynamic_labels[valid_mask]
                            valid_depth = points[valid_mask, 2]
                            
                            # Save labeled points for image
                            points_label = np.concatenate([
                                valid_points_2d,  # [N, 2] (u, v)
                                valid_depth[:, np.newaxis],  # [N, 1] (depth)
                                valid_labels_dynamic[:, np.newaxis],  # [N, 1] (dynamic label)
                            ], axis=1)  # [N, 4]
                            
                            # Generate save path
                            if metadata_path:
                                # Extract timestamp from metadata path
                                timestamp_match = None
                                if isinstance(metadata_path, str):
                                    timestamp_match = re.search(r'(\d{4}(?:_\d{2}){5})', metadata_path)
                                
                                if timestamp_match:
                                    timestamp = timestamp_match.group(1)
                                    label_save_path = osp.join(
                                        save_dir,
                                        f"{timestamp}_{agent_type}_cam{cam_idx}_{single_cam_idx}.npy"
                                    )
                                else:
                                    label_save_path = osp.join(
                                        save_dir,
                                        f"sample_{idx}_{agent_type}_cam{cam_idx}_{single_cam_idx}.npy"
                                    )
                            else:
                                label_save_path = osp.join(
                                    save_dir,
                                    f"sample_{idx}_{agent_type}_cam{cam_idx}_{single_cam_idx}.npy"
                                )
                            
                            # Create directory if needed
                            os.makedirs(osp.dirname(label_save_path), exist_ok=True)
                            
                            # Save
                            np.save(label_save_path, points_label)  # Save as [N, 4]: (u, v, depth, dynamic_label)
                            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate point labels for AirV2X dataset')
    parser.add_argument('--hypes_yaml', type=str, required=True,
                       help='Path to AirV2X yaml config file')
    parser.add_argument('--save_dir', type=str, default='samples_point_label',
                       help='Directory to save generated labels')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    generate(args.hypes_yaml, args.save_dir)
