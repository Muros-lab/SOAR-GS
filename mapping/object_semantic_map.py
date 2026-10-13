"""ObjectSemanticMap: semantic voxel map of SOAR-GS (class-level geometric complexity, semantic boundaries)"""


import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.ndimage import binary_dilation, generate_binary_structure

from utils.operations import *
from utils.common import TextColors


class ObjectSemanticMap:
    
    def __init__(self, cfg, bbox, device, detector_cfg: Optional[Dict] = None):
        self.device = device
        self.use_detector = detector_cfg is not None and detector_cfg.get('enabled', True)
        
        # Configuration
        self.min_gaussian_per_voxel = cfg.min_gaussian_per_voxel
        self.occ_thres = 0.8
        self.free_thres = 0.2
        
        # ============================================
        # CORE VOXEL GRID STRUCTURE
        # ============================================
        bbox = torch.tensor(bbox, device=device)
        extents = bbox[1] - bbox[0]
        map_resolution = torch.tensor(cfg.map_resolution, device=device)
        dim = torch.ceil(extents / map_resolution).int()
        size = extents / dim
        
        # Create voxel grid
        indices_x = torch.arange(dim[0], device=device)
        indices_y = torch.arange(dim[1], device=device)
        indices_z = torch.arange(dim[2], device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(
            indices_x, indices_y, indices_z, indexing="ij"
        )
        
        # Voxel centers in world coordinates
        centers_x = bbox[0][0] + (grid_x + 0.5) * size[0]
        centers_y = bbox[0][1] + (grid_y + 0.5) * size[1]
        centers_z = bbox[0][2] + (grid_z + 0.5) * size[2]
        
        self.voxel_centers = torch.stack(
            (centers_x, centers_y, centers_z), dim=-1
        ).view(-1, 3)
        
        self.voxel_indices = torch.floor((self.voxel_centers - bbox[0]) / size).int()
        
        # Homogeneous coordinates for projection
        self.points_3d_hom = torch.cat(
            (self.voxel_centers, torch.ones((self.voxel_centers.shape[0], 1), device=device)),
            dim=-1,
        )  # (N, 4)
        
        self.dim = dim
        self.size = size
        self.bbox = bbox
        
        # Structuring elements for morphology operations
        self.occ_structure_element = self._create_spherical_structuring_element(
            np.max(np.array(cfg.safety_margin) / size.cpu().numpy())
        )
        self.frontier_structure_element = generate_binary_structure(3, 1)
        
        # ============================================
        # OCCUPANCY INFORMATION (Geometric)
        # ============================================
        N = torch.prod(dim).item()
        self.voxel_log_odds = torch.zeros(N, device=device)  # Log-odds representation
        self.unexplored_mask = torch.ones(N, dtype=torch.bool, device=device)
        self.planning_mask = torch.zeros(N, device=device)
        self.roi_mask = torch.zeros(N, dtype=torch.bool, device=device)
        self.voxel_normal = torch.zeros((N, 3), device=device)
        
        # ============================================
        # SEMANTIC INFORMATION (Objects)
        # ============================================
        if self.use_detector:
            print(f"\n{TextColors.CYAN}[ObjectSemanticMap] Initializing with object detection{TextColors.RESET}")
            
            from detector import ObjectDetector, SemanticProjector
            
            # Object detector (YOLO-World + SAM)
            self.detector = ObjectDetector(
                yolo_model=detector_cfg.get('yolo_model', 'yolov8x-worldv2'),
                efficient_sam_model=detector_cfg.get('sam_model', 'vits'),
                class_names=detector_cfg.get('class_names', []),
                confidence_threshold=detector_cfg.get('confidence_threshold', 0.4),
                weights_dir=detector_cfg.get('weights_dir', './weights'),
                device=device,
            )
            
            # Semantic projector (2D → 3D)
            self.projector = SemanticProjector(self)
            
            # NOTE: No predefined class weights!
            # Weights are computed purely from volume statistics (adaptive)
            
            # Semantic fields (-1 = unlabeled/unknown)
            self.voxel_semantic_class = torch.full((N,), -1, dtype=torch.long, device=device)
            self.voxel_class_confidence = torch.zeros(N, device=device)
            self.voxel_object_count = torch.zeros(N, dtype=torch.long, device=device)
            self.voxel_object_scores = torch.zeros(N, device=device)
            
            # Tracking structures
            self.class_vote_history = defaultdict(lambda: defaultdict(int))
            self.object_observation_stats = {}
            self.object_volume_stats = {}
            
            # Adaptive weighting
            adaptive_cfg = detector_cfg.get('adaptive_weighting', {})
            self.use_adaptive = adaptive_cfg.get('enabled', True)
            self.adaptive_cfg = adaptive_cfg
            self.coarse_phase_complete = False
            
            # Volume stats tracking for adaptive updates
            self.last_volume_update_pose = None
            self.last_detected_classes = set()
            
            # Cache for boundary computations (invalidated on semantic update)
            self._boundary_cache_valid = False
            self._cached_boundary_mask = None
            self._cached_boundary_complexity = None
            
            # Fine exploration start target (Object Completion-Driven Planning)
            self.fine_start_target = None  # Target voxel position for fine start
            self.fine_start_used = False   # Flag to prevent repeated use
            
            # View diversity tracking (3D Spherical Variance)
            # Stores unit vectors: camera → object centroid direction per observation
            self.object_view_directions = defaultdict(list)  # {class_name: [unit_vec1, unit_vec2, ...]}
            
        else:
            print(f"\n{TextColors.YELLOW}[ObjectSemanticMap] Detector disabled{TextColors.RESET}")
            self.detector = None
            self.projector = None
        
        print(f"  Voxel grid: {dim[0]} × {dim[1]} × {dim[2]} = {N:,} voxels")
        print(f"  Resolution: {size[0]:.3f}m × {size[1]:.3f}m × {size[2]:.3f}m")
        print(f"  Bounding box: {bbox[0].cpu().numpy()} to {bbox[1].cpu().numpy()}\n")
    
    def _create_spherical_structuring_element(self, radius):
        """Create spherical structuring element for safety margin dilation."""
        size = int(np.ceil(radius)) * 2 + 1
        center = size // 2
        struct = np.zeros((size, size, size), dtype=bool)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    if np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2) <= radius:
                        struct[i, j, k] = True
        return struct
    
    # ============================================
    # COORDINATE TRANSFORMATIONS
    # ============================================
    
    def _voxelize(self, points_3d):
        """Convert 3D points to voxel indices."""
        relative = points_3d - self.bbox[0]
        voxel_idx = torch.floor(relative / self.size).int()
        valid_mask = (voxel_idx >= 0).all(dim=1) & (voxel_idx < self.dim).all(dim=1)
        return voxel_idx, valid_mask
    
    def to_linear_indices(self, voxel_indices):
        """Convert (i,j,k) indices to linear indices."""
        return (voxel_indices[:, 0] * self.dim[1] * self.dim[2] +
                voxel_indices[:, 1] * self.dim[2] + voxel_indices[:, 2])
    
    def xyz_2_index(self, xyz):
        """Convert world coordinates to voxel indices"""
        xyz = xyz.to(self.device)
        relative_positions = xyz - self.bbox[0]
        voxel_indices = torch.floor(relative_positions / self.size).int()
        return voxel_indices.tolist()
    
    def index_2_xyz(self, voxel_indices):
        """Convert voxel indices to world coordinates (voxel centers)"""
        voxel_indices = torch.tensor(voxel_indices, device=self.device)
        xyz = self.bbox[0] + (voxel_indices + 0.5) * self.size
        return xyz
    
    # ============================================
    # PROPERTIES (Computed Masks)
    # ============================================
    
    @property
    def voxel_states(self):
        return torch.sigmoid(self.voxel_log_odds)
    
    @property
    def free_mask(self):
        return self.voxel_states <= self.free_thres
    
    @property
    def occ_mask(self):
        return self.voxel_states >= self.occ_thres
    
    @property
    def free_mask_w_margin(self):
        """Free space with safety margin."""
        free_3d = self.free_mask.view(*self.dim).cpu().numpy()
        occ_3d = self.occ_mask.view(*self.dim).cpu().numpy()
        dilated_occ = binary_dilation(occ_3d, structure=self.occ_structure_element)
        safe_free = free_3d & ~dilated_occ
        return torch.from_numpy(safe_free).to(self.device).view(-1)
    
    @property
    def frontier_mask(self):
        """Frontier voxels: free space adjacent to unexplored."""
        unexplored_3d = self.unexplored_mask.view(*self.dim).cpu().numpy()
        dilated_unexplored = binary_dilation(unexplored_3d, structure=self.frontier_structure_element)
        dilated_unexplored = torch.from_numpy(dilated_unexplored).to(self.device).view(-1)
        return self.free_mask & dilated_unexplored
    
    @property
    def unknown_mask(self):
        """Alias for unexplored_mask for GUI compatibility"""
        return self.unexplored_mask
    
    def compute_boundary_mask(self) -> torch.Tensor:
        """
        Compute semantic boundary mask: voxels where different object classes meet.
        Uses 6-connectivity (face neighbors) for boundary detection.
        
        Returns:
            Boolean mask (N,) - True for boundary voxels
        """
        if not self.use_detector:
            return torch.zeros_like(self.unexplored_mask)
        
        # Return cached result if valid
        if self._boundary_cache_valid and self._cached_boundary_mask is not None:
            return self._cached_boundary_mask
        
        # Get semantic classes in 3D grid
        semantic_3d = self.voxel_semantic_class.view(*self.dim)
        boundary_mask = torch.zeros_like(self.voxel_semantic_class, dtype=torch.bool)
        
        # 6-connectivity: face neighbors only
        for axis in range(3):
            for direction in [-1, 1]:
                shifted = torch.roll(semantic_3d, shifts=direction, dims=axis)
                
                # Handle boundary conditions (edge voxels)
                if direction == -1:
                    if axis == 0:
                        shifted[-1, :, :] = semantic_3d[-1, :, :]
                    elif axis == 1:
                        shifted[:, -1, :] = semantic_3d[:, -1, :]
                    else:
                        shifted[:, :, -1] = semantic_3d[:, :, -1]
                else:
                    if axis == 0:
                        shifted[0, :, :] = semantic_3d[0, :, :]
                    elif axis == 1:
                        shifted[:, 0, :] = semantic_3d[:, 0, :]
                    else:
                        shifted[:, :, 0] = semantic_3d[:, :, 0]
                
                # Mark as boundary if neighbor has different class (and at least one is labeled)
                different_class = (semantic_3d != shifted)
                has_object = (semantic_3d >= 0) | (shifted >= 0)  # >= 0: labeled, -1: unlabeled
                boundary_mask |= (different_class & has_object).view(-1)
        
        # Cache the result
        self._cached_boundary_mask = boundary_mask
        return boundary_mask
    
    def compute_boundary_complexity(self) -> torch.Tensor:
        """
        Compute boundary complexity based on unique adjacent class count.
        Uses 6-connectivity (face neighbors) for junction detection.
        
        Counts how many unique object classes (non-zero) exist in the voxel's neighborhood:
        - 1: Only one class (no boundary between objects)
        - 2: Two object classes meet (object-object boundary)
        - 3+: Three or more classes meet (complex junction)
        
        Note: Background (class 0) is excluded to focus on object-object boundaries.
        Weight is directly n (number of classes), used by both I_explore and I_boundary.
        
        Returns:
            Unique class count per voxel (N,) - 1.0 to 4.0
        """
        if not self.use_detector:
            return torch.ones(len(self.voxel_centers), device=self.device)
        
        # Return cached result if valid
        if self._boundary_cache_valid and self._cached_boundary_complexity is not None:
            return self._cached_boundary_complexity
        
        # Get semantic classes in 3D grid
        semantic_3d = self.voxel_semantic_class.view(*self.dim)
        N = semantic_3d.numel()
        
        # Collect all neighbor classes (self + 6 neighbors = 7 total)
        neighbor_classes = [semantic_3d]
        
        # 6-connectivity: face neighbors only
        for axis in range(3):
            for direction in [-1, 1]:
                shifted = torch.roll(semantic_3d, shifts=direction, dims=axis)
                
                # Handle boundary conditions
                if direction == -1:
                    if axis == 0:
                        shifted[-1, :, :] = semantic_3d[-1, :, :]
                    elif axis == 1:
                        shifted[:, -1, :] = semantic_3d[:, -1, :]
                    else:
                        shifted[:, :, -1] = semantic_3d[:, :, -1]
                else:
                    if axis == 0:
                        shifted[0, :, :] = semantic_3d[0, :, :]
                    elif axis == 1:
                        shifted[:, 0, :] = semantic_3d[:, 0, :]
                    else:
                        shifted[:, :, 0] = semantic_3d[:, :, 0]
                
                neighbor_classes.append(shifted)
        
        # Stack: (7, N)
        all_classes = torch.stack([nc.view(-1) for nc in neighbor_classes], dim=0)
        
        # Efficient unique counting: use sorting
        sorted_classes, _ = torch.sort(all_classes, dim=0)
        
        # Count unique labeled classes per voxel (exclude unlabeled = -1)
        # This focuses on object-object boundaries only
        N = sorted_classes.shape[1]
        unique_counts = (sorted_classes[0] >= 0).float()  # >= 0: labeled classes
        
        for i in range(1, 7):
            is_new = (sorted_classes[i] != sorted_classes[i-1]) & (sorted_classes[i] >= 0)
            unique_counts += is_new.float()
        
        # Voxels with no objects get baseline of 1
        unique_counts[unique_counts == 0] = 1.0
        
        # Clamp to [1, 4] range (4+ classes = very complex junction)
        complexity = torch.clamp(unique_counts, min=1.0, max=4.0)
        
        # Cache the result
        self._cached_boundary_complexity = complexity
        self._boundary_cache_valid = True
        
        return complexity
    
    # ============================================
    # OCCUPANCY UPDATE (Frustum Culling)
    # ============================================
    
    def update_occupancy(self, dataframe):
        """
        Update occupancy using frustum culling method
        
        Args:
            dataframe: Dict containing depth, extrinsic, intrinsic
        """
        depth = dataframe["depth"]
        extrinsic = dataframe["extrinsic"]
        intrinsic = dataframe["intrinsic"]
        
        # Project voxel centers to image plane
        points_2d, points_depth = self._project_3d_points(extrinsic, intrinsic)
        
        # Frustum culling
        frustum_pass_mask, frustum_hit_mask = self._get_frustum_mask(
            points_2d, points_depth, depth
        )
        
        # Update log-odds with distance weighting
        camera_pos = extrinsic[:3, 3]
        
        # Pass voxels (free space)
        if frustum_pass_mask.any():
            dist_pass = torch.norm(
                self.voxel_centers[frustum_pass_mask] - camera_pos, dim=1
            )
            weight_pass = self._inverse_sensor_model(dist_pass)
            self.voxel_log_odds[frustum_pass_mask] -= 2.8 * weight_pass
        
        # Hit voxels (occupied)
        if frustum_hit_mask.any():
            dist_hit = torch.norm(
                self.voxel_centers[frustum_hit_mask] - camera_pos, dim=1
            )
            weight_hit = self._inverse_sensor_model(dist_hit)
            self.voxel_log_odds[frustum_hit_mask] += 2.8 * weight_hit
        
        # Clamp to prevent overflow
        self.voxel_log_odds = torch.clamp(self.voxel_log_odds, min=-4.5, max=4.5)
        
        # Update unexplored mask
        self.unexplored_mask[frustum_pass_mask | frustum_hit_mask] = False
    
    def _project_3d_points(self, extrinsic, intrinsic):
        """
        Project 3D voxel centers to image plane
        
        Args:
            extrinsic: (4, 4) camera extrinsic matrix
            intrinsic: (3, 3) camera intrinsic matrix
            
        Returns:
            points_2d: (N, 2) normalized image coordinates [0, 1]
            points_depth: (N,) depth values
        """
        # Transform to camera coordinates
        points_camera_hom = (extrinsic.inverse() @ self.points_3d_hom.T).T
        points_camera = points_camera_hom[:, :3]  # (N, 3)
        points_depth = points_camera[:, 2]  # (N)
        
        # Project to image plane
        points_image_hom = (intrinsic @ points_camera.T).T
        points_image = points_image_hom[:, :2] / points_image_hom[:, 2].unsqueeze(-1)
        
        return points_image, points_depth
    
    def _get_frustum_mask(self, points_2d, points_depth, depth_map):
        """
        Get visible points within view frustum using depth test
        
        Args:
            points_2d: (N, 2) normalized image coordinates
            points_depth: (N,) depth values
            depth_map: (H, W) depth map
            
        Returns:
            frustum_pass_mask: (N,) bool - rays pass through (free)
            frustum_hit_mask: (N,) bool - rays hit surface (occupied)
        """
        # Handle both (H, W) and (1, H, W) depth maps
        if depth_map.dim() == 3:
            depth_map = depth_map.squeeze(0)
        h, w = depth_map.shape
        
        # Front of camera
        front_mask = points_depth > 0
        
        # Scale to pixel coordinates
        points_2d[:, 0] = points_2d[:, 0] * w
        points_2d[:, 1] = points_2d[:, 1] * h
        
        # Bounds check
        x_indices = points_2d[:, 0]
        y_indices = points_2d[:, 1]
        valid_x = (x_indices >= 0) & (x_indices < w)
        valid_y = (y_indices >= 0) & (y_indices < h)
        valid_coords = valid_x & valid_y & front_mask
        
        # Get valid pixel coordinates
        valid_indices = valid_coords.nonzero(as_tuple=True)[0]
        valid_x_coords = x_indices[valid_coords].long()
        valid_y_coords = y_indices[valid_coords].long()
        
        # Depth test
        measured_depth = depth_map[valid_y_coords, valid_x_coords]
        voxel_depth = points_depth[valid_coords]
        
        # Pass: voxel before measured depth (free space)
        # Hit: voxel at measured depth (occupied)
        depth_tolerance = 0.1  # 10cm tolerance
        pass_indices = valid_indices[voxel_depth < measured_depth - depth_tolerance]
        hit_indices = valid_indices[
            torch.abs(voxel_depth - measured_depth) <= depth_tolerance
        ]
        
        # Create masks
        frustum_pass_mask = torch.zeros(len(points_depth), dtype=torch.bool, device=self.device)
        frustum_hit_mask = torch.zeros(len(points_depth), dtype=torch.bool, device=self.device)
        
        frustum_pass_mask[pass_indices] = True
        frustum_hit_mask[hit_indices] = True
        
        return frustum_pass_mask, frustum_hit_mask
    
    def _inverse_sensor_model(self, distance):
        """
        Inverse sensor model: closer observations are more reliable
        
        Args:
            distance: (N,) distances in meters
            
        Returns:
            weight: (N,) confidence weights
        """
        # Linear decay with distance
        weight = torch.clamp(1.0 - 0.1 * distance, min=0.0, max=1.0)
        return weight
    
    # ============================================
    # SEMANTIC INTEGRATION (Reuse existing code)
    # ============================================
    
    def update_semantic(self, rgb, depth, extrinsic, intrinsic):
        """
        Update semantic labels using object detection
        
        Reuses existing detection and projection pipeline
        
        Args:
            rgb: RGB image tensor
            depth: Depth map tensor
            extrinsic: Camera extrinsic matrix
            intrinsic: Camera intrinsic matrix
            
        Returns:
            segmentation_vis: Segmentation visualization for GUI (or None)
        """
        if not self.use_detector:
            return None
        
        # Convert to numpy for detector
        if rgb.dim() == 4:  # (1, 3, H, W)
            rgb_np = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:  # (3, H, W)
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        
        rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
        
        depth_np = depth.squeeze().cpu().numpy() if depth.dim() >= 2 else depth.cpu().numpy()
        extrinsic_np = extrinsic.cpu().numpy()
        intrinsic_np = intrinsic.cpu().numpy()
        
        # Object detection
        detections = self.detector.detect(rgb_np)
        
        if len(detections) == 0:
            return None, None
        
        # Create visualizations (separate detection and segmentation)
        from visualization.segmentation_vis import create_segmentation_visualization, create_detection_visualization
        segmentation_vis = create_segmentation_visualization(rgb_np, detections, alpha=0.4)
        detection_vis = create_detection_visualization(rgb_np, detections)
        
        # Project to voxels
        semantic_voxels = self.projector.project_to_voxels(
            detections, depth_np, extrinsic_np, intrinsic_np
        )
        
        if not semantic_voxels:
            return segmentation_vis, detection_vis
        
        # Update voxel semantic labels (majority voting + running average)
        for class_id, detections_list in semantic_voxels.items():
            for det_info in detections_list:
                linear_indices = det_info['linear_indices']
                confidence = det_info['confidence']
                class_name = det_info['class_name']
                
                # Track observations
                if class_name not in self.object_observation_stats:
                    self.object_observation_stats[class_name] = {
                        'total_observations': 0,
                        'unique_voxels': set()
                    }
                
                self.object_observation_stats[class_name]['total_observations'] += len(linear_indices)
                indices_list = linear_indices.tolist() if hasattr(linear_indices, 'tolist') else list(linear_indices)
                self.object_observation_stats[class_name]['unique_voxels'].update(indices_list)
                
                # Update voxels
                for linear_idx in linear_indices:
                    linear_idx_item = int(linear_idx)
                    
                    # Majority voting
                    self.class_vote_history[linear_idx_item][class_id] += 1
                    votes = self.class_vote_history[linear_idx_item]
                    dominant_class = max(votes.items(), key=lambda x: x[1])[0]
                    
                    self.voxel_semantic_class[linear_idx_item] = dominant_class
                    
                    # Running average confidence
                    count = self.voxel_object_count[linear_idx_item]
                    current_conf = self.voxel_class_confidence[linear_idx_item]
                    new_conf = (current_conf * count + confidence) / (count + 1)
                    self.voxel_class_confidence[linear_idx_item] = new_conf
                    self.voxel_object_count[linear_idx_item] += 1
        
        # Invalidate boundary cache since semantic info changed
        self._boundary_cache_valid = False
        
        # Update view diversity: record viewing direction for actually detected objects
        if hasattr(self, 'object_view_directions'):
            detected_class_names = [
                det.class_name for det in detections  # DetectionResult is a dataclass
            ]
            # Remove duplicates
            unique_classes = list(set(detected_class_names))
            camera_pose = torch.tensor(extrinsic_np, device=self.device, dtype=torch.float32)
            self.update_object_view_direction(camera_pose, unique_classes)
        
        return segmentation_vis, detection_vis
    
    # ============================================
    # MAIN UPDATE LOOP
    # ============================================
    
    def check_visible_direction(self, point):
        """
        Check the visible direction from a point (for ROI sampling)
        
        Finds free neighbor voxels and returns average direction toward them.
        
        Args:
            point: 3D point position
            
        Returns:
            view_direction: Average direction toward free neighbors, or None
        """
        voxel_index = torch.tensor(self.xyz_2_index(point), device=self.device)
        directions = torch.tensor(
            [[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]],
            device=self.device,
        )
        directions = directions[torch.any(directions != 0, dim=1)]
        neighbor_indices = voxel_index.unsqueeze(0) + directions

        in_bounds = torch.all(neighbor_indices >= 0, dim=1) & torch.all(
            neighbor_indices < self.dim, dim=1
        )
        neighbor_indices = neighbor_indices[in_bounds]
        free_neighbor_mask = self.free_mask.view(*self.dim)[
            neighbor_indices[:, 0], neighbor_indices[:, 1], neighbor_indices[:, 2]
        ]
        if torch.sum(free_neighbor_mask) > 0:
            free_neighbor_indices = neighbor_indices[free_neighbor_mask]
            free_neighbor_indices = free_neighbor_indices.view(-1, 3)
            free_neighbor_xyzs = self.index_2_xyz(free_neighbor_indices)
            view_directions = free_neighbor_xyzs - point.unsqueeze(0)
            view_directions = view_directions / view_directions.norm(
                dim=-1, keepdim=True
            )
            view_direction = torch.mean(view_directions, dim=0)
            return view_direction
        else:
            return None
    
    def update(self, dataframe, is_keyframe=True):
        """
        Update the map with a new observation (RGB-D, pose)
        
        Args:
            dataframe: Dict with rgb, depth, extrinsic, intrinsic
            is_keyframe: If True, run detection. If False, skip detection (path following)
            
        Returns:
            Tuple of (segmentation_vis, detection_vis) or (None, None)
        """
        # 1. Update occupancy (every frame - fast!)
        self.update_occupancy(dataframe)
        
        # 2. Update semantics (only at keyframes - slow!)
        if self.use_detector and is_keyframe:
            seg_vis, det_vis = self.update_semantic(
                dataframe["rgb"],
                dataframe["depth"],
                dataframe["extrinsic"],
                dataframe["intrinsic"]
            )
            # Cache for non-keyframes
            self.last_segmentation_vis = seg_vis
            self.last_detection_vis = det_vis
            return seg_vis, det_vis
        elif self.use_detector and not is_keyframe:
            # Non-keyframe: Reuse cached
            cached_seg = self.last_segmentation_vis if hasattr(self, 'last_segmentation_vis') else None
            cached_det = self.last_detection_vis if hasattr(self, 'last_detection_vis') else None
            return cached_seg, cached_det
        
        return None, None
    
    # ============================================
    # VOLUME STATISTICS (Adaptive Updates)
    # ============================================
    
    def compute_volume_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute volume statistics for each detected object class
        
        Returns:
            Dict mapping class_name to {voxel_count, volume_ratio}
        """
        if not self.use_detector:
            return {}
        
        stats = {}
        
        # Calculate total explored voxels
        explored_mask = self.free_mask | self.occ_mask
        total_explored = explored_mask.sum().item()
        
        if total_explored == 0:
            return stats
        
        # Count voxels per class (only explored voxels)
        for class_id, class_name in enumerate(self.detector.class_names):
            # Skip invalid class IDs (e.g., -1 for unlabeled voxels)
            if class_id < 0:
                continue
            class_mask = (self.voxel_semantic_class == class_id) & explored_mask
            object_voxels = class_mask.sum().item()
            
            if object_voxels > 0:
                volume_ratio = object_voxels / total_explored
                stats[class_name] = {
                    'voxel_count': object_voxels,
                    'volume_ratio': volume_ratio,
                }
        
        self.object_volume_stats = stats
        return stats
    
    def should_update_volume_stats(self, current_pose):
        """
        Check if scene has changed significantly for volume stats update
        
        Args:
            current_pose: Current camera pose (4, 4) matrix
            
        Returns:
            bool: Whether to update volume statistics
        """
        if not self.coarse_phase_complete:
            return False  # Only after coarse
        
        # First update after coarse always returns False (already updated)
        if self.last_volume_update_pose is None:
            return False
        
        # 1. Pose change detection
        pose_change = torch.norm(
            current_pose[:3, 3] - self.last_volume_update_pose[:3, 3]
        ).item()
        
        # 2. New object detection
        current_classes = set()
        for class_id, class_name in enumerate(self.detector.class_names):
            if (self.voxel_semantic_class == class_id).any():
                current_classes.add(class_name)
        
        new_classes = current_classes - self.last_detected_classes
        
        # Thresholds
        pose_threshold = 2.0  # meters
        new_class_threshold = 2  # new classes
        
        should_update = pose_change > pose_threshold or len(new_classes) >= new_class_threshold
        
        if should_update:
            from utils.common import TextColors
            print(f"{TextColors.YELLOW}[ObjectSemanticMap] Scene changed significantly:{TextColors.RESET}")
            print(f"  Pose change: {pose_change:.2f}m (threshold: {pose_threshold}m)")
            print(f"  New classes: {len(new_classes)} (threshold: {new_class_threshold})")
            print(f"  Classes: {new_classes}")
        
        return should_update
    
    # ============================================
    # OBJECT COMPLETION-DRIVEN PLANNING
    # ============================================
    
    def get_object_completions(self) -> List[Dict]:
        """
        Compute completion metrics for each detected object.
        
        Completion is based on:
        - Coverage: fraction of object voxels that have been observed
        - Confidence: average confidence across observed voxels
        
        Returns:
            List of dicts with:
            - class_name: object class name
            - class_id: object class id
            - completion: overall completion score (0-1)
            - avg_confidence: average confidence
            - voxel_count: total voxels for this class
            - low_conf_indices: linear indices of low-confidence voxels
            - centroid: center position of the object
        """
        if not self.use_detector:
            return []
        
        completions = []
        
        for class_id, class_name in enumerate(self.detector.class_names):
            class_mask = self.voxel_semantic_class == class_id
            voxel_count = class_mask.sum().item()
            
            if voxel_count == 0:
                continue
            
            # Get confidence for this class
            class_confidences = self.voxel_class_confidence[class_mask]
            avg_confidence = class_confidences.mean().item()
            
            # Observed = has confidence > 0
            observed_mask = class_confidences > 0
            observed_count = observed_mask.sum().item()
            coverage = observed_count / voxel_count if voxel_count > 0 else 0
            
            # Completion = coverage × avg_confidence
            completion = coverage * avg_confidence
            
            # Find low-confidence voxels (for targeting)
            class_indices = torch.where(class_mask)[0]
            low_conf_threshold = 0.5
            low_conf_mask = class_confidences < low_conf_threshold
            low_conf_indices = class_indices[low_conf_mask]
            
            # Compute centroid
            class_centers = self.voxel_centers[class_mask]
            centroid = class_centers.mean(dim=0)
            
            completions.append({
                'class_name': class_name,
                'class_id': class_id,
                'completion': completion,
                'avg_confidence': avg_confidence,
                'voxel_count': voxel_count,
                'low_conf_indices': low_conf_indices,
                'centroid': centroid,
            })
        
        # Sort by completion (highest first)
        completions.sort(key=lambda x: x['completion'], reverse=True)
        
        return completions
    
    def update_object_view_direction(self, camera_pose: torch.Tensor, visible_class_names: List[str]):
        """
        Record viewing direction for each observed object class.
        
        Args:
            camera_pose: (4, 4) camera extrinsic matrix
            visible_class_names: List of class names visible in this view
        """
        if not self.use_detector:
            return
        
        camera_pos = camera_pose[:3, 3].to(self.device)
        
        for class_name in visible_class_names:
            # Get class ID
            if class_name not in self.detector.class_names:
                continue
            class_id = self.detector.class_names.index(class_name)
            if class_id < 0:  # -1 means unlabeled
                continue
            
            # Get object centroid
            class_mask = self.voxel_semantic_class == class_id
            if class_mask.sum() == 0:
                continue
            
            centroid = self.voxel_centers[class_mask].mean(dim=0)
            
            # Compute viewing direction (camera → object)
            direction = centroid - camera_pos
            direction = direction / (direction.norm() + 1e-8)  # Normalize to unit vector
            
            # Store as CPU tensor
            self.object_view_directions[class_name].append(direction.cpu())
    
    def compute_view_diversity(self, class_name: str) -> float:
        """
        Compute viewing direction diversity for an object class using Spherical Variance.
        
        The Mean Resultant Length (R) measures concentration of unit vectors:
        - R ≈ 1: All directions are similar (low diversity)
        - R ≈ 0: Directions are uniformly distributed (high diversity)
        
        Spherical Variance = 1 - R
        
        Args:
            class_name: Name of the object class
            
        Returns:
            Diversity score in [0, 1], or 0.0 if insufficient observations
        """
        directions = self.object_view_directions.get(class_name, [])
        
        if len(directions) < 2:
            return 0.0  # Not enough observations to compute diversity
        
        # Stack to tensor
        dir_tensor = torch.stack(directions)  # (N, 3)
        
        # Mean resultant vector
        mean_dir = dir_tensor.mean(dim=0)
        R = mean_dir.norm().item()
        
        # Spherical variance = 1 - R
        diversity = 1.0 - R
        return diversity
    
    def get_best_fine_start_target(self, min_voxels: int = 100) -> Optional[torch.Tensor]:
        """
        Find the optimal starting point for fine exploration.
        
        Strategy (Volume-Based):
        1. Exclude structural elements (wall, floor, ceiling, etc.)
        2. Find object with largest volume (most voxels)
        3. Target its low-confidence voxels for additional observation
        
        Rationale: Largest objects are most reliably detected from 12 coarse views.
        Their low-confidence areas indicate regions needing better viewing angles.
        
        Args:
            min_voxels: Minimum voxel count to consider an object (default 100)
        
        Returns:
            Target position (3,) tensor, or None if no suitable target
        """
        completions = self.get_object_completions()
        
        if not completions:
            print(f"{TextColors.YELLOW}[Fine Start] No objects detected{TextColors.RESET}")
            return None
        
        # Exclude structural elements (not useful for reconstruction quality)
        excluded_classes = {'wall', 'floor', 'ceiling', 'door', 'window', 'column', 'pillar'}
        objects = [c for c in completions if c['class_name'].lower() not in excluded_classes]
        
        if not objects:
            print(f"{TextColors.YELLOW}[Fine Start] Only structural elements detected, using largest{TextColors.RESET}")
            objects = completions
        
        # Sort by VOLUME (voxel_count), not completion
        objects.sort(key=lambda x: x['voxel_count'], reverse=True)
        
        # Filter by minimum voxel count
        valid_objects = [o for o in objects if o['voxel_count'] >= min_voxels]
        
        if not valid_objects:
            valid_objects = [objects[0]]  # Use largest even if below threshold
            print(f"{TextColors.YELLOW}[Fine Start] No objects above {min_voxels} voxels, using largest{TextColors.RESET}")
        
        # Select largest object
        largest_object = valid_objects[0]
        
        # Find low-confidence voxels of this object
        low_conf_indices = largest_object['low_conf_indices']
        
        if len(low_conf_indices) == 0:
            # All voxels have high confidence, target centroid
            target = largest_object['centroid']
            print(f"{TextColors.CYAN}[Fine Start] '{largest_object['class_name']}' fully confident, targeting centroid{TextColors.RESET}")
        else:
            # Target low-confidence voxels (need more observation)
            target_centers = self.voxel_centers[low_conf_indices]
            target = target_centers.mean(dim=0)
            print(f"{TextColors.CYAN}[Fine Start] Targeting {len(low_conf_indices)} low-conf voxels of '{largest_object['class_name']}'{TextColors.RESET}")
        
        print(f"  Object: {largest_object['class_name']} ({largest_object['voxel_count']} voxels)")
        print(f"  Target position: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
        
        return target
    
    def compute_adaptive_weights(self, gaussian_map=None) -> Dict[str, float]:
        """
        Compute geometric complexity (D_c) for each object class.
        
        D_c uses Angular Dispersion of 2DGS normals:
        - D_c = 0: All normals aligned (planar surface)
        - D_c = 1: Normals dispersed (complex geometry)
        
        Returns:
            Raw D_c values per class (0.0 - 1.0)
        """
        if not self.object_volume_stats:
            return {}
        
        adaptive_weights = {}
        
        # Compute geometric complexity (raw D_c values)
        class_complexity = self._compute_geometric_complexity(gaussian_map) if gaussian_map else {}
        
        # Return raw D_c values directly
        for class_name in self.object_volume_stats.keys():
            D_c = class_complexity.get(class_name, 0.5)
            adaptive_weights[class_name] = D_c
        
        return adaptive_weights

    
    def _compute_geometric_complexity(self, gaussian_map) -> Dict[str, float]:
        """
        Compute geometric complexity for each class using Angular Dispersion.
        
        Angular Dispersion measures how spread out the normal vectors are:
        - D_c = 0: All normals aligned (planar surface)
        - D_c = 1: Normals completely dispersed (complex geometry)
        
        Formula: D_c = 1 - mean(|nᵢ · n̄|)
        where n̄ is the normalized mean normal of the class.
        
        Args:
            gaussian_map: GaussianMap with explicit normals from 2DGS
        
        Returns:
            Geometric complexity D_c (0.0 - 1.0) per class
        """
        class_complexity = {}
        
        if gaussian_map is None or len(gaussian_map._means) == 0:
            return class_complexity
        
        try:
            # Get Gaussian positions and normals from 2DGS
            gaussian_means = gaussian_map.get_means.detach()
            gaussian_normals = gaussian_map.get_normals.detach()
            
            # Normalize normals (ensure unit vectors)
            gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)
            
            # Map Gaussians to voxels to get their class
            voxel_indices = self._voxelize(gaussian_means)
            valid_mask = voxel_indices[1]  # in-bounds mask
            voxel_idx = voxel_indices[0][valid_mask]
            
            # Get linear indices
            linear_indices = self.to_linear_indices(voxel_idx)
            gaussian_classes = self.voxel_semantic_class[linear_indices]
            valid_normals = gaussian_normals[valid_mask].cpu().numpy()
            gaussian_classes_np = gaussian_classes.cpu().numpy()
            
            # Compute Angular Dispersion per class
            import numpy as np
            
            for class_name, stats in self.object_volume_stats.items():
                class_id = self.detector.class_names.index(class_name) if class_name in self.detector.class_names else -1
                if class_id < 0:
                    continue
                
                class_mask = gaussian_classes_np == class_id
                class_normals = valid_normals[class_mask]
                
                if len(class_normals) < 1:
                    continue  # No samples → not included → D_c defaults to 0.0 → final D_c = 1.0
                
                # Angular Dispersion: D_c = 1 - mean(|n · n̄|)
                # Step 1: Compute mean normal
                mean_normal = np.mean(class_normals, axis=0)
                mean_norm = np.linalg.norm(mean_normal)
                
                if mean_norm < 1e-6:
                    # Mean is near zero → normals cancel out → high complexity
                    D_c = 1.0
                else:
                    # Normalize mean normal
                    mean_normal = mean_normal / mean_norm
                    
                    # Step 2: Compute dot products with mean
                    dot_products = np.abs(class_normals @ mean_normal)
                    
                    # Step 3: D_c = 1 - average alignment
                    D_c = 1.0 - np.mean(dot_products)
                
                # Clamp to [0, 1] for safety
                D_c = float(np.clip(D_c, 0.0, 1.0))
                class_complexity[class_name] = D_c
            
        except Exception as e:
            print(f"  [Warning] Geometric complexity computation failed: {e}")
        
        return class_complexity
    
    def _compute_voxel_object_scores(self):
        """
        Compute importance score for each voxel based on its object class
        
        Updates self.voxel_object_scores with scores 0.0-1.0
        """
        self.voxel_object_scores = torch.zeros(len(self.voxel_centers), device=self.device)
        
        # Use adaptive weights if available
        if self.use_adaptive and self.coarse_phase_complete and self.object_volume_stats:
            weights = self.compute_adaptive_weights()
        else:
            # Before coarse phase: uniform weights for all classes
            weights = {name: 0.5 for name in self.detector.class_names}
        
        # Vectorized assignment: process all voxels at once
        # Get all class IDs
        class_ids = self.voxel_semantic_class
        
        # Create weight lookup tensor
        num_classes = len(self.detector.class_names)
        weight_tensor = torch.full((num_classes,), 0.5, device=self.device)
        for class_id, class_name in enumerate(self.detector.class_names):
            weight_tensor[class_id] = weights.get(class_name, 0.5)
        
        # Compute scores for all voxels at once
        # mask for labeled voxels (class_id >= 0, -1 means unlabeled)
        labeled_mask = class_ids >= 0
        
        # Get weights and confidence for labeled voxels
        labeled_weights = weight_tensor[class_ids[labeled_mask]]
        labeled_confidence = self.voxel_class_confidence[labeled_mask]
        
        # Score = weight × confidence
        self.voxel_object_scores[labeled_mask] = labeled_weights * labeled_confidence
    
    # ============================================
    # UTILITY UPDATE (GaussianMap Integration)
    # ============================================
    
    def update_utility(self, gaussian_map, use_confidence=True, confidence_thres=0.3):
        """
        Update ROI mask: frontiers + low-confidence gaussians
        
        Args:
            gaussian_map: GaussianMap instance
            use_confidence: Whether to include low-confidence gaussians
            confidence_thres: Confidence threshold
        """
        # Base ROI: frontier voxels
        raw_roi_mask = self.frontier_mask
        
        # Optional: Add low-confidence gaussians
        if use_confidence and gaussian_map is not None:
            gaussian_mean = gaussian_map.get_means.detach()
            gaussian_normal = gaussian_map.get_normals.detach()
            gaussian_confidence = gaussian_map.get_confidences.detach()
            gaussian_opacity = gaussian_map.get_opacities.detach()
            
            # Voxelize gaussian positions
            voxel_indices, valid_mask = self._voxelize(gaussian_mean)
            voxel_indices = voxel_indices[valid_mask]
            gaussian_normal = gaussian_normal[valid_mask]
            gaussian_confidence = gaussian_confidence[valid_mask]
            gaussian_opacity = gaussian_opacity[valid_mask]
            
            # Get high-opacity low-confidence gaussians
            confidence_mask = (gaussian_confidence < confidence_thres) & (gaussian_opacity > 0.7)
            voxel_indices = voxel_indices[confidence_mask]
            gaussian_normal = gaussian_normal[confidence_mask]
            
            # Convert to linear indices
            linear_indices = self.to_linear_indices(voxel_indices).long()
            
            # Count gaussians per voxel
            voxel_sum = torch.zeros(
                torch.prod(self.dim).item(), dtype=torch.int64, device=self.device
            )
            voxel_sum = voxel_sum.scatter_add(
                0, linear_indices, torch.ones_like(linear_indices, dtype=torch.int64)
            )
            
            # Voxels with low-confidence gaussians
            voxel_normal_sum = torch.zeros(
                (torch.prod(self.dim).item(), 3), device=self.device
            )
            voxel_normal_sum = voxel_normal_sum.scatter_add(
                0,
                linear_indices.unsqueeze(1).expand(-1, 3),
                gaussian_normal,
            )
            
            # Add to ROI
            update_mask = voxel_sum > self.min_gaussian_per_voxel
            mean_normals = voxel_normal_sum / voxel_sum.unsqueeze(1).clamp(min=1)
            self.voxel_normal[update_mask] = torch.nn.functional.normalize(
                mean_normals[update_mask], dim=-1
            )
            raw_roi_mask = raw_roi_mask | update_mask
        
        # Apply ROI filtering
        self.roi_mask = self._get_roi_mask(raw_roi_mask)
        
        # Compute voxel object scores (ONLY during coarse phase)
        if self.use_detector and not self.coarse_phase_complete:
            self._compute_voxel_object_scores()
    
    def _get_roi_mask(self, raw_roi_mask):
        """
        Filter ROI mask to free space only
        
        Args:
            raw_roi_mask: Raw ROI before filtering
            
        Returns:
            Filtered ROI mask
        """
        # ROI must be in free space
        return raw_roi_mask & self.free_mask
    
    def cal_visible_mask(self, extrinsic, intrinsic, depth):
        """
        Get mask for all visible voxels at given view
        
        Args:
            extrinsic: (4, 4) camera extrinsic matrix
            intrinsic: (3, 3) camera intrinsic matrix
            depth: (H, W) depth map
            
        Returns:
            visible_mask: (N,) bool mask of visible voxels
        """
        points_2d, points_depth = self._project_3d_points(extrinsic, intrinsic)
        frustum_mask, _ = self._get_frustum_mask(points_2d, points_depth, depth)
        return frustum_mask
    
    # ============================================
    # PATH PLANNING WITH SEMANTIC COST
    # ============================================
    
    def update_graph(self, robot_space=None):
        """
        Update graph for path planning
        
        Args:
            robot_space: Optional robot-occupied space mask
        """
        if not hasattr(self, 'graph'):
            # Initialize graph on first call
            from mapping.voxel_map import VoxelGraph
            self.graph = VoxelGraph(
                self.size.cpu().numpy(),
                self.dim.cpu().numpy(),
                self.voxel_indices.cpu().numpy()
            )
        
        # Planning mask: free space with safety margin
        planning_mask = self.free_mask_w_margin
        if robot_space is not None:
            planning_mask = planning_mask | robot_space
        
        # Update graph with current free space
        self.graph.update_graph(planning_mask.cpu().numpy())
    
    def find_path_semantic(self, start_pos, goal_pos, use_semantic=True):
        """
        Find path from start to goal using A* with semantic cost
        
        Args:
            start_pos: (3,) start position in world coordinates
            goal_pos: (3,) goal position in world coordinates
            use_semantic: Whether to use semantic cost (only after coarse phase)
            
        Returns:
            path: List of voxel positions, or None if no path found
        """
        import heapq
        import numpy as np
        
        # Convert positions to voxel indices
        start_voxel, start_valid = self._voxelize(start_pos.unsqueeze(0))
        goal_voxel, goal_valid = self._voxelize(goal_pos.unsqueeze(0))
        
        if not start_valid[0] or not goal_valid[0]:
            return None
        
        start_idx = tuple(start_voxel[0].cpu().numpy())
        goal_idx = tuple(goal_voxel[0].cpu().numpy())
        
        # A* search
        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        came_from = {}
        g_score = {start_idx: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_idx:
                return self._reconstruct_path(came_from, current)
            
            # Get neighbors from graph
            if current not in self.graph.dense_graph:
                continue
            
            for neighbor_data in self.graph.dense_graph[current]:
                if isinstance(neighbor_data, tuple):
                    neighbor, base_distance = neighbor_data
                else:
                    neighbor = neighbor_data
                    base_distance = np.linalg.norm(
                        np.array(current) - np.array(neighbor)
                    )
                
                neighbor = tuple(neighbor)
                
                # Compute cost with semantic bonus
                edge_cost = self._compute_edge_cost(
                    current, neighbor, base_distance, use_semantic
                )
                
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Heuristic: Euclidean distance to goal
                    h = np.linalg.norm(np.array(neighbor) - np.array(goal_idx))
                    f_score = tentative_g + h
                    
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None  # No path found
    
    def _compute_edge_cost(self, from_voxel, to_voxel, base_distance, use_semantic):
        """
        Compute edge cost with semantic bonus
        
        Args:
            from_voxel: (i, j, k) tuple
            to_voxel: (i, j, k) tuple
            base_distance: Euclidean distance
            use_semantic: Whether to apply semantic cost
            
        Returns:
            cost: Edge cost (lower = better)
        """
        cost = base_distance
        
        # Semantic bonus (only after coarse phase)
        if use_semantic and self.use_detector and self.coarse_phase_complete:
            # Convert to linear index
            to_linear = (
                to_voxel[0] * self.dim[1].item() * self.dim[2].item() +
                to_voxel[1] * self.dim[2].item() +
                to_voxel[2]
            )
            
            # Get object importance score for this voxel
            importance = self.voxel_object_scores[to_linear].item()
            
            # Bonus: Higher importance = Lower cost
            # Scale: importance 0-1, bonus 0-0.5 * distance
            semantic_bonus = importance * 0.5 * base_distance
            cost -= semantic_bonus
        
        return max(cost, 0.01)  # Ensure positive cost
    
    def _reconstruct_path(self, came_from, current):
        """
        Reconstruct path from A* search
        
        Args:
            came_from: Dict mapping voxel to previous voxel
            current: Goal voxel
            
        Returns:
            path: List of voxel center positions (world coordinates)
        """
        path_indices = [current]
        while current in came_from:
            current = came_from[current]
            path_indices.append(current)
        
        path_indices.reverse()
        
        # Convert voxel indices to world positions
        path_positions = []
        for idx in path_indices:
            linear_idx = (
                idx[0] * self.dim[1].item() * self.dim[2].item() +
                idx[1] * self.dim[2].item() +
                idx[2]
            )
            path_positions.append(self.voxel_centers[linear_idx])
        
        return path_positions
    
    def in_free_space(self, positions):
        """
        Check if positions are in free space
        
        Args:
            positions: (N, 3) tensor of positions in world coordinates
            
        Returns:
            free_mask: (N,) bool tensor indicating if positions are in free space
        """
        # Voxelize positions
        voxel_indices, valid_mask = self._voxelize(positions)
        
        # Initialize result
        result = torch.zeros(len(positions), dtype=torch.bool, device=self.device)
        
        if not valid_mask.any():
            return result
        
        # Get linear indices for valid positions
        valid_indices = voxel_indices[valid_mask]
        linear_indices = self.to_linear_indices(valid_indices)
        
        # Check free space with margin
        free_w_margin = self.free_mask_w_margin
        is_free = free_w_margin[linear_indices]
        
        # Update result for valid positions
        result[valid_mask] = is_free
        
        return result


