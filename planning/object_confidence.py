"""
SOAR-GS NBV planner: U(v) = α·I_explore + β·I_quality + γ·I_boundary

I_explore: Boundary-weighted exploration (only boundary-adjacent unexplored contribute)
I_quality: Quality utility with D_c weighting and view diversity multiplier
I_boundary: Observed boundary refinement (mean complexity, no propagation)
"""

from .plan_base import *
from utils.common import TextColors
from utils.operations import GaussianRenderer


class ObjectConfidencePlanner(PlanBase):
    
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.render_ratio = cfg.get('render_ratio', 0.5)
        self.alpha_explore = cfg.get('alpha_explore', 1.0)
        self.beta_quality = cfg.get('beta_quality', 1.0)
        self.gamma_boundary = cfg.get('gamma_boundary', 0.4)
        self.pixel_stride = cfg.get('pixel_stride', 4)
        self.d0 = cfg.get('d0', 3.0)
        self.use_D_c = cfg.get('use_D_c', True)
        self.use_boundary_propagation = cfg.get('use_boundary_propagation', True)
        self.propagation_hops = cfg.get('propagation_hops', 2)
        
        print(f"[ObjectConfidencePlanner] Initialized")
        print(f"  U(v) = α·I_explore + β·I_quality + γ·I_boundary")
        print(f"  α={self.alpha_explore}, β={self.beta_quality}, γ={self.gamma_boundary}, d0={self.d0}m")
        print(f"  use_D_c={self.use_D_c}, boundary_propagation={self.use_boundary_propagation}")
    
    @torch.no_grad()
    def cal_utility(self, gaussian_map, voxel_map, candidates, simulator):
        """Calculate utility for view candidates: U(v) = α·I_explore + β·I_quality + γ·I_boundary"""
        t_utility = 0
        
        render_resolution = np.round(self.render_ratio * simulator.resolution).astype(int)
        h, w = render_resolution
        depth_range = simulator.depth_range
        extrinsics = candidates.to(self.device)
        intrinsics = repeat(simulator.intrinsic, "h w -> v h w", v=len(candidates)).to(self.device)
        
        renderer = GaussianRenderer(
            extrinsics, intrinsics,
            gaussian_map.get_attr(), gaussian_map.background_color,
            (gaussian_map.scene_near, gaussian_map.scene_far),
            (h, w), self.device,
        )
        
        explore_util = torch.zeros(len(candidates))
        rendered_data = []
        actual_depths = []
        visible_masks = []  # Store for I_boundary reuse
        require_valid_mask = simulator.has_missing_surface
        
        # Pre-compute boundary data for I_boundary
        boundary_mask = None
        boundary_complexity = None
        propagated_weights = None
        unexp_mask_global = voxel_map.unexplored_mask
        
        if hasattr(voxel_map, 'compute_boundary_complexity'):
            try:
                boundary_mask = voxel_map.compute_boundary_mask()
                boundary_complexity = voxel_map.compute_boundary_complexity()
                
                if self.use_boundary_propagation and unexp_mask_global.sum() > 0:
                    propagated_weights = self._propagate_complexity_to_unexplored(
                        boundary_complexity, boundary_mask, unexp_mask_global, voxel_map
                    )
            except (AttributeError, RuntimeError) as e:
                print(f"  [I_boundary] Warning: Boundary computation failed: {e}")
        
        for i in tqdm(range(len(extrinsics)), desc=f" {TextColors.CYAN}Evaluate Candidates{TextColors.RESET}"):
            t_utility_start = time.time()
            
            (rgb, depth, normal, opacity, d2n, confidence, importance, count, _) = renderer.render_view(i)
            confidences = confidence[0]
            depths = depth[0]
            
            rendered_data.append({'depth': depths.clone(), 'confidence': confidences.clone()})
            
            t_simulator_start = time.time()
            sim_result = simulator.simulate(extrinsics[i].cpu(), require_gt=True, return_valid_mask=require_valid_mask)
            actual_depth = sim_result['depth'].squeeze(0).cpu()
            actual_depth = torch.tensor(cv2.resize(actual_depth.numpy(), render_resolution, interpolation=cv2.INTER_NEAREST))
            
            if require_valid_mask:
                valid_mask = sim_result['valid_mask']
                valid_mask = cv2.resize(valid_mask.astype(np.uint8), render_resolution, interpolation=cv2.INTER_NEAREST)
                valid_mask = torch.tensor(valid_mask).bool()
            else:
                valid_mask = torch.ones(*render_resolution).bool()
            t_simulator = time.time() - t_simulator_start
            
            actual_depths.append(actual_depth)
            
            # I_explore: Pure exploration coverage (visible unexplored / total unexplored)
            depth_voxel = depths.clone()
            depth_voxel[depth_voxel < 0.001] = 10000
            depth_voxel = torch.clamp(depth_voxel, min=depth_range[0], max=depth_range[1])
            depth_voxel[~valid_mask] = -1.0
            
            visible_mask = voxel_map.cal_visible_mask(extrinsics[i], intrinsics[i], depth_voxel)
            visible_masks.append(visible_mask)  # Store for I_boundary
            unexp_mask = voxel_map.unexplored_mask
            visible_unexp_mask = visible_mask & unexp_mask
            
            total_unexplored = torch.sum(unexp_mask).float()
            if total_unexplored > 0:
                if propagated_weights is not None:
                    # Boundary-weighted: only boundary-adjacent unexplored contribute
                    visible_weights = propagated_weights[visible_unexp_mask]
                    weighted_unexplored = visible_weights.sum()
                    explore_util[i] = weighted_unexplored / total_unexplored
                else:
                    # Fallback: simple count ratio
                    unexplored_count = visible_unexp_mask.sum().float()
                    explore_util[i] = unexplored_count / total_unexplored
            else:
                explore_util[i] = 0.0
            
            t_utility += time.time() - t_utility_start - t_simulator
        
        explore_util[torch.isnan(explore_util)] = 0.0

        I_explore_raw = explore_util.cpu()
        I_explore = self._normalize(I_explore_raw)

        I_quality_raw = self._compute_quality_utility(
            gaussian_map, voxel_map, extrinsics, intrinsics, rendered_data, actual_depths, depth_range
        )
        I_quality = self._normalize(I_quality_raw)
        
        # I_boundary: Observed boundary refinement ONLY (no propagation)
        # Measures complexity of already-observed boundaries for quality refinement
        # Propagation now handled by I_explore
        I_boundary = torch.zeros(len(extrinsics))
        I_boundary_raw = torch.zeros(len(extrinsics))
        
        if boundary_mask is not None and boundary_complexity is not None:
            # Use direct n instead of pairwise formula n(n-1)/2+1
            
            for i in range(len(extrinsics)):
                # Observed boundary only (no propagated_contrib)
                visible_boundary = visible_masks[i] & boundary_mask
                # Use mean instead of sum to remove FOV/distance bias
                if visible_boundary.sum() > 0:
                    I_boundary_raw[i] = boundary_complexity[visible_boundary].mean()
                else:
                    I_boundary_raw[i] = 0.0
            
            I_boundary = self._normalize(I_boundary_raw.cpu())

        # U(v) = α·I_explore + β·I_quality + γ·I_boundary
        alpha = self.alpha_explore
        beta = self.beta_quality
        gamma = self.gamma_boundary
        utility = alpha * I_explore + beta * I_quality + gamma * I_boundary

        # Enhanced logging
        print(f"\n  [Utility Components]")
        print(f"    I_explore  - raw: [{I_explore_raw.min():.4f}, {I_explore_raw.max():.4f}] → norm: [{I_explore.min():.4f}, {I_explore.max():.4f}]")
        print(f"    I_quality  - raw: [{I_quality_raw.min():.4f}, {I_quality_raw.max():.4f}] → norm: [{I_quality.min():.4f}, {I_quality.max():.4f}]")
        print(f"    I_boundary - raw: [{I_boundary_raw.min():.4f}, {I_boundary_raw.max():.4f}] → norm: [{I_boundary.min():.4f}, {I_boundary.max():.4f}]")
        print(f"  [Weights] α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
        
        # Store for later access by plan()
        self._last_utility_components = {
            'I_explore': I_explore,
            'I_quality': I_quality,
            'I_boundary': I_boundary,
            'utility': utility
        }
        
        return utility, t_utility
    
    def _normalize(self, x):
        """Max normalization to [0, 1]"""
        x_max = x.max()
        if x_max < 1e-8:
            return torch.zeros_like(x)
        return x / x_max
    
    def _propagate_complexity_to_unexplored(self, boundary_complexity, boundary_mask, unexplored_mask, voxel_map):
        device = boundary_complexity.device
        hops = self.propagation_hops
        
        complexity_3d = torch.zeros(*voxel_map.dim, device=device)
        boundary_indices = torch.where(boundary_mask)[0]
        
        if len(boundary_indices) > 0:
            dim1_dim2 = voxel_map.dim[1] * voxel_map.dim[2]
            x = boundary_indices // dim1_dim2
            remainder = boundary_indices % dim1_dim2
            y = remainder // voxel_map.dim[2]
            z = remainder % voxel_map.dim[2]
            
            # Use direct n instead of pairwise formula n(n-1)/2+1
            complexity_3d[x, y, z] = boundary_complexity[boundary_indices]
        
        max_complexity = complexity_3d.clone()
        for _ in range(hops):
            current = max_complexity.clone()
            for axis in range(3):
                rolled_pos = torch.roll(current, shifts=1, dims=axis)
                rolled_neg = torch.roll(current, shifts=-1, dims=axis)
                
                if axis == 0:
                    rolled_pos[0, :, :] = 0.0
                    rolled_neg[-1, :, :] = 0.0
                elif axis == 1:
                    rolled_pos[:, 0, :] = 0.0
                    rolled_neg[:, -1, :] = 0.0
                else:
                    rolled_pos[:, :, 0] = 0.0
                    rolled_neg[:, :, -1] = 0.0
                    
                max_complexity = torch.maximum(max_complexity, rolled_pos)
                max_complexity = torch.maximum(max_complexity, rolled_neg)
        
        propagated_values = max_complexity.view(-1)
        
        # Only boundary-adjacent unexplored get weight (non-boundary = 0)
        # This focuses I_explore on boundary regions where floaters occur
        final_weights = torch.zeros(len(boundary_complexity), device=device)
        update_mask = (propagated_values > 0) & unexplored_mask
        final_weights[update_mask] = propagated_values[update_mask]
        
        return final_weights
    
    def _compute_quality_utility(self, gaussian_map, voxel_map, extrinsics, intrinsics, rendered_data, actual_depths, depth_range=None):
        """Compute I_quality = mean[(1+D_c) × (1-conf) × dist_weight]
        
        D_c: Geometric complexity (Angular Dispersion), range [0, 1]
        - D_c = 0: planar surface (all normals aligned)
        - D_c = 1: complex geometry (normals dispersed)
        """
        N = len(extrinsics)
        quality_utility = torch.zeros(N)
        d0 = self.d0
        
        use_semantic = (
            hasattr(voxel_map, 'voxel_semantic_class') and
            hasattr(voxel_map, 'coarse_phase_complete') and 
            voxel_map.coarse_phase_complete
        )
        
        D_c_weights = {}
        if use_semantic and hasattr(voxel_map, 'compute_adaptive_weights'):
            D_c_weights = voxel_map.compute_adaptive_weights(gaussian_map)
        
        for i in range(N):
            actual_depth = actual_depths[i]
            confidence = rendered_data[i]['confidence']
            H, W = actual_depth.shape
            
            max_depth = depth_range[1] if depth_range else 10.0
            valid_mask = (actual_depth > 0.001) & (actual_depth < max_depth)
            if valid_mask.sum() == 0:
                continue
            
            if use_semantic and hasattr(voxel_map, 'detector'):
                pixel_intrinsic = intrinsics[i].clone()
                pixel_intrinsic[0, 0] *= W
                pixel_intrinsic[1, 1] *= H
                pixel_intrinsic[0, 2] *= W
                pixel_intrinsic[1, 2] *= H
                
                class_stats = self._compute_class_uncertainty_area(
                    actual_depth, confidence, valid_mask, extrinsics[i], pixel_intrinsic, voxel_map
                )
                
                class_qualities = []
                for class_id, stats in class_stats.items():
                    if self.use_D_c and class_id >= 0 and class_id < len(voxel_map.detector.class_names):
                        class_name = voxel_map.detector.class_names[class_id]
                        D_c = D_c_weights.get(class_name, 0.0)  # [0, 1]
                        
                        # View diversity: penalize views from similar directions
                        if hasattr(voxel_map, 'compute_view_diversity'):
                            diversity = voxel_map.compute_view_diversity(class_name)
                            diversity_multiplier = 1.0 + 0.3 * (1.0 - diversity)
                        else:
                            diversity_multiplier = 1.0
                        
                        effective_D_c = D_c * diversity_multiplier
                    else:
                        D_c = 0.0  # No complexity info
                        effective_D_c = 0.0  # Baseline: no complexity bonus
                    
                    uncertainty = stats['uncertainty']
                    mean_depth = stats.get('mean_depth', 2.0)
                    
                    # D_c-Adaptive d0: complex objects require closer observation
                    # D_c=0 (simple) → d0, D_c=1 (complex) → d0/2
                    adaptive_d0 = d0 * (1.0 - 0.5 * D_c)
                    dist_weight = 1.0 / (1.0 + (mean_depth / adaptive_d0) ** 2)
                    
                    quality = (1 + effective_D_c) * uncertainty * dist_weight
                    class_qualities.append(quality)
                
                if class_qualities:
                    I_quality = sum(class_qualities) / len(class_qualities)
                else:
                    # No objects visible → fallback to all-pixel uncertainty
                    conf_valid = confidence[valid_mask].flatten()
                    depth_valid = actual_depth[valid_mask].flatten().to(conf_valid.device)
                    uncertainty_all = 1 - conf_valid
                    dist_weight_all = 1.0 / (1.0 + (depth_valid / d0) ** 2)
                    I_quality = float((uncertainty_all * dist_weight_all).mean())
                quality_utility[i] = float(I_quality)
            else:
                conf_valid = confidence[valid_mask].flatten()
                depth_valid = actual_depth[valid_mask].flatten().to(conf_valid.device)
                
                uncertainty = 1 - conf_valid
                dist_weight = 1.0 / (1.0 + (depth_valid / d0) ** 2)
                
                I_quality = (uncertainty * dist_weight).mean()
                quality_utility[i] = float(I_quality.cpu())
        
        return quality_utility
    
    def _compute_class_uncertainty_area(self, depth, confidence, valid_mask, extrinsic, intrinsic, voxel_map):
        """Compute per-class uncertainty and mean depth from rendered view."""
        H, W = depth.shape
        class_stats = {}
        
        stride = self.pixel_stride
        v_coords, u_coords = torch.where(valid_mask)
        sample_mask = (v_coords % stride == 0) & (u_coords % stride == 0)
        v_sampled = v_coords[sample_mask]
        u_sampled = u_coords[sample_mask]
        
        if len(v_sampled) == 0:
            return class_stats
        
        depth_cpu = depth.cpu() if depth.is_cuda else depth
        confidence_cpu = confidence.cpu() if confidence.is_cuda else confidence
        intrinsic_cpu = intrinsic.cpu() if intrinsic.is_cuda else intrinsic
        extrinsic_cpu = extrinsic.cpu() if extrinsic.is_cuda else extrinsic
        
        z = depth_cpu[v_sampled, u_sampled]
        conf_sampled = confidence_cpu[v_sampled, u_sampled]
        
        positive_mask = z > 0
        if positive_mask.sum() == 0:
            return class_stats
        z = z[positive_mask]
        conf_sampled = conf_sampled[positive_mask]
        v_sampled = v_sampled[positive_mask]
        u_sampled = u_sampled[positive_mask]
        
        fx, fy = intrinsic_cpu[0, 0], intrinsic_cpu[1, 1]
        cx, cy = intrinsic_cpu[0, 2], intrinsic_cpu[1, 2]
        
        x = (u_sampled.float() - cx) * z / fx
        y = (v_sampled.float() - cy) * z / fy
        
        points_cam = torch.stack([x, y, z], dim=-1)
        R = extrinsic_cpu[:3, :3]
        t = extrinsic_cpu[:3, 3]
        points_world = (R @ points_cam.T).T + t
        
        try:
            voxel_indices, valid = voxel_map._voxelize(points_world.to(voxel_map.device))
            
            if valid.sum() == 0:
                return class_stats
            
            linear_indices = voxel_map.to_linear_indices(voxel_indices[valid])
            classes = voxel_map.voxel_semantic_class[linear_indices].cpu()
            conf_valid = conf_sampled[valid.cpu()].cpu()
            z_valid = z[valid.cpu()]
            
            scale_factor = stride * stride
            for class_id in classes.unique():
                class_id_item = class_id.item()
                if class_id_item >= 0:  # -1 = unlabeled
                    class_mask = classes == class_id
                    class_stats[class_id_item] = {
                        'uncertainty': float((1 - conf_valid[class_mask]).mean()),
                        'area': int(class_mask.sum().item() * scale_factor),
                        'mean_depth': float(z_valid[class_mask].mean())
                    }
        except Exception as e:
            print(f"  [I_quality] Error: {e}")
        
        return class_stats
