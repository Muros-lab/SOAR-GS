import numpy as np
import torch
from typing import Optional
from einops import repeat
import time
from tqdm import tqdm
import cv2

from .utils import (
    PathPlanner,
    cal_flight_time,
    inplace_rotation,
    rotation_from_z_batch,
    select_points_within_cone,
    wp2path,
)
from utils.common import FakeQueue, Planner2Gui


class RegionTracker:
    """Track visited regions to prevent getting stuck in one area."""
    
    def __init__(self, region_size: float = 0.5, max_visits: int = 3):
        self.region_size = region_size
        self.visited_regions = set()
        self.visit_counts = {}
        self.max_visits_before_penalty = max_visits
    
    def get_region_id(self, position) -> tuple:
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        ix = int(position[0] / self.region_size)
        iy = int(position[1] / self.region_size)
        iz = int(position[2] / self.region_size)
        return (ix, iy, iz)
    
    def mark_visited(self, position):
        region_id = self.get_region_id(position)
        self.visited_regions.add(region_id)
        self.visit_counts[region_id] = self.visit_counts.get(region_id, 0) + 1
    
    def get_visit_penalty(self, position) -> float:
        """Continuous exponential decay penalty after threshold visits.
        
        Formula: P = exp(-max(0, visits - τ) / τ)
        where τ = max_visits_before_penalty is the threshold.
        
        - visits ≤ τ: P = 1.0 (no penalty, exploration allowed)
        - visits = 2τ: P = 0.37 (1/e)
        - visits = 3τ: P = 0.14
        
        Preserves original intent: first τ visits are penalty-free.
        """
        region_id = self.get_region_id(position)
        visits = self.visit_counts.get(region_id, 0)
        
        # Exponential decay only after threshold
        tau = self.max_visits_before_penalty
        excess_visits = max(0, visits - tau)
        return np.exp(-excess_visits / tau)
    
    def reset(self):
        self.visited_regions.clear()
        self.visit_counts.clear()


class PlanBase:
    def __init__(self, cfg, device):
        self.device = device
        self.pitch_angle = cfg.pitch_angle
        self.robot_size = cfg.robot_size
        self.radius = cfg.radius
        self.flight_speed = 1.0
        self.pose = torch.tensor(cfg.init_pose).type(torch.float32)
        self.init = False

        self.path_planner = PathPlanner()
        self.path_length_factor = cfg.path_length_factor

        self.use_confidence = cfg.use_confidence
        self.sample_num = cfg.sample_num
        self.max_roi_sample_num = cfg.max_roi_sample_num
        
        # Pose history for redundancy prevention
        self.pose_history = []
        self.max_pose_history = cfg.get('max_pose_history', 15)
        self.min_position_dist = cfg.get('min_position_dist', 0.3)
        self.min_angle_diff = cfg.get('min_angle_diff', 15.0)
        self.redundant_pose_penalty = cfg.get('redundant_pose_penalty', 0.3)
        
        # Region tracking
        region_size = cfg.get('region_size', 0.6)
        max_visits = cfg.get('max_visits_before_penalty', 3)
        self.region_tracker = RegionTracker(region_size=region_size, max_visits=max_visits)

        self.q_planner2gui = FakeQueue()
        self.q_gui2planner = FakeQueue()

    def plan(self, map, simulator, recorder):
        gaussian_map, voxel_map = map
        t_planning = 0
        if self.init:
            t_sampling_start = time.time()
            robot_space = self.get_robot_space(voxel_map)
            voxel_map.update_graph(robot_space)
            
            # Object Completion-Driven Fine Start
            if (hasattr(voxel_map, 'fine_start_target') and 
                voxel_map.fine_start_target is not None and
                hasattr(voxel_map, 'fine_start_used') and 
                not voxel_map.fine_start_used):
                
                print(f"\n{'='*50}")
                print(f"  OBJECT COMPLETION-DRIVEN FINE START")
                print(f"{'='*50}")
                
                target = voxel_map.fine_start_target
                nbv = self._generate_nbv_for_target(target, voxel_map)
                
                if nbv is not None:
                    voxel_map.fine_start_used = True
                    
                    wp_list, wp_length_list = self.path_planner.search_goal(
                        self.pose[:3, 3].numpy(),
                        nbv[:3, 3].unsqueeze(0).numpy(),
                        voxel_map,
                    )
                    
                    if not np.isinf(wp_length_list[0]):
                        wp_indices = wp_list[0]
                        waypoints = voxel_map.index_2_xyz(wp_indices).cpu()
                        
                        camera_path, path_length = wp2path(
                            self.pose[:3, :3], nbv[:3, :3], waypoints)
                        
                        # wp2path may return an empty path when start == end
                        if len(camera_path) == 0:
                            camera_path = nbv.unsqueeze(0)
                            path_length = 0
                        
                        self.pose = nbv
                        print(f"  Moving to fine start target (path={path_length:.2f}m)")
                        print(f"{'='*50}\n")
                        
                        if recorder is not None:
                            t_planning += time.time() - t_sampling_start
                            t_flight = cal_flight_time(path_length, flight_speed=self.flight_speed)
                            recorder.update_time("planning", t_planning)
                            recorder.update_time("flight", t_flight)
                            recorder.update_path(camera_path, path_length)
                        
                        return camera_path
                    else:
                        print(f"  Target unreachable, falling back to normal planning")
                        print(f"{'='*50}\n")
                else:
                    print(f"  Could not generate valid NBV for target")
                    voxel_map.fine_start_used = True

            # View candidate sampling
            if self.max_roi_sample_num > 0:
                voxel_map.update_utility(gaussian_map, self.use_confidence)
                roi_candidates = self.generate_roi_candidates(
                    voxel_map, self.max_roi_sample_num, gaussian_map
                )
            else:
                roi_candidates = torch.tensor([])

            if self.sample_num - len(roi_candidates) > 0:
                random_candidates = self.generate_random_candidates(
                    voxel_map, self.sample_num - len(roi_candidates)
                )
            else:
                random_candidates = torch.tensor([])

            total_candidates = torch.cat((roi_candidates, random_candidates), dim=0)
            if len(total_candidates) == 0:
                # no candidate could be sampled: stay in place for this step
                print("Warning: no view candidates sampled, staying in place")
                t_planning += time.time() - t_sampling_start
                camera_path = self.pose.unsqueeze(0)
                if recorder is not None:
                    recorder.update_time("planning", t_planning)
                    recorder.update_time("flight", 0)
                    recorder.update_path(camera_path, 0)
                return camera_path
            view_xyz = total_candidates[:, :3, 3]
            view_direction = total_candidates[:, :3, 2]
            t_planning += time.time() - t_sampling_start
            self.q_planner2gui.put(Planner2Gui(view_xyz, view_direction))
            print(f"\n generate {len(roi_candidates)} roi samples, {len(random_candidates)} random samples")

            # Utility calculation
            utility_list, t_utility = self.cal_utility(
                gaussian_map, voxel_map, total_candidates, simulator
            )
            t_planning += t_utility

            # Path planning
            t_path_start = time.time()
            wp_list, wp_length_list = self.path_planner.search_goal(
                self.pose[:3, 3].numpy(),
                total_candidates[:, :3, 3].numpy(),
                voxel_map,
            )
            t_planning += time.time() - t_path_start

            # NBV selection
            score_list = self.cal_view_scores(utility_list, wp_length_list, 
                                               total_candidates, voxel_map, gaussian_map)
            nbv_id = torch.argmax(score_list)
            if nbv_id < len(roi_candidates):
                print("    → Selected from ROI candidates!")
            
            # Log selected NBV details
            if hasattr(self, '_last_utility_components') and self._last_utility_components is not None:
                uc = self._last_utility_components
                ie = uc['I_explore'][nbv_id]
                iq = uc['I_quality'][nbv_id]
                ib = uc['I_boundary'][nbv_id]
                u = uc['utility'][nbv_id]
                s = score_list[nbv_id]
                pl = wp_length_list[nbv_id]
                a, b, g = self.alpha_explore, self.beta_quality, self.gamma_boundary
                print(f"\n  [NBV] U={a:.1f}*{ie:.3f}+{b:.1f}*{iq:.3f}+{g:.1f}*{ib:.3f}={u:.3f} → Score={s:.3f}, Path={pl:.1f}m")
            
            nbv = total_candidates[nbv_id]
            wp_length = wp_length_list[nbv_id]

            # Handle unreachable NBV
            if np.isinf(wp_length):
                print("Warning: Selected NBV unreachable, finding alternative...")
                sorted_indices = torch.argsort(score_list, descending=True)
                for alt_id in sorted_indices:
                    if not np.isinf(wp_length_list[alt_id]):
                        nbv_id = alt_id
                        nbv = total_candidates[nbv_id]
                        wp_length = wp_length_list[nbv_id]
                        print(f"  Fallback to candidate {nbv_id} with path length {wp_length:.2f}")
                        break
                else:
                    print("  All candidates unreachable, staying in place")
                    nbv = torch.eye(4)
                    nbv[:3, :3] = self.pose[:3, :3]
                    nbv[:3, 3] = self.pose[:3, 3]
                    wp_indices = [voxel_map.xyz_2_index(self.pose[:3, 3])]
                    waypoints = torch.stack([self.pose[:3, 3]])
                    camera_path, path_length = wp2path(
                        self.pose[:3, :3], nbv[:3, :3], waypoints)
                    
                    # wp2path may return empty path when start==end
                    if len(camera_path) == 0:
                        camera_path = nbv.unsqueeze(0)
                    
                    self.pose = nbv
                    if recorder is not None:
                        recorder.update_time("planning", t_planning)
                        recorder.update_time("flight", 0)
                        recorder.update_path(camera_path, 0)
                    return camera_path

            wp_indices = wp_list[nbv_id]
            waypoints = voxel_map.index_2_xyz(wp_indices).cpu()

        else:
            nbv = torch.eye(4)
            nbv[:3, :3] = self.pose[:3, :3]
            # Use bbox center if init position is at origin (avoids floor-level start in MP3D)
            init_pos = self.pose[:3, 3]
            if torch.allclose(init_pos, torch.zeros(3)):
                bbox_center = ((voxel_map.bbox[0] + voxel_map.bbox[1]) / 2).cpu().float()
                init_pos = bbox_center
                print(f"  [Init] Using bbox center as start: {init_pos.tolist()}")
            nbv_index = voxel_map.xyz_2_index(init_pos)
            nbv_xyz = voxel_map.index_2_xyz([nbv_index])[0].cpu()
            nbv[:3, 3] = nbv_xyz
            waypoints = torch.stack([init_pos, nbv_xyz])
            self.init = True

        camera_path, path_length = wp2path(self.pose[:3, :3], nbv[:3, :3], waypoints)
        
        if len(camera_path) == 0:
            print("Warning: Empty path generated, staying in place")
            camera_path = self.pose.unsqueeze(0)
            path_length = 0
        
        self.pose = nbv
        self._update_pose_history(nbv)
        self.region_tracker.mark_visited(nbv[:3, 3])

        if recorder is not None:
            t_flight = cal_flight_time(path_length, flight_speed=self.flight_speed)
            recorder.update_time("planning", t_planning)
            recorder.update_time("flight", t_flight)
            recorder.update_path(camera_path, path_length)

        return camera_path

    def generate_random_candidates(self, voxel_map, num):
        """Generate random view candidates around current pose."""
        voxel_centers = voxel_map.voxel_centers.cpu().numpy()
        free_mask = voxel_map.free_mask_w_margin.cpu().numpy()

        range_from_start = np.linalg.norm(
            voxel_centers - self.pose[:3, 3].numpy(), axis=1
        )
        within_range = range_from_start <= self.radius
        valid_mask = free_mask & within_range
        valid_centers = voxel_centers[valid_mask]
        
        # Handle edge case: no valid centers or not enough
        if len(valid_centers) == 0:
            # Expand search radius as fallback
            expanded_radius = self.radius * 3
            within_range = range_from_start <= expanded_radius
            valid_mask = free_mask & within_range
            valid_centers = voxel_centers[valid_mask]
            
            if len(valid_centers) == 0:
                print(f"Warning: No valid random candidate locations found")
                return torch.tensor([])
        
        # Ensure we don't request more samples than available
        actual_num = min(num, len(valid_centers))
        if actual_num < num:
            print(f"Warning: Only {actual_num} valid centers available (requested {num})")
        
        random_indices = np.random.choice(len(valid_centers), size=actual_num, replace=False)
        view_positions = valid_centers[random_indices]
        candidates = inplace_rotation(
            view_positions, pitch_angle=self.pitch_angle, num=actual_num
        )
        return candidates

    def generate_roi_candidates(self, voxel_map, num, gaussian_map=None):
        """Generate targeted view candidates around ROI with D_c-aware prioritization."""
        roi_candidates = torch.tensor([])
        sample_per_roi = 5
        free_mask = voxel_map.free_mask_w_margin
        free_points = voxel_map.voxel_centers[free_mask]

        roi_mask = voxel_map.roi_mask
        roi_centers = voxel_map.voxel_centers[roi_mask]
        roi_normals = voxel_map.voxel_normal[roi_mask]
        
        roi_distance = torch.linalg.norm(
            roi_centers - self.pose[:3, 3].unsqueeze(0).to(self.device), dim=1
        )
        
        # Object-class-aware prioritization
        if (hasattr(voxel_map, 'coarse_phase_complete') and 
            voxel_map.coarse_phase_complete and
            hasattr(voxel_map, 'voxel_semantic_class') and
            gaussian_map is not None):
            
            D_c_weights = {}
            if hasattr(voxel_map, 'compute_adaptive_weights'):
                D_c_weights = voxel_map.compute_adaptive_weights(gaussian_map)
            
            voxel_result = voxel_map._voxelize(roi_centers)
            voxel_indices = voxel_result[0]
            valid_mask = voxel_result[1]
            
            roi_D_c = torch.zeros(len(roi_centers), device=self.device)  # D_c [0, 1]
            
            if valid_mask.any() and len(D_c_weights) > 0 and hasattr(voxel_map, 'detector'):
                roi_linear_indices = voxel_map.to_linear_indices(voxel_indices[valid_mask])
                roi_classes = voxel_map.voxel_semantic_class[roi_linear_indices]
                valid_roi_indices = torch.where(valid_mask)[0]
                
                for idx, class_id in enumerate(roi_classes):
                    class_id_int = int(class_id.item()) if hasattr(class_id, 'item') else int(class_id)
                    if class_id_int >= 0 and class_id_int < len(voxel_map.detector.class_names):
                        class_name = voxel_map.detector.class_names[class_id_int]
                        original_idx = valid_roi_indices[idx]
                        roi_D_c[original_idx] = D_c_weights.get(class_name, 0.0)
            
            D_c_distance_bonus = 0.3  # meters bonus per D_c point
            priority_score = roi_distance - D_c_distance_bonus * roi_D_c
            _, sorted_roi_index = torch.sort(priority_score)
            
            print(f"\n  [ROI] D_c bonus={D_c_distance_bonus}m per D_c point")
        else:
            _, sorted_roi_index = torch.sort(roi_distance)
        
        for roi_index in sorted_roi_index:
            roi_center = roi_centers[roi_index]
            roi_normal = roi_normals[roi_index]
            candidate_positions, candidate_views = select_points_within_cone(
                roi_center, roi_normal,
                d_close=0.3, d_far=2.0, cosine_sim=0.5,
                free_points=free_points, voxel_map=voxel_map,
                pitch_angle=self.pitch_angle,
            )
            num_candidates = len(candidate_positions)

            if num_candidates > 0:
                if num_candidates > sample_per_roi:
                    selected_index = np.random.choice(
                        range(num_candidates), size=sample_per_roi, replace=False,
                    )
                    candidate_positions = candidate_positions[selected_index]
                    candidate_views = candidate_views[selected_index]

                Ts = torch.tensor(repeat(np.eye(4), "h w -> n h w", n=len(candidate_positions)))
                Ts[:, :3, 3] = candidate_positions
                Ts[:, :3, :3] = rotation_from_z_batch(candidate_views)

                roi_candidates = torch.cat((roi_candidates, Ts), dim=0)

            if len(roi_candidates) >= num:
                return roi_candidates.type(torch.float32).cpu()

        return roi_candidates.type(torch.float32).cpu()

    def get_robot_space(self, voxel_map):
        range_from_start = torch.linalg.norm(
            voxel_map.voxel_centers - self.pose[:3, 3].unsqueeze(0).to(self.device),
            dim=1,
        )
        robot_space = range_from_start < self.robot_size
        return robot_space

    def cal_view_scores(self, view_utilities, path_lengths, candidates=None, voxel_map=None, gaussian_map=None):
        """Calculate final score using linear path cost subtraction: Score = U·P - λ·C
        
        Hybrid formulation:
        - Combined penalty P = P_sim × P_reg (direct product)
        - Path cost C linearly subtracted with weight λ
        - λ controls cost sensitivity: higher λ = penalize longer paths more
        """
        path_lengths = torch.tensor(path_lengths)
        valid_candidate_mask = ~torch.isinf(path_lengths)
        
        # Cost: max normalization to [0, 1]
        normalized_cost = torch.zeros_like(path_lengths)
        if valid_candidate_mask.sum() > 0:
            valid_paths = path_lengths[valid_candidate_mask]
            max_path = valid_paths.max()
            if max_path > 1e-8:
                normalized_cost = path_lengths / max_path
            else:
                normalized_cost = torch.zeros_like(path_lengths)
        normalized_cost[~valid_candidate_mask] = 1.0  # Unreachable = max cost

        # Utilities (already normalized in cal_utility)
        utilities = view_utilities.clone()
        utilities[torch.isnan(utilities)] = 0
        
        utility_max = utilities.max()
        print(f"  [Score] U max={utility_max:.4f}")
        
        # Compute penalties
        similarity_penalty = torch.ones(len(utilities))
        region_penalty = torch.ones(len(utilities))
        
        if candidates is not None and len(self.pose_history) > 0:
            similarity_penalty = self._compute_pose_similarity_penalty(candidates)
        
        if candidates is not None:
            region_penalty = self._compute_region_visit_penalty(candidates)
        
        # Combined penalty: sequential multiplication (stronger avoidance)
        combined_penalty = similarity_penalty * region_penalty
        
        # Effective utility = U × P
        effective_utility = utilities * combined_penalty
        
        # Linear path cost subtraction: Score = U·P - λ·C
        if torch.all(effective_utility == 0):
            # Fallback: random selection among reachable candidates
            view_scores = torch.rand_like(utilities)
            view_scores[~valid_candidate_mask] = -1.0  # Exclude unreachable
        else:
            view_scores = effective_utility - self.path_length_factor * normalized_cost
        
        # Log for debugging
        print(f"  [Score] P_sim: [{similarity_penalty.min():.2f}, {similarity_penalty.max():.2f}]")
        print(f"  [Score] P_reg: [{region_penalty.min():.2f}, {region_penalty.max():.2f}]")
        print(f"  [Score] Combined P: [{combined_penalty.min():.2f}, {combined_penalty.max():.2f}]")
        print(f"  [Score] Path cost (λ={self.path_length_factor}): [{normalized_cost.min():.2f}, {normalized_cost.max():.2f}]")
        print(f"  [Score] Final: [{view_scores.min():.4f}, {view_scores.max():.4f}]")
        
        return view_scores
    
    def _generate_nbv_for_target(self, target_point: torch.Tensor, voxel_map) -> Optional[torch.Tensor]:
        """Generate a view pose that observes the target point."""
        free_mask = voxel_map.free_mask_w_margin
        free_points = voxel_map.voxel_centers[free_mask]
        
        if len(free_points) == 0:
            return None
        
        distances = torch.norm(free_points - target_point.to(free_points.device), dim=1)
        valid_mask = (distances >= 0.5) & (distances <= 2.5)
        valid_points = free_points[valid_mask]
        valid_distances = distances[valid_mask]
        
        if len(valid_points) == 0:
            valid_mask = distances <= 4.0
            valid_points = free_points[valid_mask]
            valid_distances = distances[valid_mask]
        
        if len(valid_points) == 0:
            return None
        
        closest_idx = torch.argmin(valid_distances)
        camera_pos = valid_points[closest_idx].cpu()
        
        view_dir = target_point.cpu() - camera_pos
        view_dir = view_dir / torch.norm(view_dir)
        
        z_axis = view_dir
        world_up = torch.tensor([0., 0., 1.])
        if abs(torch.dot(z_axis, world_up)) > 0.99:
            world_up = torch.tensor([0., 1., 0.])
        
        x_axis = torch.cross(world_up, z_axis)
        x_axis = x_axis / torch.norm(x_axis)
        y_axis = torch.cross(z_axis, x_axis)
        
        nbv = torch.eye(4)
        nbv[:3, 0] = x_axis
        nbv[:3, 1] = y_axis
        nbv[:3, 2] = z_axis
        nbv[:3, 3] = camera_pos
        
        return nbv
    
    def _compute_region_visit_penalty(self, candidates):
        """Compute penalty for candidates in visited regions."""
        N = len(candidates)
        penalty = torch.ones(N)
        
        for i in range(N):
            candidate_pos = candidates[i, :3, 3]
            penalty[i] = self.region_tracker.get_visit_penalty(candidate_pos)
        
        return penalty
    
    def _compute_pose_similarity_penalty(self, candidates):
        """Compute smooth penalty for candidates similar to recent poses.
        
        Uses Gaussian decay instead of binary threshold to avoid cliff effects.
        P = product over history of (1 - exp(-d²/2σ_d²) · exp(-θ²/2σ_θ²))
        """
        N = len(candidates)
        penalty = torch.ones(N)
        
        if len(self.pose_history) == 0:
            return penalty
        
        candidate_positions = candidates[:, :3, 3]
        candidate_directions = candidates[:, :3, 2]
        
        # Decay parameters (sigma values)
        sigma_d = self.min_position_dist  # Use existing threshold as sigma
        sigma_theta = self.min_angle_diff  # degrees
        
        for i in range(N):
            similarity_factor = 1.0
            
            for past_pose in self.pose_history:
                past_pos = past_pose[:3, 3]
                past_dir = past_pose[:3, 2]
                
                # Position distance
                pos_dist = torch.norm(candidate_positions[i] - past_pos)
                
                # Angle difference (degrees)
                cos_angle = torch.dot(candidate_directions[i], past_dir)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                angle_diff = torch.acos(cos_angle) * 180 / np.pi
                
                # Gaussian similarity: high when close in both position and angle
                pos_similarity = torch.exp(-0.5 * (pos_dist / sigma_d) ** 2)
                angle_similarity = torch.exp(-0.5 * (angle_diff / sigma_theta) ** 2)
                combined_similarity = pos_similarity * angle_similarity
                
                # Accumulate: penalty decreases as similarity increases
                # Each similar past pose reduces the penalty multiplicatively
                similarity_factor *= (1.0 - combined_similarity * (1.0 - self.redundant_pose_penalty))
            
            penalty[i] = similarity_factor
        
        return penalty
    
    def _update_pose_history(self, pose):
        self.pose_history.append(pose.clone())
        if len(self.pose_history) > self.max_pose_history:
            self.pose_history.pop(0)
    
    def cal_utility(self):
        raise NotImplementedError
