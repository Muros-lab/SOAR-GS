import torch
import time

from utils.operations import *
from utils.common import Camera, Mapper2Gui, FakeQueue, TextColors
from .gaussian_map import GaussianMap
from .voxel_map import VoxelMap


class IncrementalMapper:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # map instance
        self.gaussian_map = None
        self.voxel_map = None

        # gui related
        self.use_gui = False
        self.q_mapper2gui = FakeQueue()
        self.q_gui2mapper = FakeQueue()
        self.pause = False
        self.init = False
        self.pending_keyframes = []  # Accumulate keyframes to prevent loss

    @property
    def current_map(self):
        return self.gaussian_map, self.voxel_map

    def load_recorder(self, recorder):
        print("\n ----------load mission recorder----------")
        self.recorder = recorder

    def load_simulator(self, simulator):
        print("\n ----------load simulator----------")
        self.simulator = simulator

    def load_planner(self, planner):
        print("\n ----------load planner----------")
        self.planner = planner

    def init_map(self):
        print("\n ----------initialize map----------")
        self.gaussian_map = GaussianMap(self.cfg.gaussian_map, self.device)
        
        # Initialize object-semantic voxel map if detector config is available
        # Try full_cfg first (set by get_mapper), then fall back to self.cfg
        full_cfg = getattr(self, 'full_cfg', self.cfg)
        detector_cfg = getattr(full_cfg, 'detector', None)
        
        
        if detector_cfg is not None and detector_cfg.get('enabled', True):
            print("[Mapper] Using ObjectSemanticMap with object detection")
            from mapping.object_semantic_map import ObjectSemanticMap
            self.voxel_map = ObjectSemanticMap(
                self.cfg.voxel_map, 
                self.simulator.bbox, 
                self.device,
                detector_cfg=detector_cfg
            )
        else:
            print("[Mapper] Using standard Voxel Map (no detector)")
            self.voxel_map = VoxelMap(self.cfg.voxel_map, self.simulator.bbox, self.device)

    def get_new_dataframe(self, i):
        # return way points to the nbv
        path = self.planner.plan(self.current_map, self.simulator, self.recorder)

        # for visualization only; noise-free frames, so the GUI does not draw from the
        # random stream used for the depth noise of the keyframes
        if self.use_gui:
            for pose in path:
                dataframe = self.simulator.simulate(pose, require_gt=True)
                camera_frame = Camera.init_from_mapper(None, dataframe)
                self.q_mapper2gui.put(
                    Mapper2Gui(
                        current_frame=camera_frame,
                    )
                )
                time.sleep(0.05)

        # dataframe at nbv as keyframe
        dataframe = self.simulator.simulate(path[-1])
        # Note: Camera creation moved to after voxel_map.update to include segmentation
        return dataframe

    def run(self):
        torch.cuda.empty_cache()
        self.init_map()
        frame_id = 0
        
        # Check if adaptive weighting is enabled
        use_adaptive = (
            hasattr(self.voxel_map, 'use_adaptive') and 
            self.voxel_map.use_adaptive
        )
        
        if use_adaptive:
            # Set initial position to bbox center if at origin (avoids floor-level start in MP3D)
            if torch.allclose(self.planner.pose[:3, 3], torch.zeros(3)):
                bbox = self.simulator.bbox
                bbox_center = (bbox[0] + bbox[1]) / 2
                self.planner.pose[:3, 3] = torch.tensor(bbox_center, dtype=torch.float32)
                print(f"  [Init] Coarse start position set to bbox center: {bbox_center.tolist()}")
            
            print(f"\n{TextColors.MAGENTA}========== PHASE 1: COARSE EXPLORATION =========={TextColors.RESET}")
            frame_id = self.run_coarse_exploration(frame_id)
            
            print(f"\n{TextColors.MAGENTA}========== Computing Adaptive Weights =========={TextColors.RESET}")
            self.finalize_coarse_phase()
            
            print(f"\n{TextColors.MAGENTA}========== PHASE 2: FINE EXPLORATION =========={TextColors.RESET}")

        print(
            f"\n {TextColors.MAGENTA}----------Start Active Reconstruction----------{TextColors.RESET}"
        )
        while self.recorder is None or self.recorder.is_alive:
            # pause information from gui
            if not self.q_gui2mapper.empty():
                data_gui2mapper = self.q_gui2mapper.get_nowait()
                self.pause = data_gui2mapper.flag_pause
            if self.pause:
                time.sleep(0.05)
                continue

            print(
                f"\n {TextColors.MAGENTA}----------Step {frame_id+1}----------{TextColors.RESET}"
            )

            print(f"\n {TextColors.GREEN}-----Planning:{TextColors.RESET}")
            dataframe = self.get_new_dataframe(frame_id)
            dataframe = {k: v.to(self.device) for k, v in dataframe.items()}

            print(f"\n {TextColors.GREEN}-----Mapping:{TextColors.RESET}")
            t_mapper_start = time.time()

            # update gaussian map
            self.gaussian_map.update(dataframe)

            # Update voxel map and get segmentation/detection visualizations
            seg_vis, det_vis = self.voxel_map.update(dataframe)
            if seg_vis is not None:
                dataframe["segmentation"] = seg_vis
            if det_vis is not None:
                dataframe["detection"] = det_vis

            t_mapper = time.time() - t_mapper_start
            
            # Create Camera with segmentation and accumulate as keyframe
            camera_frame = Camera.init_from_mapper(frame_id, dataframe)
            self.pending_keyframes.append(camera_frame)
            
            # Send current frame for live visualization
            self.q_mapper2gui.put(
                Mapper2Gui(
                    current_frame=camera_frame,
                )
            )
            
            frame_id += 1

            # Send map to gui with accumulated keyframes (prevents keyframe loss)
            self.q_mapper2gui.put(
                Mapper2Gui(
                    gaussians=self.gaussian_map,
                    voxels=self.voxel_map,
                    keyframes=self.pending_keyframes.copy(),  # Send accumulated keyframes
                )
            )
            self.pending_keyframes.clear()  # Clear after sending

            # update recorder or/and save map
            if self.recorder is not None:
                self.recorder.update_time("mapping", t_mapper)
                self.recorder.log()
                self.recorder.save_dataframe(dataframe, f"{frame_id:03}")
                if self.recorder.require_record:
                    self.recorder.save_map(self.gaussian_map, f"{frame_id:03}", self.voxel_map)
                    self.recorder.save_path()
            
            # Volume statistics management (ObjectSemanticMap)
            if hasattr(self.voxel_map, 'coarse_phase_complete'):
                # Mark coarse phase complete after first NBV
                if not self.voxel_map.coarse_phase_complete and frame_id >= 1:
                    print(f"\n{TextColors.GREEN}========== Coarse Phase Complete =========={TextColors.RESET}")
                    print(f"  Completed coarse exploration + first NBV")
                    
                    # Compute initial volume statistics
                    self.voxel_map.compute_volume_statistics()
                    self.voxel_map.coarse_phase_complete = True
                    
                    # Initialize tracking for adaptive updates
                    self.voxel_map.last_volume_update_pose = dataframe["extrinsic"].clone()
                    
                    # Track detected classes
                    detected_classes = set()
                    if hasattr(self.voxel_map, 'detector') and self.voxel_map.detector is not None:
                        for class_id, class_name in enumerate(self.voxel_map.detector.class_names):
                            if (self.voxel_map.voxel_semantic_class == class_id).any():
                                detected_classes.add(class_name)
                    self.voxel_map.last_detected_classes = detected_classes
                    
                    # Object Completion-Driven Planning: Set fine start target
                    if hasattr(self.voxel_map, 'get_best_fine_start_target'):
                        self.voxel_map.fine_start_target = self.voxel_map.get_best_fine_start_target()
                        self.voxel_map.fine_start_used = False
                    
                    print(f"{TextColors.GREEN}==========================================={TextColors.RESET}\n")
                
                # Adaptive volume statistics update (after coarse)
                elif self.voxel_map.coarse_phase_complete:
                    if self.voxel_map.should_update_volume_stats(dataframe["extrinsic"]):
                        self.voxel_map.compute_volume_statistics()
                        self.voxel_map.last_volume_update_pose = dataframe["extrinsic"].clone()
                        
                        # Update detected classes
                        detected_classes = set()
                        if hasattr(self.voxel_map, 'detector') and self.voxel_map.detector is not None:
                            for class_id, class_name in enumerate(self.voxel_map.detector.class_names):
                                if (self.voxel_map.voxel_semantic_class == class_id).any():
                                    detected_classes.add(class_name)
                        self.voxel_map.last_detected_classes = detected_classes
                    if self.recorder is not None:
                        self.recorder.save_path()
            time.sleep(0.1)

        print(
            f"\n {TextColors.MAGENTA}----------Finish Reconstruction Mission----------{TextColors.RESET}"
        )
    
    def run_coarse_exploration(self, start_frame_id):
        """
        Coarse exploration via 360-degree rotation around Z-axis at fixed XY position
        
        Position (XY) stays constant, only yaw (Z-axis rotation) changes
        
        Returns:
            Last frame_id
        """
        import math
        
        cfg = self.voxel_map.adaptive_cfg.get('coarse_phase', {})
        num_views = cfg.get('num_views', 12)
        vertical_angles = cfg.get('vertical_angles', [0])
       
        print(f"\nCoarse exploration: 360° Z-axis rotation")
        print(f"  Views: {num_views} × {len(vertical_angles)} levels")
        print(f"  XY position: FIXED (only yaw rotation around Z)")
        
        frame_id = start_frame_id
        initial_pose = self.planner.pose.clone()
        initial_position = initial_pose[:3, 3].cpu().numpy()
        
        print(f"  Initial position: X={initial_position[0]:.2f}, Y={initial_position[1]:.2f}, Z={initial_position[2]:.2f}")
        
        for v_angle in vertical_angles:
            for i in range(num_views):
                h_angle = (360.0 / num_views) * i  # Z-axis rotation (yaw)
                
                # Generate rotation pose (XY fixed, only Z-axis rotation)
                pose = self._generate_rotation_pose(initial_pose, h_angle, v_angle)
                
                # Verify position unchanged
                current_pos = pose[:3, 3].cpu().numpy()
                pos_str = f"({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})"
                
                # Simulate and update
                print(f"  [{frame_id+1:2d}] Yaw={h_angle:5.1f}°, Pitch={v_angle:4.1f}°, Pos={pos_str}")
                dataframe = self.simulator.simulate(pose)
                
                # Send visualization of rotation (without mapping yet)
                if self.use_gui:
                    # Create camera for path visualization (no segmentation yet)
                    camera_frame_viz = Camera.init_from_mapper(None, dataframe, with_measurement=False)
                    self.q_mapper2gui.put(
                        Mapper2Gui(
                            current_frame=camera_frame_viz,
                        )
                    )
                    time.sleep(0.15)  # Longer delay for clear visualization of each rotation step
                
                dataframe = {k: v.to(self.device) for k, v in dataframe.items()}
                
                # Update maps (includes semantic detection)
                t_mapper_start = time.time()
                self.gaussian_map.update(dataframe)
                
                # Update voxel map and get segmentation/detection visualizations
                seg_vis, det_vis = self.voxel_map.update(dataframe, is_keyframe=True)
                if seg_vis is not None:
                    dataframe["segmentation"] = seg_vis
                if det_vis is not None:
                    dataframe["detection"] = det_vis
                
                t_mapper = time.time() - t_mapper_start
                
                # Create Camera with segmentation for keyframe
                camera_frame = Camera.init_from_mapper(frame_id, dataframe)
                
                # Send updated maps to GUI
                if self.use_gui:
                    self.q_mapper2gui.put(
                        Mapper2Gui(
                            gaussians=self.gaussian_map,
                            voxels=self.voxel_map,
                            current_frame=camera_frame,
                        )
                    )
                    # Wait for GUI to process keyframe (prevents get_latest_queue from dropping frames)
                    time.sleep(0.3)
                
                # Update recorder
                if self.recorder is not None:
                    self.recorder.update_time("mapping", t_mapper)
                    self.recorder.log()
                    self.recorder.save_dataframe(dataframe, f"{frame_id:03}")
                    # Save rotation pose to global_path
                    if self.recorder.record_global_path:
                        camera_data = {"pose": pose, "name": frame_id}
                        self.recorder.global_path_dict[self.recorder.pose_id] = camera_data
                        self.recorder.pose_id += 1
                    self.recorder.save_map(self.gaussian_map, f"{frame_id:03}", self.voxel_map)
                
                frame_id += 1
        
        print(f"\nCoarse complete: {frame_id - start_frame_id} views collected")
        print(f"Position remained at: X={initial_position[0]:.2f}, Y={initial_position[1]:.2f}, Z={initial_position[2]:.2f}")
        return frame_id
    
    def _generate_rotation_pose(self, base_pose, h_angle_deg, v_angle_deg):
        """
        Generate camera pose with Z-axis rotation (yaw) at fixed XY position
        
        Standard robotics/drone convention (Z-up):
        - X: forward
        - Y: left
        - Z: up
        
        Coarse exploration: XY position FIXED, only yaw rotation around Z-axis
        
        Args:
            base_pose: Initial 4x4 pose matrix
            h_angle_deg: Horizontal rotation (yaw around Z-axis) in degrees
            v_angle_deg: Vertical tilt (pitch) in degrees (usually 0)
        
        Returns:
            4x4 pose matrix with rotated camera direction
        """
        import math
        
        pose = base_pose.clone()
        
        # Keep XY position FIXED (only Z-axis rotation)
        position = base_pose[:3, 3].clone()
        
        # Convert angles to radians
        yaw = math.radians(h_angle_deg)    # Rotation around Z-axis
        pitch = math.radians(v_angle_deg)  # Tilt (usually 0)
        
        # Z-up coordinate system (standard robotics)
        # Rotation matrix around Z-axis (yaw):
        # [cos(θ) -sin(θ)  0]
        # [sin(θ)  cos(θ)  0]
        # [  0       0     1]
        
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        
        # After Z-axis rotation, camera forward direction
        # In Z-up: X is forward initially, rotates in XY plane
        forward = torch.tensor([
            cos_yaw * cos_pitch,   # X component
            sin_yaw * cos_pitch,   # Y component  
            -sin_pitch             # Z component (tilt down is negative)
        ], dtype=torch.float32)
        
        # Up vector (Z-axis) - NEGATED to fix Y-axis mirror issue
        world_up = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        
        # Right vector = forward × up
        right = torch.cross(forward, world_up)
        right = right / (torch.norm(right) + 1e-8)
        
        # Recompute up = right × forward (for orthogonality)
        up = torch.cross(right, forward)
        up = up / (torch.norm(up) + 1e-8)
        
        # Build rotation matrix
        # Camera coordinate system: right, up, -forward
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # Camera looks along -Z in camera frame
        pose[:3, 3] = position  # XY FIXED!
        
        return pose
    
    def finalize_coarse_phase(self):
        """
        Compute volume statistics and activate adaptive weights
        """
        if not hasattr(self.voxel_map, 'compute_volume_statistics'):
            return
        
        # Compute volume stats
        stats = self.voxel_map.compute_volume_statistics()
        
        print("\n--- Object Volume Statistics ---")
        sorted_stats = sorted(
            stats.items(), 
            key=lambda x: x[1]['volume_ratio'], 
            reverse=True
        )
        
        for class_name, data in sorted_stats:
            volume_ratio = data['volume_ratio']
            voxel_count = data['voxel_count']
            print(f"  {class_name:20s}: {voxel_count:6d} voxels ({volume_ratio*100:5.1f}%)")
        
        # Mark coarse phase as complete
        self.voxel_map.coarse_phase_complete = True
        
        # Object Completion-Driven Planning: Set fine start target
        if hasattr(self.voxel_map, 'get_best_fine_start_target'):
            self.voxel_map.fine_start_target = self.voxel_map.get_best_fine_start_target()
            self.voxel_map.fine_start_used = False
            if self.voxel_map.fine_start_target is not None:
                print(f"  Fine start target set (largest object)")
        
        # Compute and display adaptive weights
        if hasattr(self.voxel_map, 'compute_adaptive_weights'):
            adaptive = self.voxel_map.compute_adaptive_weights(self.gaussian_map)
