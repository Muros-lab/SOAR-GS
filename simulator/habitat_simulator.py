import habitat_sim
import numpy as np
import quaternion
import torch
import os
import logging
import trimesh
from .utils import *

# turn off non-critical log from simulator
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class HabitatSimulator:
    def __init__(self, simulator_cfg, scene_cfg):
        print("\n ----------configure habitat simulator----------")

        # get simulator backend config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.gpu_device_id = 0
        if not os.path.exists(scene_cfg.scene_id):
            raise FileNotFoundError(
                f"Scene not found: {scene_cfg.scene_id}. Download the data (see README) and run from the repo root."
            )
        if not os.path.exists(scene_cfg.mesh_path):
            raise FileNotFoundError(
                f"Scene mesh not found: {scene_cfg.mesh_path}. Download the data (see README) and run from the repo root."
            )
        backend_cfg.scene_id = scene_cfg.scene_id
        backend_cfg.enable_physics = simulator_cfg.physics.enable
        self.scene_name = scene_cfg.scene_name
        self.has_missing_surface = scene_cfg.has_missing_surface
        self.mesh = trimesh.load(scene_cfg.mesh_path)
        # MP3D .obj files contain multiple sub-meshes → trimesh returns Scene
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(self.mesh.dump())
        self.bbox = np.array(self.mesh.bounding_box.bounds)
        
        # MP3D: Habitat transforms Z-up scenes to Y-up internally.
        # SOAR-GS uses Z-up internally (matching Replica), so we need
        # to convert poses from Z-up to Y-up when sending to Habitat.
        self.mesh_up_axis = getattr(scene_cfg, 'mesh_up_axis', 'y')
        if self.mesh_up_axis == 'z':
            # Z-up to Y-up: X=X, Y=Z, Z=-Y
            self._z2y_transform = np.array([
                [1,  0,  0, 0],
                [0,  0,  1, 0],
                [0, -1,  0, 0],
                [0,  0,  0, 1]
            ], dtype=np.float64)
            print(f"  [Coordinate] MP3D mode: poses will be Z-up→Y-up for Habitat")
        else:
            self._z2y_transform = None

        # get sensor config
        sensor_specs = []
        self.resolution = np.array(simulator_cfg.sensor.resolution)
        H, W = self.resolution
        self.fov = np.array(simulator_cfg.sensor.fov)
        vfov, hfov = self.fov
        self.intrinsic = compute_camera_intrinsic(
            H, W, vfov, hfov, normalize=simulator_cfg.sensor.normalize
        )
        self.depth_noise_co = simulator_cfg.sensor.depth_noise_co
        self.depth_range = simulator_cfg.sensor.depth_range

        # all sensors have the same intrinsic
        for sensor_type in simulator_cfg.sensor.sensor_type:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_type
            sensor_spec.sensor_type = SENSOR_TYPE[sensor_type]
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_spec.resolution = [H, W]
            sensor_spec.vfov = vfov
            sensor_spec.hfov = hfov
            sensor_spec.position = simulator_cfg.sensor.position
            sensor_specs.append(sensor_spec)

        # get agent config
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        print(
            "\n scene:",
            self.scene_name,
            "\n bounding_box:",
            self.bbox.tolist(),
            "\n resolution:",
            self.resolution.tolist(),
            "\n fov:",
            self.fov.tolist(),
            "\n depth_range:",
            self.depth_range,
            "\n depth_noise_co:",
            self.depth_noise_co,
        )

        # spawn simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        if backend_cfg.enable_physics:
            self.sim.set_gravity(simulator_cfg.physics.gravity)
        
        # Load semantic mapping if semantic sensor is enabled
        self.semantic_id_to_class = None
        if "semantic" in simulator_cfg.sensor.sensor_type:
            self._load_semantic_mapping(scene_cfg)

        print("\n ----------load habitat simulator----------")
        self.data = {}
    
    def _load_semantic_mapping(self, scene_cfg):
        """Load instance ID to class ID mapping from info_semantic.json"""
        import json
        
        # Try to find info_semantic.json
        scene_dir = os.path.dirname(scene_cfg.scene_id)
        semantic_info_path = os.path.join(scene_dir, "info_semantic.json")
        
        if os.path.exists(semantic_info_path):
            with open(semantic_info_path, "r") as f:
                semantic_info = json.load(f)
            
            # Build class name to ID mapping
            self.class_name_to_id = {}
            for cls in semantic_info.get("classes", []):
                self.class_name_to_id[cls["name"]] = cls["id"]
            
            # Build instance ID to class ID mapping
            # id_to_label maps object instance ID to class ID
            id_to_label = semantic_info.get("id_to_label", [])
            self.semantic_id_to_class = {i: cls_id for i, cls_id in enumerate(id_to_label)}
            
            print(f"Loaded semantic mapping: {len(self.semantic_id_to_class)} instances, {len(self.class_name_to_id)} classes")
        else:
            print(f"Warning: Semantic info not found at {semantic_info_path}")
            self.semantic_id_to_class = None


    def simulate(self, c2w, valid_mask_only=False, require_gt=False, return_valid_mask=False):
        # simulate agent motion
        c2w_habitat = opencv_to_opengl_camera(c2w.numpy())
        
        # MP3D: convert from Z-up (SOAR-GS internal) to Y-up (Habitat)
        if self._z2y_transform is not None:
            c2w_habitat = self._z2y_transform @ c2w_habitat
        
        orientation = quaternion.from_rotation_matrix(c2w_habitat[:3, :3])
        position = np.array(c2w_habitat[:3, 3])
        agent_state = habitat_sim.agent.AgentState(position, orientation)
        self.sim.get_agent(0).set_state(agent_state)

        # get observations
        obs = self.sim.get_sensor_observations()
        color = obs.get("color", None)
        depth = obs.get("depth", None)
        valid_mask = None

        # use for planning purpose to exclude missing surfaces
        if valid_mask_only and depth is not None:
            valid_mask = depth > 0
            return valid_mask

        # use for mapping and test
        else:
            if color is not None:
                rgb = color[:, :, :3] / 255.0
                rgb = torch.from_numpy(rgb.astype(np.float32))
                rgb = rgb.permute(2, 0, 1)  # (C, H, W)

            if depth is not None:
                valid_mask = depth > 0  # missing surface return depth=0

                if not require_gt:
                    # for mapping
                    range_mask = (depth > self.depth_range[0]) & (
                        depth < self.depth_range[1]
                    )
                    depth_noise_std = depth * self.depth_noise_co
                    depth_noise = np.random.normal(scale=depth_noise_std)
                    depth += depth_noise
                    depth[~range_mask] = -1.0  # depth = -1 for out of range

                depth[~valid_mask] = -2.0  # depth = -2 for missing surfaces
                depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(
                    0
                )  # (1, H, W)

            data_frame = {
                "extrinsic": c2w,
                "intrinsic": self.intrinsic,
                "rgb": rgb,
                "depth": depth,
                "depth_range": torch.tensor(self.depth_range),
            }
            
            # Add semantic observation if available (for per-class evaluation)
            semantic = obs.get("semantic", None)
            if semantic is not None:
                # Convert instance IDs to class IDs if mapping is available
                if self.semantic_id_to_class is not None:
                    semantic_class = np.zeros_like(semantic, dtype=np.int32)
                    for inst_id, cls_id in self.semantic_id_to_class.items():
                        semantic_class[semantic == inst_id] = cls_id
                    data_frame["semantic"] = torch.from_numpy(semantic_class)
                else:
                    # Fallback: use instance IDs directly
                    data_frame["semantic"] = torch.from_numpy(semantic.astype(np.int32))

            
            # Return valid_mask together if requested (avoids second simulate call)
            if return_valid_mask:
                data_frame["valid_mask"] = valid_mask

            return data_frame

