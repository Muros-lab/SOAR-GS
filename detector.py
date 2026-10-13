"""
YOLO-World + EfficientSAM Object Detection & Segmentation Pipeline
"""


import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectionResult:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    mask: np.ndarray  # H x W binary mask


class ObjectDetector:
    
    def __init__(
        self,
        yolo_model: str = "yolov8x-worldv2",
        efficient_sam_model: str = "vits",  # "vits" or "vitt"
        device: str = "cuda",
        confidence_threshold: float = 0.4,
        class_names: Optional[List[str]] = None,
        weights_dir: Optional[str] = None,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        self.weights_dir = Path(weights_dir) if weights_dir else Path("./weights")
        
        # Initialize models
        print(f"[ObjectDetector] Loading YOLO-World: {yolo_model}")
        self.yolo = self._load_yolo_world(yolo_model)
        
        print(f"[ObjectDetector] Loading EfficientSAM: {efficient_sam_model}")
        self.sam = self._load_efficient_sam(efficient_sam_model)
        
        print(f"[ObjectDetector] Ready. Device: {device}")
        
    def _load_yolo_world(self, model_name: str):
        """Load YOLO-World model"""
        from ultralytics import YOLO
        
        # loaded from weights_dir; ultralytics downloads the official file there if it is missing
        model = YOLO(str(self.weights_dir / f"{model_name}.pt"))
        
        # Set custom classes for open-vocabulary detection
        if self.class_names:
            model.set_classes(self.class_names)
            print(f"[YOLO-World] Custom classes: {self.class_names}")
        
        return model
    
    def _load_efficient_sam(self, model_type: str = "vits"):
        """
        Load EfficientSAM model

        Args:
            model_type: "vits" (small) or "vitt" (tiny)

        Same architecture and checkpoint as efficient_sam.build_efficient_sam_vits/vitt,
        but the checkpoint is read from weights_dir instead of the current directory.
        """
        try:
            from efficient_sam.efficient_sam import build_efficient_sam
        except ImportError:
            raise ImportError(
                "EfficientSAM not installed. Install with:\n"
                "  pip install git+https://github.com/yformer/EfficientSAM.git"
            )

        if model_type == "vits":
            embed_dim, num_heads = 384, 6
        else:
            embed_dim, num_heads = 192, 3
        checkpoint = self.weights_dir / f"efficient_sam_{model_type}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"EfficientSAM checkpoint not found: {checkpoint}. Download it from "
                f"https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_{model_type}.pt.zip "
                f"and unzip it into {self.weights_dir}."
            )

        model = build_efficient_sam(
            encoder_patch_embed_dim=embed_dim,
            encoder_num_heads=num_heads,
            checkpoint=str(checkpoint),
        ).eval()
        model = model.to(self.device)
        model.eval()
        return model

    def set_classes(self, class_names: List[str]):
        """Update detection classes (YOLO-World open-vocabulary)"""
        self.class_names = class_names
        self.yolo.set_classes(class_names)
        print(f"[ObjectDetector] Updated classes: {class_names}")
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run full detection + segmentation pipeline
        
        Args:
            image: RGB image (H, W, 3), uint8
            
        Returns:
            List of DetectionResult
        """
        # Step 1: YOLO-World detection
        detections = self._run_yolo(image)
        
        if len(detections) == 0:
            return []
        
        # Step 2: EfficientSAM segmentation for each detection
        results = self._run_efficient_sam(image, detections)
        
        return results
    
    def _run_yolo(self, image: np.ndarray) -> List[Dict]:
        """Run YOLO-World detection"""
        results = self.yolo(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].cpu().numpy(),
                    "confidence": boxes.conf[i].cpu().item(),
                    "class_id": int(boxes.cls[i].cpu().item()),
                    "class_name": result.names[int(boxes.cls[i].cpu().item())],
                }
                detections.append(det)
                
        return detections
    
    def _run_efficient_sam(
        self, 
        image: np.ndarray, 
        detections: List[Dict]
    ) -> List[DetectionResult]:
        """Run EfficientSAM segmentation with bbox prompts"""
        results = []
        H, W = image.shape[:2]
        
        # Prepare image tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            
            # EfficientSAM uses box as two corner points
            # point format: top-left and bottom-right corners
            # labels: 2 = top-left, 3 = bottom-right (box prompt)
            pts_sampled = torch.tensor([
                [bbox[0], bbox[1]],  # top-left (x1, y1)
                [bbox[2], bbox[3]],  # bottom-right (x2, y2)
            ]).float().reshape(1, 1, 2, 2).to(self.device)  # (1, 1, 2, 2)
            
            pts_labels = torch.tensor([2, 3]).reshape(1, 1, 2).to(self.device)  # (1, 1, 2)
            
            # Run EfficientSAM
            with torch.no_grad():
                predicted_masks, _ = self.sam(
                    image_tensor,
                    pts_sampled,
                    pts_labels,
                )
            
            # Post-process mask
            # Output shape: [1, 1, 3, H, W] - 3 mask candidates
            # The first candidate is used, thresholded at 0.5
            mask = predicted_masks[0, 0, 0].cpu().numpy()  # (H, W)
            mask = (mask > 0.5).astype(np.uint8)
            
            # Resize to match input image if needed
            if mask.shape != (H, W):
                import cv2
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            result = DetectionResult(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=bbox,
                mask=mask,
            )
            results.append(result)
            
        return results


class SemanticProjector:
    """Project 2D segmentation masks to 3D voxel space."""
    
    def __init__(self, voxel_map):
        self.voxel_map = voxel_map
        self.device = voxel_map.device
        
    def project_to_voxels(
        self,
        detections: List[DetectionResult],
        depth: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> Dict[int, List[Dict]]:
        """
        Project 2D masks to 3D voxel indices
        
        Args:
            detections: List of DetectionResult
            depth: Depth map (H, W)
            extrinsic: Camera extrinsic matrix (4, 4)
            intrinsic: Camera intrinsic matrix (3, 3)
            
        Returns:
            Dict mapping class_id to list of voxel info
        """
        H, W = depth.shape
        semantic_voxels = {}
        
        for det in detections:
            mask = det.mask.astype(bool)
            class_id = det.class_id
            
            # Get pixel coordinates where mask is True
            ys, xs = np.where(mask)
            
            if len(xs) == 0:
                continue
                
            # Get depth values at mask pixels
            depths = depth[ys, xs]
            valid_mask = depths > 0
            
            xs = xs[valid_mask]
            ys = ys[valid_mask]
            depths = depths[valid_mask]
            
            if len(xs) == 0:
                continue
            
            # Backproject to 3D (pass actual resolution)
            points_3d = self._backproject(xs, ys, depths, extrinsic, intrinsic, H, W)
            
            # Voxelize
            voxel_indices, linear_indices = self._to_voxel_indices(points_3d)
            
            if voxel_indices is None or len(voxel_indices) == 0:
                continue
            
            if class_id not in semantic_voxels:
                semantic_voxels[class_id] = []
                
            semantic_voxels[class_id].append({
                "voxel_indices": voxel_indices,
                "linear_indices": linear_indices,
                "confidence": det.confidence,
                "class_name": det.class_name,
                "num_points": len(points_3d),
            })
            
        return semantic_voxels
    
    def _backproject(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        depths: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        H: int = 512,
        W: int = 512,
    ) -> np.ndarray:
        """Backproject 2D pixels to 3D world coordinates
        
        Habitat simulator provides:
        - extrinsic: 4x4 camera-to-world matrix (OpenCV convention)
        - depth: depth values in meters (NOT centimeters!)
        - intrinsic: normalized intrinsic (may need to scale by resolution)
        
        Args:
            xs, ys: pixel coordinates
            depths: depth values at those pixels
            extrinsic: camera-to-world matrix
            intrinsic: camera intrinsic matrix
            H, W: image height and width (for normalized intrinsic scaling)
        """
        
        # Check if intrinsic is normalized (values < 1) or pixel-based
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        # If normalized intrinsic, scale to pixel coordinates
        if fx < 10:  # Normalized intrinsic has small values
            fx = fx * W
            fy = fy * H
            cx = cx * W
            cy = cy * H
        
        # Pixel → Camera coordinates (OpenCV: Z forward, X right, Y down)
        x_cam = (xs - cx) * depths / fx
        y_cam = (ys - cy) * depths / fy
        z_cam = depths
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)
        
        # Camera → World using extrinsic (C2W matrix)
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        points_world = (R @ points_cam.T).T + t
        
        return points_world
    
    def _to_voxel_indices(
        self, 
        points_3d: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert 3D points to voxel indices"""
        if len(points_3d) == 0:
            return None, None
       
        points_tensor = torch.tensor(
            points_3d, 
            dtype=torch.float32, 
            device=self.device
        )
        voxel_indices, valid_mask = self.voxel_map._voxelize(points_tensor)
        
        num_valid = torch.sum(valid_mask).item()
        if num_valid == 0:
            return None, None
        
        valid_indices = voxel_indices[valid_mask]
        linear_indices = self.voxel_map.to_linear_indices(valid_indices)
        linear_indices_unique = torch.unique(linear_indices)
        
        return valid_indices.cpu().numpy(), linear_indices_unique.cpu().numpy()
