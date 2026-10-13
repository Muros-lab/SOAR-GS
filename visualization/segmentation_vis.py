"""
Segmentation visualization utilities
"""
import numpy as np
import cv2
from typing import List


def create_segmentation_visualization(
    rgb: np.ndarray,
    detections: List,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create colored segmentation overlay visualization
    
    Args:
        rgb: RGB image (H, W, 3), uint8
        detections: List of DetectionResult objects
        alpha: Blending factor for overlay (0.0 = transparent, 1.0 = opaque)
    
    Returns:
        Segmentation visualization (H, W, 3), uint8
    """
    if len(detections) == 0:
        # Return original RGB if no detections
        return rgb.copy()
    
    H, W = rgb.shape[:2]
    
    # Create overlay canvas
    overlay = rgb.copy()
    seg_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Define color palette (bright, distinguishable colors)
    color_palette = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
        [0, 128, 255],    # Light Blue
        [128, 255, 0],    # Lime
        [255, 0, 128],    # Pink
        [0, 255, 128],    # Spring Green
        [128, 128, 255],  # Lavender
        [255, 128, 128],  # Light Red
        [128, 255, 128],  # Light Green
        [128, 128, 255],  # Light Blue
        [200, 100, 50],   # Brown
        [50, 200, 100],   # Teal
        [100, 50, 200],   # Violet
        [200, 200, 50],   # Olive
    ]
    
    # Draw each detection
    for i, det in enumerate(detections):
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]
        
        # Get mask
        mask = det.mask > 0.5  # Binary mask
        
        # Fill mask with color
        seg_mask[mask] = color
        
        # Draw bounding box
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{det.class_name}: {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Background for text
        cv2.rectangle(
            overlay,
            (x1, y1 - text_h - 4),
            (x1 + text_w + 4, y1),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            overlay,
            label,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    # Blend segmentation mask with original image
    result = cv2.addWeighted(overlay, 1.0, seg_mask, alpha, 0)
    
    return result


def create_detection_visualization(
    rgb: np.ndarray,
    detections: List,
) -> np.ndarray:
    """
    Create detection-only visualization (bounding boxes + labels, no masks)
    
    Args:
        rgb: RGB image (H, W, 3), uint8
        detections: List of DetectionResult objects
    
    Returns:
        Detection visualization (H, W, 3), uint8
    """
    if len(detections) == 0:
        return rgb.copy()
    
    result = rgb.copy()
    
    # Same color palette as segmentation for consistency
    color_palette = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
        [0, 128, 255],    # Light Blue
        [128, 255, 0],    # Lime
        [255, 0, 128],    # Pink
        [0, 255, 128],    # Spring Green
    ]
    
    for i, det in enumerate(detections):
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]
        
        # Draw bounding box
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{det.class_name}: {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Background for text
        cv2.rectangle(
            result,
            (x1, y1 - text_h - 4),
            (x1 + text_w + 4, y1),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            result,
            label,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return result


def create_simple_segmentation_visualization(
    height: int,
    width: int,
    detections: List
) -> np.ndarray:
    """
    Create simple segmentation mask visualization without RGB overlay
    
    Args:
        height: Image height
        width: Image width
        detections: List of DetectionResult objects
    
    Returns:
        Segmentation visualization (H, W, 3), uint8
    """
    if len(detections) == 0:
        # Return black image if no detections
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    seg_vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Color palette
    color_palette = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [255, 128, 0], [128, 0, 255], [0, 128, 255],
        [128, 255, 0], [255, 0, 128], [0, 255, 128],
    ]
    
    # Draw each detection
    for i, det in enumerate(detections):
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]
        
        # Get mask
        mask = det.mask > 0.5
        
        # Fill mask with color
        seg_vis[mask] = color
    
    return seg_vis
