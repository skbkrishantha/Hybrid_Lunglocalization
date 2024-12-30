# utils/bbox_utils.py

import torch
import torchvision.transforms.functional as TF
from config import Config

def adjust_predictions(predictions, shrink_factor=Config.BBOX_SHRINK_FACTOR):
    """
    Apply post-processing adjustments to predictions
    """
    adjusted_pred = predictions.clone()
    
    # Calculate centers
    centers_x = (predictions[:, 0] + predictions[:, 2]) / 2
    centers_y = (predictions[:, 1] + predictions[:, 3]) / 2
    
    # Calculate widths and heights
    widths = predictions[:, 2] - predictions[:, 0]
    heights = predictions[:, 3] - predictions[:, 1]
    
    # Adjust box sizes
    adjusted_pred[:, 0] = centers_x - (widths * shrink_factor) / 2  # x1
    adjusted_pred[:, 1] = centers_y - (heights * shrink_factor) / 2 # y1
    adjusted_pred[:, 2] = centers_x + (widths * shrink_factor) / 2  # x2
    adjusted_pred[:, 3] = centers_y + (heights * shrink_factor) / 2 # y2
    
    return adjusted_pred

def rotate_bbox(bbox, angle):
    """
    Rotate bounding box coordinates
    """
    # Convert to center, width, height format
    center_x = (bbox[:, 0] + bbox[:, 2]) / 2
    center_y = (bbox[:, 1] + bbox[:, 3]) / 2
    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]
    
    # Convert angle to radians
    angle_rad = torch.tensor(angle) * torch.pi / 180
    
    # Create rotation matrix
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    
    # Rotate center point
    new_center_x = center_x * cos_theta - center_y * sin_theta
    new_center_y = center_x * sin_theta + center_y * cos_theta
    
    # Convert back to corner format
    rotated_bbox = torch.zeros_like(bbox)
    rotated_bbox[:, 0] = new_center_x - width / 2
    rotated_bbox[:, 1] = new_center_y - height / 2
    rotated_bbox[:, 2] = new_center_x + width / 2
    rotated_bbox[:, 3] = new_center_y + height / 2
    
    return rotated_bbox

def validate_with_tta(model, images, device=None):
    """
    Test-time augmentation for validation
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original prediction
        pred = model(images)
        predictions.append(pred)
        
        # Rotated predictions
        for angle in Config.TTA_ANGLES:
            rotated = TF.rotate(images, angle)
            pred = model(rotated)
            # Inverse rotate the predictions
            pred = rotate_bbox(pred, -angle)
            predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    final_pred = adjust_predictions(final_pred)
    
    return final_pred