# utils/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class IoULoss(nn.Module):
    """IoU Loss for bounding box regression"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        # Calculate intersection areas
        x1 = torch.max(pred[:, 0], target[:, 0])
        y1 = torch.max(pred[:, 1], target[:, 1])
        x2 = torch.min(pred[:, 2], target[:, 2])
        y2 = torch.min(pred[:, 3], target[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = pred_area + target_area - intersection + self.eps
        
        # Calculate IoU
        iou = intersection / union
        loss = 1 - iou
        
        return loss.mean()

class BBoxLoss(nn.Module):
    """Combined bounding box loss with size penalty"""
    def __init__(self, size_penalty_weight=0.1, eps=1e-6):
        super().__init__()
        self.size_penalty_weight = size_penalty_weight
        self.eps = eps
    
    def forward(self, pred, target):
        # Create new tensors instead of modifying in place
        pred_detached = pred.clone()
        target_detached = target.clone()
        
        # Basic coordinate loss (L1)
        coord_loss = F.l1_loss(pred_detached, target_detached)
        
        # Calculate box sizes
        pred_width = pred_detached[:, 2] - pred_detached[:, 0]
        pred_height = pred_detached[:, 3] - pred_detached[:, 1]
        target_width = target_detached[:, 2] - target_detached[:, 0]
        target_height = target_detached[:, 3] - target_detached[:, 1]
        
        # Penalize oversized predictions
        width_penalty = torch.maximum(pred_width - target_width, torch.zeros_like(pred_width))
        height_penalty = torch.maximum(pred_height - target_height, torch.zeros_like(pred_height))
        size_penalty = torch.mean(width_penalty + height_penalty)
        
        # Combined loss
        total_loss = coord_loss + self.size_penalty_weight * size_penalty
        
        return total_loss