import torch
import torch.nn as nn
import numpy as np
from .efficient_net import ImprovedEfficientNet 
from .unet import UNet

class HybridModel(nn.Module):
   def __init__(self, efficient_weights_path=None):
       super().__init__()
       
       # EfficientNet backbone
       self.efficient_net = ImprovedEfficientNet('efficientnet-b3')
       
       # U-Net for segmentation
       self.unet = UNet(in_channels=1, out_channels=1)
       
       # Refinement modules
       self.attention = nn.Sequential(
           nn.Conv2d(1, 16, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(16, 1, 1),
           nn.Sigmoid()
       )
       
       self.boundary_refine = nn.Sequential(
           nn.Conv2d(1, 32, 3, padding=1),
           nn.ReLU(), 
           nn.Conv2d(32, 16, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(16, 1, 1),
           nn.Sigmoid()
       )
       
       # Coordinate prediction
       self.coordinate_head = nn.Sequential(
           nn.Linear(8, 256),
           nn.ReLU(),
           nn.BatchNorm1d(256),
           nn.Dropout(0.2),
           nn.Linear(256, 64), 
           nn.ReLU(),
           nn.BatchNorm1d(64),
           nn.Linear(64, 4),
           nn.Sigmoid()
       )
       
       self._init_weights()
       
       if efficient_weights_path:
           self._load_weights(efficient_weights_path)
           
   def _init_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight)
               if m.bias is not None:
                   nn.init.zeros_(m.bias)
           elif isinstance(m, nn.Linear):
               nn.init.xavier_uniform_(m.weight)
               nn.init.zeros_(m.bias)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.ones_(m.weight)
               nn.init.zeros_(m.bias)
               
   def _load_weights(self, path):
       try:
           checkpoint = torch.load(path, map_location='cpu')
           state_dict = checkpoint.get('model_state_dict', checkpoint)
           cleaned_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
           self.efficient_net.load_state_dict(cleaned_dict, strict=False)
       except Exception as e:
           print(f"Warning loading weights: {str(e)}")

   def forward(self, x):
       # Initial bbox prediction
       bbox_features = self.efficient_net(x)
       
       # Segmentation with refinement
       x_gray = x[:, 0:1]
       mask = self.unet(x_gray)
       attention_mask = self.attention(mask)
       refined_mask = mask * attention_mask
       final_mask = self.boundary_refine(refined_mask)
       
       # Convert mask to bbox coordinates
       mask_bbox = self.mask_to_bbox_tensor(final_mask)
       
       # Combine and refine coordinates
       combined_features = torch.cat([bbox_features, mask_bbox], dim=1)
       final_bbox = self.coordinate_head(combined_features)
       
       # Ensure proper coordinate ordering
       x1, y1, x2, y2 = torch.split(final_bbox, 1, dim=1)
       x2 = x1 + torch.relu(x2 - x1) + 1e-3  # Ensure x2 > x1
       y2 = y1 + torch.relu(y2 - y1) + 1e-3  # Ensure y2 > y1
       final_bbox = torch.cat([x1, y1, x2, y2], dim=1)
       
       return final_bbox, final_mask

   @staticmethod
   def mask_to_bbox_tensor(mask):
       device = mask.device
       batch_size = mask.size(0)
       bbox = torch.zeros(batch_size, 4, device=device)
       
       for i in range(batch_size):
           mask_np = mask[i, 0].detach().cpu().numpy()
           rows = np.any(mask_np > 0.5, axis=1)
           cols = np.any(mask_np > 0.5, axis=0)
           
           if np.any(rows) and np.any(cols):
               rmin, rmax = np.where(rows)[0][[0, -1]]
               cmin, cmax = np.where(cols)[0][[0, -1]]
               
               h, w = mask_np.shape
               bbox_coords = [
                   max(0.0, min(1.0, cmin/w)),
                   max(0.0, min(1.0, rmin/h)),
                   max(0.0, min(1.0, cmax/w)),
                   max(0.0, min(1.0, rmax/h))
               ]
               bbox[i] = torch.tensor(bbox_coords, device=device)
       
       return bbox
    
    

   def compute_iou_loss(self, pred, target, eps=1e-6):
       x1 = torch.max(pred[:, 0], target[:, 0])
       y1 = torch.max(pred[:, 1], target[:, 1])
       x2 = torch.min(pred[:, 2], target[:, 2])
       y2 = torch.min(pred[:, 3], target[:, 3])
       
       intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
       pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
       target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
       union = pred_area + target_area - intersection + eps
       return 1 - (intersection / union).mean()
   
   def compute_loss(self, pred_bbox, pred_mask, target_bbox, target_mask):
        # Scaled losses
       l1_loss = nn.L1Loss()(pred_bbox, target_bbox) * 100
       iou_loss = self.compute_iou_loss(pred_bbox, target_bbox) * 10
       seg_loss = nn.BCELoss()(pred_mask, target_mask)
        
        # Add small epsilon to prevent complete zero
       total_loss = l1_loss + iou_loss + seg_loss + 1e-8
       return total_loss
   
   def get_boundary_weight(self, mask, kernel_size=3):
       # Compute boundary regions for weighted segmentation loss
        padded = F.pad(mask, [kernel_size//2]*4, mode='replicate')
        dilated = F.max_pool2d(padded, kernel_size, stride=1)
        eroded = -F.max_pool2d(-padded, kernel_size, stride=1)
        boundary = torch.abs(dilated - eroded)
        weight = 1.0 + 2.0 * boundary
        return weight

    

