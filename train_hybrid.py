import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.hybrid_model import HybridModel

class HybridLoss(nn.Module):
    def __init__(self, bbox_weight=0.4, mask_weight=0.4, boundary_weight=0.2):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.mask_weight = mask_weight 
        self.boundary_weight = boundary_weight
        
        self.bbox_loss = nn.L1Loss()
        self.mask_loss = nn.BCELoss()
        self.boundary_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred_bbox, pred_mask, target_bbox, target_mask):
        # Bbox loss
        bbox_loss = self.bbox_loss(pred_bbox, target_bbox)
        
        # Mask loss
        mask_loss = self.mask_loss(pred_mask, target_mask)
        
        # Boundary loss using image gradients
        pred_grad = self.compute_gradients(pred_mask)
        target_grad = self.compute_gradients(target_mask)
        boundary_loss = self.boundary_loss(pred_grad, target_grad)
        
        return (self.bbox_weight * bbox_loss + 
                self.mask_weight * mask_loss +
                self.boundary_weight * boundary_loss)

def train_hybrid_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            images, bbox_targets, mask_targets = batch
            images = images.to(device)
            bbox_targets = bbox_targets.to(device)
            mask_targets = mask_targets.to(device)
            
            optimizer.zero_grad()
            pred_bbox, pred_mask = model(images)
            loss = criterion(pred_bbox, pred_mask, bbox_targets, mask_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, bbox_targets, mask_targets = batch
                images = images.to(device)
                bbox_targets = bbox_targets.to(device)
                mask_targets = mask_targets.to(device)
                
                pred_bbox, pred_mask = model(images)
                loss = criterion(pred_bbox, pred_mask, bbox_targets, mask_targets)
                val_loss += loss.item()
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_hybrid_model.pth')
        
        scheduler.step()

if __name__ == "__main__":
    # Initialize model
    model = HybridModel(efficient_weights_path='weights2/model_epoch_40.pth')
    
    # Create data loaders (you'll need to implement this)
    train_loader = create_data_loader(train_data)
    val_loader = create_data_loader(val_data)
    
    # Train model
    train_hybrid_model(model, train_loader, val_loader)