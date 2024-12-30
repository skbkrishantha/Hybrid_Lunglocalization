import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.hybrid_model import HybridModel
from data_loader import HybridDataset
from config_hy import TrainingConfig as Config

def setup_training():
    """Setup training environment"""
    # Create output directories
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.checkpoints_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = HybridModel(efficient_weights_path=Config.efficient_weights)
    model = model.to(device)
    
    # Load data
    df = pd.read_csv(Config.train_csv)
    print(f"Total samples: {len(df)}")
    
    # Split data
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Create datasets
    train_dataset = HybridDataset(train_df, Config)
    val_dataset = HybridDataset(val_df, Config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )
    
    return model, train_loader, val_loader, device

def main():
    # Setup training
    model, train_loader, val_loader, device = setup_training()
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(os.path.join(Config.output_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(Config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (images, bboxes, masks) in enumerate(train_loader):
            images = images.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            pred_bbox, pred_mask = model(images)
            
            loss = model.compute_loss(pred_bbox, pred_mask, bboxes, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{Config.num_epochs} "
                      f"[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, bboxes, masks in val_loader:
                images = images.to(device)
                bboxes = bboxes.to(device)
                masks = masks.to(device)
                
                pred_bbox, pred_mask = model(images)
                loss = model.compute_loss(pred_bbox, pred_mask, bboxes, masks)
                val_loss += loss.item()
        
        # Log metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(Config.checkpoints_dir, 'best_model.pth'))
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(Config.checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        scheduler.step()
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()