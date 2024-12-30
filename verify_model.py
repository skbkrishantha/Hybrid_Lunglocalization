import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from models.efficient_net import ImprovedEfficientNet

def verify_predictions(model, device):
    """Verify model predictions are valid"""
    model.eval()
    with torch.no_grad():
        # Test input
        x = torch.randn(5, 3, 512, 512).to(device)
        
        # Get predictions
        pred = model(x)
        
        print("\nPrediction Verification:")
        print(f"Shape: {pred.shape}")
        print("\nCoordinate ranges:")
        for i, name in enumerate(['x1', 'y1', 'x2', 'y2']):
            values = pred[:, i].cpu().numpy()
            print(f"{name}: [{values.min():.3f}, {values.max():.3f}]")
            
        # Verify box validity
        valid = True
        for i in range(len(pred)):
            x1, y1, x2, y2 = pred[i].cpu().numpy()
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                valid = False
                print(f"\nInvalid box {i}: {[x1, y1, x2, y2]}")
        
        print(f"\nAll boxes valid: {valid}")
        return valid

def visualize_sample_predictions(model, device):
    """Visualize sample predictions"""
    model.eval()
    with torch.no_grad():
        # Create sample images (random noise for testing)
        samples = torch.randn(5, 3, 512, 512).to(device)
        
        # Get predictions
        preds = model(samples)
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(min(5, len(preds))):
            # Get predictions
            x1, y1, x2, y2 = preds[i].cpu().numpy()
            
            # Create simple visualization
            img = samples[i].cpu().numpy()[0]  # Take first channel
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
            
            axes[i].imshow(img, cmap='gray')
            
            # Draw predicted box
            w, h = 512, 512
            rect = plt.Rectangle((x1*w, y1*h), (x2-x1)*w, (y2-y1)*h,
                               fill=False, color='r', linewidth=2)
            axes[i].add_patch(rect)
            
            axes[i].set_title(f'Sample {i}\nBox: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create new model
    print("\nCreating new model...")
    model = ImprovedEfficientNet('efficientnet-b3').to(device)
    
    # Verify predictions
    print("\nVerifying predictions...")
    is_valid = verify_predictions(model, device)
    
    if is_valid:
        print("\nModel predictions are valid!")
        print("\nVisualizing sample predictions...")
        visualize_sample_predictions(model, device)
        
        # Save the model if it's working correctly
        save_path = 'weights/verified_model.pth'
        torch.save(model.state_dict(), save_path)
        print(f"\nSaved verified model to {save_path}")
    else:
        print("\nModel predictions are invalid! Please check the model architecture.")

if __name__ == "__main__":
    main()