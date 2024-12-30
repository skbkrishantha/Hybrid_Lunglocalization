import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.hybrid_model import HybridModel

def evaluate_hybrid_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    bbox_ious = []
    seg_dices = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, bbox_targets, mask_targets = batch
            images = images.to(device)
            bbox_targets = bbox_targets.to(device)
            mask_targets = mask_targets.to(device)
            
            # Get predictions
            pred_bbox, pred_mask = model(images)
            
            # Calculate metrics
            iou = calculate_iou(pred_bbox, bbox_targets)
            dice = calculate_dice(pred_mask, mask_targets)
            
            bbox_ious.extend(iou.cpu().numpy())
            seg_dices.extend(dice.cpu().numpy())
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average IoU: {np.mean(bbox_ious):.4f} ± {np.std(bbox_ious):.4f}")
    print(f"Average Dice: {np.mean(seg_dices):.4f} ± {np.std(seg_dices):.4f}")
    
    # Visualize results
    plot_results(bbox_ious, seg_dices)

def plot_results(ious, dices):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(ious, bins=50)
    plt.title('IoU Distribution')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(dices, bins=50)
    plt.title('Dice Score Distribution')
    plt.xlabel('Dice Score')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('hybrid_results.png')
    plt.close()

if __name__ == "__main__":
    # Load model
    model = HybridModel()
    model.load_state_dict(torch.load('best_hybrid_model.pth'))
    
    # Create test loader
    test_loader = create_data_loader(test_data)
    
    # Evaluate
    evaluate_hybrid_model(model, test_loader)