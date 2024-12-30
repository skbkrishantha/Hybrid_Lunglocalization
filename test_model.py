# test_model.py
import torch
from models.hybrid_model import HybridModel

def test_model_setup():
    # Initialize model
    model = HybridModel(efficient_weights_path='weights2/model_epoch_40.pth')
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Test forward pass
    bbox, mask = model(dummy_input)
    
    print("Model Output Shapes:")
    print(f"Bounding Box: {bbox.shape}")
    print(f"Segmentation Mask: {mask.shape}")

if __name__ == "__main__":
    test_model_setup()