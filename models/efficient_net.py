import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from .attention import SpatialAttention 

def check_state_dict():
    """Check structure of saved state dict"""
    state_dict = torch.load('weights/verified_model.pth', weights_only=True)
    print("\nState dict keys:")
    for key in sorted(state_dict.keys()):
        print(key)
    return state_dict

# models/efficient_net.py

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from .attention import SpatialAttention
from config import Config

class ImprovedEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b3'):
        super().__init__()
        self.net = EfficientNet.from_pretrained(model_name)
        in_features = self.net._fc.in_features
        self.spatial_attention = SpatialAttention()
        
        # Add learnable scaling parameters
        self.scale_factors = nn.Parameter(torch.tensor([0.4, 0.4, 0.6, 0.6]))
        self.offsets = nn.Parameter(torch.tensor([0.1, 0.1, 0.5, 0.5]))
        
        self.last_linear = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(),
            nn.BatchNorm1d(in_features//2),
            nn.Dropout(0.3),
            nn.Linear(in_features//2, 4)
        )
        
        # Freeze batch norm layers
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        # Extract features
        features = self.net.extract_features(x)
        
        # Apply attention
        attention = self.spatial_attention(features)
        features = features * attention
        
        # Global average pooling
        x = self.net._avg_pooling(features)
        x = x.view(x.size(0), -1)
        
        # Get predictions
        x = self.last_linear(x)
        
        # Create new tensor for processed predictions
        batch_size = x.size(0)
        processed_pred = torch.zeros(batch_size, 4, device=x.device, dtype=x.dtype)
        
        # Apply learned scaling and offset without in-place operations
        processed_pred = torch.stack([
            torch.sigmoid(x[:, 0]) * self.scale_factors[0] + self.offsets[0],
            torch.sigmoid(x[:, 1]) * self.scale_factors[1] + self.offsets[1],
            torch.sigmoid(x[:, 2]) * self.scale_factors[2] + self.offsets[2],
            torch.sigmoid(x[:, 3]) * self.scale_factors[3] + self.offsets[3]
        ], dim=1)
        
        # Ensure minimum box size without in-place operations
        min_size = Config.BBOX_MIN_SIZE
        x2 = torch.maximum(processed_pred[:, 2], processed_pred[:, 0] + min_size)
        y2 = torch.maximum(processed_pred[:, 3], processed_pred[:, 1] + min_size)
        
        processed_pred = torch.stack([
            processed_pred[:, 0],
            processed_pred[:, 1],
            x2,
            y2
        ], dim=1)
        
        return processed_pred

    def inference(self, x):
            with torch.no_grad():
                predictions = self.forward(x)
                adjusted_predictions = adjust_predictions(predictions)
            return adjusted_predictions



def load_model(checkpoint_path='weights/verified_model.pth'):
    """Load model with proper error handling"""
    try:
        # Check saved state structure
        print("Checking saved state...")
        state_dict = check_state_dict()
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImprovedEfficientNet('efficientnet-b3').to(device)
        
        # Try loading weights with strict=False first
        print("\nLoading weights...")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("\nWarning: Some keys didn't match:")
            if incompatible.missing_keys:
                print("Missing keys:", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                print("Unexpected keys:", incompatible.unexpected_keys)
        
        model.eval()
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def validate_on_test_data():
    """Validate model on real test data"""
    # Load model
    model, device = load_model()
    if model is None:
        print("Failed to load model!")
        return None, None
    
    # Rest of validation code...
    # (rest of the function remains the same)

# Check state dict and try loading
if __name__ == "__main__":
    print("Checking state dict structure...")
    state_dict = check_state_dict()
    
    print("\nTrying to load model...")
    model, device = load_model()
    
    if model is not None:
        print("\nValidating model...")
        results, ious = validate_on_test_data()