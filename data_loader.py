import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import os

class HybridDataset(Dataset):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.base_path = os.path.join(config.train_data_dir, 'train')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Construct DICOM path
            study_id = row['StudyInstanceUID']
            series_id = row['SeriesInstanceUID']
            image_id = row['SOPInstanceUID']
            
            dcm_path = os.path.join(self.base_path, study_id, series_id, f"{image_id}.dcm")
            
            if not os.path.exists(dcm_path):
                raise FileNotFoundError(f"DICOM file not found: {dcm_path}")
            
            # Load and process image
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array.astype(np.float32)
            
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                image = image * dcm.RescaleSlope + dcm.RescaleIntercept
            
            image = (image - image.min()) / (image.max() - image.min())
            image = np.stack([image] * 3, axis=0)
            
            # Create binary mask with correct dimensions
            mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)
            
            # Get bounding box coordinates and create mask
            bbox = [0, 0, 1, 1]  # Replace with actual bbox coordinates
            y1, x1, y2, x2 = [int(coord * image.shape[1]) for coord in bbox]
            mask[0, y1:y2, x1:x2] = 1
            
            return (
                torch.FloatTensor(image),
                torch.FloatTensor(bbox),
                torch.FloatTensor(mask)
            )
            
        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            raise
    
    def create_mask(self, bbox, shape):
        """Create binary mask from bbox"""
        mask = np.zeros(shape, dtype=np.float32)
        h, w = shape
        x1, y1, x2, y2 = [
            int(bbox[0] * w),
            int(bbox[1] * h),
            int(bbox[2] * w),
            int(bbox[3] * h)
        ]
        mask[y1:y2, x1:x2] = 1
        return mask