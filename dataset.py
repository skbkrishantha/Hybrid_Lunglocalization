# File: dataset.py
import numpy as np
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
import glob
import os

def window(x, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return x

class LungDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size, transform=None):
        self.image_dict = image_dict
        self.bbox_dict = bbox_dict
        self.image_list = image_list
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data = pydicom.dcmread(f'/storage/scratch2/sandaruwanh/rsna_lg/train/{study_id}/{series_id}/{self.image_list[index]}.dcm')
        
        # Process image
        x = data.pixel_array.astype(np.float32)
        x = x * data.RescaleSlope + data.RescaleIntercept
        x1 = window(x, WL=100, WW=700)
        x = np.stack([x1, x1, x1], axis=2)
        x = cv2.resize(x, (self.target_size, self.target_size))
        
        # Get bounding box
        bboxes = [self.bbox_dict[self.image_list[index]]]
        class_labels = ['lung']
        
        if self.transform:
            transformed = self.transform(image=x, bboxes=bboxes, class_labels=class_labels)
            x = transformed['image']
            bboxes = transformed['bboxes']
        
        x = x.transpose(2, 0, 1)
        y = torch.tensor(bboxes[0], dtype=torch.float32)
        
        return torch.from_numpy(x), y