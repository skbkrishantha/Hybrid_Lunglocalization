import os
import glob
import pickle
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

class DataPreprocessor:
    def __init__(self):
        self.train_dir = '/storage/scratch2/sandaruwanh/rsna_lg/train'
        self.output_dir = ensure_directory('process_input/splitall')
        self.lung_bbox_path = 'lung_bbox.csv'
        
    def create_series_dict(self):
        """Creates a dictionary containing information about each series"""
        series_dict = {}
        
        # Load bbox data
        print("Loading bbox data...")
        bbox_df = pd.read_csv(self.lung_bbox_path)
        valid_image_ids = set(bbox_df['Image'].unique())
        print(f"Found {len(valid_image_ids)} images with bounding boxes")
        
        study_dirs = sorted(glob.glob(os.path.join(self.train_dir, '*')))
        print(f"Found {len(study_dirs)} study directories")
        
        print("Creating series dictionary...")
        for study_dir in tqdm(study_dirs):
            study_id = os.path.basename(study_dir)
            series_dirs = sorted(glob.glob(os.path.join(study_dir, '*')))
            
            for series_dir in series_dirs:
                series_id = os.path.basename(series_dir)
                series_key = f"{study_id}_{series_id}"
                
                dicom_files = sorted(glob.glob(os.path.join(series_dir, '*.dcm')))
                if not dicom_files:
                    continue
                
                try:
                    z_positions = []
                    image_list = []
                    
                    for dcm_file in dicom_files:
                        image_id = os.path.splitext(os.path.basename(dcm_file))[0]
                        if image_id in valid_image_ids:
                            try:
                                dcm = pydicom.dcmread(dcm_file)
                                z_positions.append(float(dcm.ImagePositionPatient[2]))
                                image_list.append(image_id)
                            except Exception as e:
                                print(f"Error reading DICOM {image_id}: {str(e)}")
                    
                    if image_list:
                        sorted_indices = np.argsort(z_positions)
                        sorted_image_list = [image_list[i] for i in sorted_indices]
                        sorted_z_positions = [z_positions[i] for i in sorted_indices]
                        
                        series_dict[series_key] = {
                            'study_id': study_id,
                            'series_id': series_id,
                            'image_list': image_list,
                            'sorted_image_list': sorted_image_list,
                            'z_positions': sorted_z_positions,
                            'num_images': len(sorted_image_list)
                        }
                    
                except Exception as e:
                    print(f"Error processing series {series_key}: {str(e)}")
                    continue
        
        print(f"\nProcessing summary:")
        print(f"Total series processed: {len(series_dict)}")
        return series_dict
    
    def create_image_dict(self, series_dict):
        """Creates a dictionary containing information about each image"""
        image_dict = {}
        print("\nCreating image dictionary...")
        
        for series_key, series_info in tqdm(series_dict.items()):
            for idx, image_id in enumerate(series_info['sorted_image_list']):
                image_dict[image_id] = {
                    'series_id': series_key,
                    'z_position': series_info['z_positions'][idx],
                    'image_index': idx,
                    'total_images': series_info['num_images']
                }
        
        return image_dict
    
    def create_train_val_split(self, series_dict, train_ratio=0.8, random_seed=42):
        """Creates train/validation split of series"""
        series_list = list(series_dict.keys())
        
        print(f"\nPreparing train/val split:")
        print(f"Total series available: {len(series_list)}")
        
        # Sort by number of images
        series_list.sort(key=lambda x: series_dict[x]['num_images'], reverse=True)
        
        np.random.seed(random_seed)
        np.random.shuffle(series_list)
        
        split_idx = int(len(series_list) * train_ratio)
        train_series = series_list[:split_idx]
        val_series = series_list[split_idx:]
        
        print(f"Split details:")
        print(f"Training series: {len(train_series)}")
        print(f"Validation series: {len(val_series)}")
        
        return train_series, val_series
    
    def process(self):
        """Main processing function"""
        print(f"Starting preprocessing...")
        print(f"Train directory: {self.train_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Create dictionaries
        series_dict = self.create_series_dict()
        print(f"\nFound {len(series_dict)} valid series")
        
        if not series_dict:
            raise ValueError("No valid series found! Check your data paths and criteria.")
        
        image_dict = self.create_image_dict(series_dict)
        print(f"Processed {len(image_dict)} images")
        
        # Create train/val split
        train_series, val_series = self.create_train_val_split(series_dict)
        
        # Save processed data
        print("\nSaving processed data...")
        save_data = {
            'series_dict.pickle': series_dict,
            'image_dict.pickle': image_dict,
            'series_list_train.pickle': train_series,
            'series_list_valid.pickle': val_series
        }
        
        for filename, data in save_data.items():
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {filepath}")
        
        print("\nPreprocessing complete!")
        
        # Print final statistics
        print("\nFinal Statistics:")
        print(f"Total series: {len(series_dict)}")
        print(f"Total images: {len(image_dict)}")
        print(f"Training series: {len(train_series)}")
        print(f"Validation series: {len(val_series)}")
        
        return series_dict, image_dict, train_series, val_series


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    try:
        preprocessor.process()
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        import traceback
        print(traceback.format_exc())