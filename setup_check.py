import os
import sys
from pathlib import Path

def check_directories():
    required_dirs = [
        'weights',
        'models',
        'utils'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Creating missing directory: {dir_name}")
            os.makedirs(dir_name)
    return True

def check_files():
    required_files = [
        'config.py',
        'train.py',
        'dataset.py',
        'models/__init__.py',
        'models/efficient_net.py',
        'models/attention.py',
        'utils/__init__.py',
        'utils/metrics.py',
    ]
    
    missing_files = []
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_files:
        print("Missing files:")
        for file_name in missing_files:
            print(f"  - {file_name}")
        return False
    return True

def check_data_paths():
    try:
        from config import Config
        
        paths_to_check = [
            (Config.train_data_dir, "Training data directory"),
            (Config.bbox_csv_path, "Bounding box CSV file"),
            (Config.series_list_path, "Series list pickle file"),
            (Config.series_dict_path, "Series dict pickle file"),
            (Config.image_dict_path, "Image dict pickle file")
        ]
        
        missing_paths = []
        for path, description in paths_to_check:
            if not os.path.exists(path):
                missing_paths.append((path, description))
        
        if missing_paths:
            print("\nWarning: Some data paths are not found:")
            for path, description in missing_paths:
                print(f"  - {description}: '{path}'")
            print("\nPlease verify these paths in config.py match your data directory structure.")
            return False
        return True
    except ImportError as e:
        print(f"Error importing Config: {e}")
        return False
    except Exception as e:
        print(f"Error checking data paths: {e}")
        return False

def check_gpu_availability():
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. GPU training will not be possible!")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        
        return True
    except ImportError:
        print("Error: PyTorch is not installed!")
        return False

def check_dependencies():
    required_packages = [
        'torch',
        'torchvision',
        'albumentations',
        'efficientnet_pytorch',
        'opencv-python',
        'pydicom',
        'tqdm',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\nMissing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def main():
    print("Running setup checks...")
    
    checks = [
        (check_dependencies, "Checking required packages..."),
        (check_directories, "Checking required directories..."),
        (check_files, "Checking required files..."),
        (check_data_paths, "Checking data paths..."),
        (check_gpu_availability, "Checking GPU availability...")
    ]
    
    all_passed = True
    for check_func, message in checks:
        print(f"\n{message}")
        if not check_func():
            all_passed = False
            
    if all_passed:
        print("\nAll checks passed! You can start training using ./run.sh")
    else:
        print("\nSome checks failed. Please fix the issues before starting training.")
        print("If data paths are the only failing checks, verify the paths in config.py")

if __name__ == "__main__":
    main()