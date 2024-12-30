import albumentations as A

class Config:
    # Existing settings
    # Model settings
    model_name = 'efficientnet-b3'
    image_size = 512
    
    # Training settings
    seed = 2023
    batch_size = 32
    epochs = 40
    warmup_epochs = 2
    base_lr = 2e-4
    min_lr = 1e-6
    weight_decay = 0.01
    num_workers = 6
    
    # Data paths
    train_data_dir = '/storage/scratch2/sandaruwanh/rsna_lg'
    bbox_csv_path = 'lung_bbox.csv'
    
    # Pickle file paths
    series_list_path = 'process_input/splitall/series_list_train.pickle'
    series_list_valid_path = 'process_input/splitall/series_list_valid.pickle'
    series_dict_path = 'process_input/splitall/series_dict.pickle'
    image_dict_path = 'process_input/splitall/image_dict.pickle'
    
    # Checkpoint settings
    save_frequency = 5
    val_frequency = 1
    
    # New bounding box related parameters
    BBOX_SIZE_PENALTY_WEIGHT = 0.1  # Weight for the size penalty in BBoxLoss
    BBOX_MIN_SIZE = 0.2             # Minimum allowed size for predicted boxes
    BBOX_SHRINK_FACTOR = 0.95       # Factor to shrink predictions during post-processing
    
    # Test Time Augmentation settings
    USE_TTA = True                  # Whether to use test-time augmentation
    TTA_ANGLES = [-5, -2.5, 2.5, 5] # Rotation angles for TTA
    
    # Loss weights
    BBOX_LOSS_WEIGHT = 0.5          # Weight for bbox loss in combined loss
    IOU_LOSS_WEIGHT = 0.5           # Weight for IoU loss in combined loss
    
    # Existing augmentation settings
    train_transforms = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.8),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        ], p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=15, 
            border_mode=0, 
            p=0.5
        ),
        A.GridDistortion(p=0.3),
        A.CoarseDropout(
            max_holes=2,
            max_height=int(0.3*image_size),
            max_width=int(0.3*image_size),
            min_height=int(0.1*image_size),
            min_width=int(0.1*image_size),
            fill_value=0,
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
    ], bbox_params=A.BboxParams(
        format='albumentations',
        label_fields=['class_labels']
    ))
    
    # Validation transforms (for TTA)
    val_transforms = A.Compose([
        # Minimal transforms for validation
        A.Resize(height=image_size, width=image_size),
    ], bbox_params=A.BboxParams(
        format='albumentations',
        label_fields=['class_labels']
    ))
    GRAD_CLIP_NORM = 1.0