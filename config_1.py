import albumentations as A

class Config:
    # Training settings
    seed = 2023
    image_size = 512
    batch_size = 32  # For RTX A6000 (48GB VRAM)
    epochs = 40
    warmup_epochs = 2
    base_lr = 1e-4  # Adjusted for dataset size
    min_lr = 1e-6
    weight_decay = 0.01
    num_workers = 6
    model_name = 'efficientnet-b3'
    
    # Data settings
    train_data_dir = '/storage/scratch2/sandaruwanh/rsna_lg/train'
    bbox_csv_path = 'lung_bbox.csv'
    series_list_path = 'process_input/splitall/series_list_train.pickle'
    series_dict_path = 'process_input/splitall/series_dict.pickle'
    image_dict_path = 'process_input/splitall/image_dict.pickle'
    
    # Let's adjust the training augmentations based on our dataset
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