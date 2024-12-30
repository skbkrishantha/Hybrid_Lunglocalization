class TrainingConfig:
    # Data paths
    train_data_dir = '/storage/scratch2/sandaruwanh/rsna_lg'
    train_csv = '../balanced_train.csv'
    
    # Model parameters
    efficient_weights = 'weights/model_epoch_40.pth'
    image_size = 512
    
    # Training parameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    num_workers = 4
    
    # Loss weights
    bbox_weight = 0.5
    seg_weight = 0.5
    
    # Save paths
    output_dir = 'hybrid_model_output'
    checkpoints_dir = 'hybrid_model_output/checkpoints'