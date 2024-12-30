import os
import pickle
import numpy as np

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def verify_preprocessing():
    base_dir = 'process_input/splitall'
    required_files = [
        'series_dict.pickle',
        'image_dict.pickle',
        'series_list_train.pickle',
        'series_list_valid.pickle'
    ]
    
    print("Verifying preprocessed data...")
    
    # Check if files exist
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        print("\nError: Missing required files:")
        for filename in missing_files:
            print(f"- {filename}")
        return False
    
    # Load and verify data
    try:
        series_dict = load_pickle(os.path.join(base_dir, 'series_dict.pickle'))
        image_dict = load_pickle(os.path.join(base_dir, 'image_dict.pickle'))
        train_series = load_pickle(os.path.join(base_dir, 'series_list_train.pickle'))
        valid_series = load_pickle(os.path.join(base_dir, 'series_list_valid.pickle'))
        
        print("\nData statistics:")
        print(f"Total series: {len(series_dict)}")
        print(f"Total images: {len(image_dict)}")
        print(f"Training series: {len(train_series)}")
        print(f"Validation series: {len(valid_series)}")
        
        # Verify data consistency
        assert all(series_id in series_dict for series_id in train_series), "Invalid training series"
        assert all(series_id in series_dict for series_id in valid_series), "Invalid validation series"
        
        # Verify image references
        for series_id, series_info in series_dict.items():
            assert all(img_id in image_dict for img_id in series_info['sorted_image_list']), \
                f"Missing images for series {series_id}"
        
        print("\nAll verifications passed!")
        return True
        
    except Exception as e:
        print(f"\nError verifying data: {str(e)}")
        return False

if __name__ == "__main__":
    verify_preprocessing()