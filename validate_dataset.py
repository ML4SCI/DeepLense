import os
import yaml
import numpy as np

def load_config(config_path='config.yaml'):
    """Loads dataset parameters from the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def validate_data(config):
    """Validates the HR and LR datasets for consistency."""
    print("⏳ Starting dataset validation...")
    
    try:
        hr_path = config['dataset']['hr_path']
        lr_path = config['dataset']['lr_path']
        scale = config['dataset']['scale_factor']
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")

    if not os.path.exists(hr_path) or not os.path.exists(lr_path):
        raise FileNotFoundError("Dataset directories not found. Please check paths in config.yaml.")

    # Filter out hidden files (like macOS .DS_Store)
    hr_files = sorted([f for f in os.listdir(hr_path) if not f.startswith('.')])
    lr_files = sorted([f for f in os.listdir(lr_path) if not f.startswith('.')])

    # 1. Check Image Counts
    if len(hr_files) != len(lr_files):
        raise ValueError(f"❌ Count mismatch! HR contains {len(hr_files)} files, LR contains {len(lr_files)} files.")
    print(f"✅ Image counts match: {len(hr_files)} pairs found.")

    # 2. Check Filename Consistency
    for hr_f, lr_f in zip(hr_files, lr_files):
        if hr_f != lr_f:
            raise ValueError(f"❌ Filename mismatch detected: '{hr_f}' vs '{lr_f}'. Files must align perfectly.")
    print("✅ Filenames are strictly consistent across HR and LR folders.")

    # 3. Shape Validation (Sample first 5 files to save compute time)
    print("⏳ Checking image dimensions against scale factor...")
    for f in hr_files[:5]:
        if f.endswith('.npy'):
            hr_img = np.load(os.path.join(hr_path, f), allow_pickle=True)
            lr_img = np.load(os.path.join(lr_path, f), allow_pickle=True)
            
            # Handle object arrays vs standard arrays
            if hr_img.dtype == object:
                hr_img = np.stack([np.array(r) for r in hr_img])
                lr_img = np.stack([np.array(r) for r in lr_img])

            hr_shape = hr_img.shape[-1] # Assuming (C, H, W) or (H, W)
            lr_shape = lr_img.shape[-1]

            if hr_shape != lr_shape * scale:
                raise ValueError(f"❌ Shape mismatch in {f}! HR width ({hr_shape}) != LR width ({lr_shape}) * scale ({scale})")
                
    print("✅ Image shapes correctly match the scale factor.")
    print("\n🎉 Dataset validation passed successfully! Ready for training.")

if __name__ == "__main__":
    cfg = load_config()
    validate_data(cfg)