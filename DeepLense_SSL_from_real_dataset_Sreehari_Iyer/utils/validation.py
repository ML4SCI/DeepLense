import os
import glob

def validate_ssl_dataset(root_dir, hr_folder, lr_folder):
    """
    Validates that HR and LR .npy files exist and match.
    Directly addresses the 'Inconsistent Dataset Handling' issue.
    """
    hr_path = os.path.join(root_dir, hr_folder)
    lr_path = os.path.join(root_dir, lr_folder)

    if not os.path.exists(hr_path) or not os.path.exists(lr_path):
        raise FileNotFoundError(f"Dataset folders not found at {root_dir}")

    hr_files = sorted(glob.glob(os.path.join(hr_path, "*.npy")))
    lr_files = sorted(glob.glob(os.path.join(lr_path, "*.npy")))

    if len(hr_files) == 0:
        raise ValueError(f"No .npy files found in {hr_path}")
    
    if len(hr_files) != len(lr_files):
        raise ValueError(f"Mismatch: Found {len(hr_files)} HR and {len(lr_files)} LR files.")

    print(f"✅ Validation Successful: {len(hr_files)} pairs ready for training.")
    return True