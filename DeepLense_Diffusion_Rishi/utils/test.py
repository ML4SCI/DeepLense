import torch
import numpy as np
import os
import torchvision.transforms as Transforms
import matplotlib.pyplot as plt

# Dataset paths defined in the issue
DATASET_PATHS = [
    '../Data/cdm_regress_multi_param_model_ii/cdm_regress_multi_param/',
    '../Data/npy_lenses-20240731T044737Z-001/npy_lenses/',
    '../Data/real_lenses_dataset/lenses'
]

def run_test_pipeline():
    # Ensure output directory exists for saving plots
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Modular transform pipeline
    preprocess = Transforms.Compose([
        Transforms.CenterCrop(64),
    ])

    for path in DATASET_PATHS:
        # Check if directory exists before processing
        if not os.path.exists(path):
            print(f"Skipping: {path} (Directory not found)")
            continue

        print(f"Processing: {path}")
        
        # Filter for .npy files
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        
        if not files:
            print(f"No .npy files found in {path}")
            continue

        # Load a sample file for testing
        sample_path = os.path.join(path, files[0])
        
        try:
            data = np.load(sample_path)
            
            # Simple min-max normalization
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            
            # Convert to tensor and apply transforms
            img_tensor = torch.from_numpy(data)
            img_tensor = preprocess(img_tensor)
            
            # Reorder dimensions if necessary for matplotlib (C, H, W) -> (H, W, C)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
            
            plt.imshow(img_tensor.numpy())
            plot_filename = f"test_{os.path.basename(os.path.normpath(path))}.jpg"
            plt.savefig(os.path.join("plots", plot_filename))
            print(f"Result saved to plots/{plot_filename}")
            
        except Exception as e:
            print(f"Could not process {files[0]}: {e}")

if __name__ == "__main__":
    run_test_pipeline()