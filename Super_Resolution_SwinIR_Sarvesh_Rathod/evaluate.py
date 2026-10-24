import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LensingDataset
from model import build_model
from pytorch_msssim import ssim
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'data_diff'
CHECKPOINT_PATH = 'swinir_epoch_10.pth' # Update this after training

def calculate_psnr(img1, img2):
    mse = nn.MSELoss()(img1, img2).item()
    if mse == 0:
        return 100
    return 10 * math.log10(1 / mse)

def evaluate():
    print(f"Evaluating on {DEVICE}")
    
    # Load Test Data
    # Assuming standard split logic or separate test file
    # For now, reusing the loader logic, but ideally we load 'test_HR.npy' if it existed
    # We will just split the training data again for demonstration if explicit test set isn't saved
    full_dataset = LensingDataset(
        hr_path=os.path.join(DATA_DIR, 'train_HR.npy'),
        lr_path=os.path.join(DATA_DIR, 'train_LR.npy')
    )
    
    # Use the last 10% as test (same as training split logic)
    test_size = int(0.1 * len(full_dataset))
    # Note: rigorous testing requires setting the seed or saving split indices
    # Here we just take the last chunk for simplicity as per common Kaggle practices found in this repo
    
    # Manual slicing for deterministic testing
    test_dataset = torch.utils.data.Subset(full_dataset, range(len(full_dataset)-test_size, len(full_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = build_model().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print("Warning: No checkpoint found. Evaluating untrained model.")
    
    model.eval()
    
    avg_psnr = 0.0
    avg_ssim = 0.0
    
    # Visualization buffer
    visuals = []
    
    with torch.no_grad():
        for i, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = model(lr)
            
            # Metrics
            psnr = calculate_psnr(sr, hr)
            # SSIM requires (N, C, H, W)
            ssim_val = ssim(sr, hr, data_range=1.0, size_average=True).item()
            
            avg_psnr += psnr
            avg_ssim += ssim_val
            
            # Save first 3 for visualization
            if i < 3:
                visuals.append((lr.cpu(), sr.cpu(), hr.cpu()))
                
    avg_psnr /= len(test_loader)
    avg_ssim /= len(test_loader)
    
    print(f"Test Results - Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, (lr, sr, hr) in enumerate(visuals):
        # LR
        axes[i, 0].imshow(lr[0, 0, :, :], cmap='magma')
        axes[i, 0].set_title(f"Low Res Input {i}")
        axes[i, 0].axis('off')
        
        # SR
        axes[i, 1].imshow(sr[0, 0, :, :], cmap='magma')
        axes[i, 1].set_title(f"SwinIR Output {i}")
        axes[i, 1].axis('off')
        
        # HR
        axes[i, 2].imshow(hr[0, 0, :, :], cmap='magma')
        axes[i, 2].set_title(f"Ground Truth {i}")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig('results.png')
    print("Saved visual comparison to results.png")

if __name__ == "__main__":
    evaluate()
