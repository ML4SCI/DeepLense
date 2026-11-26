"""
Unsupervised Super-Resolution Training Script for DEEPLENSE2
PR #2: Implements autoencoder-based SR on simulated lensing images

Key differences from PR #1 (train_srcnn_minimal.py):
- Unsupervised (no paired LR/HR needed)
- Uses autoencoder architecture (DEEPLENSE2 requirement)
- Works with real DeepLense simulation data
- Includes evaluation metrics (PSNR, SSIM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# Import our autoencoder components
# (These will be in the same directory)
try:
    from autoencoder_sr import (
        SuperResolutionAutoencoder,
        PerceptualLoss,
        apply_degradation
    )
except ImportError:
    print("⚠ autoencoder_sr.py not found. Make sure it's in the same directory.")
    print("  Defining minimal versions here...")
    
    # Minimal fallback if import fails
    class SuperResolutionAutoencoder(nn.Module):
        def __init__(self, in_channels=1, base_channels=64):
            super().__init__()
            # Simplified version - use the full version from autoencoder_sr.py
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, in_channels, 3, padding=1),
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    class PerceptualLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()
        
        def forward(self, output, target):
            loss = self.mse(output, target)
            return loss, loss, torch.tensor(0.0)
    
    def apply_degradation(img, scale_factor=2, noise_level=0.05):
        import torch.nn.functional as F
        b, c, h, w = img.shape
        lr_size = (h // scale_factor, w // scale_factor)
        img_lr = F.interpolate(img, size=lr_size, mode='bilinear', align_corners=False)
        if noise_level > 0:
            noise = torch.randn_like(img_lr) * noise_level
            img_lr = img_lr + noise
        img_degraded = F.interpolate(img_lr, size=(h, w), mode='bicubic', align_corners=False)
        return img_degraded


# ============================================================================
# DATASET
# ============================================================================

class DeepLenseDataset(Dataset):
    """
    Dataset for DeepLense simulations
    
    Supports two modes:
    1. Real data: Load from .npy files (Model I/II/III)
    2. Synthetic: Generate dummy Einstein rings (fallback)
    """
    
    def __init__(self, data_path=None, n_samples=100, img_size=64, use_synthetic=False):
        self.img_size = img_size
        
        if use_synthetic or data_path is None:
            print(f"📊 Generating {n_samples} synthetic lensing images...")
            self.data = self._generate_synthetic_data(n_samples)
            self.source = "synthetic"
        else:
            print(f"📊 Loading data from: {data_path}")
            self.data = self._load_real_data(data_path)
            self.source = "real"
        
        print(f"✅ Dataset ready: {self.data.shape} ({self.source})")
    
    def _generate_synthetic_data(self, n_samples):
        """Generate synthetic Einstein rings (same as PR #1)"""
        images = []
        
        for i in range(n_samples):
            x = np.linspace(-1, 1, self.img_size)
            y = np.linspace(-1, 1, self.img_size)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            
            ring_radius = 0.4 + np.random.rand() * 0.3
            ring_width = 0.05 + np.random.rand() * 0.05
            ring = np.exp(-((R - ring_radius)**2) / ring_width)
            
            background = np.random.rand() * 0.3
            noise = np.random.randn(self.img_size, self.img_size) * 0.05
            
            img = ring + background + noise
            img = np.clip(img, 0, 1)
            images.append(img[None, ...])
        
        return torch.FloatTensor(np.array(images))
    
    def _load_real_data(self, data_path):
        """Load real DeepLense simulation data"""
        data_path = Path(data_path)
        
        if data_path.is_file() and data_path.suffix == '.npy':
            # Single .npy file
            data = np.load(data_path)
        elif data_path.is_dir():
            # Directory of .npy files
            npy_files = list(data_path.glob('*.npy'))
            if not npy_files:
                raise ValueError(f"No .npy files found in {data_path}")
            print(f"   Found {len(npy_files)} .npy files, using first one")
            data = np.load(npy_files[0])
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        
        # Ensure proper shape: (N, 1, H, W)
        if len(data.shape) == 3:
            data = data[:, None, :, :]
        elif len(data.shape) == 4 and data.shape[1] != 1:
            # If multiple channels, take first
            data = data[:, :1, :, :]
        
        # Resize if needed
        if data.shape[2] != self.img_size or data.shape[3] != self.img_size:
            print(f"   Resizing from {data.shape[2]}x{data.shape[3]} to {self.img_size}x{self.img_size}")
            import torch.nn.functional as F
            data_tensor = torch.FloatTensor(data)
            data_tensor = F.interpolate(data_tensor, size=(self.img_size, self.img_size), mode='bilinear')
            data = data_tensor.numpy()
        
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        return torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio
    Higher is better (typically 20-50 dB)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Calculate Structural Similarity Index
    Range: [-1, 1], higher is better
    Simplified version - full SSIM implementation is more complex
    """
    # Simplified SSIM using correlation
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    sigma1 = torch.std(img1)
    sigma2 = torch.std(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    
    return ssim.item()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_autoencoder(args):
    """Main training loop"""
    
    print("="*70)
    print("UNSUPERVISED AUTOENCODER SR - DEEPLENSE2 PR #2")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\n🖥  Device: {device}")
    
    # Dataset
    print(f"\n📊 Loading dataset...")
    dataset = DeepLenseDataset(
        data_path=args.data_path,
        n_samples=args.n_samples,
        img_size=args.img_size,
        use_synthetic=args.use_synthetic
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Model
    print(f"\n🏗  Building model...")
    model = SuperResolutionAutoencoder(
        in_channels=1,
        base_channels=args.base_channels
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = PerceptualLoss(alpha=args.alpha, beta=args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'percep_loss': [],
        'psnr': [],
        'ssim': []
    }
    
    model.train()
    
    for epoch in range(args.epochs):
        epoch_losses = {'total': 0, 'recon': 0, 'percep': 0}
        epoch_metrics = {'psnr': 0, 'ssim': 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, hr_imgs in enumerate(pbar):
            hr_imgs = hr_imgs.to(device)
            
            # Apply degradation (unsupervised: create LR from HR)
            lr_imgs = apply_degradation(
                hr_imgs,
                scale_factor=args.scale_factor,
                noise_level=args.noise_level
            )
            
            # Forward pass
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            
            # Calculate loss
            total_loss, recon_loss, percep_loss = criterion(sr_imgs, hr_imgs)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['percep'] += percep_loss.item()
            
            # Calculate metrics (on first batch)
            if batch_idx == 0:
                with torch.no_grad():
                    psnr = calculate_psnr(sr_imgs, hr_imgs)
                    ssim = calculate_ssim(sr_imgs[0], hr_imgs[0])
                    epoch_metrics['psnr'] = psnr
                    epoch_metrics['ssim'] = ssim
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'psnr': f"{epoch_metrics['psnr']:.2f}dB"
            })
        
        # Average losses
        n_batches = len(dataloader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        # Store history
        history['total_loss'].append(avg_losses['total'])
        history['recon_loss'].append(avg_losses['recon'])
        history['percep_loss'].append(avg_losses['percep'])
        history['psnr'].append(epoch_metrics['psnr'])
        history['ssim'].append(epoch_metrics['ssim'])
        
        # Learning rate scheduling
        scheduler.step(avg_losses['total'])
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"  Loss: {avg_losses['total']:.6f} (Recon: {avg_losses['recon']:.6f}, Percep: {avg_losses['percep']:.6f})")
        print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB, SSIM: {epoch_metrics['ssim']:.4f}")
        print()
    
    print(f"✅ Training complete!\n")
    
    # Save model
    if args.save_model:
        Path(args.output_dir).mkdir(exist_ok=True)
        model_path = Path(args.output_dir) / "autoencoder_sr_deeplense.pth"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved: {model_path}\n")
    
    # Visualize results
    visualize_results(model, dataset, device, args.output_dir, history)
    
    return model, history


def visualize_results(model, dataset, device, output_dir, history):
    """Create comprehensive visualizations"""
    
    print(f"📊 Creating visualizations...")
    
    model.eval()
    Path(output_dir).mkdir(exist_ok=True)
    
    with torch.no_grad():
        # Get sample
        hr_img = dataset[0].unsqueeze(0).to(device)
        lr_img = apply_degradation(hr_img)
        sr_img = model(lr_img)
        
        # Move to CPU
        hr_img = hr_img.cpu()
        lr_img = lr_img.cpu()
        sr_img = sr_img.cpu()
        
        # Calculate metrics
        psnr_lr = calculate_psnr(lr_img, hr_img)
        psnr_sr = calculate_psnr(sr_img, hr_img)
        ssim_lr = calculate_ssim(lr_img[0], hr_img[0])
        ssim_sr = calculate_ssim(sr_img[0], hr_img[0])
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(lr_img[0, 0], cmap='viridis')
        axes[0].set_title(f'Low Resolution\nPSNR: {psnr_lr:.2f}dB, SSIM: {ssim_lr:.3f}')
        axes[0].axis('off')
        
        axes[1].imshow(sr_img[0, 0], cmap='viridis')
        axes[1].set_title(f'Super-Resolved (Autoencoder)\nPSNR: {psnr_sr:.2f}dB, SSIM: {ssim_sr:.3f}')
        axes[1].axis('off')
        
        axes[2].imshow(hr_img[0, 0], cmap='viridis')
        axes[2].set_title('High Resolution (Target)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "autoencoder_sr_result.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(history['psnr'])
        axes[0, 1].set_title('PSNR (Higher is better)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('dB')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(history['recon_loss'], label='Reconstruction')
        axes[1, 0].plot(history['percep_loss'], label='Perceptual')
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(history['ssim'])
        axes[1, 1].set_title('SSIM (Higher is better)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}/")
    print(f"   - autoencoder_sr_result.png")
    print(f"   - training_curves.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unsupervised Autoencoder SR for DEEPLENSE2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data args
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to DeepLense .npy file or directory')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic data (fallback if no real data)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples (for synthetic data)')
    parser.add_argument('--img-size', type=int, default=64,
                        help='Image size')
    
    # Model args
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base number of channels in autoencoder')
    parser.add_argument('--scale-factor', type=int, default=2,
                        help='SR scale factor')
    parser.add_argument('--noise-level', type=float, default=0.05,
                        help='Noise level for degradation')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Weight for perceptual loss')
    
    # System args
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode')
    parser.add_argument('--output-dir', type=str, default='outputs_pr2',
                        help='Output directory')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model')
    
    args = parser.parse_args()
    
    # Run training
    model, history = train_autoencoder(args)
    
    print("="*70)
    print("DONE! PR #2 READY")
    print("="*70)


if __name__ == "__main__":
    main()