"""
Minimal SRCNN (Super-Resolution CNN) for DeepLense

This script provides a baseline implementation of super-resolution for gravitational
lensing images. It supports the DEEPLENSE2 proposal by establishing reproducible
infrastructure for SR experiments.

Author: GSoC 2026 Contributor
Project: DEEPLENSE2 - Unsupervised Super-Resolution and Analysis of Real Lensing Images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# MODEL DEFINITION

class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN)
    
    A simple 3-layer CNN that learns an end-to-end mapping from low-resolution
    to high-resolution images. Originally proposed for natural images, adapted
    here for gravitational lensing observations.
    
    Architecture:
        - Conv1: Feature extraction (64 filters, 9x9)
        - Conv2: Non-linear mapping (32 filters, 5x5)  
        - Conv3: Reconstruction (num_channels filters, 5x5)
    
    Args:
        num_channels (int): Number of image channels (default: 1 for grayscale)
        
    Input:
        Low-resolution image upsampled to target size via bicubic interpolation
        
    Output:
        Super-resolved image at target resolution
        
    Reference:
        Dong et al. "Image Super-Resolution Using Deep Convolutional Networks"
        https://arxiv.org/abs/1501.00092
    """
    
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        
        # Feature extraction layer
        # Extracts overlapping patches and represents them as high-dimensional vectors
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping layer
        # Maps high-dimensional vectors to another space
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction layer
        # Aggregates predictions to produce final high-resolution image
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input LR image, shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Super-resolved image, shape (B, C, H, W)
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


# DUMMY DATASET (for Phase 1 - will replace with real data later)

class DummyLensingDataset(Dataset):
    """
    Generate synthetic gravitational lensing images for testing
    
    Creates Einstein ring patterns with realistic characteristics:
    - Variable ring radius (simulating different lens masses)
    - Random background sources (simulating source galaxies)
    - Gaussian noise (simulating observational noise)
    
    This dummy dataset allows pipeline testing without requiring large
    real datasets. Will be replaced with actual simulated/real lensing
    images in future PRs.
    
    Args:
        n_samples (int): Number of images to generate
        img_size (int): Size of images (square)
        scale_factor (int): SR scale factor (2 = 2x upsampling)
        
    Returns:
        Tuple of (LR_image, HR_image) where LR is upsampled to HR size
    """
    
    def __init__(self, n_samples=100, img_size=64, scale_factor=2):
        self.n_samples = n_samples
        self.img_size = img_size
        self.scale_factor = scale_factor
        
        print(f"Generating {n_samples} dummy lensing images...")
        self.hr_images = self._generate_lensing_images()
        self.lr_images = self._downsample(self.hr_images)
        print("✅ Dummy dataset ready")
        
    def _generate_lensing_images(self):
        """
        Create synthetic Einstein ring patterns
        
        Simulates strong gravitational lensing where a massive foreground
        object (lens) bends light from a background source, creating a ring.
        
        Returns:
            torch.Tensor: HR images, shape (n_samples, 1, img_size, img_size)
        """
        images = []
        
        for i in range(self.n_samples):
            # Create coordinate grid centered at origin
            x = np.linspace(-1, 1, self.img_size)
            y = np.linspace(-1, 1, self.img_size)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)  # Radial distance from center
            
            # Synthetic Einstein ring with variable parameters
            # ring_radius: Einstein radius (depends on lens mass and geometry)
            # ring_width: Thickness of ring (depends on source size)
            ring_radius = 0.4 + np.random.rand() * 0.3
            ring_width = 0.05 + np.random.rand() * 0.05
            ring = np.exp(-((R - ring_radius)**2) / ring_width)
            
            # Add background galaxies (random intensity)
            background = np.random.rand() * 0.3
            
            # Add observational noise (Gaussian)
            noise = np.random.randn(self.img_size, self.img_size) * 0.05
            
            # Combine and normalize
            img = ring + background + noise
            img = np.clip(img, 0, 1)
            
            images.append(img[None, ...])  # Add channel dimension
        
        return torch.FloatTensor(np.array(images))
    
    def _downsample(self, images):
        """
        Create LR images by downsampling HR images
        
        Mimics the observation process where we have lower resolution data
        and want to super-resolve it. Uses bicubic interpolation for both
        downsampling and upsampling back to original size.
        
        Args:
            images (torch.Tensor): HR images
            
        Returns:
            torch.Tensor: LR images upsampled to HR size (model input)
        """
        # Downsample to low resolution
        lr_size = self.img_size // self.scale_factor
        
        lr_images = torch.nn.functional.interpolate(
            images, 
            size=(lr_size, lr_size), 
            mode='bilinear',
            align_corners=False
        )
        
        # Upsample back to original size (this becomes model input)
        # The model learns to improve upon this bicubic upsampling
        lr_images_upsampled = torch.nn.functional.interpolate(
            lr_images,
            size=(self.img_size, self.img_size),
            mode='bicubic',
            align_corners=False
        )
        
        return lr_images_upsampled
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single LR/HR image pair
        
        Args:
            idx (int): Index of sample
            
        Returns:
            tuple: (LR_image, HR_image)
        """
        return self.lr_images[idx], self.hr_images[idx]



# TRAINING FUNCTION

def train_srcnn(args):
    """
    Main training loop for SRCNN
    
    Trains the model to minimize MSE between super-resolved and high-resolution
    images. This is a supervised approach; future PRs will explore unsupervised
    methods as per DEEPLENSE2 proposal.
    
    Args:
        args: Command-line arguments containing hyperparameters
        
    Returns:
        tuple: (trained_model, list_of_losses)
    """
    
    print("="*70)
    print("SRCNN TRAINING FOR DEEPLENSE")
    print("="*70)
    
    # Setup device (CUDA if available, CPU otherwise)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create dataset and dataloader
    print(f"\nCreating dataset...")
    train_dataset = DummyLensingDataset(
        n_samples=args.n_samples,
        img_size=args.img_size,
        scale_factor=args.scale_factor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility (avoid multiprocessing issues)
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = SRCNN(num_channels=1).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    model.train()
    losses = []
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            # Move data to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.6f}")
    
    print(f"\n✅ Training complete!")
    
    # Save model if requested
    if args.save_model:
        Path(args.output_dir).mkdir(exist_ok=True)
        model_path = Path(args.output_dir) / "srcnn_deeplense.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\n💾 Model saved: {model_path}")
    
    # Create visualization
    visualize_results(model, train_dataset, device, args.output_dir)
    
    return model, losses


def visualize_results(model, dataset, device, output_dir):
    """
    Create side-by-side comparison visualization
    
    Generates a figure showing:
    - Input LR image (bicubic upsampled)
    - Super-resolved output (SRCNN)
    - Ground truth HR image
    
    Args:
        model: Trained SRCNN model
        dataset: Dataset to sample from
        device: torch.device for computation
        output_dir: Directory to save visualization
    """
    
    print(f"\nCreating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Get first sample from dataset
        lr_img, hr_img = dataset[0]
        lr_img = lr_img.unsqueeze(0).to(device)
        sr_img = model(lr_img).cpu()
        lr_img = lr_img.cpu()
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(lr_img[0, 0], cmap='viridis')
        axes[0].set_title('Low Resolution (Input)')
        axes[0].axis('off')
        
        axes[1].imshow(sr_img[0, 0], cmap='viridis')
        axes[1].set_title('Super-Resolved (SRCNN)')
        axes[1].axis('off')
        
        axes[2].imshow(hr_img[0], cmap='viridis')
        axes[2].set_title('High Resolution (Target)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        Path(output_dir).mkdir(exist_ok=True)
        output_path = Path(output_dir) / "srcnn_result.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_path}")
        
        plt.close()


# MAIN

def main():
    """
    Parse arguments and run training
    
    Command-line interface allows full control over training parameters
    without modifying code. All paths are configurable to avoid hardcoding.
    """
    
    parser = argparse.ArgumentParser(
        description='SRCNN for DeepLense Super-Resolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--img-size', type=int, default=64,
                        help='Image size (square images)')
    parser.add_argument('--scale-factor', type=int, default=2,
                        help='Super-resolution scale factor')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    
    # System arguments
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (disable CUDA)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for saving outputs')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model weights')
    
    args = parser.parse_args()
    
    # Run training
    model, losses = train_srcnn(args)
    
    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()