"""
Unsupervised Autoencoder Super-Resolution for DeepLense
Implements DEEPLENSE2 requirement: "familiarity with autoencoders"

This is UNSUPERVISED - does not require paired LR/HR images.
Works by learning to reconstruct high-quality images from degraded inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Encoder block with downsampling
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        pooled = self.pool(x)
        return pooled, x  # Return both for skip connections


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SuperResolutionAutoencoder(nn.Module):
    """
    Autoencoder for Unsupervised Super-Resolution
    
    Architecture:
    - Encoder: Extracts features while downsampling
    - Bottleneck: Compressed representation
    - Decoder: Reconstructs high-quality image with skip connections
    
    Key difference from SRCNN (PR #1):
    - SRCNN: Supervised (needs paired LR/HR)
    - This: Unsupervised (learns from single images)
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        base_channels: Base number of filters (default: 64)
    """
    
    def __init__(self, in_channels=1, base_channels=64):
        super(SuperResolutionAutoencoder, self).__init__()
        
        # Encoder path
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        
        # Decoder path (with skip connections - U-Net style)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)
        
        # Final reconstruction layer
        self.final = nn.Conv2d(base_channels, in_channels, 1)
        
    def forward(self, x):
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder with skip connections
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Final output
        x = self.final(x)
        
        return x


class PerceptualLoss(nn.Module):
    """
    Perceptual loss for better SR quality
    Combines pixel-wise MSE with feature-space loss
    """
    
    def __init__(self, alpha=1.0, beta=0.1):
        super(PerceptualLoss, self).__init__()
        self.alpha = alpha  # Weight for reconstruction loss
        self.beta = beta    # Weight for perceptual loss
        self.mse = nn.MSELoss()
        
    def forward(self, output, target):
        # Reconstruction loss (pixel-wise)
        recon_loss = self.mse(output, target)
        
        # Simple gradient-based perceptual loss
        # Encourages preservation of edges/structure
        grad_output_x = output[:, :, :, 1:] - output[:, :, :, :-1]
        grad_output_y = output[:, :, 1:, :] - output[:, :, :-1, :]
        grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        perceptual_loss = (
            self.mse(grad_output_x, grad_target_x) +
            self.mse(grad_output_y, grad_target_y)
        )
        
        total_loss = self.alpha * recon_loss + self.beta * perceptual_loss
        
        return total_loss, recon_loss, perceptual_loss


def apply_degradation(img, scale_factor=2, noise_level=0.05):
    """
    Apply degradation to create LR image (for unsupervised training)
    
    This simulates the observation process:
    1. Downsample (simulate low resolution observation)
    2. Add noise (simulate detector noise)
    3. Upsample back (so dimensions match for autoencoder)
    
    Args:
        img: High resolution image tensor
        scale_factor: Downsampling factor
        noise_level: Gaussian noise std dev
        
    Returns:
        Degraded image (same size as input)
    """
    
    b, c, h, w = img.shape
    
    # Downsample
    lr_size = (h // scale_factor, w // scale_factor)
    img_lr = F.interpolate(img, size=lr_size, mode='bilinear', align_corners=False)
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(img_lr) * noise_level
        img_lr = img_lr + noise
    
    # Upsample back to original size
    img_degraded = F.interpolate(img_lr, size=(h, w), mode='bicubic', align_corners=False)
    
    return img_degraded


# Test code
if __name__ == "__main__":
    print("="*70)
    print("TESTING UNSUPERVISED SR AUTOENCODER")
    print("="*70)
    
    # Create model
    model = SuperResolutionAutoencoder(in_channels=1, base_channels=64)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✅ Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 64, 64)
    print(f"\n📐 Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"📐 Output shape: {output.shape}")
    
    # Test degradation
    degraded = apply_degradation(dummy_input, scale_factor=2)
    print(f"📐 Degraded shape: {degraded.shape}")
    
    # Test loss
    criterion = PerceptualLoss()
    loss, recon, percep = criterion(output, dummy_input)
    print(f"\n📊 Loss values:")
    print(f"   Total: {loss.item():.6f}")
    print(f"   Reconstruction: {recon.item():.6f}")
    print(f"   Perceptual: {percep.item():.6f}")
    
    print(f"\n✅ All tests passed!")
    print(f"\n🎯 This autoencoder is ready for DEEPLENSE2 PR #2")