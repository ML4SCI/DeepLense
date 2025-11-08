"""
Thanks to Gemini 2.5 for pairing here and there.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import NanoDiT
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from contextlib import nullcontext
from dataset import NpyDataset, diffusion_transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from ema_pytorch import EMA

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision('high')

NUM_CLASSES = 3  
IMG_SIZE = 64
IMG_CHANNELS = 1 
LATENT_DIM = 512
PATCH_SIZE = 2
MODEL_DEPTH = 6
MODEL_HEADS = 8
MAX_GRAD_NORM = 1.0
NUM_ODE_STEPS = 50
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WARMUP_EPOCHS = 10
BATCH_SIZE = 160
EPOCHS = 600
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
COMPILE = True
AMP_DTYPE = torch.float32
SAMPLE_INTERVAL = 1
NUM_SAMPLES_PER_CLASS = 4
CFG_SCALE = 5.0
CHECKPOINT_SAVE_INTERVAL = 3
DATA_DIR = "/speech/advait/rooshil/nanoDiT/Model_II_normalized"
CKPT_SAVE_DIR = "diffusion_training/dit_conditional_ckpts"
IMG_SAVE_DIR = "diffusion_training/dit_conditional_images"
NPY_SAVE_DIR = "diffusion_training/dit_conditional_npy"

os.makedirs(CKPT_SAVE_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(NPY_SAVE_DIR, exist_ok=True)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def flow_lerp(x):
    bsz = x.shape[0]
    t = torch.rand((bsz,)).to(x.device)
    t_broadcast = t.view(bsz, 1, 1, 1)
    z1 = torch.randn_like(x)
    zt = (1 - t_broadcast) * x + t_broadcast * z1
    return zt, z1, t

@torch.no_grad()
def sample_conditional_cfg(
    model, target_classes_list, ode_steps, cfg_scale=5.0, num_samples_per_cls=1, null_token_id=NUM_CLASSES
):
    model.eval()
    num_target_cls = len(target_classes_list)
    total_images_to_sample = num_samples_per_cls * num_target_cls

    z = torch.randn((total_images_to_sample, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)

    sample_cls_labels_list = []
    for c_idx in target_classes_list:
        sample_cls_labels_list.extend([c_idx] * num_samples_per_cls)
    conditional_labels = torch.tensor(sample_cls_labels_list, device=DEVICE).long()

    using_cfg = cfg_scale >= 1.0
    y = conditional_labels
    if using_cfg:
        n = conditional_labels.shape[0]
        y_null = torch.tensor([null_token_id] * n, device=conditional_labels.device)
        y = torch.cat([y, y_null], 0)

    bsz = z.shape[0]
    dt = 1.0 / ode_steps
    dt = torch.tensor([dt] * bsz).to(z.device).view([bsz, *([1] * len(z.shape[1:]))])
    
    for i in range(ode_steps, 0, -1):
        z_final = torch.cat([z, z], 0) if using_cfg else z
        t = i / ode_steps
        t = torch.tensor([t] * z_final.shape[0]).to(z_final.device, dtype=z_final.dtype)
        predicted_velocity = model(z_final, t, y)
        if using_cfg:
            pred_cond, pred_uncond = predicted_velocity.chunk(2, dim=0)
            predicted_velocity = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        z = (z - dt * predicted_velocity).to(z.dtype)
    
    images = (z + 1) / 2.0
    images = torch.clamp(images, 0.0, 1.0)

    model.train()
    return images, conditional_labels

def plot_loss_and_lr(epoch_losses, learning_rates, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epoch_losses, label='Training Loss', color='blue', linewidth=2)
    if len(epoch_losses) > 10:
        window_size = max(1, len(epoch_losses) // 20)
        smoothed = np.convolve(epoch_losses, np.ones(window_size)/window_size, mode='valid')
        x_smooth = np.arange(window_size - 1, len(epoch_losses))
        ax1.plot(x_smooth, smoothed, label=f'Smoothed', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(learning_rates, label='Learning Rate', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics.png", dpi=150)
    plt.close()

def train():
    model = NanoDiT(
        input_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IMG_CHANNELS,
        hidden_size=LATENT_DIM,
        depth=MODEL_DEPTH,
        num_heads=MODEL_HEADS,
        num_classes=NUM_CLASSES,
        timestep_freq_scale=1000,
    ).to(DEVICE)
    
    ema = EMA(model, include_online_model=False)
    
    if COMPILE:
        print("`torch.compile()` enabled.")
        model.compile(mode='max-autotune')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = torch.GradScaler() if AMP_DTYPE is not None else None
    criterion = nn.MSELoss()
    
    train_dataset = NpyDataset(root_dir=DATA_DIR, transform=diffusion_transforms)
    train_classes = list(set(train_dataset.class_to_idx.values()))
    assert NUM_CLASSES == len(train_classes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
    )
    
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = len(train_dataloader) * WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=MIN_LEARNING_RATE / LEARNING_RATE
    )
    
    amp_context = (
        torch.autocast(device_type=torch.device(DEVICE).type, dtype=AMP_DTYPE) 
        if AMP_DTYPE is not None
        else nullcontext()
    )
    if AMP_DTYPE:
        print(f"Using automatic mixed-precision in {AMP_DTYPE}")

    print(f"Training on {DEVICE}")
    print(f"Using custom model: {type(model).__name__}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Warmup epochs: {WARMUP_EPOCHS}, Total steps: {total_steps}")
    
    all_losses = []
    epoch_avg_losses = []
    learning_rates = []
    global_step = 0
    
    epoch_pbar = tqdm(range(EPOCHS), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_losses = []
        
        step_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", position=1, leave=False)
        
        for step, (real_images, class_ids) in enumerate(step_pbar):
            optimizer.zero_grad()
            real_images = real_images.to(DEVICE, non_blocking=True)
            class_ids = class_ids.to(DEVICE, non_blocking=True)

            zt, z1, t = flow_lerp(real_images)
            target_velocity_field = z1 - real_images
            
            with amp_context:
                predicted_velocity = model(zt, t, class_ids)
                loss = criterion(target_velocity_field, predicted_velocity)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()

            scheduler.step()
            ema.update()
            
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            all_losses.append(loss_val)
            current_lr = scheduler.get_last_lr()[0]
            
            global_step += 1
            
            step_pbar.set_postfix({
                'Loss': f'{loss_val:.6f}',
                'Avg Loss': f'{np.mean(epoch_losses):.6f}',
                'LR': f'{current_lr:.2e}',
                'Step': global_step
            })

        avg_epoch_loss = np.mean(epoch_losses)
        epoch_avg_losses.append(avg_epoch_loss)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        epoch_pbar.set_postfix({
            'Avg Loss': f'{avg_epoch_loss:.6f}',
            'Latest': f'{epoch_losses[-1]:.6f}',
            'LR': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        plot_loss_and_lr(epoch_avg_losses, learning_rates, "diffusion_training")

        if (epoch + 1) % SAMPLE_INTERVAL == 0 or epoch == EPOCHS - 1:
            print(f"\nSampling images at epoch {epoch + 1}...")
            classes_to_sample_list = list(range(min(NUM_CLASSES, 5)))
            
            generated_sample_images, _ = sample_conditional_cfg(
                    model, classes_to_sample_list, ode_steps=NUM_ODE_STEPS,
                    cfg_scale=CFG_SCALE, num_samples_per_cls=NUM_SAMPLES_PER_CLASS
                )
            
            if generated_sample_images.nelement() > 0:
                for i, img in enumerate(generated_sample_images):
                    class_id = classes_to_sample_list[i // NUM_SAMPLES_PER_CLASS]
                    img_np = img.cpu().numpy().squeeze()
                    npy_dir = f"{NPY_SAVE_DIR}/sample_epoch_{epoch + 1}"
                    os.makedirs(npy_dir, exist_ok=True)
                    npy_file_path = f"{npy_dir}/class_{class_id}_sample_{i % NUM_SAMPLES_PER_CLASS}.npy"
                    np.save(npy_file_path, img_np)
                    
                grid = torchvision.utils.make_grid(generated_sample_images, nrow=NUM_SAMPLES_PER_CLASS)
                torchvision.utils.save_image(grid, f"{IMG_SAVE_DIR}/sample_epoch_{epoch + 1}.png")
                print(f"Saved EMA sample images to sample_epoch_{epoch + 1}.png")

        if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'global_step': global_step
            }
            torch.save(checkpoint, f"{CKPT_SAVE_DIR}/dit_conditional_epoch_{epoch + 1}.pt")
            print(f"Saved checkpoint with EMA at epoch {epoch + 1}")

    print("Training finished.")
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': EPOCHS,
        'global_step': global_step
    }
    torch.save(final_checkpoint, f"{CKPT_SAVE_DIR}/dit_conditional_final.pt")
    print("Saved final checkpoint with EMA")

if __name__ == "__main__":
    train()
