import torch
import numpy as np
import torchvision
from model import NanoDiT
from ema_pytorch import EMA
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image import FrechetInceptionDistance

torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.cuda.manual_seed_all(666)
torch.set_float32_matmul_precision('high')

NUM_CLASSES = 3
IMG_SIZE = 64
IMG_CHANNELS = 1
LATENT_DIM = 512
PATCH_SIZE = 2
MODEL_DEPTH = 6
MODEL_HEADS = 8
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path, compile=False):
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
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])
    
    ema.copy_params_from_ema_to_model()
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Device: {DEVICE}")
    # if compile:
    #     print("Compiling model for performance...")
    # model = torch.compile(mode='max-autotune')
    
    return model

@torch.inference_mode()
def generate_image(model, target_class, ode_steps=100, cfg_scale=3.0, num_samples=1, save_path=None):
    model.eval()
    null_token_id = NUM_CLASSES
    
    z = torch.randn((num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)
    
    conditional_labels = torch.tensor([target_class] * num_samples, device=DEVICE).long()
    
    using_cfg = cfg_scale >= 1.0
    y = conditional_labels
    if using_cfg:
        y_null = torch.tensor([null_token_id] * num_samples, device=DEVICE)
        y = torch.cat([y, y_null], 0)
    
    dt = 1.0 / ode_steps
    dt = torch.tensor([dt] * num_samples).to(z.device).view([num_samples, *([1] * len(z.shape[1:]))])
    
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
    
    return images


def calculate_fid_from_npy_dirs(real_dir, generated_dir, device="cuda"):
    fid = FrechetInceptionDistance().to(device)
    
    real_files = [f for f in os.listdir(real_dir) if f.endswith('.npy')]
    generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.npy')]
    
    print(f"Found {len(real_files)} real images and {len(generated_files)} generated images")
    
    def process_npy_batch(file_paths, base_dir):
        images = []
        for file_path in file_paths:
            img = np.load(os.path.join(base_dir, file_path))  # Shape: (1, 64, 64)
            img = torch.from_numpy(img).float()
            img = img.repeat(3, 1, 1)  # (3, 64, 64)
            images.append(img)
            
        return torch.stack(images)
    
    batch_size = 50
    
    # Process real images
    print("Processing real images...")
    for i in tqdm(range(0, len(real_files), batch_size)):
        batch_files = real_files[i:i+batch_size]
        real_batch = process_npy_batch(batch_files, real_dir).to(device)
        # Convert to uint8 [0, 255] as expected by FID
        real_batch = (real_batch * 255).to(torch.uint8)
        fid.update(real_batch, real=True)
    
    # Process generated images
    print("Processing generated images...")
    for i in tqdm(range(0, len(generated_files), batch_size)):
        batch_files = generated_files[i:i+batch_size]
        generated_batch = process_npy_batch(batch_files, generated_dir).to(device)
        generated_batch = (generated_batch * 255).to(torch.uint8)
        fid.update(generated_batch, real=False)
    
    # Compute FID
    fid_score = fid.compute()
    
    return fid_score


if __name__ == "__main__":
    real_dir = "real_npy_stacked"
    generated_dir = "generated_npy_stacked"
    
    model = load_model("/speech/advait/rooshil/nanoDiT/in_progress/diffusion_training/dit_conditional_ckpts/dit_conditional_epoch_69.pt", compile=True)
    classes = ['axion', 'cdm', 'no_sub']
    os.makedirs('generated_npy', exist_ok=True)
    os.makedirs('generated_npy_stacked', exist_ok=True)
    for class_name in classes:
        os.makedirs(f'generated_npy/{class_name}', exist_ok=True)
        for i in tqdm(range(400), desc=f"Generating samples for {class_name}"):
            sample = generate_image(model, classes.index(class_name))
            sample = sample.squeeze().cpu().numpy()
            np.save(f'generated_npy/{class_name}/{i}.npy', sample)
            if class_name == 'axion':
                np.save(f'generated_npy_stacked/{i}.npy', sample)
            if class_name == 'cdm':
                np.save(f'generated_npy_stacked/{400+i}.npy', sample)
            if class_name == 'no_sub':
                np.save(f'generated_npy_stacked/{800+i}.npy', sample)
    
    fid_score = calculate_fid_from_npy_dirs(real_dir, generated_dir, device=DEVICE)
    fid_axion = calculate_fid_from_npy_dirs(f'real_npy/axion', f'generated_npy/axion', device=DEVICE)
    fid_cdm = calculate_fid_from_npy_dirs(f'real_npy/cdm', f'generated_npy/cdm', device=DEVICE)
    fid_no_sub = calculate_fid_from_npy_dirs(f'real_npy/no_sub', f'generated_npy/no_sub', device=DEVICE)
    
    print(f"Overall FID Score: {fid_score:.4f}")
    print(f"FID Score for Axion: {fid_axion:.4f}")
    print(f"FID Score for CDM: {fid_cdm:.4f}")
    print(f"FID Score for No Sub: {fid_no_sub:.4f}")
    # gen_regular, _ = sample_conditional_cfg(
    #     model, [1], ode_steps=120,
    #     cfg_scale=1.5, num_samples_per_cls=4
    # )

    # # Test 2: EMA model, no CFG  
    # gen_ema, _ = sample_conditional_cfg(
    #     model_ema, [1], ode_steps=120,
    #     cfg_scale=1.5, num_samples_per_cls=4
    # )
    
    # torchvision.utils.save_image(gen_regular, "cdm_regular_2.png")
    # torchvision.utils.save_image(gen_ema, "cdm_ema_2.png")  
    
    


