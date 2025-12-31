# External imports
import torch
import numpy as np
import argparse
import os
import random
from tqdm import tqdm
import torch.nn.functional as F
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Internal imports
from differentiable_lensing import DifferentiableLensing
import data
from sisr import SISR

# === small helpers ===========================================================
def strtobool(x):
    """
    Convert a string-like boolean into a Python bool.
    - Accepts things like 'true' (case-insensitive) -> True, everything else -> False.
    NOTE: this is very strict. Consider using distutils.util.strtobool for more options.
    """
    if x.lower().strip() == 'true': return True
    else: return False

def wmse_loss(y1, y2, w):
    """
    Weighted MSE wrapper: returns mean((y1-y2)^2 * w).
    - y1, y2, w: tensors broadcastable to same shape.
    - Note: no epsilon / stability tweak; if w contains zeros everywhere the mean will be 0 which might be OK.
    """
    return torch.mean((y1-y2)**2*w)

# === argument parsing =======================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='the LR of the optimizer')
    parser.add_argument('--seed', type=int, default=0,
                        help='the seed of the experiment')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if True, cuda will be enabled when possible')
    parser.add_argument('--log-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, training will be logged with Tensorboard')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='the number of images loaded to at any moment')
    parser.add_argument('--vdl-weight', type=float, default=0.0,
                        help='weight of the vdl loss')

    # Performance / architecture options
    parser.add_argument('--resolution', type=float,
                        help='arcsecond per pixel resolution the images are captured in')
    parser.add_argument('--magnification', type=int, default=2,
                        help='magnification value achieved by the SR network')
    parser.add_argument('--n-mag', type=int, default=1,
                        help='number of times the magnification value is applied by the SR network')
    parser.add_argument('--residual-depth', type=int, default=3,
                        help='the number of residual layers in the SR network')
    parser.add_argument('--in-channels', type=int, default=2,
                        help='the number of channels in the images')
    parser.add_argument('--latent-space-size', type=int, default=64,
                        help='the number of neurons in the latent space(s)')
    parser.add_argument('--image-shape', type=int, default=64,
                        help='the shape of the (square) image in one axis')
    parser.add_argument('--theta-e', type=float, default=0.75,
                        help='the value of the einstein radius used to compute the deflection field')
    args = parser.parse_args()

    # derived args
    args.effective_magnification = int(args.magnification ** args.n_mag)
    args.target_shape = args.image_shape * args.effective_magnification
    args.target_resolution = args.resolution / args.effective_magnification
    # device choice: OK here, but later code redefines device inconsistently (see below).
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print('[SYS] Device is set to %s'%args.device)
    return args

if __name__ == '__main__':
    args = parse_args()
    run_name = f'{args.exp_name}'

    # seeds for repeatability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # determinism for cuDNN
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = args.batch_size

    # --- dataset loading ---------------------------------------------------
    # The code constructs three datasets and concats them.
    # data.LensingDataset(root, types, n) — ensure data.LensingDataset handles these args.
    train_dataset_no_sub = data.LensingDataset('train/',['no_sub'],5000)
    val_dataset_no_sub = data.LensingDataset('val/',['no_sub'],2000)

    train_dataset_axion = data.LensingDataset('train/',['axion'],5000)
    val_dataset_axion = data.LensingDataset('val/',['axion'],2000)

    train_dataset_cdm = data.LensingDataset('train/',['cdm'],5000)
    val_dataset_cdm = data.LensingDataset('val/',['cdm'],2000)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset_no_sub, train_dataset_axion, train_dataset_cdm])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_no_sub, val_dataset_axion, val_dataset_cdm])

    train_dataset, train_rest = torch.utils.data.random_split(train_dataset, [0.34, 0.66])
    val_dataset, val_rest = torch.utils.data.random_split(val_dataset, [0.34, 0.66])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=min(8, os.cpu_count()))
    val_dataloader = torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=min(8, os.cpu_count()))

    # NOTE doc comment: "This configuration will load 5000 (low-resolution) images in total"
    # — but actual counts depend on the LensingDataset and the random_split above.

        # --- model / modules ----------------------------------------------------
    # Instantiate model and optimizer
    model = SISR(magnification=args.magnification, n_mag=args.n_mag, residual_depth=args.residual_depth, in_channels=args.in_channels, latent_channel_count=args.latent_space_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # differentiable lensing module: note that the DifferentiableLensing __init__
    # in the other file accepts `device` and `alpha`. Here alpha=None so internal set_alpha may not be called.
    lensing_module = DifferentiableLensing(device=device, alpha=None, target_resolution=args.target_resolution, target_shape=args.target_shape).to(args.device)

    # TensorBoard logging
    if args.log_train:
        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )

    # --- load precomputed sparse mappings and maps --------------------------
    # These files must exist in working dir. They are moved to args.device below.
    cross_grid_to_log = torch.load('scatter_to_log_128.pt').to(args.device)
    cross_grid_forward_from_log = torch.load('forward_from_log_128.pt').to(args.device)
    cross_grid_from_log = torch.load('scatter_from_log_128.pt').to(args.device)
    cross_grid_backward = torch.load('sparse_grid_fracs_euclid_backward.pt').to(args.device)

    # convergence maps
    source_convergence_map = torch.load('source_convergence_map.pt').to(args.device)
    image_convergence_map = torch.load('image_convergence_map.pt').to(args.device)

    # --- PSF kernel setup ---------------------------------------------------
    # gaussian_kernel returns (Z, X, Y) as numpy arrays in the previous file.
    # Converting to torch.tensor is fine but be mindful of dtype/device.
    psf, _, _ = lensing_module.gaussian_kernel(fwhm_arcsec=0.16, pixscale_arcsec=args.target_resolution)
    psf = torch.tensor(psf, dtype=torch.float32, device=args.device).unsqueeze(0).unsqueeze(0)

    # --- training loop ------------------------------------------------------
    for epoch in range(args.epochs):
        losses = []
        for i,lr_image in enumerate(tqdm(train_dataloader, desc=f"Training epoch {epoch+1} of {args.epochs}")):
            # lr_image shape expectation: (B, 1, H, W) maybe — they call .squeeze(1) below
            lr_image = lr_image.float().to(device).squeeze(1)

            # Source reconstruction through backward lensing (using precomputed sparse mapping)
            # cross_grid_fill expects I_img shaped (B, C, Ix, Iy); ensure lr_image matches.
            reconstructed_source = lensing_module.cross_grid_fill(lr_image, [cross_grid_backward]) # rename to lensing

            # Upscaling using a neural network: concatenate source and lr image along channels
            model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
            upscaled_source_ = model(model_feed)

            # Image construction through forward lensing: apply a chain of sparse mappings
            upscaled_image_ = lensing_module.cross_grid_fill(upscaled_source_, [cross_grid_to_log, cross_grid_forward_from_log, cross_grid_from_log])
            # Convolve with PSF. NOTE: padding='same' requires PyTorch >= 1.11 (or newer); older versions don't accept it.
            convolved_upscaled_image_ = F.conv2d(upscaled_image_, psf, padding="same")

            # Downsampling and upsampling:
            downsampled_image = F.interpolate(convolved_upscaled_image_, scale_factor=1/args.effective_magnification)
            interpolated_image = F.interpolate(lr_image, scale_factor=args.effective_magnification)
            downsampled_source = F.interpolate(upscaled_source_, scale_factor=1/args.effective_magnification)

            # Losses: weighted MSE
            image_reconstruction_loss = wmse_loss(downsampled_image, lr_image, source_convergence_map)
            source_reconstruction_loss = wmse_loss(downsampled_source, reconstructed_source, image_convergence_map)

            # Variation density (an optional regularizer)
            interpolated_image_vd = lensing_module.compute_variation_density(interpolated_image)
            convolved_upscaled_image_vd = lensing_module.compute_variation_density(convolved_upscaled_image_)

            total_loss = image_reconstruction_loss + source_reconstruction_loss + args.vdl_weight * F.mse_loss(interpolated_image_vd, convolved_upscaled_image_vd)
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            losses.append(total_loss.detach().item())

        # Logging/tracking after training loop
        if args.log_train:
            writer.add_scalar("train_loss/total", np.mean(losses), global_step=epoch)
        print('[SYS] Train loss at epoch %d: %.6f'%(epoch+1, np.mean(losses)))
                # --- validation loop -------------------------------------------------
        for i,lr_image in enumerate(tqdm(val_dataloader, desc=f"Validation epoch {epoch+1} of {args.epochs}")):
            lr_image = lr_image.float().to(device).squeeze(1)

            # Source reconstruction
            reconstructed_source = lensing_module.cross_grid_fill(lr_image, [cross_grid_backward]) # rename to lensing

            # Upscaling
            model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
            with torch.no_grad():
                upscaled_source_ = model(model_feed)

            # Image construction
            upscaled_image_ = lensing_module.cross_grid_fill(upscaled_source_, [cross_grid_to_log, cross_grid_forward_from_log, cross_grid_from_log])
            convolved_upscaled_image_ = F.conv2d(upscaled_image_, psf, padding="same")

            # Downsampling
            downsampled_image = F.interpolate(convolved_upscaled_image_, scale_factor=1/args.effective_magnification)
            interpolated_image = F.interpolate(lr_image, scale_factor=args.effective_magnification)
            downsampled_source = F.interpolate(upscaled_source_, scale_factor=1/args.effective_magnification)

            # Losses
            image_reconstruction_loss = wmse_loss(downsampled_image, lr_image, source_convergence_map)
            source_reconstruction_loss = wmse_loss(downsampled_source, reconstructed_source, image_convergence_map)

            interpolated_image_vd = lensing_module.compute_variation_density(interpolated_image)
            convolved_upscaled_image_vd = lensing_module.compute_variation_density(convolved_upscaled_image_)

            total_loss = image_reconstruction_loss + source_reconstruction_loss + args.vdl_weight * F.mse_loss(interpolated_image_vd, convolved_upscaled_image_vd)

            losses.append(total_loss.detach().item())

        # validation logging
        if args.log_train:
            writer.add_scalar("val_loss/total", np.mean(losses), global_step=i)
        print('[SYS] Validation loss at epoch %d: %.6f'%(epoch+1, np.mean(losses)))

    # save model weights at the end
    torch.save(model.state_dict(), '%s_weights.pt'%args.exp_name)
