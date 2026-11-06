from ssl_components.utils import *
from ssl_components.model import *
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = 'data'
BATCH_SIZE = 32
NUM_WORKERS = 10
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 10
CHECKPOINT_PATH = 'checkpoints/lens_model_i_ssl.pt'
VAL_LOSS_PLOT_PATH = 'plots/val_loss_curve.png'
EPSILON = 1e-6

# Reconstruction weights (input-anchored; these are the terms that drive the encoder to learn).
REC_WEIGHT = 1.0         # preprocessed-image MSE — the main data term
RAW_REC_WEIGHT = 0.1     # log+z-normalised raw flux — secondary data term


POISSON_WEIGHT = 0
CURL_WEIGHT = 0
SMOOTH_PSI_WEIGHT = 0
SMOOTH_KAPPA_WEIGHT = 0
POS_KAPPA_WEIGHT = 0

# Kept for reference (unused by the active loss).
SOURCE_TV_WEIGHT = 1e-4
SOURCE_POSITIVE_WEIGHT = 1e-3
POTENTIAL_SMOOTHNESS_WEIGHT = 1e-4
ALPHA_MAGNITUDE_WEIGHT = 1e-5

physics = Physics()

def preprocess_lens_image(image):
    image = torch.clamp(image, min=EPSILON)
    image_max = image.amax(dim=(-2, -1), keepdim=True).clamp_min(EPSILON)
    log_contrast = torch.log(image_max / image).pow(2)

    kernel_x = torch.tensor(
        [[[-0.5, 0.0, 0.5]]],
        dtype=log_contrast.dtype,
    )
    kernel_y = torch.tensor(
        [[[-0.5], [0.0], [0.5]]],
        dtype=log_contrast.dtype,
    )

    mixed_gradient = F.conv2d(log_contrast.unsqueeze(0), kernel_x.unsqueeze(0), padding=(0, 1))
    mixed_gradient = F.conv2d(mixed_gradient, kernel_y.unsqueeze(0), padding=(1, 0))
    return torch.abs(torch.tanh(mixed_gradient.squeeze(0)))

class LensDataset(Dataset):
    def __init__(self, root, split='train'):
        super().__init__()
        split_dir = Path(root) / 'Model_I' / split
        if not split_dir.exists():
            raise FileNotFoundError(f'Could not find split directory: {split_dir}')

        self.image_paths = sorted(split_dir.rglob('*.npy'))
        if len(self.image_paths) == 0:
            raise ValueError(f'No .npy files found in {split_dir}')
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.load(image_path, allow_pickle=True)
        if image.dtype == object:
            image = image[0]
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = image[None, :, :]
        elif image.ndim == 3 and image.shape[0] != 1:
            image = image[:1]

        image = torch.from_numpy(image).unsqueeze(0)
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

        # Raw target: log + per-sample z-normalise so the loss has a learnable scale.
        # Computed on the numpy array before converting to torch.
        raw = np.log(np.clip(image.numpy(), EPSILON, None))
        raw = (raw - raw.mean()) / (raw.std() + 1e-6)
        raw = torch.from_numpy(raw.astype(np.float32))

        if image.shape[0]==148:
            print(f"148 at {image_path}")
        return preprocess_lens_image(image), raw
    
train_dataset = LensDataset(root=ROOT, split='train') 
val_dataset = LensDataset(root=ROOT, split='val')

train_dataloader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              num_workers=NUM_WORKERS,
                              shuffle=True)

val_dataloader = DataLoader(val_dataset, 
                              batch_size=BATCH_SIZE, 
                              num_workers=NUM_WORKERS,
                              shuffle=False)


def total_variation(x):
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w




def physics_reconstruction_loss(model, batch):
    """Input-anchored loss: MSE on the preprocessed image is the main data term;
    a secondary MSE on the (log + z-normalised) raw flux head keeps the physical
    scale represented. The remaining terms are small-weight physics regularisers."""
    preprocessed, raw = batch
    _, _, psi, kappa, img_recon, raw_recon = model(preprocessed)

    L_rec = F.mse_loss(img_recon, preprocessed)
    L_raw_rec = F.mse_loss(raw_recon, raw)
    L_poisson = physics.poisson_loss(psi, kappa)
    L_curl = physics.deflection_consistency_loss(psi)
    L_smooth_psi = total_variation(psi)
    L_smooth_kappa = total_variation(kappa)
    L_pos_kappa = F.relu(-kappa).pow(2).mean()

    return (
        REC_WEIGHT * L_rec
        + RAW_REC_WEIGHT * L_raw_rec
        + POISSON_WEIGHT * L_poisson
        + CURL_WEIGHT * L_curl
        + SMOOTH_PSI_WEIGHT * L_smooth_psi
        + SMOOTH_KAPPA_WEIGHT * L_smooth_kappa
        + POS_KAPPA_WEIGHT * L_pos_kappa
    )


def train_one_epoch(model, dataloader, optimiser):
    model.train()
    running_loss = 0
    total = 0
    for preprocessed, raw in tqdm(dataloader):
        preprocessed = preprocessed.to(DEVICE, non_blocking=True)
        raw = raw.to(DEVICE, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        loss = physics_reconstruction_loss(model, (preprocessed, raw))
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * preprocessed.size(0)
        total += preprocessed.size(0)
    return running_loss / total


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    running_loss = 0
    total = 0
    for preprocessed, raw in dataloader:
        preprocessed = preprocessed.to(DEVICE, non_blocking=True)
        raw = raw.to(DEVICE, non_blocking=True)
        loss = physics_reconstruction_loss(model, (preprocessed, raw))
        running_loss += loss.item() * preprocessed.size(0)
        total += preprocessed.size(0)
    return running_loss / total


def main():
    model = LensModel(64, 16, 1).to(DEVICE)
    optimiser = optim.SGD(model.parameters(), 
                            weight_decay=1e-4,
                            lr=1e-3)
    best_val_loss = float('inf')
    val_losses = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss = train_one_epoch(model, train_dataloader, optimiser)
        val_loss = validate(model, val_dataloader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        print(f'Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}')

    plot_path = Path(VAL_LOSS_PLOT_PATH)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation loss')
    plt.title('Validation loss over epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Saved validation loss curve to {plot_path}')


if __name__ == '__main__':
    main()
