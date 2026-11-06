from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = 'data'
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 30
CHECKPOINT_PATH = 'checkpoints/arc_classifier_model_iii.pt'
NUM_ANGLE_SAMPLES = 256
NUM_PSD_BINS = 32
NUM_FOURIER_COEFFS = 32
EPSILON = 1e-8


def load_image(path: Path) -> np.ndarray:
    image = np.load(path, allow_pickle=True)
    if isinstance(image, np.ndarray) and image.dtype == object:
        image = image[0]
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[None, :, :]
    elif image.ndim == 3 and image.shape[0] != 1:
        image = image[:1]
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image.astype(np.float32)


def estimate_arc_signal(image: np.ndarray, n_angles: int = NUM_ANGLE_SAMPLES) -> np.ndarray:
    """Project the brightest circular arc into a 1D brightness signal I(s).

    The image is treated as a lensing map. We estimate the ring center from the
    bright pixels, identify an annular radius with maximal radial power, and then
    average the intensity over a narrow angular band at that radius. The result is
    a signal indexed by angle, which is equivalent to arc length s = r * theta.
    """
    if image.ndim == 3:
        image = image[0]

    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros(n_angles, dtype=np.float32)

    bright_mask = image > np.percentile(image, 80)
    yy, xx = np.indices(image.shape)
    if bright_mask.any():
        center_y = yy[bright_mask].mean()
        center_x = xx[bright_mask].mean()
    else:
        center_y = image.shape[0] / 2.0
        center_x = image.shape[1] / 2.0

    center = np.array([center_x, center_y], dtype=np.float32)
    y_dist = yy - center[1]
    x_dist = xx - center[0]
    r = np.hypot(y_dist, x_dist)

    radial_bins = np.linspace(0.0, float(r.max()), 256, endpoint=False)
    radial_centers = radial_bins + (radial_bins[1] - radial_bins[0]) / 2.0
    counts, _ = np.histogram(r.ravel(), bins=256, range=(0.0, float(r.max()) if r.max() > 0 else 1.0))
    weights, _ = np.histogram(r.ravel(), bins=256, range=(0.0, float(r.max()) if r.max() > 0 else 1.0), weights=image.ravel())
    with np.errstate(divide='ignore', invalid='ignore'):
        radial_profile = np.divide(weights, counts, out=np.zeros_like(weights, dtype=np.float32), where=counts > 0)

    if radial_profile.max() <= 0:
        radius = max(1.0, min(image.shape) * 0.30)
    else:
        radius = float(radial_centers[np.argmax(radial_profile)])
        radius = max(radius, 2.0)

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    signal = np.empty(n_angles, dtype=np.float32)
    annulus_half_width = max(2, int(radius * 0.08))
    angle_half_width = 2.0 * np.pi / n_angles * 2.0

    for idx, theta in enumerate(angles):
        theta_grid = np.arctan2(y_dist, x_dist)
        delta_theta = np.abs(np.angle(np.exp(1j * (theta_grid - theta))))
        mask = (np.abs(r - radius) <= annulus_half_width) & (delta_theta <= angle_half_width)

        if mask.any():
            signal[idx] = image[mask].mean()
        else:
            signal[idx] = image[np.abs(r - radius) <= annulus_half_width].mean() if np.any(np.abs(r - radius) <= annulus_half_width) else image.mean()

    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - signal.mean()
    return signal


def compute_arc_descriptors(image: np.ndarray) -> np.ndarray:
    signal = estimate_arc_signal(image)

    variance = float(np.var(signal))
    magnitude_fft = np.fft.rfft(signal)
    psd = np.abs(magnitude_fft) ** 2
    psd = np.log(np.maximum(psd[:NUM_PSD_BINS], EPSILON))
    fourier_coeffs = np.real(magnitude_fft[:NUM_FOURIER_COEFFS])

    features = np.concatenate(
        [
            np.array([variance], dtype=np.float32),
            psd.astype(np.float32),
            fourier_coeffs.astype(np.float32),
        ]
    )
    return features


class ArcFeatureDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', class_to_idx: dict | None = None) -> None:
        super().__init__()
        split_dir = Path(root) / 'Model_IV' / split
        if not split_dir.exists():
            raise FileNotFoundError(f'Could not find split directory: {split_dir}')

        if class_to_idx is None:
            class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
            class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}

        self.class_to_idx = class_to_idx
        self.samples = []
        for class_name, class_index in self.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.rglob('*.npy')):
                self.samples.append((image_path, class_index))

        if len(self.samples) == 0:
            raise ValueError(f'No labeled .npy files were found in {split_dir}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = load_image(image_path)
        features = torch.from_numpy(compute_arc_descriptors(image))
        label = torch.tensor(label, dtype=torch.long)
        return features, label


class ArcMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_dataloaders(root: str, batch_size: int, num_workers: int, max_samples: int | None = None):
    train_dataset = ArcFeatureDataset(root=root, split='train')
    if max_samples is not None:
        train_dataset.samples = train_dataset.samples[:max_samples]

    val_dataset = ArcFeatureDataset(
        root=root,
        split='val',
        class_to_idx=train_dataset.class_to_idx,
    )
    if max_samples is not None:
        val_dataset.samples = val_dataset.samples[:max_samples]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_dataset.class_to_idx


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(dataloader, leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += features.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        running_loss += loss.item() * features.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += features.size(0)

    return running_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a small MLP on arc-based brightness descriptors for Model III lensing classes.'
    )
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap for quick smoke tests.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, class_to_idx = build_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    feature_dim = 1 + NUM_PSD_BINS + NUM_FOURIER_COEFFS
    model = ArcMLP(input_dim=feature_dim, num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_accuracy = 0.0
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = Path(args.checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'feature_dim': feature_dim,
                    'epoch': epoch,
                    'val_accuracy': val_accuracy,
                },
                checkpoint_path,
            )

        print(
            f'Epoch {epoch}: '
            f'train_loss={train_loss:.6f} train_acc={train_accuracy:.4f} '
            f'val_loss={val_loss:.6f} val_acc={val_accuracy:.4f}'
        )

    print(f'classes={class_to_idx}')
    print(f'best_val_acc={best_val_accuracy:.4f} checkpoint={args.checkpoint}')


if __name__ == '__main__':
    main()
