from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_components.utils import *
from ssl_components.model import LensModel

ROOT = 'data'
BATCH_SIZE = 64
NUM_WORKERS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 30
ENCODER_CHECKPOINT_PATH = 'checkpoints/lens_model_iv.pt'
CLASSIFIER_CHECKPOINT_PATH = 'checkpoints/random_lens_classifier.pt'
EPSILON = 1e-6


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


class LensClassificationDataset(Dataset):
    def __init__(self, root, split='train', class_to_idx=None):
        super().__init__()
        self.split_dir = Path(root) / 'Model_III' / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f'Could not find split directory: {self.split_dir}')

        if class_to_idx is None:
            class_names = sorted(path.name for path in self.split_dir.iterdir() if path.is_dir())
            class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}

        self.class_to_idx = class_to_idx
        self.samples = []
        for class_name, class_index in self.class_to_idx.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.rglob('*.npy')):
                self.samples.append((image_path, class_index))

        if len(self.samples) == 0:
            raise ValueError(f'No labelled .npy files found in {self.split_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = np.load(image_path, allow_pickle=True)
        if image.dtype == object:
            image = image[0]
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = image[None, :, :]
        elif image.ndim == 3 and image.shape[0] != 1:
            image = image[:1]
        image = torch.from_numpy(image)
        return preprocess_lens_image(image), torch.tensor(label, dtype=torch.long)


class RandomEncoderClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes, hidden_dim=256, feature_index=1):
        super().__init__()
        self.encoder = encoder
        self.feature_index = feature_index
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            encoder_output = self.encoder(x)
            if isinstance(encoder_output, tuple):
                features = encoder_output[self.feature_index]
            else:
                features = encoder_output

            if features.ndim == 4:
                features = features.flatten(2).transpose(1, 2)

            random_features = torch.randn_like(features)

        pooled_features = random_features.mean(dim=1)
        return self.classifier(pooled_features)


def build_dataloaders(root, batch_size, num_workers):
    train_dataset = LensClassificationDataset(root=root, split='train')
    val_dataset = LensClassificationDataset(
        root=root,
        split='val',
        class_to_idx=train_dataset.class_to_idx,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataloader, val_dataloader, train_dataset.class_to_idx


def infer_encoder_embed_dim(encoder):
    if hasattr(encoder, 'projector') and hasattr(encoder.projector, 'out_features'):
        return encoder.projector.out_features
    if hasattr(encoder, 'norm') and hasattr(encoder.norm, 'normalized_shape'):
        return encoder.norm.normalized_shape[0]
    raise ValueError('Could not infer encoder embedding dimension.')


def load_frozen_encoder(checkpoint_path, device):
    lens_model = LensModel(64, 4, 1).to(device)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        missing_keys, unexpected_keys = lens_model.load_state_dict(state_dict, strict=False)
        print(f'Loaded encoder checkpoint: {checkpoint_path}')
        if missing_keys:
            print(f'Missing checkpoint keys: {len(missing_keys)}')
        if unexpected_keys:
            print(f'Unexpected checkpoint keys: {len(unexpected_keys)}')
    else:
        print(f'No encoder checkpoint found at {checkpoint_path}; using random frozen encoder.')
    return lens_model.vit_encoder


def train_one_epoch(model, dataloader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description='Train a small MLP classifier on random vectors matching the frozen LensModel encoder output shape.')
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--feature-index', type=int, default=1, help='Encoder tuple output to mimic: 0 for tokens, 1 for projected latent.')
    parser.add_argument('--encoder-checkpoint', default=ENCODER_CHECKPOINT_PATH)
    parser.add_argument('--classifier-checkpoint', default=CLASSIFIER_CHECKPOINT_PATH)
    parser.add_argument('--device', default=DEVICE)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_dataloader, val_dataloader, class_to_idx = build_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    encoder = load_frozen_encoder(args.encoder_checkpoint, device)
    embed_dim = infer_encoder_embed_dim(encoder)
    model = RandomEncoderClassifier(
        encoder=encoder,
        embed_dim=embed_dim,
        num_classes=len(class_to_idx),
        hidden_dim=args.hidden_dim,
        feature_index=args.feature_index,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_accuracy = 0.0
    for epoch in tqdm(range(args.epochs)):
        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, criterion, optimiser, device)
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = Path(args.classifier_checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
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
    print(f'best_val_acc={best_val_accuracy:.4f} checkpoint={args.classifier_checkpoint}')


if __name__ == '__main__':
    main()
