from pathlib import Path
import argparse
import csv
import math

from ssl_components.utils import *
from ssl_components.model import LensModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = 'data/regression'
BATCH_SIZE = 64
NUM_WORKERS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 100
ENCODER_CHECKPOINT_PATH = 'checkpoints/lens_model_i_physics.pt'
REGRESSOR_CHECKPOINT_PATH = 'checkpoints/einstein_radius_regressor.pt'
EPSILON = 1e-6
RANDOM_SEED = 42


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


class EinsteinRadiusDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.image_dir = self.root / 'images'
        self.labels_path = self.root / 'labels.csv'
        if not self.image_dir.exists():
            raise FileNotFoundError(f'Could not find image directory: {self.image_dir}')
        if not self.labels_path.exists():
            raise FileNotFoundError(f'Could not find labels file: {self.labels_path}')

        self.samples = []
        with self.labels_path.open(newline='') as labels_file:
            for row in csv.DictReader(labels_file):
                image_path = self.image_dir / row['filename']
                if image_path.exists():
                    self.samples.append((image_path, float(row['einstein_radius'])))

        if len(self.samples) == 0:
            raise ValueError(f'No labelled images found in {self.root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, radius = self.samples[index]
        image = np.load(image_path, allow_pickle=True)
        if image.dtype == object:
            image = image[0]
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = image[None, :, :]
        elif image.ndim == 3 and image.shape[0] != 1:
            image = image[:1]

        image = torch.from_numpy(image)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(64, 64),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        return preprocess_lens_image(image), torch.tensor(radius, dtype=torch.float32)


class FrozenEncoderRegressor(nn.Module):
    def __init__(self, encoder, embed_dim, hidden_dim=512):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder.transformer_blocks[-1].parameters():
            parameter.requires_grad = False
        for parameter in self.encoder.projector.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            encoder_output = self.encoder(x)
            features = encoder_output[1] if isinstance(encoder_output, tuple) else encoder_output

        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        pooled_features = features.mean(dim=1)
        return self.regressor(pooled_features).squeeze(1)


def load_frozen_encoder(checkpoint_path, device):
    lens_model = LensModel(64, 16, 1).to(device)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Could not find SSL encoder checkpoint: {checkpoint_path}')

    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    lens_model.load_state_dict(state_dict)
    print(f'Loaded SSL encoder checkpoint: {checkpoint_path}')
    return lens_model.vit_encoder


def build_dataloaders(root, batch_size, num_workers, seed):
    dataset = EinsteinRadiusDataset(root)
    train_size = math.floor(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
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
    return train_dataloader, val_dataloader, train_size, val_size


def train_one_epoch(model, dataloader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    total = 0

    for images, targets in tqdm(dataloader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets = []
    all_predictions = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(images)
        loss = criterion(predictions, targets)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_targets.append(targets.cpu())
        all_predictions.append(predictions.cpu())

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()
    mae = mean_absolute_error(targets, predictions)
    rmse = mean_squared_error(targets, predictions) ** 0.5
    r2 = r2_score(targets, predictions)
    return running_loss / total, mae, rmse, r2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a small Einstein-radius regression head on a frozen SSL lens encoder.'
    )
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--encoder-checkpoint', default=ENCODER_CHECKPOINT_PATH)
    parser.add_argument('--checkpoint', default=REGRESSOR_CHECKPOINT_PATH)
    parser.add_argument('--device', default=DEVICE)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_dataloader, val_dataloader, train_size, val_size = build_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    encoder = load_frozen_encoder(args.encoder_checkpoint, device)
    embed_dim = encoder.projector.out_features
    model = FrozenEncoderRegressor(
        encoder=encoder,
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=0.1)
    optimiser = optim.AdamW(
        [
            {'params': model.regressor.parameters(), 'lr': args.lr},
            {
                'params': [
                    parameter for parameter in model.encoder.parameters()
                    if parameter.requires_grad
                ],
                'lr': args.lr * 0.01,
            },
        ],
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float('inf')
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimiser, device)
        val_loss, val_mae, val_rmse, val_r2 = validate(model, val_dataloader, criterion, device)
        print(
            f'Epoch {epoch}: '
            f'train_loss={train_loss:.6f} val_loss={val_loss:.6f} '
            f'val_mae={val_mae:.6f} val_rmse={val_rmse:.6f} val_r2={val_r2:.6f}'
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            checkpoint_path = Path(args.checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'train_size': train_size,
                    'val_size': val_size,
                },
                checkpoint_path,
            )

    print(f'data={args.root} train={train_size} val={val_size}')
    print(f'best_val_rmse={best_val_rmse:.6f} checkpoint={args.checkpoint}')


if __name__ == '__main__':
    main()
