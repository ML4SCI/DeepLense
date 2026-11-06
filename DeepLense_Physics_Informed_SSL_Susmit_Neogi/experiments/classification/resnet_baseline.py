from pathlib import Path
import argparse

import torchvision.models as models
from ssl_components.utils import *
from sklearn.metrics import classification_report


ROOT = 'data'
BATCH_SIZE = 64
NUM_WORKERS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 30
CHECKPOINT_PATH = 'checkpoints/resnet_model_i.pt'
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
        self.split_dir = Path(root) / 'Model_I' / split
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
        image = F.interpolate(
            image.unsqueeze(0),
            size=(64, 64),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        return preprocess_lens_image(image), torch.tensor(label, dtype=torch.long)


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


def build_model(num_classes):
    # torchvision has no ResNet-150; ResNet-152 is the closest standard variant.
    model = models.resnet152(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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
    all_labels = []
    all_predictions = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (predictions == labels).sum().item()
        total += images.size(0)
        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())

    return (
        running_loss / total,
        correct / total,
        torch.cat(all_labels),
        torch.cat(all_predictions),
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet-152 classifier on Model II lensing images.')
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_dataloader, val_dataloader, class_to_idx = build_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_accuracy = 0.0
    target_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    for epoch in tqdm(range(args.epochs)):
        train_loss, train_accuracy = train_one_epoch(
            model, train_dataloader, criterion, optimiser, device
        )
        val_loss, val_accuracy, val_labels, val_predictions = validate(
            model, val_dataloader, criterion, device
        )

        print(f'\nValidation classification report (epoch {epoch}):')
        print(
            classification_report(
                val_labels.numpy(),
                val_predictions.numpy(),
                labels=list(range(len(target_names))),
                target_names=target_names,
                zero_division=0,
            )
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = Path(args.checkpoint)
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
    print(f'best_val_acc={best_val_accuracy:.4f} checkpoint={args.checkpoint}')


if __name__ == '__main__':
    main()
