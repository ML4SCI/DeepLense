from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.my_dataset_vit import MyDatasetViT


def get_vit_transforms(image_size: int):
    """
    Returns transforms for Vision Transformer models.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Lambda(lambda x: x.unsqueeze(0))  # add channel dim
    ])


def create_dataloaders(cfg):
    """
    Create train, validation, and test dataloaders from config.
    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = cfg["data"]

    dataset_root = Path(data_cfg["dataset_root"])

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    # Fail fast if dataset is missing
    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split not found: {split_dir}\n"
                "Please ensure the dataset archive is extracted correctly."
            )

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    image_size = data_cfg.get("image_size", 64)

    vit_transforms = get_vit_transforms(image_size)

    train_dataset = MyDatasetViT(train_dir, vit_transforms)
    val_dataset = MyDatasetViT(val_dir, vit_transforms)
    test_dataset = MyDatasetViT(test_dir, vit_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
