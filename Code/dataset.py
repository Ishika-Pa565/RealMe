"""
dataset.py - Dataset loaders and augmentation pipeline for RealMe.
Label convention: fake = 1.0, real = 0.0
"""

import io
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class RandomJPEGCompression:
    """Simulate online image compression artifacts."""
    def __init__(self, quality_range=(60, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


def get_transforms(is_train: bool = True):
    """
    Returns image transformation pipeline.
    Input resolution: 224x224 normalized with ImageNet statistics.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            RandomJPEGCompression(quality_range=(65, 95), p=0.4),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


class DeepfakeDataset(Dataset):
    """
    Custom Dataset for Deepfake Detection.
    Maps 'fake' subfolder to label 1.0 and 'real' subfolder to label 0.0.
    """
    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

        # fake = 1.0, real = 0.0
        fake_dir = self.root_dir / "fake"
        real_dir = self.root_dir / "real"

        if fake_dir.exists():
            for p in fake_dir.iterdir():
                if p.suffix.lower() in valid_extensions:
                    self.samples.append((p, 1.0))

        if real_dir.exists():
            for p in real_dir.iterdir():
                if p.suffix.lower() in valid_extensions:
                    self.samples.append((p, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(data_dir: str | Path, batch_size: int = 32, subset: int | None = None, num_workers: int = 0):
    """
    Constructs train, validation, and test dataloaders.
    Expected directory structure:
        data_dir/
            train/ (real, fake)
            valid/ (real, fake)
            test/  (real, fake)
    """
    base_path = Path(data_dir)

    train_dir = base_path / "train"
    val_dir = base_path / "valid"
    test_dir = base_path / "test"

    train_ds = DeepfakeDataset(train_dir, transform=get_transforms(is_train=True))
    val_ds = DeepfakeDataset(val_dir, transform=get_transforms(is_train=False))
    test_ds = DeepfakeDataset(test_dir, transform=get_transforms(is_train=False))

    if subset is not None and subset > 0:
        if len(train_ds) > subset:
            train_indices = random.sample(range(len(train_ds)), subset)
            train_ds = Subset(train_ds, train_indices)
        val_subset = max(1, int(subset * 0.2))
        if len(val_ds) > val_subset:
            val_indices = random.sample(range(len(val_ds)), val_subset)
            val_ds = Subset(val_ds, val_indices)
        if len(test_ds) > val_subset:
            test_indices = random.sample(range(len(test_ds)), val_subset)
            test_ds = Subset(test_ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
