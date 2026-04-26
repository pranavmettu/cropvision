"""Dataset and transform helpers for ImageFolder plant disease data."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_imagefolder(
    data_dir: Path,
    train: bool = True,
    max_images_per_class: int | None = None,
) -> datasets.ImageFolder | Subset:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. TODO: download PlantVillage or another "
            "ImageFolder-style dataset and place class folders under data/raw/plantvillage/."
        )
    transform = get_train_transforms() if train else get_eval_transforms()
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    if len(dataset.classes) < 2:
        raise ValueError(f"Expected at least 2 class folders in {data_dir}, found {len(dataset.classes)}.")
    if max_images_per_class is not None and max_images_per_class > 0:
        indices: list[int] = []
        per_class_counts = {class_idx: 0 for class_idx in range(len(dataset.classes))}
        for idx, (_, class_idx) in enumerate(dataset.samples):
            if per_class_counts[class_idx] < max_images_per_class:
                indices.append(idx)
                per_class_counts[class_idx] += 1
        dataset = Subset(dataset, indices)
    return dataset


def create_train_val_datasets(
    data_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
    max_images_per_class: int | None = None,
) -> Tuple[Dataset, Dataset, list[str]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")

    base_dataset = load_imagefolder(data_dir, train=True, max_images_per_class=max_images_per_class)
    class_names = base_dataset.dataset.classes if isinstance(base_dataset, Subset) else base_dataset.classes
    val_size = max(1, int(len(base_dataset) * val_split))
    train_size = len(base_dataset) - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small for a train/validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(base_dataset, [train_size, val_size], generator=generator)

    # Validation should use deterministic transforms while preserving the split indices.
    eval_dataset = load_imagefolder(data_dir, train=False, max_images_per_class=max_images_per_class)
    val_dataset.dataset = eval_dataset
    return train_dataset, val_dataset, class_names


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    max_images_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dataset, val_dataset, class_names = create_train_val_datasets(data_dir, val_split, seed, max_images_per_class)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_names
