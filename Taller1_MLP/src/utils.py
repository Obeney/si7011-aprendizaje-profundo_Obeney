"""Utilidades compartidas para Taller 1 — regresión logística."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

KAGGLE_DATA_DIR = Path(
    "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray"
)


def get_data_dir() -> Path:
    """Ruta al dataset según el entorno (Kaggle o local)."""
    if KAGGLE_DATA_DIR.exists():
        return KAGGLE_DATA_DIR
    local = Path(os.environ.get("CHEST_XRAY_DIR", "data/chest_xray"))
    return local


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transforms, eval_transforms


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
