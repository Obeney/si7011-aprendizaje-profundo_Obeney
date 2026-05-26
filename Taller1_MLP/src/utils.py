"""Utilidades compartidas — Taller 1 MLP Intel Image Classification."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

IMG_SIZE = 150
N_FEATURES = IMG_SIZE * IMG_SIZE
N_CLASSES = 6

KAGGLE_TRAIN = Path(
    "/kaggle/input/datasets/puneet6060/intel-image-classification/seg_train/seg_train"
)
KAGGLE_VAL = Path(
    "/kaggle/input/datasets/puneet6060/intel-image-classification/seg_test/seg_test"
)
KAGGLE_PRED = Path(
    "/kaggle/input/datasets/puneet6060/intel-image-classification/seg_pred/seg_pred"
)


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
