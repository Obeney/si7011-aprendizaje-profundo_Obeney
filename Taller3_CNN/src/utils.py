"""Utilidades — Taller 3 RNN Bike Sharing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("data/bike_processed")
SEQUENCE_LENGTH = 24
DELAY = 1
SAMPLING_RATE = 1
BATCH_SIZE = 256


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
