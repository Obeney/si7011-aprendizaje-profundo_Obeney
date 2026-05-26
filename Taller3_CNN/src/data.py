"""Dataset secuencial para series de tiempo."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import DELAY, SAMPLING_RATE, SEQUENCE_LENGTH


class TimeseriesDataset(Dataset):
    """Genera ventanas [seq_len, features] → target normalizado."""

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = SEQUENCE_LENGTH,
        sampling_rate: int = SAMPLING_RATE,
        start_index: int = 0,
        end_index: int | None = None,
        delay: int = DELAY,
    ) -> None:
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.delay = delay
        end_index = end_index if end_index is not None else len(data) - delay
        self.indices = np.arange(start_index, end_index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        steps = np.arange(
            start,
            start + self.sequence_length * self.sampling_rate,
            self.sampling_rate,
        )
        x = self.data[steps]
        y = self.targets[start + self.delay]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
