"""Modelo MLP para clasificación de imágenes aplanadas."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Red densa: input_dim -> 512 -> 128 -> num_classes."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
