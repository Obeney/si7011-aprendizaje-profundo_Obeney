"""Modelo de regresión logística para imágenes aplanadas."""

from __future__ import annotations

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Una capa lineal sobre el vector aplanado de la imagen."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)
