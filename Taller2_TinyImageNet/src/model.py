"""ResNet34 con transfer learning para 200 clases."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .utils import NUM_CLASSES


def build_resnet34(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
