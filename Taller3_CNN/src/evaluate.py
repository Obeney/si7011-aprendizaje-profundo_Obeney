"""Evaluación en test."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import evaluate_mse_mae


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    count_std: float,
) -> dict[str, float]:
    criterion = nn.MSELoss()
    _, mae = evaluate_mse_mae(model, loader, criterion, device, count_std)

    preds, targets = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        preds.extend(model(x).cpu().numpy())
        targets.extend(y.numpy())

    preds = np.array(preds) * count_std
    targets = np.array(targets) * count_std
    errors = preds - targets
    rmse = float(np.sqrt(np.mean(errors**2)))

    return {"mae": mae, "rmse": rmse, "max_abs_error": float(np.abs(errors).max())}
