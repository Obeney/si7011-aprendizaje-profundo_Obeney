"""Entrenamiento con early stopping."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / n


@torch.no_grad()
def evaluate_mse_mae(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    count_std: float = 1.0,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        total_loss += criterion(preds, y).item() * x.size(0)
        total_mae += torch.abs(preds - y).sum().item()
    mse = total_loss / n
    mae_orig = (total_mae / n) * count_std
    return mse, mae_orig


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    count_std: float,
    epochs: int = 150,
    patience: int = 20,
    lr: float = 1e-3,
) -> tuple[nn.Module, dict[str, Any]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_wts = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = evaluate_mse_mae(
            model, val_loader, criterion, device, count_std
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} bikes"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_loss)

    model.load_state_dict(best_wts)
    return model, history
