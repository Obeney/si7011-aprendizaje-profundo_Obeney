"""Entrenamiento, evaluación e inferencia."""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 200,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    metric.reset()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        metric.update(outputs, labels)

    return running_loss / len(loader.dataset), metric.compute().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 200,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    metric.reset()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        metric.update(outputs, labels)

    return running_loss / len(loader.dataset), metric.compute().item()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 10,
    lr: float = 1e-4,
    num_classes: int = 200,
) -> tuple[nn.Module, dict[str, Any], float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{max_epochs} | LR: {lr_now:.6f} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

    model.load_state_dict(best_wts)
    return model, history, best_val_acc


@torch.no_grad()
def predict_test(
    model: nn.Module,
    test_loader: DataLoader,
    filenames: list[str],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    predictions: list[int] = []
    for images in test_loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1).cpu().tolist()
        predictions.extend(preds)
    return pd.DataFrame({"File": filenames, "Encoded_Label": predictions})
