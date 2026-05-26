"""Bucle de entrenamiento y validación."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_f1: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n = len(loader.dataset)
    preds_list: list[float] = []
    labels_list: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        preds_list.extend(preds.cpu().numpy().flatten())
        labels_list.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / n
    acc = accuracy_score(labels_list, preds_list)
    f1 = f1_score(labels_list, preds_list)
    return avg_loss, acc, f1


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
) -> list[EpochMetrics]:
    history: list[EpochMetrics] = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        history.append(
            EpochMetrics(train_loss, val_loss, val_acc, val_f1)
        )
        print(
            f"Epoch {epoch + 1:2d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

    return history
