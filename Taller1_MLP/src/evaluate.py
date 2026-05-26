"""Métricas y visualización en test."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    model.eval()
    preds_list: list[float] = []
    labels_list: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        preds_list.extend(preds.cpu().numpy().flatten())
        labels_list.extend(labels.cpu().numpy().flatten())

    return preds_list, labels_list


def print_report(y_true: list[float], y_pred: list[float]) -> confusion_matrix:
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusión:")
    print(cm)
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
    return cm


def plot_confusion_matrix(
    cm,
    save_path: Path | None = None,
    title: str = "Matriz de confusión en test",
) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
