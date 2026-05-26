"""Generación de predicciones para el conjunto de competencia."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_competition(
    model: nn.Module,
    loader: DataLoader,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    ids: list[str] = []
    preds: list[str] = []

    for images, names in loader:
        x = images.view(images.size(0), -1).to(device)
        indices = model(x).argmax(dim=1).cpu().tolist()
        labels = [idx_to_class[i] for i in indices]
        ids.extend(list(names))
        preds.extend(labels)

    return pd.DataFrame({"id": ids, "predicted_label": preds})


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
