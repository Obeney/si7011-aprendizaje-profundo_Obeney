"""Dataset CSV + rutas de imágenes Tiny ImageNet."""

from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CSVImageDataset(Dataset):
    """Lee rutas desde un CSV y carga imágenes RGB."""

    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None) -> None:
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.has_labels = "Encoded_Label" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["File"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.has_labels:
            return img, torch.tensor(row["Encoded_Label"], dtype=torch.long)
        return img
