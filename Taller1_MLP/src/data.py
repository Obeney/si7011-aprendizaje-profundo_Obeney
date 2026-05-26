"""DataModule para Chest X-Ray (PyTorch puro)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .utils import build_transforms, get_data_dir


class ChestXRayDataModule:
    """Carga train/val/test con ImageFolder y pesos de clase."""

    def __init__(
        self,
        data_dir: Path | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else get_data_dir()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms, self.eval_transforms = build_transforms()

    def setup(self) -> None:
        self.train_set = datasets.ImageFolder(
            self.data_dir / "train", transform=self.train_transforms
        )
        self.val_set = datasets.ImageFolder(
            self.data_dir / "val", transform=self.eval_transforms
        )
        self.test_set = datasets.ImageFolder(
            self.data_dir / "test", transform=self.eval_transforms
        )

        self.classes = self.train_set.classes
        self.num_classes = len(self.classes)
        self.class_to_idx = self.train_set.class_to_idx

        targets = torch.tensor(self.train_set.targets)
        class_counts = torch.bincount(targets).float()
        self.class_weights = (1.0 / class_counts) / (1.0 / class_counts).sum()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
