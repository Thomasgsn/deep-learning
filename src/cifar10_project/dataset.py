"""Chargement, découpage et statistiques CIFAR-10.

Le jeu officiel de test n'est construit que par get_test_loader(), appelé lors de
l'évaluation finale. La validation est un découpage stratifié du train officiel.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .utils import CLASS_NAMES, worker_seed

# Statistiques usuelles CIFAR-10, calculées sur le train officiel.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class PILToTensor:
    """Convertit une image RGB sans dépendre du pont NumPy de PyTorch."""
    def __call__(self, image):
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError(f"Image PIL attendue, reçu : {type(image).__name__}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        # frombuffer évite torch.from_numpy, incompatible avec certaines paires
        # PyTorch compilé pour NumPy 1.x / NumPy 2.x.
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return pixels.view(image.height, image.width, 3).permute(2, 0, 1).to(torch.float32).div_(255.0)


def make_transforms(augmentation: str = "standard") -> tuple[transforms.Compose, transforms.Compose]:
    """Transformations train/évaluation; les choix sont nommés et comparables."""
    if augmentation not in {"none", "standard", "jitter", "erase"}:
        raise ValueError(f"Augmentation inconnue: {augmentation}")
    train_ops: list = []
    if augmentation != "none":
        train_ops.append(transforms.RandomCrop(32, padding=4))
    if augmentation in {"standard", "jitter", "erase"}:
        train_ops.append(transforms.RandomHorizontalFlip())
    if augmentation == "jitter":
        # torchvision 0.17 et NumPy 2 lèvent OverflowError sur le hue PIL négatif.
        # Ces trois variations suffisent pour tester une perturbation photométrique légère.
        train_ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15))
    train_ops += [PILToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if augmentation == "erase":
        train_ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)))
    eval_ops = transforms.Compose([PILToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return transforms.Compose(train_ops), eval_ops


class TransformSubset(Dataset):
    """Sous-ensemble partageant les mêmes images mais avec son propre transform."""
    def __init__(self, dataset: Dataset, indices: np.ndarray, transform) -> None:
        self.dataset, self.indices, self.transform = dataset, indices.tolist(), transform
        self.targets = np.asarray(dataset.targets)[indices].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, label = self.dataset.data[self.indices[item]], int(self.dataset.targets[self.indices[item]])
        from PIL import Image
        image = Image.fromarray(image)
        return self.transform(image), label


def stratified_indices(targets: list[int] | np.ndarray, validation_fraction: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(targets))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, np.asarray(targets)))
    return train_idx, val_idx


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, workers: int, device_type: str) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device_type == "cuda", persistent_workers=workers > 0,
                      worker_init_fn=worker_seed, generator=generator)


def get_train_validation_loaders(root: str | Path, batch_size: int = 128, validation_fraction: float = 0.1,
                                 seed: int = 42, augmentation: str = "standard", workers: int = 2,
                                 device_type: str = "cpu", download: bool = True,
                                 train_limit: int | None = None) -> tuple[DataLoader, DataLoader, dict]:
    train_transform, eval_transform = make_transforms(augmentation)
    raw = datasets.CIFAR10(root=str(root), train=True, transform=None, download=download)
    train_idx, val_idx = stratified_indices(raw.targets, validation_fraction, seed)
    # La phase exploratoire prend un sous-ensemble stratifié du seul train; la validation reste fixe.
    if train_limit is not None and train_limit < len(train_idx):
        sub_split = StratifiedShuffleSplit(n_splits=1, train_size=train_limit, random_state=seed + 1)
        relative, _ = next(sub_split.split(train_idx, np.asarray(raw.targets)[train_idx]))
        train_idx = train_idx[relative]
    train_set = TransformSubset(raw, train_idx, train_transform)
    val_set = TransformSubset(raw, val_idx, eval_transform)
    info = {"train_size": len(train_set), "validation_size": len(val_set), "validation_fraction": validation_fraction,
            "train_class_counts": dict(Counter(train_set.targets)), "validation_class_counts": dict(Counter(val_set.targets)),
            "class_names": list(CLASS_NAMES), "mean": list(CIFAR_MEAN), "std": list(CIFAR_STD), "augmentation": augmentation}
    return (_loader(train_set, batch_size, True, seed, workers, device_type),
            _loader(val_set, batch_size, False, seed, workers, device_type), info)


def get_test_loader(root: str | Path, batch_size: int = 256, workers: int = 2, seed: int = 42,
                    device_type: str = "cpu", download: bool = True) -> DataLoader:
    """À réserver à l'évaluation finale, après verrouillage du modèle."""
    _, eval_transform = make_transforms("none")
    dataset = datasets.CIFAR10(root=str(root), train=False, transform=eval_transform, download=download)
    return _loader(dataset, batch_size, False, seed, workers, device_type)


def dataset_statistics(root: str | Path, download: bool = True) -> dict:
    data = datasets.CIFAR10(root=str(root), train=True, download=download)
    counts = Counter(data.targets)
    return {"samples": len(data), "image_shape_hwc": list(data.data.shape[1:]),
            "class_counts": {CLASS_NAMES[i]: counts[i] for i in range(10)},
            "pixel_mean_0_1": (data.data.astype(np.float64) / 255.0).mean((0, 1, 2)).tolist(),
            "pixel_std_0_1": (data.data.astype(np.float64) / 255.0).std((0, 1, 2)).tolist()}
