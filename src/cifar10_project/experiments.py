"""Lanceurs des phases exploratoire, complète et optimisation limitée."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import train_model
from .utils import save_json


def run_phase(phase: str, data_dir="data", device="auto", epochs: int | None = None) -> list[dict]:
    if phase == "exploration":
        models = [("mlp", "mlp"), ("cnn", "cnn"), ("resnet18", "resnet18"), ("wide_resnet", "wide_resnet")]
        common = {"epochs": epochs or 5, "train_limit": 4000, "batch_size": 128, "patience": epochs or 5,
                  "scheduler": "cosine", "augmentation": "standard", "optimizer": "adamw", "learning_rate": .001}
    elif phase == "complete":
        models = [("cnn", "cnn"), ("resnet18", "resnet18"), ("wide_resnet", "wide_resnet")]
        common = {"epochs": epochs or 30, "batch_size": 128, "patience": 8, "scheduler": "cosine",
                  "augmentation": "standard", "optimizer": "sgd", "learning_rate": .1, "weight_decay": 5e-4}
    elif phase == "optimisation":
        # Recherche courte sur 4 000 images. Chaque variante ne modifie qu'un facteur,
        # puis seule la meilleure configuration doit être confirmée sur le train complet.
        base = {"epochs": epochs or 10, "train_limit": 4000, "batch_size": 128,
                "patience": epochs or 10, "scheduler": "cosine", "augmentation": "standard",
                "optimizer": "sgd", "learning_rate": .1, "weight_decay": 5e-4,
                "mix": "none", "label_smoothing": 0.0}
        variants = [
            ("ref", base),
            ("mixup", {**base, "mix": "mixup", "mix_alpha": .2}),
            ("label_smoothing", {**base, "label_smoothing": .1}),
        ]
        results = []
        for name, config in variants:
            full = {**config, "name": f"optim_subset_{name}", "model": "resnet18",
                    "data_dir": data_dir, "device": device}
            results.append(train_model(full))
        save_json(Path("results/metrics") / "optimisation_runs.json", results)
        return results
    else:
        raise ValueError("phase doit être exploration, complete ou optimisation")
    results = []
    for model_name, label in models:
        config = {**common, "name": f"{phase}_{label}", "model": model_name, "data_dir": data_dir, "device": device}
        results.append(train_model(config))
    save_json(Path("results/metrics") / f"{phase}_runs.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["exploration", "complete", "optimisation"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    run_phase(args.phase, args.data_dir, args.device, args.epochs)

if __name__ == "__main__":
    main()
