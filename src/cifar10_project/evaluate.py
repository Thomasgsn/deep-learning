"""Évaluation finale sur le test officiel et métriques détaillées."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .dataset import get_test_loader
from .models import build_model
from .utils import CLASS_NAMES, choose_device, save_json


def evaluate_checkpoint(checkpoint_path: str | Path, data_dir: str | Path = "data", output_dir: str | Path = "results/metrics",
                        device_name: str = "auto", batch_size: int = 256, workers: int = 2) -> dict:
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"], **config.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = get_test_loader(data_dir, batch_size=batch_size, workers=workers, device_type=device.type)
    y_true, y_pred, confidences = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=1)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            confidences.extend(probabilities.max(dim=1).values.cpu().tolist())
            y_true.extend(labels.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(10), zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=range(10))
    result = {"name": config["name"], "checkpoint": str(checkpoint_path), "dataset": "CIFAR-10 test officiel",
              "n_test": len(y_true), "accuracy": accuracy_score(y_true, y_pred),
              "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()), "f1_macro": float(f1.mean()),
              "per_class": classification_report(y_true, y_pred, labels=range(10), target_names=CLASS_NAMES,
                                                  output_dict=True, zero_division=0),
              "confusion_matrix": matrix.tolist(), "mean_confidence": float(np.mean(confidences)),
              "inference_ms_per_image": elapsed * 1000 / len(y_true),
              "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "device": str(device)}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / f"{config['name']}_test.json", result)
    np.save(out / f"{config['name']}_confusion.npy", matrix)
    print(f"Test {config['name']}: accuracy={result['accuracy']:.4f}, macro-F1={result['f1_macro']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/metrics")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.data_dir, args.output_dir, args.device)

if __name__ == "__main__":
    main()
