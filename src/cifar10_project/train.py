"""Boucle d'entraînement et sauvegarde du meilleur checkpoint de validation."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .dataset import get_train_validation_loaders
from .models import build_model
from .utils import choose_device, count_parameters, now, save_json, seed_everything


def mixed_batch(images: torch.Tensor, labels: torch.Tensor, method: str, alpha: float):
    if method == "none":
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    if method == "mixup":
        return lam * images + (1 - lam) * images[permutation], labels, labels[permutation], lam
    if method == "cutmix":
        height, width = images.shape[-2:]
        ratio = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(width * ratio), int(height * ratio)
        cx, cy = np.random.randint(width), np.random.randint(height)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
        return mixed, labels, labels[permutation], lam
    raise ValueError(f"Méthode de mélange inconnue: {method}")


def run_epoch(model, loader, criterion, device, optimizer=None, mix="none", mix_alpha=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
            inputs, y_a, y_b, lam = mixed_batch(images, labels, mix, mix_alpha)
            logits = model(inputs)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                logits = model(images)
                loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        prediction = logits.argmax(dim=1)
        if training and lam < 1.0:
            # Avec MixUp/CutMix, une prédiction est correcte au prorata de chaque étiquette mélangée.
            correct += (lam * (prediction == y_a).sum() + (1 - lam) * (prediction == y_b).sum()).item()
        else:
            correct += (prediction == labels).sum().item()
        total += labels.size(0)
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def train_model(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    # Algorithmes déterministes : ×2,5 plus lent sur MPS (mesuré sur ResNet-18), donc désactivés par défaut.
    seed_everything(seed, deterministic=bool(config.get("deterministic", False)))
    device = choose_device(config.get("device", "auto"))
    root = Path(config.get("data_dir", "data"))
    output_dir = Path(config.get("output_dir", "checkpoints")) / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, dataset_info = get_train_validation_loaders(
        root, batch_size=config.get("batch_size", 128), validation_fraction=config.get("validation_fraction", .1),
        seed=seed, augmentation=config.get("augmentation", "standard"), workers=config.get("workers", 2),
        device_type=device.type, train_limit=config.get("train_limit"))
    model = build_model(config["model"], **config.get("model_kwargs", {})).to(device)
    parameters = count_parameters(model)
    optimizer_name = config.get("optimizer", "sgd").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.get("learning_rate", .1), momentum=.9,
                                    weight_decay=config.get("weight_decay", 5e-4), nesterov=True)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-3),
                                      weight_decay=config.get("weight_decay", 1e-4))
    else:
        raise ValueError("optimizer doit être 'sgd' ou 'adamw'")
    epochs = int(config.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1)) if config.get("scheduler", "cosine") == "cosine" else None
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    patience = int(config.get("patience", 8))
    best_loss, stale, history = float("inf"), 0, []
    writer = SummaryWriter(log_dir=str(Path("runs") / config["name"])) if config.get("tensorboard", False) else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = now()
    for epoch in range(1, epochs + 1):
        epoch_start = now()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer,
                                  config.get("mix", "none"), float(config.get("mix_alpha", 1.0)))
        val_metrics = run_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
               "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
               "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
               "epoch_seconds": now() - epoch_start}
        history.append(row)
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        print(f"[{config['name']}] {epoch:03d}/{epochs} train_acc={row['train_accuracy']:.4f} "
              f"val_acc={row['validation_accuracy']:.4f} val_loss={row['validation_loss']:.4f}")
        if val_metrics["loss"] < best_loss:
            best_loss, stale = val_metrics["loss"], 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "validation_metrics": val_metrics, "dataset_info": dataset_info}, output_dir / "best.pt")
        else:
            stale += 1
        if stale >= patience:
            print(f"Arrêt anticipé après {epoch} époques (patience={patience}).")
            break
        if config.get("max_seconds") and now() - start >= float(config["max_seconds"]):
            print("Arrêt à la limite de temps configurée.")
            break
    elapsed = now() - start
    if writer:
        writer.close()
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(history[0]) if history else ["epoch"])
        writer_csv.writeheader()
        writer_csv.writerows(history)
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    inference = benchmark_inference(model, val_loader, device)
    result = {"name": config["name"], "model": config["model"], "parameters": parameters,
              "best_epoch": best["epoch"], "best_validation_accuracy": best["validation_metrics"]["accuracy"],
              "best_validation_loss": best["validation_metrics"]["loss"], "training_seconds": elapsed,
              "inference_ms_per_image": inference, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "device": str(device), "epochs_completed": len(history), "checkpoint": str(output_dir / "best.pt"),
              "config": config, "dataset": dataset_info}
    save_json(output_dir / "summary.json", result)
    return result


def benchmark_inference(model, loader, device) -> float:
    model.eval()
    examples, start = 0, now()
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            model(images)
            examples += len(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (now() - start) * 1000 / max(examples, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Fichier JSON de configuration")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train_model(config)

if __name__ == "__main__":
    main()
