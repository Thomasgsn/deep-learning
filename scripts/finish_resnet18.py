"""Termine complete_resnet18 après une interruption à l'époque 29/30.

Les poids de l'époque 29 sont rechargés (l'état de l'optimiseur n'a pas été sauvegardé) et la 30e époque
est jouée au learning rate cosinus correspondant, avec un momentum réinitialisé. Le résumé est marqué
`resumed_from_checkpoint` et `training_seconds_estimated` : à signaler dans le rapport.
L'historique 1-29 est reconstruit depuis logs/complete.log (train_loss, lr et durées non conservés).
"""
import csv
import math
import re
from pathlib import Path

import torch
from torch import nn

from cifar10_project.dataset import get_train_validation_loaders
from cifar10_project.models import build_model
from cifar10_project.train import benchmark_inference, run_epoch
from cifar10_project.utils import choose_device, count_parameters, now, save_json, seed_everything

def main():
    NAME, EPOCHS, DONE = "complete_resnet18", 30, 29
    output_dir = Path("checkpoints") / NAME
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == DONE, f"best.pt est à l'époque {checkpoint['epoch']}, pas {DONE}"
    config = checkpoint["config"]
    lr = config["learning_rate"] * (1 + math.cos(math.pi * DONE / EPOCHS)) / 2

    seed_everything(int(config.get("seed", 42)), deterministic=False)
    device = choose_device("auto")
    train_loader, val_loader, dataset_info = get_train_validation_loaders(
        "data", batch_size=config["batch_size"], seed=int(config.get("seed", 42)), augmentation=config["augmentation"],
        workers=2, device_type=device.type)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=config["weight_decay"], nesterov=True)
    criterion = nn.CrossEntropyLoss()

    start = now()
    train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
    val_metrics = run_epoch(model, val_loader, criterion, device)
    last_seconds = now() - start
    print(f"époque 30 lr={lr:.5f} train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
          f"val_loss={val_metrics['loss']:.4f}")

    best_epoch, best_metrics = DONE, checkpoint["validation_metrics"]
    if val_metrics["loss"] < best_metrics["loss"]:
        best_epoch, best_metrics = EPOCHS, val_metrics
        torch.save({"model": model.state_dict(), "config": config, "epoch": EPOCHS,
                    "validation_metrics": val_metrics, "dataset_info": dataset_info}, output_dir / "best.pt")
    else:
        model.load_state_dict(checkpoint["model"])

    pattern = re.compile(rf"\[{NAME}\] (\d+)/\d+ train_acc=([\d.]+) val_acc=([\d.]+) val_loss=([\d.]+)")
    rows = {}
    for line in Path("logs/complete.log").read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            epoch = int(match[1])
            rows[epoch] = {"epoch": epoch, "learning_rate": config["learning_rate"] * (1 + math.cos(math.pi * (epoch - 1) / EPOCHS)) / 2,
                           "train_loss": "", "train_accuracy": float(match[2]),
                           "validation_loss": float(match[4]), "validation_accuracy": float(match[3]), "epoch_seconds": ""}
    rows[EPOCHS] = {"epoch": EPOCHS, "learning_rate": lr, "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"], "validation_loss": val_metrics["loss"],
                    "validation_accuracy": val_metrics["accuracy"], "epoch_seconds": last_seconds}
    assert sorted(rows) == list(range(1, EPOCHS + 1)), "historique incomplet dans logs/complete.log"
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[1]))
        writer.writeheader()
        writer.writerows(rows[e] for e in sorted(rows))

    result = {"name": NAME, "model": config["model"], "parameters": count_parameters(model), "best_epoch": best_epoch,
              "best_validation_accuracy": best_metrics["accuracy"], "best_validation_loss": best_metrics["loss"],
              "training_seconds": last_seconds * EPOCHS, "training_seconds_estimated": True,
              "resumed_from_checkpoint": f"époque {DONE}, optimiseur réinitialisé",
              "inference_ms_per_image": benchmark_inference(model, val_loader, device), "peak_gpu_memory_mb": None,
              "device": str(device), "epochs_completed": EPOCHS, "checkpoint": str(output_dir / "best.pt"),
              "config": config, "dataset": dataset_info}
    save_json(output_dir / "summary.json", result)
    print(f"OK : meilleur epoch {best_epoch}, val_acc={best_metrics['accuracy']:.4f}")



if __name__ == "__main__":
    main()
