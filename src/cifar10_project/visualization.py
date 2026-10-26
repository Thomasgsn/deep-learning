"""Graphiques de l'exploration, des apprentissages et des résultats."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision import datasets

from .utils import CLASS_NAMES, choose_device


def plot_dataset(data_dir="data", output="results/figures/dataset_examples.png"):
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for label, ax in enumerate(axes.flat):
        image = next(dataset.data[i] for i, target in enumerate(dataset.targets) if target == label)
        ax.imshow(image)
        ax.set_title(CLASS_NAMES[label])
        ax.axis("off")
    fig.suptitle("Un exemple par classe de CIFAR-10")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    counts = pd.Series(dataset.targets).value_counts().reindex(range(10))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(CLASS_NAMES, counts.values, color="#4776a8")
    ax.set_ylabel("Nombre d'images")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Répartition des classes dans le train officiel")
    fig.tight_layout()
    fig.savefig(Path(output).with_name("class_distribution.png"), dpi=160)
    plt.close(fig)


def plot_history(history_path: str | Path, output: str | Path):
    frame = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label="Entraînement")
    axes[0].plot(frame.epoch, frame.validation_loss, label="Validation")
    axes[0].set(title="Fonction de coût", xlabel="Époque", ylabel="Cross-entropy")
    axes[1].plot(frame.epoch, frame.train_accuracy, label="Entraînement")
    axes[1].plot(frame.epoch, frame.validation_accuracy, label="Validation")
    axes[1].set(title="Exactitude", xlabel="Époque", ylabel="Accuracy")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_confusion(metrics_path: str | Path, output: str | Path):
    result = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    matrix = np.asarray(result["confusion_matrix"])
    # Les valeurs brutes sont conservées dans JSON; normalisation par ligne pour lire les erreurs relatives.
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(normalized, cmap="Blues", vmin=0, vmax=1, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set(title=f"Matrice de confusion normalisée — {result['name']}", xlabel="Classe prédite", ylabel="Classe réelle")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_per_class_metrics(metrics_path: str | Path, output: str | Path):
    result = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    rows = []
    for class_name in CLASS_NAMES:
        metrics = result["per_class"][class_name]
        rows.append({"classe": class_name, "précision": metrics["precision"],
                     "rappel": metrics["recall"], "F1": metrics["f1-score"]})
    frame = pd.DataFrame(rows).set_index("classe")
    ax = frame.plot(kind="bar", figsize=(11, 4.5), ylim=(0, 1), color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set(title=f"Métriques par classe — {result['name']}", ylabel="Score", xlabel="Classe")
    ax.grid(axis="y", alpha=.25)
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def plot_comparison(results_dir="checkpoints", output="results/comparisons/model_comparison.csv", prefix=""):
    """`prefix` restreint la comparaison à une phase (ex. « complete_ ») pour ne pas mélanger les protocoles."""
    rows = []
    for path in sorted(Path(results_dir).glob(f"{prefix}*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = {"name": summary["name"], "model": summary["model"], "parameters": summary["parameters"],
               "validation_accuracy": summary["best_validation_accuracy"], "training_seconds": summary["training_seconds"],
               "inference_ms_per_image": summary["inference_ms_per_image"], "device": summary["device"]}
        test_path = Path("results/metrics") / f"{summary['name']}_test.json"
        if test_path.exists():
            test = json.loads(test_path.read_text(encoding="utf-8"))
            row.update(test_accuracy=test["accuracy"], test_f1_macro=test["f1_macro"])
        rows.append(row)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise FileNotFoundError(f"Aucun summary.json trouvé dans {results_dir}; lancez d'abord des entraînements.")
    frame = pd.DataFrame(rows).sort_values("validation_accuracy", ascending=False)
    frame.to_csv(out, index=False)
    labels = frame.name.str.replace("_", " ", regex=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("validation_accuracy", "Accuracy validation", 1.0, "#4776a8"),
        ("parameters", "Paramètres (millions)", None, "#72a050"),
        ("training_seconds", "Temps d'entraînement (minutes)", None, "#d07c3e"),
        ("inference_ms_per_image", "Inférence modèle (ms/image)", None, "#8d6cab"),
    ]
    values_by_panel = [
        frame.validation_accuracy, frame.parameters / 1e6,
        frame.training_seconds / 60, frame.inference_ms_per_image,
    ]
    for ax, (column, title, upper, color), values in zip(axes.flat, panels, values_by_panel):
        bars = ax.bar(labels, values, color=color)
        if upper is not None:
            ax.set_ylim(0, upper)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.2)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    fig.suptitle("Comparaison sur validation — test réservé à l'évaluation finale")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    return frame


def plot_prediction_examples(checkpoint_path, data_dir="data", output="results/figures/prediction_examples.png", errors=True, count=10):
    """Illustrations d'erreurs ou de bonnes prédictions sur test (après sélection du modèle)."""
    from .dataset import CIFAR_MEAN, CIFAR_STD, PILToTensor
    from .models import build_model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = choose_device("auto")
    model = build_model(config["model"], **config.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model"])
    data = datasets.CIFAR10(root=data_dir, train=False, download=True)
    mean, std = np.array(CIFAR_MEAN, dtype=np.float32), np.array(CIFAR_STD, dtype=np.float32)
    found = []
    model.eval()
    with torch.inference_mode():
        for index, (image, label) in enumerate(data):
            tensor = PILToTensor()(image)
            tensor = ((tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]).unsqueeze(0).to(device)
            logits = model(tensor)
            probabilities = logits.softmax(1)
            prediction = int(logits.argmax(1).item())
            confidence = float(probabilities[0, prediction].item())
            if (prediction != label) == errors:
                found.append((np.asarray(image), label, prediction, confidence))
                if len(found) >= count:
                    break
    if not found:
        raise ValueError("Aucun exemple correspondant trouvé.")
    cols = min(5, len(found)); rows = int(np.ceil(len(found) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows), squeeze=False)
    for ax, (image, label, prediction, confidence) in zip(axes.flat, found):
        ax.imshow(image)
        ax.set_title(f"Vrai : {CLASS_NAMES[label]}\nPrédit : {CLASS_NAMES[prediction]} ({confidence:.0%})", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(found):]: ax.axis("off")
    fig.suptitle("Erreurs de classification" if errors else "Prédictions correctes")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
