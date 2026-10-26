"""Smoke test CPU sans téléchargement : forward, loss/backward, métriques et checkpoint."""
from pathlib import Path
import tempfile
import torch
from torch import nn
from cifar10_project.models import build_model
from cifar10_project.train import run_epoch
from cifar10_project.utils import count_parameters


def main():
    batch = torch.randn(4, 3, 32, 32)
    temporary_root = Path(".tmp")
    temporary_root.mkdir(exist_ok=True)
    for name in ("mlp", "cnn", "resnet18", "wide_resnet"):
        model = build_model(name)
        output = model(batch)
        assert output.shape == (4, 10), (name, output.shape)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = run_epoch(model, [(batch, torch.tensor([0, 1, 2, 3]))], nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)
        assert 0 <= metrics["accuracy"] <= 1 and metrics["loss"] > 0 and count_parameters(model) > 0
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = build_model(name); restored.load_state_dict(torch.load(path, weights_only=True))
            assert restored(batch).shape == (4, 10)
        print(f"OK {name}: {count_parameters(model):,} paramètres; entraînement, métriques et sauvegarde/restauration")

if __name__ == "__main__": main()
