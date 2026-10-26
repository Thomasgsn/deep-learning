"""Utilitaires communs pour des expériences reproductibles sur CIFAR-10."""
from __future__ import annotations

import json
import hashlib
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

CLASS_NAMES = ("avion", "automobile", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:  # compatibilité avec les versions plus anciennes de PyTorch
        torch.use_deterministic_algorithms(deterministic)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA a été demandé mais n'est pas disponible.")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS a été demandé mais n'est pas disponible.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def file_sha256(path: str | Path) -> str:
    """Calcule l'empreinte d'un artefact sans le charger entièrement en mémoire."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: str | Path = ".") -> str:
    """Empreinte stable du code Python effectivement présent dans le dépôt."""
    root = Path(root).resolve()
    files = sorted(
        path for directory in (root / "src", root / "scripts")
        if directory.exists()
        for path in directory.rglob("*.py")
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    """Versions et matériel nécessaires à la traçabilité d'une expérience."""
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "indisponible"
    try:
        repository = Path.cwd().resolve().as_posix()
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-c", f"safe.directory={repository}", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit = "indisponible"
        dirty = None
    if torch.cuda.is_available():
        accelerator = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "Apple MPS"
    else:
        accelerator = "CPU"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_runtime": torch.version.cuda,
        "accelerator": accelerator,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_tree_sha256": source_tree_sha256(),
        "command": " ".join(sys.argv),
    }


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Objet non sérialisable : {type(value).__name__}")


def now() -> float:
    return time.perf_counter()


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
