"""Exporte un checkpoint final allégé, vérifiable et directement évaluable."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cifar10_project.utils import file_sha256, save_json


def export_checkpoint(source: Path, output: Path) -> dict:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = {"model", "config", "epoch", "validation_metrics"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint source incomplet : {sorted(missing)}")
    payload = {key: checkpoint[key] for key in (
        "model", "config", "epoch", "validation_metrics", "dataset_info"
    ) if key in checkpoint}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    manifest = {
        "file": output.as_posix(),
        "sha256": file_sha256(output),
        "source": source.as_posix(),
        "epoch": payload["epoch"],
        "validation_metrics": payload["validation_metrics"],
        "config": payload["config"],
    }
    save_json(output.with_suffix(".json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_model.pt"))
    args = parser.parse_args()
    manifest = export_checkpoint(args.source, args.output)
    print(f"{manifest['file']} sha256={manifest['sha256']}")


if __name__ == "__main__":
    main()
