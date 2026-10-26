"""Mesure la latence modèle seul d'un checkpoint pour plusieurs tailles de batch."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cifar10_project.dataset import get_test_loader
from cifar10_project.models import build_model
from cifar10_project.train import benchmark_inference_details
from cifar10_project.utils import choose_device, file_sha256, runtime_metadata, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/metrics/final_inference_benchmark.json"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 128])
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    seed_everything(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    model = build_model(config["model"], **config.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model"])
    measurements = {}
    for batch_size in args.batches:
        loader = get_test_loader(args.data_dir, batch_size=batch_size, workers=0,
                                 seed=int(config.get("seed", 42)), device_type=device.type)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        details = benchmark_inference_details(model, loader, device, warmup_batches=10, measure_batches=50)
        details["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        )
        measurements[str(batch_size)] = details
    result = {
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "measurements": measurements,
        "environment": runtime_metadata(),
    }
    save_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
