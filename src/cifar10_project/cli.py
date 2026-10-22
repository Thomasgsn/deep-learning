"""Commandes pratiques pour CIFAR-10."""
import argparse
from .dataset import dataset_statistics
from .utils import save_json
from .visualization import plot_comparison, plot_confusion, plot_dataset, plot_history, plot_prediction_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stats = sub.add_parser("stats"); stats.add_argument("--data-dir", default="data")
    figures = sub.add_parser("figures"); figures.add_argument("--data-dir", default="data")
    figures.add_argument("--checkpoint"); figures.add_argument("--history")
    compare = sub.add_parser("compare"); compare.add_argument("--checkpoints", default="checkpoints")
    compare.add_argument("--prefix", default="", help="Préfixe de phase : exploration_, complete_ ou optim_")
    args = parser.parse_args()
    if args.command == "stats":
        result = dataset_statistics(args.data_dir)
        save_json("results/metrics/dataset_statistics.json", result)
        print(result)
    elif args.command == "figures":
        plot_dataset(args.data_dir)
        if args.history:
            from pathlib import Path
            plot_history(args.history, Path("results/figures") / (Path(args.history).parent.name + "_learning.png"))
        if args.checkpoint:
            from pathlib import Path
            metrics = Path("results/metrics") / (Path(args.checkpoint).parent.name + "_test.json")
            if metrics.exists(): plot_confusion(metrics, Path("results/figures") / (metrics.stem + "_confusion.png"))
            plot_prediction_examples(args.checkpoint, args.data_dir, "results/figures/errors.png", errors=True)
            plot_prediction_examples(args.checkpoint, args.data_dir, "results/figures/correct.png", errors=False)
    elif args.command == "compare":
        output = f"results/comparisons/{args.prefix.rstrip('_') or 'model'}_comparison.csv"
        print(plot_comparison(args.checkpoints, output, args.prefix).to_string(index=False))

if __name__ == "__main__":
    main()
