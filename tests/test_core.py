import numpy as np
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from cifar10_project.dataset import TransformSubset, make_transforms, stratified_indices
from cifar10_project.evaluate import evaluate_checkpoint
from cifar10_project.models import build_model
from cifar10_project.train import mixed_batch, train_model
from cifar10_project.visualization import plot_comparison, plot_confusion, plot_history
from cifar10_project.utils import count_parameters


def test_stratified_split_is_reproducible_and_balanced():
    targets = np.repeat(np.arange(10), 100)
    train, validation = stratified_indices(targets, .2, 19)
    assert np.array_equal(validation, stratified_indices(targets, .2, 19)[1])
    assert set(train).isdisjoint(validation)
    assert len(validation) == 200
    assert np.all(np.bincount(targets[validation]) == 20)


def test_models_forward_and_parameter_count():
    batch = torch.randn(2, 3, 32, 32)
    for name in ("mlp", "cnn", "resnet18", "wide_resnet"):
        model = build_model(name)
        assert model(batch).shape == (2, 10)
        assert count_parameters(model) > 0


def test_mixup_and_cutmix_preserve_shapes_and_labels():
    images, labels = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
    for method in ("mixup", "cutmix"):
        result, first, second, weight = mixed_batch(images, labels, method, 1.0)
        assert result.shape == images.shape
        assert first.shape == labels.shape == second.shape
        assert 0 <= weight <= 1


def test_metrics_are_finite():
    truth, prediction = [0, 1, 1, 2], [0, 1, 2, 2]
    assert accuracy_score(truth, prediction) == .75
    assert 0 <= f1_score(truth, prediction, average="macro") <= 1


def test_none_augmentation_is_deterministic_and_matches_evaluation():
    train_transform, eval_transform = make_transforms("none")
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    assert torch.equal(train_transform(image), eval_transform(image))


def test_supported_augmentations_return_cifar_tensors():
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    for name in ("none", "standard", "jitter", "erase"):
        train_transform, _ = make_transforms(name)
        tensor = train_transform(image)
        assert tensor.shape == (3, 32, 32)
        assert tensor.dtype == torch.float32


def test_dataset_subset_and_dataloader_pipeline():
    class TinyCIFAR:
        data = np.zeros((8, 32, 32, 3), dtype=np.uint8)
        targets = [i % 2 for i in range(8)]

    transform, _ = make_transforms("standard")
    subset = TransformSubset(TinyCIFAR(), np.arange(8), transform)
    images, labels = next(iter(DataLoader(subset, batch_size=4, num_workers=0)))
    assert images.shape == (4, 3, 32, 32)
    assert labels.tolist() == [0, 1, 0, 1]


def test_one_epoch_saves_and_reloads_best_checkpoint(tmp_path, monkeypatch):
    train_data = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 10)
    val_data = TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4))
    loaders = (DataLoader(train_data, batch_size=4), DataLoader(val_data, batch_size=4))
    monkeypatch.setattr(
        "cifar10_project.train.get_train_validation_loaders",
        lambda *args, **kwargs: (*loaders, {"train_size": 8, "validation_size": 4}),
    )
    config = {
        "name": "unit_train", "model": "mlp", "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "checkpoints"), "device": "cpu", "seed": 7,
        "batch_size": 4, "workers": 0, "epochs": 1, "patience": 1,
        "optimizer": "adamw", "learning_rate": 0.001, "scheduler": "cosine",
        "augmentation": "none", "tensorboard": False,
    }
    result = train_model(config)
    assert result["epochs_completed"] == 1
    assert result["checkpoint"]
    assert (tmp_path / "checkpoints" / "unit_train" / "best.pt").exists()
    assert (tmp_path / "checkpoints" / "unit_train" / "history.csv").exists()


def test_checkpoint_evaluation_writes_real_metrics(tmp_path, monkeypatch):
    model = build_model("mlp")
    config = {"name": "unit_eval", "model": "mlp", "model_kwargs": {}}
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    test_data = TensorDataset(torch.randn(6, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4, 5]))
    monkeypatch.setattr("cifar10_project.evaluate.get_test_loader", lambda *args, **kwargs: DataLoader(test_data, batch_size=3))
    result = evaluate_checkpoint(checkpoint, tmp_path / "data", tmp_path / "metrics", "cpu", batch_size=3, workers=0)
    assert result["n_test"] == 6
    assert sum(map(sum, result["confusion_matrix"])) == 6
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["f1_macro"] <= 1
    assert (tmp_path / "metrics" / "unit_eval_test.json").exists()
    assert (tmp_path / "metrics" / "unit_eval_confusion.npy").exists()


def test_result_plots_and_comparison_are_generated(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        "epoch,train_loss,train_accuracy,validation_loss,validation_accuracy\n1,1.2,0.6,1.1,0.65\n",
        encoding="utf-8",
    )
    history_plot = tmp_path / "history.png"
    plot_history(history, history_plot)
    assert history_plot.exists() and history_plot.stat().st_size > 0

    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        '{"name":"unit","confusion_matrix":[' + ','.join(['[1,0,0,0,0,0,0,0,0,0]'] + ['[0,0,0,0,0,0,0,0,0,0]'] * 9) + ']}',
        encoding="utf-8",
    )
    confusion_plot = tmp_path / "confusion.png"
    plot_confusion(metrics, confusion_plot)
    assert confusion_plot.exists() and confusion_plot.stat().st_size > 0

    model_dir = tmp_path / "checkpoints" / "unit_model"
    model_dir.mkdir(parents=True)
    (model_dir / "summary.json").write_text(
        '{"name":"unit_model","model":"mlp","parameters":100,"best_validation_accuracy":0.65,'
        '"training_seconds":1.0,"inference_ms_per_image":0.2,"device":"cpu"}',
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.csv"
    frame = plot_comparison(tmp_path / "checkpoints", comparison)
    assert len(frame) == 1
    assert comparison.exists() and comparison.with_suffix(".png").exists()


def test_mixup_training_accuracy_is_weighted_by_mix_ratio(monkeypatch):
    from torch import nn
    from cifar10_project.train import run_epoch

    images, labels = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
    monkeypatch.setattr("cifar10_project.train.mixed_batch", lambda *args: (images, labels, labels.flip(0), 0.75))
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    metrics = run_epoch(model, [(images, labels)], nn.CrossEntropyLoss(), torch.device("cpu"),
                        torch.optim.SGD(model.parameters(), lr=0.0), mix="mixup")
    assert 0 <= metrics["accuracy"] <= 1


def test_comparison_prefix_filters_phases(tmp_path):
    for name in ("exploration_a", "complete_a"):
        directory = tmp_path / "checkpoints" / name
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            f'{{"name":"{name}","model":"mlp","parameters":1,"best_validation_accuracy":0.5,'
            '"training_seconds":1.0,"inference_ms_per_image":0.2,"device":"cpu"}', encoding="utf-8")
    frame = plot_comparison(tmp_path / "checkpoints", tmp_path / "c.csv", prefix="complete_")
    assert list(frame.name) == ["complete_a"]
