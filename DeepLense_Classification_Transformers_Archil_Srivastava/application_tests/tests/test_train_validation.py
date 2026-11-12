
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

import train as train_module


class DummyModel(torch.nn.Module):
    """Small model used to test the training loop without real training."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.layer(x)


def make_fake_metrics():
    """Return the metric structure expected by train()."""
    metrics = {
        "loss": 0.5,
        "accuracy": 0.8,
        "micro_auroc": 0.85,
        "macro_auroc": 0.85,
        "ground_truth": torch.tensor([0, 1]),
        "logits": torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
    }

    for label in train_module.LABELS:
        metrics[f"{label}_auroc"] = 0.85

    return metrics


@pytest.fixture
def setup_training(monkeypatch, tmp_path):
    """Prepare a small, isolated training environment."""

    model = DummyModel()

    # Three batches of simple dummy data.
    images = torch.randn(3, 1)
    labels = torch.tensor([0, 1, 0])

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=1,
        shuffle=False,
    )

    # Replace actual training with a harmless dummy function.
    monkeypatch.setattr(
        train_module,
        "train_step",
        lambda *args, **kwargs: torch.tensor(0.5),
    )

    # Count how many times validation is called.
    validation_calls = []

    def fake_evaluate(*args, **kwargs):
        validation_calls.append(1)
        return make_fake_metrics()

    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)

    # Mock W&B so no online account or login is required.
    fake_run = SimpleNamespace(
        dir=str(tmp_path),
        summary={},
    )

    fake_wandb = SimpleNamespace(
        watch=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: None,
        save=lambda *args, **kwargs: None,
        run=fake_run,
        plot=SimpleNamespace(
            roc_curve=lambda *args, **kwargs: "fake_roc_curve"
        ),
    )

    monkeypatch.setattr(train_module, "wandb", fake_wandb)

    # Avoid writing actual model checkpoints.
    monkeypatch.setattr(
        train_module.torch,
        "save",
        lambda *args, **kwargs: None,
    )

    return model, loader, validation_calls


def run_training(model, loader, log_interval):
    return train_module.train(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=None,
        optimizer=None,
        scheduler=None,
        epochs=1,
        device="cpu",
        log_interval=log_interval,
    )


def test_validation_when_batches_less_than_interval(setup_training):
    model, loader, validation_calls = setup_training

    run_training(model, loader, log_interval=100)

    assert len(validation_calls) == 1


def test_validation_at_interval_and_epoch_end(setup_training):
    model, loader, validation_calls = setup_training

    # Three batches: validate at batch 2 and batch 3 (epoch end).
    run_training(model, loader, log_interval=2)

    assert len(validation_calls) == 2


def test_no_duplicate_validation_on_final_batch(setup_training):
    model, loader, validation_calls = setup_training

    # Three batches with interval 3: final batch is also an interval.
    run_training(model, loader, log_interval=3)

    assert len(validation_calls) == 1


@pytest.mark.parametrize("invalid_interval", [0, -1, -10])
def test_invalid_log_interval_raises_error(setup_training, invalid_interval):
    model, loader, validation_calls = setup_training

    with pytest.raises(ValueError, match="log_interval must be a positive integer"):
        run_training(model, loader, log_interval=invalid_interval)

    assert len(validation_calls) == 0
