"""Task-agnostic downstream training and held-out evaluation.

``finetune`` trains one task, validates every ``val_interval`` epochs (and on
the last epoch), keeps the checkpoint with the best validation ``task.monitor``
and, if ``test_dataset`` is set, evaluates *only that checkpoint* on the
held-out set. ``evaluate`` re-runs the held-out evaluation for any saved
checkpoint, on any test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..config import FinetuneConfig, config_to_dict
from ..data import make_loader
from ..models import adapter_param_groups, count_parameters, is_encoder
from ..tasks import Task, build_task
from ..utils import append_csv, check_finite, logger, make_run_dir, resolve_device, set_seed, write_json


def _autocast(config: FinetuneConfig, device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda")


def _optimizer(model: nn.Module, config: FinetuneConfig) -> torch.optim.Optimizer:
    uses_adapter = is_encoder(config.backbone) and config.adaptation not in ("full", "linear_probe")
    if uses_adapter:
        groups = adapter_param_groups(model, config.lr, config.weight_decay, config.adaptation, config.loraplus_ratio)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config.lr, "weight_decay": config.weight_decay}]
    return torch.optim.AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)


@torch.no_grad()
def run_inference(task: Task, model: nn.Module, loader, device: torch.device) -> dict[str, Any]:
    """Run ``model`` over ``loader`` and return ``task.metrics`` plus raw outputs."""
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(*task.model_input(batch))  # full precision: metrics must not depend on bf16 rounding
        check_finite("model output", output)
        for key, value in task.collect(output, batch).items():
            collected.setdefault(key, []).append(value)
    outputs = {key: np.concatenate(values) for key, values in collected.items()}
    return {"metrics": task.metrics(outputs), "outputs": outputs}


def _is_better(task: Task, value: float, best: float | None) -> bool:
    if best is None:
        return True
    return value > best if task.higher_is_better else value < best


def _build_trained_model(checkpoint: dict, device: torch.device) -> tuple[Task, nn.Module, FinetuneConfig]:
    config = FinetuneConfig(**checkpoint["config"])
    config.pretrained = None  # the adapted weights are already inside the checkpoint
    task = build_task(config)
    task.load_state_dict(checkpoint.get("task_state", {}))
    model = task.build_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    return task, model.to(device), config


def evaluate(checkpoint_path: str | Path, test_root: str, device_name: str = "cuda", batch_size: int | None = None, workers: int = 8) -> dict[str, Any]:
    """Held-out evaluation of a saved downstream checkpoint on ``test_root``."""
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task, model, config = _build_trained_model(checkpoint, device)
    loader = make_loader(task.build_test_dataset(test_root), batch_size or config.batch_size, shuffle=False, workers=workers)
    result = run_inference(task, model, loader, device)
    metrics = {"test_root": str(test_root), "checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"], **result["metrics"]}
    logger.info("Held-out %s on %s: %s", task.name, test_root, {k: v for k, v in metrics.items() if isinstance(v, float)})
    return metrics


def finetune(config: FinetuneConfig) -> Path:
    """Train ``config.task`` and return the run directory."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    task = build_task(config)
    shots = "full" if config.shots is None else f"shots{config.shots}"
    tag = config.adaptation if is_encoder(config.backbone) else "scratch"
    run_dir = make_run_dir(config.output_dir, f"{config.task}/{config.backbone}_{tag}", config.run_name or f"{config.train_dataset}_{shots}_seed{config.seed}")
    write_json(run_dir / "config.json", config_to_dict(config))

    train_set, val_set = task.build_datasets()
    train_loader = make_loader(train_set, config.batch_size, shuffle=True, workers=config.workers)
    val_loader = make_loader(val_set, config.batch_size, shuffle=False, workers=config.workers)
    model = task.build_model().to(device)
    trainable = count_parameters(model, trainable_only=True)
    logger.info(
        "%s | %s + %s | train %d / val %d | trainable %.3fM of %.2fM | run dir %s",
        task.name, config.backbone, tag, len(train_set), len(val_set), trainable / 1e6, count_parameters(model) / 1e6, run_dir,
    )

    best_value: float | None = None
    best_epoch = 0
    if trainable > 0:
        optimizer = _optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
        steps = len(train_loader) if config.max_steps_per_epoch is None else min(len(train_loader), config.max_steps_per_epoch)
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(train_loader, total=steps, desc=f"{task.name} {epoch}/{config.epochs}", leave=False), start=1):
                if step > steps:
                    break
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with _autocast(config, device):
                    loss = task.loss(model(*task.model_input(batch)), batch)
                check_finite("training loss", loss)
                (loss / config.grad_accum).backward()
                if step % config.grad_accum == 0 or step == steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            scheduler.step()

            if epoch % config.val_interval and epoch != config.epochs:
                continue
            val = run_inference(task, model, val_loader, device)["metrics"]
            append_csv(run_dir / "val_metrics.csv", {"epoch": epoch, "train_loss": float(np.mean(losses)), **val})
            logger.info("epoch %d | train loss %.4f | val %s %.4f", epoch, np.mean(losses), task.monitor, val[task.monitor])
            if _is_better(task, val[task.monitor], best_value):
                best_value, best_epoch = val[task.monitor], epoch
                _save(run_dir / "best.pt", model, task, config, epoch, val)
    else:
        # Parameter-free baseline (bicubic): nothing to train, just record validation.
        val = run_inference(task, model, val_loader, device)["metrics"]
        best_value = val[task.monitor]
        _save(run_dir / "best.pt", model, task, config, 0, val)

    summary: dict[str, Any] = {"best_epoch": best_epoch, f"val_{task.monitor}": best_value, "trainable_parameters": trainable}
    if config.test_dataset:
        summary["test"] = evaluate(run_dir / "best.pt", f"{config.data_root}/{config.test_dataset}", config.device, config.batch_size, config.workers)
    write_json(run_dir / "summary.json", summary)
    logger.info("Done. Best validation %s = %.4f at epoch %d.", task.monitor, best_value, best_epoch)
    return run_dir


def _save(path: Path, model: nn.Module, task: Task, config: FinetuneConfig, epoch: int, val_metrics: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "task_state": task.state_dict(),
            "config": config_to_dict(config),
            "epoch": epoch,
            "val_metrics": {k: v for k, v in val_metrics.items() if np.isscalar(v)},
        },
        path,
    )
