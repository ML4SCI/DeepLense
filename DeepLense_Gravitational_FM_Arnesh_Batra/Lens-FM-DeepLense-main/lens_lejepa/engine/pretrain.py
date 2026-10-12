"""Self-supervised pretraining.

Two families share this loop:

* teacher-free LeJEPA methods (``lens_lejepa``, ``lejepa``): encoder + projector;
* masked-prediction references (``ijepa``, ``lens_jepa``, ``lens_jepa_sym``,
  ``lens_jepa_focus``): encoder + predictor + EMA teacher + block masks.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import PretrainConfig, config_to_dict
from ..data import UnlabelledLensDataset, make_loader
from ..data.io import standardise
from ..models import BlockMaskCollator, JEPAPredictor, LeJEPAProjector, MaskConfig, build_encoder, count_parameters
from ..ssl import LensLeJEPAObjective, MaskedJEPAObjective, ema_update, pretraining_diagnostics
from ..utils import WarmupCosine, append_csv, check_finite, logger, make_run_dir, resolve_device, set_seed, write_json


def _param_groups(*modules: nn.Module) -> list[dict]:
    """Weight decay on matrices only; biases and norm parameters are excluded."""
    decay, no_decay = [], []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim == 1 or name.endswith("bias") else decay).append(param)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0, "no_decay": True}]


class _Setup:
    """Everything that differs between the two method families."""

    def __init__(self, config: PretrainConfig, device: torch.device, dataset) -> None:
        self.config = config
        self.encoder = build_encoder(config.encoder, config.image_size, config.patch_size).to(device)
        self.teacher = None
        if config.is_masked:
            m = config.masked
            grid = config.image_size // config.patch_size
            collator = BlockMaskCollator(
                MaskConfig(grid, m.npred, m.pred_scale, m.aspect_ratio, 1, m.enc_scale, m.min_keep), seed=config.seed
            )
            self.loader = DataLoader(
                dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, collate_fn=collator,
                pin_memory=torch.cuda.is_available(), persistent_workers=config.workers > 0,
            )
            self.head = JEPAPredictor(grid * grid, self.encoder.embed_dim, m.predictor_dim, m.predictor_depth, self.encoder.num_heads).to(device)
            self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
            self.objective = MaskedJEPAObjective(config.method, m, config.patch_size)
        else:
            self.loader = make_loader(dataset, config.batch_size, shuffle=True, workers=config.workers)
            self.head = LeJEPAProjector(self.encoder.embed_dim, config.projector_dim).to(device)
            self.objective = LensLeJEPAObjective(config.objective, config.patch_size).to(device)

    def step(self, batch, device: torch.device):
        if self.teacher is not None:
            images, masks_enc, masks_pred = batch
            images = standardise(images.to(device, non_blocking=True))
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            return self.objective(self.encoder, self.teacher, self.head, images, masks_enc, masks_pred)
        return self.objective(self.encoder, self.head, batch["image"].to(device, non_blocking=True))

    def state(self) -> dict:
        state = {"encoder": self.encoder.state_dict(), "head": self.head.state_dict()}
        if self.teacher is not None:
            state["teacher"] = self.teacher.state_dict()
        else:
            state["projector"] = state["head"]  # name used by the original LeJEPA checkpoints
            state["sigreg_step"] = int(self.objective.sigreg.step)
        return state

    def load(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.head.load_state_dict(state.get("head", state.get("projector")))
        if self.teacher is not None:
            self.teacher.load_state_dict(state["teacher"])
        else:
            self.objective.sigreg.step.fill_(state["sigreg_step"])


def pretrain(config: PretrainConfig, resume: str | Path | None = None) -> Path:
    """Train an encoder and return the run directory.

    Writes ``last_pretrain.pt`` and ``pretrain_metrics.csv`` after every epoch,
    so a run can be resumed with ``resume=<run_dir>/last_pretrain.pt``.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = make_run_dir(config.output_dir, f"pretrain/{config.method}", config.run_name)
    write_json(run_dir / "config.json", config_to_dict(config))

    dataset = UnlabelledLensDataset(f"{config.data_root}/{config.dataset}", config.image_size, config.max_samples, config.seed)
    setup = _Setup(config, device, dataset)
    loader = setup.loader
    steps_per_epoch = len(loader) if config.max_steps_per_epoch is None else min(len(loader), config.max_steps_per_epoch)
    trained = [setup.encoder, setup.head]
    optimizer = torch.optim.AdamW(_param_groups(*trained), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * steps_per_epoch
    lr_schedule = WarmupCosine(optimizer, "lr", config.start_lr, config.lr, config.final_lr, config.warmup_epochs * steps_per_epoch, total_steps)
    wd_schedule = WarmupCosine(optimizer, "weight_decay", config.weight_decay, config.weight_decay, config.final_weight_decay, 0, total_steps)
    logger.info(
        "Pretraining %s on %d images | %s encoder %.2fM params | %d steps/epoch | run dir %s",
        config.method, len(dataset), config.encoder, count_parameters(setup.encoder) / 1e6, steps_per_epoch, run_dir,
    )

    start_epoch, collapse_epochs = 1, 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        setup.load(state)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        collapse_epochs = state.get("collapse_epochs", 0)
        lr_schedule.step_count = wd_schedule.step_count = state["step"]
        if config.is_masked:
            loader.collate_fn.iteration = state["step"] - 1
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda")
    parameters = [p for module in trained for p in module.parameters()]
    for epoch in range(start_epoch, config.epochs + 1):
        for module in trained:
            module.train()
        sums: dict[str, float] = {}
        grad_norms: list[float] = []
        progress = tqdm(loader, total=steps_per_epoch, desc=f"pretrain {epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            if step > steps_per_epoch:
                break
            lr_schedule.step()
            wd_schedule.step()
            with autocast:
                loss, parts = setup.step(batch, device)
            check_finite("pretraining loss", loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(float(nn.utils.clip_grad_norm_(parameters, float("inf"))))
            optimizer.step()
            if setup.teacher is not None:
                # EMA momentum rises linearly from ema_start to ema_end over training.
                m = config.masked
                done = lr_schedule.step_count - 1
                ema_update(setup.teacher, setup.encoder, m.ema_start + done * (m.ema_end - m.ema_start) / max(total_steps - 1, 1))
            for key, value in {"loss": loss, **{k: v for k, v in parts.items() if k != "canonical_tokens"}}.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Diagnostics on the last batch of the epoch: detect representation collapse.
        diagnostics = pretraining_diagnostics(parts["canonical_tokens"])
        metrics = {"epoch": epoch, **{k: v / steps_per_epoch for k, v in sums.items()}, "grad_norm": float(np.mean(grad_norms)), **diagnostics}
        collapsed = diagnostics["feature_std"] < config.collapse_std_threshold or diagnostics["effective_rank_fraction"] < config.collapse_rank_threshold
        collapse_epochs = collapse_epochs + 1 if collapsed else 0
        append_csv(run_dir / "pretrain_metrics.csv", metrics)
        extra = " | ".join(f"{k} {metrics[k]:.2e}" for k in ("d4", "ring", "prediction", "symmetry") if k in metrics)
        logger.info(
            "epoch %d | loss %.4f | %s | std %.3f | eff-rank %.3f",
            epoch, metrics["loss"], extra, diagnostics["feature_std"], diagnostics["effective_rank_fraction"],
        )
        if collapse_epochs >= config.collapse_patience:
            raise RuntimeError(f"Representation collapsed for {collapse_epochs} epochs: {diagnostics}. Run rejected.")
        if not math.isfinite(metrics["loss"]):
            raise FloatingPointError("Non-finite epoch loss.")
        # Only healthy checkpoints are written, so a collapsed run can never be used downstream.
        torch.save(
            {
                "method": config.method,
                "epoch": epoch,
                "step": lr_schedule.step_count,
                **setup.state(),
                "optimizer": optimizer.state_dict(),
                "collapse_epochs": collapse_epochs,
                "config": config_to_dict(config),
                "diagnostics": diagnostics,
            },
            run_dir / "last_pretrain.pt",
        )
    logger.info("Pretraining finished: %s", run_dir / "last_pretrain.pt")
    return run_dir
