"""Collapse diagnostics recorded once per pretraining epoch."""

from __future__ import annotations

import torch


@torch.no_grad()
def representation_stats(features: torch.Tensor, max_dims: int = 128) -> dict[str, float]:
    """Feature standard deviation and effective-rank fraction of ``[N, D]`` features.

    Effective rank is ``exp(entropy)`` of the normalised covariance spectrum
    (Roy & Vetterli, 2007), divided by the number of dimensions used, so 1.0
    means perfectly isotropic and ~0 means collapsed. Only the first
    ``max_dims`` channels are used to keep the eigen-decomposition cheap.
    """
    x = features.detach().float().reshape(-1, features.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    feature_std = float(x.std(dim=0, unbiased=False).mean())
    sample = x[:, :max_dims]
    covariance = sample.T @ sample / max(sample.shape[0], 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-12)
    p = eigenvalues / eigenvalues.sum()
    effective_rank = float(torch.exp(-(p * p.log()).sum())) / sample.shape[1]
    return {"feature_std": feature_std, "effective_rank_fraction": effective_rank}


@torch.no_grad()
def pretraining_diagnostics(tokens: torch.Tensor) -> dict[str, float]:
    """Pooled (per-image) and patch-level statistics of an encoder batch."""
    pooled = representation_stats(tokens.mean(dim=1))
    patch = representation_stats(tokens)
    return {
        **pooled,
        "patch_feature_std": patch["feature_std"],
        "patch_effective_rank_fraction": patch["effective_rank_fraction"],
    }
