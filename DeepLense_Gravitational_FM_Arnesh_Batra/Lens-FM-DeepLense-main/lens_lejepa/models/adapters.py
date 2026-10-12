"""Parameter-efficient adapters for a frozen encoder.

Every adapter wraps an ``nn.Linear`` and leaves the original weight frozen.
Supported methods (``--set adaptation=...``):

=============  =================================================================
``lora``       ``W x + (alpha / r) B A x``                       (Hu et al., 2022)
``rslora``     ``W x + (alpha / sqrt(r)) B A x``        (Kalajdzievski, 2023)
``dora``       learned row magnitude times normalised ``W + dW``  (Liu et al., 2024)
``loraplus``   LoRA, with ``B`` trained at ``loraplus_ratio`` x the LR
``gated``      LoRA with a learned sigmoid gate on every rank component
``loha``       Hadamard product of two low-rank updates
``layerwise``  LoRA whose rank grows with block depth (``rank_schedule``)
=============  =================================================================

Attribute names (``base``, ``lora_a``, ``lora_b``, ...) match the original
experiment code so released checkpoints load unchanged.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from torch import nn

ADAPTERS = ("lora", "rslora", "dora", "loraplus", "gated", "loha", "layerwise")
# Names used in the original experiment configs.
_ALIASES = {"standard": "lora", "adaptive": "gated"}


def canonical_adapter_name(name: str) -> str:
    name = _ALIASES.get(str(name).lower(), str(name).lower())
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter {name!r}; choose from {ADAPTERS}.")
    return name


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable rank-``r`` update ``B A``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, rank_stabilised: bool = False) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("Adapter rank must be positive.")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / math.sqrt(rank) if rank_stabilised else alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)  # the adapted model starts identical to the base

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b.weight @ self.lora_a.weight) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class GatedLoRALinear(LoRALinear):
    """LoRA with a learned soft gate per rank component (AdaLoRA-style budget)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__(base, rank, alpha, dropout)
        self.rank_logits = nn.Parameter(torch.full((rank,), 4.0))  # sigmoid(4) ~ 0.98

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.lora_a(self.dropout(x)) * torch.sigmoid(self.rank_logits)
        return self.base(x) + self.lora_b(gated) * self.scaling


class DoRALinear(LoRALinear):
    """Weight-decomposed LoRA: a learned magnitude times a normalised direction."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__(base, rank, alpha, dropout)
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1, keepdim=True).clamp_min(1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direction = self.base.weight + self.delta_weight()
        weight = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8) * self.magnitude
        return F.linear(self.dropout(x), weight, self.base.bias)


class LoHALinear(nn.Module):
    """Low-rank Hadamard product update, ``(B1 A1) * (B2 A2)``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("Adapter rank must be positive.")
        self.base = base.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a1 = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b1 = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a2 = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b2 = nn.Linear(rank, base.out_features, bias=False)
        for layer in (self.lora_a1, self.lora_a2, self.lora_b2):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b1.weight)

    def delta_weight(self) -> torch.Tensor:
        first = self.lora_b1.weight @ self.lora_a1.weight
        second = self.lora_b2.weight @ self.lora_a2.weight
        return first * second * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(self.dropout(x), self.delta_weight())


def _block_index(name: str) -> int | None:
    match = re.search(r"blocks\.(\d+)\.", name)
    return None if match is None else int(match.group(1))


def _layerwise_rank(name: str, schedule: list[int], max_depth: int) -> int:
    depth = _block_index(name)
    if depth is None or max_depth <= 0:
        return schedule[0]
    stage = min(len(schedule) - 1, int(round(depth / max_depth * (len(schedule) - 1))))
    return schedule[stage]


def _make_adapter(layer: nn.Linear, method: str, rank: int, alpha: float, dropout: float) -> nn.Module:
    if method in ("lora", "loraplus", "layerwise"):
        return LoRALinear(layer, rank, alpha, dropout)
    if method == "rslora":
        return LoRALinear(layer, rank, alpha, dropout, rank_stabilised=True)
    if method == "gated":
        return GatedLoRALinear(layer, rank, alpha, dropout)
    if method == "dora":
        return DoRALinear(layer, rank, alpha, dropout)
    if method == "loha":
        return LoHALinear(layer, rank, alpha, dropout)
    raise ValueError(method)


def inject_adapters(
    encoder: nn.Module,
    method: str,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    rank_schedule: list[int] | None = None,
) -> list[str]:
    """Freeze ``encoder`` and wrap every ``nn.Linear`` inside it with an adapter.

    For a 12-block ViT that is 48 projections (qkv, proj, fc1, fc2 per block).
    Returns the names of the wrapped layers.
    """
    method = canonical_adapter_name(method)
    if method == "layerwise" and not rank_schedule:
        raise ValueError("The layerwise adapter needs a rank_schedule, e.g. [8, 16, 32, 64].")
    encoder.requires_grad_(False)
    # Linear layers inside nn.MultiheadAttention (out_proj) are left alone:
    # the fused attention kernel reads their weights directly.
    targets = [
        (f"{parent_name}.{child_name}".lstrip("."), parent, child_name, child)
        for parent_name, parent in list(encoder.named_modules())
        if not isinstance(parent, nn.MultiheadAttention)
        for child_name, child in list(parent.named_children())
        if isinstance(child, nn.Linear)
    ]
    if not targets:
        raise RuntimeError("No nn.Linear layers found to adapt.")
    depths = [d for d in (_block_index(name) for name, *_ in targets) if d is not None]
    max_depth = max(depths) if depths else 0
    for name, parent, child_name, child in targets:
        layer_rank = _layerwise_rank(name, rank_schedule, max_depth) if method == "layerwise" else rank
        setattr(parent, child_name, _make_adapter(child, method, layer_rank, alpha, dropout))
    return [name for name, *_ in targets]


def adapter_param_groups(model: nn.Module, lr: float, weight_decay: float, method: str, loraplus_ratio: float = 2.0) -> list[dict]:
    """Optimizer groups over trainable parameters; LoRA+ gets a faster ``B``."""
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Model has no trainable parameters.")
    if canonical_adapter_name(method) != "loraplus":
        return [{"params": [p for _, p in trainable], "lr": lr, "weight_decay": weight_decay}]
    b_params = [p for name, p in trainable if ".lora_b" in name]
    other = [p for name, p in trainable if ".lora_b" not in name]
    return [
        {"params": other, "lr": lr, "weight_decay": weight_decay},
        {"params": b_params, "lr": lr * loraplus_ratio, "weight_decay": weight_decay},
    ]
