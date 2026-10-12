"""I-JEPA masking: token gathering helpers and the block mask sampler."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather the tokens listed in each ``[B, K]`` mask and stack masks along the batch."""
    return torch.cat([torch.gather(x, 1, m.unsqueeze(-1).expand(-1, -1, x.shape[-1])) for m in masks], dim=0)


def repeat_interleave_batch(x: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    """Repeat each of the ``len(x) // batch_size`` chunks of ``x`` ``repeat`` times."""
    chunks = len(x) // batch_size
    return torch.cat([torch.cat([x[i * batch_size : (i + 1) * batch_size]] * repeat, dim=0) for i in range(chunks)], dim=0)


@dataclass
class MaskConfig:
    grid_size: int = 10
    npred: int = 4                                   # target blocks per image
    pred_scale: tuple[float, float] = (0.15, 0.2)    # fraction of the grid per target block
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    nenc: int = 1                                    # context masks per image
    enc_scale: tuple[float, float] = (0.85, 1.0)     # fraction of the free patches kept as context
    min_keep: int = 1


class BlockMaskCollator:
    """Collate images and sample one I-JEPA mask set per batch.

    ``npred`` rectangular target blocks are placed uniformly at random. The
    context is a random subset of the patches that lie outside *every* target
    block, so no target patch ever leaks into the context. Returns
    ``(images, masks_enc, masks_pred)`` with masks as lists of ``[B, K]`` index tensors.
    """

    def __init__(self, config: MaskConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self.iteration = -1

    def _generator(self) -> torch.Generator:
        self.iteration += 1
        info = torch.utils.data.get_worker_info()
        worker = info.id if info is not None else 0
        return torch.Generator().manual_seed(self.seed + worker * 1_000_003 + self.iteration)

    def _block_size(self, g: torch.Generator) -> tuple[int, int]:
        c, n = self.config, self.config.grid_size
        scale = c.pred_scale[0] + torch.rand(1, generator=g).item() * (c.pred_scale[1] - c.pred_scale[0])
        ratio = c.aspect_ratio[0] + torch.rand(1, generator=g).item() * (c.aspect_ratio[1] - c.aspect_ratio[0])
        keep = int(n * n * scale)
        return min(n, max(1, round(sqrt(keep * ratio)))), min(n, max(1, round(sqrt(keep / ratio))))

    def _block(self, size: tuple[int, int], g: torch.Generator) -> torch.Tensor:
        n = self.config.grid_size
        h, w = size
        top = int(torch.randint(0, n - h + 1, (1,), generator=g))
        left = int(torch.randint(0, n - w + 1, (1,), generator=g))
        grid = torch.zeros(n, n, dtype=torch.bool)
        grid[top : top + h, left : left + w] = True
        return grid.flatten().nonzero().flatten()

    def __call__(self, batch):
        images = torch.utils.data.default_collate(list(batch))
        if isinstance(images, dict):
            images = images["image"]
        c, total = self.config, self.config.grid_size**2
        g = self._generator()
        size = self._block_size(g)
        for _ in range(128):
            targets = [self._block(size, g) for _ in range(c.npred)]
            covered = torch.unique(torch.cat(targets))
            if total - covered.numel() >= c.min_keep:
                break
        else:
            raise RuntimeError("Target blocks leave too little context; reduce pred_scale or npred.")
        free = torch.ones(total, dtype=torch.bool)
        free[covered] = False
        free = free.nonzero().flatten()
        contexts = []
        for _ in range(c.nenc):
            fraction = c.enc_scale[0] + torch.rand(1, generator=g).item() * (c.enc_scale[1] - c.enc_scale[0])
            count = max(c.min_keep, round(free.numel() * fraction))
            contexts.append(free[torch.randperm(free.numel(), generator=g)[:count]])
        b = images.shape[0]
        masks_pred = [m.unsqueeze(0).expand(b, -1).contiguous() for m in targets]
        masks_enc = [m.unsqueeze(0).expand(b, -1).contiguous() for m in contexts]
        return images, masks_enc, masks_pred
