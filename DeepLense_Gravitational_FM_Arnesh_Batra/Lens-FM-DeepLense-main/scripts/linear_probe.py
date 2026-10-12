"""Frozen linear probe: how much class information a frozen encoder exposes.

Pooled features are extracted once per encoder and dataset, then a logistic
regression is fitted for every label budget on the same splits as the adapter
runs. The regularisation strength is chosen on the validation split only.

    python scripts/linear_probe.py --encoder vit_small \\
        --checkpoint runs/pretrain/lens_lejepa/lens_lejepa_model_i_seed42/last_pretrain.pt \\
        --data-root ../datasets --output probe_lens_lejepa.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lens_lejepa.data import CLASSES, LensClassificationDataset, few_shot_split, stratified_split
from lens_lejepa.metrics import classification_metrics
from lens_lejepa.models import load_pretrained_encoder


@torch.no_grad()
def extract(encoder, root: str, image_size: int, device: torch.device, workers: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = LensClassificationDataset(root, image_size)
    loader = DataLoader(dataset, batch_size=256, num_workers=workers)
    features = [encoder(batch["image"].to(device)).mean(dim=1).float().cpu() for batch in loader]
    return torch.cat(features).numpy(), dataset.labels


def fit(train_x, train_y, val_x, val_y):
    scaler = StandardScaler().fit(train_x)
    best = None
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(train_x), train_y)
        score = (model.predict(scaler.transform(val_x)) == val_y).mean()
        if best is None or score > best[0]:
            best = (score, c, model)
    return scaler, best[2], best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="vit_small")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="../datasets")
    parser.add_argument("--datasets", nargs="+", default=["Model_II", "Model_III"])
    parser.add_argument("--shots", nargs="+", default=["100", "500", "1000", "5000", "full"])
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_pretrained_encoder(args.encoder, args.image_size, 16, args.checkpoint).to(device).eval()
    results = []
    for name in args.datasets:
        train_x, train_y = extract(encoder, f"{args.data_root}/{name}", args.image_size, device, args.workers)
        test_x, test_y = extract(encoder, f"{args.data_root}/{name}_test", args.image_size, device, args.workers)
        for shots in (args.shots if name == "Model_II" else ["full"]):
            if shots == "full":
                tr, va = stratified_split(train_y, 0.8, args.seed)
            else:
                tr, va = few_shot_split(train_y, int(shots), 0.25, args.seed)
            scaler, model, c = fit(train_x[tr], train_y[tr], train_x[va], train_y[va])
            metrics = classification_metrics(test_y, model.predict_proba(scaler.transform(test_x)), CLASSES)
            results.append({"dataset": f"{name}_test", "shots": shots, "C": c, **metrics})
            print(name, shots, {k: round(metrics[k], 4) for k in ("accuracy", "mean_auroc", "macro_f1")}, flush=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
