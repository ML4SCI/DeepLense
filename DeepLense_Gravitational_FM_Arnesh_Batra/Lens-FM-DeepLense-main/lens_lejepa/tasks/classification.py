"""Dark matter substructure classification: axion vs. CDM vs. no substructure."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset

from ..data import CLASSES, LensClassificationDataset, few_shot_split, stratified_split
from ..metrics import classification_metrics
from ..models import EncoderClassifier, Lensiformer, LensPINN, SupervisedViT, build_resnet18
from .base import Task


class ClassificationTask(Task):
    """Three-class substructure classification (axion / CDM / no substructure)."""

    name = "classification"
    monitor = "accuracy"
    higher_is_better = True

    def build_datasets(self):
        c = self.config
        root = f"{c.data_root}/{c.train_dataset}"
        dataset = LensClassificationDataset(root, c.image_size, cache=c.cache, distortion=self.needs_distortion)
        train_dataset = dataset
        if c.train_jitter > 0:  # shifted copies for training only; validation stays centred
            train_dataset = LensClassificationDataset(root, c.image_size, cache=c.cache, distortion=self.needs_distortion, jitter=c.train_jitter)
        if c.shots is None:
            train_idx, val_idx = stratified_split(dataset.labels, c.train_fraction, c.seed)
        else:
            train_idx, val_idx = few_shot_split(dataset.labels, c.shots, c.val_fraction, c.seed)
        return Subset(train_dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())

    def build_test_dataset(self, root: str):
        return LensClassificationDataset(root, self.config.image_size, cache=self.config.cache, distortion=self.needs_distortion)

    def build_head(self, encoder: nn.Module) -> nn.Module:
        return EncoderClassifier(encoder, num_classes=len(CLASSES))

    def build_baseline(self, name: str) -> nn.Module:
        c, k = self.config, len(CLASSES)
        baselines = {
            "resnet18": lambda: build_resnet18(k),
            "vit": lambda: SupervisedViT(c.image_size, c.patch_size, k),
            "vitsd": lambda: SupervisedViT(c.image_size, c.patch_size, k, shifted_patches=True),
            "lensiformer": lambda: Lensiformer(c.image_size, c.patch_size, num_classes=k),
            "lenspinn": lambda: LensPINN(num_classes=k),
        }
        if name in baselines:
            return baselines[name]()
        return super().build_baseline(name)

    def loss(self, output, batch):
        return F.cross_entropy(output.float(), batch["label"], label_smoothing=self.config.label_smoothing)

    def collect(self, output, batch):
        return {
            "probabilities": torch.softmax(output.float(), dim=1).cpu().numpy(),
            "labels": batch["label"].cpu().numpy(),
        }

    def metrics(self, outputs: dict[str, np.ndarray]) -> dict:
        return classification_metrics(outputs["labels"], outputs["probabilities"], CLASSES)
