# Adding a downstream task

A task is one class. The training loop (`lens_lejepa/engine/finetune.py`) never needs to change.

## 1. Write the dataset

Datasets return a **dictionary** per item. Put it in `lens_lejepa/data/datasets.py` (or next to your task):

```python
class EinsteinRadiusDataset(Dataset):
    def __init__(self, root, image_size):
        self.paths, self.labels = list_class_files(root)
        self.image_size = image_size
        self.targets = ...  # one float per file

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = load_lens_image(self.paths[index])
        return {
            "image": to_model_tensor(image, self.image_size, standardize=True),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }
```

## 2. Write the task

```python
# lens_lejepa/tasks/einstein_radius.py
from .base import Task
from . import register

@register
class EinsteinRadiusTask(Task):
    """Predict the Einstein radius in pixels."""

    name = "einstein_radius"
    monitor = "mae"            # validation metric used to pick the checkpoint
    higher_is_better = False

    def build_datasets(self):
        c = self.config
        dataset = EinsteinRadiusDataset(f"{c.data_root}/{c.train_dataset}", c.image_size)
        train_idx, val_idx = quantile_split(dataset.targets, c.train_fraction, c.seed)
        return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())

    def build_test_dataset(self, root):
        return EinsteinRadiusDataset(root, self.config.image_size)

    def build_head(self, encoder):
        return EncoderRegressor(encoder)          # any nn.Module that wraps the encoder

    def loss(self, output, batch):
        return F.smooth_l1_loss(output.float(), batch["target"])

    def collect(self, output, batch):             # per-batch arrays for the metrics
        return {"pred": output.float().cpu().numpy(), "true": batch["target"].cpu().numpy()}

    def metrics(self, outputs):
        return {"mae": float(np.abs(outputs["pred"] - outputs["true"]).mean())}
```

Optional hooks:

| Method | Default | Override when |
|---|---|---|
| `model_input(batch)` | `batch["image"]` | the model reads a different key (see `super_resolution.py`) |
| `build_baseline(name)` | error | you want `backbone=<name>` to train a from-scratch model |
| `state_dict()` / `load_state_dict()` | nothing | the task has state that evaluation needs, e.g. a target normaliser |

## 3. Register it

`@register` adds the class to `TASKS`. Add `from . import einstein_radius  # noqa: F401` at the **end** of
`lens_lejepa/tasks/__init__.py` (after `register` is defined) so the decorator runs, then:

```bash
python -m lens_lejepa tasks
python -m lens_lejepa finetune --config configs/downstream/regression.yaml --set task=einstein_radius
```

Everything else (adapters, bf16 training, validation schedule, best-checkpoint selection, held-out evaluation,
`summary.json`) is inherited.

## 4. Test it

Add a case to `tests/test_pipeline.py::test_finetune_and_evaluate`. The `fake_root` fixture in `tests/conftest.py`
builds a tiny dataset on disk, so the test runs on CPU in seconds.
