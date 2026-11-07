"""The results table only reports an adaptation for runs that actually adapt an encoder."""

import importlib.util
import json
from pathlib import Path

import pytest

from lens_lejepa.config import FinetuneConfig, config_to_dict

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_results.py"


def _load_collect():
    spec = importlib.util.spec_from_file_location("collect_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect


@pytest.mark.parametrize(
    ("backbone", "adaptation", "expected"),
    [
        ("vit", "rslora", "-"),  # supervised baseline trained from scratch
        ("vitsd", "rslora", "-"),
        ("resnet18", "rslora", "-"),
        ("lens_jepa", "lora", "lora"),  # pretrained encoder with LoRA
        ("vit_small", "rslora", "rslora"),
    ],
)
def test_adaptation_column(tmp_path, backbone, adaptation, expected):
    run = tmp_path / "classification" / "group" / "Model_II_full_seed42"
    run.mkdir(parents=True)
    config = config_to_dict(FinetuneConfig(backbone=backbone, adaptation=adaptation))
    (run / "config.json").write_text(json.dumps(config))
    (run / "summary.json").write_text(json.dumps({"test": {"test_root": "../datasets/Model_II_test", "accuracy": 0.9}}))

    (row,) = _load_collect()(tmp_path)

    assert row["backbone"] == backbone
    assert row["adaptation"] == expected
