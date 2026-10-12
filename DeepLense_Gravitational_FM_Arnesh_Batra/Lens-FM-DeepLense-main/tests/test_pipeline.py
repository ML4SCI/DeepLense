"""End-to-end CPU runs of every command on a tiny synthetic dataset."""

import json

import pytest
import torch

from lens_lejepa.config import FinetuneConfig, PretrainConfig, load_config
from lens_lejepa.engine import evaluate, finetune, pretrain
from lens_lejepa.ssl import LensLeJEPAObjective, ObjectiveConfig, SIGReg
from lens_lejepa.models import LeJEPAProjector, build_vit


def test_config_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("method: lejepa\nobjective:\n  d4_weight: 0.3\n")
    config = load_config(PretrainConfig, path, ["epochs=3", "objective.rings=4"])
    assert config.epochs == 3 and config.objective.rings == 4 and config.objective.d4_weight == 0.3
    assert config.objective.use_lens_priors is False  # the control never uses the priors
    assert load_config(FinetuneConfig, None, ["shots=null", "rank_schedule=[8,16]"]).rank_schedule == [8, 16]
    with pytest.raises(ValueError):
        load_config(FinetuneConfig, None, ["not_a_field=1"])


def test_sigreg_is_reproducible_and_detects_non_gaussian():
    torch.manual_seed(0)
    gaussian = torch.randn(512, 32)
    a, b = SIGReg(64), SIGReg(64)
    assert torch.equal(a(gaussian), b(gaussian))
    assert SIGReg(64)(torch.zeros(512, 32) + 3.0) > 10 * SIGReg(64)(gaussian)


@pytest.mark.parametrize("use_priors", [True, False])
def test_objective_backpropagates(use_priors):
    vit = build_vit("vit_tiny", 32, 16)
    projector = LeJEPAProjector(vit.embed_dim, 16)
    objective = LensLeJEPAObjective(ObjectiveConfig(use_lens_priors=use_priors, sigreg_slices=16), patch_size=16)
    loss, parts = objective(vit, projector, torch.rand(8, 1, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)
    assert (parts["d4"] > 0) == use_priors
    assert vit.patch_embed.proj.weight.grad is not None


def _pretrain_config(root, out, method="lens_lejepa", encoder="vit_tiny"):
    return PretrainConfig(
        method=method, data_root=str(root.parent), dataset=root.name, image_size=32, encoder=encoder,
        projector_dim=16, epochs=2, batch_size=12, warmup_epochs=1, workers=0, device="cpu",
        output_dir=str(out), run_name="t", collapse_patience=100,
        objective={"sigreg_slices": 16}, masked={"predictor_dim": 48, "predictor_depth": 1, "npred": 1, "pred_scale": [0.25, 0.25]},
    )


@pytest.mark.parametrize("method,encoder", [("ijepa", "vit_tiny"), ("lens_jepa", "lens_jepa"), ("lens_jepa_sym", "lens_jepa"), ("lens_jepa_focus", "lens_jepa")])
def test_masked_reference_pretraining(fake_root, tmp_path, method, encoder):
    run = pretrain(_pretrain_config(fake_root, tmp_path, method, encoder))
    state = torch.load(run / "last_pretrain.pt", weights_only=False)
    assert state["epoch"] == 2 and "teacher" in state
    config = _pretrain_config(fake_root, tmp_path, method, encoder)
    config.epochs = 3
    assert torch.load(pretrain(config, resume=run / "last_pretrain.pt") / "last_pretrain.pt", weights_only=False)["epoch"] == 3


def test_pretrain_and_resume(fake_root, tmp_path):
    run = pretrain(_pretrain_config(fake_root, tmp_path))
    state = torch.load(run / "last_pretrain.pt", weights_only=False)
    assert state["epoch"] == 2
    config = _pretrain_config(fake_root, tmp_path)
    config.epochs = 3
    run = pretrain(config, resume=run / "last_pretrain.pt")
    assert torch.load(run / "last_pretrain.pt", weights_only=False)["epoch"] == 3
    assert (run / "pretrain_metrics.csv").read_text().count("\n") == 4


@pytest.mark.parametrize(
    "task,backbone,adaptation",
    [
        ("classification", "vit_tiny", "rslora"),
        ("classification", "vit_tiny", "full"),
        ("classification", "resnet18", "rslora"),
        ("classification", "vit", "rslora"),
        ("classification", "vitsd", "rslora"),
        ("classification", "lensiformer", "rslora"),
        ("classification", "lenspinn", "rslora"),
        ("regression", "lenspinn", "rslora"),
        ("classification", "lens_jepa", "lora"),
        ("regression", "vit_tiny", "loraplus"),
        ("super_resolution", "vit_tiny", "lora"),
        ("super_resolution", "rcan", "lora"),
        ("super_resolution", "edsr", "lora"),
        ("super_resolution", "srresnet", "lora"),
        ("super_resolution", "fsrcnn", "lora"),
        ("super_resolution", "rdn", "lora"),
        ("super_resolution", "bicubic", "lora"),
    ],
)
def test_finetune_and_evaluate(fake_root, tmp_path, task, backbone, adaptation):
    from lens_lejepa.models import is_encoder

    method = "lens_jepa" if backbone == "lens_jepa" else "lens_lejepa"
    ssl_run = pretrain(_pretrain_config(fake_root, tmp_path / "ssl", method, backbone if backbone == "lens_jepa" else "vit_tiny"))
    config = FinetuneConfig(
        task=task, data_root=str(fake_root.parent), train_dataset=fake_root.name, test_dataset=fake_root.name,
        image_size=32, backbone=backbone, adaptation=adaptation, rank=4, alpha=8,
        pretrained=str(ssl_run / "last_pretrain.pt") if is_encoder(backbone) else None,
        epochs=2, batch_size=8, val_interval=1, workers=0, device="cpu", output_dir=str(tmp_path), run_name="t",
    )
    run = finetune(config)
    summary = json.loads((run / "summary.json").read_text())
    reloaded = evaluate(run / "best.pt", str(fake_root), "cpu", workers=0)
    for key, value in summary["test"].items():
        if isinstance(value, float):
            assert reloaded[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key
