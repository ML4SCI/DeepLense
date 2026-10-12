import math

import pytest
import torch

from lens_lejepa.models import (
    ADAPTERS,
    EncoderClassifier,
    PatchSuperResolution,
    build_vit,
    count_parameters,
    inject_adapters,
)
from lens_lejepa.models.adapters import LoRALinear


def test_vit_small_shapes_and_size():
    vit = build_vit("vit_small", 160, 16)
    tokens = vit(torch.randn(2, 1, 160, 160))
    assert tokens.shape == (2, 100, 384)
    assert count_parameters(vit) == 21_431_424  # ViT-S/16 as reported in the paper
    assert not vit.pos_embed.requires_grad


def test_rslora_on_vit_small_matches_paper_count():
    vit = build_vit("vit_small", 160, 16)
    targets = inject_adapters(vit, "rslora", rank=32, alpha=64)
    model = EncoderClassifier(vit)
    assert len(targets) == 48  # qkv, proj, fc1, fc2 in each of 12 blocks
    assert count_parameters(model, trainable_only=True) == 2_360_451


@pytest.mark.parametrize("method", [m for m in ADAPTERS if m != "layerwise"] + ["layerwise"])
def test_adapters_start_as_the_frozen_model(method):
    torch.manual_seed(0)
    vit = build_vit("vit_tiny", 32, 16).eval()
    x = torch.randn(2, 1, 32, 32)
    before = vit(x)
    inject_adapters(vit, method, rank=4, alpha=8, rank_schedule=[2, 4] if method == "layerwise" else None)
    assert torch.allclose(vit(x), before, atol=1e-5)
    trainable = [name for name, p in vit.named_parameters() if p.requires_grad]
    assert trainable and all(any(k in name for k in ("lora_", "magnitude", "rank_logits")) for name in trainable)


def test_rslora_scaling():
    layer = torch.nn.Linear(8, 8)
    assert LoRALinear(layer, rank=16, alpha=64).scaling == pytest.approx(4.0)
    assert LoRALinear(layer, rank=16, alpha=64, rank_stabilised=True).scaling == pytest.approx(64 / math.sqrt(16))


def test_patch_super_resolution_output():
    vit = build_vit("vit_tiny", 32, 16)
    out = PatchSuperResolution(vit, 32, 16, output_size=64)(torch.randn(3, 1, 32, 32))
    assert out.shape == (3, 1, 64, 64)
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("name", ["vit", "vitsd"])
def test_supervised_vits(name):
    from lens_lejepa.models import SupervisedViT

    model = SupervisedViT(32, 16, 3, dim=64, heads=2, mlp_dim=64, shifted_patches=name == "vitsd")
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 3)


def test_lensiformer_and_lenspinn():
    from lens_lejepa.models import Lensiformer, LensPINN, LensPINNRegressor

    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    probabilities, source = Lensiformer(32, 16, embed_dim=32, num_heads=2).forward_with_source(images, distortion)
    assert probabilities.shape == (2, 3) and torch.allclose(probabilities.sum(1), torch.ones(2), atol=1e-5)
    assert source.shape == (2, 1, 32, 32)
    assert LensPINN()(images, distortion).shape == (2, 3)
    assert LensPINNRegressor()(images, distortion).shape == (2,)


def test_lens_jepa_encoder_and_masks():
    from lens_lejepa.models import BlockMaskCollator, JEPAPredictor, LensJEPAEncoder, MaskConfig

    encoder = LensJEPAEncoder(32, 16)
    assert count_parameters(LensJEPAEncoder(160, 16)) == 2_454_018  # the 2.45M reference backbone
    collator = BlockMaskCollator(MaskConfig(grid_size=2, npred=1, pred_scale=(0.25, 0.25)))
    images, masks_enc, masks_pred = collator([{"image": torch.randn(1, 32, 32)} for _ in range(3)])
    assert not set(masks_enc[0][0].tolist()) & set(masks_pred[0][0].tolist())  # no target leaks into context
    context = encoder(images, masks_enc)
    predicted = JEPAPredictor(4, encoder.embed_dim, 32, depth=1, num_heads=2)(context, masks_enc, masks_pred)
    assert predicted.shape == (3, masks_pred[0].shape[1], encoder.embed_dim)
