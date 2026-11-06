import sys
import types
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock timm if missing in environment
try:
    import timm.models.resnet  # noqa: F401
except ImportError:
    timm_mod = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_resnet = types.ModuleType("timm.models.resnet")
    timm_resnet.resnet50 = lambda *args, **kwargs: nn.Identity()
    timm_resnet.resnet18 = lambda *args, **kwargs: nn.Identity()
    sys.modules["timm"] = timm_mod
    sys.modules["timm.models"] = timm_models
    sys.modules["timm.models.resnet"] = timm_resnet

# Mock models.backbone for standalone classifier import if needed
if "models.backbone" not in sys.modules:
    mb = types.ModuleType("models.backbone")
    mb.Backbone = None
    sys.modules["models.backbone"] = mb

from selfsupervised.models.classifier import Classifier
from selfsupervised.models.patch_embedding_layer import (
    patch_embedding,
    ConvPatchEmbed,
    channel_vit_patch_embedding,
)
from selfsupervised.models.backbone import Backbone


class DummyBackbone(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        return torch.randn(x.shape[0], self.embed_dim)


def test_classifier_linear_probe():
    """Verify Classifier runs in linear_probe mode with torch.no_grad()."""
    backbone = DummyBackbone(embed_dim=32)
    clf = Classifier(mode="linear_probe", backbone=backbone)
    x = torch.randn(2, 3, 64, 64)
    out = clf(x)
    assert out.shape == (2, 2)
    assert clf.freeze_backbone is True


def test_classifier_finetune():
    """Verify Classifier runs in finetune mode."""
    backbone = DummyBackbone(embed_dim=32)
    clf = Classifier(mode="finetune", backbone=backbone)
    x = torch.randn(2, 3, 64, 64)
    out = clf(x)
    assert out.shape == (2, 2)
    assert clf.freeze_backbone is False


def test_patch_embedding_with_padding():
    """Verify patch_embedding correctly calculates padding without NameError."""
    # Test non-divisible size with padding=True
    pe = patch_embedding(
        image_size=(70, 70),
        patch_size=(16, 16),
        input_channels=3,
        patch_embedding_dim=64,
        padding=True,
    )
    x = torch.randn(2, 3, 70, 70)
    out = pe(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 64


def test_conv_patch_embed_with_padding():
    """Verify ConvPatchEmbed correctly calculates padding without NameError."""
    cpe = ConvPatchEmbed(
        image_size=(70, 70),
        patch_size=16,
        input_channels=3,
        patch_embedding_dim=64,
        padding=True,
    )
    x = torch.randn(2, 3, 70, 70)
    out, grid_size = cpe(x)
    assert out.shape[0] == 2
    assert isinstance(grid_size, tuple)


def test_channel_vit_patch_embedding_with_padding():
    """Verify channel_vit_patch_embedding calculates padding without NameError."""
    cvpe = channel_vit_patch_embedding(
        image_size=(70, 70),
        patch_size=(16, 16),
        input_channels=3,
        patch_embedding_dim=64,
        padding=True,
    )
    x = torch.randn(2, 3, 70, 70)
    out = cvpe(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 64


def test_backbone_unknown_arch_exits():
    """Verify unknown architecture in Backbone exits cleanly via sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        Backbone(image_size=64, input_channels=3, arch="unknown_backbone")
    assert exc_info.value.code == 1
