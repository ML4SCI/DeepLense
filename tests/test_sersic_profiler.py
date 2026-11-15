import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

def test_sersic_law_gradient_stability_at_origin():
    """Verify sersic_law calculates non-NaN gradients across origin coordinates for multiple Sérsic indices."""
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.GAN.models import SersicProfiler

    device = torch.device('cpu')
    profiler = SersicProfiler(resolution=0.04, device=device, discriminators=[])

    x = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    y = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    x_center = torch.tensor([0.0])
    y_center = torch.tensor([0.0])

    for n_sersic in [1.0, 2.0, 4.0]:
        profiler.sersic_args = [1.0, n_sersic, 1.0]
        I = profiler.sersic_law(x, y, x_center, y_center, R_sersic=1.0)
        loss = I.sum()
        loss.backward()

        assert not torch.isnan(x.grad).any(), f"NaN gradient detected for n={n_sersic} at x"
        assert not torch.isnan(y.grad).any(), f"NaN gradient detected for n={n_sersic} at y"
        assert torch.isfinite(x.grad).all(), f"Infinite gradient detected for n={n_sersic} at x"
        assert torch.isfinite(y.grad).all(), f"Infinite gradient detected for n={n_sersic} at y"

        x.grad.zero_()
        y.grad.zero_()

def test_sersic_law_half_precision_stability():
    """Verify sersic_law gradient stability under float16 mixed-precision emulation."""
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.GAN.models import SersicProfiler

    device = torch.device('cpu')
    profiler = SersicProfiler(resolution=0.04, device=device, discriminators=[])

    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float16, requires_grad=True)
    y = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float16, requires_grad=True)
    x_center = torch.tensor([0.0], dtype=torch.float16)
    y_center = torch.tensor([0.0], dtype=torch.float16)

    for n_sersic in [1.0, 2.0, 4.0]:
        profiler.sersic_args = [1.0, n_sersic, 1.0]
        I = profiler.sersic_law(x, y, x_center, y_center, R_sersic=1.0)
        loss = I.sum()
        loss.backward()

        assert not torch.isnan(x.grad).any(), f"FP16 NaN gradient detected for n={n_sersic} at x"
        assert not torch.isnan(y.grad).any(), f"FP16 NaN gradient detected for n={n_sersic} at y"

        x.grad.zero_()
        y.grad.zero_()

def test_create_lensing_gradient_flow_and_strides():
    """Verify create_lensing backward flow through flipped tensors and grid sampling."""
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.GAN.models import SersicProfiler

    device = torch.device('cpu')
    profiler = SersicProfiler(resolution=0.04, device=device, discriminators=[])

    B, C, H, W = 2, 1, 16, 16
    LR = torch.rand(B, C, H, W, requires_grad=True)
    alpha = torch.zeros(B, 2, H * 2, W * 2, requires_grad=True)
    alpha_interp = torch.zeros(B, 2, H, W, requires_grad=True)

    source, S_lens, I_lens_LR, I_lens = profiler.create_lensing(
        LR, alpha_interp, alpha, resolution=0.04, resolution_=0.02, magnification=2, R_sersic=1.0
    )

    loss = I_lens.sum() + I_lens_LR.sum() + source.sum()
    loss.backward()

    assert not torch.isnan(alpha.grad).any(), "NaN gradient in alpha"
    assert torch.isfinite(alpha.grad).all(), "Infinite gradient in alpha"

def test_sersic_profiler_models_cross_verification():
    """Verify numerical stability across all 6 model implementations."""
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.GAN.models import SersicProfiler as GAN_Profiler
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.RCAN.models import SersicProfiler as RCAN_Profiler
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.DC.models import SersicProfiler as DC_Profiler
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.sparse.models import SersicProfiler as Sparse_Profiler
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.fGAN.models import SersicProfiler as FGAN_Profiler
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.vanilla.models import SersicProfiler as Vanilla_Profiler

    device = torch.device('cpu')
    instances = [
        GAN_Profiler(resolution=0.04, device=device, discriminators=[]),
        RCAN_Profiler(resolution=0.04, device=device),
        DC_Profiler(resolution=0.04, device=device),
        Sparse_Profiler(resolution=0.04, device=device),
        FGAN_Profiler(resolution=0.04, device=device, discriminators=[], feature_extractor=None),
        Vanilla_Profiler(resolution=0.04, device=device),
    ]

    for p in instances:
        x = torch.tensor([0.0], requires_grad=True)
        y = torch.tensor([0.0], requires_grad=True)
        I = p.sersic_law(x, y, torch.tensor([0.0]), torch.tensor([0.0]), R_sersic=1.0)
        I.backward()
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(y.grad).any()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(y.grad).all()

def test_sersic_normalization_zero_background_stability():
    """Verify create_sersic handles constant zero/uniform backgrounds without NaN division."""
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.GAN.models import SersicProfiler

    device = torch.device('cpu')
    profiler = SersicProfiler(resolution=0.04, device=device, discriminators=[])
    # Uniform zero image patch
    zero_patch = torch.zeros(1, 1, 32, 32)
    s_profile = profiler.create_sersic(zero_patch, resolution=0.04, R_sersic=1.0)
    assert not torch.isnan(s_profile).any()
    assert torch.isfinite(s_profile).all()

if __name__ == '__main__':
    test_sersic_law_gradient_stability_at_origin()
    print("test_sersic_law_gradient_stability_at_origin PASSED")
    test_sersic_law_half_precision_stability()
    print("test_sersic_law_half_precision_stability PASSED")
    test_create_lensing_gradient_flow_and_strides()
    print("test_create_lensing_gradient_flow_and_strides PASSED")
    test_sersic_profiler_models_cross_verification()
    print("test_sersic_profiler_models_cross_verification PASSED")
    test_sersic_normalization_zero_background_stability()
    print("test_sersic_normalization_zero_background_stability PASSED")
    print("All tests passed successfully!")
