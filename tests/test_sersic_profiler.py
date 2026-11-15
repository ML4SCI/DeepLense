import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

def test_sersic_law_gradient_stability_at_origin():
    """Verify sersic_law calculates non-NaN gradients across origin coordinates."""
    from DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar.GAN.models import SersicProfiler

    device = torch.device('cpu')
    profiler = SersicProfiler(resolution=0.04, device=device, discriminators=[])

    x = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    y = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    x_center = torch.tensor([0.0])
    y_center = torch.tensor([0.0])

    for n_sersic in [1.0, 2.0, 4.0]:
        I = profiler.sersic_law(x, y, x_center, y_center, R_sersic=1.0)
        loss = I.sum()
        loss.backward()

        assert not torch.isnan(x.grad).any(), f"NaN gradient detected for n={n_sersic} at x"
        assert not torch.isnan(y.grad).any(), f"NaN gradient detected for n={n_sersic} at y"
        assert torch.isfinite(x.grad).all(), f"Infinite gradient detected for n={n_sersic} at x"
        assert torch.isfinite(y.grad).all(), f"Infinite gradient detected for n={n_sersic} at y"

        x.grad.zero_()
        y.grad.zero_()

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
    test_sersic_profiler_models_cross_verification()
    print("test_sersic_profiler_models_cross_verification PASSED")
    test_sersic_normalization_zero_background_stability()
    print("test_sersic_normalization_zero_background_stability PASSED")
    print("All tests passed successfully!")
