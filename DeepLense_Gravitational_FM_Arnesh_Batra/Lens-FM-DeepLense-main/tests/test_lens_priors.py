"""Numerical checks of the claims in the paper's appendix ("Why the patch
permutation is exact"), run against the training code itself."""

import itertools

import pytest
import torch

from lens_lejepa.ssl import D4_SIZE, arc_weights, d4_patch_permutation, d4_token_loss, d4_transform, ring_descriptor, ring_masks

GRID, PATCH = 10, 16
SIZE = GRID * PATCH


def patchify(x: torch.Tensor) -> torch.Tensor:
    """[B, C, N, N] -> [B, G*G, C, P, P] in row-major patch order."""
    b, c, _, _ = x.shape
    return x.reshape(b, c, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, GRID * GRID, c, PATCH, PATCH)


@pytest.mark.parametrize("g", range(D4_SIZE))
def test_d4_moves_whole_patches(g):
    """E(g x)_p == g E(x)_{pi_g(p)} pixel for pixel (Proposition 1)."""
    x = torch.randn(2, 1, SIZE, SIZE)
    lhs = patchify(d4_transform(x, g))
    rhs = d4_transform(patchify(x)[:, d4_patch_permutation(GRID, g)], g)
    assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("g", range(D4_SIZE))
def test_permutation_is_bijection(g):
    perm = d4_patch_permutation(GRID, g)
    assert torch.equal(perm.sort().values, torch.arange(GRID * GRID))


def test_identity_and_group_homomorphism():
    """pi_e is the identity and pi_a o pi_b == pi_{ab} for all 64 pairs."""
    assert torch.equal(d4_patch_permutation(GRID, 0), torch.arange(GRID * GRID))
    x = torch.randn(1, 1, SIZE, SIZE)
    for a, b in itertools.product(range(D4_SIZE), repeat=2):
        composed = d4_transform(d4_transform(x, b), a)
        ab = next(k for k in range(D4_SIZE) if torch.equal(d4_transform(x, k), composed))
        pa, pb = d4_patch_permutation(GRID, a), d4_patch_permutation(GRID, b)
        # Patch p of (a(b x)) comes from patch pa[p] of (b x), i.e. patch pb[pa[p]] of x.
        assert torch.equal(pb[pa], d4_patch_permutation(GRID, ab))


@pytest.mark.parametrize("g", range(D4_SIZE))
def test_rings_are_d4_invariant(g):
    """pi_g(R_k) == R_k for every ring (Proposition 2)."""
    masks = ring_masks(GRID, 3)
    assert torch.equal(masks[:, d4_patch_permutation(GRID, g)], masks)


def test_rings_partition_the_grid():
    masks = ring_masks(GRID, 3)
    assert torch.equal(masks.sum(0), torch.ones(GRID * GRID))
    assert (masks.sum(1) > 0).all()


def test_arc_weights_have_unit_mean():
    weights = arc_weights(torch.rand(4, 1, SIZE, SIZE), PATCH, strength=0.75)
    assert weights.shape == (4, GRID * GRID)
    assert torch.allclose(weights.mean(1), torch.ones(4), atol=1e-5)
    assert torch.allclose(arc_weights(torch.rand(2, 1, SIZE, SIZE), PATCH, 0.0), torch.ones(2, GRID * GRID))


@pytest.mark.parametrize("g", range(D4_SIZE))
def test_losses_vanish_for_an_equivariant_encoder(g):
    """A patch-wise encoder is exactly D4-equivariant up to the in-patch action,
    so with a D4-invariant patch feature both priors must be zero."""
    def encoder(x):  # per-patch sorted pixels: invariant to any in-patch permutation
        return patchify(x).flatten(2).sort(dim=-1).values

    x = torch.rand(3, 1, SIZE, SIZE)
    xg = d4_transform(x, g)
    za, zb = encoder(x), encoder(xg)
    assert d4_token_loss(za, zb, d4_patch_permutation(GRID, g)).item() < 1e-10
    ra, rb = ring_descriptor(za, x, PATCH, 3, 0.75), ring_descriptor(zb, xg, PATCH, 3, 0.75)
    assert torch.allclose(ra, rb, atol=1e-5)
