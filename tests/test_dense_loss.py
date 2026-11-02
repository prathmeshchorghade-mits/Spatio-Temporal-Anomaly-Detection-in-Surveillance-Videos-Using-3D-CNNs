"""Smoke test: dense_mil_loss correctness checks.

Verifies:
    - Loss is finite for random inputs.
    - Gradient flows back through the first conv layer.
    - Loss approaches zero when anomalous top-k is far above normal top-k.
    - Proximity-aware variant also produces finite loss with gradients.
    - All returned loss component keys are present.
"""

import pytest
import torch
from training.losses.dense_mil import dense_mil_loss
from training.models.dense_c3d import DenseC3D, POOL_CONFIGS


B, S, T_, H_, W_ = 2, 4, 4, 14, 14   # batch, segments, T', H', W' for g14


def _make_vox(requires_grad: bool = True):
    return torch.randn(B, S, T_, H_, W_, requires_grad=requires_grad)


def test_loss_is_finite() -> None:
    """Loss and all components must be finite for random inputs."""
    vox_a = _make_vox()
    vox_n = _make_vox()
    result = dense_mil_loss(vox_a, vox_n)
    for key, val in result.items():
        assert torch.isfinite(val), f"Component '{key}' is not finite: {val}"


def test_gradient_reaches_first_conv() -> None:
    """Gradient must propagate back to the first conv weight via the model."""
    model = DenseC3D(pools=POOL_CONFIGS["g14"], variant_name="g14")
    model.train()

    x_a = torch.randn(B * S, 3, 16, 112, 112)
    x_n = torch.randn(B * S, 3, 16, 112, 112)

    vox_a = model(x_a).view(B, S, *model(x_a).shape[1:])  # (B, S, T', H', W')
    # Recompute cleanly to keep graph
    vox_a = model(x_a).view(B, S, 4, 14, 14)
    vox_n = model(x_n).view(B, S, 4, 14, 14)

    loss_dict = dense_mil_loss(vox_a, vox_n)
    loss_dict["total"].backward()

    first_conv = list(model.features[0].parameters())[0]
    assert first_conv.grad is not None, "No gradient at first conv weight"
    assert not torch.isnan(first_conv.grad).any(), "NaN gradient at first conv"


def test_loss_near_zero_when_anomalous_wins() -> None:
    """Rank loss ≈ 0 when anomalous top-k >> normal top-k (margin satisfied)."""
    # Anomalous maps: very high logits; normal maps: very low logits
    vox_a = torch.full((B, S, T_, H_, W_), fill_value=10.0)
    vox_n = torch.full((B, S, T_, H_, W_), fill_value=-10.0)
    result = dense_mil_loss(vox_a, vox_n, lam_sp=0.0, lam_tv=0.0, lam_norm=0.0)
    assert result["rank"].item() < 1e-4, (
        f"Rank loss should be ~0 when anomalous >> normal, got {result['rank'].item()}"
    )


def test_all_loss_keys_present() -> None:
    """Return dict must contain all expected keys."""
    expected_keys = {"total", "rank", "sparse", "tv_space", "tv_time", "norm"}
    vox_a = _make_vox()
    vox_n = _make_vox()
    result = dense_mil_loss(vox_a, vox_n)
    assert expected_keys == set(result.keys()), (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


def test_proximity_aware_variant() -> None:
    """Proximity-aware top-k variant produces finite loss with gradients."""
    vox_a = _make_vox()
    vox_n = _make_vox()
    result = dense_mil_loss(vox_a, vox_n, proximity_window=(2, 3, 3))
    assert torch.isfinite(result["total"]), "Proximity variant: loss not finite"
    result["total"].backward()
    assert vox_a.grad is not None, "Proximity variant: no gradient on vox_a"
