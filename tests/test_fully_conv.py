"""Smoke test: fully-convolutional property of DenseC3D.

The same weights must run on both 112×112 (training crop) and 112×144
(inference frame) inputs without shape errors.  The temporal output T'=4
must be preserved for both input sizes.
"""

import pytest
import torch
from training.models.dense_c3d import DenseC3D, POOL_CONFIGS


@pytest.mark.parametrize("variant", ["g7", "g14", "g28"])
def test_same_weights_different_spatial_sizes(variant: str) -> None:
    """One model instance produces valid outputs for both spatial sizes."""
    model = DenseC3D(pools=POOL_CONFIGS[variant], variant_name=variant)
    model.eval()

    sizes = [(112, 112), (112, 144), (128, 160)]
    for H, W in sizes:
        x = torch.randn(1, 3, 16, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.ndim == 4, f"[{variant}] Expected 4-D output, got {out.ndim}-D"
        assert out.shape[0] == 1, "Batch dim mismatch"
        assert out.shape[1] == 4, (
            f"[{variant}] T' must be 4 for H={H}, W={W}, got T'={out.shape[1]}"
        )
        assert torch.isfinite(out).all(), (
            f"[{variant}] Non-finite values for input size ({H}, {W})"
        )


@pytest.mark.parametrize("variant", ["g7", "g14", "g28"])
def test_weights_unchanged_after_inference(variant: str) -> None:
    """Inference on multiple sizes must not change model weights."""
    model = DenseC3D(pools=POOL_CONFIGS[variant], variant_name=variant)
    model.eval()

    # Capture initial weight snapshot
    params_before = {
        name: p.clone() for name, p in model.named_parameters()
    }

    for H, W in [(112, 112), (112, 144)]:
        x = torch.randn(1, 3, 16, H, W)
        with torch.no_grad():
            _ = model(x)

    for name, p in model.named_parameters():
        assert torch.equal(params_before[name], p), (
            f"[{variant}] Weight '{name}' changed during inference"
        )
