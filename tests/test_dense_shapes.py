"""Smoke test: verify DenseC3D output shapes for all resolution variants.

Tests both 112×112 (training crop) and 112×144 (inference) input sizes.
Verifies:
    - Temporal output T' = 4 for all variants.
    - Spatial outputs: g7→7×7, g14→14×14, g28→28×28 for 112×112 input.
    - Same weights run on 112×144 without error (fully convolutional property).
"""

import pytest
import torch
from training.models.dense_c3d import DenseC3D, POOL_CONFIGS


BATCH = 2
CLIPS = 3   # clips per bag (S)
C, T = 3, 16

# Expected spatial output for 112×112 crop input
EXPECTED_SHAPES = {
    "g7":  (4, 7, 7),
    "g14": (4, 14, 14),
    "g28": (4, 28, 28),
}


@pytest.mark.parametrize("variant", ["g7", "g14", "g28"])
def test_output_shape_112x112(variant: str) -> None:
    """Voxel map shape is correct for 112×112 training crop input."""
    model = DenseC3D(pools=POOL_CONFIGS[variant], variant_name=variant)
    model.eval()
    x = torch.randn(BATCH, C, T, 112, 112)
    with torch.no_grad():
        out = model(x)
    expected_T, expected_H, expected_W = EXPECTED_SHAPES[variant]
    assert out.shape == (BATCH, expected_T, expected_H, expected_W), (
        f"[{variant}] Expected {(BATCH, expected_T, expected_H, expected_W)}, got {out.shape}"
    )


@pytest.mark.parametrize("variant", ["g7", "g14", "g28"])
def test_output_shape_112x144(variant: str) -> None:
    """Same weights run on 112×144 inference frame (fully convolutional)."""
    model = DenseC3D(pools=POOL_CONFIGS[variant], variant_name=variant)
    model.eval()
    x = torch.randn(BATCH, C, T, 112, 144)
    with torch.no_grad():
        out = model(x)
    # T' must still be 4; spatial dims can differ
    assert out.shape[0] == BATCH, "Batch dim mismatch"
    assert out.shape[1] == EXPECTED_SHAPES[variant][0], (
        f"[{variant}] Expected T'=4, got T'={out.shape[1]}"
    )
    # Must produce a valid tensor (not raise an error)
    assert not torch.isnan(out).any(), "NaN in output"


@pytest.mark.parametrize("variant", ["g7", "g14", "g28"])
def test_bag_reshape(variant: str) -> None:
    """Bag inference: flat [B*S, C, T, H, W] output reshapes to [B, S, T', H', W']."""
    model = DenseC3D(pools=POOL_CONFIGS[variant], variant_name=variant)
    model.eval()
    x = torch.randn(BATCH * CLIPS, C, T, 112, 112)
    with torch.no_grad():
        out = model(x)   # (B*S, T', H', W')
    expected_T, expected_H, expected_W = EXPECTED_SHAPES[variant]
    bag = out.view(BATCH, CLIPS, expected_T, expected_H, expected_W)
    assert bag.shape == (BATCH, CLIPS, expected_T, expected_H, expected_W)


@pytest.mark.parametrize("variant", ["g7", "g14", "g28"])
def test_clip_logit_shape(variant: str) -> None:
    """clip_logit returns a scalar per batch item."""
    model = DenseC3D(pools=POOL_CONFIGS[variant], variant_name=variant)
    model.eval()
    x = torch.randn(BATCH, C, T, 112, 112)
    with torch.no_grad():
        vox = model(x)
        scores = model.clip_logit(vox, k_frac=0.1)
    assert scores.shape == (BATCH,), f"Expected ({BATCH},), got {scores.shape}"
