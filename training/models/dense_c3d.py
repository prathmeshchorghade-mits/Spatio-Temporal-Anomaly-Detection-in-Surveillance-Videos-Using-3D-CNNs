"""Dense fully-convolutional 3D CNN for voxel-level MIL anomaly localization.

Architecture (Phase 2):
    - 4 convolutional blocks with 3x3x3 kernels, BN, ReLU
    - NO global pooling or FC layers — fully convolutional
    - Output: voxel logit map [B, T', H', W'] where T'=4, H'/W' depends on variant
    - Three resolution variants controlled by pool configurations:
        g7  (stride 16): 7×7 spatial map from 112×112 input
        g14 (stride 8):  14×14 spatial map from 112×112 input
        g28 (stride 4):  28×28 spatial map from 112×112 input

Shape contract: input [B, 3, 16, H, W] → voxel map [B, T'=4, H', W']
    For bag inference: input [B*S, 3, 16, H, W] → reshape to [B, S, T', H', W']
    The same weights run on 112×112 (training crop) and 112×144 (inference frame).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


# Pool configurations for the three resolution variants.
# Each entry is a tuple of 4 pool specs (one per block).
# Each spec is either a kernel/stride tuple (kt, kh, kw) or None (no pool).
# Temporal: blocks 2 and 3 pool time by 2; block 1 pools time by 1.
# Spatial: cumulative spatial stride determines output resolution.
POOL_CONFIGS: Dict[str, Tuple] = {
    # Cumulative spatial stride = 2*2*4*1 = 16 → 112/16 = 7
    "g7":  ((1, 2, 2), (2, 2, 2), (2, 4, 4), None),
    # Cumulative spatial stride = 2*2*2*1 = 8  → 112/8  = 14
    "g14": ((1, 2, 2), (2, 2, 2), (2, 2, 2), None),
    # Cumulative spatial stride = 2*2*1*1  = 4  → 112/4  = 28
    # block1 pool(1,2,2)->56×56, block2 pool(2,2,2)->28×28, block3 pool(2,1,1)->28×28, T=16→16→8→4
    "g28": ((1, 2, 2), (2, 2, 2), (2, 1, 1), None),
}


def _conv_block(
    cin: int,
    cout: int,
    pool: Optional[Tuple[int, int, int]],
) -> nn.Sequential:
    """Single 3D conv block: Conv3d → BN → ReLU → [MaxPool3d].

    Args:
        cin:  Input channels.
        cout: Output channels.
        pool: Optional (kt, kh, kw) for MaxPool3d with equal kernel and stride.
              Pass None to skip pooling.
    """
    layers: list = [
        nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    ]
    if pool is not None:
        layers.append(nn.MaxPool3d(kernel_size=pool, stride=pool))
    return nn.Sequential(*layers)


class DenseC3D(nn.Module):
    """Fully-convolutional dense 3D CNN for voxel-level MIL localization.

    Args:
        widths: Channel widths for the 4 conv blocks. Default: (32, 64, 128, 256).
        pools:  Pool spec tuple for each block (4 entries). Use POOL_CONFIGS["g14"]
                as the default (14×14 spatial output from 112×112 input, T'=4).
        variant_name: Optional string tag stored as an attribute for logging.

    Inputs:
        x: Float tensor of shape (B, 3, 16, H, W).

    Outputs:
        forward(x):     Voxel logit map (B, T', H', W') — raw logits, no sigmoid.
        clip_logit(vox, k_frac): Clip-level score = mean of top-k voxel logits.
    """

    def __init__(
        self,
        widths: Tuple[int, ...] = (32, 64, 128, 256),
        pools: Tuple = POOL_CONFIGS["g14"],
        variant_name: str = "g14",
    ):
        super().__init__()
        self.variant_name = variant_name

        cs = (3,) + tuple(widths)
        self.features = nn.Sequential(
            *[_conv_block(cs[i], cs[i + 1], pools[i]) for i in range(4)]
        )
        # 1×1×1 conv: maps channel dimension to a single voxel logit
        self.head = nn.Conv3d(widths[-1], 1, kernel_size=1, bias=True)

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        """Kaiming He init for Conv3d; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute voxel logit map.

        Args:
            x: Input tensor of shape (B, 3, 16, H, W).

        Returns:
            Voxel logit map of shape (B, T', H', W').
        """
        feats = self.features(x)          # (B, C_last, T', H', W')
        logits = self.head(feats)         # (B, 1, T', H', W')
        return logits.squeeze(1)          # (B, T', H', W')

    # ------------------------------------------------------------------
    # Clip-level scoring (used at inference and for the MIL loss)
    # ------------------------------------------------------------------

    def clip_logit(
        self,
        vox: torch.Tensor,
        k_frac: float = 0.1,
    ) -> torch.Tensor:
        """Reduce a voxel map to a scalar clip score via top-k mean.

        Taking the mean of the top-k fraction rather than the single maximum
        is more robust to isolated noisy voxels.

        Args:
            vox:    Voxel logit map of shape (B, T', H', W') or (T', H', W').
            k_frac: Fraction of voxels to average. E.g. 0.1 → top-10%.

        Returns:
            Scalar clip score per example: shape (B,) or scalar.
        """
        squeezed = vox.dim() == 3
        if squeezed:
            vox = vox.unsqueeze(0)

        flat = vox.flatten(start_dim=1)           # (B, N)
        k = max(1, int(k_frac * flat.shape[1]))
        scores = flat.topk(k, dim=1).values.mean(dim=1)   # (B,)

        return scores.squeeze(0) if squeezed else scores

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"variant={self.variant_name}"


# ---------------------------------------------------------------------------
# Parameter counter (mirrors the one in scratch_3dcnn.py)
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Print and return the number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    return trainable
