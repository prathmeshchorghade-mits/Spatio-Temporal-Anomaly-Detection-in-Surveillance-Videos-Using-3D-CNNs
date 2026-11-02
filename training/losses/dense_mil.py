"""Dense voxel-level MIL loss for Phase 2 localization training.

Loss components:
    1. Ranking loss  — hinge on logits: top-k anomalous voxels beat top-k normal voxels.
    2. Sparsity      — penalise mean activation of anomalous maps (small extent).
    3. Smoothness    — spatial TV (within each clip map) + temporal TV (across clips).
    4. Normal-bag    — L2 suppression of all normal voxels toward zero.

Optional proximity-aware variant:
    Instead of globally selecting top-k voxels, find the contiguous space-time window
    of size (wt, wh, ww) that maximises the mean score (via sliding-window max pooling),
    then average the top-k voxels within that window.  This is the ablation row E5
    described in the implementation plan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Top-k helpers
# ---------------------------------------------------------------------------

def topk_mean(x: torch.Tensor, k_frac: float) -> torch.Tensor:
    """Mean of the top-k_frac fraction of values along dim 1.

    Args:
        x:      Tensor of shape (B, N) — flattened voxel logits.
        k_frac: Fraction in (0, 1].  k = max(1, round(k_frac * N)).

    Returns:
        Tensor of shape (B,).
    """
    k = max(1, int(k_frac * x.shape[1]))
    return x.topk(k, dim=1).values.mean(dim=1)


def proximity_topk_mean(
    vox: torch.Tensor,
    k_frac: float,
    window: tuple = (2, 3, 3),
) -> torch.Tensor:
    """Proximity-aware top-k: select voxels within the best contiguous window.

    Finds the spatial-temporal window (wt × wh × ww) with the highest mean
    *sigmoid* score using 3-D average pooling, crops to that window, then
    takes the top-k mean of the raw logits within it.

    Args:
        vox:    Voxel logit map (B, S, T', H', W').
        k_frac: Top-k fraction applied inside the best window.
        window: (wt, wh, ww) window size for sliding-window search.

    Returns:
        Tensor of shape (B,) — one score per bag.
    """
    B, S, T, H, W = vox.shape
    # Flatten segments into the batch dimension for pooling
    flat = vox.reshape(B * S, 1, T, H, W)          # (B*S, 1, T, H, W)
    prob_flat = torch.sigmoid(flat)

    wt, wh, ww = window
    # Pad so the sliding window always covers a full region
    pad_t = wt // 2
    pad_h = wh // 2
    pad_w = ww // 2

    # Average-pool to find the best window centre
    pooled = F.avg_pool3d(
        prob_flat,
        kernel_size=window,
        stride=1,
        padding=(pad_t, pad_h, pad_w),
    )                                               # (B*S, 1, T, H, W)

    # For each sample, find the index of the maximum pooled value
    pooled_flat = pooled.view(B * S, -1)            # (B*S, T*H*W)
    best_idx = pooled_flat.argmax(dim=1)            # (B*S,)

    # Convert flat index back to (t, h, w)
    best_t = best_idx // (H * W)
    best_h = (best_idx % (H * W)) // W
    best_w = best_idx % W

    # Gather logits within the window around the best centre
    scores_list = []
    logit_flat = vox.reshape(B * S, T, H, W)
    for i in range(B * S):
        t0 = max(0, best_t[i].item() - pad_t)
        t1 = min(T, best_t[i].item() + pad_t + 1)
        h0 = max(0, best_h[i].item() - pad_h)
        h1 = min(H, best_h[i].item() + pad_h + 1)
        w0 = max(0, best_w[i].item() - pad_w)
        w1 = min(W, best_w[i].item() + pad_w + 1)
        window_logits = logit_flat[i, t0:t1, h0:h1, w0:w1].flatten()
        k = max(1, int(k_frac * window_logits.numel()))
        scores_list.append(window_logits.topk(k).values.mean())

    scores_per_clip = torch.stack(scores_list)      # (B*S,)
    return scores_per_clip.reshape(B, S).mean(dim=1)  # (B,)


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

def dense_mil_loss(
    vox_a: torch.Tensor,
    vox_n: torch.Tensor,
    k_frac: float = 0.01,
    margin: float = 1.0,
    lam_sp: float = 8e-5,
    lam_tv: float = 8e-5,
    lam_norm: float = 1e-2,
    proximity_window: Optional[tuple] = None,
) -> dict:
    """Dense voxel-level MIL ranking loss with sparsity, smoothness, and normal-bag terms.

    Args:
        vox_a:            Voxel logit maps for anomalous bags: (B, S, T', H', W').
        vox_n:            Voxel logit maps for normal bags:    (B, S, T', H', W').
        k_frac:           Top-k fraction for ranking.  Default 0.01 = top-1% of voxels.
        margin:           Hinge margin (applied on logits).  Default 1.0.
        lam_sp:           Sparsity regularisation weight.
        lam_tv:           TV smoothness regularisation weight (spatial + temporal).
        lam_norm:         Normal-bag L2 suppression weight.
        proximity_window: If not None, use proximity-aware top-k with this window size
                          (wt, wh, ww). Otherwise use global top-k.

    Returns:
        Dict with keys: 'total', 'rank', 'sparse', 'tv_space', 'tv_time', 'norm'.
    """
    B = vox_a.shape[0]

    # ---- 1. Ranking loss on logits (avoids sigmoid saturation) ----
    if proximity_window is not None:
        top_a = proximity_topk_mean(vox_a, k_frac, window=proximity_window)
        top_n = proximity_topk_mean(vox_n, k_frac, window=proximity_window)
    else:
        a_flat = vox_a.flatten(start_dim=1)         # (B, S*T'*H'*W')
        n_flat = vox_n.flatten(start_dim=1)
        top_a = topk_mean(a_flat, k_frac)           # (B,)
        top_n = topk_mean(n_flat, k_frac)           # (B,)

    rank = F.relu(margin - top_a + top_n).mean()

    # ---- 2. Sparsity: mean activation of anomalous bags ----
    pa = torch.sigmoid(vox_a)                       # (B, S, T', H', W')
    sparse = pa.mean()

    # ---- 3. Smoothness: spatial TV + temporal TV ----
    # Spatial TV: differences along height and width dimensions (last two dims)
    tv_h = (pa[..., 1:, :] - pa[..., :-1, :]).abs().mean()
    tv_w = (pa[..., :, 1:] - pa[..., :, :-1]).abs().mean()
    tv_space = tv_h + tv_w

    # Temporal TV: differences between consecutive segment scores
    # Pool spatial dims to get per-segment scalar: (B, S)
    seg_scores = pa.mean(dim=(2, 3, 4))             # mean over T', H', W'
    tv_time = (seg_scores[:, 1:] - seg_scores[:, :-1]).pow(2).mean()

    # ---- 4. Normal-bag suppression ----
    pn = torch.sigmoid(vox_n)
    norm_loss = pn.pow(2).mean()

    # ---- Total ----
    total = (
        rank
        + lam_sp * sparse
        + lam_tv * (tv_space + tv_time)
        + lam_norm * norm_loss
    )

    return {
        "total":    total,
        "rank":     rank,
        "sparse":   lam_sp * sparse,
        "tv_space": lam_tv * tv_space,
        "tv_time":  lam_tv * tv_time,
        "norm":     lam_norm * norm_loss,
    }
