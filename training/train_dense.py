"""Training loop for DenseC3D voxel-level MIL anomaly detection.

Provides one-epoch training, validation, checkpointing, and the main
``train_dense()`` driver with early stopping on validation AUC.

Key differences from training/train.py (clip-level):
    - Model outputs voxel maps [B*S, T', H', W']; loss expects [B, S, T', H', W'].
    - Validation uses model.clip_logit() to produce clip scores.
    - Per-epoch logs include: val_auc, mean_voxel_sparsity, border_mass_frac.
"""

import os
import csv
import random
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from training.config import ProjectConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RNG helpers (reuse same pattern as train.py)
# ---------------------------------------------------------------------------

def get_rng_states() -> Dict[str, Any]:
    """Capture RNG states for Python, NumPy, and PyTorch (CPU + CUDA)."""
    states: Dict[str, Any] = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore previously captured RNG states."""
    random.setstate(states["random"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


# ---------------------------------------------------------------------------
# Checkpointing (same signature as train.py for checkpoint compatibility)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[Any],
    epoch: int,
    val_auc: float,
    config: Dict[str, Any],
    rng_states: Dict[str, Any],
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save full training state to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "val_auc": val_auc,
        "config": config,
        "rng_states": rng_states,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load checkpoint and restore all states. Returns the checkpoint dict."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if checkpoint.get("rng_states"):
        set_rng_states(checkpoint["rng_states"])
    return checkpoint


# ---------------------------------------------------------------------------
# Voxel monitoring helpers (Phase 3 logging additions)
# ---------------------------------------------------------------------------

def _compute_voxel_sparsity(vox: torch.Tensor) -> float:
    """Mean sigmoid activation across all voxels in the batch.

    A value near 0 means the model is sparse (good); near 1 means saturated.
    Args:
        vox: Voxel logit map [B, S, T', H', W'] or [N, T', H', W'].
    """
    return torch.sigmoid(vox).mean().item()


def _compute_border_mass_fraction(vox: torch.Tensor, border_frac: float = 0.15) -> float:
    """Fraction of heatmap activation mass in the spatial border region.

    A high border fraction is an early warning for border/letterbox shortcuts.
    Args:
        vox:         Voxel logit map [..., H', W'].
        border_frac: Width of the border as a fraction of the spatial dimension.
    Returns:
        Scalar in [0, 1]: fraction of total probability mass at the border.
    """
    probs = torch.sigmoid(vox)
    H, W = probs.shape[-2], probs.shape[-1]
    bh = max(1, int(border_frac * H))
    bw = max(1, int(border_frac * W))

    total_mass = probs.sum().item() + 1e-8
    # Border: top/bottom rows and left/right columns
    border = (
        probs[..., :bh, :].sum()
        + probs[..., -bh:, :].sum()
        + probs[..., bh:-bh, :bw].sum()
        + probs[..., bh:-bh, -bw:].sum()
    )
    return (border.item() / total_mass)


# ---------------------------------------------------------------------------
# Single-epoch training
# ---------------------------------------------------------------------------

def train_one_epoch_dense(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn,          # dense_mil_loss callable
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    device: torch.device,
    clips_per_bag: int,
    k_frac: float = 0.01,
    margin: float = 1.0,
    lam_sp: float = 8e-5,
    lam_tv: float = 8e-5,
    lam_norm: float = 1e-2,
    proximity_window=None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one training epoch over MIL bag-pairs with dense voxel-level loss.

    Returns a dict of averaged metrics for the epoch.
    """
    model.train()
    S = clips_per_bag
    keys = ["total", "rank", "sparse", "tv_space", "tv_time", "norm"]
    running = {k: 0.0 for k in keys}
    running_sparsity = 0.0
    running_border = 0.0
    num_batches = 0

    for anom_bags, norm_bags in tqdm(dataloader, desc="Training", leave=False):
        # anom_bags / norm_bags: (B, S, C, T, H, W)
        anom_bags = anom_bags.to(device)
        norm_bags = norm_bags.to(device)

        B, K, C, T, H, W = anom_bags.shape
        assert K == S, f"Expected {S} clips per bag, got {K}"

        # Flatten bags for single forward pass: (B*S, C, T, H, W)
        anom_flat = anom_bags.view(B * S, C, T, H, W)
        norm_flat = norm_bags.view(B * S, C, T, H, W)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        device_type = device.type if device.type in ("cuda", "cpu") else "cpu"

        with torch.autocast(device_type=device_type, enabled=use_amp):
            # Forward: voxel maps [B*S, T', H', W']
            vox_a_flat = model(anom_flat)
            vox_n_flat = model(norm_flat)

            # Infer T', H', W' from output
            _, Tp, Hp, Wp = vox_a_flat.shape

            # Reshape to [B, S, T', H', W'] for dense loss
            vox_a = vox_a_flat.view(B, S, Tp, Hp, Wp)
            vox_n = vox_n_flat.view(B, S, Tp, Hp, Wp)

            loss_dict = loss_fn(
                vox_a, vox_n,
                k_frac=k_frac,
                margin=margin,
                lam_sp=lam_sp,
                lam_tv=lam_tv,
                lam_norm=lam_norm,
                proximity_window=proximity_window,
            )
            loss = loss_dict["total"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # Accumulate loss components
        for k in keys:
            v = loss_dict.get(k, 0.0)
            running[k] += v.item() if torch.is_tensor(v) else float(v)

        # Phase 3 monitoring (detached, no grad)
        with torch.no_grad():
            running_sparsity += _compute_voxel_sparsity(vox_a)
            running_border += _compute_border_mass_fraction(vox_a)

        num_batches += 1

    n = max(num_batches, 1)
    result = {k: v / n for k, v in running.items()}
    result["mean_voxel_sparsity"] = running_sparsity / n
    result["border_mass_frac"] = running_border / n
    return result


# ---------------------------------------------------------------------------
# Validation (video-level AUC via clip_logit → max)
# ---------------------------------------------------------------------------

def validate_dense(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    batch_size: int = 16,
    k_frac: float = 0.1,
) -> Dict[str, Any]:
    """Score every clip in each validation video using clip_logit and compute video-level AUC.

    Supports MILBagValidationDataset which returns
    ``(clips_tensor, label, video_path, timestamps)`` per item.
    """
    model.eval()
    video_scores = []
    video_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Validation", leave=False):
            item = val_dataset[i]
            clips = item[0]   # (N, C, T, H, W)
            label = item[1]   # int

            clips = clips.to(device)
            clip_scores = []

            for j in range(0, clips.size(0), batch_size):
                batch = clips[j: j + batch_size]    # (bs, C, T, H, W)
                device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
                with torch.autocast(device_type=device_type):
                    vox = model(batch)               # (bs, T', H', W')
                scores = model.clip_logit(vox, k_frac=k_frac)  # (bs,)
                # Convert logit score to probability
                clip_scores.append(torch.sigmoid(scores).cpu())

            clip_scores = torch.cat(clip_scores)     # (N,)
            video_scores.append(clip_scores.max().item())
            video_labels.append(int(label))

    video_scores_np = np.array(video_scores)
    video_labels_np = np.array(video_labels)

    if len(np.unique(video_labels_np)) > 1:
        auc = float(roc_auc_score(video_labels_np, video_scores_np))
    else:
        auc = 0.0

    return {
        "val_auc": auc,
        "val_video_scores": video_scores_np.tolist(),
        "val_video_labels": video_labels_np.tolist(),
    }


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train_dense(
    config: ProjectConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_dataset,
    loss_fn,
    device: torch.device,
    variant_tag: str = "g14",
) -> str:
    """Full training loop for DenseC3D with early stopping on validation AUC.

    Returns the path to the best checkpoint.
    """
    from training.losses.dense_mil import dense_mil_loss  # local import to avoid circular

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    use_amp = config.AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    seed = config.SEED
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    run_dir = os.path.join(config.RUNS_DIR, f"dense_{variant_tag}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    best_path = os.path.join(config.CHECKPOINT_DIR, f"dense_{variant_tag}_best_seed{seed}.pt")
    last_path = os.path.join(config.CHECKPOINT_DIR, f"dense_{variant_tag}_last_seed{seed}.pt")
    log_csv = os.path.join(run_dir, "training_log.csv")

    # Resume from last checkpoint if available
    start_epoch = 1
    best_auc = 0.0
    if os.path.exists(last_path):
        logger.info("Resuming from %s", last_path)
        ckpt = load_checkpoint(last_path, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("val_auc", 0.0)
        print(f"▶ Resumed from epoch {ckpt['epoch']}, val_auc={best_auc:.4f}")

    # CSV header
    if start_epoch == 1:
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr",
                "train_total", "train_rank", "train_sparse",
                "train_tv_space", "train_tv_time", "train_norm",
                "mean_voxel_sparsity", "border_mass_frac",
                "val_auc",
            ])

    patience_counter = 0
    proximity_window = getattr(config, "DENSE_PROXIMITY_WINDOW", None)
    if proximity_window is not None:
        proximity_window = tuple(proximity_window)

    for epoch in range(start_epoch, config.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.EPOCHS}  |  LR={current_lr:.2e}  |  Variant={variant_tag}")
        print(f"{'='*60}")

        # --- train ---
        train_metrics = train_one_epoch_dense(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            clips_per_bag=config.CLIPS_PER_BAG,
            k_frac=config.DENSE_K_FRAC,
            margin=config.DENSE_MARGIN,
            lam_sp=config.DENSE_LAM_SP,
            lam_tv=config.DENSE_LAM_TV,
            lam_norm=config.DENSE_LAM_NORM,
            proximity_window=proximity_window,
            grad_clip=config.GRAD_CLIP,
        )

        # --- validate ---
        val_metrics = validate_dense(model, val_dataset, device, batch_size=16)
        val_auc = val_metrics["val_auc"]

        # --- print ---
        print(
            f"Train loss: {train_metrics['total']:.4f} "
            f"(rank={train_metrics['rank']:.4f}, "
            f"sparse={train_metrics['sparse']:.6f}, "
            f"tv_space={train_metrics['tv_space']:.6f}, "
            f"tv_time={train_metrics['tv_time']:.6f}, "
            f"norm={train_metrics['norm']:.6f})"
        )
        print(
            f"Monitoring → voxel_sparsity={train_metrics['mean_voxel_sparsity']:.4f}  "
            f"border_mass={train_metrics['border_mass_frac']:.4f}"
        )
        print(f"Val AUC: {val_auc:.4f}")

        # --- log CSV ---
        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, current_lr,
                train_metrics["total"], train_metrics["rank"],
                train_metrics["sparse"], train_metrics["tv_space"],
                train_metrics["tv_time"], train_metrics["norm"],
                train_metrics["mean_voxel_sparsity"],
                train_metrics["border_mass_frac"],
                val_auc,
            ])

        scheduler.step()

        # --- checkpoint ---
        rng_states = get_rng_states()
        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}

        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, val_auc, config_dict, rng_states, last_path,
            extra={"variant": variant_tag},
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, val_auc, config_dict, rng_states, best_path,
                extra={"variant": variant_tag},
            )
            print(f"★ New best model saved (AUC={best_auc:.4f}) → {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.PATIENCE})")
            if patience_counter >= config.PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break

    print(f"\n✓ Training complete. Best AUC: {best_auc:.4f}")
    print(f"  Best checkpoint: {best_path}")
    return best_path


__all__ = [
    "get_rng_states",
    "set_rng_states",
    "save_checkpoint",
    "load_checkpoint",
    "train_one_epoch_dense",
    "validate_dense",
    "train_dense",
]
