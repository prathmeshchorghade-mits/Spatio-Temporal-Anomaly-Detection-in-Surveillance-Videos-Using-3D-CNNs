"""Training loop for MIL 3D CNN video anomaly detection.

Provides one-epoch training, validation, checkpointing with full state,
and the main ``train()`` driver with early stopping on validation AUC.
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from training.config import ProjectConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RNG state helpers — needed for exact resume after Colab disconnect
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
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    val_auc: float,
    config: Dict[str, Any],
    rng_states: Dict[str, Any],
    path: str,
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
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    """Load checkpoint and restore all states.  Returns the checkpoint dict."""
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
# Single-epoch training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one training epoch over MIL bag-pairs.

    Returns a dict of averaged loss components for the epoch.
    """
    model.train()
    running = {"total": 0.0, "rank": 0.0, "sparse": 0.0, "smooth": 0.0}
    num_batches = 0

    for anom_bags, norm_bags in tqdm(dataloader, desc="Training", leave=False):
        # anom_bags / norm_bags: (B, clips_per_bag, C, T, H, W)
        anom_bags = anom_bags.to(device)
        norm_bags = norm_bags.to(device)

        B, K, C, T, H, W = anom_bags.shape

        # Flatten bags → (B*K, C, T, H, W)
        anom_flat = anom_bags.view(-1, C, T, H, W)
        norm_flat = norm_bags.view(-1, C, T, H, W)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        with autocast(enabled=use_amp):
            anom_logits = model(anom_flat).view(B, K)   # (B, K)
            norm_logits = model(norm_flat).view(B, K)   # (B, K)

            # MILRankingLoss returns {'total', 'rank', 'sparse', 'smooth'}
            loss_dict = criterion(anom_logits, norm_logits)
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

        for k in running:
            val = loss_dict[k]
            running[k] += val.item() if torch.is_tensor(val) else float(val)
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in running.items()}


# ---------------------------------------------------------------------------
# Validation (video-level AUC via max clip score)
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Score every clip in each validation video and compute video-level AUC."""
    model.eval()
    video_scores = []
    video_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Validation", leave=False):
            # MILBagValidationDataset returns (clips, label, path, timestamps)
            item = val_dataset[i]
            clips = item[0]          # (N, C, T, H, W)
            label = item[1]          # int

            clips = clips.to(device)
            scores = []
            for j in range(0, clips.size(0), batch_size):
                batch = clips[j : j + batch_size]
                with autocast():
                    logits = model(batch)
                batch_scores = torch.sigmoid(logits).squeeze(-1)
                scores.append(batch_scores.cpu())

            scores = torch.cat(scores)
            video_scores.append(scores.max().item())
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

def train(
    config: ProjectConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_dataset,
    criterion: nn.Module,
    device: torch.device,
) -> str:
    """Full training loop with early stopping on validation AUC.

    Returns the path to the best checkpoint.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    scaler = GradScaler(enabled=config.AMP) if config.AMP else None

    # Paths
    seed = config.SEED
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    run_dir = os.path.join(config.RUNS_DIR, f"run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    best_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_seed{seed}.pt")
    last_path = os.path.join(config.CHECKPOINT_DIR, f"last_model_seed{seed}.pt")
    log_csv = os.path.join(run_dir, "training_log.csv")

    # Resume from last checkpoint if it exists
    start_epoch = 1
    best_auc = 0.0
    if os.path.exists(last_path):
        logger.info("Resuming from %s", last_path)
        ckpt = load_checkpoint(last_path, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("val_auc", 0.0)
        print(f"▶ Resumed from epoch {ckpt['epoch']}, val_auc={best_auc:.4f}")

    # Initialise CSV header (only if starting fresh)
    if start_epoch == 1:
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr",
                "train_total", "train_rank", "train_sparse", "train_smooth",
                "val_auc",
            ])

    patience_counter = 0

    for epoch in range(start_epoch, config.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.EPOCHS}  |  LR={current_lr:.2e}")
        print(f"{'='*60}")

        # --- train ---
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, grad_clip=config.GRAD_CLIP,
        )

        # --- validate ---
        val_metrics = validate(model, val_dataset, device, batch_size=16)
        val_auc = val_metrics["val_auc"]

        # --- log ---
        print(
            f"Train loss: {train_metrics['total']:.4f} "
            f"(rank={train_metrics['rank']:.4f}, "
            f"sparse={train_metrics['sparse']:.6f}, "
            f"smooth={train_metrics['smooth']:.6f})"
        )
        print(f"Val AUC: {val_auc:.4f}")

        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, current_lr,
                train_metrics["total"], train_metrics["rank"],
                train_metrics["sparse"], train_metrics["smooth"],
                val_auc,
            ])

        scheduler.step()

        # --- checkpoint ---
        rng_states = get_rng_states()
        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}

        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, val_auc, config_dict, rng_states, last_path,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, val_auc, config_dict, rng_states, best_path,
            )
            print(f"★ New best model saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.PATIENCE})")
            if patience_counter >= config.PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break

    print(f"\n✓ Training complete. Best AUC: {best_auc:.4f}")
    print(f"  Best checkpoint: {best_path}")
    return best_path
