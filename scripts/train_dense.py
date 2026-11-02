#!/usr/bin/env python3
"""Training runner for Phase 2 DenseC3D model.

Usage examples:
    # Basic: point at your dataset root (must contain manifest_train.csv / manifest_val.csv)
    python scripts/train_dense.py --data-root /path/to/your/dataset

    # With transcoding from raw UCF-Crime (generates manifests automatically):
    python scripts/train_dense.py --raw-dir /path/to/raw_ucfcrime --data-root ./data/processed

    # Choose resolution variant (g7, g14, g28):
    python scripts/train_dense.py --data-root /path/to/dataset --variant g14

    # Multi-seed ablation (e.g. seeds 0, 1, 2):
    python scripts/train_dense.py --data-root /path/to/dataset --seeds 0 1 2

    # All options:
    python scripts/train_dense.py \\
        --data-root /path/to/dataset \\
        --variant g14 \\
        --seeds 0 1 2 \\
        --epochs 60 \\
        --batch-size 4 \\
        --clips-per-bag 32 \\
        --lr 1e-4 \\
        --checkpoint-dir ./checkpoints \\
        --runs-dir ./runs \\
        --no-amp
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where the script is called from
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from training.config import ProjectConfig, set_seed, get_device, save_config
from training.models.dense_c3d import DenseC3D, POOL_CONFIGS, count_parameters
from training.losses.dense_mil import dense_mil_loss
from training.dataset.transforms import VideoTransform
from training.dataset.mil_dataset import (
    UCFCrimeManifest,
    MILBagDataset,
    MILBagValidationDataset,
    mil_collate_fn,
)
from training.train_dense import train_dense

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_dense")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DenseC3D (Phase 2) on UCF-Crime with dense voxel-level MIL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    data_grp = p.add_argument_group("Data")
    data_grp.add_argument(
        "--data-root", type=str, required=True,
        help=(
            "Directory that contains (or will contain) manifest_train.csv and "
            "manifest_val.csv. If --raw-dir is also provided, prepare_data.py is "
            "run first to generate manifests here."
        ),
    )
    data_grp.add_argument(
        "--raw-dir", type=str, default=None,
        help=(
            "Path to the raw UCF-Crime video directory. If set, the data "
            "preparation step is run automatically before training."
        ),
    )
    data_grp.add_argument(
        "--target-fps", type=int, default=10,
        help="Target FPS when transcoding (only used if --raw-dir is set).",
    )

    # ---- Model ----
    model_grp = p.add_argument_group("Model")
    model_grp.add_argument(
        "--variant", type=str, default="g14", choices=["g7", "g14", "g28"],
        help="DenseC3D resolution variant (g7=7×7, g14=14×14, g28=28×28 spatial output).",
    )
    model_grp.add_argument(
        "--widths", type=int, nargs=4, default=[32, 64, 128, 256],
        metavar=("W1", "W2", "W3", "W4"),
        help="Channel widths for the 4 conv blocks.",
    )

    # ---- Loss ----
    loss_grp = p.add_argument_group("Loss")
    loss_grp.add_argument("--k-frac", type=float, default=0.01,
                          help="Top-k fraction for voxel ranking (0.01 = top-1%%).")
    loss_grp.add_argument("--margin", type=float, default=1.0,
                          help="Hinge margin for ranking loss.")
    loss_grp.add_argument("--lam-sp", type=float, default=8e-5,
                          help="Sparsity regularisation weight.")
    loss_grp.add_argument("--lam-tv", type=float, default=8e-5,
                          help="Total-variation smoothness weight.")
    loss_grp.add_argument("--lam-norm", type=float, default=1e-2,
                          help="Normal-bag suppression weight.")
    loss_grp.add_argument(
        "--proximity-window", type=int, nargs=3, default=None,
        metavar=("WT", "WH", "WW"),
        help="Enable proximity-aware top-k with this window size (e.g. 2 3 3). "
             "Omit to use global top-k.",
    )

    # ---- Training ----
    train_grp = p.add_argument_group("Training")
    train_grp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                           help="Random seeds for multi-seed training.")
    train_grp.add_argument("--epochs", type=int, default=60)
    train_grp.add_argument("--patience", type=int, default=12,
                           help="Early-stopping patience (epochs without val AUC improvement).")
    train_grp.add_argument("--batch-size", type=int, default=4,
                           help="Number of bag-pairs per batch.")
    train_grp.add_argument("--clips-per-bag", type=int, default=32,
                           help="Number of clips sampled per video bag.")
    train_grp.add_argument("--lr", type=float, default=1e-4)
    train_grp.add_argument("--weight-decay", type=float, default=5e-5)
    train_grp.add_argument("--grad-clip", type=float, default=1.0)
    train_grp.add_argument("--num-workers", type=int, default=2,
                           help="DataLoader worker processes (set 0 on Windows/MPS).")
    train_grp.add_argument("--no-amp", action="store_true",
                           help="Disable automatic mixed precision (AMP).")

    # ---- Output paths ----
    out_grp = p.add_argument_group("Output paths")
    out_grp.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    out_grp.add_argument("--runs-dir", type=str, default="./runs")
    out_grp.add_argument("--output-dir", type=str, default="./outputs")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Optional data-preparation step
# ---------------------------------------------------------------------------

def maybe_prepare_data(args: argparse.Namespace) -> None:
    """Run prepare_data.py if --raw-dir is provided and manifests are missing."""
    data_root = Path(args.data_root)
    train_manifest = data_root / "manifest_train.csv"
    val_manifest = data_root / "manifest_val.csv"

    if train_manifest.exists() and val_manifest.exists():
        logger.info("Manifests found — skipping data preparation.")
        return

    if args.raw_dir is None:
        logger.error(
            "manifest_train.csv / manifest_val.csv not found in %s.\n"
            "Either:\n"
            "  (a) Pass --raw-dir /path/to/raw_ucfcrime to auto-generate them, or\n"
            "  (b) Run scripts/prepare_data.py manually first.",
            data_root,
        )
        sys.exit(1)

    logger.info("Running data preparation (this may take a while)…")
    prepare_script = ROOT_DIR / "scripts" / "prepare_data.py"
    cmd = [
        sys.executable, str(prepare_script),
        "--raw-dir", args.raw_dir,
        "--processed-dir", str(data_root),
        "--manifest-dir", str(data_root),
        "--target-fps", str(args.target_fps),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Data preparation failed (exit code %d).", result.returncode)
        sys.exit(result.returncode)
    logger.info("Data preparation complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Step 0: prepare data if needed ----
    maybe_prepare_data(args)

    # ---- Step 1: device ----
    device = get_device()
    logger.info("Using device: %s", device)

    # ---- Step 2: base config ----
    config = ProjectConfig()
    config.DATA_ROOT       = args.data_root
    config.CHECKPOINT_DIR  = args.checkpoint_dir
    config.RUNS_DIR        = args.runs_dir
    config.OUTPUT_DIR      = args.output_dir
    config.EPOCHS          = args.epochs
    config.PATIENCE        = args.patience
    config.BATCH_SIZE      = args.batch_size
    config.CLIPS_PER_BAG   = args.clips_per_bag
    config.LR              = args.lr
    config.WEIGHT_DECAY    = args.weight_decay
    config.GRAD_CLIP       = args.grad_clip
    config.AMP             = not args.no_amp
    # Dense-specific
    config.DENSE_VARIANT          = args.variant
    config.DENSE_WIDTHS           = args.widths
    config.DENSE_K_FRAC           = args.k_frac
    config.DENSE_MARGIN           = args.margin
    config.DENSE_LAM_SP           = args.lam_sp
    config.DENSE_LAM_TV           = args.lam_tv
    config.DENSE_LAM_NORM         = args.lam_norm
    config.DENSE_PROXIMITY_WINDOW = (
        list(args.proximity_window) if args.proximity_window else None
    )

    # ---- Step 3: transforms ----
    train_transform = VideoTransform(mode="train")
    val_transform   = VideoTransform(mode="val")

    # ---- Step 4: manifests ----
    data_root = Path(args.data_root)
    try:
        train_manifest = UCFCrimeManifest(str(data_root / "manifest_train.csv"))
        val_manifest   = UCFCrimeManifest(str(data_root / "manifest_val.csv"))
    except FileNotFoundError as e:
        logger.error("Could not load manifests: %s", e)
        sys.exit(1)

    logger.info(
        "Train: %d normal, %d anomalous",
        len(train_manifest.normal_videos), len(train_manifest.anomalous_videos),
    )
    logger.info(
        "Val:   %d normal, %d anomalous",
        len(val_manifest.normal_videos), len(val_manifest.anomalous_videos),
    )

    # ---- Step 5: validation dataset (shared across seeds) ----
    val_dataset = MILBagValidationDataset(
        manifest=val_manifest,
        transform=val_transform,
        clip_length=config.CLIP_LENGTH,
        clip_stride=config.CLIP_STRIDE,
        target_fps=config.TARGET_FPS,
    )

    # ---- Step 6: multi-seed training loop ----
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  DenseC3D [{args.variant}]  |  Seed {seed}")
        print(f"{'='*60}")

        config.SEED = seed
        set_seed(seed)

        # Per-seed run dir + config snapshot
        run_dir = os.path.join(config.RUNS_DIR, f"dense_{args.variant}_seed{seed}")
        os.makedirs(run_dir, exist_ok=True)
        save_config(config, os.path.join(run_dir, "run_config.json"))

        # Training dataset (re-created per seed for reproducible sampling)
        train_dataset = MILBagDataset(
            manifest=train_manifest,
            transform=train_transform,
            clips_per_bag=config.CLIPS_PER_BAG,
            clip_length=config.CLIP_LENGTH,
            target_fps=config.TARGET_FPS,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=mil_collate_fn,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        # Model
        model = DenseC3D(
            widths=tuple(config.DENSE_WIDTHS),
            pools=POOL_CONFIGS[config.DENSE_VARIANT],
            variant_name=config.DENSE_VARIANT,
        ).to(device)

        logger.info("DenseC3D [%s] parameter count:", args.variant)
        count_parameters(model)

        # Train
        best_path = train_dense(
            config=config,
            model=model,
            train_loader=train_loader,
            val_dataset=val_dataset,
            loss_fn=dense_mil_loss,
            device=device,
            variant_tag=args.variant,
        )
        print(f"✓ Seed {seed} done. Best checkpoint: {best_path}")

    print(f"\n✅ All seeds {args.seeds} complete for variant [{args.variant}].")


if __name__ == "__main__":
    main()
