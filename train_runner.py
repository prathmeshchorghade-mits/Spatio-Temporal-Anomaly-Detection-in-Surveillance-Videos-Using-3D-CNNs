import os
import sys

from training.config import ProjectConfig, set_seed, get_device, save_config
from training.model.scratch_3dcnn import Scratch3DCNN, count_parameters
from training.loss import MILRankingLoss
from training.dataset.transforms import VideoTransform
from training.dataset.mil_dataset import (
    UCFCrimeManifest, MILBagDataset, MILBagValidationDataset,
    mil_collate_fn
)
from training.train import train
import torch
from torch.utils.data import DataLoader

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    seeds = [42, 123, 456]
    print(f"Starting multi-seed training run with seeds: {seeds}")

    # Transforms
    train_transform = VideoTransform(mode='train')
    val_transform = VideoTransform(mode='val')

    # Base configuration
    config = ProjectConfig()
    config.EPOCHS = 60
    config.PATIENCE = 12
    config.BATCH_SIZE = 4
    config.AMP = True
    
    try:
        train_manifest = UCFCrimeManifest(os.path.join(config.DATA_ROOT, 'manifest_train.csv'))
        val_manifest = UCFCrimeManifest(os.path.join(config.DATA_ROOT, 'manifest_val.csv'))
    except FileNotFoundError as e:
        print(f"Error loading manifests: {e}")
        print("Please ensure you have run the data preparation step first (prepare_data.py).")
        sys.exit(1)

    print(f"Train: {len(train_manifest.normal_videos)} normal, {len(train_manifest.anomalous_videos)} anomalous")
    print(f"Val: {len(val_manifest.normal_videos)} normal, {len(val_manifest.anomalous_videos)} anomalous")

    val_dataset = MILBagValidationDataset(
        manifest=val_manifest,
        transform=val_transform,
        clip_length=config.CLIP_LENGTH,
        clip_stride=config.CLIP_STRIDE,
        target_fps=config.TARGET_FPS,
    )

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Starting Training for Seed: {seed}")
        print(f"{'='*60}")
        
        config.SEED = seed
        set_seed(seed)
        
        # Save run config
        run_dir = os.path.join(config.RUNS_DIR, f'run_seed{seed}')
        os.makedirs(run_dir, exist_ok=True)
        save_config(config, os.path.join(run_dir, 'run_config.json'))
        
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
            num_workers=2,
            collate_fn=mil_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        model = Scratch3DCNN().to(device)
        
        criterion = MILRankingLoss(
            margin=config.RANKING_MARGIN,
            lambda_sparse=config.LAMBDA_SPARSE,
            lambda_smooth=config.LAMBDA_SMOOTH,
            topk=config.TOPK,
        )
        
        best_model_path = train(
            config=config,
            model=model,
            train_loader=train_loader,
            val_dataset=val_dataset,
            criterion=criterion,
            device=device,
        )
        print(f"Finished seed {seed}. Best model saved to: {best_model_path}")

if __name__ == '__main__':
    main()
