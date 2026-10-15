# Implementation Plan

> **Note:** The comprehensive 10-phase implementation plan, technical specifications, experiment matrix, and 16-week timeline have been expanded into the root master file:  
> 📄 [**`implementation_plan.md`**](file:///Users/prathmesh/Documents/Anomaly%20detection/implementation_plan.md)  
> 
> Please refer to [**`summary.md`**](file:///Users/prathmesh/Documents/Anomaly%20detection/summary.md) for the consolidated codebase inventory and implementation roadmap.

---

## Quick Reference: Top-Level Execution Scripts (Phases 1, 3, 4, 5)

The pipeline uses the following top-level CLI scripts to execute tasks on the target machine:

### 1. `prepare_data.py` (Data Pipeline — Phase 1)
- Ingest raw UCF-Crime dataset.
- Transcode videos to 10 FPS at low resolution (128×171) via OpenCV / FFmpeg.
- Validate frame count (minimum 16 frames); drop corrupted files.
- Generate manifests: `manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv` following official splits.

### 2. `train_runner.py` / `run_training.py` (Training Engine — Phases 3 & 4)
- Load config parameters from `ProjectConfig`.
- Instantiate datasets (`MILBagDataset` / `MILBagValidationDataset`).
- Instantiate model (`DenseC3D` for dense voxel localization or `Scratch3DCNN` for clip baseline).
- Train with AdamW, AMP, cosine annealing, and resume-safe checkpointing.
- Save `best_model.pt` (selected strictly on validation AUC) and `last_model.pt`.

### 3. `run_evaluation.py` (Evaluation & Metrics — Phases 4, 7, 8)
- Load frozen `best_model.pt`.
- Calibrate threshold on validation split; freeze into `calibrated_threshold.json`.
- Compute test frame-level ROC-AUC, PR-AUC, and false alarms per hour.
- Compute spatial localization metrics (pointing game, energy-in-region, pixel AUC, IoU).
- Run shortcut audit and anomalous-only AUC.

### 4. `predict_video.py` (Inference & Heatmap Demo — Phases 5 & 6)
- Ingest single `.mp4` video.
- Extract 16-frame sliding clips (stride 8) $\to$ predict voxel logits.
- Run `inference/heatmaps.py` to upsample and average overlapping clips.
- Apply per-pixel EMA temporal smoothing.
- Render annotated video with heatmap overlay, score plot, and anomaly intervals.
