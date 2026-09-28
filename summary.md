# Project Summary — Spatio-Temporal Video Anomaly Detection

> **Project:** Capstone project for Madhav Institute of Technology and Science (MITS)
> **Objective:** Train a PyTorch 3D CNN from scratch in local environment to score suspicious behaviour in surveillance videos using UCF-Crime dataset with MIL (Multiple Instance Learning) ranking loss.
> **Research Question:** *How far can a compact, from-scratch spatio-temporal model get on weak video-level labels, and what closes the gap to pretrained pipelines?*

---

## Table of Contents

- [What Has Been Implemented](#what-has-been-implemented)
- [File-by-File Explanation](#file-by-file-explanation)
- [Things To Be Implemented](#things-to-be-implemented)

---

## What Has Been Implemented

The project follows a **six-phase implementation plan** (Phase 0–5) mapping to five local environment notebooks. The core library (Phase 2 — Model & Loss) and shared infrastructure are complete; the local environment-dependent phases remain.

### ✅ Core Library — Model & Loss (Phase 2)

The 3D CNN architecture, MIL ranking loss, and supporting training infrastructure have been built as importable Python modules (not local environment-only notebook cells), enabling local CPU testing before GPU deployment:

| Component | Status | Description |
|---|---|---|
| **Configuration system** | ✅ Done | `ProjectConfig` dataclass with JSON save/load, seed control, device selection |
| **3D CNN model** | ✅ Done | Custom C3D-inspired architecture with 4 conv blocks, global adaptive pooling, and MLP anomaly head |
| **MIL ranking loss** | ✅ Done | Hinge ranking loss (logit-based) + temporal sparsity + temporal smoothness regularization; supports max-MIL and top-k MIL; includes smoke tests |
| **Video transforms** | ✅ Done | Training (random crop, flip, color jitter), validation/test (center crop), inference (no crop, full frame 112×144) modes with Kinetics-400 normalization |
| **MIL bag dataset** | ✅ Done | Bag-pair training dataset (32 segments, re-jittered per epoch) and full-video sliding-window validation dataset with OpenCV decoding |
| **UCF-Crime manifest** | ✅ Done | CSV-based manifest loader with video path, label, category, and split support |
| **Clip sampling** | ✅ Done | Temporal segment sampling for MIL bags and sliding-window extraction for validation/inference |
| **Evaluation metrics** | ✅ Done | Video-level AUC, frame/clip-level ROC-AUC & PR-AUC, false alarm rate per hour, threshold calibration (validation-only), category breakdown, ROC/PR curve plotting |
| **Training loop** | ✅ Done | Full training driver with AdamW optimizer, cosine annealing LR, AMP, gradient clipping, early stopping on val AUC, resume-safe checkpointing (model, optimizer, scheduler, scaler, RNG states), and CSV logging |
| **Postprocessing** | ✅ Done | EMA smoothing (α=0.3), hysteresis thresholding with run-length persistence, clip-to-timestamp conversion, score CSV generation, score timeline plotting |

### ✅ Documentation

| Document | Status | Description |
|---|---|---|
| `README.md` | ✅ Done | Project overview, architecture diagram, 6-phase implementation plan, technical config tables, proposed enhancements, differentiation strategy, risk mitigation |
| `SYSTEM_ARCHITECTURE.md` | ✅ Done | Quality goals, training architecture, data contracts, technical configuration, decision logic, ablation ladder, explainability plan, verification gates |
| `docs/implementation.md` | ✅ Done | Phased implementation guide, team responsibilities, timeline, dependency order |

---

## File-by-File Explanation

### Project Root

| File | Description |
|---|---|
| [`README.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/README.md) | Main project documentation. Describes the system's purpose, research question, Mermaid architecture diagrams, six-phase implementation plan (Phase 0–5), technical configuration summary tables, team responsibilities, proposed enhancements (ablation ladder, Grad-CAM, cross-dataset eval), differentiation strategy (campus-safety, category head, seed ensemble), risk mitigation plan, local environment workflow, and academic references. |
| [`SYSTEM_ARCHITECTURE.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/SYSTEM_ARCHITECTURE.md) | Detailed system architecture. Covers scope and quality goals, pre-transcode data pipeline, training architecture flow, data contracts (JSON schema), technical configuration tables (clip/MIL/training/inference settings), 3-seed reporting, offline inference decision logic, output artefact naming, local environment notebook boundaries, experiment artefact design, proposed enhancements (ablation, explainability, evaluation rigor, differentiation), verification gates, and risk mitigations. |
| [`.gitignore`](file:///Users/prathmesh/Documents/Anomaly%20detection/.gitignore) | Git ignore rules for data files, checkpoints, and outputs. |

### `training/` — Training Package

| File | Lines | Description |
|---|---|---|
| [`__init__.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/__init__.py) | 18 | Package init. Exports `ProjectConfig`, `set_seed`, `get_device`, `save_config`, `load_config`. |
| [`config.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/config.py) | 238 | **Central configuration module.** Defines the `ProjectConfig` dataclass with all hyperparameters: clip settings (length=16, stride=8, fps=10), spatial dimensions (resize 128×171, crop 112×112), Kinetics-400 normalization stats, MIL settings (32 clips/bag, margin=0.5), training settings (AdamW LR=1e-4, 60 epochs, early stopping patience=12, AMP, 3 seeds), inference thresholds (EMA α=0.3, validation-calibrated enter threshold, exit delta=0.1), and local disk paths. Also provides `set_seed()` for reproducibility (Python, NumPy, PyTorch, cuDNN), `get_device()` for hardware selection, and `save_config()`/`load_config()` for JSON serialization. Supports case-insensitive key mapping. |
| [`loss.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/loss.py) | 115 | **MIL Ranking Loss module.** Implements `MILRankingLoss(nn.Module)` with three components: (1) **Ranking loss** — hinge loss on raw logits pushing top anomalous clip above top normal clip by a margin (`max(0, m - topk_anom + topk_norm)`); supports both max-MIL and top-k MIL modes. (2) **Sparsity loss** — encourages anomalies to be rare (penalizes mean sigmoid score). (3) **Smoothness loss** — penalizes rapid score changes between adjacent clips. Returns a dict of `{total, rank, sparse, smooth}` losses. Includes a `_smoke_test()` function that validates gradients flow for both max-MIL and top-k modes. |
| [`metrics.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/metrics.py) | 172 | **Evaluation metrics module.** Functions: `compute_video_level_auc()` — ROC-AUC on max clip scores per video; `compute_frame_level_auc()` — ROC-AUC and PR-AUC at clip/frame level; `compute_false_alarm_rate()` — counts false alarms per video-hour on normal videos using burst-based detection; `calibrate_threshold()` — finds the optimal threshold achieving a target FPR on the validation set (never on test); `compute_category_breakdown()` — per-category AUC, precision, recall, F1 in a DataFrame; `plot_roc_pr_curves()` — side-by-side ROC and PR curve plots saved to disk. |
| [`train.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/train.py) | 333 | **Training loop module.** Implements: `get_rng_states()`/`set_rng_states()` for capturing and restoring Python/NumPy/PyTorch RNG states (needed for exact resume after local environment disconnect); `save_checkpoint()`/`load_checkpoint()` for full state persistence (model, optimizer, scheduler, AMP scaler, epoch, val_auc, config, RNG states); `train_one_epoch()` for processing MIL bag-pairs with AMP, gradient clipping, and loss decomposition tracking; `validate()` for scoring all clips per video and computing video-level AUC (used for early stopping, not MIL loss); `train()` main driver with AdamW optimizer, cosine annealing LR scheduler, CSV logging of all metrics per epoch, automatic resume from last checkpoint, and early stopping on validation AUC with configurable patience. |

### `training/dataset/` — Dataset & Transforms

| File | Lines | Description |
|---|---|---|
| [`__init__.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/dataset/__init__.py) | 2 | Minimal package init (placeholder comment). |
| [`transforms.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/dataset/transforms.py) | 186 | **Video transformation pipeline.** `VideoTransform` class with 4 modes: *train* (resize → random crop → random horizontal flip → random brightness/contrast jitter → normalize), *val/test* (resize → center crop → normalize), *inference* (resize to inference dims 112×144, no crop → normalize). Uses Kinetics-400 channel-wise mean/std. Also provides `decode_video_clips()` for OpenCV-based sliding-window clip extraction with FPS resampling, and `temporal_segment_sampling()` for Sultani-style MIL bag creation (divides video into 32 equal segments, picks one start index per segment with optional jitter, re-jittered every epoch). |
| [`mil_dataset.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/dataset/mil_dataset.py) | 310 | **MIL Bag Dataset module.** Contains: `UCFCrimeManifest` — loads a CSV manifest with video_path, label, category, split columns and provides lookup helpers; `MILBagDataset(Dataset)` — training dataset that returns (anomalous_bag, normal_bag) pairs where each bag is `(clips_per_bag, C, T, H, W)`, using segment sampling with retry logic for robustness; `MILBagValidationDataset(Dataset)` — validation dataset that extracts *all* clips from each video using a sliding window (stride 8) and returns `(clips_tensor, label, video_path, timestamps)`; `mil_collate_fn()` for stacking bag-pairs; `val_collate_fn()` for variable-length clip lists. Handles FPS resampling via frame stride calculation and pads short clips by repeating the last frame. |

### `training/model/` — Neural Network Architecture

| File | Lines | Description |
|---|---|---|
| [`__init__.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/model/__init__.py) | 4 | Exports `Scratch3DCNN` and `count_parameters`. |
| [`scratch_3dcnn.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/model/scratch_3dcnn.py) | 143 | **Custom 3D CNN model.** C3D-inspired architecture built from scratch (no pretrained weights). **Architecture:** Block 1 (Conv3D 3→64, BN, ReLU, MaxPool 1×2×2) → Block 2 (Conv3D 64→128, BN, ReLU, MaxPool 2×2×2) → Block 3 (two Conv3D 128→256→256, BN, ReLU, MaxPool 2×2×2) → Block 4 (two Conv3D 256→512→512, BN, ReLU, MaxPool 2×2×2) → AdaptiveAvgPool3D (1,1,1) → Dropout(0.6) → FC 512→128 → ReLU → Dropout(0.6) → FC 128→1. All 3×3×3 kernels with padding=1. Uses Kaiming He initialization. Outputs raw logits (no sigmoid — prevents saturation in MIL ranking loss). `forward_with_embeddings()` also returns the 512-dim feature vector. `count_parameters()` utility prints total/trainable parameter counts. Adaptive pooling allows variable spatial input sizes (112×112 for training, 112×144 for inference). |

### `inference/` — Inference & Postprocessing

| File | Lines | Description |
|---|---|---|
| [`__init__.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/inference/__init__.py) | 2 | Empty package init. |
| [`postprocess.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/inference/postprocess.py) | 155 | **Score postprocessing module.** `ema_smooth()` — exponential moving average filter for score smoothing (α=0.3). `hysteresis_threshold()` — state-machine based anomaly interval detection using enter/exit thresholds with run-length persistence (requires N=3 consecutive clips above threshold to enter, M=3 consecutive below to exit). `intervals_to_timestamps()` — converts clip-index intervals to timestamp intervals. `generate_score_csv()` — writes per-clip CSV with raw/smoothed scores and anomaly flag (`{name}_scores.csv`). `plot_score_timeline()` — matplotlib visualization of raw vs smoothed scores with red-shaded anomaly intervals (`{name}_score_plot.png`). |

### `docs/` — Documentation

| File | Lines | Description |
|---|---|---|
| [`implementation.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/docs/implementation.md) | 237 | Phased implementation guide (Phase 0–7) covering decisions lock, local scaffold + testing, local environment setup, data pipeline, training, evaluation, video inference, and integration/presentation. Includes team responsibility matrix, estimated weekly timeline, and Mermaid dependency flowchart. |

### `output/` — Generated Outputs

| File | Description |
|---|---|
| `output/pdf/MITS_SpatioTemporal_Anomaly_Detection_Milestone_Report.pdf` | Milestone report PDF for the capstone project. |

---

## Things To Be Implemented

### 🔲 Phase 0 — Environment Setup (Notebook `01_setup.ipynb`)

- [ ] Create local environment notebook that selects GPU runtime
- [ ] Mount local disk and create persistent folder structure (`data/`, `checkpoints/`, `runs/`, `outputs/`)
- [ ] Install and verify pinned packages (`opencv-python-headless`, `scikit-learn`, `matplotlib`)
- [ ] Freeze package versions to `requirements.txt`
- [ ] Set global random seed; log `environment.json` to Drive with package versions, GPU info, and seed

### 🔲 Phase 1 — Data Pipeline (Notebook `02_prepare_data.ipynb`)

- [ ] Pre-transcode videos to low-res, 10 fps H.264; copy to local environment local disk for fast I/O
- [ ] Scan UCF-Crime root and build manifest with video path, label, category, **file hash**
- [ ] Run quality pass — reject corrupted files and videos decoding to < 16 frames; log rejections with reasons
- [ ] Use **official UCF-Crime split** (290 test videos preserved); carve validation from official train set
- [ ] Visually spot-check decoded clips and confirm tensor shapes/timestamps
- [ ] **Deliverable:** `manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`, data-quality report

### 🔲 Phase 3 — Training (Notebook `03_train.ipynb`)

- [ ] Create local environment notebook that imports the local training modules
- [ ] Wire up the full training pipeline (data → model → loss → optimizer → checkpoints)
- [ ] Train on the UCF-Crime dataset (or an approved subset for debugging first)
- [ ] Log and save periodic/best checkpoints to local disk
- [ ] **Run with 3 different seeds** to report mean ± std
- [ ] **Deliverable:** `best_model.pt`, `last_model.pt`, training log CSV, `run_config.json`

### 🔲 Phase 4 — Evaluation & Threshold Calibration (Notebook `04_evaluate.ipynb`)

- [ ] Load the **frozen best checkpoint** only — never the last mid-training checkpoint
- [ ] Compute frame-level ROC-AUC and PR-AUC on official test set; run 3 seeds, report mean ± std
- [ ] Compute false alarms per video-hour on held-out normal videos
- [ ] **Calibrate threshold on validation split only**; freeze to `calibrated_threshold.json`
- [ ] Run frozen threshold once on test split for final metrics — never re-tune after seeing test results
- [ ] Break results by UCF-Crime category and by **failure-condition proxies**: mean brightness (low light), Laplacian variance (blur), global-motion magnitude (camera shake), person count (crowding)
- [ ] **Deliverable:** `metrics.json`, ROC/PR curves, `calibrated_threshold.json`, category-wise error breakdown

### 🔲 Phase 5 — Offline Inference & Visualization (Notebook `05_predict_video.ipynb`)

- [ ] Reuse exact Phase 1 inference preprocessing to score videos clip by clip
- [ ] Apply EMA smoothing + hysteresis thresholding with calibrated threshold
- [ ] **`inference/decoder.py`** — Video decoder module (OpenCV/FFmpeg) at target FPS
- [ ] **`inference/clip_assembler.py`** — Assemble decoded frames into 16-frame clips with stride 8
- [ ] **`inference/overlays.py`** — On-frame score overlay and colored border during flagged intervals
- [ ] **`inference/predict_video.py`** — End-to-end inference: decode → clip → score → postprocess → outputs
- [ ] **Deliverable:** per-video `{name}_scores.csv`, `{name}_score_plot.png`, `{name}_annotated.mp4`

### 🔲 Proposed Enhancements

#### Ablation Ladder (Must Include)

- [ ] Frame-differencing baseline (classical)
- [ ] Base model (max-MIL)
- [ ] Top-k MIL instead of max-only
- [ ] (2+1)D or depthwise-separable convolutions
- [ ] Added frame-difference motion channel
- [ ] Self-supervised pretraining (playback-speed or clip-order prediction)
- [ ] *(Optional)* Temporal head (temporal conv or self-attention) over clip embeddings
- [ ] *(Optional)* Pretrained frozen-feature reference as upper bound

#### Explainability & Failure Analysis

- [ ] **Shortcut analysis** — trivial classifier on camera quality, brightness, scene identity to detect dataset biases
- [ ] **3D Grad-CAM** overlay on flagged clips in annotated MP4
- [ ] Automatic failure-condition proxies (brightness, Laplacian variance, global-motion, person count)

#### Evaluation Rigor

- [ ] **Cross-dataset generalization** — run frozen model on ShanghaiTech Campus (report domain shift honestly)
- [ ] **AUC vs. FLOPs plot** — model variants with parameters, FLOPs, CPU clips/second (ONNX export)

#### Differentiation Strategy

- [ ] **Campus-safety use case** — scope to concrete deployment scenario
- [ ] **Category head + triage scoring** — 13-way head with severity tiers (*"possible fighting, 02:14–02:31, severity high, confidence medium"*)
- [ ] **Seed ensemble uncertainty** — leverage 3 seeds as deep ensemble; validate with risk-coverage curve
- [ ] *(Optional)* **Self-training** — use ensemble high-confidence agreement for pseudo segment-level labels

### 🔲 Testing (`tests/`)

- [ ] **Unit tests** for transforms (output shape `[3, 16, 112, 112]`, dtype, determinism with fixed seed)
- [ ] **Unit tests** for model forward pass on CPU (correct output shape, gradient flow)
- [ ] **Unit tests** for MIL loss (already has smoke test, needs proper pytest coverage)
- [ ] **Unit tests** for postprocessing (EMA, hysteresis state machine, edge cases)
- [ ] **Integration test** — train a tiny model on synthetic data, run inference, verify end-to-end
- [ ] **Parity test** — confirm training and inference transforms produce identical outputs for the same input

### 🔲 Miscellaneous

- [ ] Add a `requirements.txt` with pinned package versions (generated in Phase 0)
- [ ] Create a `notebooks/` directory with organized local environment notebooks
- [ ] Add `training/evaluate.py` as a standalone evaluation module (currently evaluation is embedded in `train.py` and `metrics.py`)
- [ ] Populate `inference/__init__.py` with proper exports
- [ ] Populate `training/dataset/__init__.py` with proper exports
- [ ] Consider adding TensorBoard logging support alongside CSV logging
- [ ] Update `docs/implementation.md` to align with the 6-phase plan

---

## Summary Statistics

| Metric | Value |
|---|---|
| **Total Python source files** | 10 |
| **Total lines of code** | ~1,795 |
| **Documentation files** | 4 (README, SYSTEM_ARCHITECTURE, implementation guide, summary) |
| **Implementation phases** | 6 (Phase 0–5) mapping to 5 local environment notebooks |
| **Phases completed** | Phase 2 (model & loss, core library — all importable modules) |
| **Phases remaining** | Phase 0 (env setup), Phase 1 (data pipeline), Phase 3 (training), Phase 4 (evaluation), Phase 5 (inference) |
| **Proposed enhancements** | Ablation ladder, Grad-CAM, shortcut analysis, cross-dataset eval, campus-safety use case, category head, seed ensemble |
| **Seeds for reporting** | 3 (mean ± std) |
| **Current focus** | Core library complete; ready for local environment integration |
