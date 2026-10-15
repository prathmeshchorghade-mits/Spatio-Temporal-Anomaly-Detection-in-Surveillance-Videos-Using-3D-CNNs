# Project Decision Log

This document logs all major technical and architectural decisions made throughout the project's lifecycle.

---

### [DECISION-001] Transition to Local Training Environment
- **Date:** 2026-09-29
- **Status:** IMPLEMENTED
- **Category:** ARCHITECTURE
- **Files Changed:** `README.md`, `SYSTEM_ARCHITECTURE.md`, `docs/implementation.md`, `summary.md`
- **Reason:** The original plan relied on Google Colab and Google Drive for training and storage. Training the dataset locally is preferred for better control, stability, and unrestricted execution time.
- **Consideration:** Considered keeping the Colab workflow as a secondary option, but maintaining dual setups creates unnecessary overhead. Considered migrating to an enterprise cloud solution (AWS/GCP), but local execution is more straightforward for immediate capstone needs.
- **Decision:** Decided to fully strip out Google Colab and Google Drive references and rely entirely on local Python execution with local disk storage (`data/`, `checkpoints/`, `runs/`, `outputs/`).
- **Code Change Summary:** Updated all project markdown documentation to reflect local paths and local environment setups instead of Colab notebooks.

---

### [DECISION-002] Create NVIDIA GPU Environment Setup
- **Date:** 2026-09-29
- **Status:** IMPLEMENTED
- **Category:** CONFIG
- **Files Changed:** `requirements.txt`, `setup_env.py`
- **Reason:** With the move to local training, a reproducible environment setup is required specifically tailored for NVIDIA GPUs (CUDA), as 3D CNN training is computationally heavy.
- **Consideration:** Considered using Conda `environment.yml` but decided on a simpler `requirements.txt` paired with a Python setup script to avoid enforcing a specific package manager.
- **Decision:** Created a `requirements.txt` specifically pinning PyTorch with CUDA 12.1 (`cu121`), and a `setup_env.py` script to generate necessary project directories and verify CUDA availability.
- **Code Change Summary:** Created `requirements.txt` with PyTorch CUDA wheels and `setup_env.py` for directory scaffolding and environment logging.

---

### [DECISION-003] Clean and Rearrange Codebase
- **Date:** 2026-09-29
- **Status:** IMPLEMENTED
- **Category:** ARCHITECTURE
- **Files Changed:** `tmp/`, `output/`, `docs/milestone_report/`, `update_plan.py`
- **Reason:** The repository contained residual folders (`tmp/`, `output/`) from a previously generated PDF milestone report, as well as a temporary `update_plan.py` script. These cluttered the root directory.
- **Consideration:** Considered deleting the milestone report assets entirely since it is a generated artifact, but keeping them in `docs/milestone_report/` preserves project history.
- **Decision:** Consolidated all milestone report assets (PDF, images, generation script) into `docs/milestone_report/`. Deleted the `tmp/` and `output/` directories. Removed temporary python scripts.
- **Code Change Summary:** Moved files and deleted unnecessary root directories.

---

### [DECISION-004] Phase 0 — Lock Research-Track Operating Decisions
- **Date:** 2026-10-15
- **Status:** IMPLEMENTED
- **Category:** RESEARCH / CONFIG
- **Files Changed:** `docs/planning/decision.md`, `requirements.txt`
- **Reason:** Before any Phase 1+ implementation begins, every fixed operating parameter must be documented and frozen. This prevents post-hoc tuning on results and ensures reproducibility — a core requirement for research publication.

---

#### 0.1 Input Pipeline

| Parameter | Value | Rationale |
|---|---|---|
| **Clip length (T)** | 16 frames | Standard for C3D-family models; matches existing `ProjectConfig.CLIP_LENGTH` |
| **Clip stride (S)** | 8 frames | 50% overlap balances temporal coverage and compute; matches `ProjectConfig.CLIP_STRIDE` |
| **Target FPS** | 10 fps | UCF-Crime raw fps varies (24-30); 10 fps reduces storage 3× and is standard in MIL-VAD |
| **Resize (H×W)** | 128 × 171 | Preserves 4:3 aspect ratio before cropping |
| **Train crop** | 112 × 112 | Random spatial crop; standard C3D input size |
| **Inference spatial** | 112 × 144 | Full frame (no crop) — the fully convolutional `DenseC3D` supports arbitrary spatial dims |
| **Normalization** | Kinetics-400 mean/std (RGB) | `[0.43216, 0.394666, 0.37645]` / `[0.22803, 0.22145, 0.216989]` — pretrained norm kept for comparability |

#### 0.2 Model Architecture (DenseC3D)

| Parameter | Value | Rationale |
|---|---|---|
| **Backbone** | 4 conv3d blocks (3→32→64→128→256), BN, ReLU | Compact from-scratch design; no pretrained weights |
| **Head** | 1×1×1 Conv3d → 1 channel (voxel logit) | Fully convolutional; no global pooling, no FC layers |
| **Temporal pooling** | Blocks 2 & 3 only (16 → 4 output temporal frames) | Preserves temporal resolution for localization |
| **Grid variants** | `g7` (stride 16), `g14` (stride 8), `g28` (stride 4) | Resolution ablation study; pool specs differ per variant |
| **Dropout** | None in dense head (only 1×1 conv) | Dense MIL loss provides implicit regularization |
| **Weight init** | Kaiming He (fan_out, ReLU) | Standard for conv nets trained from scratch |
| **Fully convolutional property** | Same weights run on 112×112 (train) and 112×144 (inference) | No spatial-size dependency; enables multi-resolution eval |

**Pool configurations (frozen):**

| Variant | Block 1 pool | Block 2 pool | Block 3 pool | Block 4 pool | Output map (112×112) | Output map (112×144) |
|---|---|---|---|---|---|---|
| `g7` | (1,2,2) | (2,2,2) | (2,2,2) | (1,2,2) | 4×7×7 | 4×7×9 |
| `g14` | (1,2,2) | (2,2,2) | (2,2,2) | None | 4×14×14 | 4×14×18 |
| `g28` | (1,2,2) | (2,2,2) | None | None | 4×28×28 | 4×28×36 |

#### 0.3 Loss Function (Dense Voxel MIL)

| Parameter | Value | Rationale |
|---|---|---|
| **Ranking margin (m)** | 1.0 (dense), 0.5 (clip baseline) | Logit-space hinge; 1.0 avoids sigmoid saturation for dense maps |
| **Top-k fraction (k_frac)** | 0.01 (1% of voxels) | Sparse anomaly prior: anomalies occupy a small fraction of the scene |
| **λ_sparse** | 8e-5 | Penalizes mean sigmoid activation of anomalous maps |
| **λ_tv (smoothness)** | 8e-5 | Spatial total variation + temporal segment difference |
| **λ_norm** | 1e-2 | Penalizes sigmoid² of normal-bag voxels toward zero |
| **Hinge target** | Logits (not probabilities) | Avoids gradient vanishing from sigmoid saturation |

#### 0.4 Training Protocol

| Parameter | Value | Rationale |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay; standard for small models |
| **Learning rate** | 1e-4 | Matches existing config; validated in capstone baseline |
| **Weight decay** | 5e-5 | Light regularization; heavier risks underfitting small model |
| **LR schedule** | Cosine annealing to 0 | Smooth decay; no warmup (small model converges fast) |
| **Max epochs** | 60 | Early stopping usually triggers before 60 |
| **Early stopping patience** | 12 epochs | On validation frame-level AUC |
| **Batch size** | 4 bag-pairs | 1 anomalous + 1 normal per pair; 4 pairs per step |
| **Clips per bag** | 32 segments | Uniformly sampled from video; matches existing MIL pipeline |
| **AMP** | Enabled (float16) | 1.5–2× throughput on Ampere+ GPUs |
| **Gradient clipping** | Max norm = 1.0 | Prevents exploding gradients from hinge loss |
| **Seeds (ablation)** | 0, 1, 2 | 3 seeds per experiment row E0–E8 |
| **Seeds (ensemble)** | 0, 1, 2, 3, 4 | 5 seeds for Phase 9 ensemble uncertainty |

#### 0.5 Data Protocol

| Parameter | Value | Rationale |
|---|---|---|
| **Dataset** | UCF-Crime (official split) | 1610 train videos, 290 test videos |
| **Validation carving** | ~15% from training set (stratified) | Never touch test until all decisions are frozen |
| **Test protocol** | Single evaluation after full freeze | Prevents test-set overfitting |
| **Localization GT** | Box annotations (Liu & Ma 2019 / ST-UCF-Crime) | Week-1 task: confirm public availability |
| **Mini pixel-mask set** | 60–100 videos, ~10 frames each, SAM 2-prompted, human-verified | Your own contribution; released as annotations only |
| **Cross-dataset** | ShanghaiTech Campus (frame-level AUC) | Frozen model, no fine-tuning; report drop honestly |

#### 0.6 Inference & Post-Processing

| Parameter | Value | Rationale |
|---|---|---|
| **EMA alpha** | 0.3 | Temporal smoothing of per-frame anomaly scores |
| **Hysteresis enter threshold** | 0.5 | Score above which an anomaly event opens |
| **Hysteresis exit delta** | 0.1 | Event closes when score < (0.5 − 0.1) = 0.4 |
| **Hysteresis run length** | 3 consecutive clips | Persistence filter to suppress single-clip noise |
| **Heatmap upsampling** | Bilinear to frame resolution | From voxel map [T',H',W'] to [T,H,W] |
| **Localization threshold** | 0.5 × per-frame max (fixed rule, no tuning) | Threshold-free metrics are primary |

#### 0.7 Evaluation Metrics (Frozen)

**Frame-level detection (primary):**
- ROC-AUC on official test set (mean ± std over 3 seeds)
- PR-AUC on official test set
- Anomalous-videos-only AUC (source confound control)

**Localization (primary, threshold-free):**
- Pointing game accuracy
- Energy-in-region ratio
- Pixel-AUC on annotated anomalous frames

**Localization (secondary, fixed threshold):**
- IoU / TIoU at 0.5 × per-frame max
- RBDR / TBDR (Ramachandra & Jones, α=β=0.1)

**Efficiency:**
- Parameters, FLOPs (fvcore), GPU clips/s, CPU clips/s (ONNX Runtime)
- Localization quality vs. FLOPs plot

**Uncertainty (Phase 9):**
- Risk-coverage AUC
- ECE and reliability diagram
- Spearman(per-frame mean voxel MI, 1 − IoU)

#### 0.8 Experiment Matrix (Frozen IDs)

| ID | Variant | Seeds | Locked? |
|---|---|---|---|
| E0 | Frame-differencing baseline | n/a | ✅ |
| E1 | Clip-level max-MIL + Grad-CAM | 3 | ✅ |
| E2 | Dense `g7`, top-k MIL | 3 | ✅ |
| E3 | Dense `g14` | 3 | ✅ |
| E4 | Dense `g28` | 3 | ✅ |
| E5 | Best grid + proximity-aware top-k | 3 | ✅ |
| E6 | Best + (2+1)D convolutions | 3 | ✅ |
| E7 | Best + frame-difference motion channel | 3 | ✅ |
| E8 | Best + augmentation/adversarial mitigation | 3 | ✅ |
| E9 | Best config, 5-seed ensemble | 5 | ✅ |
| E10 | Frozen pretrained features + dense head | 3 | ✅ |

#### 0.9 Environment Freeze

- **Python:** 3.10+
- **PyTorch:** 2.2.0+cu121 (pinned)
- **New research dependencies added to `requirements.txt`:**
  - `fvcore>=0.1.5` — FLOPs profiling
  - `onnx>=1.14.0` — model export
  - `onnxruntime>=1.16.0` — CPU throughput benchmarking
  - `pytest>=7.4.0` — smoke tests and CI
  - `pyyaml>=6.0` — config serialization
  - `numpy>=1.24.0` — pinned floor
  - SAM 2 — installed separately at Phase 7 (inference only, not a training dependency)

#### 0.10 Guiding Principles

1. **Validation-only model selection.** The "best" configuration is always chosen by validation AUC. Test set is touched exactly once, after all choices are frozen.
2. **No novelty over-claim.** Dense weakly supervised localization exists. Our claim is: reproducible study with compact from-scratch models, localization-aware audit, and ensemble uncertainty.
3. **Report negatives honestly.** If from-scratch maps are barely above center-prior, reframe as "what from-scratch weak supervision can and cannot localize."
4. **Annotations-only release.** UCF-Crime videos are not redistributed; only code, configs, checkpoints, and annotation files are released.
5. **Deterministic seeds.** Every run logs its seed, RNG state snapshot, and config JSON alongside the checkpoint.

- **Consideration:** Considered starting with a higher margin (m=2.0) for the dense loss, but 1.0 is already well-separated in logit space and avoids pushing the model toward extreme logit magnitudes early in training. Also considered k_frac=0.05 (5%) but 1% better reflects the spatial sparsity of real anomalies. Both can be revisited as ablation rows but the defaults are frozen here.
- **Decision:** All operating parameters above are LOCKED. Any changes require a new DECISION entry with explicit justification. Phase 1+ implementation may now proceed against these constants.
- **Code Change Summary:** Updated `decision.md` with full Phase 0 specification. Updated `requirements.txt` with research-track dependencies (fvcore, onnx, onnxruntime, pytest, pyyaml, numpy floor pin).
