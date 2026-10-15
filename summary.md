# Project Summary — Dense Weakly Supervised Anomaly Localization with a Compact From-Scratch 3D CNN

> **Working Title:** Dense Weakly Supervised Anomaly Localization with a Compact From-Scratch 3D CNN: A Localization-Aware Shortcut Audit and Ensemble Uncertainty  
> **Base Repo:** Spatio-Temporal Video Anomaly Detection (MITS Capstone & Research Track)  
> **Date:** 14 Oct 2026 | **Companion File:** [`implementation_plan.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/implementation_plan.md)  
> **Research Question:** *How far can dense spatio-temporal anomaly localization emerge from weak video-level labels in a compact 3D CNN trained from scratch, how much does the model rely on non-event shortcuts, and can ensemble uncertainty enable risk-controlled triage?*

---

## Table of Contents

1. [Executive Overview & Novelty Positioning](#1-executive-overview--novelty-positioning)
2. [The Three Core Contributions](#2-the-three-core-contributions)
3. [What Has Been Implemented (Current Codebase Status)](#3-what-has-been-implemented-current-codebase-status)
4. [File-by-File Codebase Inventory](#4-file-by-file-codebase-inventory)
5. [Core Library Upgrades Required (Phases 2–3)](#5-core-library-upgrades-required-phases-23)
6. [Phased Implementation Plan (Phases 0–10)](#6-phased-implementation-plan-phases-010)
7. [Experiment Matrix (E0–E10)](#7-experiment-matrix-e0e10)
8. [16-Week Timeline & Go/No-Go Decision Gates](#8-16-week-timeline--gono-go-decision-gates)
9. [Deliverables Checklist](#9-deliverables-checklist)
10. [Risks, Fallback Framing, and Claims to Avoid](#10-risks-fallback-framing-and-claims-to-avoid)

---

## 1. Executive Overview & Novelty Positioning

A comprehensive literature check establishes that weakly supervised spatial localization on UCF-Crime is **not new**:
- **Landi et al. (2019):** Action tubes on UCFCrime2Local bounding boxes using pretrained backbones.
- **Liu & Ma (ACM MM 2019):** UCF-Crime box annotations and background-bias analysis with pretrained features.
- **WSSTAD (IJCAI 2021):** Weakly supervised tube localization using ST-UCF-Crime and proposal networks.
- **STPrompt (ACM MM 2024):** Training-free spatial localization leveraging frozen CLIP prompts.
- **Karim et al. (arXiv 2606.29498):** Patch-level MIL over a coarse $7 \times 7$ grid with frozen UniFormer backbones.
- **Localizing to Debias (arXiv 2608.12045) & Auditing Frame-Level AUC (arXiv 2608.11985):** Patch-level benchmarks, granularity, and scene bias audits.

### The Research Gaps Addressed
1. **From-Scratch Compact Localization:** All existing spatial localization work relies on heavy, pretrained, frozen feature extractors. Nobody studies whether dense localization emerges when training a compact 3D CNN **from scratch**, nor its CPU/FLOPs efficiency.
2. **Resolution Ladder:** Prior work limits maps to coarse $7 \times 7$ patches. A formal resolution ablation ($7 \times 7$ vs. $14 \times 14$ vs. $28 \times 28$ voxel maps) resolves accuracy vs. compute trade-offs.
3. **Localization-Aware Shortcut Audit:** Prior audits inspect frame-level AUC or generic background correlations. A **region-intervention audit** directly tests causality by deleting the ground-truth event region vs. an equal-area random background control.
4. **Spatio-Temporal Uncertainty:** Moves beyond clip-level evidential scores by computing voxel-wise epistemic uncertainty and risk-coverage curves for alert rejection.

> [!IMPORTANT]
> **Honest Scientific Claims:** We do not claim state-of-the-art frame AUC or TIoU. Pretrained foundation pipelines will outperform from-scratch models on raw numbers. Our contribution is an open, reproducible audit of compact from-scratch dense localization, efficiency, shortcuts, and uncertainty.

---

## 2. The Three Core Contributions

```
                                  [ SYSTEM PIPELINE ]
  Raw Video (UCF-Crime)
       │  [Phase 1: Transcode 10 fps, Manifests, Official Split]
       ▼
  16-Frame Clips (Stride 8)
       │  [Phases 2-3: DenseC3D From-Scratch + Voxel MIL Loss + AMP]
       ▼
  Voxel Logits [T', H', W']
       │  [Phase 6: Upsampling, Overlap-Averaging, EMA Temporal Smoothing]
       ▼
  Dense Heatmap [T, H, W]
   ├── Frame Anomaly Timeline    ──► Phase 4: ROC-AUC, PR-AUC, False Alarms/Hour
   ├── Spatio-Temporal Heatmaps  ──► Phase 7 (Contribution A): Pointing Game, Energy-in-Region,
   │                                           Pixel AUC, IoU, RBDR/TBDR vs. Grad-CAM & Center Prior
   ├── Region Interventions      ──► Phase 8 (Contribution B): Cue Classifiers, Anomalous-Only AUC,
   │                                           Deletion vs. Random Control, Mitigation
   └── 5-Seed Ensemble           ──► Phase 9 (Contribution D): Voxel Entropy/MI, Risk-Coverage,
                                               Rule-Based Triage
```

- **Contribution A: Dense Weakly Supervised Localization & Efficiency Study**
  - Fully convolutional 3D CNN (`DenseC3D`) outputting voxel logits `[T', H', W']` without bounding box proposals.
  - Voxel-level MIL ranking loss with spatial TV, temporal smoothness, sparsity, and normal-bag suppression.
  - Multi-resolution analysis (`g7`, `g14`, `g28`), (2+1)D convolutions, and motion channel ablations.
  - Profiling parameters, GFLOPs (`fvcore`), and throughput on GPU and CPU (`onnxruntime`).
- **Contribution B: Localization-Aware Shortcut Audit & Mitigation**
  - Cue-only classifiers (brightness, blur, resolution, bitrate, motion, frozen CLIP scene embeddings).
  - Three-AUC evaluation: standard test AUC, **anomalous-videos-only AUC** (eliminating source/camera bias), and per-category breakdown.
  - Region-intervention causality check: event region deletion vs. equal-area random background control.
  - Background reliance score and validation of a concrete mitigation (quality-matched augmentation / adversarial head).
- **Contribution D: Ensemble Uncertainty & Operational Triage**
  - 5-seed deep ensemble yielding voxel-level predictive entropy and mutual information (epistemic uncertainty).
  - Risk-coverage rejection curves for alert precision and pointing-game hit rate.
  - Rule-based operational severity triage (score, duration, uncertainty) producing actionable surveillance alerts.

---

## 3. What Has Been Implemented (Current Codebase Status)

The repository contains the foundation built during Phases 1–2 of the capstone project:

| Component | Status | Details in Codebase |
|---|---|---|
| **Configuration system** | ✅ Done | `ProjectConfig` dataclass in [`training/config.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/config.py) with JSON serialization, seed locking, paths |
| **Clip baseline model** | ✅ Done | `Scratch3DCNN` in [`training/model/scratch_3dcnn.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/model/scratch_3dcnn.py) (4 conv blocks, GAP, scalar MLP head) |
| **Clip-level MIL loss** | ✅ Done | `MILRankingLoss` in [`training/loss.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/loss.py) with max/top-k ranking, sparsity, and temporal smoothness |
| **Video transforms** | ✅ Done | `VideoTransform` in [`training/dataset/transforms.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/dataset/transforms.py) (train/val/inference modes, Kinetics-400 norm) |
| **MIL bag dataset** | ✅ Done | `MILBagDataset` & `MILBagValidationDataset` in [`training/dataset/mil_dataset.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/dataset/mil_dataset.py) |
| **Evaluation metrics** | ✅ Done | Clip/frame ROC-AUC, PR-AUC, false alarm rate, threshold calibration in [`training/metrics.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/metrics.py) |
| **Training engine** | ✅ Done | AdamW, cosine annealing, AMP, gradient clipping, resume-safe checkpoints in [`training/train.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/train.py) |
| **Postprocessing** | ✅ Done | EMA smoothing, hysteresis state-machine, score plotting in [`inference/postprocess.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/inference/postprocess.py) |
| **Data script** | ✅ Done | Video scanning and transcoding utility in [`prepare_data.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/prepare_data.py) |

---

## 4. File-by-File Codebase Inventory

### Project Root & Documentation
| File | Lines | Purpose |
|---|---|---|
| [`implementation_plan.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/implementation_plan.md) | 260 | The 10-phase master implementation plan, code templates, experiment matrix, and timeline |
| [`README.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/README.md) | 480 | Capstone overview, architecture diagrams, technical configuration, and deployment guide |
| [`SYSTEM_ARCHITECTURE.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/SYSTEM_ARCHITECTURE.md) | 390 | Engineering architecture, data contracts, ablation ladders, and verification gates |
| [`prepare_data.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/prepare_data.py) | 247 | Dataset processing script: transcode videos to 10 FPS, validate frame counts, emit manifests |
| [`train_runner.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/train_runner.py) | 85 | CLI runner invoking the training pipeline |
| [`milestone_4.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/milestone_4.md) | 69 | Core library milestone sign-off report |

### `training/` Package
| File | Lines | Purpose |
|---|---|---|
| [`training/config.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/config.py) | 238 | Hyperparameters dataclass, seed management (`set_seed`), device selection |
| [`training/loss.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/loss.py) | 115 | `MILRankingLoss`: Hinge ranking loss on clip logits, sparsity, temporal smoothness |
| [`training/metrics.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/metrics.py) | 172 | Frame/video ROC-AUC, PR-AUC, false alarms per hour, threshold calibration |
| [`training/train.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/train.py) | 333 | Full training loop, AMP, RNG state serialization, early stopping, validation driver |
| [`training/dataset/transforms.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/dataset/transforms.py) | 186 | Video spatial transforms (crop/flip/jitter/normalize) and segment sampling |
| [`training/dataset/mil_dataset.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/dataset/mil_dataset.py) | 310 | `UCFCrimeManifest`, `MILBagDataset` (32 segments), `MILBagValidationDataset` |
| [`training/model/scratch_3dcnn.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/training/model/scratch_3dcnn.py) | 143 | `Scratch3DCNN`: C3D-inspired 4-block architecture with Global Avg Pooling + scalar MLP |

### `inference/` Package
| File | Lines | Purpose |
|---|---|---|
| [`inference/postprocess.py`](file:///Users/prathmesh/Documents/Anomaly%20detection/inference/postprocess.py) | 155 | EMA filter, hysteresis thresholding state machine, CSV timeline generation, score plotting |

---

## 5. Core Library Upgrades Required (Phases 2–3)

Before launching full multi-resolution training, three code extensions must be added to the core library:

### 1. `training/model/dense_c3d.py` (Dense 3D CNN)
Replace Global Average Pooling and the MLP with a $1 \times 1 \times 1$ conv head outputting `[B, 1, T', H', W']`.
- **Temporal Downsampling:** 16 frames $\to T'=4$ (pooling in blocks 2 and 3).
- **Spatial Grid Options:**
  - `g7`: Stride 16 $\to 7 \times 7$ feature map from $112 \times 112$ input.
  - `g14`: Stride 8 $\to 14 \times 14$ feature map.
  - `g28`: Stride 4 $\to 28 \times 28$ feature map.
- **Top-$k$ Clip Aggregator:** Method `clip_logit(vox, k_frac=0.1)` computing mean top-$k$ voxel logits to produce clip-level scores for temporal evaluation.

### 2. `training/loss_dense.py` or updated `training/loss.py` (Voxel MIL Loss)
Voxel-level Multiple Instance Learning loss:
```python
def dense_mil_loss(vox_a, vox_n, k_frac=0.01, margin=1.0,
                   lam_sp=8e-5, lam_tv=8e-5, lam_norm=1e-2):
    # 1. Hinge ranking on top-k voxels (anomalous bag > normal bag)
    # 2. Voxel sparsity penalty on anomalous bags
    # 3. Spatial Total Variation (TV) + temporal smoothness
    # 4. Normal-bag suppression penalty (penalize any high normal voxel)
```

### 3. Four Core Smoke Tests (`tests/`)
- `test_dense_shapes.py`: Verify voxel map shapes for $112 \times 112$ (train) and $112 \times 144$ (inference) across `g7`, `g14`, `g28`.
- `test_dense_loss.py`: Verify loss is finite, gradient flows to block 1 convs, and loss zeroes out when positive top-$k$ is well separated from negative.
- `test_fully_conv.py`: Confirm weights run seamlessly across variable spatial resolutions without modification.
- `test_heatmap_projection.py`: Ensure a one-hot voxel maps accurately to expected pixel spatial bounds.

---

## 6. Phased Implementation Plan (Phases 0–10)

### Phase 0 — Decisions & Environment Lock
- [ ] Record fixed settings: clip length $T=16$, stride 8, target FPS 10, spatial size $112 \times 112$, margin $m=1.0$.
- [ ] Export pinned `requirements.txt` (`torch`, `torchvision`, `opencv-python-headless`, `fvcore`, `onnx`, `onnxruntime`, `sam2`).
- [ ] Verify local NVIDIA GPU environment with AMP.

### Phase 1 — Data Pipeline & Integrity Pass
- [ ] Transcode raw UCF-Crime videos to 10 FPS, lower resolution (OpenCV/FFmpeg).
- [ ] Generate manifests: `manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv` using official split (1610 train / 290 test).
- [ ] Carve deterministic validation split from training videos (never touching the 290 test videos).
- [ ] Quality pass: reject corrupted files and clips $< 16$ frames; record reasons in `data_quality_report.json`.

### Phase 2 — Dense Architecture & Voxel Loss
- [ ] Implement `DenseC3D` supporting `g7`, `g14`, and `g28` configurations.
- [ ] Implement `dense_mil_loss` with top-$k$ voxel hinge, spatial TV, temporal smoothness, and normal suppression.
- [ ] Implement proximity-aware top-$k$ voxel selection variant.
- [ ] Pass the 4 unit smoke tests in `tests/`.

### Phase 3 — Training Engine Additions
- [ ] Extend `training/train.py` to log: validation frame-level AUC, mean voxel sparsity, and border mass fraction (early shortcut warning).
- [ ] Train clip-level baseline (E0, E1) and initial dense models using 3 seeds (0, 1, 2).

### Phase 4 — Temporal Evaluation & Calibration
- [ ] Load frozen best checkpoint (selected strictly by validation AUC).
- [ ] Calibrate binary threshold on validation set targeting 5% FPR $\to$ freeze into `calibrated_threshold.json`.
- [ ] Compute official test set frame-level ROC-AUC, PR-AUC, and false alarm rate per hour (mean $\pm$ std).

### Phase 5 — Offline Inference Base
- [ ] Video sliding-window decoder (stride 8) generating clip sequence.
- [ ] EMA smoothing ($\alpha=0.35$) and hysteresis state-machine for interval demarcation.
- [ ] Emit per-video `{name}_scores.csv` and timeline plot `{name}_score_plot.png`.

### Phase 6 — Dense Spatio-Temporal Heatmap Pipeline (`inference/heatmaps.py`)
- [ ] Upsample voxel maps `[T', H', W']` bilinearly/trilinearly to full frame resolution $(H, W)$.
- [ ] Overlap-averaging accumulator across stride-8 sliding clips.
- [ ] Per-pixel temporal EMA smoothing ($\alpha=0.3$).
- [ ] Save output `{name}_heat.npz` (float16) and render overlay `{name}_annotated.mp4`.

### Phase 7 — Localization Evaluation (Contribution A)
- [ ] **Bounding Box GT:** Ingest and parse official test box annotations (UCFCrime2Local / ST-UCF-Crime).
- [ ] **Mini-Mask Benchmark:** Prompt SAM 2 using bounding boxes on 60–100 anomalous test videos (~10 frames/video); manually verify and correct in CVAT/labelme.
- [ ] **Threshold-Free Spatial Metrics:** Pointing game accuracy, Energy-in-region fraction, Pixel ROC-AUC / PR-AUC.
- [ ] **Thresholded Metrics:** Spatial IoU (fixed threshold at $0.5 \times \text{frame\_max}$), RBDR/TBDR ($\alpha=\beta=0.1$).
- [ ] **Baselines:** Evaluate Random map, Center-prior Gaussian, and 3D Grad-CAM on clip-level model.
- [ ] **Efficiency Profiling:** Parameters, GFLOPs (`fvcore`), GPU FPS, and CPU FPS (`onnxruntime`).

### Phase 8 — Localization-Aware Shortcut Audit (Contribution B)
- [ ] **Cue Extractors & Classifiers:** Extract mean brightness, blur (Laplacian variance), resolution, bitrate, motion magnitude, and frozen CLIP/DINOv2 scene embeddings. Train logistic regression & gradient boosted trees.
- [ ] **Three-AUC Analysis:** Compare Standard Test AUC vs. **Anomalous-Videos-Only AUC** vs. Category AUC.
- [ ] **Score-Cue Correlations:** Spearman correlation between model scores and quality cues.
- [ ] **Region-Intervention Experiment:**
  - Measure clip score drop when ground-truth event region is deleted (blurred/mean-filled).
  - Measure clip score drop when an equal-area random background region is deleted.
- [ ] **Background Reliance Score:** Compute $1 - \frac{\text{energy\_in\_region}}{1 - \text{area\_fraction}}$.
- [ ] **Mitigation:** Implement and ablate quality-matched augmentation or scene-adversarial GRL head.

### Phase 9 — Ensemble Uncertainty & Operational Triage (Contribution D)
- [ ] Train 5 seeds of the best model configuration.
- [ ] Compute voxel-level mean $\mu$, total predictive entropy $H$, and mutual information $I$ (epistemic uncertainty).
- [ ] Construct **risk-coverage curves**: reject top-uncertain alerts and measure alert precision and localization hit rate gains.
- [ ] Calibration assessment: ECE and reliability diagrams on validation clips.
- [ ] Implement rule-based operational triage: produce formatted alerts:
  `"Anomaly detected [02:14-02:31] | Severity: HIGH | Confidence: MEDIUM (MI: 0.18)"`.

### Phase 10 — Cross-Dataset Check & Reproducibility
- [ ] Run frozen best model on ShanghaiTech Campus (zero-shot transfer frame AUC).
- [ ] Prepare `REPRODUCE.md` with exact seeds, checkpoints, and runbook.
- [ ] Package mini-mask annotations for open-source release.

---

## 7. Experiment Matrix (E0–E10)

Every variant is selected strictly by **validation split AUC** before running test set evaluation:

| ID | Model / Variant | Description | Seeds | Evaluation Metrics |
|---|---|---|---|---|
| **E0** | Motion Differencing | Classical pixel frame-differencing baseline | n/a | Frame ROC-AUC |
| **E1** | Clip Baseline + Grad-CAM | Original `Scratch3DCNN` + 3D Grad-CAM | 3 | Frame AUC, Pointing Game |
| **E2** | Dense `g7` | Stride 16, $7 \times 7$ grid, top-$k$ voxel MIL | 3 | Frame AUC, Pointing Game, Pixel AUC, FLOPs |
| **E3** | Dense `g14` | Stride 8, $14 \times 14$ grid, top-$k$ voxel MIL | 3 | Frame AUC, Pointing Game, Pixel AUC, FLOPs |
| **E4** | Dense `g28` | Stride 4, $28 \times 28$ grid, top-$k$ voxel MIL | 3 | Frame AUC, Pointing Game, Pixel AUC, FLOPs |
| **E5** | Proximity Dense | Best grid + proximity-aware top-$k$ selection | 3 | Localization metrics, Energy-in-Region |
| **E6** | (2+1)D Dense | Best grid + factorized (2+1)D convolutions | 3 | Params, FLOPs, Frame AUC, Localization |
| **E7** | Motion Channel | Best grid + RGB + frame-diff motion channel | 3 | Frame AUC, Localization |
| **E8** | Bias Mitigation | Best model + quality-matched augmentation / GRL | 3 | Shortcut audit table, Anomalous-only AUC |
| **E9** | Deep Ensemble | 5 seeds of winning configuration | 5 | Entropy/MI maps, Risk-Coverage curves |
| **E10** | Pretrained Reference | Frozen features (I3D/UniFormer) + dense head | 3 | Upper-bound reference ceiling |
| *Opt*| Self-Supervised | Clip-order / playback-speed pretraining | 3 | Gap-closing ablation |

---

## 8. 16-Week Timeline & Go/No-Go Decision Gates

```
W1-W2: Literature, Box Ingestion, Transcoding, Smoke Tests
  └─► [Gate 1]: Confirm literature gap & obtain boxes
W3-W4: DenseC3D & Voxel MIL Loss, Baselines E0/E1
W5-W7: Multi-Resolution Grid Training E2, E3, E4
  └─► [Gate 2]: Pointing Game MUST outperform Center-Prior Baseline
W7-W8: SAM 2 Mini-Masks, Full Spatial Metrics, Efficiency Table
W9-W10: Ablations (E5-E8), Shortcut Audit & Mitigation
W11-W12: 5-Seed Ensemble, Uncertainty Maps, Risk-Coverage, Triage
W13: Pretrained Upper Bound (E10), ShanghaiTech Transfer
W14-W15: Paper Writing, Visual Figures, Reproducibility Audit
W16: Internal Review & Submission
```

### Critical Decision Gates
- **Gate 1 (Week 1): Literature & Ground Truth Availability**
  - Verify full texts of recent preprints (arXiv 2608.12045, 2608.11985). If from-scratch dense localization is already published, narrow focus to the localization-aware audit and efficiency profiling.
  - Confirm public downloadability of UCF-Crime bounding boxes. If inaccessible, pivot to annotating an in-house test subset immediately.
- **Gate 2 (Week 6): Center-Prior Sanity Gate**
  - The pointing game score of `g7`/`g14` must statistically beat a center-prior Gaussian heatmap. If from-scratch maps fail to beat the center prior, switch to the documented **fallback framing**: *"What from-scratch weak supervision can and cannot localize"*, carried by the audit, uncertainty, and released masks.

---

## 9. Deliverables Checklist

- [ ] Pinned `requirements.txt` and `environment.json`
- [ ] Transcoded videos and manifests (`manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`)
- [ ] Data quality report (`data_quality_report.json`)
- [ ] Core modules: `DenseC3D`, `dense_mil_loss`, passing unit smoke tests
- [ ] Checkpoints for E1–E10 saved to `./checkpoints/`
- [ ] Frame threshold calibration: `calibrated_threshold.json`
- [ ] SAM 2 mini-mask benchmark dataset with datasheet
- [ ] Dense spatio-temporal heatmap generator (`inference/heatmaps.py`)
- [ ] Efficiency analysis: parameter count, GFLOPs table, GPU/CPU throughput plots
- [ ] Shortcut audit report: cue classifiers, anomalous-videos-only AUC, region deletion results
- [ ] Ensemble uncertainty suite: entropy/MI maps, risk-coverage curves, triage policy
- [ ] ShanghaiTech cross-dataset evaluation report
- [ ] Paper draft and `REPRODUCE.md`

---

## 10. Risks, Fallback Framing, and Claims to Avoid

### Risk Mitigation Matrix
| Identified Risk | Mitigation Strategy |
|---|---|
| From-scratch maps fail to beat center prior | Reframe as an empirical limits study (*"What from-scratch weak supervision can and cannot localize"*). The audit (B), uncertainty (D), and mini-mask dataset provide solid workshop contributions. |
| Test boxes are inaccessible | Annotate 60–100 test videos in CVAT with bounding boxes, prompt SAM 2, and release the benchmark. |
| Frame AUC is inflated by background/camera cues | Report **anomalous-videos-only AUC** alongside standard AUC in all summary tables. |
| Overfitting to test split | Select all hyperparameters and thresholds strictly on validation; evaluate test set exactly once. |
| Dataset licensing constraints | Distribute code and annotation files only; never host raw UCF-Crime video files. |

### Claims Strictly Avoided
1. Do **not** claim state-of-the-art frame AUC or spatial localization against pretrained foundation models.
2. Do **not** claim "first ever" without rigorous full-text literature verification.
3. Do **not** claim "pixel ground truth segmentation" (masks are box-prompted and human-verified, not drawn from scratch).
4. Do **not** claim well-calibrated clip probabilities (labels are video-level; calibration is approximate).

---

## Summary Statistics

| Metric | Target Value |
|---|---|
| **Total Implementation Phases** | 11 (Phases 0–10) |
| **Total Planned Experiments** | 11 core runs (E0–E10) + 1 optional |
| **Grid Resolutions Evaluated** | 3 (`g7`: stride 16, `g14`: stride 8, `g28`: stride 4) |
| **Ensemble Seeds** | 5 seeds for epistemic uncertainty quantification |
| **Evaluation Metrics** | 8 (Frame AUC, Anomalous-only AUC, Pointing Game, Energy-in-Region, Pixel AUC, IoU, RBDR/TBDR, Risk-Coverage AU-RC) |
| **Total Project Timeline** | 16 weeks |
