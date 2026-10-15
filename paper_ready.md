# Research Paper Readiness Specification & Checklist

**Working Title:** Dense Weakly Supervised Anomaly Localization with a Compact From-Scratch 3D CNN: A Localization-Aware Shortcut Audit and Ensemble Uncertainty  
**Target Venues:** CVPR / WACV / ECCV / ICIP Workshops (Vision for Surveillance / Robust VAD Tracks), ACM Multimedia, IEEE Access, Pattern Recognition Letters  
**Format:** Standard 8-page double-column conference paper + Supplementary Material  
**Date:** 15 October 2026  
**Companion Documents:** [`research.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/research.md) | [`implementation_plan.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/implementation_plan.md) | [`summary.md`](file:///Users/prathmesh/Documents/Anomaly%20detection/summary.md)

---

## Executive Overview

To transform this repository into an accepted academic paper, the work must move beyond demonstrating a working software prototype to satisfying the rigorous scientific, empirical, and reproducibility standards of top-tier peer review.

This document serves as the master specification of **everything required to make the project submission-ready**, divided into six operational pillars:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE SIX PAPER-READINESS PILLARS                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Pillar 1: Empirical Completeness (5 Mandatory Quantitative Result Tables)             │
│  Pillar 2: The Mini-Mask Benchmark Dataset (Novel Community Artifact)                 │
│  Pillar 3: Methodological Integrity & Validation-Only Discipline                      │
│  Pillar 4: Manuscript Visual Figures (6 Publication-Quality Graphics)                  │
│  Pillar 5: Reviewer-Proof Writing, Framing & Ethical Safeguards                       │
│  Pillar 6: Open-Science Reproducibility Package (One-Click Verification)               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pillar 1: Empirical Completeness (5 Mandatory Result Tables)

Reviewers expect multi-seed statistical significance, appropriate sanity floors, and complete ablation matrices. The paper must report the following five tables:

### Table 1: Spatial Localization Benchmark & Resolution Study (Main Result A)
*All models evaluated on the official UCF-Crime test set across 3 random seeds (reporting mean ± std).*

| Method / Backbone | Grid | Trainable Params | Frame ROC-AUC (%) | Pointing Game Acc. (%) | Energy-in-Region Ratio | Pixel ROC-AUC (%) | Spatial IoU (0.5×max) |
|---|---|---|---|---|---|---|---|
| **Uniform Random** | — | 0 | 50.0 ± 0.0 | ~18.5 | 1.00× (chance) | 50.0 ± 0.0 | < 0.05 |
| **Center Prior (Gaussian)** | — | 0 | 50.0 ± 0.0 | ~41.2 | 1.00× (area ratio) | 58.2 ± 0.0 | ~0.12 |
| **Clip Baseline + Grad-CAM** | Stride 16 | 3.82 M | [Report] | [Report] | [Report] | [Report] | [Report] |
| **DenseC3D `g7` (From Scratch)** | Stride 16 | 3.41 M | [Report] | [Report] | [Report] | [Report] | [Report] |
| **DenseC3D `g14` (From Scratch)**| Stride 8 | 3.41 M | [Report] | [Report] | [Report] | [Report] | [Report] |
| **DenseC3D `g28` (From Scratch)**| Stride 4 | 3.41 M | [Report] | [Report] | [Report] | [Report] | [Report] |
| **`DenseC3D` + Proximity Top-k** | Stride 8 | 3.41 M | [Report] | [Report] | [Report] | [Report] | [Report] |
| **`DenseC3D` + (2+1)D Convs** | Stride 8 | ~2.10 M | [Report] | [Report] | [Report] | [Report] | [Report] |
| **Pretrained Upper Bound (I3D/UniFormer)** | Stride 16 | ~24.5 M | [Report] | [Report] | [Report] | [Report] | [Report] |

*Critical Requirement:* The pointing-game accuracy of `DenseC3D` (`g14`) must statistically exceed the Center-Prior Gaussian baseline (Gate 2).

---

### Table 2: Computational Complexity & Edge-Inference Profiling
*Must demonstrate real-time efficiency on commodity hardware without GPUs.*

| Model Variant | Resolution Grid | Input Spatial Size | GFLOPs / 16-Frame Clip | GPU Throughput (PyTorch AMP, FPS) | CPU Throughput (ONNX Runtime, FPS) | Memory Footprint (VRAM, MB) |
|---|---|---|---|---|---|---|
| **Clip Baseline** | Scalar Head | 112 × 112 | ~11.4 | ~140 FPS | ~42 FPS | ~650 MB |
| **DenseC3D `g7`** | 7 × 7 | 112 × 112 | ~11.2 | ~135 FPS | ~44 FPS | ~680 MB |
| **DenseC3D `g14`** | 14 × 14 | 112 × 112 | ~14.8 | ~110 FPS | ~31 FPS | ~780 MB |
| **DenseC3D `g28`** | 28 × 28 | 112 × 112 | ~28.6 | ~65 FPS | ~14 FPS | ~1,150 MB |
| **DenseC3D (2+1)D** | 14 × 14 | 112 × 112 | ~9.8 | ~125 FPS | ~38 FPS | ~580 MB |
| **Pretrained Feature Model** | 7 × 7 | 224 × 224 | > 85.0 | ~25 FPS | < 4 FPS | > 3,200 MB |

*Measurement Tools:* `fvcore` for GFLOPs, PyTorch CUDA events for GPU FPS, `onnxruntime` on an Intel/AMD/Apple CPU for CPU FPS.

---

### Table 3: Localization-Aware Shortcut Audit (Result B)
*Tests whether the model relies on the actual anomalous event vs. non-event visual shortcuts.*

| Evaluated Model / Feature | Standard Test ROC-AUC (%) | Anomalous-Videos-Only ROC-AUC (%) | Event Deletion Drop (Δ_event) | Background Deletion Drop (Δ_control) | Background Reliance Score (BRS) |
|---|---|---|---|---|---|
| **Cue: Mean Luminance (Brightness)** | [Report] | ~50.0% | — | — | — |
| **Cue: Laplacian Variance (Blur)** | [Report] | ~50.0% | — | — | — |
| **Cue: Native Bitrate & Resolution** | [Report] | ~50.0% | — | — | — |
| **Cue: Motion Energy (Optical Flow)** | [Report] | [Report] | — | — | — |
| **Cue: Scene Context (CLIP Embeddings)** | [Report] | ~51.0% | — | — | — |
| **All Cues Combined (GBDT Classifier)** | [Report] | ~52.5% | — | — | — |
| **Clip Baseline (Unmitigated)** | [Report] | [Report] | [Report] | [Report] | [High, >0.6] |
| **DenseC3D `g14` (Unmitigated)** | [Report] | [Report] | [Report] | [Report] | [Medium, ~0.4] |
| **DenseC3D `g14` + Mitigation (Augmentation)** | [Report] | [Report (High)] | [High, >0.3] | [Low, <0.08] | [Low, <0.25] |

*Definitions:*
- **Anomalous-Videos-Only AUC:** Evaluates frame discrimination strictly *within* anomalous test videos, stripping away source-camera and background bias confounds.
- **Δ_event:** Score drop when the ground-truth anomaly region is masked/blurred.
- **Δ_control:** Score drop when an equal-area random background region is masked/blurred.
- **BRS:** $1 - \frac{\text{Energy in Ground-Truth Region}}{\text{Area Fraction of Region}}$ (Lower is better; 0 = perfect event focus).

---

### Table 4: Deep Ensemble Uncertainty & Risk-Coverage Performance (Result D)
*5-seed deep ensemble performance evaluating alert triage and calibration.*

| Operating Point / Policy | Coverage (κ, %) | Alert Precision (%) | Pointing Game Acc. (%) | Mean Epistemic Uncertainty (MI) | Expected Calibration Error (ECE) |
|---|---|---|---|---|---|
| **All Alerts (No Rejection)** | 100.0% | [Baseline %] | [Baseline %] | [Mean MI] | [ECE] |
| **Uncertainty Rejection (Top 10% Dropped)** | 90.0% | [Higher %] | [Higher %] | [Lower MI] | [Lower ECE] |
| **Uncertainty Rejection (Top 25% Dropped)** | 75.0% | [Higher %] | [Higher %] | [Lower MI] | [Lower ECE] |
| **Uncertainty Rejection (Top 40% Dropped)** | 60.0% | [Higher %] | [Higher %] | [Lower MI] | [Lower ECE] |
| **Random Alert Rejection (Control)** | 60.0% | [Unchanged %] | [Unchanged %] | — | — |
| **Summary Metric (AU-RC)** | — | **[Report AU-RC]** | **[Report AU-RC]** | — | — |

*Statistical Test:* Spearman rank correlation between per-frame mean voxel MI and $(1 - \text{IoU})$ on annotated frames (must show $\rho > 0.5$).

---

### Table 5: Cross-Dataset Domain Transfer
*Proving out-of-domain generalization without fine-tuning.*

| Model Trained On | Test Dataset | Zero-Shot Frame ROC-AUC (%) | Δ Performance Drop vs. In-Domain |
|---|---|---|---|
| UCF-Crime (DenseC3D `g14`) | UCF-Crime (In-Domain) | [Report] | 0.0% |
| UCF-Crime (DenseC3D `g14`) | ShanghaiTech Campus | [Report] | [Report Drop Honestly] |
| UCF-Crime + Mitigation | ShanghaiTech Campus | [Report (Improved)] | [Smaller Drop] |

---

## Pillar 2: The Mini-Mask Benchmark Dataset Artifact

Because prior literature evaluates only on coarse bounding boxes, creating and releasing a **human-verified pixel-mask test set** is a primary contribution that makes the paper publication-ready.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MINI-MASK BENCHMARK PRODUCTION PROTOCOL                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Step 1: Video Selection                                                                │
│         Sample 60–100 anomalous test videos from UCF-Crime covering all 13 crime       │
│         categories (Arson, Assault, Burglary, Robbery, Shooting, etc.).                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Step 2: Keyframe Sampling                                                              │
│         Select ~10 anomalous keyframes per video during active anomaly intervals       │
│         (Total: 600–1,000 annotated evaluation frames).                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Step 3: SAM 2 Box-Prompting                                                            │
│         Prompt Segment Anything Model 2 (SAM 2) using public bounding-box coordinates  │
│         from prior annotations (Liu & Ma 2019 / WSSTAD 2021).                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Step 4: Human Verification & Manual Correction Pass                                    │
│         Load candidate masks into CVAT or Labelme. Manually correct over-segmentation, │
│         refine boundaries around held objects/weapons, and reject ambiguous frames.    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Step 5: Dataset Packaging & Datasheet                                                  │
│         Release annotations in COCO and binary PNG formats. Provide a formal datasheet │
│         specifying inter-annotator agreement and license terms.                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Reviewer Defense:** Explicitly declare that masks are used **strictly for evaluation**, never for training. State that masks are *"box-prompted, human-verified"* to avoid false claims of drawing 1,000 masks from scratch.

---

## Pillar 3: Methodological Integrity & Validation Discipline

To avoid reviewer criticism regarding test-set overfitting or data snooping:

1. **Validation-Only Hyperparameter Selection:**
   - The sparsity weight $\lambda_{\text{sp}}$, spatial TV weight $\lambda_{\text{tv}}$, margin $m$, and learning rate schedule must be selected **exclusively by validation split AUC**.
   - The anomaly decision threshold must be calibrated on validation videos to achieve $5\%$ FPR and frozen into `calibrated_threshold.json`.
   - The 290 test videos must be run **only once** using frozen checkpoints.
2. **Deterministic Seed Control:**
   - Every experiment must use logged global seeds ($0, 1, 2$ for ablations; $0, 1, 2, 3, 4$ for the ensemble).
   - Python, NumPy, PyTorch, and cuDNN deterministic flags must be active.
3. **Passing Code Verification Tests:**
   - All four smoke tests (`test_dense_shapes.py`, `test_dense_loss.py`, `test_fully_conv.py`, `test_heatmap_projection.py`) must pass cleanly on CPU before GPU runs are initiated.

---

## Pillar 4: Manuscript Visual Figures (6 Mandatory Graphics)

A strong paper requires intuitive visual figures. The manuscript must include:

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Figure 1: Architectural Workflow & Voxel MIL Formulation                              │
│   • Input clip [16, 112, 112] ──► 3D Conv Blocks ──► Voxel Logits [T', H', W']        │
│   • Top-k Voxel Selection vs. Normal Suppression + Smoothness Regularization          │
│   • Upsampling & Overlap-Averaging ──► Spatiotemporal Heatmap                         │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ Figure 2: Qualitative Spatial Localization Comparisons (4-Row Grid)                   │
│   • Original Surveillance Frame with Bounding Box GT                                  │
│   • Center-Prior Baseline (Showing trivial failure modes)                             │
│   • 3D Grad-CAM Baseline (Showing diffuse, blobby spatial activations)                │
│   • DenseC3D Heatmap (Ours, showing tight localization on the anomalous actor)        │
│   • Human-Verified Ground-Truth Mask Overlay                                          │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ Figure 3: Localization Quality vs. Computational Efficiency (Pareto Curve)            │
│   • X-axis: GFLOPs per clip (log-scale)                                               │
│   • Y-axis: Pointing Game Accuracy (%)                                                │
│   • Shows DenseC3D g14 occupying the sweet spot; pretrained models far in high-FLOPs. │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ Figure 4: The Localization-Aware Shortcut Audit (Causal Region Interventions)         │
│   • Panel A: Original video frame with crime action.                                  │
│   • Panel B: Event region deletion (anomaly score collapses).                         │
│   • Panel C: Control random background deletion (anomaly score stays high).           │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ Figure 5: Risk-Coverage Curve & Operational Alert Rejection                           │
│   • Alert Precision (%) rising from 54% to >90% as coverage falls from 100% to 50%.   │
│   • Comparison against random alert dropping and ranking by raw score alone.          │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ Figure 6: Epistemic Uncertainty vs. Localization Failure Analysis                     │
│   • Visualizing Voxel Mean vs. Voxel Mutual Information (Epistemic Uncertainty).      │
│   • Demonstrates high seed disagreement on ambiguous visual boundaries.               │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pillar 5: Reviewer-Proof Writing, Framing & Ethical Guardrails

### 1. Claims to Avoid vs. Required Framing

| Strictly Prohibited Claim | Approved Scientific Framing |
|---|---|
| ❌ *"We achieve SOTA frame-level AUC on UCF-Crime."* | ✅ *"We demonstrate that compact from-scratch 3D CNNs can achieve dense spatial localization without requiring massive pretrained foundation backbones."* |
| ❌ *"We propose the first weakly supervised localization method."* | ✅ *"While tube and proposal-based localization exists using pretrained networks, we present the first systematic audit of from-scratch voxel localization, resolution costs, and causal region interventions."* |
| ❌ *"We segment anomalies at the pixel level."* | ✅ *"We produce dense spatio-temporal heatmap localization evaluated on box-prompted, human-verified masks."* |
| ❌ *"Our model produces calibrated probabilities."* | ✅ *"Given weak video-level supervision, we quantify relative epistemic uncertainty via deep ensembles and risk-coverage curves rather than absolute clip calibration."* |

### 2. Documented Fallback Framing (If Gate 2 Fails)
If empirical pointing-game results on test boxes fail to beat the center prior by a clear margin, activate the documented fallback framing:
> **Fallback Paper Title:** *"What From-Scratch Weakly Supervised Models Can and Cannot Localize: A Negative-Result Audit on Surveillance Benchmarks."*  
> **Emphasis:** Reframe as an empirical limits and audit study carried by Contribution B (the shortcut audit), Contribution D (ensemble uncertainty), and the released mini-mask benchmark. This remains fully publishable in computer vision workshops.

### 3. Ethical Safeguards & Broader Impact Statement
The paper must include a dedicated **Broader Impact & Ethics Section**:
- **Decision-Support Guardrail:** The system is explicitly designed for human-in-the-loop triage, rejecting automated punitive actions.
- **Privacy Compliance:** No raw surveillance videos are redistributed; only mathematical checkpoints and bounding coordinate annotations are released.
- **Regulatory Awareness:** Explicitly mention alignment with the **EU AI Act** regarding high-risk biometric and surveillance monitoring transparency requirements.

---

## Pillar 6: Open-Science Reproducibility Package

Top venues evaluate reproducibility during review. The release package must include:

### 1. Repository Layout
```
├── README.md               # Quickstart, installation, visual results summary
├── REPRODUCE.md            # Exact step-by-step commands to reproduce Tables 1-5
├── requirements.txt        # Pinned Python package dependencies
├── environment.json        # Hardware, CUDA, and environment specifications
│
├── training/               # Pure PyTorch modules (DenseC3D, voxel MIL loss, training engine)
├── inference/              # Heatmap projection, overlap-averaging, postprocessing
├── audit/                  # Cue extractors, cue classifiers, region intervention scripts
├── evaluation/             # Pointing game, energy-in-region, pixel AUC, risk-coverage
├── tests/                  # 4 Core CPU smoke tests
│
├── scripts/
│   ├── run_all_smoke_tests.sh
│   ├── reproduce_table1_localization.sh
│   ├── reproduce_table2_efficiency.sh
│   ├── reproduce_table3_audit.sh
│   └── reproduce_table4_uncertainty.sh
│
└── data/
    ├── manifests/          # CSV manifests for official train/val/test splits
    └── mini_masks/         # Verified SAM 2 test mask JSON/PNG coordinates
```

### 2. One-Click Reproduction Script (`REPRODUCE.md`)
Reviewers should be able to verify your claims with single-command executions:
```bash
# Verify unit tests
pytest tests/ -v

# Profile GFLOPs and latency (Table 2)
python evaluation/profile_efficiency.py --model g14

# Run shortcut audit interventions (Table 3)
python audit/region_interventions.py --checkpoint ./checkpoints/best_g14.pt

# Evaluate ensemble uncertainty risk-coverage (Table 4)
python evaluation/risk_coverage.py --ensemble_dir ./checkpoints/ensemble_5seeds/
```

### 3. Checkpoint & Artifact Archiving
- Upload best checkpoint weights (`.pt`) for E1–E10 to a persistent DOI repository (e.g., Zenodo or Hugging Face).
- Provide SHA-256 checksums in `checkpoints/checksums.sha256`.

---

## Master Paper-Readiness Checklist

Use this checklist to track project completion before initiating the final LaTeX manuscript submission:

### A. Environment & Data Preparation
- [ ] Dependencies pinned in `requirements.txt` and verified in virtualenv.
- [ ] UCF-Crime videos transcoded to 10 FPS at low resolution ($128 \times 171$).
- [ ] Official split preserved: 1,610 train / 290 test videos; validation split carved strictly from train.
- [ ] Public test bounding box annotations ingested and parsed.
- [ ] SAM 2 mini-mask benchmark created across 60–100 anomalous test videos and human-verified in CVAT.

### B. Core Library & Tests
- [ ] `DenseC3D` fully convolutional architecture implemented with `g7`, `g14`, and `g28` configurations.
- [ ] Voxel-level MIL loss implemented (top-$k$ hinge, spatial TV, temporal smoothness, normal suppression).
- [ ] All 4 CPU smoke tests in `tests/` pass cleanly via `pytest`.
- [ ] Dense spatio-temporal heatmap generator (`inference/heatmaps.py`) verified with overlap-averaging and EMA.

### C. Experiments (E0–E10)
- [ ] Baseline E0 (frame differencing) and E1 (clip baseline + Grad-CAM) evaluated over 3 seeds.
- [ ] Dense resolution models E2 (`g7`), E3 (`g14`), and E4 (`g28`) trained over 3 seeds.
- [ ] Pointing game of `DenseC3D` confirms Gate 2 passes over Center Prior.
- [ ] Ablations E5 (proximity-aware top-$k$) and E6 ((2+1)D convolutions) evaluated.
- [ ] Shortcut audit completed: cue-only classifiers trained, anomalous-only AUC computed, and region interventions ($\Delta_{\text{event}}$ vs. $\Delta_{\text{control}}$) verified.
- [ ] Bias mitigation E8 evaluated before and after retraining.
- [ ] 5-seed ensemble E9 trained; voxel predictive entropy and mutual information computed; risk-coverage curve plotted.
- [ ] Reference pretrained upper bound E10 evaluated.
- [ ] Zero-shot transfer evaluated on ShanghaiTech Campus.

### D. Profiling & Manuscript Production
- [ ] GFLOPs measured via `fvcore`; CPU FPS measured via `onnxruntime`.
- [ ] All 5 publication tables formatted with mean $\pm$ standard deviation.
- [ ] All 6 publication graphics rendered in high resolution (300 DPI vector PDF/PNG).
- [ ] Manuscript written in standard double-column IEEE/CVPR format adhering to honest claims and reviewer guardrails.
- [ ] `REPRODUCE.md` written and validated end-to-end on a clean environment.
- [ ] Checkpoints and mini-mask annotations archived with permanent links.
