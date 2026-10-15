# Detailed Milestone Report: Spatio-Temporal Video Anomaly Detection

**Project:** Capstone project for Madhav Institute of Technology and Science (MITS)
**Objective:** Train a PyTorch 3D CNN from scratch in a local environment to score suspicious behaviour in surveillance videos using the UCF-Crime dataset with Multiple Instance Learning (MIL) ranking loss.
**Research Question:** *How far can a compact, from-scratch spatio-temporal model get on weak video-level labels, and what closes the gap to pretrained pipelines?*

---

## 1. Executive Summary

This document serves as a comprehensive milestone report for the Spatio-Temporal Video Anomaly Detection capstone project. The project is structured across a six-phase implementation plan (Phase 0–5). At the current milestone, **Phase 2 (Core Library)** and the foundational infrastructure have been successfully designed, implemented, and unit-tested. The core library establishes a robust, highly configurable PyTorch ecosystem that enables local CPU testing before deployment to high-performance GPUs.

## 2. Implemented Architecture and Core Library (Phase 2)

The foundation of the anomaly detection pipeline is composed of several critical Python modules:

### 2.1 Configuration and State Management (`training/config.py`)
- **Centralized Hyperparameters:** The `ProjectConfig` dataclass manages all aspects of the pipeline. This includes clip settings (length=16, stride=8, fps=10), spatial dimensions (resize 128x171, crop 112x112), and training parameters (AdamW LR=1e-4, 60 epochs, early stopping patience=12).
- **Reproducibility:** A rigorous `set_seed()` function ensures exact reproducibility across Python, NumPy, PyTorch, and cuDNN.
- **Serialization:** Full JSON serialization/deserialization enables precise tracking and reloading of experiment parameters.

### 2.2 Neural Network Architecture (`training/model/scratch_3dcnn.py`)
- **Custom 3D CNN:** A C3D-inspired architecture built completely from scratch without relying on pretrained weights.
- **Structural Flow:** 
  - *Block 1:* Conv3D 3→64, BN, ReLU, MaxPool 1×2×2
  - *Block 2:* Conv3D 64→128, BN, ReLU, MaxPool 2×2×2
  - *Block 3:* Two Conv3D 128→256→256, BN, ReLU, MaxPool 2×2×2
  - *Block 4:* Two Conv3D 256→512→512, BN, ReLU, MaxPool 2×2×2
- **Features:** A global adaptive pooling layer (allows input size flexibility, e.g., 112x112 for training, 112x144 for inference) followed by an MLP head (Dropout, FC, ReLU). Outputs raw logits rather than sigmoid probabilities to prevent gradient saturation within the MIL ranking loss.

### 2.3 Multiple Instance Learning (MIL) Loss (`training/loss.py`)
- **Composite Loss Function:** Implements a sophisticated `MILRankingLoss` that integrates three core components:
  1. **Ranking Loss:** A hinge loss pushing the highest-scoring anomalous clip above the highest-scoring normal clip by a configurable margin, supporting both max-MIL and top-k MIL modes.
  2. **Sparsity Regularization:** Penalizes the mean sigmoid score across the video, encouraging anomalies to be rare occurrences.
  3. **Smoothness Regularization:** Penalizes rapid, unnatural fluctuations in anomaly scores between adjacent temporal clips.

### 2.4 Data Pipeline and Transformations (`training/dataset/`)
- **Transform Pipeline (`transforms.py`):** Configured with distinct, specialized pathways:
  - *Training:* Resize, random crop, random horizontal flip, brightness/contrast jitter, and Kinetics-400 normalization.
  - *Validation/Test:* Resize, center crop, and normalize.
  - *Inference:* Resize to non-square inference dimensions (112x144) without cropping, preserving the full frame.
- **MIL Bag Dataset (`mil_dataset.py`):**
  - Training bags utilize Sultani-style segment sampling, dividing videos into 32 segments with epoch-by-epoch re-jittering to augment data.
  - Validation sets extract all clips using a sliding window (stride 8) with OpenCV decoding for comprehensive, frame-accurate evaluation.
  - A built-in manifest loader (`UCFCrimeManifest`) handles dataset mapping, splits, and categories via CSV.

### 2.5 Training Loop and Engine (`training/train.py`)
- **Optimization Strategy:** Employs AdamW optimization coupled with a cosine annealing learning rate scheduler.
- **Stability Mechanisms:** Integrates Automatic Mixed Precision (AMP) for speed and gradient clipping for stability.
- **Resilience:** Implements resume-safe checkpointing that saves model weights, optimizer/scheduler states, AMP scaler, epoch details, and deeply nested RNG states for perfect mid-training resumption in unstable local environments.
- **Early Stopping:** Monitored via video-level AUC on the validation split.

### 2.6 Evaluation and Metrics (`training/metrics.py`)
- **AUC Calculations:** Computes comprehensive metrics including video-level ROC-AUC, alongside frame/clip-level ROC-AUC and PR-AUC.
- **False Alarm Modeling:** Tracks false alarms per hour on normal videos using burst-based detection.
- **Threshold Calibration:** Determines the optimal separation threshold dynamically based exclusively on the validation set to achieve targeted False Positive Rates.
- **Visualizations:** Generates side-by-side ROC and PR curve plots automatically.

### 2.7 Inference and Postprocessing (`inference/postprocess.py`)
- **Score Smoothing:** Applies an Exponential Moving Average (EMA) filter (α=0.3) to stabilize raw model predictions over time.
- **Hysteresis Thresholding:** Employs a state-machine-based interval detection strategy, requiring consecutive threshold breaches (run-length persistence, e.g., N=3 consecutive clips above threshold) to enter or exit an anomaly state, preventing flickering.
- **Data Export:** Seamlessly converts clip-index intervals to timestamp intervals, outputting per-video score CSVs and annotated timeline plots.

## 3. Pending Implementation (Upcoming Phases)

While the core library is complete, the following phases represent the remaining implementation roadmap:

- **Phase 0 (Environment Setup):** Establish persistent disk volumes, generate pinned `requirements.txt`, and configure the GPU computing environment.
- **Phase 1 (Data Pipeline):** Execute mass pre-transcoding of UCF-Crime videos to a standardized 10 fps H.264 format, calculate file hashes, and perform quality passes to filter corrupted inputs.
- **Phase 3 (Training Integration):** Wire the entire training pipeline across local notebooks, execute training on complete datasets across 3 discrete seeds to ensure statistical significance, and log comprehensive checkpoints.
- **Phase 4 (Evaluation):** Finalize threshold calibration on the validation split and produce detailed category-wise error breakdowns and proxy metric correlations (e.g., evaluating failures based on low light, blur, or camera shake).
- **Phase 5 (Offline Inference):** Assemble final video overlays with visual score timelines and color-coded anomaly borders, yielding annotated MP4 deliverables.

## 4. Planned Enhancements & Differentiation

To ensure academic and practical rigor, the following extensions are scoped:
- **Ablation Studies:** Evaluating classical frame-differencing baselines, top-k MIL variations, (2+1)D convolutions, and motion channel integrations.
- **Explainability:** Utilizing 3D Grad-CAM overlays to demystify model activations and analyzing potential model shortcuts (e.g., reliance on camera quality or brightness).
- **Differentiation Strategy:** Scoping a specialized "campus-safety" use case, introducing a 13-way category head for triage scoring, and utilizing seed ensembles for uncertainty estimation.

## 5. Summary Statistics

| Metric | Value |
|---|---|
| **Total Python Source Files** | 10 |
| **Total Lines of Code** | ~1,795 |
| **Documentation Files** | 4 (README, SYSTEM_ARCHITECTURE, implementation guide, summary) |
| **Phases Completed** | Phase 2 (Core Library — Model & Loss) |
| **Phases Remaining** | Phases 0, 1, 3, 4, 5 |
| **Robustness Mechanisms** | 3-seed reporting, hysteresis postprocessing, full RNG state persistence |

---
This milestone signifies the successful translation of the theoretical system architecture into a functional, highly configurable codebase. The project is now technically ready for full-scale data processing and intensive model training.
