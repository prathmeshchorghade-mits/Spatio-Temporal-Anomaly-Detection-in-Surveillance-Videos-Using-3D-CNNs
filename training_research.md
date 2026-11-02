# Research Training Guide: Dense Spatio-Temporal Anomaly Localization

This guide details how to configure, execute, monitor, and ablate the **DenseC3D** weakly supervised anomaly localization model developed under Phase 2 and Phase 3 of the research track.

---

## 1. Overview & System Transition

The project has transitioned from a clip-level classification model to a dense, voxel-level spatiotemporal localization architecture:

| Component | Base Model (`Scratch3DCNN`) | Research Model (`DenseC3D`) |
|---|---|---|
| **Output Type** | Scalar clip-level logit `[B, 1]` | 4D Voxel logit map `[B, T'=4, H', W']` |
| **Head Architecture** | Global Adaptive Avg Pool + 2-layer MLP | Fully Convolutional 1×1×1 Conv3D |
| **Spatial Output** | Global pooled (1×1) | Dense grid (`g7`: 7×7, `g14`: 14×14, `g28`: 28×28) |
| **Loss Function** | Clip max-margin MIL | Dense voxel MIL (Ranking + Sparsity + Spatial/Temporal TV + Normal Suppression) |
| **Localization Mechanism** | Post-hoc 3D Grad-CAM (coarse attribution) | Direct feedforward dense voxel prediction |
| **Monitoring** | Validation AUC | Val AUC + Mean Voxel Sparsity + Border Mass Fraction |

---

## 2. Quickstart: How to Run Training

Always execute using the Python 3.11 environment where PyTorch and dependencies are configured:

```bash
PYTHON_BIN="/usr/local/opt/python@3.11/bin/python3.11"
```

### Scenario A: Training with Preprocessed Data & Manifests
If your processed dataset folder already contains `manifest_train.csv` and `manifest_val.csv`:

```bash
$PYTHON_BIN scripts/train_dense.py \
    --data-root "/path/to/your/processed_dataset" \
    --variant g14 \
    --seeds 0 \
    --epochs 60 \
    --batch-size 4
```

### Scenario B: Auto-Transcoding Raw UCF-Crime Videos
If pointing to raw UCF-Crime MP4 video archives, the pipeline will automatically transcode videos to 10 FPS (128×171) and construct stratified train/val manifests before launching training:

```bash
$PYTHON_BIN scripts/train_dense.py \
    --raw-dir "/path/to/raw_ucfcrime_directory" \
    --data-root "./data/ucf_crime_processed" \
    --variant g14 \
    --seeds 0
```

### Scenario C: Multi-Seed Ablation Ladder
To train across seeds (seeds 0, 1, 2) for rigorous error reporting:

```bash
$PYTHON_BIN scripts/train_dense.py \
    --data-root "/path/to/your/dataset" \
    --variant g14 \
    --seeds 0 1 2 \
    --epochs 60 \
    --batch-size 4
```

---

## 3. Parameter Guide: What Can Be Changed and What It Does

The table below outlines every configurable training parameter, default settings, and expected behavioral impact:

| Parameter Flag | Type & Default | Description | Impact of Modifying |
|---|---|---|---|
| `--variant` | `str`: `g14` (`g7`, `g14`, `g28`) | Spatial resolution grid variant. | **`g7` (stride 16):** Coarsest map (7×7), lowest FLOPs, largest receptive field.<br>**`g14` (stride 8):** Balanced tradeoff (14×14), recommended baseline.<br>**`g28` (stride 4):** Finest map (28×28), highest FLOPs, sharpest localization boundaries, but more prone to spatial noise. |
| `--widths` | `list`: `[32, 64, 128, 256]` | Conv channel capacity across the 4 blocks. | Increasing widths (e.g. `[64, 128, 256, 512]`) expands representation capacity but increases VRAM and risk of overfitting from-scratch on weak labels. |
| `--k-frac` | `float`: `0.01` (1%) | Top fraction of voxels aggregated for bag ranking. | **Higher (e.g. `0.05`):** Dilutes ranking score across normal and anomalous voxels; leads to blurry, diffuse heatmaps.<br>**Lower (e.g. `0.002`):** Focuses ranking on extreme peaks; sensitive to isolated single-pixel noise. |
| `--margin` | `float`: `1.0` | Hinge loss margin ($m$) applied directly on logits. | Enforces $S_{\text{anom}} - S_{\text{norm}} \ge m$. Increasing margin forces stronger separation between anomalous and normal activations. |
| `--lam-sp` | `float`: `8e-5` | L1 sparsity penalty on anomalous bag activations ($\sigma(V_{\text{anom}})$). | **Higher (e.g. `5e-4`):** Suppresses small/subtle anomalies; model may collapse to all zeros.<br>**Lower (e.g. `1e-5`):** Model activates indiscriminately over whole scenes, degrading localization. |
| `--lam-tv` | `float`: `8e-5` | Total Variation weight (spatial adjacent differences + temporal segment differences). | **Higher (e.g. `5e-4`):** Over-smooths heatmaps into uniform spatial blobs.<br>**Lower (e.g. `1e-5`):** Produces speckled, discontinuous high-frequency spatial noise across frames. |
| `--lam-norm` | `float`: `1e-2` | L2 penalty suppressing all normal bag voxels ($\sigma(V_{\text{norm}})^2$). | Crucial anchor signal. Forcing all normal video voxels toward zero prevents scene background features from gaining positive logits. |
| `--proximity-window` | `int x 3`: `None` (e.g. `2 3 3`) | Spatial-temporal contiguous window pooling before top-$k$ selection. | Replaces global top-$k$ with proximity-aware top-$k$ (experiment E5). Forces anomalous evidence to be clustered in space and time rather than scattered randomly across the video bag. |
| `--batch-size` | `int`: `4` | Number of bag-pairs (1 positive + 1 negative video bag) per batch. | Each batch processes $2 \times B \times S = 2 \times 4 \times 32 = 256$ clips. If encountering CUDA OOM, reduce to `2` (batch processes 128 clips). |
| `--clips-per-bag` | `int`: `32` | Number of 16-frame clips sampled across the video timeline. | Higher values give denser temporal coverage of long videos at the cost of additional GPU memory. |
| `--lr` | `float`: `1e-4` | Initial learning rate for AdamW optimizer with Cosine Annealing. | Default tuned for from-scratch 3D CNN convergence without destabilizing early Batch Normalization statistics. |
| `--weight-decay` | `float`: `5e-5` | AdamW L2 weight regularization. | Prevents filter blow-up when training from scratch without pre-trained backbones. |
| `--no-amp` | `flag` | Disable Automatic Mixed Precision. | AMP is enabled by default on CUDA. Pass `--no-amp` when training on CPU or debugging numerical precision. |

---

## 4. Experiment Matrix Mapping

When conducting ablation studies for the research paper, map command line invocations to experiment IDs:

| Experiment ID | Command Configuration | Purpose / Focus |
|---|---|---|
| **E2** | `python scripts/train_dense.py --variant g7 --seeds 0 1 2` | Stride 16 spatial resolution baseline (7×7 map) |
| **E3** | `python scripts/train_dense.py --variant g14 --seeds 0 1 2` | Stride 8 spatial resolution baseline (14×14 map) |
| **E4** | `python scripts/train_dense.py --variant g28 --seeds 0 1 2` | Stride 4 spatial resolution baseline (28×28 map) |
| **E5** | `python scripts/train_dense.py --variant g14 --proximity-window 2 3 3 --seeds 0 1 2` | Proximity-aware top-$k$ spatio-temporal clustering |
| **E8** | `python scripts/train_dense.py --variant g14 --lam-norm 2e-2` | Ablation on normal background suppression anchor |
| **E9** | `python scripts/train_dense.py --variant g14 --seeds 0 1 2 3 4` | 5-seed ensemble for epistemic uncertainty quantification |

---

## 5. Monitoring & Diagnostic Metrics

During training, each epoch reports diagnostic metrics alongside loss and validation AUC:

```
Monitoring → voxel_sparsity=0.0312  border_mass=0.1420
Val AUC: 0.8124
```

1. **`mean_voxel_sparsity` ($\sigma(V)$ mean):**
   - **Healthy range:** `0.01` to `0.08`
   - **Warning (< 0.005):** Representation collapse; model is predicting near-zero everywhere due to over-penalization (`--lam-sp` too high).
   - **Warning (> 0.25):** Saturation; model is failing to localize and marking huge background swathes as anomalous (`--lam-sp` too low).

2. **`border_mass_frac`:**
   - **Healthy range:** `< 0.20`
   - **Warning (> 0.35):** Letterbox shortcut detection. The model has discovered that compression artifacts or black border bars correlate with video sources, focusing attention on the image perimeter rather than the central action.
