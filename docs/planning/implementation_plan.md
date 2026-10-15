# Implementation Plan

**Project:** Dense Weakly Supervised Anomaly Localization with a Compact From-Scratch 3D CNN: Localization-Aware Shortcut Audit and Ensemble Uncertainty  
**Base repo:** Spatio-Temporal Video Anomaly Detection (MITS capstone)  
**Plan date:** 14 Oct 2026  

This plan extends the existing six-phase capstone plan (Phases 0-5). Phases 0-5 stay as written (manifests, official UCF-Crime split, 16-frame clips, MIL training, frozen-threshold evaluation, inference). The new work is Phases 6-10, plus changes to the model head and loss in Phases 2-3.

---

## 0. Positioning (read this first)

A literature check on 14 Oct 2026 showed that weakly supervised spatial localization on UCF-Crime is **already published**, so "we localize anomalies from video-level labels" is not a novelty claim by itself:

| Work | What it does | Backbone |
|---|---|---|
| Landi et al., "Anomaly Locality in Video Surveillance" (2019) | Action tubes, UCFCrime2Local boxes | Pretrained |
| Liu & Ma, "Exploring Background-Bias" (ACM MM 2019) | Box annotations for UCF-Crime, background-bias analysis | Pretrained |
| WSSTAD (IJCAI 2021) | Weakly supervised tube localization, ST-UCF-Crime | Pretrained proposals |
| STPrompt (ACM MM 2024) | Spatial localization with CLIP prompts | Frozen CLIP |
| "Learning Where and When" (arXiv 2606.29498) | Patch-level MIL, proximity-aware top-k, 7x7 grid | Frozen UniFormer |
| "Localizing to Debias" (arXiv 2608.12045) | Patch-level benchmark and baseline, detector-free | Not verified (full text unreadable) |
| "Auditing Frame-Level AUC..." (arXiv 2608.11985) | Granularity, resolution, scene bias of frame-level AUC | Audit paper |

**Gap this project targets (all unverified against full texts, so confirm in week 1):**

1. Every localization method above sits on **pretrained frozen features**. Nobody (that I found) studies whether dense localization **emerges in a compact model trained from scratch**, or what it costs in FLOPs and CPU throughput.
2. Existing maps are coarse (7x7 patches). A **resolution study** (7x7 vs 14x14 vs 28x28 voxel maps) is a concrete, answerable question.
3. Existing bias analyses audit frame-level AUC or background bias in general. A **localization-aware** audit asks: "is the evidence inside the annotated anomaly region, and does removing the region actually change the score?"
4. Uncertainty exists in VAD (e.g. evidential reasoning) but **spatio-temporal uncertainty maps with risk-coverage for localization** is a different object.

**Claim to write:** a reproducible study of dense, weakly supervised localization with compact from-scratch 3D CNNs, with a localization-aware shortcut audit and ensemble uncertainty. **Do not claim** state-of-the-art AUC or TIoU. Pretrained pipelines will win on raw numbers.

---

## 1. Contribution map

| ID | Contribution | Role in paper | New phases |
|---|---|---|---|
| A | Dense voxel-level MIL localization with a from-scratch 3D CNN, resolution/efficiency study | Main | 2, 3, 6, 7 |
| B | Localization-aware shortcut and background-reliance audit, with a mitigation | Validity check, second main result | 8 |
| D | Seed-ensemble uncertainty maps, risk-coverage, triage rule | Add-on | 9 |

---

## 2. Compute and data requirements

**Hardware (local):** one NVIDIA GPU with at least 12 GB VRAM (8 GB works with batch of 2 bag-pairs and AMP). 16+ GB RAM, 300 GB free disk (raw UCF-Crime is about 128 hours; transcoded 10 fps low-res copy is far smaller).

**Software (pin in requirements.txt):** Python 3.10+, PyTorch 2.x with CUDA, torchvision, opencv-python-headless, numpy, pandas, scikit-learn, matplotlib, tqdm, pyyaml, fvcore or ptflops (FLOPs), onnx and onnxruntime (CPU throughput), pytest. For pseudo-masks: SAM 2 (inference only, used for the evaluation mini-benchmark, not for training). For the audit only: a frozen image encoder (CLIP or DINOv2) to embed scene content.

**Data:**
- UCF-Crime videos, official split (1610 train, 290 test), preserved as in Phase 1.
- Box annotations for UCF-Crime test videos from prior work (Liu & Ma 2019 and/or ST-UCF-Crime from WSSTAD). **Week 1 task:** confirm which are publicly downloadable and their licenses/format. If neither is accessible, the localization evaluation falls back to your own annotated subset (Phase 7).
- ShanghaiTech Campus for the cross-dataset check (frame-level only unless masks are available).

**Compute estimate (rough, depends on your GPU):** the main model is small, so one 60-epoch run is likely hours, not days. Budget: 5 variants x 3 seeds for the ablation ladder plus 5 seeds for the final ensemble. Measure one epoch in Phase 3 and extrapolate before committing to the full grid.

---

## 3. Changes to the existing phases

### Phase 2 (model): make the head dense

Replace "global pooling + MLP" with a fully convolutional dense head. Keep the backbone from-scratch (3x3x3 conv blocks, BN, ReLU).

Output stride options (these are your resolution ablation):

| Variant | Spatial stride | Map for 112x112 input | Map for 112x144 input |
|---|---|---|---|
| `g7` | 16 | 7x7 | 7x9 |
| `g14` | 8 | 14x14 | 14x18 |
| `g28` | 4 | 28x28 | 28x36 |

Temporal downsampling: 16 frames to T'=4 (pool time in blocks 2 and 3 only).

```python
# training/models/dense_c3d.py
import torch, torch.nn as nn

def block(cin, cout, pool):
    layers = [nn.Conv3d(cin, cout, 3, padding=1, bias=False),
              nn.BatchNorm3d(cout), nn.ReLU(inplace=True)]
    if pool is not None:
        layers.append(nn.MaxPool3d(kernel_size=pool, stride=pool))
    return nn.Sequential(*layers)

class DenseC3D(nn.Module):
    """Fully convolutional: input [B,3,16,H,W] -> logits [B,T',H',W'].
    pools: tuple of 4 pool specs, e.g. g14 = ((1,2,2),(2,2,2),(2,2,2),None)
    """
    def __init__(self, widths=(32, 64, 128, 256),
                 pools=((1,2,2),(2,2,2),(2,2,2),None)):
        super().__init__()
        cs = (3,) + tuple(widths)
        self.features = nn.Sequential(*[block(cs[i], cs[i+1], pools[i]) for i in range(4)])
        self.head = nn.Conv3d(widths[-1], 1, kernel_size=1)   # voxel logit

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)          # [B,T',H',W']

    def clip_logit(self, vox, k_frac=0.1):
        # clip score = mean of top-k voxel logits (robust to one noisy voxel)
        flat = vox.flatten(1)
        k = max(1, int(k_frac * flat.shape[1]))
        return flat.topk(k, dim=1).values.mean(1)
```

Shape contract: for a bag of S clips, input `[B*S,3,16,H,W]`, voxel map `[B*S,T',H',W']`, reshape to `[B,S,T',H',W']`. Because there is no fully connected layer, the same weights run on the 112x112 training crop and the full 112x144 inference frame.

### Phase 2 (loss): voxel-level MIL

```python
# training/losses/dense_mil.py
import torch, torch.nn.functional as F

def topk_mean(x, k_frac):                       # x: [B, N]
    k = max(1, int(k_frac * x.shape[1]))
    return x.topk(k, dim=1).values.mean(1)

def dense_mil_loss(vox_a, vox_n, k_frac=0.01, margin=1.0,
                   lam_sp=8e-5, lam_tv=8e-5, lam_norm=1e-2):
    """vox_a, vox_n: logits [B,S,T',H',W'] for anomalous / normal bags."""
    B = vox_a.shape[0]
    a = vox_a.flatten(1); n = vox_n.flatten(1)
    # 1) ranking on logits: top-k voxels of the anomalous bag beat top-k of the normal bag
    rank = F.relu(margin - topk_mean(a, k_frac) + topk_mean(n, k_frac)).mean()
    # 2) sparsity: anomalous evidence should be small in extent
    sparse = torch.sigmoid(vox_a).mean()
    # 3) smoothness: spatial TV inside each clip map + temporal difference between
    #    consecutive sampled segments (pooled over space)
    pa = torch.sigmoid(vox_a)
    tv_space = (pa[..., 1:, :] - pa[..., :-1, :]).abs().mean() + \
               (pa[..., :, 1:] - pa[..., :, :-1]).abs().mean()
    seg = pa.mean(dim=(2, 3, 4))                              # [B,S]
    tv_time = (seg[:, 1:] - seg[:, :-1]).pow(2).mean()
    # 4) normal bags: every voxel should be low (cheap, strong negative signal)
    norm = torch.sigmoid(vox_n).pow(2).mean()
    return rank + lam_sp * sparse + lam_tv * (tv_space + tv_time) + lam_norm * norm
```

Notes: hinge on logits (margin about 1.0) avoids the saturation problem already noted in the README. Treat `k_frac`, `lam_*` as hyperparameters; tune on the validation split by validation AUC only. Add a **proximity-aware variant** (select top-k voxels in a contiguous space-time window rather than globally) as one ablation row; this reproduces the idea from 2606.29498 on your backbone and shows whether it helps from scratch.

### Phase 3 (training): additions

- Everything in the existing Phase 3 stays (AdamW, cosine, AMP, resume-safe checkpoints, best by validation AUC, CSV logs).
- Log, per epoch: validation frame-level AUC, mean voxel sparsity, and fraction of heatmap mass in the border region (a cheap early warning for border/letterbox shortcuts).
- Seeds: 0, 1, 2 for the ablation table; seeds 0-4 for the final ensemble (Phase 9).

### Smoke tests to add (`tests/`)

- `test_dense_shapes.py`: voxel map shape per variant for 112x112 and 112x144 inputs.
- `test_dense_loss.py`: loss is finite, gradient reaches the first conv, loss is zero when anomalous top-k is far above normal top-k.
- `test_fully_conv.py`: the same weights run on both input sizes.
- `test_heatmap_projection.py`: projecting a one-hot voxel to frame coordinates lands at the expected pixel box.

---

## 4. New phases

### Phase 6: Heatmap generation and projection (`inference/heatmaps.py`)

1. Run the model on stride-8 overlapping clips (existing Phase 5 logic) and keep the voxel map `[T',H',W']` per clip.
2. Upsample each map bilinearly to frame resolution.
3. Assign each clip's maps to the frames it covers; average where clips overlap.
4. Apply temporal EMA (alpha 0.3, as in Phase 5) per pixel.
5. Output: `{name}_heat.npz` (float16 `[T,H,W]`) and an overlay MP4 (replaces the 3D Grad-CAM overlay in the original plan; keep Grad-CAM as a baseline in Phase 7).

### Phase 7: Localization evaluation (Contribution A)

**Ground truth.**
- Boxes from the public annotations (week-1 check). Typically annotated on every 16th frame of anomalous test videos, so evaluate on annotated frames only.
- **Mini pixel-mask set (your own contribution):** pick 60-100 anomalous test videos, about 10 annotated frames each. Prompt SAM 2 with the GT box to get a mask, then manually verify and fix every mask in CVAT or labelme. Report this honestly as "box-prompted, human-verified masks". Release only annotations, not videos (UCF-Crime licensing).

**Threshold-free metrics (primary, because validation videos have no boxes to tune on):**
- **Pointing game:** is the heatmap argmax inside the GT box/mask?
- **Energy-in-region:** fraction of heatmap mass inside the GT region, compared to the region's area fraction (a random map scores equal to area fraction).
- **Pixel AUC:** ROC-AUC of heatmap values against the GT mask, on anomalous frames.

**Threshold-based metrics (secondary, fixed rule):**
- Threshold at 0.5 x per-frame maximum (no tuning on test), connected components to boxes.
- IoU in anomalous frames (reported in prior work as "TIoU"), and RBDR/TBDR (Ramachandra & Jones protocol, alpha=beta=0.1) so numbers can be placed next to the literature.
- If you want a tuned threshold, carve a dev subset (20%) from the annotated videos, tune there, and report only on the remaining 80%. State this in the paper.

**Baselines on the same data and backbone:**
1. Random map and center-prior map (sanity floor).
2. 3D Grad-CAM on the clip-level model (`g_global` = original Phase 2 model).
3. `g7`, `g14`, `g28` dense models (resolution study).
4. Dense model with proximity-aware top-k.
5. Reference upper bound: frozen pretrained features (e.g. I3D or VideoMAE) with the same dense head, clearly labeled as a pretrained reference.

**Efficiency table:** parameters, FLOPs (fvcore), GPU clips/s, CPU clips/s via ONNX Runtime. Plot localization quality (pointing game or pixel AUC) against FLOPs.

**Frame-level detection** (unchanged from Phase 4): ROC-AUC and PR-AUC on the official test set, mean +/- std over 3 seeds. Also report AUC **within anomalous test videos only** (see Phase 8).

### Phase 8: Localization-aware shortcut audit (Contribution B)

Goal: show how much of the performance comes from cues unrelated to the event, and whether the dense model's evidence lies on the event.

**8.1 Cue-only baselines.** Per clip, compute: mean brightness, contrast, Laplacian variance (blur), native resolution, fps, bitrate (file size over duration), global-motion magnitude, and a scene embedding (frozen CLIP/DINOv2 of the middle frame, reduced with PCA to 16-32 dims). Train logistic regression and gradient boosting on the training split, evaluate frame-level AUC on test with the same pipeline as the model. Report AUC of each cue family and of all cues combined.

**8.2 Source confound control.** Normal and anomalous videos come from different sources, which inflates frame-level AUC. Report three AUCs side by side: standard (all test videos), **anomalous-videos-only** (distinguishes anomalous from normal frames inside the same video), and per-category. A model whose AUC collapses in the anomalous-only setting is relying on source cues.

**8.3 Score-cue correlation.** Spearman correlation of model scores with each cue, per video and per clip.

**8.4 Region interventions (needs boxes/masks).**
- *Deletion:* replace the GT region with a blurred or mean-filled patch, measure the drop in clip score.
- *Insertion/background-only:* keep only the GT region, blank everything else.
- *Control:* same operations on an equal-area random region outside the box.
A model that uses the event shows a large deletion drop and a small control drop. A background-reliant model shows the reverse or no difference.

**8.5 Background reliance score.** `1 - energy_in_region / (1 - area_fraction)` clipped to [0, 1], averaged per video, reported per category.

**8.6 Mitigation (pick one, ablate).**
- Quality-matched augmentation: random brightness, contrast, blur, JPEG/H.264 re-encoding applied to **both** normal and anomalous videos, with strengths sampled from a shared distribution.
- Scene-adversarial head: gradient-reversal head predicting the scene cluster id from backbone features.
- Background-suppression term: penalize voxel scores on regions with low motion energy (frame-difference mask) in anomalous bags.

Report cue-only AUC, anomalous-only AUC, deletion drop, and pointing game before and after the mitigation. This is the table that makes B a result and not just an analysis.

**Positioning against prior audits:** cite 2608.11985 (frame-level AUC audit) and Liu & Ma 2019 (background bias) explicitly, and state that your audit adds region interventions on a from-scratch dense model.

### Phase 9: Ensemble uncertainty and triage (Contribution D)

1. Train **5 seeds** of the final configuration (3 seeds make a weak ensemble; use 5 if compute allows, otherwise state the limit).
2. For each voxel: mean probability `p_bar`, predictive entropy `H(p_bar)`, and mutual information `H(p_bar) - mean_m H(p_m)` as the epistemic term.
3. Clip-level uncertainty: top-k mean of voxel MI. Video-level: max over clips.
4. **Risk-coverage curve:** sort alerts by uncertainty, drop the most uncertain, plot alert precision (and localization hit rate) versus coverage. The expected result is rising precision as coverage falls; report the area under the risk-coverage curve and compare against ranking by score alone and by random.
5. **Calibration:** ECE and reliability diagram on validation clips with pseudo-labels from the frozen threshold; report honestly that video-level weak labels make clip-level calibration only approximate.
6. **Spatial check:** Spearman correlation between per-frame mean voxel MI and (1 - IoU) on annotated frames.

```python
# training/uncertainty.py
import torch

def ensemble_stats(prob_list):                 # list of M tensors [T,H,W]
    P = torch.stack(prob_list)                 # [M,T,H,W]
    pbar = P.mean(0)
    H = lambda p: -(p*torch.log(p+1e-8) + (1-p)*torch.log(1-p+1e-8))
    total = H(pbar)
    aleatoric = H(P).mean(0)
    mi = total - aleatoric                     # epistemic
    return pbar, total, mi

def risk_coverage(unc, correct):               # numpy 1-D arrays per alert
    import numpy as np
    order = np.argsort(unc)                    # most certain first
    c = np.cumsum(correct[order]) / (np.arange(len(order)) + 1)
    cov = (np.arange(len(order)) + 1) / len(order)
    return cov, c                              # precision vs coverage
```

**Triage rule (documented, rule-based, not learned):**
`severity = tier(score_smoothed, duration_s, mi)` with tiers low/medium/high from fixed cutoffs fit on validation, and a confidence label from uncertainty (`confidence = high` if MI below the validation median). Output: "anomaly, 02:14-02:31, severity high, confidence medium". The 13-way category head from the original plan is **optional and deferred**: it needs separate validation and dilutes the paper. Add it only if everything else is done.

### Phase 10: Cross-dataset check and reproducibility

- Run the frozen best model unmodified on ShanghaiTech (frame-level AUC; report the drop honestly).
- Release: code, configs, trained checkpoints, annotation files for the mini mask set, and a `REPRODUCE.md` with exact commands and seeds.

---

## 5. Experiment matrix

| ID | Variant | Seeds | Reports |
|---|---|---|---|
| E0 | Frame-differencing baseline | n/a | AUC |
| E1 | Clip-level max-MIL (original Phase 2 model) + Grad-CAM | 3 | AUC, localization |
| E2 | Dense `g7`, top-k MIL | 3 | AUC, localization, FLOPs |
| E3 | Dense `g14` | 3 | same |
| E4 | Dense `g28` | 3 | same |
| E5 | Best grid + proximity-aware top-k | 3 | same |
| E6 | Best + (2+1)D convolutions | 3 | same |
| E7 | Best + frame-difference motion channel | 3 | same |
| E8 | Best + augmentation/adversarial mitigation (Phase 8.6) | 3 | audit table |
| E9 | Best config, 5-seed ensemble | 5 | uncertainty, risk-coverage |
| E10 | Frozen pretrained features + dense head (reference) | 3 | upper bound |
| Optional | Self-supervised pretraining (speed or order prediction) | 3 | gap-closing |

Rule: pick the "best" configuration by **validation AUC only**, never by test localization. Freeze it before running Phase 7 on test.

---

## 6. Timeline (16 weeks)

| Weeks | Work | Gate |
|---|---|---|
| 1 | Re-verify literature with full texts (especially 2608.12045, 2608.11985); confirm box annotations are obtainable; finish Phases 0-1 | **Go/no-go 1:** if a paper already does from-scratch dense localization with a shortcut audit, narrow the claim (see section 8) |
| 2-3 | Dense model, loss, smoke tests; baseline E0, E1 | Shapes and gradients pass |
| 4-6 | Train E2-E4, first localization eval on boxes | **Go/no-go 2:** if pointing game is not clearly above the center-prior baseline, switch to the fallback framing |
| 7-8 | Mini pixel-mask set, full Phase 7 metrics, efficiency table | Annotations verified |
| 9-10 | E5-E8, audit (Phase 8) | Audit table complete |
| 11-12 | Ensemble, uncertainty, triage (Phase 9) | Risk-coverage curve |
| 13 | E10 reference, ShanghaiTech check | All tables frozen |
| 14-15 | Write paper, figures | Draft |
| 16 | Internal review, reproducibility pass, submit | Venue deadline |

Check venue deadlines now; workshop deadlines for spring 2027 conferences are typically in the months after this plan starts, but verify the actual dates.

---

## 7. Deliverables checklist

- [ ] `requirements.txt`, `environment.json`, seeds logged
- [ ] manifests and data-quality report (Phase 1)
- [ ] `DenseC3D` + `dense_mil_loss` with passing tests
- [ ] checkpoints (best by validation AUC) for E1-E10
- [ ] `calibrated_threshold.json` (frame-level) and fixed localization rule documented
- [ ] heatmap overlays and score CSVs for example videos
- [ ] mini pixel-mask annotations with a short datasheet
- [ ] efficiency table and AUC/localization vs FLOPs plot
- [ ] audit tables (cue-only, anomalous-only AUC, deletion, background reliance, mitigation)
- [ ] ensemble uncertainty maps, risk-coverage and calibration plots
- [ ] ShanghaiTech result
- [ ] paper draft, `REPRODUCE.md`

---

## 8. Risks and fallback framing

| Risk | Mitigation |
|---|---|
| From-scratch dense maps are barely better than a center prior | Report it. Reframe as "what from-scratch weak supervision can and cannot localize", with the audit (B) and uncertainty (D) as the main results. A well-run negative result with a released annotation set is publishable at workshop level |
| Box annotations not accessible | Annotate your own subset (Phase 7) and enlarge it; cite the original annotation papers |
| Another paper already covers the exact combination | Narrow to the parts still open, usually the localization-aware audit on a compact model, or the efficiency study |
| Frame-level AUC inflated by source confound | Always report anomalous-only AUC next to standard AUC |
| Overfitting to the 290-video test set | Test touched once, after all choices are frozen on validation |
| Ensemble of 3 too small | Use 5 seeds, or state the limitation |
| Licensing of UCF-Crime | Release annotations and code only, never videos |

---

## 9. Claims to avoid

- State-of-the-art AUC or localization on UCF-Crime.
- "First" localization method, or "first" audit. Say "to our knowledge" only after the full-text literature check.
- Pixel-accurate ground truth (the masks are box-prompted and human-verified, not independently drawn).
- Calibrated uncertainty at clip level (labels are video-level).
