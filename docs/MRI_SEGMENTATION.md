# MRI Brain-Tumour Segmentation — Whole Tumour (binary)

*Generated from run artifacts on 2026-08-13 08:13 UTC.*

This document covers the **Segmentation** workstream only. The X-ray
**Classification** workstream is separate, complete, and untouched by this work;
no result, chart, model, metric or upload is shared between the two.

## 1. Task

| Item | Value |
|---|---|
| Task | Binary **whole tumour vs background** segmentation on brain MRI |
| Model | **2D slice-wise U-Net using four MRI sequences** |
| Input | All four MRI sequences as channels — FLAIR · T1w · T1gd · T2w |
| Output | 1 binary channel (logits; sigmoid applied only for metrics/inference) |
| Binary rule | `mask = (label > 0)` |
| Slice axis | axial (index axis 2 of the 240x240x155 volume) |

> Slice-wise 2D U-Net — 4 input channels (one per MRI sequence), 1 binary output channel (whole tumour). This is not a 3D U-Net.

This is **not** multi-class subregion segmentation: the three source tumour
labels are merged into one foreground class before training.

## 2. Dataset source

| Item | Value |
|---|---|
| Dataset | Medical Segmentation Decathlon — Task01_BrainTumour (BraTS) |
| Identifier | `decathlon_task01_brain_tumour` |
| Download URL | `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar` |
| Project homepage | http://medicaldecathlon.com/ |
| Download date (UTC) | 2026-08-13 |
| Licence | CC BY-SA 4.0 (Medical Segmentation Decathlon) |
| Shipped `name` | BRATS |
| Shipped `release` | 2.0 04/05/2018 |
| Shipped `reference` | https://www.med.upenn.edu/sbia/brats2017.html |
| Shipped `numTraining` | 484 |
| Raw location | `data/raw/mri/decathlon_task01_brain_tumour/` |

The pre-existing tiny sample at `data/raw/mri/brats/` is legacy course data.
It was left completely untouched and is not used anywhere in this pipeline.

### 2.1 Exact source label mapping

Read from the shipped `dataset.json` and asserted in code
(`src/segmentation/mri/inventory.py::verify_dataset_json`):

| Source label | Meaning | Mapped to project target |
|---|---|---|
| `0` | background | background (0) |
| `1` | edema | **whole tumour (1)** |
| `2` | non-enhancing tumor | **whole tumour (1)** |
| `3` | enhancing tumour | **whole tumour (1)** |

Observed label values across all 484 verified volumes: `[0, 1, 2, 3]` — no undocumented values present.

## 3. Inventory and integrity

| Check | Result |
|---|---|
| Matched image/label pairs | 484 |
| Images without a label | 0 |
| Labels without an image | 0 |
| Volumes passing full verification | 484 |
| Volumes failing verification | 0 |
| Image shapes observed | (240, 240, 155, 4) |
| Modalities per volume | 4 — FLAIR · T1w · T1gd · T2w |
| Total axial slices | 75,020 |
| Slices containing tumour | 33,755 |

Every volume was fully decompressed and read voxel-by-voxel, so a truncated
or corrupt `.nii.gz` would have failed here rather than mid-training. Image and
label affines were compared per case to confirm spatial alignment.

## 4. Patient-safe split

- Split unit: **patient/case — never slices or patches**
- Seed: `42` (fixed, reproducible)
- Stratified on: whole-tumour burden quartile

| Split | Cases | % | Slices | Tumour slices | Mean tumour fraction |
|---|---:|---:|---:|---:|---:|
| Train | 339 | 70.0% | 52,545 | 23,647 | 0.01153 |
| Validation | 73 | 15.1% | 11,315 | 5,216 | 0.01271 |
| Held-out internal test | 72 | 14.9% | 11,160 | 4,892 | 0.01059 |
| **Total** | **484** | 100% | 75,020 | 33,755 | |

**Leakage check — zero patient overlap: True**

- train ∩ validation: 0 patients
- train ∩ test: 0 patients
- validation ∩ test: 0 patients

### 4.1 Protected held-out test split

> Project-created internal held-out test split drawn from the same public Decathlon training archive. It is NOT an external or clinical test set.

The test split was **not** used for:

- model architecture choice
- hyperparameter tuning
- early stopping
- threshold selection

It was read exactly once, by `scripts/mri_eval_heldout_test.py`, after the
validation-selected checkpoint was frozen. That script records a receipt
(`metrics/heldout_test_evaluation_receipt.json`) containing the checkpoint hash and
refuses to run a second time without an explicit override.

Manifests are frozen by SHA-256 in `data/processed/mri/split_frozen.json`; both the
training and the test scripts refuse to run if a manifest has changed.

## 5. Preprocessing

`4 MRI sequences → nonzero-voxel normalization → 2D slice preparation → binary whole-tumour mask`

| Step | Definition |
|---|---|
| Load | `.nii.gz` read directly with **nibabel**; image/label affines verified equal |
| Channels | four sequences stacked as input channels |
| Normalization | per volume, per modality **z-score over nonzero brain voxels only**; background written back as exactly `0.0` |
| Binary target | `mask = (label > 0)` |
| Slice axis | axial (index axis 2 of the 240x240x155 volume) |
| Input size | **192 × 192** |
| Image resampling | bilinear (image only) |
| Mask resampling | nearest-neighbour (masks only — labels are never interpolated) |
| Train sample shape | image `(4, 192, 192)`, mask `(1, 192, 192)` |

### 5.1 Input size choice (192×192)

Measured on NVIDIA GeForce RTX 5070 Ti Laptop GPU (12.82 GB) by
`scripts/mri_gpu_smoke_test.py` — real forward+backward passes, AMP enabled:

| Input size | Batch | Peak allocated | Peak reserved | Step time |
|---:|---:|---:|---:|---:|
| 160² | 8 | 0.43 GB | 0.50 GB | 28 ms |
| 160² | 16 | 0.74 GB | 0.91 GB | 42 ms |
| 160² | 24 | 1.06 GB | 1.36 GB | 68 ms |
| 160² | 32 | 1.38 GB | 1.76 GB | 94 ms |
| 192² | 8 | 0.57 GB | 0.69 GB | 30 ms |
| 192² | 16 | 1.02 GB | 1.30 GB | 71 ms |
| 192² | 24 **←chosen** | 1.48 GB | 1.90 GB | 99 ms |
| 192² | 32 | 1.92 GB | 2.50 GB | 120 ms |

192×192 at batch 24 peaks at 1.90 GB reserved of 12.82 GB — ample headroom, and it preserves more spatial detail than 160×160.

### 5.2 Slice retention (avoiding empty-slice domination)

| Split | Policy |
|---|---|
| Train | every tumour-containing slice, plus a deterministic per-case sample of `0.35 ×` that many empty brain slices (RNG seeded from the case id, seed `42`) |
| Validation | **every** slice of the volume, unfiltered |
| Held-out test | **every** slice of the volume, unfiltered |

Sampling runs per case, *after* the patient split is assigned, so it can never move
a patient or a slice across a split boundary. Evaluation splits are deliberately
unfiltered so per-case Dice cannot be flattered by dropping hard or empty slices.

| Cache | Cases | Slices | Tumour slices | Empty slices |
|---|---:|---:|---:|---:|
| train | 339 | 31,843 | 23,585 | 8,258 |
| valid | 73 | 11,315 | 5,204 | 6,111 |

### 5.3 Visual validation of mask alignment

`scripts/mri_visual_validation.py` rendered 8 overlays from
randomly drawn **training** cases and ran quantitative checks on 8 cases:

| Check | Threshold | Result |
|---|---|---|
| Tumour voxels falling on zero-intensity background | ≤ 0.05 | pass |
| FLAIR intensity contrast (tumour − brain, in sd) | ≥ 0.2 | pass |
| Tumour centroid inside brain bounding box | required | pass |

All checks passed: **True**. Figure: `outputs/segmentation/mri_unet_whole_tumour_2d_192/figures/visual_validation_train_overlays.png`

## 6. Model and training

| Setting | Value |
|---|---|
| Model | 2D slice-wise U-Net using four MRI sequences |
| Architecture | `mri_unet_2d`, encoder features [32, 64, 128, 256], 7,763,329 parameters |
| Input channels | 4 |
| Output channels | 1 (binary) |
| Loss | `L = 0.5 * BCEWithLogits + 0.5 * (1 - SoftDice), Dice smoothing eps=1.0, per-sample Dice averaged over batch` |
| Optimizer | AdamW, lr `0.0003`, weight decay `1e-05` |
| Scheduler | ReduceLROnPlateau(mode=max, factor=0.50, patience=3) |
| Batch size | 24 |
| Epochs configured / completed / best | 30 / 30 / **29** |
| Early stopping | patience 7 on validation Dice |
| Selection metric | validation micro Dice (validation split only) |
| Mixed precision | True |
| Gradient clipping | 1.0 |
| Seed / deterministic | 42 / True |
| Augmentation | hflip p=0.5; integer shift +/-6.25%; brain-masked intensity jitter p=0.5 |
| Training duration | 83.6 min |
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| torch / CUDA | 2.11.0+cu128 / 12.8 |

### 6.1 Loss formula

```
L = w_bce * BCEWithLogits(z, y) + w_dice * (1 - SoftDice(sigmoid(z), y))

SoftDice = (2 * sum(p*y) + eps) / (sum(p) + sum(y) + eps)

w_bce  = 0.5
w_dice = 0.5
eps    = 1.0
```

Soft Dice is computed per sample and then averaged over the batch, so a single
large tumour cannot dominate the batch term. Whole-tumour voxels are only a few
percent of a brain volume: plain BCE collapses toward empty masks, and plain Dice
is ill-conditioned on slices with no tumour, so both terms are kept.

## 7. Results

Dice and IoU are **per-case volumetric**: confusion counts are accumulated over
every axial slice of a case, then the metric is computed once from the case totals.
Decision threshold is fixed at `0.5` and was **not** tuned.

### 7.1 Validation (used for model selection)

| Metric | Dice | IoU |
|---|---:|---:|
| Mean (per case) | 0.8965 | 0.8190 |
| Median | 0.9165 | 0.8459 |
| IQR | 0.8837 – 0.9407 | 0.7917 – 0.8880 |
| 95% CI (bootstrap) | 0.8794 – 0.9113 | 0.7936 – 0.8413 |
| Micro (voxel-pooled) | 0.9174 | 0.8473 |

Cases: 73 (73 with ground-truth tumour). Precision 0.9224, recall 0.9124.

### 7.2 Held-out internal test (one-time, after checkpoint freeze)

| Metric | Dice | IoU |
|---|---:|---:|
| Mean (per case) | 0.8758 | 0.7868 |
| Median | 0.9006 | 0.8192 |
| 95% CI (bootstrap) | 0.8578 – 0.8922 | 0.7606 – 0.8113 |
| Micro (voxel-pooled) | 0.8965 | 0.8124 |

Cases: 72. Evaluated once at 2026-08-13T08:12:08+00:00 against
checkpoint `models/mri_unet_whole_tumour_2d_192_best.pt` (epoch 29, sha256 `ec4222c0b34087ad…`).

> Validation selected the model. This held-out number is a single generalization
> check and nothing was changed after seeing it.

## 8. Artifacts

Run directory: `outputs/segmentation/mri_unet_whole_tumour_2d_192/`

| Artifact | Path (relative to the run directory) |
|---|---|
| Best checkpoint | `models/mri_unet_whole_tumour_2d_192_best.pt` |
| Last checkpoint | `models/mri_unet_whole_tumour_2d_192_last.pt` |
| Training history | `metrics/training_history.csv` |
| Validation per case | `metrics/validation_per_case_metrics.csv` |
| Validation aggregate | `metrics/validation_aggregate_metrics.json` |
| Validation overlays | `figures/validation_qualitative_examples.png` |
| Training history figure | `figures/training_history.png` |
| Config used | `config_used.yaml` |
| Split summary | `split_summary_used.json` |
| Held-out test per-case metrics | `metrics/heldout_test_per_case_metrics.csv` |
| Held-out test aggregate metrics | `metrics/heldout_test_aggregate_metrics.json` |
| Held-out test overlays | `figures/heldout_test_qualitative_examples.png` |
| Held-out test receipt | `metrics/heldout_test_evaluation_receipt.json` |

Manifests and splits: `data/processed/mri/`

| File | Purpose |
|---|---|
| `mri_manifest.csv` | every usable case with shape, modality count, label stats, split |
| `train_manifest.csv` / `valid_manifest.csv` / `test_manifest.csv` | frozen per-split manifests |
| `split_summary.json` | split policy, counts, leakage check, source metadata |
| `split_frozen.json` | SHA-256 of each manifest (freeze record) |
| `dataset_inventory.json` | full Phase 1 per-case integrity inventory |

Presentation samples (copies only, originals untouched):

- `data/presentation_samples/segmentation_validation/`
- `data/presentation_samples/segmentation_test/`

## 9. Limitations

1. **Not clinical validation.** No external validation, no prospective evaluation,
   no evidence of diagnostic readiness. Academic demonstration only.
2. **The held-out test split is internal.** It comes from the same public Decathlon
   training archive, the same scanners and the same acquisition protocols as the
   training data. It measures generalization to unseen *patients*, not to unseen
   *sites*, scanners, sequences or populations.
3. **Whole tumour only.** Clinically important subregions (enhancing tumour, tumour
   core, edema) are merged into one class and are not distinguished.
4. **2D slice-wise, not 3D.** The model sees one axial slice at a time and has no
   through-plane context, so predictions can be inconsistent between adjacent slices.
   A 3D model would typically do better on volumetric coherence.
5. **Fixed threshold of 0.5**, not tuned; a tuned operating point could trade
   precision against recall differently.
6. **Requires all four co-registered sequences.** The model cannot run on a single
   MRI image or on a generic PNG, and the demo enforces this.
7. **Resampled to a fixed square input**, which changes voxel geometry; metrics are
   reported at 192×192 model resolution, not at native 240×240 acquisition
   resolution.
8. **Single training run, single seed.** No cross-validation and no seed-variance
   estimate, so the reported numbers carry run-to-run uncertainty beyond the
   bootstrap CI shown.

## 10. Reproducing

```bash
python scripts/mri_prepare_dataset.py          # Phase 1  download extract + inventory
python scripts/mri_build_manifests.py          # Phase 2  patient-safe frozen splits
python scripts/mri_gpu_smoke_test.py           # Phase 3  input-size evidence
python scripts/mri_build_slice_cache.py --splits train valid
python scripts/mri_visual_validation.py        # Phase 3  alignment gate
python scripts/run_mri_unet_whole_tumour.py    # Phase 4  train + validate
python scripts/mri_eval_heldout_test.py        # Phase 5  ONE-TIME test evaluation
python scripts/mri_build_presentation_samples.py
python scripts/mri_verify_handoff.py           # Phase 9  verification
```

