# AI-Assisted Medical Image Analysis

**1) Chest X-ray Disease Classification  ·  2) Brain MRI Tumour Segmentation**

A reproducible PyTorch project for a master's-level final assignment.

> ### ⚠️ Academic prototype — not a medical device
> This project is an **educational prototype** built for coursework. It **cannot
> diagnose patients**, does **not** assist or replace clinicians, has **no
> clinical validation** and **no regulatory approval**. It must never be used for
> any decision about a real person's care. See [`docs/ETHICS.md`](docs/ETHICS.md).

> ### Status: code complete, no data, no results
> The course dataset has not been added yet. **No model has been trained and no
> metric has been produced.** Every table and figure appears only after you add
> data and run the notebooks. Nothing in this repository contains fabricated
> results.

---

## Contents

1. [What this project does](#1-what-this-project-does)
2. [Environment setup](#2-environment-setup)
3. [Where to put the data](#3-where-to-put-the-data)
4. [Notebook order](#4-notebook-order)
5. [Project workflow](#5-project-workflow)
6. [Configuration](#6-configuration)
7. [Evaluation metrics](#7-evaluation-metrics)
8. [Repository layout](#8-repository-layout)
9. [Reproducibility](#9-reproducibility)
10. [Ethical and clinical limitations](#10-ethical-and-clinical-limitations)
11. [Screenshot checklist](#11-screenshot-checklist)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What this project does

### Part A — Chest X-ray classification

- Configurable **binary or multi-label** disease classification (one sigmoid
  output per label).
- **Patient-level** train/validation/test split with an automated leakage check.
- EDA: class distribution, image-size audit, missing-file detection, samples.
- Two models: a **from-scratch CNN baseline** and **ImageNet-pretrained
  DenseNet-121 / EfficientNet** transfer learning.
- Class imbalance handled with `pos_weight` in `BCEWithLogitsLoss` (focal loss
  also available).
- Metrics: ROC-AUC, PR-AUC, precision, recall/sensitivity, specificity, F1,
  confusion matrix — with **thresholds selected on validation data only** and
  **bootstrap confidence intervals**. Accuracy is reported as secondary only.
- **Grad-CAM** attribution maps, including failure cases.

### Part B — Brain MRI tumour segmentation

- Paired image/mask discovery with **full pair validation** before training.
- **Patient-level** (not slice-level) splitting.
- EDA: mask coverage, tumour-area distribution, overlays, dimension audit.
- **U-Net** implemented from scratch (plus an optional pretrained ResNet-34
  encoder variant).
- **Dice + BCE** loss (Tversky also available).
- Metrics: Dice, IoU/Jaccard, sensitivity, specificity, per-slice distribution,
  bootstrap CIs resampled by patient.
- Qualitative overlays for best, typical **and worst** cases; predicted masks
  saved to disk.

### Engineering

- All reusable logic in `src/` — notebooks orchestrate, they do not implement.
- All tunables in `configs/*.yaml`; every run snapshots the config it used.
- Seeded everywhere; stable, hash-based patient splits.
- Actionable error messages when data is missing or mis-structured.

---

## 2. Environment setup

**Requirements:** Python 3.11, Windows/Linux/macOS. A CUDA GPU is strongly
recommended but not required — everything falls back to CPU.

### Step 1 — create and activate a virtual environment

```powershell
# Windows PowerShell, from the project root
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Step 2 — install PyTorch (do this first, separately)

The correct wheel depends on your GPU, so PyTorch is **not** installed by a plain
`pip install -r requirements.txt`.

```powershell
# NVIDIA GPU — CUDA 12.8 wheels.
# Required for RTX 50-series (Blackwell, compute capability 12.0);
# also works for RTX 30/40-series.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```powershell
# No NVIDIA GPU (CPU only) — much smaller download, much slower training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The CUDA download is roughly 2–3 GB. If it drops out mid-download, retry with
`--retries 10 --timeout 120`.

Verify before moving on — the version string must end in `+cu…`, not `+cpu`:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 3 — install everything else

```powershell
pip install -r requirements.txt
```

> `requirements.txt` deliberately does **not** list `torch`/`torchvision`. If it
> did, this command would pull the CPU build from PyPI and silently replace the
> CUDA build you just installed. Do not add them to the file. If you ever see
> `torch.cuda.is_available()` turn `False` after a `pip install`, this is why —
> re-run step 2.

### Step 4 — register the Jupyter kernel

```powershell
python -m ipykernel install --user --name medicaldiagnostic --display-name "Python 3 (medicaldiagnostic)"
```

### Step 5 — verify

```powershell
jupyter lab
```

Open **`notebooks/00_environment_check.ipynb`**, select the
*Python 3 (medicaldiagnostic)* kernel, and run every cell. It checks the Python
version, packages, GPU, project imports, folder structure, seeding, and runs a
forward pass through both model families on synthetic tensors. Do not continue
until it passes.

> **TensorFlow is not used anywhere in this project** and must not be installed
> into this environment.

---

## 3. Where to put the data

Nothing under `data/` is committed to git. Full layout documentation with config
snippets: **[`data/README.md`](data/README.md)**.

### Chest X-ray → `data/raw/xray/`

Default layout — one folder per class, optional `train/`/`test/` level:

```
data/raw/xray/
├── train/
│   ├── NORMAL/           <- .jpeg / .png / .jpg
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Then set in **both** `configs/xray_baseline.yaml` and `configs/xray_transfer.yaml`:

```yaml
data:
  root: data/raw/xray
  layout: folders
  labels: [PNEUMONIA]           # must match the class folder names
  patient_id_regex: '^(person\d+)'   # or null if filenames carry no patient ID
```

For **multi-label** datasets (e.g. NIH ChestX-ray14) use `layout: csv` and point
`csv_path`, `image_column`, `patient_column` and `label_column` at your metadata
file — see [`data/README.md`](data/README.md) §1 Layout B.

### Brain MRI → `data/raw/mri/`

Default layout — one folder per patient, masks marked by a suffix:

```
data/raw/mri/
├── TCGA_CS_4941_19960909/
│   ├── TCGA_CS_4941_19960909_1.tif
│   ├── TCGA_CS_4941_19960909_1_mask.tif
│   └── ...
└── TCGA_CS_4942_19970222/
```

Then set in `configs/mri_unet.yaml`:

```yaml
data:
  root: data/raw/mri
  layout: suffix
  mask_suffix: "_mask"
```

Mirrored `images/` + `masks/` trees are supported via `layout: parallel`.
NIfTI/DICOM volumes must be converted to 2D slices first
([`data/README.md`](data/README.md) §3).

If a path or label name is wrong, the code raises an error naming the exact path,
what it found instead, and which config key to change.

---

## 4. Notebook order

Run in this order. Each notebook depends on the previous ones in its track.

| # | Notebook | Purpose | Produces |
|---|---|---|---|
| **00** | `00_environment_check.ipynb` | Verify Python, packages, GPU, imports, seeding | — |
| **01** | `01_xray_eda_and_preparation.ipynb` | X-ray EDA + patient-level split | `data/processed/xray_manifest.csv` |
| **02** | `02_xray_training.ipynb` | Train baseline CNN **and** DenseNet-121 | checkpoints, training history |
| **03** | `03_xray_evaluation_and_gradcam.ipynb` | Test metrics, CIs, Grad-CAM | metrics CSV/JSON, figures |
| **04** | `04_mri_eda_and_preparation.ipynb` | MRI pairing + validation + split | `data/processed/mri_pairs.csv` |
| **05** | `05_mri_unet_training.ipynb` | Train U-Net with Dice+BCE | checkpoint, training history |
| **06** | `06_mri_evaluation_and_visualization.ipynb` | Dice/IoU, CIs, overlays | metrics, overlays, predicted masks |

**Tracks are independent.** 01→02→03 (X-ray) and 04→05→06 (MRI) can be run in
either order after notebook 00.

**Both training notebooks have a `SMOKE_TEST` switch.** Set it to `True` first —
it trains 1 epoch on a small subset to prove the pipeline runs end to end. Then
set it back to `False` for the real run. Smoke-test numbers are not reportable.

---

## 5. Project workflow

```
                    configs/*.yaml  ──────────────┐
                                                  │ (paths, image size, labels,
                                                  │  batch size, epochs, lr, seed)
                                                  ▼
   data/raw/xray/   ──►  [01] manifest ──► integrity checks ──► EDA
                              │                                   │
                              └──► patient-level split ──► leakage check ✓
                                          │
                                          ▼
                          data/processed/xray_manifest.csv
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                          ▼
        [02] baseline CNN (scratch)              [02] DenseNet-121 (pretrained)
                     │        weighted BCE, val-based early stopping        │
                     └────────────────────┬────────────────────┘
                                          ▼
        [03] thresholds from VAL ──► test metrics ──► bootstrap CIs ──► Grad-CAM
                                          │
                                          ▼
                            outputs/classification/<run>/


   data/raw/mri/    ──►  [04] pair discovery ──► PAIR VALIDATION ✓ ──► EDA
                                          │
                                          └──► patient-level split ──► leakage ✓
                                          ▼
                            data/processed/mri_pairs.csv
                                          │
                                          ▼
                     [05] U-Net + Dice&BCE, val-Dice early stopping
                                          │
                                          ▼
        [06] threshold from VAL ──► Dice / IoU ──► CIs by patient ──► overlays
                                          │
                                          ▼
                            outputs/segmentation/<run>/
```

Two rules hold throughout, and they are what make the results defensible:

1. **Split by patient, never by image.** Verified automatically by
   `assert_no_patient_leakage`, which raises rather than warns.
2. **The test set is used exactly once, for reporting.** Every threshold, epoch
   and architecture decision is made on validation data.

Each run directory contains:

```
outputs/<task>/<run_name>/
├── config_used.yaml        exact config that produced this run
├── run_metadata.json       timestamp, Python/PyTorch/CUDA/GPU, best epoch
├── models/                 best + last checkpoints (git-ignored)
├── metrics/                training history, thresholds, test metrics, CIs
├── figures/                curves, ROC/PR, confusion matrices, Grad-CAM, overlays
└── predictions/            per-image predictions, predicted masks
```

---

## 6. Configuration

Nothing is hard-coded in the notebooks. Four YAML files:

| File | Purpose |
|---|---|
| `configs/base.yaml` | shared defaults: `seed`, `deterministic`, `device`, workers, AMP |
| `configs/xray_baseline.yaml` | simple CNN from scratch |
| `configs/xray_transfer.yaml` | pretrained DenseNet-121 |
| `configs/mri_unet.yaml` | U-Net segmentation |

Each inherits `base.yaml` and overrides what it needs. Every key is commented
in-file. The values you are most likely to change:

| Key | Meaning |
|---|---|
| `data.root`, `data.layout`, `data.labels` | **where your data is and what to predict** |
| `data.patient_id_regex` | how to extract the patient ID (prevents leakage) |
| `data.image_size` | 224 for classification, 256 for segmentation |
| `train.batch_size`, `train.epochs`, `train.lr` | the usual knobs |
| `train.loss` | `weighted_bce` / `focal` — `dice_bce` / `dice` / `tversky` |
| `seed` | global random seed |
| `eval.threshold_strategy` | `f1`, `youden` or `min_recall` |

Load a config, optionally overriding values for a one-off experiment:

```python
from src.common import load_config

cfg = load_config("xray_transfer.yaml")
cfg = load_config("xray_transfer.yaml", overrides={"train": {"epochs": 3}})
```

---

## 7. Evaluation metrics

Full rationale in **[`docs/METRICS.md`](docs/METRICS.md)**. In short:

**Classification** — ROC-AUC and PR-AUC (threshold-free ranking quality),
sensitivity/specificity/precision/F1 at a threshold chosen on **validation**,
confusion matrix, and bootstrap 95% CIs. **Accuracy is secondary**: on a task
where 2% of images are positive, predicting "negative" always scores 98%
accuracy and detects nothing.

**Segmentation** — Dice (primary) and IoU/Jaccard, reported as the mean over
*tumour-containing* slices as well as over all slices, plus the per-slice
distribution and bootstrap CIs resampled **by patient**. Pixel accuracy is not
reported at all: "all background" would exceed 98%.

**Both tasks:** the threshold is selected on validation and frozen before the
test set is touched.

---

## 8. Repository layout

```
medicaldiagnostic_finalproject/
├── data/
│   ├── raw/
│   │   ├── xray/                  <- PUT CHEST X-RAY DATA HERE
│   │   └── mri/                   <- PUT BRAIN MRI DATA HERE
│   ├── processed/                 <- manifests generated by notebooks 01 / 04
│   └── README.md                  <- detailed data layout guide
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_xray_eda_and_preparation.ipynb
│   ├── 02_xray_training.ipynb
│   ├── 03_xray_evaluation_and_gradcam.ipynb
│   ├── 04_mri_eda_and_preparation.ipynb
│   ├── 05_mri_unet_training.ipynb
│   └── 06_mri_evaluation_and_visualization.ipynb
├── src/
│   ├── common/                    shared: config, paths, seeding, device, IO, plots, errors
│   ├── classification/            manifest, splits, dataset, transforms, models,
│   │                              losses, metrics, train, evaluate, gradcam
│   └── segmentation/              pairing, splits, dataset, unet, losses,
│                                  metrics, train, evaluate
├── configs/
│   ├── base.yaml
│   ├── xray_baseline.yaml
│   ├── xray_transfer.yaml
│   └── mri_unet.yaml
├── outputs/
│   ├── classification/            per-run models, metrics, figures, predictions
│   └── segmentation/
├── docs/
│   ├── ETHICS.md                  clinical limitations and required disclaimer
│   ├── METRICS.md                 why each metric, how to read it
│   ├── REPRODUCIBILITY.md         what is guaranteed and what is not
│   ├── SCREENSHOT_CHECKLIST.md    figures to collect for the report
│   └── TROUBLESHOOTING.md         common errors and fixes
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 9. Reproducibility

Seeds are set for Python, NumPy and PyTorch; DataLoader shuffling and workers are
seeded; patient splits are assigned by hashing the patient ID with the seed, so
adding images does not reshuffle unrelated patients; every run snapshots its
config and environment.

Bit-exact reproduction across different GPUs, CUDA versions or PyTorch builds is
**not** achievable, and mixed precision introduces small accumulation
differences. Both limits are documented in
**[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** — state them in the
report rather than claiming exact reproducibility.

---

## 10. Ethical and clinical limitations

Read **[`docs/ETHICS.md`](docs/ETHICS.md)** in full before writing conclusions.
The essentials:

- This is coursework. It is not a medical device, is not clinically validated,
  and cannot diagnose anyone.
- Results come from a single public dataset; generalisation to other hospitals,
  scanners or populations is unknown and untested.
- Public chest X-ray labels are often text-mined from reports and contain errors;
  segmentation masks reflect a single annotator's interpretation.
- Models readily learn shortcuts (text markers, tubes, scanner artefacts) rather
  than pathology — the Grad-CAM section exists partly to make this visible.
- Dice and AUC measure statistical agreement, not clinical usefulness.

Required disclaimer for the report and title slide:

> This work is an academic prototype developed for coursework. It is not a
> medical device, has not been clinically validated, and must not be used for
> diagnosis or any clinical decision-making. All results are limited to the
> specific public dataset used and do not generalise to clinical practice.

---

## 11. Screenshot checklist

**[`docs/SCREENSHOT_CHECKLIST.md`](docs/SCREENSHOT_CHECKLIST.md)** lists every
figure and cell output worth capturing, with its exact output path. The items
that most often separate a strong submission from an average one:

- the **patient leakage check** output (both tasks);
- the **pair validation** output (MRI);
- the **threshold sweep** figures — they justify the operating point;
- **confidence intervals** on the headline metrics;
- **failure cases**: Grad-CAM on an FP/FN, and worst-case segmentation overlays.

---

## 12. Troubleshooting

Common errors and fixes: **[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)**.

Quick answers:

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run notebooks from `notebooks/` with the project kernel; the bootstrap cell handles `sys.path`. |
| `DataNotFoundError` | The message names the path and the config key — fix `data.root` in the config. |
| `DataLayoutError: No images matched the configured labels` | `data.labels` must match your class folder names. |
| CUDA out of memory | Lower `train.batch_size`, then `data.image_size`, then `model.features`. |
| Dice stays ~0 | Re-run notebook 04's pair validation; check the augmentation alignment cell in notebook 05. |
| Training is extremely slow | Confirm section 3 of notebook 00 reports CUDA available. |

---

## Citation

Cite the datasets you use, with their licences, in the report. This repository
contains no data and no pretrained medical model — only code.
