# AI-Assisted Medical Image Analysis

**1) Chest X-ray Disease Classification  ·  2) Brain MRI Tumour Segmentation**

Academic PyTorch + Streamlit prototype for a master's final project.

> **Not a medical device.** Educational only — never use for real patient care. See [`docs/ETHICS.md`](docs/ETHICS.md).

---

## Team quick start (Windows, Saturday presentation)

Teammates who are GitHub collaborators:

```powershell
git clone https://github.com/ninojosol/medicaldiagnostic_finalproject.git
cd medicaldiagnostic_finalproject
.\scripts\setup_windows.ps1          # Python 3.11 venv + torch + requirements
.\scripts\install_presentation_assets.ps1 -FromRelease   # checkpoints + MRI demo samples
.\scripts\run_app.ps1                # streamlit run app/streamlit_app.py
```

Full guide (datasets, troubleshooting, CUDA/CPU): **[`docs/TEAM_SETUP.md`](docs/TEAM_SETUP.md)**  
Asset layout / shared-folder fallback: **[`TEAM_ASSETS.md`](TEAM_ASSETS.md)**  
Release tag: **`v1.0-presentation-demo`**

| In the GitHub clone | Via Release / local only |
|---------------------|---------------------------|
| App + training/eval code | Four X-ray `*_best.pt` + MRI U-Net `*_best.pt` |
| Metrics, thresholds, figures | Small MRI presentation `.nii.gz` samples |
| Tracked X-ray presentation PNGs | Full raw X-ray / full Decathlon MRI archive |

The **full MRI dataset is not included in GitHub.**

---

## What this project does

### Part A — Chest X-ray classification

Multi-label NIH-style classification with a from-scratch CNN, DenseNet-121, EfficientNet-B0, and an experimental ViT-B/16 comparator. Validation thresholds, ROC/PR, Grad-CAM, Streamlit four-model inference demo.

### Part B — Brain MRI tumour segmentation

2D slice-wise U-Net on four MRI sequences (MSD Task01 / BraTS), whole-tumour binary masks, Dice/IoU reporting, Streamlit overlay demo on small approved presentation samples.

---

## Environment (summary)

**Python 3.11.** Install **PyTorch first**, then `requirements.txt` (torch is intentionally omitted from the requirements file so CUDA wheels are not overwritten).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or .../cpu
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Or use `.\scripts\setup_windows.ps1` / `.\scripts\run_app.ps1`.

---

## Data

Nothing under `data/raw/` is committed. Layout details: [`data/README.md`](data/README.md).  
MRI pipeline notes: [`docs/MRI_SEGMENTATION.md`](docs/MRI_SEGMENTATION.md).

---

## Notebooks & training

| Track | Notebooks / scripts |
|-------|---------------------|
| Env check | `notebooks/00_environment_check.ipynb` |
| X-ray | `01`–`03` + `scripts/run_xray_*.py` |
| MRI | `04`–`06` + `scripts/run_mri_unet_whole_tumour.py` |

Configs live in `configs/*.yaml`. Outputs land under `outputs/classification/` and `outputs/segmentation/` (weights gitignored).

---

## Docs

| Doc | Purpose |
|-----|---------|
| [`docs/TEAM_SETUP.md`](docs/TEAM_SETUP.md) | Collaborator Windows setup |
| [`docs/ETHICS.md`](docs/ETHICS.md) | Clinical / ethics limits |
| [`docs/METRICS.md`](docs/METRICS.md) | Metric rationale |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Seeds and limits |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Common errors |

---

## Citation

Cite the public datasets and licences you use in the report. This repository ships **code** (and small tracked X-ray demo PNGs); model weights and MRI demo volumes are distributed separately via the presentation Release.
