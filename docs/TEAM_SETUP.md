# Team setup (Windows) — presentation handoff

Goal: clone the repo, install Python deps, fetch the **presentation asset bundle**, and run the Streamlit app for Saturday prep.

> Academic prototype only — not a medical device. Do not use with real patient care decisions.

The **full MRI Decathlon dataset is not in GitHub**. Checkpoints and small MRI demo volumes ship via the GitHub Release (or a shared folder fallback).

---

## 1. Clone

```powershell
git clone https://github.com/ninojosol/medicaldiagnostic_finalproject.git
cd medicaldiagnostic_finalproject
```

---

## 2. Create Python 3.11 virtual environment

```powershell
py -3.11 -m venv .venv
```

If `py` is missing, install [Python 3.11](https://www.python.org/downloads/release/python-3119/) and tick **Add python.exe to PATH**, then retry.

---

## 3. Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, see [Troubleshooting](#troubleshooting).

---

## 4. Install dependencies

**PyTorch first** (do not put torch in `requirements.txt` — a plain pip install would replace CUDA wheels with CPU):

```powershell
# NVIDIA GPU (CUDA 12.8) — preferred if you have a compatible GPU
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# OR CPU only
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then:

```powershell
pip install -r requirements.txt
```

Verify:

```powershell
python -c "import torch, streamlit, nibabel; print(torch.__version__, torch.cuda.is_available(), streamlit.__version__)"
```

---

## 5. Public datasets (optional for full Data Prep panels)

| Need | Source | Local path |
|------|--------|------------|
| NIH ChestX-ray **small** course set (images + CSVs) | Course / NIH public ChestX-ray14 subset used in class | `data/raw/xray/nih/` (`images-small/`, `test.csv`, …) |
| Full MSD Task01 BrainTumour | [Medical Segmentation Decathlon](http://medicaldecathlon.com/) / MONAI S3 `Task01_BrainTumour.tar` | `data/raw/mri/decathlon_task01_brain_tumour/` |

**For Saturday presentation inference you do not need the full MRI archive.** Use the Release asset bundle below.

Processed X-ray CSVs (`data/processed/xray/*.csv`) are optional metadata for Data Prep charts. Inference Demo works from uploaded images + checkpoints.

---

## 6. Presentation checkpoints & demo assets (required)

### Preferred — GitHub Release

1. Open: https://github.com/ninojosol/medicaldiagnostic_finalproject/releases/tag/v1.0-presentation-demo  
2. Download `presentation-assets-v1.0.zip`.  
3. From the **repo root**, with the venv activated:

```powershell
.\scripts\install_presentation_assets.ps1 -ZipPath path\to\presentation-assets-v1.0.zip
```

Or let the script download it (requires `gh` auth or a public release):

```powershell
.\scripts\install_presentation_assets.ps1 -FromRelease
```

This places:

- four X-ray `*_best.pt` files under `outputs/classification/.../models/`
- MRI U-Net `*_best.pt` under `outputs/segmentation/mri_unet_whole_tumour_2d_192/models/`
- eight small BraTS presentation volumes under `data/presentation_samples/segmentation_{validation,test}/`

Checksums: `SHA256SUMS.txt` inside the zip (verified by the install script).

### Fallback — shared folder

If the Release is unavailable, follow **[`TEAM_ASSETS.md`](../TEAM_ASSETS.md)** (exact layout, destinations, SHA-256).

---

## 7. Run the app

```powershell
.\scripts\run_app.ps1
# or:
streamlit run app/streamlit_app.py
```

Browser: http://localhost:8501

---

## What is / is not in GitHub

| In the clone | Not in GitHub (Release / local only) |
|--------------|--------------------------------------|
| App + `src/` code | Model checkpoints (`.pt`) |
| Classification metrics / thresholds / figures | Full raw X-ray / MRI archives |
| X-ray presentation PNGs already tracked | MRI presentation `.nii.gz` samples |
| Lightweight MRI run metrics / figures | Slice cache, `.npz` predictions |
| Setup scripts & this guide | Secrets, `.env`, personal paths |

---

## Troubleshooting

### Python not found

- Install Python **3.11** from python.org.  
- Use `py -3.11` (Windows launcher).  
- Confirm: `py -3.11 --version`.

### PowerShell execution policy

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Or activate without changing policy:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

### PyTorch CUDA unavailable

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- If version ends in `+cpu`, reinstall with the `cu128` index URL above.  
- CPU-only still runs the demo; inference is slower (especially ViT).

### Missing checkpoint

Error text points at a `.pt` path. Run `install_presentation_assets.ps1` and confirm:

```text
outputs/classification/xray_baseline_cnn_from_scratch_multilabel_320/models/*_best.pt
outputs/classification/xray_finetuned_densenet_multilabel_320/models/*_best.pt
outputs/classification/xray_finetuned_efficientnet_b0_multilabel_320/models/*_best.pt
outputs/classification/xray_finetuned_vit_b16_multilabel_224/models/*_best.pt
outputs/segmentation/mri_unet_whole_tumour_2d_192/models/*_best.pt
```

### Missing dataset / model bundle

- Classification Data Prep may warn without `data/raw/xray/nih/images-small/` — Inference Demo still works with uploads if checkpoints are installed.  
- Segmentation sample picker needs the Release MRI samples under `data/presentation_samples/segmentation_*`.  
- Full Decathlon download is only for re-training, not for the Saturday demo.

### Port already in use

```powershell
streamlit run app/streamlit_app.py --server.port 8502
```
