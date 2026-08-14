# Presentation assets — external bundle (fallback)

Use this when the GitHub Release `v1.0-presentation-demo` cannot be downloaded.
Prefer the Release when available: see [`docs/TEAM_SETUP.md`](docs/TEAM_SETUP.md).

## Bundle name

`presentation-assets-v1.0.zip` (folder inside: `presentation-assets/`)

## Layout

```text
presentation-assets/
├── classification/
│   ├── checkpoints/
│   │   ├── xray_baseline_cnn_from_scratch_multilabel_320_best.pt
│   │   ├── xray_finetuned_densenet_multilabel_320_best.pt
│   │   ├── xray_finetuned_efficientnet_b0_multilabel_320_best.pt
│   │   └── xray_finetuned_vit_b16_multilabel_224_best.pt
│   ├── configs/          # optional copies; repo already has config_used.yaml
│   ├── thresholds/       # optional; repo already has metrics/thresholds.json
│   └── metadata/         # optional; repo already has run_metadata.json
├── segmentation/
│   ├── checkpoint/
│   │   └── mri_unet_whole_tumour_2d_192_best.pt
│   ├── config/
│   │   └── config_used.yaml
│   ├── metadata/
│   │   └── run_metadata.json
│   └── small_presentation_samples/
│       ├── segmentation_validation/
│       └── segmentation_test/
└── SHA256SUMS.txt
```

## Destination paths (after install)

Run from the **repository root**:

```powershell
.\scripts\install_presentation_assets.ps1 -ZipPath path\to\presentation-assets-v1.0.zip
```

Manual mapping:

| Bundle path | Project path |
|-------------|--------------|
| `classification/checkpoints/xray_baseline_*_best.pt` | `outputs/classification/xray_baseline_cnn_from_scratch_multilabel_320/models/` |
| `classification/checkpoints/xray_finetuned_densenet_*_best.pt` | `outputs/classification/xray_finetuned_densenet_multilabel_320/models/` |
| `classification/checkpoints/xray_finetuned_efficientnet_*_best.pt` | `outputs/classification/xray_finetuned_efficientnet_b0_multilabel_320/models/` |
| `classification/checkpoints/xray_finetuned_vit_*_best.pt` | `outputs/classification/xray_finetuned_vit_b16_multilabel_224/models/` |
| `segmentation/checkpoint/mri_unet_*_best.pt` | `outputs/segmentation/mri_unet_whole_tumour_2d_192/models/` |
| `segmentation/small_presentation_samples/segmentation_*` | `data/presentation_samples/segmentation_*` |

## SHA-256

Every build writes `SHA256SUMS.txt` at the bundle root (37 files in `v1.0`).
`scripts/install_presentation_assets.ps1` verifies all hashes before copying.

Rebuild locally (maintainers only):

```powershell
.\scripts\build_presentation_assets.ps1
```

Current `v1.0` zip: `presentation-assets-v1.0.zip` (~507 MB).

## Exclusions (never put these in the shared bundle)

- Full raw X-ray / MRI archives  
- `data/processed/mri/slice_cache/`  
- `*.npz` / prediction mask dumps  
- Training `*_last.pt` / stage-1 checkpoints (not needed for the app)  
- Secrets, `.env`, personal absolute paths  

## Size note

ViT-B/16 best checkpoint is ~327 MB. Total zip is typically ~500–550 MB compressed poorly (already compressed weights + NIfTI). Plan disk accordingly.
