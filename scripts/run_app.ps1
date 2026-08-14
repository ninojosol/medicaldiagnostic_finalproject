#Requires -Version 5.1
<#
.SYNOPSIS
  Launch the Streamlit presentation app.
.PARAMETER Port
  Server port (default 8501).
#>
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Missing .venv. Run .\scripts\setup_windows.ps1 first."
}

$missing = @()
$checks = @(
    "outputs\classification\xray_baseline_cnn_from_scratch_multilabel_320\models\xray_baseline_cnn_from_scratch_multilabel_320_best.pt",
    "outputs\classification\xray_finetuned_densenet_multilabel_320\models\xray_finetuned_densenet_multilabel_320_best.pt",
    "outputs\classification\xray_finetuned_efficientnet_b0_multilabel_320\models\xray_finetuned_efficientnet_b0_multilabel_320_best.pt",
    "outputs\classification\xray_finetuned_vit_b16_multilabel_224\models\xray_finetuned_vit_b16_multilabel_224_best.pt",
    "outputs\segmentation\mri_unet_whole_tumour_2d_192\models\mri_unet_whole_tumour_2d_192_best.pt"
)
foreach ($rel in $checks) {
    if (-not (Test-Path (Join-Path $Root $rel))) { $missing += $rel }
}
if ($missing.Count -gt 0) {
    Write-Warning "Missing checkpoint(s). Inference demos will fail until you install assets:"
    $missing | ForEach-Object { Write-Warning "  $_" }
    Write-Warning "Run: .\scripts\install_presentation_assets.ps1 -FromRelease"
    Write-Warning "See: docs\TEAM_SETUP.md"
}

Write-Host "Starting Streamlit on port $Port ..."
& $python -m streamlit run (Join-Path $Root "app\streamlit_app.py") --server.port $Port
