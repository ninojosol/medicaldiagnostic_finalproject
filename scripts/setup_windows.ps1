#Requires -Version 5.1
<#
.SYNOPSIS
  Create .venv (Python 3.11), install PyTorch, then requirements.txt.
.PARAMETER CpuOnly
  Install CPU-only torch wheels instead of CUDA 12.8.
#>
param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Project root: $Root"

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python launcher 'py' not found. Install Python 3.11 from python.org and re-run."
}

& py -3.11 --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python 3.11 is required. Install it, then re-run this script."
}

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating .venv"
    & py -3.11 -m venv .venv
}

$python = Join-Path $Root ".venv\Scripts\python.exe"
$pip = Join-Path $Root ".venv\Scripts\pip.exe"

Write-Host "==> Upgrading pip"
& $python -m pip install --upgrade pip

if ($CpuOnly) {
    Write-Host "==> Installing PyTorch (CPU)"
    & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    Write-Host "==> Installing PyTorch (CUDA 12.8 wheels)"
    & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
}

Write-Host "==> Installing requirements.txt"
& $pip install -r requirements.txt

Write-Host "==> Verifying imports"
& $python -c "import torch, streamlit, nibabel, pandas, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('streamlit ok')"

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next: .\scripts\install_presentation_assets.ps1 -FromRelease"
Write-Host "Then:  .\scripts\run_app.ps1"
Write-Host "Guide: docs\TEAM_SETUP.md"
