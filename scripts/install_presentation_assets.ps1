#Requires -Version 5.1
<#
.SYNOPSIS
  Install presentation-assets into the project paths the Streamlit app expects.
.PARAMETER ZipPath
  Path to presentation-assets-v1.0.zip
.PARAMETER FromRelease
  Download the zip from GitHub release v1.0-presentation-demo
.PARAMETER SkipHashCheck
  Skip SHA-256 verification (not recommended)
#>
param(
    [string]$ZipPath = "",
    [switch]$FromRelease,
    [switch]$SkipHashCheck
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ReleaseTag = "v1.0-presentation-demo"
$AssetName = "presentation-assets-v1.0.zip"
$Staging = Join-Path $Root "presentation-assets"
$DownloadDir = Join-Path $Root "dist"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

if ($FromRelease) {
    Ensure-Dir $DownloadDir
    $ZipPath = Join-Path $DownloadDir $AssetName
    Write-Host "==> Downloading $AssetName from release $ReleaseTag"
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $gh) {
        # Common install location after winget
        $candidate = Join-Path $env:ProgramFiles "GitHub CLI\gh.exe"
        if (Test-Path $candidate) {
            $gh = Get-Command $candidate
        }
    }
    if (-not $gh) {
        Write-Error "GitHub CLI (gh) not found. Download the zip manually from the Releases page, then re-run with -ZipPath."
    }
    & $gh.Source release download $ReleaseTag -R ninojosol/medicaldiagnostic_finalproject -p $AssetName -D $DownloadDir --clobber
    if (-not (Test-Path $ZipPath)) {
        Write-Error "Download failed: $ZipPath not found."
    }
}

if (-not $ZipPath) {
    Write-Error "Provide -ZipPath or -FromRelease. See docs/TEAM_SETUP.md and TEAM_ASSETS.md."
}
if (-not (Test-Path $ZipPath)) {
    Write-Error "Zip not found: $ZipPath"
}

Write-Host "==> Extracting $ZipPath"
if (Test-Path $Staging) {
    Remove-Item -LiteralPath $Staging -Recurse -Force
}
Expand-Archive -LiteralPath $ZipPath -DestinationPath $Root -Force
if (-not (Test-Path $Staging)) {
    # zip may contain a single top-level folder already named presentation-assets
    $inner = Get-ChildItem $Root -Directory | Where-Object { $_.Name -like "presentation-assets*" } | Select-Object -First 1
    if ($inner -and (Test-Path (Join-Path $inner.FullName "SHA256SUMS.txt"))) {
        $Staging = $inner.FullName
    } else {
        Write-Error "Expected folder presentation-assets/ after extract."
    }
}

$sumsFile = Join-Path $Staging "SHA256SUMS.txt"
if (-not (Test-Path $sumsFile)) {
    Write-Error "SHA256SUMS.txt missing inside the bundle."
}

if (-not $SkipHashCheck) {
    Write-Host "==> Verifying SHA-256"
    $failures = 0
    Get-Content $sumsFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line) { return }
        $parts = $line -split "\s+", 2
        if ($parts.Count -lt 2) { return }
        $expected = $parts[0].ToLowerInvariant()
        $rel = $parts[1].Trim() -replace "/", "\"
        $file = Join-Path $Staging $rel
        if (-not (Test-Path $file)) {
            Write-Warning "Missing listed file: $rel"
            $failures++
            return
        }
        $actual = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            Write-Warning "Hash mismatch: $rel"
            $failures++
        }
    }
    if ($failures -gt 0) {
        Write-Error "$failures checksum failure(s). Aborting install."
    }
    Write-Host "  checksums OK"
}

function Install-File([string]$Src, [string]$Dst) {
    if (-not (Test-Path $Src)) { Write-Error "Missing in bundle: $Src" }
    Ensure-Dir (Split-Path $Dst -Parent)
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
    Write-Host "  -> $Dst"
}

Write-Host "==> Installing classification checkpoints"
$ckpt = Join-Path $Staging "classification\checkpoints"
Install-File (Join-Path $ckpt "xray_baseline_cnn_from_scratch_multilabel_320_best.pt") `
    (Join-Path $Root "outputs\classification\xray_baseline_cnn_from_scratch_multilabel_320\models\xray_baseline_cnn_from_scratch_multilabel_320_best.pt")
Install-File (Join-Path $ckpt "xray_finetuned_densenet_multilabel_320_best.pt") `
    (Join-Path $Root "outputs\classification\xray_finetuned_densenet_multilabel_320\models\xray_finetuned_densenet_multilabel_320_best.pt")
Install-File (Join-Path $ckpt "xray_finetuned_efficientnet_b0_multilabel_320_best.pt") `
    (Join-Path $Root "outputs\classification\xray_finetuned_efficientnet_b0_multilabel_320\models\xray_finetuned_efficientnet_b0_multilabel_320_best.pt")
Install-File (Join-Path $ckpt "xray_finetuned_vit_b16_multilabel_224_best.pt") `
    (Join-Path $Root "outputs\classification\xray_finetuned_vit_b16_multilabel_224\models\xray_finetuned_vit_b16_multilabel_224_best.pt")

Write-Host "==> Installing segmentation checkpoint + samples"
Install-File (Join-Path $Staging "segmentation\checkpoint\mri_unet_whole_tumour_2d_192_best.pt") `
    (Join-Path $Root "outputs\segmentation\mri_unet_whole_tumour_2d_192\models\mri_unet_whole_tumour_2d_192_best.pt")

# Ensure run metadata exists (repo may already have it; bundle copy is authoritative for missing clones)
$segMetaSrc = Join-Path $Staging "segmentation\metadata\run_metadata.json"
$segCfgSrc = Join-Path $Staging "segmentation\config\config_used.yaml"
$segRun = Join-Path $Root "outputs\segmentation\mri_unet_whole_tumour_2d_192"
Ensure-Dir $segRun
if (Test-Path $segMetaSrc) {
    Install-File $segMetaSrc (Join-Path $segRun "run_metadata.json")
}
if (Test-Path $segCfgSrc) {
    Install-File $segCfgSrc (Join-Path $segRun "config_used.yaml")
}

$presRoot = Join-Path $Root "data\presentation_samples"
Ensure-Dir $presRoot
foreach ($split in @("segmentation_validation", "segmentation_test")) {
    $src = Join-Path $Staging "segmentation\small_presentation_samples\$split"
    $dst = Join-Path $presRoot $split
    if (-not (Test-Path $src)) { Write-Error "Bundle missing samples: $src" }
    if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
    Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
    Write-Host "  -> $dst"
}

Write-Host ""
Write-Host "Assets installed. Run: .\scripts\run_app.ps1"
