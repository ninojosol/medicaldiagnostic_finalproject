#Requires -Version 5.1
<#
.SYNOPSIS
  Build presentation-assets/ + presentation-assets-v1.0.zip with SHA256SUMS.txt.
  Run on the machine that already has trained checkpoints and MRI presentation samples.
#>
param(
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $OutDir) {
    $OutDir = Join-Path $Root "presentation-assets"
}

$ZipPath = Join-Path $Root "presentation-assets-v1.0.zip"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Copy-Required([string]$Src, [string]$Dst) {
    if (-not (Test-Path $Src)) {
        Write-Error "Required source missing: $Src"
    }
    Ensure-Dir (Split-Path $Dst -Parent)
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
    Write-Host "  + $Src"
}

if (Test-Path $OutDir) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
Ensure-Dir $OutDir

Write-Host "==> Staging classification checkpoints"
$clsRoot = Join-Path $OutDir "classification"
Ensure-Dir (Join-Path $clsRoot "checkpoints")
Ensure-Dir (Join-Path $clsRoot "configs")
Ensure-Dir (Join-Path $clsRoot "thresholds")
Ensure-Dir (Join-Path $clsRoot "metadata")

$clsMaps = @(
    @{
        Pt = "outputs\classification\xray_baseline_cnn_from_scratch_multilabel_320\models\xray_baseline_cnn_from_scratch_multilabel_320_best.pt"
        Name = "xray_baseline_cnn_from_scratch_multilabel_320_best.pt"
        Run = "outputs\classification\xray_baseline_cnn_from_scratch_multilabel_320"
    },
    @{
        Pt = "outputs\classification\xray_finetuned_densenet_multilabel_320\models\xray_finetuned_densenet_multilabel_320_best.pt"
        Name = "xray_finetuned_densenet_multilabel_320_best.pt"
        Run = "outputs\classification\xray_finetuned_densenet_multilabel_320"
    },
    @{
        Pt = "outputs\classification\xray_finetuned_efficientnet_b0_multilabel_320\models\xray_finetuned_efficientnet_b0_multilabel_320_best.pt"
        Name = "xray_finetuned_efficientnet_b0_multilabel_320_best.pt"
        Run = "outputs\classification\xray_finetuned_efficientnet_b0_multilabel_320"
    },
    @{
        Pt = "outputs\classification\xray_finetuned_vit_b16_multilabel_224\models\xray_finetuned_vit_b16_multilabel_224_best.pt"
        Name = "xray_finetuned_vit_b16_multilabel_224_best.pt"
        Run = "outputs\classification\xray_finetuned_vit_b16_multilabel_224"
    }
)

foreach ($m in $clsMaps) {
    Copy-Required (Join-Path $Root $m.Pt) (Join-Path $clsRoot "checkpoints\$($m.Name)")
    $run = Join-Path $Root $m.Run
    Copy-Required (Join-Path $run "config_used.yaml") (Join-Path $clsRoot "configs\$($m.Name -replace '_best\.pt$','_config_used.yaml')")
    Copy-Required (Join-Path $run "metrics\thresholds.json") (Join-Path $clsRoot "thresholds\$($m.Name -replace '_best\.pt$','_thresholds.json')")
    Copy-Required (Join-Path $run "run_metadata.json") (Join-Path $clsRoot "metadata\$($m.Name -replace '_best\.pt$','_run_metadata.json')")
}

Write-Host "==> Staging segmentation checkpoint + samples"
$segRoot = Join-Path $OutDir "segmentation"
$segRun = Join-Path $Root "outputs\segmentation\mri_unet_whole_tumour_2d_192"
Ensure-Dir (Join-Path $segRoot "checkpoint")
Ensure-Dir (Join-Path $segRoot "config")
Ensure-Dir (Join-Path $segRoot "metadata")
Ensure-Dir (Join-Path $segRoot "small_presentation_samples")

Copy-Required `
    (Join-Path $segRun "models\mri_unet_whole_tumour_2d_192_best.pt") `
    (Join-Path $segRoot "checkpoint\mri_unet_whole_tumour_2d_192_best.pt")
Copy-Required (Join-Path $segRun "config_used.yaml") (Join-Path $segRoot "config\config_used.yaml")
Copy-Required (Join-Path $segRun "run_metadata.json") (Join-Path $segRoot "metadata\run_metadata.json")

foreach ($split in @("segmentation_validation", "segmentation_test")) {
    $src = Join-Path $Root "data\presentation_samples\$split"
    $dst = Join-Path $segRoot "small_presentation_samples\$split"
    if (-not (Test-Path $src)) {
        Write-Error "Missing presentation samples: $src"
    }
    Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
    Write-Host "  + $src"
}

Write-Host "==> Writing SHA256SUMS.txt"
$sumsPath = Join-Path $OutDir "SHA256SUMS.txt"
$lines = New-Object System.Collections.Generic.List[string]
Get-ChildItem -LiteralPath $OutDir -Recurse -File |
    Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
    Sort-Object FullName |
    ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $rel = $_.FullName.Substring($OutDir.Length).TrimStart("\", "/") -replace "\\", "/"
        $lines.Add("$hash  $rel")
    }
$lines | Set-Content -LiteralPath $sumsPath -Encoding ascii

Write-Host "==> Creating zip: $ZipPath"
if (Test-Path $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
Compress-Archive -Path $OutDir -DestinationPath $ZipPath -CompressionLevel Optimal

$zipMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "Done. Bundle: $ZipPath ($zipMb MB)"
Write-Host "Upload with: gh release create v1.0-presentation-demo `"$ZipPath`" --title `"v1.0 presentation demo assets`" --notes-file docs/TEAM_SETUP.md"
