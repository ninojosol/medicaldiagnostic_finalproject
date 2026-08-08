# =============================================================================
# daily_push.ps1 - Safe one-command daily Git sync for this project
# =============================================================================
# What it does (in order):
#   1. Verify this folder is a Git repo on branch "main" with an "origin" remote
#   2. Show git status --short
#   3. If nothing to sync locally, exit successfully
#   4. git pull --rebase origin main  (BEFORE staging)
#   5. Stage with git add -A, block unsafe medical/model paths
#   6. Commit with "Daily sync: YYYY-MM-DD HH:mm" (local Windows time)
#   7. Push to origin/main and print a success summary
#
# Safety rules:
#   - Never force-push, reset, discard, or delete local files
#   - On blocked paths: unstage only (git restore --staged .)
#   - On rebase conflict: stop and tell you how to fix it manually
#   - Does not ask for or store GitHub passwords/tokens
#   - Does not run git init or change remotes
# =============================================================================

$ErrorActionPreference = 'Stop'

# Exact success/idle message uses an em dash (U+2014), built at runtime so this
# script stays ASCII-safe for Windows PowerShell without relying on a UTF-8 BOM.
$EmDash = [char]0x2014
$NothingToCommitMsg = "Nothing to commit $EmDash repository is already up to date."

function Write-Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host $Message -ForegroundColor Green
}

function Write-Err([string]$Message) {
    Write-Host $Message -ForegroundColor Red
}

function Write-Warn([string]$Message) {
    Write-Host $Message -ForegroundColor Yellow
}

function Test-IsUnsafeStagedPath {
    # Returns $true if a staged path matches any blocked pattern.
    # Paths are matched with forward slashes for consistency on Windows.
    param([Parameter(Mandatory = $true)][string]$Path)

    $p = $Path -replace '\\', '/'

    # Virtual environment
    if ($p -match '(^|/)\.venv(/|$)') { return $true }

    # Dataset directories that must never be committed
    if ($p -match '(^|/)data/raw(/|$)') { return $true }
    if ($p -match '(^|/)data/processed(/|$)') { return $true }

    # Medical imaging formats
    if ($p -match '\.dcm$') { return $true }
    if ($p -match '\.nii$') { return $true }
    if ($p -match '\.nii\.gz$') { return $true }
    if ($p -match '\.mhd$') { return $true }
    if ($p -match '\.raw$') { return $true }

    # Model weights / checkpoints / serialized objects
    if ($p -match '\.pt$') { return $true }
    if ($p -match '\.pth$') { return $true }
    if ($p -match '\.ckpt$') { return $true }
    if ($p -match '\.onnx$') { return $true }
    if ($p -match '\.safetensors$') { return $true }
    if ($p -match '\.h5$') { return $true }
    if ($p -match '\.pkl$') { return $true }
    if ($p -match '\.joblib$') { return $true }

    # Prediction mask outputs (any nested run folder)
    if ($p -match '(^|/)outputs/.+/predictions/masks(/|$)') { return $true }

    # NumPy arrays
    if ($p -match '\.npy$') { return $true }
    if ($p -match '\.npz$') { return $true }

    return $false
}

# Resolve project root = parent of the scripts/ folder that contains this file
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location -LiteralPath $ProjectRoot

Write-Host ""
Write-Info "=== Daily Git Sync ==="
Write-Host "Project root: $ProjectRoot"
Write-Host ""

try {
    # -------------------------------------------------------------------------
    # 1) Confirm we are inside a Git repository
    # -------------------------------------------------------------------------
    $insideRepo = $null
    try {
        $insideRepo = (& git rev-parse --is-inside-work-tree 2>$null)
    } catch {
        $insideRepo = $null
    }

    if ($insideRepo -ne 'true') {
        Write-Err "ERROR: This folder is not a Git repository."
        Write-Err "Expected a .git directory under: $ProjectRoot"
        Write-Err "This script will not run 'git init' or change remotes."
        exit 1
    }

    # -------------------------------------------------------------------------
    # 2) Confirm origin remote exists (do not create or modify remotes)
    # -------------------------------------------------------------------------
    $originUrl = $null
    try {
        $originUrl = (& git remote get-url origin 2>$null)
    } catch {
        $originUrl = $null
    }

    if (-not $originUrl) {
        Write-Err "ERROR: Remote 'origin' was not found."
        Write-Err "Configure origin yourself (this script will not add remotes)."
        exit 1
    }

    # -------------------------------------------------------------------------
    # 3) Confirm current branch is main
    # -------------------------------------------------------------------------
    $currentBranch = (& git branch --show-current).Trim()
    if ($currentBranch -ne 'main') {
        Write-Err "ERROR: Current branch is '$currentBranch', but this script only syncs 'main'."
        Write-Err "Switch to main first, then re-run this script."
        exit 1
    }

    Write-Ok "Repository OK | branch: main | origin: $originUrl"
    Write-Host ""

    # -------------------------------------------------------------------------
    # 4) Show status before doing anything
    # -------------------------------------------------------------------------
    Write-Info "--- git status --short (before sync) ---"
    & git status --short
    Write-Host ""

    $statusBefore = & git status --porcelain
    $hasLocalChanges = -not [string]::IsNullOrWhiteSpace((@($statusBefore) -join "`n").Trim())

    if (-not $hasLocalChanges) {
        Write-Ok $NothingToCommitMsg
        exit 0
    }

    # -------------------------------------------------------------------------
    # 5) Pull / rebase BEFORE staging local changes
    #    Stops immediately on failure or conflicts. Never force-pushes or resets.
    # -------------------------------------------------------------------------
    Write-Info "Pulling with rebase: git pull --rebase origin main"
    & git pull --rebase origin main
    if ($LASTEXITCODE -ne 0) {
        Write-Err "ERROR: git pull --rebase failed or has conflicts."
        Write-Host ""
        Write-Warn "Resolve the conflict manually, then continue:"
        Write-Warn "  1. Open conflicted files and fix them"
        Write-Warn '  2. git add <resolved-files>'
        Write-Warn "  3. git rebase --continue"
        Write-Warn "To cancel the rebase instead: git rebase --abort"
        Write-Host ""
        Write-Warn "This script will NOT force-push, reset, discard, or overwrite files."
        exit 1
    }
    Write-Ok "Pull/rebase succeeded."
    Write-Host ""

    # Re-check working tree after pull (still may have local uncommitted edits)
    $statusAfterPull = & git status --porcelain
    if ([string]::IsNullOrWhiteSpace((@($statusAfterPull) -join "`n").Trim())) {
        Write-Ok $NothingToCommitMsg
        exit 0
    }

    # -------------------------------------------------------------------------
    # 6) Stage all changes (respects .gitignore)
    # -------------------------------------------------------------------------
    Write-Info "Staging changes: git add -A"
    & git add -A
    if ($LASTEXITCODE -ne 0) {
        Write-Err "ERROR: git add -A failed."
        exit 1
    }

    $stagedNames = @(& git diff --cached --name-only)
    if ($stagedNames.Count -eq 0 -or [string]::IsNullOrWhiteSpace(($stagedNames -join '').Trim())) {
        Write-Ok $NothingToCommitMsg
        exit 0
    }

    # -------------------------------------------------------------------------
    # 7) Safety gate: block medical data, models, venv, and large artifacts
    # -------------------------------------------------------------------------
    $unsafeFiles = New-Object System.Collections.Generic.List[string]
    foreach ($path in $stagedNames) {
        if (Test-IsUnsafeStagedPath -Path $path) {
            [void]$unsafeFiles.Add($path)
        }
    }

    if ($unsafeFiles.Count -gt 0) {
        Write-Err "ERROR: Unsafe staged files detected. Commit blocked."
        Write-Host ""
        Write-Warn "The following staged paths look like data, models, or environments"
        Write-Warn "and must not be pushed. Unstaging only (local files are kept):"
        Write-Host ""
        foreach ($f in $unsafeFiles) {
            Write-Host "  - $f" -ForegroundColor Yellow
        }
        Write-Host ""

        # Unstage everything; never delete working-tree files
        & git restore --staged .
        if ($LASTEXITCODE -ne 0) {
            Write-Err "ERROR: Failed to unstage with 'git restore --staged .'."
            Write-Warn "Run manually: git restore --staged ."
            exit 1
        }

        Write-Ok "Staging cleared. Your local files were NOT deleted."
        Write-Warn "Remove those paths from the commit set (or rely on .gitignore),"
        Write-Warn "then run this script again."
        exit 1
    }

    # -------------------------------------------------------------------------
    # 8) Commit with automatic local-time message
    # -------------------------------------------------------------------------
    # Exact format: Daily sync: YYYY-MM-DD HH:mm  (local Windows clock)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
    $commitMessage = "Daily sync: $timestamp"

    Write-Info "Creating commit: $commitMessage"
    & git commit -m $commitMessage
    if ($LASTEXITCODE -ne 0) {
        Write-Err "ERROR: git commit failed."
        exit 1
    }

    $commitHash = (& git rev-parse --short HEAD).Trim()

    # -------------------------------------------------------------------------
    # 9) Push to origin/main (no force)
    # -------------------------------------------------------------------------
    Write-Info "Pushing: git push origin main"
    & git push origin main
    if ($LASTEXITCODE -ne 0) {
        Write-Err "ERROR: git push origin main failed."
        Write-Warn "Your commit exists locally ($commitHash) but was not pushed."
        Write-Warn "This script never force-pushes. Fix remote access, then push manually:"
        Write-Warn "  git push origin main"
        exit 1
    }

    # -------------------------------------------------------------------------
    # 10) Success summary
    # -------------------------------------------------------------------------
    Write-Host ""
    Write-Ok "=== Sync complete ==="
    Write-Ok "Commit:  $commitHash"
    Write-Ok "Message: $commitMessage"
    Write-Ok "Branch:  main"
    Write-Ok "Remote:  $originUrl"
    Write-Host ""
    exit 0
}
catch {
    $errText = $_.Exception.Message
    Write-Err "ERROR: Unexpected failure: $errText"
    exit 1
}
