"""Phase 0 baseline: hash every protected classification artifact and source file.

Writes outputs/segmentation/_audit/classification_baseline_<stamp>.json so the
Phase 9 verification step can prove the classification workstream was untouched.

Read-only with respect to every protected path.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROTECTED_TREES = [
    "outputs/classification",
    "data/raw/xray",
    "data/processed/xray",
    "src/classification",
    "src/common",
    "outputs/figures/xray",
    "outputs/metrics/xray",
    "outputs/models/xray",
]
PROTECTED_GLOBS = ["configs/xray_*"]
PROTECTED_FILES = [
    "app/streamlit_app.py",
    "app/assets/styles.css",
    "configs/base.yaml",
    "requirements.txt",
    "run_xray_finetune_efficientnet_b0.py",
    "scripts/run_xray_baseline_cnn.py",
    "scripts/run_xray_finetune_densenet.py",
    "scripts/run_xray_finetune_vit.py",
    "src/segmentation/dataset.py",
    "src/segmentation/evaluate.py",
    "src/segmentation/losses.py",
    "src/segmentation/metrics.py",
    "src/segmentation/pairing.py",
    "src/segmentation/splits.py",
    "src/segmentation/train.py",
    "src/segmentation/unet.py",
    "src/segmentation/__init__.py",
    "configs/mri_unet.yaml",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict:
    entries: dict[str, dict] = {}
    targets: list[Path] = []

    for tree in PROTECTED_TREES:
        root = REPO / tree
        if root.is_dir():
            targets += [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for pattern in PROTECTED_GLOBS:
        targets += [p for p in REPO.glob(pattern) if p.is_file()]
    for rel in PROTECTED_FILES:
        p = REPO / rel
        if p.is_file():
            targets.append(p)

    for path in sorted(set(targets)):
        rel = path.relative_to(REPO).as_posix()
        entries[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return entries


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "outputs" / "segmentation" / "_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
    ).stdout
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()

    entries = collect()
    payload = {
        "captured_utc": stamp,
        "git_head": git_head,
        "git_status_porcelain": git_status,
        "git_clean": git_status.strip() == "",
        "file_count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries.values()),
        "files": entries,
    }

    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    out = out_dir / f"classification_{label}_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[phase0] wrote {out.relative_to(REPO).as_posix()}")
    print(f"[phase0] git HEAD          : {git_head}")
    print(f"[phase0] git clean         : {payload['git_clean']}")
    print(f"[phase0] protected files   : {len(entries)}")
    print(f"[phase0] protected bytes   : {payload['total_bytes']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
