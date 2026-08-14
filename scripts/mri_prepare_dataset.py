"""Phase 1 — extract and inventory the Decathlon Task01_BrainTumour archive.

Usage::

    python scripts/mri_prepare_dataset.py [--skip-extract] [--limit N]

Reads the downloaded tar from data/raw/mri/decathlon_task01_brain_tumour/_download/,
extracts imagesTr / labelsTr / dataset.json, verifies every labelled case, and
writes data/processed/mri/dataset_inventory.json.

Leaves the legacy data/raw/mri/brats/ sample completely untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.constants import (  # noqa: E402
    DATASET_ID,
    DATASET_JSON,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_URL,
    IMAGES_TR,
    INVENTORY_JSON,
    LABELS_TR,
    PROCESSED_MRI_DIR,
    RAW_DOWNLOAD_DIR,
    RAW_MRI_DIR,
    SLICE_AXIS,
)
from src.segmentation.mri.inventory import (  # noqa: E402
    build_inventory,
    read_dataset_json,
    verify_dataset_json,
)

TAR_NAME = "Task01_BrainTumour.tar"


def extract_archive(tar_path: Path) -> None:
    """Extract the archive flat into RAW_MRI_DIR (dropping the top-level folder)."""
    print(f"[extract] {tar_path.name} ({tar_path.stat().st_size / 1e9:.2f} GB)")
    RAW_MRI_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n = 0
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            rel = Path(*parts[1:])          # strip "Task01_BrainTumour/"
            if rel.name.startswith("._"):   # macOS resource-fork stubs
                continue
            if rel.parts[0] not in {"imagesTr", "labelsTr"} and rel.name != "dataset.json":
                continue
            dest = RAW_MRI_DIR / rel
            if dest.exists() and dest.stat().st_size == member.size:
                n += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with dest.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            n += 1
            if n % 100 == 0:
                print(f"  [extract] {n} files ({time.time() - started:.0f}s)", flush=True)
    print(f"[extract] done: {n} files in {time.time() - started:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="inventory only the first N cases (debug)")
    args = ap.parse_args()

    tar_path = RAW_DOWNLOAD_DIR / TAR_NAME

    if not args.skip_extract:
        if not tar_path.exists():
            print(f"ERROR: archive not found at {tar_path}", file=sys.stderr)
            return 2
        extract_archive(tar_path)

    if not DATASET_JSON.exists():
        print(f"ERROR: dataset.json missing at {DATASET_JSON}", file=sys.stderr)
        return 2

    # ---- report the exact source label mapping -----------------------------
    meta = read_dataset_json()
    print("\n[dataset.json] as shipped by the source:")
    print(f"  name        : {meta.get('name')}")
    print(f"  description : {meta.get('description')}")
    print(f"  reference   : {meta.get('reference')}")
    print(f"  licence     : {meta.get('licence') or meta.get('license')}")
    print(f"  release     : {meta.get('release')}")
    print(f"  tensorImageSize: {meta.get('tensorImageSize')}")
    print(f"  modality    : {meta.get('modality')}")
    print(f"  labels      : {meta.get('labels')}")
    print(f"  numTraining : {meta.get('numTraining')}   numTest: {meta.get('numTest')}")

    ok, problems = verify_dataset_json(meta)
    if not ok:
        print("\nERROR: shipped metadata does not match the assumed mapping:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 3
    print("\n[verify] shipped label mapping and modality count match the project assumption.")

    # ---- full inventory ----------------------------------------------------
    print("\n[inventory] verifying every labelled case (this reads every voxel)...")
    started = time.time()
    inv = build_inventory(IMAGES_TR, LABELS_TR, slice_axis=SLICE_AXIS, progress=True)
    inv["inventory_seconds"] = round(time.time() - started, 1)
    inv["download_url"] = DATASET_URL
    inv["download_date"] = datetime.fromtimestamp(
        tar_path.stat().st_mtime, tz=timezone.utc
    ).date().isoformat() if tar_path.exists() else datetime.now(timezone.utc).date().isoformat()
    inv["dataset_name"] = DATASET_NAME
    inv["licence"] = DATASET_LICENCE
    inv["dataset_json"] = {
        "name": meta.get("name"),
        "description": meta.get("description"),
        "reference": meta.get("reference"),
        "licence": meta.get("licence") or meta.get("license"),
        "release": meta.get("release"),
        "modality": meta.get("modality"),
        "labels": meta.get("labels"),
        "numTraining": meta.get("numTraining"),
        "numTest": meta.get("numTest"),
        "tensorImageSize": meta.get("tensorImageSize"),
    }

    PROCESSED_MRI_DIR.mkdir(parents=True, exist_ok=True)
    INVENTORY_JSON.write_text(json.dumps(inv, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("PHASE 1 INVENTORY")
    print("=" * 70)
    print(f"dataset id            : {DATASET_ID}")
    print(f"source url            : {DATASET_URL}")
    print(f"download date (UTC)   : {inv['download_date']}")
    print(f"licence               : {DATASET_LICENCE}")
    print(f"matched image/label   : {inv['n_matched_pairs']}")
    print(f"images without label  : {inv['n_images_without_label']}")
    print(f"labels without image  : {inv['n_labels_without_image']}")
    print(f"verified OK           : {inv['n_ok']}")
    print(f"failed verification   : {inv['n_failed']}")
    print(f"image shapes          : {inv['image_shape_histogram']}")
    print(f"observed label values : {inv['observed_label_values']}")
    print(f"documented mapping    : {inv['documented_label_mapping']}")
    print(f"modalities            : {inv['modality_names']}")
    print(f"tumour slices / total : {inv['total_tumour_slices']} / {inv['total_slices']}")
    print(f"inventory time        : {inv['inventory_seconds']}s")
    if inv["failures"]:
        print("\nFAILURES:")
        for f in inv["failures"][:20]:
            print(f"  {f['case_id']}: {f['problems']}")
    print(f"\nwritten -> {INVENTORY_JSON.relative_to(REPO).as_posix()}")
    return 0 if inv["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
