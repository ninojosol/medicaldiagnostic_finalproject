"""Phase 8 — assemble copied-only MRI presentation samples for the demo UI.

Creates:
    data/presentation_samples/segmentation_validation/
    data/presentation_samples/segmentation_test/

Each holds a small documented set of cases. Files are **copied** with
shutil.copyfile from the raw archive — originals are never moved, renamed or
modified, and the copies keep their original file names inside a per-case folder.

Case selection is deterministic and documented: for each split the cases are
ranked by per-case Dice from that split's frozen metrics CSV and sampled at
evenly spaced ranks (best, upper-middle, middle, lower-middle, worst-of-sample)
so the demo shows an honest spread rather than a curated highlight reel.

Usage::

    python scripts/mri_build_presentation_samples.py [--n-valid 4] [--n-test 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.constants import (  # noqa: E402
    DATASET_ID,
    DATASET_NAME,
    HELD_OUT_TEST_NOTE,
    MODALITY_NAMES,
    PRESENTATION_TEST_DIR,
    PRESENTATION_VALID_DIR,
    PROJECT_ROOT,
    SOURCE_LABELS,
    WHOLE_TUMOUR_RULE,
    run_dir,
)
from src.segmentation.mri.manifests import load_manifest  # noqa: E402


def pick_ranked(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Evenly spaced ranks across the Dice distribution (best .. worst)."""
    ranked = df.sort_values("dice", ascending=False).reset_index(drop=True)
    if len(ranked) <= n:
        return ranked
    idx = np.unique(np.linspace(0, len(ranked) - 1, n).round().astype(int))
    return ranked.iloc[idx].reset_index(drop=True)


def build(split: str, dest: Path, metrics_csv: Path, n: int, label: str) -> dict:
    manifest = load_manifest(split)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics not found for {split}: {metrics_csv}")
    metrics = pd.read_csv(metrics_csv)

    picks = pick_ranked(metrics, n)
    dest.mkdir(parents=True, exist_ok=True)

    cases = []
    for pos, row in picks.iterrows():
        case_id = row["case_id"]
        mrow = manifest[manifest["case_id"] == case_id].iloc[0]
        src_img = PROJECT_ROOT / mrow["image_path"]
        src_lab = PROJECT_ROOT / mrow["label_path"]

        case_dir = dest / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        dst_img = case_dir / src_img.name
        dst_lab = case_dir / f"label_{src_lab.name}"

        # copyfile: read-only on the source, never a move or rename of the original
        if not dst_img.exists() or dst_img.stat().st_size != src_img.stat().st_size:
            shutil.copyfile(src_img, dst_img)
        if not dst_lab.exists() or dst_lab.stat().st_size != src_lab.stat().st_size:
            shutil.copyfile(src_lab, dst_lab)

        cases.append(
            {
                "case_id": case_id,
                "split": label,
                "split_role": (
                    "validation — used for model selection"
                    if split == "valid"
                    else "held-out internal test — one-time generalization check"
                ),
                "rank_in_split": int(pos) + 1,
                "selection_rule": "evenly spaced rank across the split's Dice distribution",
                "image_file": dst_img.name,
                "label_file": dst_lab.name,
                "source_image_path": mrow["image_path"],
                "source_label_path": mrow["label_path"],
                "case_dice": float(row["dice"]),
                "case_iou": float(row["iou"]),
                "gt_tumour_voxels": int(row["gt_tumour_voxels"]),
                "pred_tumour_voxels": int(row["pred_tumour_voxels"]),
                "n_slices": int(row["n_slices"]),
                "shape": [int(mrow["shape_h"]), int(mrow["shape_w"]), int(mrow["shape_d"])],
                "modalities": list(MODALITY_NAMES),
                "image_bytes": dst_img.stat().st_size,
                "label_bytes": dst_lab.stat().st_size,
            }
        )
        print(f"  {label:<26} {case_id}  Dice={row['dice']:.4f}  "
              f"({dst_img.stat().st_size / 1e6:.1f} MB copied)")

    readme = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Copied-only presentation samples for the Streamlit Segmentation inference demo."
        ),
        "split": label,
        "split_note": (
            HELD_OUT_TEST_NOTE if split == "test"
            else "Validation split — these cases selected the checkpoint."
        ),
        "source_dataset": {"id": DATASET_ID, "name": DATASET_NAME},
        "task": {
            "target": "binary whole tumour vs background",
            "rule": WHOLE_TUMOUR_RULE,
            "source_label_mapping": {str(k): v for k, v in SOURCE_LABELS.items()},
        },
        "modalities": list(MODALITY_NAMES),
        "copy_policy": (
            "Files copied with shutil.copyfile from data/raw/mri/decathlon_task01_brain_tumour/. "
            "Originals were not moved, renamed or modified."
        ),
        "selection_rule": "evenly spaced ranks across the split's per-case Dice distribution",
        "n_cases": len(cases),
        "cases": cases,
    }
    (dest / "presentation_manifest.json").write_text(
        json.dumps(readme, indent=2), encoding="utf-8"
    )
    return readme


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-valid", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--input-size", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / "mri_unet_whole_tumour.yaml").read_text(encoding="utf-8"))
    input_size = args.input_size or int(cfg["preprocess"]["input_size"])
    out = run_dir(input_size)

    print("=" * 70)
    print("PHASE 8 — PRESENTATION SAMPLES (copies only)")
    print("=" * 70)

    v = build(
        "valid", PRESENTATION_VALID_DIR,
        out / "metrics" / "validation_per_case_metrics.csv",
        args.n_valid, "validation",
    )
    t = build(
        "test", PRESENTATION_TEST_DIR,
        out / "metrics" / "heldout_test_per_case_metrics.csv",
        args.n_test, "held-out internal test",
    )

    total_mb = sum(c["image_bytes"] + c["label_bytes"] for c in v["cases"] + t["cases"]) / 1e6
    print(f"\nvalidation samples : {v['n_cases']} -> "
          f"{PRESENTATION_VALID_DIR.relative_to(REPO).as_posix()}")
    print(f"held-out test      : {t['n_cases']} -> "
          f"{PRESENTATION_TEST_DIR.relative_to(REPO).as_posix()}")
    print(f"total copied       : {total_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
