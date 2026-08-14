"""Phase 2 — build and freeze the patient-safe MRI manifests and splits.

Usage::

    python scripts/mri_build_manifests.py

Writes to data/processed/mri/:
    mri_manifest.csv, train_manifest.csv, valid_manifest.csv, test_manifest.csv,
    split_summary.json, split_frozen.json

Touches nothing under data/processed/xray/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.constants import (  # noqa: E402
    INVENTORY_JSON,
    PROCESSED_MRI_DIR,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
)
from src.segmentation.mri.manifests import (  # noqa: E402
    assign_splits,
    build_manifest_frame,
    write_manifests,
)


def main() -> int:
    if not INVENTORY_JSON.exists():
        print(
            f"ERROR: {INVENTORY_JSON} missing. Run scripts/mri_prepare_dataset.py first.",
            file=sys.stderr,
        )
        return 2

    inv = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    if inv["n_failed"]:
        print(f"ERROR: {inv['n_failed']} case(s) failed Phase 1 verification.", file=sys.stderr)
        for f in inv["failures"][:20]:
            print(f"  {f['case_id']}: {f['problems']}", file=sys.stderr)
        return 3

    df = build_manifest_frame(inv)
    print(f"[manifest] {len(df)} usable cases")

    df = assign_splits(
        df, seed=SPLIT_SEED, fractions=SPLIT_FRACTIONS, stratify_on_tumour_burden=True
    )
    summary = write_manifests(df, inv, download_date=inv.get("download_date", "unknown"))

    print("\n" + "=" * 70)
    print("PHASE 2 PATIENT-SAFE SPLIT")
    print("=" * 70)
    print(f"seed                  : {SPLIT_SEED}   (split unit: patient/case)")
    print(f"target fractions      : {SPLIT_FRACTIONS}")
    for split in ("train", "valid", "test"):
        s = summary["splits"][split]
        print(
            f"{split:<6} cases={s['cases']:<4} ({s['percent_of_cases']:>5.2f}%)  "
            f"slices={s['total_slices']:<7} tumour_slices={s['tumour_slices']:<7} "
            f"mean_tumour_frac={s['mean_tumour_fraction']:.5f}"
        )
    lk = summary["leakage_check"]
    print(f"\nzero patient overlap  : {lk['zero_patient_overlap']}")
    print(f"  train n valid        : {len(lk['train_valid_overlap'])}")
    print(f"  train n test         : {len(lk['train_test_overlap'])}")
    print(f"  valid n test         : {len(lk['valid_test_overlap'])}")
    print(f"\nheld-out test note    : {summary['split_policy']['held_out_test_note']}")
    print("\nfrozen manifests:")
    for rel, info in summary["frozen"]["files"].items():
        print(f"  {rel:<40} rows={info['rows']:<5} sha256={info['sha256'][:16]}...")
    print(f"\nwritten -> {PROCESSED_MRI_DIR.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
