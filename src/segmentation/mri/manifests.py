"""Phase 2 — patient-safe MRI manifests and frozen train/valid/test splits.

Splitting is done **only at case (patient) level**: one Decathlon case is one
patient, and every slice of that case lands in exactly one split. No slice, patch
or augmented view ever crosses a split boundary.

The test split is a project-created *internal* held-out set. It exists to give one
final generalization number after the checkpoint is frozen — it is never an
external or clinical validation set.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    DATASET_ID,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_URL,
    HELD_OUT_TEST_NOTE,
    MANIFEST_CSV,
    MODALITY_NAMES,
    PROCESSED_MRI_DIR,
    PROJECT_ROOT,
    SLICE_AXIS,
    SLICE_AXIS_NAME,
    SOURCE_LABELS,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    SPLIT_SUMMARY_JSON,
    TEST_MANIFEST_CSV,
    TRAIN_MANIFEST_CSV,
    VALID_MANIFEST_CSV,
    WHOLE_TUMOUR_RULE,
)

MANIFEST_COLUMNS = [
    "case_id",
    "patient_id",
    "split",
    "image_path",
    "label_path",
    "shape_h",
    "shape_w",
    "shape_d",
    "modality_count",
    "modalities",
    "voxel_spacing",
    "source_label_values",
    "tumour_voxels",
    "total_voxels",
    "tumour_fraction",
    "tumour_slices",
    "total_slices",
    "slice_axis",
    "source_dataset",
]


def _rel(path: str | Path) -> str:
    """Store repo-relative POSIX paths so manifests stay portable."""
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def build_manifest_frame(inventory: dict) -> pd.DataFrame:
    """Flatten the Phase 1 inventory into one row per usable case."""
    rows = []
    for case in inventory["cases"]:
        if not case["ok"]:
            continue
        h, w, d = case["image_shape"][0], case["image_shape"][1], case["image_shape"][2]
        rows.append(
            {
                "case_id": case["case_id"],
                # One Decathlon case == one patient/subject. Kept as an explicit
                # column so the split code never has to re-derive identity.
                "patient_id": case["case_id"],
                "split": "",
                "image_path": _rel(case["image_path"]),
                "label_path": _rel(case["label_path"]),
                "shape_h": int(h),
                "shape_w": int(w),
                "shape_d": int(d),
                "modality_count": int(case["n_modalities"]),
                "modalities": "|".join(MODALITY_NAMES),
                "voxel_spacing": "x".join(str(v) for v in case["voxel_spacing"][:3]),
                "source_label_values": "|".join(str(v) for v in case["label_values"]),
                "tumour_voxels": int(case["tumour_voxels"]),
                "total_voxels": int(case["total_voxels"]),
                "tumour_fraction": float(case["tumour_fraction"]),
                "tumour_slices": int(case["tumour_slices"]),
                "total_slices": int(case["total_slices"]),
                "slice_axis": SLICE_AXIS,
                "source_dataset": DATASET_ID,
            }
        )
    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    return df[MANIFEST_COLUMNS]


def assign_splits(
    df: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    fractions: dict[str, float] | None = None,
    stratify_on_tumour_burden: bool = True,
) -> pd.DataFrame:
    """Assign train/valid/test at patient level with a fixed reproducible seed.

    Patients are stratified into quartiles of whole-tumour burden first, then
    shuffled and dealt within each stratum, so all three splits see a comparable
    spread of small and large tumours. Stratification only reorders patients — it
    never lets a patient appear twice.
    """
    fractions = fractions or SPLIT_FRACTIONS
    df = df.copy()

    patients = df["patient_id"].drop_duplicates().tolist()
    if len(patients) != len(df):
        raise ValueError("manifest has more than one row per patient_id")

    if stratify_on_tumour_burden:
        burden = df.set_index("patient_id")["tumour_fraction"]
        try:
            strata = pd.qcut(burden.rank(method="first"), q=4, labels=False)
        except ValueError:  # too few unique values
            strata = pd.Series(0, index=burden.index)
        groups = {int(s): sorted(strata.index[strata == s].tolist()) for s in sorted(set(strata))}
    else:
        groups = {0: sorted(patients)}

    rng = np.random.default_rng(seed)
    shuffled = {}
    for stratum in sorted(groups):
        members = list(groups[stratum])
        rng.shuffle(members)
        shuffled[stratum] = members

    # Interleave strata round-robin, then deal with a streaming largest-deficit
    # apportionment. Rounding per stratum can starve a split when strata are
    # small (3 patients across 4 quartiles put everyone in one split), whereas a
    # global running quota is exact at any size and still keeps tumour burden
    # spread evenly because the input order alternates between strata.
    ordered: list[str] = []
    for i in range(max(len(m) for m in shuffled.values())):
        for stratum in sorted(shuffled):
            if i < len(shuffled[stratum]):
                ordered.append(shuffled[stratum][i])

    names = ("train", "valid", "test")
    counts = {s: 0 for s in names}
    assignment: dict[str, str] = {}
    for i, cid in enumerate(ordered, start=1):
        # give the patient to whichever split is furthest below its quota so far
        target = max(names, key=lambda s: (fractions[s] * i - counts[s], -names.index(s)))
        assignment[cid] = target
        counts[target] += 1

    # With >= 3 patients every split must be non-empty; move from the largest.
    if len(ordered) >= 3:
        for s in names:
            if counts[s] == 0:
                donor = max(names, key=lambda d: counts[d])
                moved = next(c for c in reversed(ordered) if assignment[c] == donor)
                assignment[moved] = s
                counts[donor] -= 1
                counts[s] += 1

    df["split"] = df["patient_id"].map(assignment)
    if df["split"].isna().any():
        missing = df.loc[df["split"].isna(), "patient_id"].tolist()
        raise ValueError(f"unassigned patients: {missing}")
    return df


def assert_no_patient_overlap(df: pd.DataFrame) -> dict[str, list[str]]:
    """Hard leakage check across all three splits. Raises on any overlap."""
    sets = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "valid", "test")}
    overlaps = {
        "train_valid": sorted(sets["train"] & sets["valid"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "valid_test": sorted(sets["valid"] & sets["test"]),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise AssertionError(f"patient leakage detected across MRI splits: {bad}")

    covered = sum(len(v) for v in sets.values())
    if covered != len(df):
        raise AssertionError(
            f"split coverage mismatch: {covered} assigned vs {len(df)} manifest rows"
        )
    return overlaps


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifests(df: pd.DataFrame, inventory: dict, *, download_date: str) -> dict:
    """Write the four manifest CSVs + split_summary.json, then freeze by hashing."""
    PROCESSED_MRI_DIR.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["split", "case_id"]).reset_index(drop=True)
    df.to_csv(MANIFEST_CSV, index=False)
    per_split = {}
    for split, path in (
        ("train", TRAIN_MANIFEST_CSV),
        ("valid", VALID_MANIFEST_CSV),
        ("test", TEST_MANIFEST_CSV),
    ):
        sub = df[df["split"] == split].reset_index(drop=True)
        sub.to_csv(path, index=False)
        per_split[split] = sub

    overlaps = assert_no_patient_overlap(df)

    def _stats(sub: pd.DataFrame) -> dict:
        return {
            "cases": int(len(sub)),
            "percent_of_cases": round(100.0 * len(sub) / max(len(df), 1), 2),
            "total_slices": int(sub["total_slices"].sum()),
            "tumour_slices": int(sub["tumour_slices"].sum()),
            "mean_tumour_fraction": float(sub["tumour_fraction"].mean()) if len(sub) else 0.0,
            "cases_without_tumour": int((sub["tumour_voxels"] == 0).sum()),
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "download_url": DATASET_URL,
            "download_date": download_date,
            "licence": DATASET_LICENCE,
        },
        "task": {
            "target": "binary whole tumour vs background",
            "rule": WHOLE_TUMOUR_RULE,
            "source_label_mapping": {str(k): v for k, v in SOURCE_LABELS.items()},
            "tumour_source_labels": [1, 2, 3],
            "input_modalities": list(MODALITY_NAMES),
            "slice_axis": SLICE_AXIS,
            "slice_axis_name": SLICE_AXIS_NAME,
        },
        "split_policy": {
            "unit": "patient/case — never slices or patches",
            "seed": SPLIT_SEED,
            "target_fractions": SPLIT_FRACTIONS,
            "stratified_on": "whole-tumour burden quartile",
            "held_out_test_note": HELD_OUT_TEST_NOTE,
            "test_split_protected": True,
            "test_used_for_selection": False,
            "test_used_for_tuning": False,
            "test_used_for_early_stopping": False,
            "test_used_for_threshold_selection": False,
        },
        "totals": {
            "cases": int(len(df)),
            "total_slices": int(df["total_slices"].sum()),
            "tumour_slices": int(df["tumour_slices"].sum()),
        },
        "splits": {s: _stats(per_split[s]) for s in ("train", "valid", "test")},
        "leakage_check": {
            "train_valid_overlap": overlaps["train_valid"],
            "train_test_overlap": overlaps["train_test"],
            "valid_test_overlap": overlaps["valid_test"],
            "zero_patient_overlap": all(not v for v in overlaps.values()),
        },
        "inventory": {
            "n_matched_pairs": inventory["n_matched_pairs"],
            "n_images_without_label": inventory["n_images_without_label"],
            "n_labels_without_image": inventory["n_labels_without_image"],
            "n_ok": inventory["n_ok"],
            "n_failed": inventory["n_failed"],
            "image_shape_histogram": inventory["image_shape_histogram"],
            "observed_label_values": inventory["observed_label_values"],
        },
    }

    SPLIT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Freeze: record a hash of each manifest so later phases can prove they are unchanged.
    frozen = {
        "frozen_utc": summary["generated_utc"],
        "files": {
            _rel(p): {"sha256": _sha256_file(p), "rows": int(len(pd.read_csv(p)))}
            for p in (MANIFEST_CSV, TRAIN_MANIFEST_CSV, VALID_MANIFEST_CSV, TEST_MANIFEST_CSV)
        },
    }
    (PROCESSED_MRI_DIR / "split_frozen.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    summary["frozen"] = frozen
    SPLIT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_manifest(split: str | None = None) -> pd.DataFrame:
    """Load a frozen manifest. `split` in {train, valid, test} or None for all."""
    path = {
        None: MANIFEST_CSV,
        "all": MANIFEST_CSV,
        "train": TRAIN_MANIFEST_CSV,
        "valid": VALID_MANIFEST_CSV,
        "test": TEST_MANIFEST_CSV,
    }[split]
    if not path.exists():
        raise FileNotFoundError(
            f"MRI manifest missing: {path}. Run scripts/mri_build_manifests.py first."
        )
    return pd.read_csv(path)


def verify_frozen_manifests() -> dict:
    """Re-hash the manifests and compare against split_frozen.json."""
    frozen_path = PROCESSED_MRI_DIR / "split_frozen.json"
    if not frozen_path.exists():
        return {"available": False, "unchanged": None, "detail": "split_frozen.json missing"}
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    results = {}
    for rel, info in frozen["files"].items():
        p = PROJECT_ROOT / rel
        results[rel] = {
            "expected": info["sha256"],
            "actual": _sha256_file(p) if p.exists() else None,
            "match": p.exists() and _sha256_file(p) == info["sha256"],
        }
    return {
        "available": True,
        "unchanged": all(r["match"] for r in results.values()),
        "files": results,
        "frozen_utc": frozen["frozen_utc"],
    }
