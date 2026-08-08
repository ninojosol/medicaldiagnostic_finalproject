"""Reproducible, patient-level train/validation/test splitting.

Why patient-level: chest X-ray datasets contain several images of the same
patient. A random per-image split puts near-identical images of one patient in
both train and test, which inflates every metric. This is the single most common
methodological error in student medical-imaging projects, so the split is done by
patient ID here and verified afterwards with :func:`assert_no_patient_leakage`.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..common.errors import DataLayoutError


def _stable_patient_order(patient_ids: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic patient ordering that does not depend on row order.

    Hashing each ID with the seed means adding or removing images from the
    dataset does not reshuffle every other patient into a different split.
    """
    keys = [
        hashlib.md5(f"{seed}:{pid}".encode("utf-8")).hexdigest()
        for pid in patient_ids
    ]
    order = np.argsort(np.array(keys))
    return np.asarray(patient_ids)[order]


def patient_level_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                        seed: int = 42, patient_col: str = "patient_id",
                        stratify_col: str | None = None) -> pd.DataFrame:
    """Add a ``split`` column with values ``train`` / ``val`` / ``test``.

    Parameters
    ----------
    stratify_col:
        Optional column whose per-patient majority value is used to balance the
        label distribution across splits. Pass a label column name (e.g.
        ``"PNEUMONIA"``) when the dataset is imbalanced; pass ``None`` for a
        plain random patient split.

    Notes
    -----
    Split proportions are approximate: patients are indivisible, so a dataset
    where one patient owns many images cannot hit the target ratios exactly.
    """
    if patient_col not in df.columns:
        raise DataLayoutError(
            f"Column '{patient_col}' missing - cannot do a patient-level split.\n"
            f"  Columns present: {list(df.columns)}\n"
            "  Fix: set data.patient_column / data.patient_id_regex in the config."
        )
    if not 0 <= val_size < 1 or not 0 <= test_size < 1 or val_size + test_size >= 1:
        raise ValueError(f"Invalid split sizes: val_size={val_size}, test_size={test_size}")

    out = df.copy()
    groups = out[[patient_col]].copy()
    if stratify_col:
        if stratify_col not in out.columns:
            raise DataLayoutError(f"stratify_col='{stratify_col}' not in manifest columns {list(out.columns)}")
        # One stratum per patient: does this patient have the finding at all?
        groups["_stratum"] = out[stratify_col]
        strata = groups.groupby(patient_col)["_stratum"].max().astype(int)
    else:
        strata = pd.Series(0, index=pd.Index(out[patient_col].unique(), name=patient_col))

    assignment: dict[str, str] = {}
    for stratum_value in sorted(strata.unique()):
        patients = strata.index[strata == stratum_value].to_numpy()
        ordered = _stable_patient_order(patients, seed)
        n = len(ordered)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        # Guarantee at least one patient per non-empty split when possible.
        if test_size > 0 and n_test == 0 and n >= 3:
            n_test = 1
        if val_size > 0 and n_val == 0 and n >= 3:
            n_val = 1
        n_train = max(n - n_val - n_test, 0)

        for pid in ordered[:n_train]:
            assignment[pid] = "train"
        for pid in ordered[n_train:n_train + n_val]:
            assignment[pid] = "val"
        for pid in ordered[n_train + n_val:]:
            assignment[pid] = "test"

    out["split"] = out[patient_col].map(assignment)
    if out["split"].isna().any():
        raise DataLayoutError("Internal error: some rows were not assigned a split.")
    return out


def use_existing_split(df: pd.DataFrame, val_size: float = 0.15, seed: int = 42,
                       patient_col: str = "patient_id", stratify_col: str | None = None) -> pd.DataFrame:
    """Honour a dataset's official train/test split, carving ``val`` out of train.

    Many public chest X-ray datasets ship ``train/`` and ``test/`` folders but no
    validation set. Using the official test set keeps results comparable with
    published work; the validation set must still be patient-disjoint from train.
    """
    if "split" not in df.columns:
        raise DataLayoutError("No 'split' column present - use patient_level_split() instead.")

    train_mask = df["split"] == "train"
    if not train_mask.any():
        raise DataLayoutError(f"No rows with split=='train'. Values found: {df['split'].unique().tolist()}")
    if (df["split"] == "val").any():
        print("[info] dataset already provides a val split; leaving it untouched.")
        return df.copy()

    carved = patient_level_split(
        df.loc[train_mask].copy(), val_size=val_size, test_size=0.0,
        seed=seed, patient_col=patient_col, stratify_col=stratify_col,
    )
    out = df.copy()
    out.loc[train_mask, "split"] = carved["split"].to_numpy()
    return out


def assert_no_patient_leakage(df: pd.DataFrame, patient_col: str = "patient_id",
                              split_col: str = "split") -> dict:
    """Verify no patient appears in more than one split. Raises if leakage exists.

    Run this in the notebook and show the output - it is direct evidence of a
    methodologically sound split for the report.
    """
    per_split = {s: set(g[patient_col]) for s, g in df.groupby(split_col)}
    overlaps = {}
    names = sorted(per_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = per_split[a] & per_split[b]
            if shared:
                overlaps[f"{a}&{b}"] = sorted(shared)[:20]

    if overlaps:
        detail = "\n".join(f"  {k}: {len(v)}+ shared patients, e.g. {v[:5]}" for k, v in overlaps.items())
        raise DataLayoutError(f"PATIENT LEAKAGE DETECTED between splits:\n{detail}")

    result = {
        "leakage": False,
        "patients_per_split": {s: len(p) for s, p in per_split.items()},
        "images_per_split": df[split_col].value_counts().to_dict(),
    }
    print("[ok] No patient appears in more than one split.")
    for split in ["train", "val", "test"]:
        if split in per_split:
            print(f"    {split:<6s} {result['images_per_split'].get(split, 0):>7d} images "
                  f"from {len(per_split[split]):>6d} patients")
    return result


def split_summary(df: pd.DataFrame, labels: list[str], split_col: str = "split") -> pd.DataFrame:
    """Per-split image count, patient count and positive rate for each label.

    Use this table in the report to show the class balance survived the split.
    """
    rows = []
    for split, group in df.groupby(split_col):
        row = {
            "split": split,
            "n_images": len(group),
            "n_patients": group["patient_id"].nunique(),
        }
        for lab in labels:
            row[f"{lab}_n"] = int(group[lab].sum())
            row[f"{lab}_rate"] = round(float(group[lab].mean()), 4)
        rows.append(row)
    order = {"train": 0, "val": 1, "test": 2}
    return pd.DataFrame(rows).sort_values("split", key=lambda s: s.map(order).fillna(9), ignore_index=True)
