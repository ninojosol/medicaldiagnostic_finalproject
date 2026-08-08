"""Patient-level splitting for MRI segmentation.

Adjacent slices from the same MRI volume are almost identical images. Splitting
slices randomly therefore puts near-duplicates in train and test, and Dice scores
become meaningless. Splitting happens by patient (i.e. by volume) instead.

The patient-level machinery is shared with the classification module; this file
adds the segmentation-specific stratification (does the patient have any tumour
pixels at all) and the summary table.
"""

from __future__ import annotations

import pandas as pd

from ..classification.splits import assert_no_patient_leakage, patient_level_split

__all__ = ["patient_level_split_seg", "assert_no_patient_leakage", "seg_split_summary"]


def patient_level_split_seg(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                            seed: int = 42, patient_col: str = "patient_id",
                            stratify_on_tumour: bool = True) -> pd.DataFrame:
    """Split slices by patient, optionally balancing tumour-positive patients.

    ``stratify_on_tumour`` requires a ``has_tumour`` column (produced by
    :func:`src.segmentation.pairing.validate_pairs`, or computed here from
    ``tumour_pixel_fraction``).
    """
    work = df.copy()
    stratify_col = None

    if stratify_on_tumour:
        if "has_tumour" not in work.columns and "tumour_pixel_fraction" in work.columns:
            work["has_tumour"] = (work["tumour_pixel_fraction"] > 0).astype(int)
        if "has_tumour" in work.columns:
            stratify_col = "has_tumour"
        else:
            print("[info] no 'has_tumour' column - splitting without tumour stratification.")

    return patient_level_split(work, val_size=val_size, test_size=test_size, seed=seed,
                               patient_col=patient_col, stratify_col=stratify_col)


def seg_split_summary(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    """Per-split slice count, patient count, tumour-positive rate and mean coverage."""
    rows = []
    for split, group in df.groupby(split_col):
        row = {
            "split": split,
            "n_slices": len(group),
            "n_patients": group["patient_id"].nunique() if "patient_id" in group else None,
        }
        if "has_tumour" in group.columns:
            row["n_slices_with_tumour"] = int(group["has_tumour"].sum())
            row["tumour_slice_rate"] = round(float(group["has_tumour"].mean()), 4)
        if "tumour_pixel_fraction" in group.columns:
            row["mean_tumour_pixel_fraction"] = round(float(group["tumour_pixel_fraction"].mean()), 6)
        rows.append(row)
    order = {"train": 0, "val": 1, "test": 2}
    return pd.DataFrame(rows).sort_values("split", key=lambda s: s.map(order).fillna(9), ignore_index=True)
