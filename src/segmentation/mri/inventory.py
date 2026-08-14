"""Phase 1 — inventory and integrity verification of the Decathlon Task01 archive.

Verifies, for every labelled case:
  * a matching image/label pair exists (no orphans on either side),
  * both files open and fully decompress with nibabel (no corrupt gzip),
  * image is 4D with exactly four modalities, label is 3D,
  * image and label spatial extents agree,
  * affines agree (spatial alignment),
  * label values are a subset of the documented source mapping.

Everything is reported; nothing is silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from .constants import (
    DATASET_ID,
    DATASET_JSON,
    IMAGES_TR,
    LABELS_TR,
    MODALITY_NAMES,
    N_MODALITIES,
    SOURCE_LABELS,
)


def _is_case_file(p: Path) -> bool:
    """Real case files only — the archive ships macOS `._BRATS_xxx.nii.gz` stubs."""
    return p.name.endswith(".nii.gz") and not p.name.startswith("._")


def case_id_from_path(p: Path) -> str:
    return p.name[: -len(".nii.gz")]


@dataclass
class CaseRecord:
    case_id: str
    image_path: str
    label_path: str
    ok: bool = True
    problems: list[str] = field(default_factory=list)
    image_shape: tuple[int, ...] | None = None
    label_shape: tuple[int, ...] | None = None
    n_modalities: int | None = None
    voxel_spacing: tuple[float, ...] | None = None
    label_values: list[int] = field(default_factory=list)
    tumour_voxels: int = 0
    total_voxels: int = 0
    tumour_fraction: float = 0.0
    tumour_slices: int = 0
    total_slices: int = 0
    affine_match: bool | None = None
    image_bytes: int = 0
    label_bytes: int = 0


def read_dataset_json(path: Path = DATASET_JSON) -> dict:
    """Load the shipped dataset.json (source of truth for labels and modalities)."""
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def verify_dataset_json(meta: dict) -> tuple[bool, list[str]]:
    """Assert the shipped metadata matches the label mapping this project assumes."""
    problems: list[str] = []

    declared_labels = {int(k): str(v) for k, v in (meta.get("labels") or {}).items()}
    if declared_labels != SOURCE_LABELS:
        problems.append(
            f"dataset.json labels {declared_labels} != expected {SOURCE_LABELS}"
        )

    declared_mods = meta.get("modality") or {}
    declared_mods = tuple(declared_mods[k] for k in sorted(declared_mods, key=int))
    if len(declared_mods) != N_MODALITIES:
        problems.append(
            f"dataset.json declares {len(declared_mods)} modalities, expected {N_MODALITIES}"
        )
    return (not problems), problems


def inspect_case(image_path: Path, label_path: Path, *, slice_axis: int = 2) -> CaseRecord:
    """Fully load one image/label pair and record shape, labels and integrity."""
    rec = CaseRecord(
        case_id=case_id_from_path(image_path),
        image_path=image_path.as_posix(),
        label_path=label_path.as_posix(),
    )
    try:
        rec.image_bytes = image_path.stat().st_size
        rec.label_bytes = label_path.stat().st_size
    except OSError as exc:  # pragma: no cover
        rec.ok = False
        rec.problems.append(f"stat failed: {exc}")
        return rec

    try:
        img = nib.load(image_path)
        lab = nib.load(label_path)
        # Force full decompression so a truncated gzip is caught here, not mid-training.
        img_arr = np.asanyarray(img.dataobj)
        lab_arr = np.asanyarray(lab.dataobj)
    except Exception as exc:  # noqa: BLE001
        rec.ok = False
        rec.problems.append(f"unreadable/corrupt: {type(exc).__name__}: {exc}")
        return rec

    rec.image_shape = tuple(int(x) for x in img_arr.shape)
    rec.label_shape = tuple(int(x) for x in lab_arr.shape)
    rec.voxel_spacing = tuple(round(float(z), 4) for z in img.header.get_zooms())

    if img_arr.ndim != 4:
        rec.ok = False
        rec.problems.append(f"image is {img_arr.ndim}D, expected 4D (H,W,D,C)")
    else:
        rec.n_modalities = int(img_arr.shape[3])
        if rec.n_modalities != N_MODALITIES:
            rec.ok = False
            rec.problems.append(
                f"{rec.n_modalities} modalities, expected {N_MODALITIES} {MODALITY_NAMES}"
            )

    if lab_arr.ndim != 3:
        rec.ok = False
        rec.problems.append(f"label is {lab_arr.ndim}D, expected 3D")

    if img_arr.ndim == 4 and lab_arr.ndim == 3:
        if img_arr.shape[:3] != lab_arr.shape:
            rec.ok = False
            rec.problems.append(
                f"spatial mismatch: image {img_arr.shape[:3]} vs label {lab_arr.shape}"
            )
        rec.affine_match = bool(np.allclose(img.affine, lab.affine, atol=1e-4))
        if not rec.affine_match:
            rec.ok = False
            rec.problems.append("image/label affines differ — volumes are not aligned")

    if not np.all(np.isfinite(img_arr)):
        rec.ok = False
        rec.problems.append("image contains NaN/Inf voxels")

    values = np.unique(lab_arr)
    rec.label_values = [int(v) for v in values]
    unexpected = sorted(set(rec.label_values) - set(SOURCE_LABELS))
    if unexpected:
        rec.ok = False
        rec.problems.append(f"undocumented label values present: {unexpected}")

    binary = lab_arr > 0
    rec.total_voxels = int(binary.size)
    rec.tumour_voxels = int(binary.sum())
    rec.tumour_fraction = float(rec.tumour_voxels / rec.total_voxels) if rec.total_voxels else 0.0
    if lab_arr.ndim == 3:
        per_slice = binary.any(axis=tuple(a for a in range(3) if a != slice_axis))
        rec.total_slices = int(lab_arr.shape[slice_axis])
        rec.tumour_slices = int(per_slice.sum())
    if rec.tumour_voxels == 0:
        rec.problems.append("no tumour voxels in label (kept, but flagged)")

    return rec


def discover_pairs(
    images_dir: Path = IMAGES_TR, labels_dir: Path = LABELS_TR
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """Return matched (image, label) pairs plus unmatched ids on either side."""
    images = {case_id_from_path(p): p for p in sorted(images_dir.glob("*.nii.gz")) if _is_case_file(p)}
    labels = {case_id_from_path(p): p for p in sorted(labels_dir.glob("*.nii.gz")) if _is_case_file(p)}

    matched_ids = sorted(set(images) & set(labels))
    images_without_label = sorted(set(images) - set(labels))
    labels_without_image = sorted(set(labels) - set(images))
    pairs = [(images[cid], labels[cid]) for cid in matched_ids]
    return pairs, images_without_label, labels_without_image


def build_inventory(
    images_dir: Path = IMAGES_TR,
    labels_dir: Path = LABELS_TR,
    *,
    slice_axis: int = 2,
    progress: bool = True,
) -> dict:
    """Full Phase 1 inventory over every labelled case."""
    pairs, img_only, lab_only = discover_pairs(images_dir, labels_dir)

    records: list[CaseRecord] = []
    for i, (ip, lp) in enumerate(pairs, 1):
        rec = inspect_case(ip, lp, slice_axis=slice_axis)
        records.append(rec)
        if progress and (i % 25 == 0 or i == len(pairs)):
            bad = sum(1 for r in records if not r.ok)
            print(f"  [inventory] {i}/{len(pairs)} cases verified ({bad} problem(s))", flush=True)

    shapes: dict[str, int] = {}
    for r in records:
        shapes[str(r.image_shape)] = shapes.get(str(r.image_shape), 0) + 1
    label_value_sets: dict[str, int] = {}
    for r in records:
        key = str(r.label_values)
        label_value_sets[key] = label_value_sets.get(key, 0) + 1

    all_label_values = sorted({v for r in records for v in r.label_values})
    failures = [r for r in records if not r.ok]

    return {
        "dataset_id": DATASET_ID,
        "images_dir": images_dir.as_posix(),
        "labels_dir": labels_dir.as_posix(),
        "n_matched_pairs": len(pairs),
        "n_images_without_label": len(img_only),
        "n_labels_without_image": len(lab_only),
        "images_without_label": img_only,
        "labels_without_image": lab_only,
        "n_ok": len(records) - len(failures),
        "n_failed": len(failures),
        "failures": [{"case_id": r.case_id, "problems": r.problems} for r in failures],
        "image_shape_histogram": shapes,
        "label_value_set_histogram": label_value_sets,
        "observed_label_values": all_label_values,
        "documented_label_mapping": {str(k): v for k, v in SOURCE_LABELS.items()},
        "modality_names": list(MODALITY_NAMES),
        "total_tumour_voxels": int(sum(r.tumour_voxels for r in records)),
        "total_voxels": int(sum(r.total_voxels for r in records)),
        "total_tumour_slices": int(sum(r.tumour_slices for r in records)),
        "total_slices": int(sum(r.total_slices for r in records)),
        "cases": [asdict(r) for r in records],
    }
