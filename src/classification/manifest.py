"""Turn whatever the course dataset looks like on disk into ONE tidy manifest.

Every downstream step (EDA, splitting, Dataset, evaluation) reads this single
DataFrame, so adapting to a new dataset means changing the config, not the code.

Manifest schema
---------------
================  ==========================================================
column            meaning
================  ==========================================================
``image_path``    absolute path to the image file
``rel_path``      path relative to the dataset root (nice for tables/figures)
``patient_id``    grouping key used to prevent patient leakage in the split
``<label>``       one 0/1 column per entry in ``config.data.labels``
``split``         optional, only when the dataset ships an official split
================  ==========================================================

Two discovery modes are supported (``config.data.layout``):

``folders``
    ``<root>/<class_name>/*.png`` or ``<root>/<split>/<class_name>/*.png``
    (the classic Kaggle chest X-ray "NORMAL / PNEUMONIA" layout).
``csv``
    A CSV/metadata file listing images and labels - required for multi-label
    datasets such as NIH ChestX-ray14.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..common.errors import DataLayoutError, require_dir, require_file, require_nonempty
from ..common.io_utils import list_images

KNOWN_SPLIT_DIRS = {"train", "training", "val", "valid", "validation", "test", "testing"}
_SPLIT_ALIASES = {
    "training": "train", "train": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


# ---------------------------------------------------------------------------
# patient ids
# ---------------------------------------------------------------------------
def extract_patient_id(path: Path, regex: str | None = None,
                       exclude_dirs: set[str] | None = None) -> str:
    """Derive a patient identifier from a file path.

    ``regex`` should contain one capturing group, applied to the file NAME first
    and then to the full path. Typical values:

    * NIH ChestX-ray14 (``00000013_005.png``) -> ``r"^(\\d+)_"``
    * ``patient00123_view1.jpeg``             -> ``r"(patient\\d+)"``

    Without a regex we fall back to the parent directory name, but only when that
    directory is not a split folder and not a class folder (``exclude_dirs``
    holds the class names, which the caller knows). Otherwise the file stem is
    used - i.e. one image = one "patient", the safe assumption when the dataset
    carries no grouping information.

    Using a class folder as the patient ID would be a silent disaster: every
    positive image would share one ID and the whole class would land in a single
    split. Hence the explicit exclusion rather than a guess.
    """
    if regex:
        for text in (path.name, str(path)):
            match = re.search(regex, text)
            if match:
                return match.group(1) if match.groups() else match.group(0)

    excluded = {name.lower() for name in (exclude_dirs or set())} | KNOWN_SPLIT_DIRS
    parent = path.parent.name
    if parent and parent.lower() not in excluded:
        return parent
    return path.stem


# ---------------------------------------------------------------------------
# discovery: folder layout
# ---------------------------------------------------------------------------
def manifest_from_folders(root: Path, labels: list[str], patient_id_regex: str | None = None,
                          case_insensitive: bool = True,
                          unlisted_as_negative: bool = True) -> pd.DataFrame:
    """Scan ``<root>[/<split>]/<class_name>/*.img`` into a one-hot manifest.

    ``unlisted_as_negative`` (default True) decides what happens to images in
    class folders that are **not** listed in ``labels``. Keeping them as
    negatives is what makes ``labels: [PNEUMONIA]`` mean "PNEUMONIA vs everything
    else" on the classic ``NORMAL/ PNEUMONIA/`` layout - dropping them instead
    would leave a dataset that is 100% positive, which trains and evaluates
    without error while being completely meaningless.

    Set it to False only when the unlisted folders are genuinely a different
    task rather than the negative class.
    """
    root = require_dir(root, what="X-ray dataset", hint="a folder containing one sub-folder per class")

    lookup = {(lab.lower() if case_insensitive else lab): lab for lab in labels}
    rows: list[dict] = []
    positive_dirs: set[str] = set()
    negative_dirs: set[str] = set()
    n_negative = 0

    for image_path in list_images(root, recursive=True):
        rel = image_path.relative_to(root)
        parts = list(rel.parts[:-1])
        if not parts:
            negative_dirs.add("<dataset root>")
            continue

        class_dir = parts[-1]
        key = class_dir.lower() if case_insensitive else class_dir
        label = lookup.get(key)

        if label is None:
            if not unlisted_as_negative:
                negative_dirs.add(class_dir)
                continue
            negative_dirs.add(class_dir)
            n_negative += 1
        else:
            positive_dirs.add(class_dir)

        split = None
        if len(parts) >= 2:
            split = _SPLIT_ALIASES.get(parts[-2].lower())

        row = {
            "image_path": str(image_path),
            "rel_path": str(rel).replace("\\", "/"),
            # The class folder must never become the patient ID - see extract_patient_id.
            "patient_id": extract_patient_id(image_path, patient_id_regex,
                                             exclude_dirs=set(labels) | set(lookup)
                                             | positive_dirs | negative_dirs),
        }
        # label is None -> an all-zero row, i.e. the negative class.
        row.update({lab: int(lab == label) for lab in labels})
        if split:
            row["split"] = split
        rows.append(row)

    if not rows or not positive_dirs:
        raise DataLayoutError(
            f"No images matched the configured labels {labels} under {root}.\n"
            f"  Class folders actually found: {sorted(negative_dirs) or '(none)'}\n"
            "  Fix: set data.labels in your config to one or more of the folder names "
            "above, or re-arrange the dataset as <root>/<class_name>/*.png"
        )

    if negative_dirs and unlisted_as_negative:
        print(f"[info] {n_negative} image(s) from folder(s) {sorted(negative_dirs)} kept as the "
              f"NEGATIVE class (not listed in data.labels).")
        print("       Set data.unlisted_folders_as_negative: false to drop them instead.")
    elif negative_dirs:
        print(f"[warn] dropped images from folder(s) not in data.labels: {sorted(negative_dirs)}")

    return pd.DataFrame(rows).sort_values("rel_path", ignore_index=True)


# ---------------------------------------------------------------------------
# discovery: CSV layout
# ---------------------------------------------------------------------------
def manifest_from_csv(csv_path: Path, image_root: Path, labels: list[str],
                      image_column: str = "image", patient_column: str | None = None,
                      label_column: str | None = None, label_separator: str = "|",
                      patient_id_regex: str | None = None,
                      split_column: str | None = None) -> pd.DataFrame:
    """Read a metadata CSV into the standard manifest.

    Three label encodings are handled automatically:

    1. one 0/1 column per label already present in the CSV (multi-label);
    2. ``label_column`` holding a single class name (binary / multi-class);
    3. ``label_column`` holding separator-joined findings, e.g.
       ``"Effusion|Infiltration"`` (NIH-style multi-label).
    """
    csv_path = require_file(csv_path, what="X-ray label CSV",
                            hint="a CSV with at least an image-path column and label information")
    image_root = require_dir(image_root, what="X-ray image root",
                             hint="the folder the CSV image paths are relative to")

    df = pd.read_csv(csv_path)
    if image_column not in df.columns:
        raise DataLayoutError(
            f"Column '{image_column}' not found in {csv_path.name}.\n"
            f"  Columns present: {list(df.columns)}\n"
            "  Fix: set data.image_column in your config to the correct column name."
        )

    out = pd.DataFrame()
    resolved = df[image_column].astype(str).map(lambda p: _resolve_image(p, image_root))
    out["image_path"] = resolved.map(str)
    out["rel_path"] = df[image_column].astype(str).str.replace("\\", "/", regex=False)

    if patient_column and patient_column in df.columns:
        out["patient_id"] = df[patient_column].astype(str)
    else:
        if patient_column:
            print(f"[warn] patient column '{patient_column}' not in CSV; deriving IDs from filenames instead.")
        out["patient_id"] = resolved.map(lambda p: extract_patient_id(p, patient_id_regex))

    present = [lab for lab in labels if lab in df.columns]
    if len(present) == len(labels):
        for lab in labels:
            out[lab] = df[lab].fillna(0).astype(float).round().astype(int).clip(0, 1)
    elif label_column and label_column in df.columns:
        raw = df[label_column].astype(str).fillna("")
        for lab in labels:
            out[lab] = raw.map(
                lambda v, lab=lab: int(lab in [t.strip() for t in v.split(label_separator)])
            )
    else:
        raise DataLayoutError(
            "Could not build labels from the CSV.\n"
            f"  Configured labels: {labels}\n"
            f"  Label columns found in CSV: {present or '(none)'}\n"
            f"  data.label_column = {label_column!r} (present: {bool(label_column) and label_column in df.columns})\n"
            f"  Columns in CSV: {list(df.columns)}\n"
            "  Fix: either add one 0/1 column per label, or set data.label_column to the "
            "column holding the class name / separator-joined findings."
        )

    if split_column and split_column in df.columns:
        out["split"] = df[split_column].astype(str).str.lower().map(_SPLIT_ALIASES)

    return out.sort_values("rel_path", ignore_index=True)


def _resolve_image(rel: str, image_root: Path) -> Path:
    """Resolve a CSV path entry against the image root, tolerating bare filenames."""
    p = Path(rel)
    if p.is_absolute():
        return p
    direct = image_root / p
    if direct.exists():
        return direct
    # Some CSVs list only the file name while images sit in sub-folders.
    matches = list(image_root.rglob(p.name))
    return matches[0] if matches else direct


# ---------------------------------------------------------------------------
# public entry point + integrity checks
# ---------------------------------------------------------------------------
def build_manifest(cfg) -> pd.DataFrame:
    """Build the manifest described by ``cfg.data``."""
    data = cfg.data
    labels = list(data.labels)
    if not labels:
        raise DataLayoutError("config data.labels is empty - list at least one label.")

    layout = str(data.get("layout", "folders")).lower()
    root = Path(cfg.path("data.root"))
    regex = data.get("patient_id_regex")

    if layout == "folders":
        df = manifest_from_folders(
            root, labels, patient_id_regex=regex,
            unlisted_as_negative=bool(data.get("unlisted_folders_as_negative", True)),
        )
    elif layout == "csv":
        df = manifest_from_csv(
            csv_path=Path(cfg.path("data.csv_path")),
            image_root=Path(cfg.path("data.image_root") if data.get("image_root") else root),
            labels=labels,
            image_column=data.get("image_column", "image"),
            patient_column=data.get("patient_column"),
            label_column=data.get("label_column"),
            label_separator=data.get("label_separator", "|"),
            patient_id_regex=regex,
            split_column=data.get("split_column"),
        )
    else:
        raise DataLayoutError(f"Unknown data.layout={layout!r}. Use 'folders' or 'csv'.")

    require_nonempty(df, what="labelled images", where=root)
    return df


def check_manifest(df: pd.DataFrame, labels: list[str], verbose: bool = True) -> dict:
    """Integrity report: missing files, duplicates, label coverage, patient counts.

    Returns a dict suitable for saving as JSON in the EDA notebook. Nothing is
    dropped here - the caller decides what to do about problems.
    """
    exists = df["image_path"].map(lambda p: Path(p).is_file())
    missing = df.loc[~exists, "image_path"].tolist()
    dup_paths = df.loc[df.duplicated("image_path", keep=False), "image_path"].tolist()

    positives = {lab: int(df[lab].sum()) for lab in labels}
    n_labels_per_image = df[labels].sum(axis=1)

    report = {
        "n_rows": int(len(df)),
        "n_unique_images": int(df["image_path"].nunique()),
        "n_patients": int(df["patient_id"].nunique()),
        "images_per_patient_mean": round(float(len(df) / max(df["patient_id"].nunique(), 1)), 2),
        "n_missing_files": len(missing),
        "missing_files_sample": missing[:10],
        "n_duplicate_paths": len(dup_paths),
        "duplicate_paths_sample": dup_paths[:10],
        "positives_per_label": positives,
        "n_images_with_no_label": int((n_labels_per_image == 0).sum()),
        "n_images_multi_label": int((n_labels_per_image > 1).sum()),
        "task_type": "multilabel" if len(labels) > 1 and (n_labels_per_image > 1).any() else
                     ("multiclass" if len(labels) > 2 else "binary"),
    }

    if verbose:
        print(f"Rows                : {report['n_rows']}")
        print(f"Unique images       : {report['n_unique_images']}")
        print(f"Unique patients     : {report['n_patients']} "
              f"(~{report['images_per_patient_mean']} images/patient)")
        print(f"Missing files       : {report['n_missing_files']}")
        print(f"Duplicate paths     : {report['n_duplicate_paths']}")
        print(f"Images with 0 labels: {report['n_images_with_no_label']}")
        print(f"Multi-label images  : {report['n_images_multi_label']}")
        print(f"Inferred task type  : {report['task_type']}")
        print("Positives per label :")
        for lab, count in positives.items():
            print(f"    {lab:<24s} {count:>7d}  ({count / max(len(df), 1):.2%})")
        if report["n_missing_files"]:
            print("\n[warn] Some image files listed in the manifest do not exist on disk.")
            print("       Drop them before training: df = df[df.image_path.map(lambda p: Path(p).is_file())]")

    return report


def image_size_report(df: pd.DataFrame, sample: int = 300, seed: int = 42) -> pd.DataFrame:
    """Open a random sample of images and report width/height/mode.

    Sampling keeps EDA fast on datasets with tens of thousands of images.
    """
    from PIL import Image

    subset = df.sample(min(sample, len(df)), random_state=seed)
    rows = []
    for path in subset["image_path"]:
        try:
            with Image.open(path) as im:
                rows.append({"image_path": path, "width": im.width, "height": im.height,
                             "mode": im.mode, "format": im.format})
        except Exception as exc:  # unreadable/corrupt file - worth reporting, not crashing
            rows.append({"image_path": path, "width": None, "height": None,
                         "mode": None, "format": None, "error": str(exc)})
    return pd.DataFrame(rows)
