"""Discover and VALIDATE image/mask pairs for brain MRI tumour segmentation.

Pairing is the number-one source of silent failure in a segmentation project: if
image ``i`` is matched with mask ``j``, training still "works" - the loss goes
down a little and Dice stays near zero - and it can take days to notice. Every
pair is therefore checked here (existence, readability, matching dimensions,
binary mask values) before a single batch is loaded.

Two dataset layouts are supported (``config.data.layout``):

``suffix`` (default, e.g. the LGG / TCGA brain MRI dataset)
    Images and masks live in the same folder, distinguished by a suffix::

        data/raw/mri/TCGA_CS_4941/TCGA_CS_4941_19960909_1.tif
        data/raw/mri/TCGA_CS_4941/TCGA_CS_4941_19960909_1_mask.tif

    The patient ID is the containing folder name.

``parallel``
    Two mirrored directory trees::

        data/raw/mri/images/patient001_slice12.png
        data/raw/mri/masks/patient001_slice12.png
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.errors import DataLayoutError, require_dir, require_nonempty
from ..common.io_utils import list_images


def pairs_from_suffix_layout(root: Path, mask_suffix: str = "_mask",
                             patient_id_regex: str | None = None) -> pd.DataFrame:
    """Pair ``<name><ext>`` with ``<name><mask_suffix><ext>`` in the same folder."""
    root = require_dir(root, what="MRI dataset",
                       hint=f"folders of slices where masks end with '{mask_suffix}'")

    all_images = list_images(root, recursive=True)
    masks = {p for p in all_images if mask_suffix in p.stem}
    images = [p for p in all_images if mask_suffix not in p.stem]
    require_nonempty(images, what="MRI slice images", where=root,
                     hint=f"image files whose names do NOT contain '{mask_suffix}'")

    by_stem = {(p.parent, p.stem): p for p in masks}
    rows, unpaired = [], []
    for image in images:
        mask = by_stem.get((image.parent, image.stem + mask_suffix))
        if mask is None:  # tolerate a different mask extension
            candidates = [m for m in masks
                          if m.parent == image.parent and m.stem == image.stem + mask_suffix]
            mask = candidates[0] if candidates else None
        if mask is None:
            unpaired.append(str(image))
            continue
        rows.append({
            "image_path": str(image),
            "mask_path": str(mask),
            "rel_path": str(image.relative_to(root)).replace("\\", "/"),
            "patient_id": _patient_id(image, root, patient_id_regex),
        })

    if not rows:
        raise DataLayoutError(
            f"Found {len(images)} images under {root} but could not pair ANY of them with a mask.\n"
            f"  Looked for: <image_stem>{mask_suffix}<ext> in the same folder.\n"
            f"  Example image: {images[0].name}\n"
            f"  Files present in that folder: "
            f"{sorted(p.name for p in images[0].parent.iterdir())[:8]}\n"
            "  Fix: set data.mask_suffix in the config, or switch to data.layout: parallel."
        )
    if unpaired:
        print(f"[warn] {len(unpaired)} image(s) had no matching mask and were dropped, "
              f"e.g. {unpaired[:3]}")

    return pd.DataFrame(rows).sort_values("rel_path", ignore_index=True)


def pairs_from_parallel_layout(image_dir: Path, mask_dir: Path,
                               mask_suffix: str = "", patient_id_regex: str | None = None) -> pd.DataFrame:
    """Pair mirrored ``images/`` and ``masks/`` trees by relative path or file stem."""
    image_dir = require_dir(image_dir, what="MRI images", hint="a folder of MRI slice images")
    mask_dir = require_dir(mask_dir, what="MRI masks", hint="a folder of binary tumour masks")

    images = require_nonempty(list_images(image_dir), what="MRI images", where=image_dir)
    masks = list_images(mask_dir)
    by_stem: dict[str, Path] = {}
    for m in masks:
        stem = m.stem[: -len(mask_suffix)] if mask_suffix and m.stem.endswith(mask_suffix) else m.stem
        by_stem.setdefault(stem, m)

    rows, unpaired = [], []
    for image in images:
        rel = image.relative_to(image_dir)
        candidate = mask_dir / rel.with_name(rel.stem + mask_suffix + rel.suffix)
        mask = candidate if candidate.exists() else by_stem.get(image.stem)
        if mask is None:
            unpaired.append(str(image))
            continue
        rows.append({
            "image_path": str(image),
            "mask_path": str(mask),
            "rel_path": str(rel).replace("\\", "/"),
            "patient_id": _patient_id(image, image_dir, patient_id_regex),
        })

    if not rows:
        raise DataLayoutError(
            f"Could not pair any of the {len(images)} images in {image_dir} "
            f"with the {len(masks)} masks in {mask_dir}.\n"
            f"  Example image stem: '{images[0].stem}'\n"
            f"  Example mask stems: {[m.stem for m in masks[:5]]}\n"
            "  Fix: set data.mask_suffix so image and mask names line up."
        )
    if unpaired:
        print(f"[warn] {len(unpaired)} image(s) had no matching mask and were dropped.")

    return pd.DataFrame(rows).sort_values("rel_path", ignore_index=True)


def _patient_id(path: Path, root: Path, regex: str | None) -> str:
    """Patient ID from a regex, else the first sub-folder under the dataset root."""
    if regex:
        for text in (path.name, str(path)):
            match = re.search(regex, text)
            if match:
                return match.group(1) if match.groups() else match.group(0)
    try:
        rel = path.relative_to(root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    return path.parent.name or path.stem


def build_pairs(cfg) -> pd.DataFrame:
    """Build the image/mask pair table described by ``cfg.data``."""
    data = cfg.data
    layout = str(data.get("layout", "suffix")).lower()
    mask_suffix = str(data.get("mask_suffix", "_mask"))
    regex = data.get("patient_id_regex")

    if layout == "suffix":
        return pairs_from_suffix_layout(Path(cfg.path("data.root")), mask_suffix, regex)
    if layout == "parallel":
        root = Path(cfg.path("data.root"))
        image_dir = Path(cfg.path("data.image_dir")) if data.get("image_dir") else root / "images"
        mask_dir = Path(cfg.path("data.mask_dir")) if data.get("mask_dir") else root / "masks"
        return pairs_from_parallel_layout(image_dir, mask_dir, mask_suffix, regex)
    raise DataLayoutError(f"Unknown data.layout={layout!r}. Use 'suffix' or 'parallel'.")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_pairs(df: pd.DataFrame, sample: int | None = None, seed: int = 42,
                   verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Open every pair and check it is usable. Returns ``(per_pair_table, summary)``.

    Checks performed
    ----------------
    * both files exist and can be decoded;
    * image and mask have identical height/width;
    * the mask is effectively binary (at most 2 distinct values after thresholding);
    * records tumour pixel fraction, which the EDA notebook plots.

    Nothing is dropped automatically - the notebook decides, and the summary
    tells you exactly what would be dropped and why.
    """
    from PIL import Image

    subset = df if sample is None or sample >= len(df) else df.sample(sample, random_state=seed)
    rows = []

    for record in subset.itertuples(index=False):
        row = {"image_path": record.image_path, "mask_path": record.mask_path,
               "patient_id": getattr(record, "patient_id", ""), "ok": False, "problem": ""}
        try:
            with Image.open(record.image_path) as im:
                image = np.array(im)
            with Image.open(record.mask_path) as mm:
                mask = np.array(mm.convert("L"))
        except Exception as exc:
            row["problem"] = f"unreadable: {exc}"
            rows.append(row)
            continue

        row.update({
            "image_h": image.shape[0], "image_w": image.shape[1],
            "image_channels": 1 if image.ndim == 2 else image.shape[2],
            "mask_h": mask.shape[0], "mask_w": mask.shape[1],
            "image_dtype": str(image.dtype), "mask_dtype": str(mask.dtype),
            "mask_unique_values": int(len(np.unique(mask))),
            "mask_max": int(mask.max()),
        })

        binary = (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)
        row["tumour_pixel_fraction"] = round(float(binary.mean()), 6)
        row["has_tumour"] = int(binary.sum() > 0)

        problems = []
        if image.shape[:2] != mask.shape[:2]:
            problems.append(f"size mismatch image{image.shape[:2]} vs mask{mask.shape[:2]}")
        if row["mask_unique_values"] > 2 and len(np.unique(binary)) > 2:
            problems.append(f"mask is not binary ({row['mask_unique_values']} distinct values)")
        row["problem"] = "; ".join(problems)
        row["ok"] = not problems
        rows.append(row)

    table = pd.DataFrame(rows)
    bad = table[~table["ok"]]
    summary = {
        "n_pairs_checked": int(len(table)),
        "n_ok": int(table["ok"].sum()),
        "n_problems": int(len(bad)),
        "problems_sample": bad[["image_path", "problem"]].head(10).to_dict(orient="records"),
        "n_patients": int(table["patient_id"].nunique()) if "patient_id" in table else 0,
        "n_slices_with_tumour": int(table["has_tumour"].sum()) if "has_tumour" in table else 0,
        "empty_mask_rate": round(float(1 - table["has_tumour"].mean()), 4) if "has_tumour" in table else None,
        "mean_tumour_pixel_fraction": round(float(table["tumour_pixel_fraction"].mean()), 6)
        if "tumour_pixel_fraction" in table else None,
        "image_sizes": table.groupby(["image_h", "image_w"]).size().to_dict()
        if "image_h" in table else {},
    }

    if verbose:
        print(f"Pairs checked        : {summary['n_pairs_checked']}")
        print(f"Valid pairs          : {summary['n_ok']}")
        print(f"Problem pairs        : {summary['n_problems']}")
        print(f"Patients             : {summary['n_patients']}")
        print(f"Slices with tumour   : {summary['n_slices_with_tumour']} "
              f"({1 - (summary['empty_mask_rate'] or 0):.1%})")
        print(f"Empty-mask rate      : {summary['empty_mask_rate']}")
        print(f"Mean tumour fraction : {summary['mean_tumour_pixel_fraction']} "
              "  <- foreground/background imbalance")
        if summary["n_problems"]:
            print("\n[warn] problem pairs found - inspect before training:")
            for item in summary["problems_sample"][:5]:
                print(f"    {Path(item['image_path']).name}: {item['problem']}")
        else:
            print("\n[ok] every checked pair is readable, size-matched and binary.")

    return table, summary
