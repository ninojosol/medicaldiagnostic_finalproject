"""Build the 2D slice cache used to train the U-Net at a reasonable throughput.

Decompressing a 143 MB NIfTI volume for every mini-batch is far too slow, so each
split is materialised once into a memory-mapped array of already-preprocessed
slices. The cache is a *training-throughput* device only:

  * cached slices are float16 (halves I/O; the z-scored range is roughly -4..13,
    where float16 resolution is ~1e-2 — immaterial for this task);
  * the **reported** validation metrics, the held-out test metrics and the
    Streamlit inference demo do NOT read the cache. They all go through
    `evaluation.evaluate_case()`, which reads the original NIfTI in float32 via
    `preprocess.prepare_case`/`prepare_slice` — one identical code path.

Slice-retention policy (deterministic, seeded per case):

  train  : every tumour-containing slice, plus `empty_ratio` x n_tumour empty
           brain slices drawn with a per-case RNG seeded from the case id.
  valid  : every slice of the volume, unfiltered.
  test   : every slice of the volume, unfiltered.

Sampling is applied *after* the patient split, per case, so it can never move a
patient or a slice across a split boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import N_MODALITIES, PROCESSED_MRI_DIR, PROJECT_ROOT
from .preprocess import (
    PreprocessConfig,
    brain_slice_indices,
    prepare_case,
    prepare_slice,
    tumour_slice_indices,
)

CACHE_ROOT = PROCESSED_MRI_DIR / "slice_cache"


def cache_dir(split: str, input_size: int) -> Path:
    return CACHE_ROOT / f"{split}_{input_size}"


def _case_rng(case_id: str, seed: int) -> np.random.Generator:
    """Per-case deterministic RNG — reproducible and independent of case order."""
    digest = hashlib.sha256(f"{case_id}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def select_slice_indices(
    norm_volume: np.ndarray,
    binary_label: np.ndarray,
    *,
    split: str,
    case_id: str,
    cfg: PreprocessConfig,
    empty_ratio: float,
    seed: int,
    min_brain_fraction: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Return the slice indices retained for one case, plus a policy record."""
    n_slices = binary_label.shape[cfg.slice_axis]
    tumour_idx = tumour_slice_indices(binary_label, slice_axis=cfg.slice_axis)

    if split != "train":
        # No selection at all for evaluation splits: per-case Dice is computed
        # over the whole volume so the number cannot be flattered by subsetting.
        chosen = np.arange(n_slices, dtype=int)
        return chosen, {
            "policy": "all slices (no filtering)",
            "n_total_slices": int(n_slices),
            "n_tumour_slices": int(tumour_idx.size),
            "n_empty_kept": int(n_slices - tumour_idx.size),
        }

    brain_idx = brain_slice_indices(
        norm_volume, slice_axis=cfg.slice_axis, min_fraction=min_brain_fraction
    )
    empty_pool = np.setdiff1d(brain_idx, tumour_idx, assume_unique=False)
    n_empty = int(round(empty_ratio * tumour_idx.size))
    n_empty = int(min(n_empty, empty_pool.size))

    if n_empty > 0:
        rng = _case_rng(case_id, seed)
        empty_sel = np.sort(rng.choice(empty_pool, size=n_empty, replace=False))
    else:
        empty_sel = np.array([], dtype=int)

    chosen = np.sort(np.concatenate([tumour_idx, empty_sel])).astype(int)
    return chosen, {
        "policy": f"all tumour slices + {empty_ratio:.2f}x deterministic empty brain slices",
        "n_total_slices": int(n_slices),
        "n_tumour_slices": int(tumour_idx.size),
        "n_empty_pool": int(empty_pool.size),
        "n_empty_kept": int(empty_sel.size),
    }


@dataclass
class CacheResult:
    split: str
    path: Path
    n_slices: int
    n_cases: int
    n_tumour_slices: int
    n_empty_slices: int
    bytes_on_disk: int


def build_split_cache(
    manifest: pd.DataFrame,
    split: str,
    cfg: PreprocessConfig,
    *,
    empty_ratio: float = 0.35,
    seed: int = 42,
    overwrite: bool = False,
    progress: bool = True,
) -> CacheResult:
    """Materialise one split's slices into memory-mapped float16/uint8 arrays."""
    out = cache_dir(split, cfg.input_size)
    meta_path = out / "cache_meta.json"

    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") and meta.get("config") == cfg.as_dict():
            print(f"[cache] {split}: reusing existing cache ({meta['n_slices']} slices)")
            return CacheResult(
                split=split,
                path=out,
                n_slices=meta["n_slices"],
                n_cases=meta["n_cases"],
                n_tumour_slices=meta["n_tumour_slices"],
                n_empty_slices=meta["n_slices"] - meta["n_tumour_slices"],
                bytes_on_disk=meta.get("bytes_on_disk", 0),
            )

    out.mkdir(parents=True, exist_ok=True)
    rows = manifest[manifest["split"] == split].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no cases in split '{split}'")

    # Capacity is derived from the manifest, which already carries per-case
    # tumour_slices and total_slices, so each 143 MB volume is decompressed
    # exactly ONCE. For train the figure is an upper bound (the empty pool can be
    # smaller than the requested sample); rows past the true count are simply
    # never referenced by index.csv.
    size = cfg.input_size
    if split == "train":
        capacity = int(
            sum(int(r) + int(round(empty_ratio * int(r))) for r in rows["tumour_slices"])
        )
    else:
        capacity = int(rows["total_slices"].sum())

    img_path = out / "images.npy"
    msk_path = out / "masks.npy"
    images = np.lib.format.open_memmap(
        img_path, mode="w+", dtype=np.float16, shape=(capacity, N_MODALITIES, size, size)
    )
    masks = np.lib.format.open_memmap(
        msk_path, mode="w+", dtype=np.uint8, shape=(capacity, 1, size, size)
    )
    print(f"  [cache/{split}] allocated {capacity} slice slots "
          f"({(images.nbytes + masks.nbytes) / 1e9:.2f} GB)", flush=True)

    index_rows: list[dict] = []
    n_tumour = 0
    cursor = 0
    policy_note = ""
    for j, row in rows.iterrows():
        norm, binary = prepare_case(
            PROJECT_ROOT / row["image_path"], PROJECT_ROOT / row["label_path"], cfg
        )
        idx, policy = select_slice_indices(
            norm, binary, split=split, case_id=row["case_id"],
            cfg=cfg, empty_ratio=empty_ratio, seed=seed,
        )
        policy_note = policy_note or policy["policy"]

        if cursor + idx.size > capacity:  # defensive; should not happen
            raise RuntimeError(
                f"slice cache overflow for '{split}': needed {cursor + idx.size} > {capacity}"
            )

        for sidx in idx:
            image, mask = prepare_slice(norm, binary, int(sidx), cfg)
            images[cursor] = image.numpy().astype(np.float16)
            masks[cursor] = mask.numpy().astype(np.uint8)
            tumour_px = int(mask.sum().item())
            n_tumour += 1 if tumour_px > 0 else 0
            index_rows.append(
                {
                    "row_index": cursor,
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "split": split,
                    "slice_index": int(sidx),
                    "tumour_pixels": tumour_px,
                    "has_tumour": int(tumour_px > 0),
                }
            )
            cursor += 1
        del norm, binary
        if progress and ((j + 1) % 20 == 0 or j + 1 == len(rows)):
            print(f"  [cache/{split}] {j + 1}/{len(rows)} cases -> {cursor} slices", flush=True)

    total = cursor

    images.flush()
    masks.flush()
    del images, masks

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(out / "index.csv", index=False)

    bytes_on_disk = img_path.stat().st_size + msk_path.stat().st_size
    meta = {
        "complete": True,
        "split": split,
        "config": cfg.as_dict(),
        "empty_ratio": empty_ratio if split == "train" else None,
        "seed": seed,
        "n_cases": int(len(rows)),
        "n_slices": int(total),
        "allocated_slots": int(capacity),
        "n_tumour_slices": int(n_tumour),
        "n_empty_slices": int(total - n_tumour),
        "bytes_on_disk": int(bytes_on_disk),
        "dtype_images": "float16",
        "dtype_masks": "uint8",
        "retention_policy": policy_note,
        "note": (
            "Training-throughput cache only. Reported validation, held-out test and "
            "Streamlit inference read the original NIfTI in float32 via "
            "evaluation.evaluate_case()."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[cache] {split}: {total} slices from {len(rows)} cases "
        f"({n_tumour} tumour / {total - n_tumour} empty), {bytes_on_disk / 1e9:.2f} GB"
    )
    return CacheResult(
        split=split,
        path=out,
        n_slices=total,
        n_cases=len(rows),
        n_tumour_slices=n_tumour,
        n_empty_slices=total - n_tumour,
        bytes_on_disk=bytes_on_disk,
    )


def load_cache_meta(split: str, input_size: int) -> dict:
    path = cache_dir(split, input_size) / "cache_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"slice cache missing for split '{split}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))
