"""Phase 3 — materialise the 2D slice cache for one or more splits.

Usage::

    python scripts/mri_build_slice_cache.py --splits train valid
    python scripts/mri_build_slice_cache.py --splits test      # only at Phase 5

The test split is NOT built by default: it is only needed for the single final
held-out evaluation, and even then the reported test metrics come from the
NIfTI float32 path, not this cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.cache import CACHE_ROOT, build_split_cache  # noqa: E402
from src.segmentation.mri.manifests import load_manifest  # noqa: E402
from src.segmentation.mri.preprocess import PreprocessConfig  # noqa: E402

CONFIG = REPO / "configs" / "mri_unet_whole_tumour.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "valid"],
                    choices=["train", "valid", "test"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pcfg = PreprocessConfig(
        input_size=int(cfg_raw["preprocess"]["input_size"]),
        slice_axis=int(cfg_raw["data"]["slice_axis"]),
    )
    empty_ratio = float(cfg_raw["slices"]["empty_slice_ratio"])
    seed = int(cfg_raw["slices"]["sampling_seed"])

    manifest = load_manifest("all")
    print(f"[cache] input_size={pcfg.input_size} empty_ratio={empty_ratio} seed={seed}")
    print(f"[cache] root: {CACHE_ROOT.relative_to(REPO).as_posix()}")

    results = {}
    for split in args.splits:
        res = build_split_cache(
            manifest, split, pcfg,
            empty_ratio=empty_ratio, seed=seed,
            overwrite=args.overwrite, progress=True,
        )
        results[split] = {
            "n_cases": res.n_cases,
            "n_slices": res.n_slices,
            "n_tumour_slices": res.n_tumour_slices,
            "n_empty_slices": res.n_empty_slices,
            "gb": round(res.bytes_on_disk / 1e9, 2),
        }

    print("\n" + "=" * 70)
    print("SLICE CACHE SUMMARY")
    print("=" * 70)
    for split, r in results.items():
        print(
            f"{split:<6} cases={r['n_cases']:<4} slices={r['n_slices']:<7} "
            f"tumour={r['n_tumour_slices']:<7} empty={r['n_empty_slices']:<7} {r['gb']:.2f} GB"
        )

    audit = REPO / "outputs" / "segmentation" / "_audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "slice_cache_summary.json").write_text(
        json.dumps({"input_size": pcfg.input_size, "empty_ratio": empty_ratio,
                    "seed": seed, "splits": results}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
