"""Phase 3.11 — visual validation gate: prove masks align with tumour anatomy.

Renders random image/mask overlays from the TRAIN split only (never valid/test),
and runs quantitative alignment assertions that would catch a flipped, transposed
or off-by-one mask even if nobody looked at the picture:

  1. tumour voxels must sit inside brain tissue — the fraction of tumour voxels
     falling on zero-intensity background must be tiny;
  2. FLAIR intensity inside the tumour must be measurably higher than in the
     surrounding brain (whole tumour is hyperintense on FLAIR — this is the
     property that breaks first if the mask is misaligned);
  3. tumour centroid must lie within the brain bounding box.

Exits non-zero and prints a STOP banner if alignment looks wrong.

Usage::

    python scripts/mri_visual_validation.py [--n 8] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.constants import MODALITY_NAMES, PROJECT_ROOT, run_dir  # noqa: E402
from src.segmentation.mri.manifests import load_manifest  # noqa: E402
from src.segmentation.mri.preprocess import (  # noqa: E402
    PreprocessConfig,
    prepare_case,
    prepare_slice,
    tumour_slice_indices,
)
from src.segmentation.mri.viz import overlay_masks, save_overlay_grid, to_uint8_rgb  # noqa: E402

CONFIG = REPO / "configs" / "mri_unet_whole_tumour.yaml"

# Thresholds for the automated alignment assertions.
MAX_TUMOUR_ON_BACKGROUND = 0.05   # <=5% of tumour voxels may fall on zero intensity
MIN_FLAIR_CONTRAST = 0.20         # tumour FLAIR mean must exceed brain mean by >=0.20 sd


def check_alignment(norm_volume: np.ndarray, binary: np.ndarray, cfg: PreprocessConfig) -> dict:
    """Quantitative alignment checks on one case, at full volume resolution."""
    flair = norm_volume[..., 0]           # channel 0 is FLAIR
    brain = (norm_volume != 0).any(axis=3)
    tumour = binary.astype(bool)

    n_tumour = int(tumour.sum())
    if n_tumour == 0:
        return {"skipped": True, "reason": "case has no tumour voxels"}

    on_background = float((tumour & ~brain).sum() / n_tumour)

    tumour_flair = float(flair[tumour].mean())
    brain_only = brain & ~tumour
    brain_flair = float(flair[brain_only].mean()) if brain_only.any() else 0.0
    contrast = tumour_flair - brain_flair

    coords = np.argwhere(tumour)
    centroid = coords.mean(axis=0)
    bcoords = np.argwhere(brain)
    bmin, bmax = bcoords.min(axis=0), bcoords.max(axis=0)
    centroid_inside = bool(np.all(centroid >= bmin) and np.all(centroid <= bmax))

    passed = (
        on_background <= MAX_TUMOUR_ON_BACKGROUND
        and contrast >= MIN_FLAIR_CONTRAST
        and centroid_inside
    )
    return {
        "skipped": False,
        "passed": passed,
        "n_tumour_voxels": n_tumour,
        "tumour_on_background_fraction": round(on_background, 5),
        "flair_tumour_mean": round(tumour_flair, 4),
        "flair_brain_mean": round(brain_flair, 4),
        "flair_contrast": round(contrast, 4),
        "tumour_centroid": [round(float(c), 1) for c in centroid],
        "brain_bbox_min": [int(v) for v in bmin],
        "brain_bbox_max": [int(v) for v in bmax],
        "centroid_inside_brain_bbox": centroid_inside,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of random cases (min 6 required)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_size = int(cfg_raw["preprocess"]["input_size"])
    pcfg = PreprocessConfig(
        input_size=input_size, slice_axis=int(cfg_raw["data"]["slice_axis"])
    )

    # TRAIN split only — the visual gate must not peek at valid or test data.
    train = load_manifest("train")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(train), size=min(args.n, len(train)), replace=False)

    out_root = run_dir(input_size) / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    samples, checks = [], []
    for i in picks:
        row = train.iloc[int(i)]
        norm, binary = prepare_case(
            PROJECT_ROOT / row["image_path"], PROJECT_ROOT / row["label_path"], pcfg
        )
        chk = check_alignment(norm, binary, pcfg)
        chk["case_id"] = row["case_id"]
        checks.append(chk)

        tidx = tumour_slice_indices(binary, slice_axis=pcfg.slice_axis)
        if tidx.size == 0:
            continue
        # middle tumour slice — the most representative view of the lesion
        sidx = int(tidx[len(tidx) // 2])
        image, mask = prepare_slice(norm, binary, sidx, pcfg)
        img_np = image.numpy()
        m_np = mask.squeeze(0).numpy()

        samples.append(
            {
                "label": f"{row['case_id']}\nslice {sidx}\nDice-target voxels: {int(m_np.sum())}",
                "panels": [
                    to_uint8_rgb(img_np[0]),                       # FLAIR
                    to_uint8_rgb(img_np[3]),                       # T2w
                    (np.stack([m_np] * 3, -1) * 255).astype(np.uint8),  # mask
                    overlay_masks(img_np[0], gt=m_np),             # overlay
                ],
            }
        )
        print(
            f"  {row['case_id']}: slice {sidx}  tumour_px={int(m_np.sum()):<6} "
            f"on_bg={chk.get('tumour_on_background_fraction')} "
            f"flair_contrast={chk.get('flair_contrast')} "
            f"{'PASS' if chk.get('passed') else 'FAIL'}",
            flush=True,
        )
        del norm, binary

    fig_path = out_root / "visual_validation_train_overlays.png"
    save_overlay_grid(
        samples,
        fig_path,
        title=(
            f"Visual validation — MRI slice / whole-tumour mask alignment "
            f"(train split, {input_size}x{input_size})"
        ),
        columns=(f"{MODALITY_NAMES[0]}", f"{MODALITY_NAMES[3]}",
                 "Ground-truth mask", "Overlay (amber = tumour)"),
    )

    evaluated = [c for c in checks if not c.get("skipped")]
    failures = [c for c in evaluated if not c["passed"]]
    report = {
        "input_size": input_size,
        "n_cases_checked": len(evaluated),
        "n_overlays_rendered": len(samples),
        "thresholds": {
            "max_tumour_on_background_fraction": MAX_TUMOUR_ON_BACKGROUND,
            "min_flair_contrast_sd": MIN_FLAIR_CONTRAST,
        },
        "all_passed": not failures,
        "failures": failures,
        "checks": checks,
        "figure": fig_path.relative_to(REPO).as_posix(),
    }
    (out_root.parent / "metrics").mkdir(parents=True, exist_ok=True)
    (out_root.parent / "metrics" / "visual_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("PHASE 3 VISUAL VALIDATION")
    print("=" * 70)
    print(f"cases checked   : {len(evaluated)}")
    print(f"overlays        : {len(samples)} -> {fig_path.relative_to(REPO).as_posix()}")
    mean_contrast = np.mean([c["flair_contrast"] for c in evaluated]) if evaluated else 0
    mean_onbg = np.mean([c["tumour_on_background_fraction"] for c in evaluated]) if evaluated else 0
    print(f"mean FLAIR contrast (tumour - brain, in sd): {mean_contrast:.3f}")
    print(f"mean tumour-on-background fraction        : {mean_onbg:.5f}")

    if failures:
        print("\n" + "!" * 70)
        print("STOP — MASK ALIGNMENT LOOKS WRONG. Do not train.")
        print("!" * 70)
        for f in failures:
            print(f"  {f['case_id']}: {f}")
        return 1

    print("\nALIGNMENT OK — masks sit on FLAIR-hyperintense tissue inside the brain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
