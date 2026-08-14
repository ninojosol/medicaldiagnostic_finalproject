"""Phase 5 — the single, one-time held-out internal test evaluation.

Run ONLY after training has finished and the validation-selected checkpoint is
frozen. The script enforces this itself:

  * it refuses to run unless run_metadata.json exists (training completed);
  * it hashes the best checkpoint and records that hash in the test result;
  * it refuses to run a second time unless --force-rerun is passed, and if the
    checkpoint hash changed it says so loudly.

Nothing here retrains, tunes, or selects anything. The threshold is read from the
frozen config, not chosen here.

This is a project-created INTERNAL held-out split from the same public Decathlon
training archive — not an external or clinical test set, and not clinical validation.

Usage::

    python scripts/mri_eval_heldout_test.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.constants import (  # noqa: E402
    HELD_OUT_TEST_NOTE,
    MODEL_LABEL,
    run_dir,
    run_dir_name,
)
from src.segmentation.mri.evaluation import evaluate_case, evaluate_split  # noqa: E402
from src.segmentation.mri.manifests import load_manifest, verify_frozen_manifests  # noqa: E402
from src.segmentation.mri.metrics import aggregate_case_metrics, bootstrap_ci  # noqa: E402
from src.segmentation.mri.preprocess import PreprocessConfig, prepare_slice  # noqa: E402
from src.segmentation.mri.train import load_checkpoint, resolve_device, seed_everything  # noqa: E402
from src.segmentation.mri.unet import build_mri_unet  # noqa: E402
from src.segmentation.mri.viz import overlay_masks, save_overlay_grid, to_uint8_rgb  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    cfg_path = REPO / "configs" / "mri_unet_whole_tumour.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    input_size = args.input_size or int(cfg["preprocess"]["input_size"])
    out = run_dir(input_size)
    name = run_dir_name(input_size)

    meta_path = out / "run_metadata.json"
    if not meta_path.exists():
        print("ERROR: run_metadata.json missing — training has not completed. "
              "The held-out test split must not be touched yet.", file=sys.stderr)
        return 2
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    best_path = out / "models" / f"{name}_best.pt"
    if not best_path.exists():
        print(f"ERROR: frozen checkpoint missing at {best_path}", file=sys.stderr)
        return 2
    ckpt_hash = sha256(best_path)

    receipt_path = out / "metrics" / "heldout_test_evaluation_receipt.json"
    if receipt_path.exists() and not args.force_rerun:
        prev = json.loads(receipt_path.read_text(encoding="utf-8"))
        print("=" * 74)
        print("HELD-OUT TEST ALREADY EVALUATED — refusing to run again.")
        print("=" * 74)
        print(f"  evaluated at    : {prev['evaluated_utc']}")
        print(f"  checkpoint sha  : {prev['checkpoint_sha256'][:16]}...")
        print(f"  mean Dice       : {prev['mean_dice']:.4f}")
        print("\nThis split is a one-shot generalization check. Re-running it and picking "
              "the better number would turn it into a selection set.")
        if prev["checkpoint_sha256"] != ckpt_hash:
            print("\nWARNING: the checkpoint on disk has CHANGED since that evaluation.")
        return 0

    frozen = verify_frozen_manifests()
    if frozen["available"] and not frozen["unchanged"]:
        print("ERROR: MRI split manifests differ from the frozen hashes.", file=sys.stderr)
        return 3

    seed = int(cfg["seed"])
    threshold = float(cfg["eval"]["threshold"])  # frozen, not chosen here
    seed_everything(seed, deterministic=bool(cfg["deterministic"]))
    device = resolve_device(cfg["device"])
    pcfg = PreprocessConfig(input_size=input_size, slice_axis=int(cfg["data"]["slice_axis"]))

    model = build_mri_unet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        features=tuple(cfg["model"]["features"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    ckpt = load_checkpoint(best_path, model, device=device)

    test_manifest = load_manifest("test")

    print("=" * 74)
    print("PHASE 5 — ONE-TIME HELD-OUT INTERNAL TEST EVALUATION")
    print("=" * 74)
    print(f"model          : {MODEL_LABEL}")
    print(f"checkpoint     : {name}_best.pt (epoch {ckpt['epoch']}, sha {ckpt_hash[:16]}...)")
    print(f"threshold      : {threshold} (frozen from config, not tuned here)")
    print(f"test cases     : {len(test_manifest)}")
    print(f"note           : {HELD_OUT_TEST_NOTE}")
    print()

    rows = evaluate_split(
        model, test_manifest, pcfg, device=device, threshold=threshold,
        batch_size=int(cfg["eval"]["batch_size"]), label="held-out test",
    )
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics" / "heldout_test_per_case_metrics.csv", index=False)

    agg = aggregate_case_metrics(rows)
    agg["dice_bootstrap_ci"] = bootstrap_ci(
        [r["dice"] for r in rows], n_boot=int(cfg["eval"]["n_bootstrap"]), seed=seed
    )
    agg["iou_bootstrap_ci"] = bootstrap_ci(
        [r["iou"] for r in rows], n_boot=int(cfg["eval"]["n_bootstrap"]), seed=seed
    )
    agg.update({
        "split": "held_out_internal_test",
        "purpose": "one-time generalization check AFTER the checkpoint was frozen",
        "threshold": threshold,
        "threshold_tuned_on_test": False,
        "used_for_model_selection": False,
        "checkpoint": f"models/{name}_best.pt",
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_sha256": ckpt_hash,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": HELD_OUT_TEST_NOTE,
        "not_clinical_validation": (
            "Not clinical validation, not external validation, not evidence of "
            "diagnostic readiness."
        ),
        "validation_comparison": {
            "validation_mean_dice": run_meta["evaluation"]["validation"]["mean_dice"],
            "validation_mean_iou": run_meta["evaluation"]["validation"]["mean_iou"],
        },
    })
    (out / "metrics" / "heldout_test_aggregate_metrics.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{
        "split": "held_out_internal_test",
        "n_cases": agg["n_cases"],
        "mean_dice": agg["mean_dice"],
        "mean_iou": agg["mean_iou"],
        "median_dice": agg["dice"]["median"],
        "micro_dice": agg["micro"]["dice"],
        "micro_iou": agg["micro"]["iou"],
        "precision": agg["micro"]["precision"],
        "recall": agg["micro"]["recall"],
        "threshold": threshold,
    }]).to_csv(out / "metrics" / "heldout_test_aggregate_metrics.csv", index=False)

    # ---- qualitative test overlays ----------------------------------------
    n_overlays = int(cfg["eval"]["n_test_overlays"])
    ranked = df.sort_values("dice", ascending=False).reset_index(drop=True)
    picks = np.unique(np.linspace(0, len(ranked) - 1, n_overlays).round().astype(int))
    pred_dir = out / "predictions" / "heldout_test"
    pred_dir.mkdir(parents=True, exist_ok=True)

    samples, saved = [], []
    for pos in picks:
        row = ranked.iloc[int(pos)]
        mrow = test_manifest[test_manifest["case_id"] == row["case_id"]].iloc[0]
        res = evaluate_case(
            model, mrow["image_path"], mrow["label_path"], pcfg,
            device=device, threshold=threshold,
            batch_size=int(cfg["eval"]["batch_size"]), return_arrays=True,
        )
        gt = res["gt_masks"]
        areas = gt.reshape(gt.shape[0], -1).sum(axis=1)
        sidx = int(areas.argmax()) if areas.max() > 0 else gt.shape[0] // 2
        image, _ = prepare_slice(res["norm_volume"], None, sidx, pcfg)
        img_np = image.numpy()
        g, p = gt[sidx], res["pred_masks"][sidx]
        samples.append({
            "label": f"{row['case_id']}\nslice {sidx}\nDice {row['dice']:.3f}",
            "panels": [
                to_uint8_rgb(img_np[0]),
                (np.stack([g] * 3, -1) * 255).astype(np.uint8),
                (np.stack([p] * 3, -1) * 255).astype(np.uint8),
                overlay_masks(img_np[0], gt=g, pred=p),
            ],
        })
        np.savez_compressed(
            pred_dir / f"{row['case_id']}_slice{sidx}.npz",
            flair=img_np[0], gt=g, pred=p,
            dice=float(row["dice"]), iou=float(row["iou"]),
            slice_index=sidx, split="held_out_internal_test",
        )
        saved.append({
            "case_id": row["case_id"], "slice_index": sidx,
            "dice": float(row["dice"]), "iou": float(row["iou"]),
            "file": f"predictions/heldout_test/{row['case_id']}_slice{sidx}.npz",
        })
        del res

    save_overlay_grid(
        samples, out / "figures" / "heldout_test_qualitative_examples.png",
        title=(f"Held-out internal test samples — {MODEL_LABEL} ({input_size}x{input_size}) · "
               "amber = ground truth, cyan = prediction, green = agreement"),
        columns=("FLAIR", "Ground truth", "Prediction", "Overlay"),
    )
    (out / "predictions" / "heldout_test_saved_index.json").write_text(
        json.dumps(saved, indent=2), encoding="utf-8"
    )

    receipt = {
        "evaluated_utc": agg["evaluated_utc"],
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "n_cases": agg["n_cases"],
        "mean_dice": agg["mean_dice"],
        "mean_iou": agg["mean_iou"],
        "threshold": threshold,
        "run_count": 1,
        "note": "One-time evaluation. Re-running requires --force-rerun and invalidates the claim.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("HELD-OUT INTERNAL TEST RESULT (one-time, checkpoint frozen)")
    print("=" * 74)
    print(f"cases                 : {agg['n_cases']}")
    print(f"mean per-case Dice    : {agg['mean_dice']:.4f}  "
          f"(95% CI {agg['dice_bootstrap_ci']['lo']:.4f}-{agg['dice_bootstrap_ci']['hi']:.4f})")
    print(f"mean per-case IoU     : {agg['mean_iou']:.4f}  "
          f"(95% CI {agg['iou_bootstrap_ci']['lo']:.4f}-{agg['iou_bootstrap_ci']['hi']:.4f})")
    print(f"median Dice           : {agg['dice']['median']:.4f}")
    print(f"micro Dice / IoU      : {agg['micro']['dice']:.4f} / {agg['micro']['iou']:.4f}")
    print(f"precision / recall    : {agg['micro']['precision']:.4f} / {agg['micro']['recall']:.4f}")
    print(f"\nvalidation comparison : mean Dice "
          f"{agg['validation_comparison']['validation_mean_dice']:.4f} (model selection)")
    print(f"\n{HELD_OUT_TEST_NOTE}")
    print("Not clinical validation. Not external validation. Not diagnostic readiness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
