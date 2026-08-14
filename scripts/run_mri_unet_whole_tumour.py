"""Phase 4 — train the 2D slice-wise U-Net for binary whole-tumour segmentation.

Model: 2D slice-wise U-Net using four MRI sequences (4 in / 1 binary out).

Everything that selects the model — checkpointing, LR schedule, early stopping —
uses the VALIDATION split only. The held-out internal test split is never read by
this script; it has no code path to it. Test evaluation is a separate, one-time
script (scripts/mri_eval_heldout_test.py) run after the checkpoint is frozen.

Usage::

    python scripts/run_mri_unet_whole_tumour.py
    python scripts/run_mri_unet_whole_tumour.py --epochs 2 --limit-train 400   # debug
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.constants import (  # noqa: E402
    DATASET_ID,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_URL,
    HELD_OUT_TEST_NOTE,
    MODALITY_NAMES,
    MODEL_ARCH_NOTE,
    MODEL_LABEL,
    PROJECT_ROOT,
    SLICE_AXIS_NAME,
    SOURCE_LABELS,
    SPLIT_SUMMARY_JSON,
    WHOLE_TUMOUR_RULE,
    run_dir,
    run_dir_name,
)
from src.segmentation.mri.cache import load_cache_meta  # noqa: E402
from src.segmentation.mri.dataset import SliceAugment, build_dataloaders  # noqa: E402
from src.segmentation.mri.evaluation import evaluate_case  # noqa: E402
from src.segmentation.mri.losses import DiceBCELoss  # noqa: E402
from src.segmentation.mri.manifests import load_manifest, verify_frozen_manifests  # noqa: E402
from src.segmentation.mri.metrics import aggregate_case_metrics, bootstrap_ci  # noqa: E402
from src.segmentation.mri.preprocess import PreprocessConfig, prepare_case, prepare_slice  # noqa: E402
from src.segmentation.mri.train import (  # noqa: E402
    load_checkpoint,
    resolve_device,
    seed_everything,
    train_model,
)
from src.segmentation.mri.unet import build_mri_unet, count_parameters  # noqa: E402
from src.segmentation.mri.viz import (  # noqa: E402
    overlay_masks,
    save_overlay_grid,
    save_training_history_figure,
    to_uint8_rgb,
)

CONFIG = REPO / "configs" / "mri_unet_whole_tumour.yaml"


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit-valid-cases", type=int, default=0,
                    help="debug only: cap the number of validation cases re-scored at the end")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    input_size = int(cfg["preprocess"]["input_size"])
    seed = int(cfg["seed"])
    threshold = float(cfg["eval"]["threshold"])

    out = run_dir(input_size)
    for sub in ("models", "metrics", "figures", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    seed_everything(seed, deterministic=bool(cfg["deterministic"]))
    device = resolve_device(cfg["device"])

    # ---- guard: manifests must be the frozen ones --------------------------
    frozen = verify_frozen_manifests()
    if frozen["available"] and not frozen["unchanged"]:
        print("ERROR: MRI split manifests differ from the frozen hashes.", file=sys.stderr)
        return 3

    pcfg = PreprocessConfig(
        input_size=input_size, slice_axis=int(cfg["data"]["slice_axis"])
    )

    train_meta = load_cache_meta("train", input_size)
    valid_meta = load_cache_meta("valid", input_size)
    manifest_all = load_manifest("all")
    n_train_cases = int((manifest_all["split"] == "train").sum())
    n_valid_cases = int((manifest_all["split"] == "valid").sum())
    n_test_cases = int((manifest_all["split"] == "test").sum())

    print("=" * 74)
    print(f"MRI WHOLE-TUMOUR SEGMENTATION — {MODEL_LABEL}")
    print("=" * 74)
    print(f"run dir        : outputs/segmentation/{run_dir_name(input_size)}")
    print(f"device         : {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")
    print(f"input size     : {input_size}x{input_size}   modalities: {list(MODALITY_NAMES)}")
    print(f"target         : binary whole tumour — {WHOLE_TUMOUR_RULE}")
    print(f"cases          : train={n_train_cases} valid={n_valid_cases} "
          f"test={n_test_cases} (test NOT read here)")
    print(f"slices         : train={train_meta['n_slices']} valid={valid_meta['n_slices']}")
    print()

    # ---- model / loss / data ----------------------------------------------
    model = build_mri_unet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        features=tuple(cfg["model"]["features"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    n_params = count_parameters(model)
    print(f"model params   : {n_params:,}")

    loss_fn = DiceBCELoss(
        bce_weight=cfg["train"]["bce_weight"],
        dice_weight=cfg["train"]["dice_weight"],
        smooth=cfg["train"]["dice_smooth"],
        pos_weight=cfg["train"].get("pos_weight"),
    ).to(device)
    print(f"loss           : {loss_fn.describe()}")

    augment = SliceAugment(hflip_p=0.5, vflip_p=0.0, max_shift_frac=0.0625,
                           intensity_scale=0.1, intensity_shift=0.1, p_intensity=0.5)
    train_loader, valid_loader, train_ds, valid_ds = build_dataloaders(
        pcfg,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        seed=seed,
        augment=augment,
        pin_memory=device.type == "cuda",
    )
    print(f"batches        : train={len(train_loader)} valid={len(valid_loader)}\n")

    # ---- train -------------------------------------------------------------
    started = time.time()
    state = train_model(
        model, train_loader, valid_loader, loss_fn,
        device=device,
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        amp=bool(cfg["train"]["amp"]),
        grad_clip=cfg["train"].get("grad_clip"),
        threshold=threshold,
        early_stopping_patience=int(cfg["train"]["early_stopping_patience"]),
        scheduler_patience=int(cfg["train"]["scheduler_patience"]),
        scheduler_factor=float(cfg["train"]["scheduler_factor"]),
        min_lr=float(cfg["train"]["min_lr"]),
        checkpoint_dir=out / "models",
        run_name=run_dir_name(input_size),
        history_csv=out / "metrics" / "training_history.csv",
        train_dataset=train_ds,
    )
    training_seconds = time.time() - started
    print(f"\n[train] finished in {training_seconds / 60:.1f} min — "
          f"best epoch {state.best_epoch}, validation Dice {state.best_metric:.4f}")

    # ---- CHECKPOINT IS NOW FROZEN -----------------------------------------
    best_path = out / "models" / f"{run_dir_name(input_size)}_best.pt"
    ckpt = load_checkpoint(best_path, model, device=device)
    print(f"[freeze] reloaded best checkpoint from epoch {ckpt['epoch']}")

    # ---- reported validation metrics: NIfTI float32 path -------------------
    # Identical function to the held-out test evaluation and the Streamlit demo.
    valid_manifest = load_manifest("valid")
    if args.limit_valid_cases:
        valid_manifest = valid_manifest.head(args.limit_valid_cases)

    print(f"\n[validation] per-case volumetric scoring over {len(valid_manifest)} cases "
          f"(original NIfTI, float32)")
    from src.segmentation.mri.evaluation import evaluate_split

    val_rows = evaluate_split(
        model, valid_manifest, pcfg, device=device, threshold=threshold,
        batch_size=int(cfg["eval"]["batch_size"]), label="validation",
    )
    val_df = pd.DataFrame(val_rows)
    val_csv = out / "metrics" / "validation_per_case_metrics.csv"
    val_df.to_csv(val_csv, index=False)

    val_agg = aggregate_case_metrics(val_rows)
    val_agg["dice_bootstrap_ci"] = bootstrap_ci(
        [r["dice"] for r in val_rows], n_boot=int(cfg["eval"]["n_bootstrap"]), seed=seed
    )
    val_agg["iou_bootstrap_ci"] = bootstrap_ci(
        [r["iou"] for r in val_rows], n_boot=int(cfg["eval"]["n_bootstrap"]), seed=seed
    )
    val_agg["threshold"] = threshold
    val_agg["split"] = "validation"
    val_agg["purpose"] = "model selection — this split chose the checkpoint"
    val_agg["best_epoch"] = state.best_epoch
    val_agg["in_training_valid_micro_dice"] = state.best_metric
    (out / "metrics" / "validation_aggregate_metrics.json").write_text(
        json.dumps(val_agg, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{
        "split": "validation",
        "n_cases": val_agg["n_cases"],
        "mean_dice": val_agg["mean_dice"],
        "mean_iou": val_agg["mean_iou"],
        "median_dice": val_agg["dice"]["median"],
        "micro_dice": val_agg["micro"]["dice"],
        "micro_iou": val_agg["micro"]["iou"],
        "precision": val_agg["micro"]["precision"],
        "recall": val_agg["micro"]["recall"],
        "threshold": threshold,
    }]).to_csv(out / "metrics" / "validation_aggregate_metrics.csv", index=False)

    print(f"\n[validation] mean per-case Dice = {val_agg['mean_dice']:.4f}  "
          f"mean IoU = {val_agg['mean_iou']:.4f}  micro Dice = {val_agg['micro']['dice']:.4f}")

    # ---- qualitative validation overlays -----------------------------------
    n_overlays = int(cfg["eval"]["n_validation_overlays"])
    ranked = val_df.sort_values("dice", ascending=False).reset_index(drop=True)
    pick_idx = np.unique(np.linspace(0, len(ranked) - 1, n_overlays).round().astype(int))
    pred_dir = out / "predictions" / "validation"
    pred_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    saved_cases = []
    for rank_pos in pick_idx:
        row = ranked.iloc[int(rank_pos)]
        mrow = valid_manifest[valid_manifest["case_id"] == row["case_id"]].iloc[0]
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
            slice_index=sidx, split="validation",
        )
        saved_cases.append({
            "case_id": row["case_id"], "slice_index": sidx,
            "dice": float(row["dice"]), "iou": float(row["iou"]),
            "file": f"predictions/validation/{row['case_id']}_slice{sidx}.npz",
        })
        del res

    save_overlay_grid(
        samples, out / "figures" / "validation_qualitative_examples.png",
        title=(f"Validation samples — {MODEL_LABEL} ({input_size}x{input_size}) · "
               "amber = ground truth, cyan = prediction, green = agreement"),
        columns=("FLAIR", "Ground truth", "Prediction", "Overlay"),
    )
    (out / "predictions" / "validation_saved_index.json").write_text(
        json.dumps(saved_cases, indent=2), encoding="utf-8"
    )

    save_training_history_figure(
        out / "metrics" / "training_history.csv",
        out / "figures" / "training_history.png",
        title=f"Training history — {MODEL_LABEL} ({input_size}x{input_size})",
    )

    # ---- artifacts: config used + run metadata -----------------------------
    shutil.copyfile(args.config, out / "config_used.yaml")
    if SPLIT_SUMMARY_JSON.exists():
        shutil.copyfile(SPLIT_SUMMARY_JSON, out / "split_summary_used.json")

    smoke_path = REPO / "outputs" / "segmentation" / "_audit" / "gpu_smoke_test.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else None
    chosen_smoke = None
    if smoke:
        chosen_smoke = next(
            (r for r in smoke["results"]
             if r["input_size"] == input_size and r["batch_size"] == cfg["train"]["batch_size"]),
            None,
        )

    metadata = {
        "run_name": run_dir_name(input_size),
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head(),
        "task": {
            "name": "binary whole-tumour MRI segmentation",
            "target": "whole tumour vs background",
            "binary_rule": WHOLE_TUMOUR_RULE,
            "source_label_mapping": {str(k): v for k, v in SOURCE_LABELS.items()},
            "not_multiclass_subregion": True,
        },
        "dataset": {
            "id": DATASET_ID, "name": DATASET_NAME, "url": DATASET_URL, "licence": DATASET_LICENCE,
            "modalities": list(MODALITY_NAMES),
            "slice_axis": SLICE_AXIS_NAME,
            "train_cases": n_train_cases, "valid_cases": n_valid_cases, "test_cases": n_test_cases,
            "train_slices": train_meta["n_slices"], "valid_slices": valid_meta["n_slices"],
            "train_tumour_slices": train_meta["n_tumour_slices"],
            "train_empty_slices": train_meta["n_empty_slices"],
            "slice_retention_policy": train_meta["retention_policy"],
            "held_out_test_note": HELD_OUT_TEST_NOTE,
        },
        "model": {
            "label": MODEL_LABEL,
            "architecture_note": MODEL_ARCH_NOTE,
            "name": cfg["model"]["name"],
            "in_channels": cfg["model"]["in_channels"],
            "out_channels": cfg["model"]["out_channels"],
            "features": cfg["model"]["features"],
            "parameters": n_params,
            "output": "logits (sigmoid applied only for metrics/inference)",
        },
        "preprocessing": pcfg.as_dict(),
        "training": {
            "epochs_configured": int(cfg["train"]["epochs"]),
            "epochs_completed": len(state.history),
            "best_epoch": state.best_epoch,
            "best_validation_micro_dice_in_training": state.best_metric,
            "batch_size": int(cfg["train"]["batch_size"]),
            "optimizer": "AdamW",
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "scheduler": "ReduceLROnPlateau(mode=max, factor=%.2f, patience=%d)" % (
                cfg["train"]["scheduler_factor"], cfg["train"]["scheduler_patience"]),
            "early_stopping_patience": int(cfg["train"]["early_stopping_patience"]),
            "selection_metric": "validation micro Dice (validation split only)",
            "loss": loss_fn.describe(),
            "amp": bool(cfg["train"]["amp"]) and device.type == "cuda",
            "grad_clip": cfg["train"].get("grad_clip"),
            "seed": seed,
            "deterministic": bool(cfg["deterministic"]),
            "num_workers": int(cfg["train"]["num_workers"]),
            "training_seconds": round(training_seconds, 1),
            "training_minutes": round(training_seconds / 60, 2),
            "augmentation": "hflip p=0.5; integer shift +/-6.25%; brain-masked intensity jitter p=0.5",
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if device.type == "cuda" else None,
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_smoke_test": chosen_smoke,
        },
        "evaluation": {
            "threshold": threshold,
            "threshold_tuned": False,
            "metric_definition": (
                "per-case volumetric Dice/IoU — confusion counts accumulated over every "
                "slice of a case, then Dice computed once from the case totals"
            ),
            "validation": {
                "n_cases": val_agg["n_cases"],
                "mean_dice": val_agg["mean_dice"],
                "mean_iou": val_agg["mean_iou"],
                "micro_dice": val_agg["micro"]["dice"],
                "micro_iou": val_agg["micro"]["iou"],
                "purpose": "model selection",
            },
            "held_out_test": "not run by this script — see scripts/mri_eval_heldout_test.py",
        },
        "protected_test_split": {
            "read_during_training": False,
            "used_for_architecture_choice": False,
            "used_for_hyperparameter_tuning": False,
            "used_for_early_stopping": False,
            "used_for_threshold_selection": False,
        },
        "artifacts": {
            "best_checkpoint": f"models/{run_dir_name(input_size)}_best.pt",
            "last_checkpoint": f"models/{run_dir_name(input_size)}_last.pt",
            "training_history": "metrics/training_history.csv",
            "validation_per_case": "metrics/validation_per_case_metrics.csv",
            "validation_aggregate": "metrics/validation_aggregate_metrics.json",
            "validation_overlays": "figures/validation_qualitative_examples.png",
            "training_history_figure": "figures/training_history.png",
            "config_used": "config_used.yaml",
            "split_summary": "split_summary_used.json",
        },
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("PHASE 4 COMPLETE")
    print("=" * 74)
    print(f"epochs completed / best : {len(state.history)} / {state.best_epoch}")
    print(f"training duration       : {training_seconds / 60:.1f} min")
    print(f"validation mean Dice    : {val_agg['mean_dice']:.4f}")
    print(f"validation mean IoU     : {val_agg['mean_iou']:.4f}")
    print(f"artifacts               : outputs/segmentation/{run_dir_name(input_size)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
