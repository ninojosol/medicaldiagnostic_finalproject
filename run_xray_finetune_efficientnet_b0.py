"""
Fine-tuned EfficientNet-B0 (validation-only) for Chest X-ray multilabel classification.

This script trains:
  1) a head-only stage with the EfficientNet-B0 backbone frozen
  2) a conservative fine-tuning stage where only the last blocks are unfrozen

It then runs validation-only evaluation to generate:
  - thresholds + per-label metrics
  - per-image validation predictions
  - training history + best checkpoint (from early stopping)

Finally, it updates `outputs/classification/model_comparison.json` by adding the new
EfficientNet-B0 validation-only results (without changing baseline/DenseNet entries).
"""

from __future__ import annotations

import json
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch

from src.common import get_device, load_config, seed_everything
from src.classification import (
    build_dataloaders,
    build_loss,
    build_model,
    compute_pos_weights,
    evaluate_classifier,
    load_checkpoint,
    train_model,
)
from src.classification.models import unfreeze_efficientnet_backbone_blocks


def _to_optional_float(x):
    # Keep JSON valid and make NaN explicitly "not computable".
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _row_to_dict(row: pd.Series) -> dict:
    out = {}
    for k, v in row.to_dict().items():
        if k in {"roc_auc", "pr_auc"}:
            out[k] = _to_optional_float(v)
        else:
            # Preserve ints/floats as normal JSON numbers.
            try:
                if isinstance(v, (np.integer,)):
                    out[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    out[k] = float(v)
                else:
                    out[k] = v
            except Exception:
                out[k] = v
    return out


def _make_validation_manifest(train_csv: Path, valid_csv: Path, image_root: Path) -> tuple[pd.DataFrame, list[str]]:
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    labels = [c for c in train_df.columns if c not in {"Image", "PatientId"}]
    if not labels:
        raise RuntimeError("No label columns detected in train_clean.csv")

    def to_manifest(df: pd.DataFrame, split: str) -> pd.DataFrame:
        out = pd.DataFrame()
        # Build absolute on-disk paths for the dataset loader.
        out["image_path"] = df["Image"].astype(str).map(lambda name: str(image_root / name)).values
        out["patient_id"] = df["PatientId"].astype(str).values
        for lab in labels:
            out[lab] = df[lab].astype(np.float32).values
        out["split"] = split
        return out

    manifest = pd.concat([to_manifest(train_df, "train"), to_manifest(valid_df, "val")], ignore_index=True)
    return manifest, labels


def _extract_val_metrics(output_root: Path, run_name: str) -> dict:
    metrics_path = output_root / run_name / "metrics" / "metrics_validation.csv"
    df = pd.read_csv(metrics_path)

    macro_row = df[df["label"] == "MACRO_AVG"].iloc[0]
    per_label_rows = df[df["label"] != "MACRO_AVG"].copy()

    return {
        "val_macro_metrics": _row_to_dict(macro_row),
        "val_metrics": {"per_label": [_row_to_dict(r) for _, r in per_label_rows.iterrows()]},
    }


def main() -> None:
    project_root = Path.cwd()
    # Allow running from repo root or elsewhere.
    if (project_root / "src").is_dir() is False:
        project_root = Path(__file__).resolve().parent

    # -- output names ---------------------------------------------------------
    stage1_run_name = "xray_finetuned_efficientnet_b0_multilabel_320_stage1_head_only"
    stage2_run_name = "xray_finetuned_efficientnet_b0_multilabel_320"
    output_root = project_root / "outputs" / "classification"

    # -- configs --------------------------------------------------------------
    template_stage1 = (
        output_root
        / "xray_finetuned_densenet_multilabel_320_stage1_head_only"
        / "config_used.yaml"
    )
    template_stage2 = output_root / "xray_finetuned_densenet_multilabel_320" / "config_used.yaml"

    if not template_stage1.is_file() or not template_stage2.is_file():
        raise FileNotFoundError(
            "DenseNet config snapshots not found. "
            "Expected: outputs/classification/xray_finetuned_densenet_multilabel_320*/config_used.yaml"
        )

    pretrained_source = "torchvision ImageNet weights (weights=DEFAULT) for efficientnet_b0"

    cfg_stage1 = load_config(
        template_stage1,
        overrides={
            "run_name": stage1_run_name,
            "model": {
                "name": "efficientnet_b0",
                "pretrained": True,
                "freeze_backbone": True,
                "pretrained_source": pretrained_source,
            },
        },
    )

    cfg_stage2 = load_config(
        template_stage2,
        overrides={
            "run_name": stage2_run_name,
            "model": {
                "name": "efficientnet_b0",
                "pretrained": True,
                # We freeze the whole backbone first; in code we selectively unfreeze the last blocks.
                "freeze_backbone": True,
                "efficientnet_unfreeze_last_blocks": 2,
                "pretrained_source": pretrained_source,
            },
        },
    )

    # -- data manifest --------------------------------------------------------
    train_csv = project_root / "data" / "processed" / "xray" / "train_clean.csv"
    valid_csv = project_root / "data" / "processed" / "xray" / "valid_clean.csv"

    image_root = Path(str(cfg_stage2.data.image_root))

    manifest, labels = _make_validation_manifest(train_csv, valid_csv, image_root=image_root)

    # -- reproducibility ------------------------------------------------------
    device = get_device("auto")
    seed_everything(cfg_stage2.get("seed", 42), deterministic=cfg_stage2.get("deterministic", True))

    # -- loaders + pos_weight -------------------------------------------------
    loaders = build_dataloaders(manifest, cfg_stage2, splits=("train", "val"))
    pos_weight = compute_pos_weights(manifest, labels, split="train")
    criterion = build_loss(cfg_stage2, pos_weight=pos_weight, device=device)

    # -------------------------------------------------------------------------
    # Stage 1: head-only training with frozen backbone
    # -------------------------------------------------------------------------
    stage1_start = time()
    stage1_model = build_model(cfg_stage1, num_labels=len(labels))
    stage1_criterion = build_loss(cfg_stage1, pos_weight=pos_weight, device=device)

    stage1_result = train_model(
        stage1_model,
        loaders,
        stage1_criterion,
        cfg_stage1,
        device,
        run_name=stage1_run_name,
    )
    print(f"[stage1] finished in {(time() - stage1_start) / 60:.1f} min")

    # -------------------------------------------------------------------------
    # Stage 2: conservative fine-tuning (only last EfficientNet blocks)
    # -------------------------------------------------------------------------
    stage2_start = time()
    stage2_model = build_model(cfg_stage2, num_labels=len(labels))
    stage2_model = unfreeze_efficientnet_backbone_blocks(
        stage2_model,
        unfreeze_last_n_blocks=int(cfg_stage2.get("model.efficientnet_unfreeze_last_blocks", 2)),
    )
    stage2_model = load_checkpoint(stage1_result["best_checkpoint"], stage2_model, device)

    stage2_result = train_model(
        stage2_model,
        loaders,
        criterion,
        cfg_stage2,
        device,
        run_name=stage2_run_name,
    )
    print(f"[stage2] finished in {(time() - stage2_start) / 60:.1f} min")

    # -------------------------------------------------------------------------
    # Validation-only evaluation + artifacts
    # -------------------------------------------------------------------------
    _ = evaluate_classifier(
        stage2_result["model"],
        loaders,
        cfg_stage2,
        device,
        run_name=stage2_run_name,
    )

    # -------------------------------------------------------------------------
    # Update model_comparison.json (preserve baseline & DenseNet exactly)
    # -------------------------------------------------------------------------
    comparison_path = output_root / "model_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    eff_metrics = _extract_val_metrics(output_root, stage2_run_name)
    efficientnet_entry = {
        "run_name": stage2_run_name,
        "stage1_run_name": stage1_run_name,
        "val_macro_metrics": eff_metrics["val_macro_metrics"],
        "val_metrics": eff_metrics["val_metrics"],
    }

    # Primary criterion: validation macro ROC-AUC.
    macro_roc = {
        "baseline_cnn": float(_to_optional_float(comparison["baseline_cnn"]["val_macro_metrics"]["roc_auc"]) or -np.inf),
        "fine_tuned_densenet": float(_to_optional_float(comparison["fine_tuned_densenet"]["val_macro_metrics"]["roc_auc"]) or -np.inf),
        "fine_tuned_efficientnet_b0": float(eff_metrics["val_macro_metrics"]["roc_auc"] or -np.inf),
    }
    sorted_keys = sorted(macro_roc, key=macro_roc.get, reverse=True)
    winner_key = sorted_keys[0]

    display_name = {
        "baseline_cnn": "Baseline CNN (from scratch)",
        "fine_tuned_densenet": "Fine-tuned DenseNet",
        "fine_tuned_efficientnet_b0": "Fine-tuned EfficientNet-B0",
    }[winner_key]

    # Secondary context (imbalance-aware): PR-AUC and F1.
    runner_key = sorted_keys[1]
    if winner_key == "baseline_cnn":
        winner_macro = comparison["baseline_cnn"]["val_macro_metrics"]
        winner_pr_auc = winner_macro.get("pr_auc")
        winner_f1 = winner_macro.get("f1")
    elif winner_key == "fine_tuned_densenet":
        winner_macro = comparison["fine_tuned_densenet"]["val_macro_metrics"]
        winner_pr_auc = winner_macro.get("pr_auc")
        winner_f1 = winner_macro.get("f1")
    else:
        winner_macro = eff_metrics["val_macro_metrics"]
        winner_pr_auc = winner_macro.get("pr_auc")
        winner_f1 = winner_macro.get("f1")

    if runner_key == "baseline_cnn":
        runner_macro = comparison["baseline_cnn"]["val_macro_metrics"]
    elif runner_key == "fine_tuned_densenet":
        runner_macro = comparison["fine_tuned_densenet"]["val_macro_metrics"]
    else:
        runner_macro = eff_metrics["val_macro_metrics"]

    selection_rationale = (
        f"{display_name} has the highest validation macro ROC-AUC "
        f"({float(winner_macro['roc_auc']):.6f}). "
        f"As secondary context, macro PR-AUC is {float(winner_pr_auc):.6f} "
        f"and macro F1 is {float(winner_f1):.6f} "
        f"(runner-up PR-AUC {float(runner_macro['pr_auc']):.6f}, F1 {float(runner_macro['f1']):.6f})."
    )

    comparison["fine_tuned_efficientnet_b0"] = efficientnet_entry
    comparison["selected_model"] = display_name
    comparison["selection_rationale"] = selection_rationale
    comparison["protected_official_test_statement"] = (
        "The protected official test set was not used for model selection, tuning, threshold selection, or validation metrics."
    )

    # Also include the explicit statement under the existing structure.
    comparison.setdefault("official_test_not_used", {})
    comparison["official_test_not_used"]["explicit_statement"] = comparison["protected_official_test_statement"]

    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Report (validation-only)
    # -------------------------------------------------------------------------
    macro = eff_metrics["val_macro_metrics"]
    print("\n=== EfficientNet-B0 validation-only macro metrics ===")
    print(f"macro ROC-AUC: {macro['roc_auc']}")
    print(f"macro PR-AUC: {macro['pr_auc']}")
    # Precision/recall/sensitivity/macro F1 live in the metrics CSV macro row.
    print(f"macro precision (PPV): {macro['precision']}")
    print(f"macro recall/sensitivity: {macro['recall_sensitivity']}")
    print(f"macro F1: {macro['f1']}")
    print("Selection winner:", display_name)


if __name__ == "__main__":
    main()

