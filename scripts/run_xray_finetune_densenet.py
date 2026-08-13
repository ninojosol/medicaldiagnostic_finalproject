"""Fine-tuned DenseNet-121 (validation-only) for chest X-ray multilabel classification.

Reproducibility script for the reported runs
``xray_finetuned_densenet_multilabel_320_stage1_head_only`` (stage 1) and
``xray_finetuned_densenet_multilabel_320`` (stage 2).

The original script was not preserved in the repository or in git history. This
file rebuilds both stages from the artifacts that *were* preserved: each run's own
``config_used.yaml`` snapshot. Those snapshots record the architecture, image size,
loss, class weights, optimizer, LR, scheduler, seed, early-stopping metric and
threshold strategy; only machine-specific absolute paths are rewritten here.

Two-stage schedule, mirroring ``run_xray_finetune_efficientnet_b0.py``:

* stage 1 - ``model.freeze_backbone: true``: train only the new 14-output head.
* stage 2 - ``model.freeze_backbone: false``: initialise from the stage-1 best
  checkpoint and fine-tune the whole network at the recorded LR.

The stage-1 -> stage-2 hand-off is inferred from ``model_comparison.json``, which
records ``stage1_run_name`` for this model, and from the two configs'
``freeze_backbone`` values. The saved artifacts do not record the hand-off
mechanism itself, so this is a faithful reconstruction of the documented schedule,
not a byte-verified replay.

Not claimed: bit-for-bit reproduction of the saved checkpoints. This script has not
been executed against the saved weights to prove equality.

Usage
-----
    python scripts/run_xray_finetune_densenet.py            # refuses to clobber artifacts
    python scripts/run_xray_finetune_densenet.py --force    # allow overwriting run dirs

The protected official test set is never loaded: only ``train_clean.csv`` and
``valid_clean.csv`` are read, and no ``test`` loader is built, so
``evaluate_classifier`` stops after its validation stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_device, load_config, seed_everything  # noqa: E402
from src.classification import (  # noqa: E402
    build_dataloaders,
    build_loss,
    build_model,
    compute_pos_weights,
    evaluate_classifier,
    load_checkpoint,
    train_model,
)

STAGE1_RUN_NAME = "xray_finetuned_densenet_multilabel_320_stage1_head_only"
STAGE2_RUN_NAME = "xray_finetuned_densenet_multilabel_320"


def make_manifest(train_csv: Path, valid_csv: Path,
                  image_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build the train/val manifest the loaders expect from the two clean CSVs.

    Label order is taken from the CSV column order (Image and PatientId excluded),
    which is the frozen NIH-14 order shared by every artifact in this project.
    """
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    labels = [c for c in train_df.columns if c not in {"Image", "PatientId"}]
    if not labels:
        raise RuntimeError(f"No label columns detected in {train_csv}")

    def to_manifest(df: pd.DataFrame, split: str) -> pd.DataFrame:
        out = pd.DataFrame()
        out["image_path"] = df["Image"].astype(str).map(lambda name: str(image_root / name)).values
        out["patient_id"] = df["PatientId"].astype(str).values
        for lab in labels:
            out[lab] = df[lab].astype(np.float32).values
        out["split"] = split
        return out

    manifest = pd.concat(
        [to_manifest(train_df, "train"), to_manifest(valid_df, "val")], ignore_index=True
    )
    return manifest, labels


def load_saved_run_config(run_name: str, output_root: Path):
    """Load a run's own config snapshot and re-point absolute paths at this checkout."""
    snapshot = output_root / run_name / "config_used.yaml"
    if not snapshot.is_file():
        raise FileNotFoundError(
            f"Missing config snapshot for '{run_name}': {snapshot}\n"
            "  This script reproduces a recorded run and needs its saved configuration."
        )
    nih_root = PROJECT_ROOT / "data" / "raw" / "xray" / "nih"
    return load_config(
        snapshot,
        overrides={
            "run_name": run_name,
            "data": {
                "root": str(nih_root),
                "image_root": str(nih_root / "images-small"),
                "csv_path": str(PROJECT_ROOT / "data" / "processed" / "xray" / "train_clean.csv"),
            },
            "output": {"root": str(output_root)},
        },
    )


def guard_existing_artifacts(run_names: list[str], output_root: Path, force: bool) -> None:
    """Refuse to overwrite saved runs unless explicitly forced."""
    clashes = [
        name for name in run_names
        if (output_root / name / "models").is_dir()
        and any((output_root / name / "models").glob("*.pt"))
    ]
    if clashes and not force:
        raise SystemExit(
            f"Refusing to overwrite existing artifacts for: {', '.join(clashes)}.\n"
            "  The reported results live there. Re-run with --force only if you intend "
            "to replace them."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting the existing saved run directories")
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "classification"
    guard_existing_artifacts([STAGE1_RUN_NAME, STAGE2_RUN_NAME], output_root, args.force)

    cfg_stage1 = load_saved_run_config(STAGE1_RUN_NAME, output_root)
    cfg_stage2 = load_saved_run_config(STAGE2_RUN_NAME, output_root)

    train_csv = PROJECT_ROOT / "data" / "processed" / "xray" / "train_clean.csv"
    valid_csv = PROJECT_ROOT / "data" / "processed" / "xray" / "valid_clean.csv"
    manifest, labels = make_manifest(train_csv, valid_csv, Path(str(cfg_stage2.data.image_root)))

    if list(labels) != list(cfg_stage2.data.labels):
        raise RuntimeError(
            "Label order in train_clean.csv does not match the saved run config:\n"
            f"  csv:    {labels}\n  config: {list(cfg_stage2.data.labels)}"
        )

    device = get_device(cfg_stage2.get("device", "auto"))
    seed_everything(cfg_stage2.get("seed", 42), deterministic=cfg_stage2.get("deterministic", True))

    # Only train/val: no test loader is built, so the official test set is untouched.
    # Both stages share the same loaders, so both see the identical split.
    loaders = build_dataloaders(manifest, cfg_stage2, splits=("train", "val"))
    pos_weight = compute_pos_weights(manifest, labels, split="train")

    # ---- stage 1: head only, backbone frozen -------------------------------
    stage1_model = build_model(cfg_stage1, num_labels=len(labels))
    stage1_criterion = build_loss(cfg_stage1, pos_weight=pos_weight, device=device)
    stage1_result = train_model(
        stage1_model, loaders, stage1_criterion, cfg_stage1, device, run_name=STAGE1_RUN_NAME
    )

    # ---- stage 2: full fine-tune, initialised from the stage-1 best --------
    stage2_model = build_model(cfg_stage2, num_labels=len(labels))
    stage2_model = load_checkpoint(stage1_result["best_checkpoint"], stage2_model, device)
    stage2_criterion = build_loss(cfg_stage2, pos_weight=pos_weight, device=device)
    stage2_result = train_model(
        stage2_model, loaders, stage2_criterion, cfg_stage2, device, run_name=STAGE2_RUN_NAME
    )

    # Validation-only: thresholds, per-label metrics, threshold sweeps and
    # predictions/validation_predictions.csv. Stops before any test stage.
    evaluate_classifier(stage2_result["model"], loaders, cfg_stage2, device,
                        run_name=STAGE2_RUN_NAME)

    print(f"\n[done] {STAGE2_RUN_NAME}: best epoch {stage2_result['best_epoch']} "
          f"({cfg_stage2.get('train.monitor')}={stage2_result['best_score']})")


if __name__ == "__main__":
    main()
