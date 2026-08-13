"""Baseline CNN (from scratch, validation-only) for chest X-ray multilabel classification.

Reproducibility script for the reported run
``xray_baseline_cnn_from_scratch_multilabel_320``.

The original script that produced that run was not preserved in the repository or
in git history. This file rebuilds the run from the artifact that *was* preserved:
the run's own ``config_used.yaml`` snapshot, which records every hyper-parameter
(architecture, image size, loss, class weights, optimizer, LR, scheduler, seed,
early-stopping metric, threshold strategy). Only the machine-specific absolute
paths are rewritten to this checkout.

It trains a single stage - the baseline has no frozen/unfrozen schedule - and then
runs validation-only evaluation, writing thresholds, per-label validation metrics,
threshold sweeps and ``predictions/validation_predictions.csv``.

Not claimed: bit-for-bit reproduction of the saved checkpoint. Ordering inside
cuDNN kernels, driver version and library versions all shift results slightly, and
this script has not been executed against the saved weights to prove equality.
What is claimed and checkable: the configuration below is the configuration the
saved run recorded.

Usage
-----
    python scripts/run_xray_baseline_cnn.py            # refuses to clobber artifacts
    python scripts/run_xray_baseline_cnn.py --force    # allow overwriting the run dir

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
    train_model,
)

RUN_NAME = "xray_baseline_cnn_from_scratch_multilabel_320"


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


def guard_existing_artifacts(run_name: str, output_root: Path, force: bool) -> None:
    """Refuse to overwrite a saved run unless explicitly forced."""
    existing = output_root / run_name / "models"
    if not force and existing.is_dir() and any(existing.glob("*.pt")):
        raise SystemExit(
            f"Refusing to overwrite existing artifacts in {output_root / run_name}.\n"
            "  The reported results live here. Re-run with --force only if you intend "
            "to replace them."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting the existing saved run directory")
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "classification"
    guard_existing_artifacts(RUN_NAME, output_root, args.force)

    cfg = load_saved_run_config(RUN_NAME, output_root)

    train_csv = PROJECT_ROOT / "data" / "processed" / "xray" / "train_clean.csv"
    valid_csv = PROJECT_ROOT / "data" / "processed" / "xray" / "valid_clean.csv"
    manifest, labels = make_manifest(train_csv, valid_csv, Path(str(cfg.data.image_root)))

    if list(labels) != list(cfg.data.labels):
        raise RuntimeError(
            "Label order in train_clean.csv does not match the saved run config:\n"
            f"  csv:    {labels}\n  config: {list(cfg.data.labels)}"
        )

    device = get_device(cfg.get("device", "auto"))
    seed_everything(cfg.get("seed", 42), deterministic=cfg.get("deterministic", True))

    # Only train/val: no test loader is built, so the official test set is untouched.
    loaders = build_dataloaders(manifest, cfg, splits=("train", "val"))
    pos_weight = compute_pos_weights(manifest, labels, split="train")
    criterion = build_loss(cfg, pos_weight=pos_weight, device=device)

    # model.name == 'simple_cnn' -> SimpleCNN, randomly initialised. The saved run
    # also records model.pretrained: true; SimpleCNN ignores it for weights, but
    # build_transforms reads it and applies ImageNet mean/std normalization. Kept
    # as-is so the input pipeline matches the reported run (and the other two
    # models) exactly.
    model = build_model(cfg, num_labels=len(labels))

    result = train_model(model, loaders, criterion, cfg, device, run_name=RUN_NAME)

    # Validation-only: thresholds, per-label metrics, threshold sweeps and
    # predictions/validation_predictions.csv. Stops before any test stage.
    evaluate_classifier(result["model"], loaders, cfg, device, run_name=RUN_NAME)

    print(f"\n[done] {RUN_NAME}: best epoch {result['best_epoch']} "
          f"({cfg.get('train.monitor')}={result['best_score']})")


if __name__ == "__main__":
    main()
