"""Pretrained ViT-B/16 (validation-only) for chest X-ray multilabel classification.

EXPERIMENTAL COMPARATOR - not the selected model
------------------------------------------------
This is a *fourth* comparator added alongside the three frozen runs (Baseline CNN,
fine-tuned DenseNet-121, fine-tuned EfficientNet-B0). Adding it does not change the
project's selected model.

``outputs/classification/model_comparison.json`` is the FROZEN record of the original
three-model comparison. This script treats it as **read-only input** and never writes,
reformats, renames or otherwise touches it. Its SHA-256 is captured before the read and
re-checked after every write this script performs; a mismatch aborts with an error.

All ViT results go into a separate file instead:

    outputs/classification/model_comparison_with_vit_experimental.json

That file copies the frozen record verbatim (including its original ``selected_model``
and ``selection_rationale``, preserved as historical metadata), adds the
``fine_tuned_vit_b16`` entry, and records the ViT-vs-EfficientNet-B0 outcome under a
separate ``experimental_selection`` field. The original three-model selection is never
replaced anywhere. Macro ROC-AUC on validation is the primary criterion; macro PR-AUC
is supporting context only.

Runs
----
* ``xray_finetuned_vit_b16_multilabel_224_stage1_head_only`` - the ImageNet backbone
  is entirely frozen and only the new 14-output head is trained (a linear probe).
* ``xray_finetuned_vit_b16_multilabel_224`` - initialised from the stage-1 best
  checkpoint, then the last ``model.vit_unfreeze_last_blocks`` encoder blocks plus the
  final encoder LayerNorm are re-opened and fine-tuned at a ViT-appropriate LR.

This mirrors the two-stage schedule of ``run_xray_finetune_efficientnet_b0.py``.
Unlike ``scripts/run_xray_finetune_densenet.py``, this script is not replaying a saved
run: it reads ``configs/xray_vit.yaml``, which is the authoritative input until the run
produces its own ``config_used.yaml`` snapshot.

Protected official test set
---------------------------
Only ``train_clean.csv`` and ``valid_clean.csv`` are read. No ``test`` loader is built,
so ``evaluate_classifier`` stops after its validation stage. Model selection, early
stopping and per-finding threshold selection are all validation-only, which makes the
resulting F1 / sensitivity / specificity *operating-point estimates on the same split
the thresholds were chosen on*, not held-out performance.

Usage
-----
    python scripts/run_xray_finetune_vit.py --smoke-test   # build/shape/checkpoint/artifact checks
    python scripts/run_xray_finetune_vit.py                # train; refuses to clobber artifacts
    python scripts/run_xray_finetune_vit.py --force        # allow overwriting the ViT run dirs
    python scripts/run_xray_finetune_vit.py --no-comparison-update

``--force`` only ever applies to the two ViT run directories named above. This script
has no code path that writes into the Baseline CNN, DenseNet or EfficientNet-B0 run
directories, or into the frozen ``model_comparison.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch

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
from src.classification.models import (  # noqa: E402
    build_transfer_model,
    count_parameters,
    unfreeze_vit_encoder_blocks,
)

CONFIG_NAME = "xray_vit.yaml"
STAGE1_RUN_NAME = "xray_finetuned_vit_b16_multilabel_224_stage1_head_only"
STAGE2_RUN_NAME = "xray_finetuned_vit_b16_multilabel_224"

COMPARISON_KEY = "fine_tuned_vit_b16"
COMPARISON_DISPLAY_NAME = "Fine-tuned ViT-B/16"
# Entries this script must never modify. Verified identical after any write.
PROTECTED_COMPARISON_KEYS = ("baseline_cnn", "fine_tuned_densenet", "fine_tuned_efficientnet_b0")
# The current selection recorded in the frozen file; the ViT is measured against it.
INCUMBENT_KEY = "fine_tuned_efficientnet_b0"
INCUMBENT_DISPLAY_NAME = "Fine-tuned EfficientNet-B0"

# Read-only input: the frozen three-model record. Never written by this script.
FROZEN_COMPARISON_FILENAME = "model_comparison.json"
# The only comparison artifact this script creates or maintains.
EXPERIMENTAL_COMPARISON_FILENAME = "model_comparison_with_vit_experimental.json"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, used to prove the frozen record was not modified."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative_to_project(path: Path) -> str:
    """Repo-relative posix path, so the artifact is portable across checkouts."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def make_manifest(train_csv: Path, valid_csv: Path,
                  image_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build the train/val manifest the loaders expect from the two clean CSVs.

    Identical to the manifest construction in the DenseNet and EfficientNet-B0
    scripts, so the ViT run sees exactly the same images and the same split.
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


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_stage_configs(output_root: Path):
    """Load configs/xray_vit.yaml and derive the two stage configs.

    Absolute paths are written in so the snapshots record the same shape as the
    saved runs' ``config_used.yaml`` files.
    """
    nih_root = PROJECT_ROOT / "data" / "raw" / "xray" / "nih"
    path_overrides = {
        "data": {
            "root": str(nih_root),
            "image_root": str(nih_root / "images-small"),
            "csv_path": str(PROJECT_ROOT / "data" / "processed" / "xray" / "train_clean.csv"),
        },
        "output": {"root": str(output_root)},
    }

    base = load_config(CONFIG_NAME, overrides=path_overrides)

    stage1_block = base.get("vit_stage1_overrides", {})
    stage1_train = stage1_block.to_dict() if hasattr(stage1_block, "to_dict") else dict(stage1_block or {})

    cfg_stage1 = load_config(
        CONFIG_NAME,
        overrides={
            **path_overrides,
            "run_name": STAGE1_RUN_NAME,
            # Stage 1: the entire backbone stays frozen; only the new head trains.
            "model": {"freeze_backbone": True, "vit_unfreeze_last_blocks": 0},
            "train": stage1_train,
        },
    )
    cfg_stage2 = load_config(
        CONFIG_NAME,
        overrides={
            **path_overrides,
            "run_name": STAGE2_RUN_NAME,
            # Stage 2: built frozen, then the last N encoder blocks are re-opened
            # in code (same mechanism as the saved EfficientNet-B0 run).
            "model": {"freeze_backbone": True},
        },
    )
    return cfg_stage1, cfg_stage2


def guard_existing_artifacts(run_names: list[str], output_root: Path, force: bool) -> None:
    """Refuse to overwrite an existing ViT run directory unless --force is passed."""
    clashes = [
        name for name in run_names
        if (output_root / name).is_dir() and any((output_root / name).rglob("*"))
    ]
    if clashes and not force:
        raise SystemExit(
            f"Refusing to overwrite existing run director{'y' if len(clashes) == 1 else 'ies'}: "
            f"{', '.join(clashes)}.\n"
            "  Re-run with --force only if you intend to replace those artifacts."
        )


# ---------------------------------------------------------------------------
# smoke test (no data, no training, no artifacts written)
# ---------------------------------------------------------------------------
def run_smoke_test() -> None:
    """Config / build / shape / checkpoint / artifact-safety checks.

    Writes nothing under ``outputs/``: the comparison writer is exercised against a
    throwaway sandbox seeded with a copy of the frozen record. ``pretrained=False``
    here so the check never triggers the ~330 MB ImageNet download; the architecture,
    head shape and state-dict round-trip are identical either way.
    """
    output_root = PROJECT_ROOT / "outputs" / "classification"
    frozen_path = output_root / FROZEN_COMPARISON_FILENAME
    frozen_hash_at_start = sha256_of(frozen_path)
    experimental_existed = (output_root / EXPERIMENTAL_COMPARISON_FILENAME).is_file()
    print(f"[smoke] frozen {FROZEN_COMPARISON_FILENAME} SHA-256 before: {frozen_hash_at_start}")

    print("[smoke] loading", CONFIG_NAME)
    cfg_stage1, cfg_stage2 = load_stage_configs(PROJECT_ROOT / "outputs" / "classification")
    labels = list(cfg_stage2.data.labels)
    image_size = int(cfg_stage2.get("data.image_size", 224))
    print(f"[smoke] labels={len(labels)} image_size={image_size} "
          f"model={cfg_stage2.get('model.name')}")
    assert len(labels) == 14, f"expected 14 NIH labels, got {len(labels)}"
    assert image_size == 224, "vit_b_16 requires data.image_size: 224"
    assert float(cfg_stage2.get("augmentation.hflip", 0.0)) == 0.0, "hflip must stay disabled"
    assert cfg_stage1.get("train.epochs") == cfg_stage1.get("vit_stage1_overrides.epochs"), \
        "stage-1 overrides were not applied to train:"

    print("[smoke] building vit_b_16 (pretrained=False, no download) ...")
    model = build_transfer_model(
        name=str(cfg_stage2.get("model.name", "vit_b_16")),
        num_labels=len(labels),
        pretrained=False,
        freeze_backbone=True,
        dropout=float(cfg_stage2.get("model.dropout", 0.2)),
    )
    total, trainable = count_parameters(model)
    print(f"[smoke] head-only stage: {trainable:,}/{total:,} trainable parameters")

    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, image_size, image_size))
    print(f"[smoke] forward output shape = {tuple(logits.shape)}")
    assert tuple(logits.shape) == (2, len(labels)), \
        f"expected [2, {len(labels)}] logits, got {tuple(logits.shape)}"
    probs = torch.sigmoid(logits)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0

    print("[smoke] unfreezing last "
          f"{int(cfg_stage2.get('model.vit_unfreeze_last_blocks', 2))} encoder block(s) ...")
    unfreeze_vit_encoder_blocks(
        model, unfreeze_last_n_blocks=int(cfg_stage2.get("model.vit_unfreeze_last_blocks", 2))
    )

    # Checkpoint round-trip in the exact format train_model writes / load_checkpoint reads.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "vit_smoke.pt"
        torch.save({"model_state": model.state_dict(), "epoch": 0, "labels": labels}, ckpt)
        fresh = build_transfer_model(name=str(cfg_stage2.get("model.name", "vit_b_16")),
                                     num_labels=len(labels), pretrained=False,
                                     freeze_backbone=True,
                                     dropout=float(cfg_stage2.get("model.dropout", 0.2)))
        fresh = load_checkpoint(ckpt, fresh, torch.device("cpu"))
        with torch.no_grad():
            reloaded = fresh(torch.zeros(2, 3, image_size, image_size))
    assert torch.allclose(logits, reloaded, atol=1e-6), "checkpoint round-trip changed the outputs"
    print("[smoke] checkpoint save/load round-trip reproduces identical logits")

    _smoke_check_comparison_writer(cfg_stage2)
    _smoke_check_no_test_data()

    # The frozen record must be untouched by everything above.
    frozen_hash_at_end = sha256_of(frozen_path)
    print(f"[smoke] frozen {FROZEN_COMPARISON_FILENAME} SHA-256 after:  {frozen_hash_at_end}")
    assert frozen_hash_at_end == frozen_hash_at_start, \
        f"{FROZEN_COMPARISON_FILENAME} was modified by the smoke test"
    assert (output_root / EXPERIMENTAL_COMPARISON_FILENAME).is_file() == experimental_existed, \
        "the smoke test must not create or remove the experimental comparison file"
    print("[smoke] frozen comparison record unchanged; no comparison artifact written "
          "under outputs/")
    print("[smoke] OK - nothing was written under outputs/, no test data was read.")


def _synthetic_validation_metrics(labels: list[str], macro_roc_auc: float) -> pd.DataFrame:
    """A metrics_validation.csv shaped like the real one, for sandbox testing only."""
    columns = ["label", "threshold", "n", "n_positive", "n_negative", "roc_auc", "pr_auc",
               "accuracy", "precision", "recall_sensitivity", "specificity", "f1",
               "balanced_accuracy", "tp", "fp", "tn", "fn", "prevalence"]
    rows = []
    for label in labels:
        rows.append(dict(zip(columns, [label, 0.5, 207, 10, 197, macro_roc_auc, 0.1, 0.9, 0.1,
                                       0.5, 0.9, 0.17, 0.7, 5, 20, 177, 5, 0.048])))
    rows.append(dict(zip(columns, ["MACRO_AVG", 0.5, 2898, 129, 2769, macro_roc_auc, 0.1, 0.9,
                                   0.1, 0.5, 0.9, 0.17, 0.7, 65, 280, 2489, 64, 0.0445])))
    return pd.DataFrame(rows, columns=columns)


def _smoke_check_comparison_writer(cfg_stage2) -> None:
    """Exercise the writer in a sandbox: only the experimental file may be created.

    The sandbox is seeded with a copy of the real frozen record, so this also proves
    the writer leaves ``model_comparison.json`` byte-identical and copies the three
    protected entries through unchanged.
    """
    import shutil
    import tempfile

    labels = list(cfg_stage2.data.labels)
    real_frozen = PROJECT_ROOT / "outputs" / "classification" / FROZEN_COMPARISON_FILENAME

    for macro_roc, expected_beats in ((0.100000, False), (0.999999, True)):
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            shutil.copy2(real_frozen, sandbox / FROZEN_COMPARISON_FILENAME)
            sandbox_frozen_before = sha256_of(sandbox / FROZEN_COMPARISON_FILENAME)

            metrics_dir = sandbox / STAGE2_RUN_NAME / "metrics"
            metrics_dir.mkdir(parents=True)
            _synthetic_validation_metrics(labels, macro_roc).to_csv(
                metrics_dir / "metrics_validation.csv", index=False)

            outcome = write_experimental_comparison(
                sandbox, cfg_stage2, STAGE1_RUN_NAME, STAGE2_RUN_NAME)

            # Only one comparison artifact may exist besides the seeded frozen copy.
            written = sorted(p.name for p in sandbox.glob("*.json"))
            assert written == sorted([FROZEN_COMPARISON_FILENAME,
                                      EXPERIMENTAL_COMPARISON_FILENAME]), \
                f"unexpected comparison artifacts in sandbox: {written}"
            assert sha256_of(sandbox / FROZEN_COMPARISON_FILENAME) == sandbox_frozen_before, \
                "the writer modified the frozen record"

            frozen = json.loads((sandbox / FROZEN_COMPARISON_FILENAME).read_text(encoding="utf-8"))
            new = json.loads((sandbox / EXPERIMENTAL_COMPARISON_FILENAME).read_text(encoding="utf-8"))
            for key in PROTECTED_COMPARISON_KEYS:
                assert new[key] == frozen[key], f"'{key}' was not copied verbatim"
            assert new["selected_model"] == frozen["selected_model"], \
                "the original three-model selection was not preserved"
            assert new["frozen_three_model_selection"]["selected_model"] == frozen["selected_model"]
            assert new["selection_rationale"] == frozen["selection_rationale"], \
                "the original selection rationale was not preserved"

            vit = new[COMPARISON_KEY]
            assert vit["role"] == "experimental_comparator"
            assert vit["status"] == "experimental_comparator"
            assert vit["architecture"] == "vit_b_16"
            assert vit["input_size"] == 224
            assert vit["pretrained"] is True
            assert vit["preprocessing"] and vit["thresholds_source"] and vit["run_paths"]
            assert vit["data_source"]["official_test_loaded"] is False

            verdict = new["experimental_selection"]
            assert verdict["vit_beats_efficientnet_b0_on_macro_roc_auc"] is expected_beats
            assert outcome["beats_incumbent"] is expected_beats
            assert outcome["created"] is True

            # Re-running must refresh in place, not spawn a second artifact.
            write_experimental_comparison(sandbox, cfg_stage2, STAGE1_RUN_NAME, STAGE2_RUN_NAME)
            assert sorted(p.name for p in sandbox.glob("*.json")) == written
            assert sha256_of(sandbox / FROZEN_COMPARISON_FILENAME) == sandbox_frozen_before

            print(f"[smoke] sandbox macro ROC-AUC {macro_roc:.6f} -> beats EfficientNet-B0 = "
                  f"{expected_beats}; frozen copy untouched; only "
                  f"{EXPERIMENTAL_COMPARISON_FILENAME} written")


def _smoke_check_no_test_data() -> None:
    """Static check that this script has no path to the protected official test set.

    The forbidden tokens are assembled from fragments on purpose: if they appeared
    as plain literals, this function's own source would trip the scan it performs.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]  # skip the module docstring, which discusses the test set
    forbidden = (
        "test" + ".csv",                        # reading the official test manifest
        "loaders[" + '"test"' + "]",            # indexing a test loader
        'splits=("train", "val", ' + '"test")',  # asking build_dataloaders for a test split
    )
    for token in forbidden:
        assert token not in body, f"script references the protected test set: {token!r}"
    assert 'splits=("train", "val")' in body, "loaders must be built for train/val only"
    assert 'if "test" in loaders' in body, "the defensive test-loader guard is missing"
    print("[smoke] no test manifest read, no test loader indexed, loaders restricted to "
          "train/val; defensive guard present")


# ---------------------------------------------------------------------------
# model_comparison.json
# ---------------------------------------------------------------------------
def _to_optional_float(x):
    """Keep JSON valid and make NaN explicitly 'not computable' (as in the EffNet script)."""
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def _row_to_dict(row: pd.Series) -> dict:
    out = {}
    for k, v in row.to_dict().items():
        if k in {"roc_auc", "pr_auc"}:
            out[k] = _to_optional_float(v)
        else:
            if isinstance(v, np.integer):
                out[k] = int(v)
            elif isinstance(v, np.floating):
                out[k] = float(v)
            else:
                out[k] = v
    return out


def _extract_val_metrics(output_root: Path, run_name: str) -> dict:
    metrics_path = output_root / run_name / "metrics" / "metrics_validation.csv"
    df = pd.read_csv(metrics_path)
    macro_row = df[df["label"] == "MACRO_AVG"].iloc[0]
    per_label_rows = df[df["label"] != "MACRO_AVG"].copy()
    return {
        "val_macro_metrics": _row_to_dict(macro_row),
        "val_metrics": {"per_label": [_row_to_dict(r) for _, r in per_label_rows.iterrows()]},
    }


def _vit_entry(output_root: Path, cfg_stage2, stage1_run: str, stage2_run: str) -> dict:
    """The ``fine_tuned_vit_b16`` entry: metrics, provenance and preprocessing."""
    vit_metrics = _extract_val_metrics(output_root, stage2_run)
    stage2_dir = output_root / stage2_run
    stage1_dir = output_root / stage1_run

    preprocessing = cfg_stage2.get("model.input_preprocessing", {})
    preprocessing = (preprocessing.to_dict() if hasattr(preprocessing, "to_dict")
                     else dict(preprocessing or {}))

    return {
        "run_name": stage2_run,
        "stage1_run_name": stage1_run,
        "role": "experimental_comparator",
        "status": "experimental_comparator",
        "architecture": "vit_b_16",
        "input_size": 224,
        "pretrained": True,
        "pretrained_source": str(cfg_stage2.get("model.pretrained_source", "")),
        "pretrained_weights": "torchvision ViT_B_16_Weights.IMAGENET1K_V1 (ImageNet-1k)",
        "trainable_scope": {
            "stage1": "classifier head only; entire ViT backbone frozen (linear probe)",
            "stage2": (
                f"last {int(cfg_stage2.get('model.vit_unfreeze_last_blocks', 2))} encoder "
                "block(s) + final encoder LayerNorm + classifier head; patch embedding, "
                "class token, positional embeddings and earlier blocks stay frozen"
            ),
        },
        "run_paths": {
            "stage1_run_dir": _relative_to_project(stage1_dir),
            "stage2_run_dir": _relative_to_project(stage2_dir),
            "checkpoint": _relative_to_project(
                stage2_dir / "models" / f"{stage2_run}_best.pt"),
            "stage1_checkpoint": _relative_to_project(
                stage1_dir / "models" / f"{stage1_run}_best.pt"),
            "config_used": _relative_to_project(stage2_dir / "config_used.yaml"),
            "run_metadata": _relative_to_project(stage2_dir / "run_metadata.json"),
            "metrics_validation": _relative_to_project(
                stage2_dir / "metrics" / "metrics_validation.csv"),
            "validation_predictions": _relative_to_project(
                stage2_dir / "predictions" / "validation_predictions.csv"),
        },
        "thresholds_source": {
            "thresholds_json": _relative_to_project(
                stage2_dir / "metrics" / "thresholds.json"),
            "threshold_table": _relative_to_project(
                stage2_dir / "metrics" / "thresholds_from_validation.csv"),
            "strategy": str(cfg_stage2.get("eval.threshold_strategy", "f1")),
            "selected_on": "validation split only",
        },
        "preprocessing": preprocessing,
        "data_source": {
            "train_manifest": "data/processed/xray/train_clean.csv",
            "validation_manifest": "data/processed/xray/valid_clean.csv",
            "official_test_loaded": False,
        },
        "note": (
            "Experimental fourth comparator (pretrained ViT-B/16). Trained at 224 x 224 "
            "because ViT-B/16's positional embeddings are fixed to that resolution, while "
            "the three CNN comparators were trained at 320 x 320; this resolution "
            "difference is a confound when reading the table. Validation-only: thresholds, "
            "early stopping and model selection never touched the official test set, so "
            "F1 / sensitivity / specificity are operating-point estimates on the same "
            "split the thresholds were selected on."
        ),
        "val_macro_metrics": vit_metrics["val_macro_metrics"],
        "val_metrics": vit_metrics["val_metrics"],
    }


def write_experimental_comparison(output_root: Path, cfg_stage2,
                                  stage1_run: str, stage2_run: str) -> dict:
    """Create/refresh model_comparison_with_vit_experimental.json.

    The frozen ``model_comparison.json`` is opened read-only, hashed before and after,
    and never written. On first call the new file is seeded with a verbatim copy of the
    frozen record - including its original ``selected_model`` and ``selection_rationale``,
    which are additionally preserved under ``frozen_three_model_selection`` as historical
    metadata. The ViT entry and an ``experimental_selection`` verdict are then added.
    On later calls only the ViT entry and the verdict are refreshed.
    """
    frozen_path = output_root / FROZEN_COMPARISON_FILENAME
    experimental_path = output_root / EXPERIMENTAL_COMPARISON_FILENAME
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Missing frozen comparison record: {frozen_path}")

    frozen_hash_before = sha256_of(frozen_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_protected = {k: deepcopy(frozen.get(k)) for k in PROTECTED_COMPARISON_KEYS}

    created = not experimental_path.is_file()
    if created:
        # Verbatim copy of the frozen record: same three entries, same labels,
        # same preprocessing spec, same original selection.
        comparison = deepcopy(frozen)
        comparison["frozen_three_model_selection"] = {
            "selected_model": frozen.get("selected_model"),
            "selection_rationale": frozen.get("selection_rationale"),
            "note": (
                "Historical metadata: the original three-model selection, copied "
                "unchanged from the frozen record. The experimental ViT comparator does "
                "not replace it."
            ),
        }
        comparison["source_record"] = {
            "file": FROZEN_COMPARISON_FILENAME,
            "sha256_at_copy_time": frozen_hash_before,
            "access": "read-only; this file is a derived copy and the frozen record is never modified",
        }
    else:
        comparison = json.loads(experimental_path.read_text(encoding="utf-8"))

    comparison["experimental_comparator_note"] = (
        "Derived, experiment-only artifact. The frozen three-model record lives in "
        f"{FROZEN_COMPARISON_FILENAME} and is never modified. The ViT entry below is an "
        "experimental comparator and does not change the project's selected model."
    )
    comparison[COMPARISON_KEY] = _vit_entry(output_root, cfg_stage2, stage1_run, stage2_run)

    vit_macro = comparison[COMPARISON_KEY]["val_macro_metrics"]
    vit_roc = _to_optional_float(vit_macro.get("roc_auc"))
    vit_pr = _to_optional_float(vit_macro.get("pr_auc"))
    inc_macro = (frozen.get(INCUMBENT_KEY) or {}).get("val_macro_metrics") or {}
    inc_roc = _to_optional_float(inc_macro.get("roc_auc"))
    inc_pr = _to_optional_float(inc_macro.get("pr_auc"))

    beats_incumbent = vit_roc is not None and inc_roc is not None and vit_roc > inc_roc
    vit_txt = "unavailable" if vit_roc is None else f"{vit_roc:.6f}"
    inc_txt = "unavailable" if inc_roc is None else f"{inc_roc:.6f}"

    comparison["experimental_selection"] = {
        "primary_criterion": "validation macro ROC-AUC",
        "supporting_context": "validation macro PR-AUC",
        "candidate": COMPARISON_DISPLAY_NAME,
        "candidate_macro_roc_auc": vit_roc,
        "candidate_macro_pr_auc": vit_pr,
        "incumbent": INCUMBENT_DISPLAY_NAME,
        "incumbent_macro_roc_auc": inc_roc,
        "incumbent_macro_pr_auc": inc_pr,
        "vit_beats_efficientnet_b0_on_macro_roc_auc": beats_incumbent,
        "verdict": (
            f"{COMPARISON_DISPLAY_NAME} scored validation macro ROC-AUC {vit_txt} versus "
            f"{INCUMBENT_DISPLAY_NAME} at {inc_txt}, so it "
            + ("exceeds" if beats_incumbent else "does not exceed")
            + " the incumbent on the primary criterion."
        ),
        "effect_on_frozen_selection": (
            "None. The frozen three-model selection is unchanged and this file does not "
            "override it. Promoting the ViT would be a separate, explicit decision."
        ),
        "caveats": (
            "Validation-only comparison. Thresholds were selected on this same split, so "
            "F1 / sensitivity / specificity are operating-point estimates. ViT-B/16 ran at "
            "224 x 224 against CNNs at 320 x 320."
        ),
    }

    # The copied entries must still match the frozen record exactly.
    for key in PROTECTED_COMPARISON_KEYS:
        if comparison.get(key) != frozen_protected[key]:
            raise RuntimeError(
                f"Aborting: the copied '{key}' entry differs from the frozen record. "
                "Nothing was written."
            )

    experimental_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    frozen_hash_after = sha256_of(frozen_path)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError(
            f"Aborting: {FROZEN_COMPARISON_FILENAME} changed during this run "
            f"({frozen_hash_before} -> {frozen_hash_after}). It must never be modified."
        )

    return {
        "created": created,
        "path": experimental_path,
        "beats_incumbent": beats_incumbent,
        "vit_macro_roc_auc": vit_roc,
        "vit_macro_pr_auc": vit_pr,
        "incumbent": (INCUMBENT_DISPLAY_NAME, inc_roc),
        "frozen_sha256": frozen_hash_before,
        "frozen_selection_preserved": frozen.get("selected_model"),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting the two ViT run directories")
    parser.add_argument("--smoke-test", action="store_true",
                        help="config/build/shape/checkpoint checks only; no data, no training")
    parser.add_argument("--no-comparison-update", action="store_true",
                        help=f"skip writing {EXPERIMENTAL_COMPARISON_FILENAME} "
                             "(the frozen model_comparison.json is never written either way)")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    output_root = PROJECT_ROOT / "outputs" / "classification"
    guard_existing_artifacts([STAGE1_RUN_NAME, STAGE2_RUN_NAME], output_root, args.force)

    cfg_stage1, cfg_stage2 = load_stage_configs(output_root)

    train_csv = PROJECT_ROOT / "data" / "processed" / "xray" / "train_clean.csv"
    valid_csv = PROJECT_ROOT / "data" / "processed" / "xray" / "valid_clean.csv"
    manifest, labels = make_manifest(train_csv, valid_csv, Path(str(cfg_stage2.data.image_root)))

    if list(labels) != list(cfg_stage2.data.labels):
        raise RuntimeError(
            "Label order in train_clean.csv does not match configs/xray_vit.yaml:\n"
            f"  csv:    {labels}\n  config: {list(cfg_stage2.data.labels)}"
        )

    device = get_device(cfg_stage2.get("device", "auto"))
    seed_everything(cfg_stage2.get("seed", 42), deterministic=cfg_stage2.get("deterministic", True))

    # Only train/val: no test loader is built, so the official test set is untouched.
    # Both stages share the same loaders, so both see the identical split.
    loaders = build_dataloaders(manifest, cfg_stage2, splits=("train", "val"))
    if "test" in loaders:  # defensive: build_dataloaders was asked for train/val only
        raise RuntimeError("A test loader was built. Aborting to keep the test set protected.")
    pos_weight = compute_pos_weights(manifest, labels, split="train")

    # ---- stage 1: head only, backbone entirely frozen ----------------------
    stage1_start = time()
    stage1_model = build_model(cfg_stage1, num_labels=len(labels))
    stage1_criterion = build_loss(cfg_stage1, pos_weight=pos_weight, device=device)
    stage1_result = train_model(
        stage1_model, loaders, stage1_criterion, cfg_stage1, device, run_name=STAGE1_RUN_NAME
    )
    print(f"[stage1] finished in {(time() - stage1_start) / 60:.1f} min")

    # ---- stage 2: last N encoder blocks unfrozen, from the stage-1 best ----
    stage2_start = time()
    stage2_model = build_model(cfg_stage2, num_labels=len(labels))
    stage2_model = unfreeze_vit_encoder_blocks(
        stage2_model,
        unfreeze_last_n_blocks=int(cfg_stage2.get("model.vit_unfreeze_last_blocks", 2)),
    )
    stage2_model = load_checkpoint(stage1_result["best_checkpoint"], stage2_model, device)
    stage2_criterion = build_loss(cfg_stage2, pos_weight=pos_weight, device=device)
    stage2_result = train_model(
        stage2_model, loaders, stage2_criterion, cfg_stage2, device, run_name=STAGE2_RUN_NAME
    )
    print(f"[stage2] finished in {(time() - stage2_start) / 60:.1f} min")

    # ---- validation-only evaluation ---------------------------------------
    # Writes thresholds, per-finding metrics, threshold sweeps and
    # predictions/validation_predictions.csv. Stops before any test stage.
    evaluate_classifier(stage2_result["model"], loaders, cfg_stage2, device,
                        run_name=STAGE2_RUN_NAME)

    print(f"\n[done] {STAGE2_RUN_NAME}: best epoch {stage2_result['best_epoch']} "
          f"({cfg_stage2.get('train.monitor')}={stage2_result['best_score']})")

    if args.no_comparison_update:
        print(f"[compare] --no-comparison-update: {EXPERIMENTAL_COMPARISON_FILENAME} not written.")
        return

    outcome = write_experimental_comparison(
        output_root, cfg_stage2, STAGE1_RUN_NAME, STAGE2_RUN_NAME
    )
    print("\n=== ViT-B/16 validation-only macro metrics (experimental comparator) ===")
    print(f"macro ROC-AUC: {outcome['vit_macro_roc_auc']}")
    print(f"macro PR-AUC:  {outcome['vit_macro_pr_auc']}")
    inc_name, inc_roc = outcome["incumbent"]
    inc_txt = "unavailable" if inc_roc is None else f"{inc_roc:.6f}"
    verb = "EXCEEDS" if outcome["beats_incumbent"] else "does NOT exceed"
    print(f"ViT {verb} {inc_name} ({inc_txt}) on validation macro ROC-AUC.")
    print(f"{'Created' if outcome['created'] else 'Updated'}: {outcome['path']}")
    print(f"Frozen selection preserved unchanged: {outcome['frozen_selection_preserved']}")
    print(f"{FROZEN_COMPARISON_FILENAME} untouched (SHA-256 {outcome['frozen_sha256'][:16]}…)")
    print("Threshold-based F1 / sensitivity / specificity are operating-point estimates: "
          "the thresholds were selected on this same validation split.")
    print("The protected official test set was not loaded at any point in this run.")


if __name__ == "__main__":
    main()
