"""End-to-end evaluation of a trained chest X-ray classifier.

Protocol enforced here:

1. predict on **validation** -> select per-label thresholds;
2. predict on **test** -> compute every metric with those frozen thresholds;
3. bootstrap confidence intervals on the test predictions;
4. write metrics (CSV + JSON), curves, confusion matrices and per-image
   predictions under ``outputs/classification/<run>/``.

The test set is touched exactly once, for reporting. No threshold, epoch or
architecture is chosen using it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common.io_utils import save_csv, save_figure, save_json
from ..common.paths import run_dirs
from .metrics import (
    bootstrap_table,
    evaluate_all_labels,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sweep,
    select_thresholds,
)
from .train import predict


def predictions_dataframe(paths, y_true: np.ndarray, y_prob: np.ndarray, labels,
                          thresholds: dict | float = 0.5) -> pd.DataFrame:
    """Per-image table: path, true labels, predicted probabilities, hard predictions.

    Saving this makes the results auditable and lets you inspect specific
    failures later without re-running the model.
    """
    y_true = np.asarray(y_true).reshape(-1, len(labels))
    y_prob = np.asarray(y_prob).reshape(-1, len(labels))

    data = {"image_path": list(paths) if paths is not None and len(paths) == len(y_true)
            else [""] * len(y_true)}
    for i, label in enumerate(labels):
        thr = thresholds.get(label, 0.5) if isinstance(thresholds, dict) else float(thresholds)
        data[f"true_{label}"] = y_true[:, i].astype(int)
        data[f"prob_{label}"] = np.round(y_prob[:, i], 6)
        data[f"pred_{label}"] = (y_prob[:, i] >= thr).astype(int)
    return pd.DataFrame(data)


def evaluate_classifier(model, loaders: dict, cfg, device, run_name: str | None = None,
                        threshold_strategy: str | None = None, n_boot: int | None = None) -> dict:
    """Run the full evaluation protocol and save every artefact.

    Returns a dict with the metric tables, thresholds and prediction DataFrames.
    """
    labels = list(cfg.data.labels)
    run_name = run_name or str(cfg.get("run_name", "run"))
    dirs = run_dirs(cfg.get("output.root", "outputs/classification"), run_name)
    strategy = threshold_strategy or str(cfg.get("eval.threshold_strategy", "f1"))
    min_recall = float(cfg.get("eval.min_recall", 0.80))
    n_boot = int(n_boot if n_boot is not None else cfg.get("eval.n_bootstrap", 1000))
    seed = int(cfg.get("seed", 42))
    model = model.to(device).eval()

    results: dict = {"run_name": run_name, "labels": labels}

    # -- 1. thresholds from validation ------------------------------------
    if "val" in loaders:
        val_true, val_prob, _, val_paths = predict(model, loaders["val"], device, return_paths=True)
        thresholds, threshold_table = select_thresholds(
            val_true, val_prob, labels, strategy=strategy, min_recall=min_recall
        )
        val_metrics = evaluate_all_labels(val_true, val_prob, labels, thresholds)
        save_csv(threshold_table, dirs["metrics"] / "thresholds_from_validation.csv")
        save_csv(val_metrics, dirs["metrics"] / "metrics_validation.csv")
        save_json(thresholds, dirs["metrics"] / "thresholds.json")
        results.update({"thresholds": thresholds, "val_metrics": val_metrics,
                        "val_true": val_true, "val_prob": val_prob})

        for i, label in enumerate(labels):
            fig = plot_threshold_sweep(val_true[:, i], val_prob[:, i],
                                       chosen=thresholds[label], label_name=label)
            save_figure(fig, dirs["figures"] / f"threshold_sweep_{_slug(label)}.png")
    else:
        print("[warn] no validation loader - falling back to a fixed 0.5 threshold. "
              "Report this limitation.")
        thresholds = {label: 0.5 for label in labels}
        results["thresholds"] = thresholds

    # -- 2. test-set metrics with frozen thresholds -----------------------
    if "test" not in loaders:
        print("[warn] no test loader - evaluation stopped after the validation stage.")
        return results

    test_true, test_prob, _, test_paths = predict(model, loaders["test"], device, return_paths=True)
    test_metrics = evaluate_all_labels(test_true, test_prob, labels, thresholds)
    save_csv(test_metrics, dirs["metrics"] / "metrics_test.csv")
    save_json(
        {"run_name": run_name, "n_test_images": int(len(test_true)),
         "threshold_strategy": strategy, "thresholds": thresholds,
         "per_label": test_metrics.to_dict(orient="records")},
        dirs["metrics"] / "metrics_test.json",
    )

    predictions = predictions_dataframe(test_paths, test_true, test_prob, labels, thresholds)
    save_csv(predictions, dirs["predictions"] / "test_predictions.csv")

    # -- 3. bootstrap CIs --------------------------------------------------
    ci_table = pd.DataFrame()
    if n_boot > 0:
        print(f"[eval] bootstrapping {n_boot} resamples for confidence intervals ...")
        ci_table = bootstrap_table(test_true, test_prob, labels, thresholds,
                                   n_boot=n_boot, seed=seed)
        save_csv(ci_table, dirs["metrics"] / "metrics_test_bootstrap_ci.csv")

    # -- 4. figures --------------------------------------------------------
    save_figure(plot_roc_curves(test_true, test_prob, labels), dirs["figures"] / "roc_curves_test.png")
    save_figure(plot_pr_curves(test_true, test_prob, labels), dirs["figures"] / "pr_curves_test.png")
    for i, label in enumerate(labels):
        y_pred = (test_prob[:, i] >= thresholds[label]).astype(int)
        save_figure(plot_confusion_matrix(test_true[:, i], y_pred, label_name=label),
                    dirs["figures"] / f"confusion_matrix_{_slug(label)}.png")

    results.update({
        "test_metrics": test_metrics,
        "bootstrap_ci": ci_table,
        "predictions": predictions,
        "test_true": test_true,
        "test_prob": test_prob,
        "test_paths": test_paths,
        "dirs": dirs,
    })

    print(f"\n[eval] test metrics for run '{run_name}':")
    print(test_metrics.round(4).to_string(index=False))
    return results


def compare_runs(result_dicts: list[dict], metric: str = "roc_auc") -> pd.DataFrame:
    """Side-by-side comparison table (baseline CNN vs transfer learning) for the report."""
    rows = []
    for res in result_dicts:
        table = res.get("test_metrics")
        if table is None:
            continue
        summary = table.iloc[-1] if len(res["labels"]) > 1 else table.iloc[0]
        rows.append({
            "run": res.get("run_name", "?"),
            "roc_auc": round(float(summary["roc_auc"]), 4),
            "pr_auc": round(float(summary["pr_auc"]), 4),
            "f1": round(float(summary["f1"]), 4),
            "recall_sensitivity": round(float(summary["recall_sensitivity"]), 4),
            "specificity": round(float(summary["specificity"]), 4),
            "accuracy_secondary": round(float(summary["accuracy"]), 4),
        })
    return pd.DataFrame(rows).sort_values(metric, ascending=False, ignore_index=True)


def _slug(text: str) -> str:
    """Filesystem-safe version of a label name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text)).strip("_").lower()
