"""Evaluation metrics for medical image classification.

Why accuracy is only a secondary metric
---------------------------------------
If a finding is present in 2% of images, a model that always predicts "negative"
scores 98% accuracy and detects nothing. For imbalanced clinical tasks the
informative metrics are:

============================  ============================================================
metric                        reads as
============================  ============================================================
ROC-AUC                       ranking quality, independent of threshold
PR-AUC (average precision)    ranking quality focused on the rare positive class
recall / sensitivity          of all patients WITH the finding, how many were caught
specificity                   of all patients WITHOUT it, how many were correctly cleared
precision (PPV)               of the positives flagged, how many were real
F1                            harmonic mean of precision and recall
============================  ============================================================

Thresholds are selected on the **validation** split only and then applied
unchanged to test. Tuning a threshold on test is a form of leakage and inflates
the reported score.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# core per-label metrics
# ---------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
                   with_auc: bool = True) -> dict:
    """Full metric set for one binary label.

    Returns NaN (not an exception) for metrics that are undefined on the given
    data - e.g. ROC-AUC when only one class is present in a split.

    ``with_auc=False`` skips ROC-AUC and PR-AUC. They are threshold-independent,
    so recomputing them inside a 99-step threshold sweep or a 1000-iteration
    bootstrap over a threshold-dependent metric is pure waste - and on a large
    test set it dominates the runtime.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    both_classes = len(np.unique(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "roc_auc": (float(roc_auc_score(y_true, y_prob))
                    if (with_auc and both_classes) else float("nan")),
        "pr_auc": (float(average_precision_score(y_true, y_prob))
                   if (with_auc and both_classes) else float("nan")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(
            0.5 * ((tp / (tp + fn) if (tp + fn) else 0.0) + (tn / (tn + fp) if (tn + fp) else 0.0))
        ),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "prevalence": float((tp + fn) / len(y_true)) if len(y_true) else float("nan"),
    }


def evaluate_all_labels(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str],
                        thresholds: dict[str, float] | float = 0.5) -> pd.DataFrame:
    """Per-label metric table plus a macro-average row.

    ``thresholds`` may be a single float applied to every label, or a dict of
    per-label thresholds produced by :func:`select_thresholds`.
    """
    y_true = np.atleast_2d(np.asarray(y_true))
    y_prob = np.atleast_2d(np.asarray(y_prob))
    if y_true.shape[1] != len(labels):
        y_true = y_true.reshape(-1, len(labels))
        y_prob = y_prob.reshape(-1, len(labels))

    rows = []
    for i, label in enumerate(labels):
        thr = thresholds.get(label, 0.5) if isinstance(thresholds, dict) else float(thresholds)
        row = {"label": label, **binary_metrics(y_true[:, i], y_prob[:, i], thr)}
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(labels) > 1:
        numeric = df.select_dtypes(include=[np.number]).mean(numeric_only=True)
        macro = {"label": "MACRO_AVG", **{k: float(v) for k, v in numeric.items()}}
        for key in ("tp", "fp", "tn", "fn", "n", "n_positive", "n_negative"):
            macro[key] = int(df[key].sum())
        df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# threshold selection - VALIDATION DATA ONLY
# ---------------------------------------------------------------------------
def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, strategy: str = "f1",
                     min_recall: float = 0.80) -> tuple[float, dict]:
    """Choose an operating threshold from validation predictions.

    Strategies
    ----------
    ``f1``
        Maximise F1 - a balanced default when both error types matter.
    ``youden``
        Maximise ``sensitivity + specificity - 1`` (Youden's J), the classic
        choice for screening-style ROC analysis.
    ``min_recall``
        Highest precision among thresholds reaching ``min_recall`` sensitivity.
        Use this when missing a positive case is the costlier error, which is the
        usual argument in a clinical setting - and state that argument explicitly
        in the report rather than defaulting to 0.5.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if len(np.unique(y_true)) < 2:
        print("[warn] only one class present; falling back to threshold 0.5")
        return 0.5, {"strategy": strategy, "reason": "single-class validation data"}

    if strategy == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        best = int(np.argmax(j))
        chosen = float(np.clip(thr[best], 0.0, 1.0))
        return chosen, {"strategy": "youden", "youden_j": float(j[best]),
                        "sensitivity": float(tpr[best]), "specificity": float(1 - fpr[best])}

    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more point than thresholds
    precision, recall = precision[:-1], recall[:-1]
    if len(thr) == 0:
        return 0.5, {"strategy": strategy, "reason": "no thresholds available"}

    if strategy == "min_recall":
        feasible = recall >= min_recall
        if not feasible.any():
            print(f"[warn] no threshold reaches recall >= {min_recall}; using best-F1 instead.")
        else:
            idx = int(np.argmax(np.where(feasible, precision, -1)))
            return float(thr[idx]), {"strategy": "min_recall", "min_recall": min_recall,
                                     "precision": float(precision[idx]), "recall": float(recall[idx])}

    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    idx = int(np.argmax(f1))
    return float(thr[idx]), {"strategy": "f1", "f1": float(f1[idx]),
                             "precision": float(precision[idx]), "recall": float(recall[idx])}


def select_thresholds(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str],
                      strategy: str = "f1", min_recall: float = 0.80) -> tuple[dict, pd.DataFrame]:
    """Per-label threshold selection. Feed it VALIDATION predictions only."""
    y_true = np.asarray(y_true).reshape(-1, len(labels))
    y_prob = np.asarray(y_prob).reshape(-1, len(labels))

    thresholds, rows = {}, []
    for i, label in enumerate(labels):
        thr, info = select_threshold(y_true[:, i], y_prob[:, i], strategy=strategy, min_recall=min_recall)
        thresholds[label] = thr
        rows.append({"label": label, "threshold": round(thr, 4), **info})

    table = pd.DataFrame(rows)
    print(f"[thresholds] selected on VALIDATION data with strategy='{strategy}':")
    print(table.to_string(index=False))
    return thresholds, table


# ---------------------------------------------------------------------------
# bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "roc_auc",
                 threshold: float = 0.5, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """Percentile bootstrap CI for one metric on one label.

    Resamples the test set with replacement ``n_boot`` times and reports the
    empirical 2.5th/97.5th percentiles. This communicates how much of the
    reported number is sampling noise - important because course-sized test sets
    are small. Resampling is at image level; with several images per patient the
    interval is mildly optimistic, which is worth stating in the report.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Only compute the AUCs when the requested metric actually is one.
    needs_auc = metric in {"roc_auc", "pr_auc"}

    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # degenerate resample - AUC undefined
        try:
            scores.append(binary_metrics(yt, yp, threshold, with_auc=needs_auc)[metric])
        except Exception:
            continue

    scores = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
    point = binary_metrics(y_true, y_prob, threshold, with_auc=needs_auc).get(metric, float("nan"))
    if len(scores) < 20:
        return {"metric": metric, "point_estimate": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot_valid": int(len(scores)),
                "note": "too few valid bootstrap resamples for a reliable CI"}

    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci_low": float(np.percentile(scores, 100 * alpha / 2)),
        "ci_high": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "ci_level": float(1 - alpha),
        "n_boot_valid": int(len(scores)),
    }


def bootstrap_table(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str],
                    thresholds: dict[str, float] | float = 0.5,
                    metrics: Sequence[str] = ("roc_auc", "pr_auc", "recall_sensitivity",
                                              "specificity", "f1"),
                    n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap CIs for several metrics x labels, formatted for the report."""
    y_true = np.asarray(y_true).reshape(-1, len(labels))
    y_prob = np.asarray(y_prob).reshape(-1, len(labels))

    rows = []
    for i, label in enumerate(labels):
        thr = thresholds.get(label, 0.5) if isinstance(thresholds, dict) else float(thresholds)
        for metric in metrics:
            res = bootstrap_ci(y_true[:, i], y_prob[:, i], metric=metric, threshold=thr,
                               n_boot=n_boot, seed=seed)
            rows.append({
                "label": label,
                "metric": metric,
                "value": round(res["point_estimate"], 4),
                "ci_low": round(res["ci_low"], 4),
                "ci_high": round(res["ci_high"], 4),
                "formatted": f"{res['point_estimate']:.3f} "
                             f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]"
                             if np.isfinite(res["ci_low"]) else f"{res['point_estimate']:.3f} [n/a]",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str],
                    title: str = "ROC curves (test set)"):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).reshape(-1, len(labels))
    y_prob = np.asarray(y_prob).reshape(-1, len(labels))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, label in enumerate(labels):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str],
                   title: str = "Precision-Recall curves (test set)"):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).reshape(-1, len(labels))
    y_prob = np.asarray(y_prob).reshape(-1, len(labels))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, label in enumerate(labels):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        base = y_true[:, i].mean()
        ax.plot(recall, precision, lw=2, label=f"{label} (AP={ap:.3f})")
        ax.axhline(base, ls=":", lw=1, color="gray")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"{title}\n(dotted line = prevalence baseline)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_name: str = "",
                          class_names: Sequence[str] = ("Negative", "Positive"), normalize: bool = False):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(np.asarray(y_true).ravel().astype(int),
                          np.asarray(y_pred).ravel().astype(int), labels=[0, 1])
    display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(display, cmap="Blues")
    ax.set_xticks([0, 1], labels=list(class_names))
    ax.set_yticks([0, 1], labels=list(class_names))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix{': ' + label_name if label_name else ''}")
    ax.grid(False)
    threshold_color = display.max() / 2
    corner = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            text = f"{display[i, j]:.2f}" if normalize else f"{cm[i, j]:,}"
            ax.text(j, i, f"{corner[i][j]}\n{text}", ha="center", va="center",
                    color="white" if display[i, j] > threshold_color else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, chosen: float | None = None,
                         label_name: str = ""):
    """Precision / recall / F1 as a function of threshold - shows WHY a threshold was picked."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    grid = np.linspace(0.01, 0.99, 99)
    # AUCs are threshold-independent, so skip them across the whole sweep.
    rows = [binary_metrics(y_true, y_prob, t, with_auc=False) for t in grid]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, name in [("precision", "Precision"), ("recall_sensitivity", "Recall / Sensitivity"),
                      ("specificity", "Specificity"), ("f1", "F1")]:
        ax.plot(grid, [r[key] for r in rows], lw=2, label=name)
    if chosen is not None:
        ax.axvline(chosen, color="red", ls="--", lw=1.5, label=f"chosen = {chosen:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Threshold sweep{': ' + label_name if label_name else ''} (validation set)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
