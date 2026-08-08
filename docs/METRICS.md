# Evaluation Metrics

Why each metric is used, how to read it, and where it comes from in the code.

---

## Part A — Chest X-ray classification

### Why accuracy is only a secondary metric

If a finding appears in 2% of images, a model that predicts "negative" for
everything scores **98% accuracy** and detects nothing. Accuracy is dominated by
the majority class, so on an imbalanced medical task it measures prevalence, not
skill. It is reported here for completeness and clearly labelled secondary.

### The metrics actually reported

Terminology: **TP** true positive, **FP** false positive, **TN** true negative,
**FN** false negative.

| Metric | Formula | Reads as | Threshold-dependent? |
|---|---|---|---|
| **ROC-AUC** | area under TPR vs FPR | probability a random positive scores higher than a random negative | no |
| **PR-AUC** (average precision) | area under precision vs recall | as above but focused on the rare positive class | no |
| **Recall / Sensitivity** | TP / (TP+FN) | of patients **with** the finding, the fraction detected | yes |
| **Specificity** | TN / (TN+FP) | of patients **without** it, the fraction correctly cleared | yes |
| **Precision (PPV)** | TP / (TP+FP) | of the cases flagged positive, the fraction that were real | yes |
| **F1** | 2·P·R / (P+R) | harmonic mean of precision and recall | yes |
| **Balanced accuracy** | (Sens + Spec) / 2 | accuracy corrected for class imbalance | yes |
| Accuracy | (TP+TN) / N | *secondary only* | yes |

**ROC-AUC vs PR-AUC.** ROC-AUC is stable under class imbalance but can look
reassuring when the positive class is tiny, because a large number of true
negatives keeps the false-positive rate low. PR-AUC has no true-negative term and
degrades visibly when precision is poor — it is the more honest summary for rare
findings. Report both. The PR baseline is the prevalence, drawn as a dotted line
in the PR figure; an AP barely above that line means the model has learned little.

### Confusion matrix

```
                 Predicted
                 Neg     Pos
Actual   Neg     TN      FP     <- false alarms: unnecessary follow-up
         Pos     FN      TP     <- FN = missed cases, usually the costliest error
```

### Threshold selection — validation data only

A model outputs a probability; turning it into a decision requires a threshold,
and 0.5 is rarely the right one for an imbalanced task. Three strategies are
available via `eval.threshold_strategy`:

| Strategy | Picks the threshold that... | Use when |
|---|---|---|
| `f1` | maximises F1 | both error types matter roughly equally (default) |
| `youden` | maximises Sensitivity + Specificity − 1 | classic screening/ROC analysis |
| `min_recall` | maximises precision subject to recall ≥ `eval.min_recall` | missing a positive is clearly worse than a false alarm |

**The threshold is always chosen on the validation split and then applied
unchanged to test.** Tuning it on test is leakage and inflates the reported
result. The threshold-sweep figure shows how precision, recall, specificity and
F1 trade off, and marks the chosen point — it is the justification for the
operating point, so include it in the report.

### Bootstrap confidence intervals

The test set is resampled with replacement (default 1000 times) and the metric
recomputed each time; the reported interval is the 2.5th–97.5th percentile.

A wide interval means the test set is too small to pin the number down. If two
models' intervals overlap substantially, **you cannot claim one is better** —
saying so is a stronger result than overstating a difference.

*Caveat to state:* resampling is at image level. When patients contribute several
images those are not independent, so the interval is mildly optimistic.

**Code:** `src/classification/metrics.py`

---

## Part B — Brain MRI segmentation

### Dice coefficient (F1 over pixels) — the primary metric

```
Dice = 2·|A ∩ B| / (|A| + |B|)
```

where A is the predicted mask and B the ground truth. Ranges 0 (no overlap) to
1 (perfect). It is the standard metric in medical segmentation because it is
insensitive to the huge background class and weights overlap directly.

### IoU / Jaccard index

```
IoU = |A ∩ B| / |A ∪ B|
```

Monotonically related to Dice (`IoU = Dice / (2 − Dice)`) but always numerically
lower, and it penalises the same error more harshly. Reported alongside Dice
because different papers quote different ones — always say which you mean.

### Three reporting decisions that change the number a lot

**1. Per-image mean, not pooled pixels.** This project reports the mean of
per-slice Dice. "Global Dice" (all pixels pooled across the test set) is also
saved, but it is dominated by large tumours and hides failures on small ones.

**2. Empty masks.** A slice with no tumour where the model predicts nothing has
an undefined Dice (0/0). Here it counts as **1.0** — correct behaviour is
rewarded. But if 60% of slices are empty, that convention inflates the average
badly, so `mean_dice_tumour_slices` is reported next to
`mean_dice_all_slices`.

**Quote `mean_dice_tumour_slices` as the primary result** and say so explicitly.

**3. Slice-level vs patient-level.** Slices from one MRI volume are near
duplicates. The bootstrap therefore resamples **by patient** where patient IDs
exist; resampling by slice would produce intervals that are far too narrow.

### Supporting metrics

| Metric | Meaning |
|---|---|
| Pixel sensitivity | fraction of true tumour pixels recovered — high with over-segmentation |
| Pixel specificity | fraction of background pixels correctly left out — always ~0.99, so nearly uninformative on its own |
| Median Dice | robust to a few catastrophic slices; report next to the mean |
| Dice distribution | the histogram is often bimodal (good slices + missed small lesions), which the mean cannot show |

**Pixel accuracy is deliberately not reported** — "all background" would score
above 98%.

### Threshold for binarisation

The U-Net outputs a probability per pixel. `eval.tune_threshold: true` sweeps the
threshold on **validation**, picks the best mean Dice, and freezes it for test —
the same discipline as the classifier. A flat sweep curve means the model is well
calibrated; a sharp peak means the reported Dice depends heavily on a value tuned
on a small validation set, which is worth flagging.

**Code:** `src/segmentation/metrics.py`

---

## Reporting checklist

For every number you put in the report:

- [ ] state which **split** it came from (test, unless explicitly noted);
- [ ] state the **threshold** used and how it was chosen;
- [ ] give a **confidence interval** where one was computed;
- [ ] for Dice, say whether it is **tumour-slices-only** or all slices;
- [ ] pair sensitivity **with** specificity — neither is meaningful alone;
- [ ] never quote accuracy as the headline result;
- [ ] never present a metric as evidence of clinical usefulness (see
      `docs/ETHICS.md`).
