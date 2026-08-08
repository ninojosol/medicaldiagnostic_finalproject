# Screenshot Checklist for the Report and Presentation

Everything below is produced by running the notebooks in order. Figures are saved
automatically under `outputs/`; the items marked *(cell output)* need a screenshot
of the notebook cell itself.

Tick each item once you have the real artefact. **Do not include a figure you did
not generate**, and do not describe results you did not measure.

---

## Setup and environment

- [ ] **Environment check** — notebook 00, section 3 *(cell output)*
      PyTorch version, CUDA availability, GPU name. Documents the hardware
      behind your results.
      → also copy these into the reproducibility table (`docs/REPRODUCIBILITY.md`)

- [ ] **Project structure** — a screenshot of the folder tree
      Shows the separation of `src/`, `notebooks/`, `configs/`, `outputs/`.

---

## Part A — Chest X-ray classification

### Data understanding (notebook 01)

- [ ] **Class distribution**
      `outputs/classification/eda/figures/class_distribution.png`
      Motivates class weighting and the choice of metrics.

- [ ] **Manifest integrity report** *(cell output, section 2)*
      Row counts, patients, missing files, positives per label.

- [ ] **Images per patient** — `.../images_per_patient.png`
      Justifies the patient-level split. Skip if one image per patient, and say so.

- [ ] **Sample images** — `.../samples_<LABEL>_1.png` and `..._0.png`
      Positive and negative examples side by side.

- [ ] **Patient leakage check** *(cell output, section 7a)* — **essential**
      Direct evidence that no patient appears in two splits.

- [ ] **Split summary table** *(cell output, section 7b)*
      Images, patients and positive rate per split.

### Training (notebook 02)

- [ ] **Class-imbalance handling** *(cell output)*
      The computed `pos_weight` per label.

- [ ] **Baseline CNN training curves**
      `outputs/classification/xray_baseline_cnn/figures/training_curves.png`

- [ ] **Transfer-learning training curves**
      `outputs/classification/xray_densenet121/figures/training_curves.png`

- [ ] **Validation comparison table** *(cell output)*
      Best epoch and validation ROC-AUC for both runs.

### Evaluation (notebook 03)

- [ ] **Model comparison on test** *(cell output / CSV)*
      `outputs/classification/model_comparison_test.csv` — the headline table.

- [ ] **ROC curves** — `.../figures/roc_curves_test.png`

- [ ] **Precision-Recall curves** — `.../figures/pr_curves_test.png`
      With the prevalence baseline visible.

- [ ] **Confusion matrix** — `.../figures/confusion_matrix_<label>.png`
      At the threshold chosen on validation.

- [ ] **Threshold sweep** — `.../figures/threshold_sweep_<label>.png`
      Justifies the operating point. Frequently missing from student reports;
      including it is a clear quality signal.

- [ ] **Bootstrap confidence intervals** *(cell output)*
      Point estimate + 95% CI per metric.

- [ ] **Grad-CAM: correct predictions** — `.../figures/gradcam_TP.png`

- [ ] **Grad-CAM: failure cases** — `.../figures/gradcam_FP.png` or `gradcam_FN.png`
      **Include at least one.** Showing only successes misrepresents the model.

- [ ] **Error analysis** *(cell output)*
      Most confident false negatives and false positives.

---

## Part B — Brain MRI segmentation

### Data understanding (notebook 04)

- [ ] **Pair validation output** *(cell output, section 2)* — **essential**
      Proves images and masks are correctly paired and size-matched.

- [ ] **Mask coverage figures**
      `outputs/segmentation/eda/figures/mask_coverage.png`
      Tumour vs non-tumour slices, and tumour area distribution. This is the
      imbalance figure that justifies Dice + BCE.

- [ ] **Sample MRI/mask overlays** — `.../sample_overlays.png`

- [ ] **Single pair check** — `.../pair_example.png`
      Slice, mask and overlay side by side.

- [ ] **Patient leakage check + split summary** *(cell output, section 7)*

### Training (notebook 05)

- [ ] **Batch sanity check** *(cell output)*
      Image range, mask values ∈ {0,1}, foreground fraction.

- [ ] **Augmentation alignment check** *(cell output figure)*
      The same slice under three augmentations, mask following the image.
      Strong evidence the pipeline is correct.

- [ ] **Training curves** — `outputs/segmentation/mri_unet/figures/training_curves.png`
      Loss and Dice, train vs validation.

### Evaluation (notebook 06)

- [ ] **Headline metrics table** *(cell output)*
      Mean Dice and IoU on tumour slices, threshold, slice counts.

- [ ] **Confidence intervals** *(cell output)*
      Dice and IoU with 95% CI, resampled by patient.

- [ ] **Dice distribution** — `.../figures/dice_distribution_test.png`
      Shows the spread the mean hides.

- [ ] **Best-case overlays** — `.../figures/overlays_best.png`

- [ ] **Worst-case overlays** — `.../figures/overlays_worst.png`
      **Required.** Best-only results are not an honest presentation.

- [ ] **Typical-case overlays** — `.../figures/overlays_typical.png`

- [ ] **Dice vs tumour size** — `.../figures/dice_vs_tumour_size.png`
      Usually reveals that small lesions drive the failures.

- [ ] **Threshold sweep** — `.../figures/threshold_sweep.png`

---

## Cross-cutting slides

- [ ] **Architecture diagram** — draw or cite one for the CNN/DenseNet and U-Net
      (explain the skip connections; they are the reason U-Net is used).

- [ ] **Workflow diagram** — the pipeline from `README.md` (data → manifest →
      split → train → evaluate → interpret).

- [ ] **Reproducibility table** — filled in from `run_metadata.json`
      (see `docs/REPRODUCIBILITY.md`).

- [ ] **Limitations slide** — from `docs/ETHICS.md` section 3.

- [ ] **Disclaimer on the title slide** — `docs/ETHICS.md` section 6.
      Verbatim.

---

## Before you submit

- [ ] Every number in the report traces to a file under `outputs/` — no
      remembered or estimated values.
- [ ] Every metric states its split, threshold and (where computed) its CI.
- [ ] Dice figures state whether they are tumour-slices-only.
- [ ] At least one failure case is shown for **each** of the two tasks.
- [ ] No claim of diagnostic ability, clinical utility or deployment readiness
      appears anywhere.
- [ ] The dataset is cited with its licence.
- [ ] `data/` and model weights are **not** committed to git.
