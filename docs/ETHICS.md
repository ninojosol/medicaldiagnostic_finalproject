# Ethical and Clinical Limitations

**Read this before writing any conclusion about the models in this project.**

---

## 1. What this project is

An **educational prototype** built for a university final project. Its purpose is
to demonstrate a sound machine-learning workflow on medical imaging data:
patient-level splitting, appropriate loss functions for imbalanced data,
threshold selection on validation data, metrics suited to clinical tasks, and
honest reporting of uncertainty.

## 2. What this project is **not**

This system:

- **cannot diagnose any patient** and must never be used to;
- **does not replace, assist or second-guess a radiologist or any clinician**;
- **is not a medical device**, and has not been assessed under any medical device
  regulation (EU MDR, FDA, or otherwise);
- **has not been clinically validated** on any prospective cohort;
- **has no regulatory approval of any kind**;
- **must not be deployed** in any setting where its output could influence a
  decision about a real person's care.

Any statement in the report, slides or code comments that implies otherwise is
wrong and should be corrected.

### Language to avoid, and what to write instead

| Do not write | Write instead |
|---|---|
| "The model diagnoses pneumonia" | "The model classifies images as pneumonia-positive under the dataset's labelling" |
| "95% accurate at detecting tumours" | "Mean Dice of X (95% CI [a, b]) on held-out test patients from this dataset" |
| "Can assist doctors" | "Demonstrates a workflow that would require extensive clinical validation before any such claim" |
| "Grad-CAM shows the lesion" | "Grad-CAM shows which image regions most influenced the score" |
| "Ready for deployment" | "A prototype for coursework; not deployable" |

---

## 3. Known technical limitations

These apply regardless of how good the numbers look. State them in the report.

**Dataset and generalisation**

- Trained and evaluated on a single dataset, likely from a small number of sites,
  scanners and acquisition protocols. Performance on data from a different
  hospital or scanner is unknown and typically much worse.
- No external validation set. All reported numbers come from the same data
  distribution as training.
- Demographic composition (age, sex, ethnicity, disease prevalence) is usually
  undocumented in public datasets. Subgroup performance is therefore unmeasured,
  and a model can perform well on average while failing on an under-represented
  group.

**Labels**

- Labels in public chest X-ray datasets are frequently derived by automated text
  mining of radiology reports and contain a meaningful error rate. The model can
  only be as correct as its labels.
- Segmentation masks reflect one annotator's interpretation. Inter-rater
  variability in tumour boundaries is substantial and is not measured here, so
  Dice is compared against one opinion, not ground truth.

**Modelling**

- Chest X-ray classification is 2D and single-view; real interpretation uses
  prior imaging, lateral views, and clinical context that this model never sees.
- MRI segmentation is done slice-by-slice with no 3D consistency between adjacent
  slices. A real volumetric assessment would differ.
- Test sets in a course project are small. Confidence intervals are wide, and two
  models whose intervals overlap cannot be said to differ.
- Shortcut learning is likely: models readily key on scanner artefacts, text
  markers, chest tubes or positioning that correlate with the label in the
  training data but carry no diagnostic meaning. The Grad-CAM section exists
  partly to make this visible.

**Metrics**

- Dice and ROC-AUC measure statistical agreement, not clinical usefulness. A
  segmentation with acceptable Dice can still be unusable for surgical planning;
  a classifier with high AUC can still be unsafe at every operating point.
- Threshold choice is a value judgement about the relative cost of a missed case
  versus a false alarm — not a property of the model. Justify it explicitly.

---

## 4. Data handling obligations

- Never commit patient data to version control. `data/` is git-ignored for this
  reason; do not override it.
- Do not upload datasets to third-party services (including cloud notebooks or AI
  assistants) unless the licence explicitly permits it.
- Check the dataset licence before use and cite it in the report.
- If you ever handle non-public clinical data, ethics approval and a data
  processing agreement are required first — an academic project is not an
  exception.

---

## 5. Why fairness and error asymmetry matter here

In a screening context a **false negative** (a missed finding) and a **false
positive** (a false alarm) are not equally costly, and neither is captured by
accuracy. This project therefore reports sensitivity and specificity separately,
selects the operating threshold explicitly, and provides a `min_recall` threshold
strategy for the case where missing a positive is deemed worse.

Whichever you choose, state the choice and the reasoning. Reporting a single
accuracy figure for an imbalanced medical task is the error this project is
designed to avoid.

---

## 6. Required disclaimer

Include this in the report and on the title slide:

> This work is an academic prototype developed for coursework. It is not a
> medical device, has not been clinically validated, and must not be used for
> diagnosis or any clinical decision-making. All results are limited to the
> specific public dataset used and do not generalise to clinical practice.
