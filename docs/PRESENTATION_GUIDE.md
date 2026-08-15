# Presentation Playbook

This guide is the team’s single preparation reference for the AI-Assisted Medical Image Analysis presentation.

> Academic research prototype only. It is not a medical device and must never be used for patient-care decisions.

## 1. One-minute project summary

We developed an academic AI-assisted medical-image-analysis prototype with two complementary tasks.

1. Chest X-ray classification predicts an independent probability for each of 14 possible findings. Several findings can be predicted from the same image, so it is a multi-label classification problem.
2. Brain MRI tumour segmentation predicts a pixel-level whole-tumour mask. It answers where the predicted region is, rather than only whether a finding may be present.

The Streamlit app makes data preparation, quality checks, model evidence, visual explanations, and controlled inference demonstrations easier to inspect. It does not replace a clinician.

## 2. Verified facts every presenter must know

| Topic | Verified fact | Safe presentation wording |
|---|---|---|
| X-ray input | One frontal grayscale chest X-ray image | “The image is converted to three channels, resized, normalised, and passed to the trained model.” |
| X-ray output | Independent probabilities for 14 NIH-style findings | “One X-ray can have multiple labels, so this is multi-label classification.” |
| MRI input | Four MRI sequences processed slice-wise | “The model receives multi-sequence MRI slices.” |
| MRI output | Binary whole-tumour mask / overlay | “Segmentation predicts the tumour region at pixel level.” |
| X-ray split | 795 training images, 207 validation images, 420 protected official-test records | “Selection and threshold tuning used validation data; the protected test set was not used for those decisions.” |
| X-ray models | Baseline CNN, fine-tuned DenseNet, fine-tuned EfficientNet-B0, experimental ViT-B/16 | “We compare a baseline, transfer-learning CNNs, and an experimental transformer comparator.” |
| Frozen selection | Fine-tuned EfficientNet-B0 was selected among the original three-model comparison | “EfficientNet-B0 is the selected model in the frozen three-model experiment.” |
| Experimental comparator | ViT-B/16 validation macro ROC-AUC 0.7349; EfficientNet-B0 0.7221 | “ViT is validation-leading experimentally, but it does not replace the frozen selection.” |
| X-ray metrics | ROC-AUC and PR-AUC are headline evidence; accuracy is secondary | “Accuracy alone is misleading with rare findings.” |
| MRI metrics | Dice is primary; IoU is supporting | “Dice measures overlap between prediction and ground-truth mask.” |
| Clinical status | No clinical validation, approval, or external validation | “This is research evidence, not clinical deployment.” |

## 3. Recommended presentation flow

| Part | Main point | Evidence to show |
|---|---|---|
| Problem and objective | Medical images are complex; the project studies two AI tasks with different outputs. | Title / architecture overview |
| Data and safety | Patient-safe splitting, validation checks, and no raw sensitive data committed to GitHub. | Data Preparation / Quality Review |
| X-ray classification | Multi-label outputs; compare baseline and transfer-learning models. | Comparison, ROC/PR, thresholds, Grad-CAM |
| MRI segmentation | U-Net predicts the whole-tumour region at pixel level. | Preprocessing, mask overlay, Dice / IoU |
| Live application | The app makes the research workflow inspectable. | One X-ray inference and one MRI overlay |
| Ethics and conclusion | Academic prototype; results are not a clinical claim. | Limitations / ethics slide |

## 4. Live-demo sequence

Do a complete dry run before presenting. The demo operator shares only the browser window and should never run training during the presentation.

1. Start at the landing page and introduce the two tracks.
2. Open the X-ray workflow.
   - Show data preparation or quality panels if available.
   - Show the comparison and explain the frozen EfficientNet-B0 selection.
   - If showing ViT, call it an experimental validation-leading comparator.
   - Upload one approved presentation image.
   - Explain probabilities and Grad-CAM carefully: Grad-CAM is a visual inspection aid, not clinical proof or causality.
3. Open the MRI workflow.
   - Select an included presentation sample.
   - Show the original slice, reference mask when available, and prediction overlay.
   - Explain Dice and IoU: higher overlap means better agreement with the reference mask.
4. End with the academic-only disclaimer, not with a disease label.

## 5. Technical checklist

- [ ] Pull the latest main branch.
- [ ] Use Python 3.11.
- [ ] Install PyTorch before requirements.txt.
- [ ] Install presentation checkpoints and MRI demo samples from the release asset bundle.
- [ ] Run scripts/run_app.ps1 or streamlit run app/streamlit_app.py.
- [ ] Confirm the browser opens at http://localhost:8501.
- [ ] Test one approved X-ray image.
- [ ] Test one MRI sample and mask overlay.
- [ ] Confirm browser zoom, display connection, and font size.
- [ ] Keep captured screenshots or video ready as a fallback.
- [ ] Never use real patient images or personally identifiable information.

## 6. Suggested team roles

| Role | Before presenting | During presentation |
|---|---|---|
| Project lead | Confirms story, slide order, and scope. | Opens and closes; connects both tasks. |
| Data / methodology presenter | Reviews sources, splits, preprocessing, and reproducibility. | Explains patient-safe splitting and leakage prevention. |
| Classification presenter | Reviews labels, models, ROC/PR, thresholds, and Grad-CAM. | Explains multi-label output and model selection. |
| Segmentation presenter | Reviews U-Net, Dice/IoU, and mask overlays. | Explains pixel-level outputs and limitations. |
| Demo operator | Installs assets and tests both workflows. | Shares screen and follows the demo sequence. |
| Backup / Q&A presenter | Reviews metrics, ethics, and limitations. | Answers questions and opens backup screenshots if needed. |

## 7. Short metric explanations

| Metric | Simple explanation |
|---|---|
| ROC-AUC | Measures how well the model ranks positives above negatives across thresholds. |
| PR-AUC | Focuses on precision and recall, which is especially useful for rare findings. |
| Recall / sensitivity | Of true positives, how many the model flags. |
| Specificity | Of true negatives, how many the model leaves unflagged. |
| Precision | Of predicted positives, how many are truly positive. |
| F1 | Balances precision and recall at one threshold. |
| Dice | Pixel overlap between predicted and reference masks; 1 means perfect overlap. |
| IoU | A stricter pixel-overlap measure that is always lower than Dice for the same prediction. |

Thresholds are selected on validation data only. Do not claim that they were tuned on, or proven by, the protected official test set.

## 8. Likely questions and concise answers

**Why not use accuracy as the headline X-ray metric?**  
Rare labels make accuracy misleading: a model can predict negative almost everywhere and still look accurate. ROC-AUC and PR-AUC are more informative.

**What is multi-label classification?**  
The image may contain more than one finding. The model gives an independent probability for each of 14 labels instead of choosing only one class.

**Why have CNN, DenseNet, EfficientNet, and ViT models?**  
The baseline CNN establishes a starting point. Transfer-learning models test stronger pretrained architectures. ViT is a transformer-based experimental comparator.

**Which model is best?**  
EfficientNet-B0 is the selected model from the frozen original three-model record. ViT-B/16 is an experimental validation-leading comparator, but it is not promoted automatically because the comparison is validation-only and input resolutions differ.

**What is the difference between classification and segmentation?**  
Classification predicts whether findings may be present. Segmentation predicts the location of a region for every pixel.

**What does Grad-CAM prove?**  
It proves nothing clinically. It is a post-hoc visualisation of regions associated with a model score and helps inspect behaviour.

**Can this system be used in a hospital?**  
No. It is an academic prototype without clinical validation, regulatory approval, external validation, or clinician-supervised deployment.

**Why use Dice rather than pixel accuracy for MRI segmentation?**  
Most pixels are background. A model could get high pixel accuracy by predicting only background, while failing to segment a tumour. Dice measures the region overlap directly.

**What are the next steps?**  
External test evaluation, larger and more representative datasets, subgroup/error analysis, clinician review, external validation, and full clinical governance before any real-world use.

## 9. Claims to use and claims to avoid

| Use | Avoid |
|---|---|
| “The model produced validation results on the project dataset.” | “The model is clinically accurate.” |
| “The protected official test set was not used for selection or threshold tuning.” | “These are final clinical-test results.” |
| “The app displays research predictions and visual explanations.” | “The app diagnoses patients.” |
| “Grad-CAM helps inspect model behaviour.” | “Grad-CAM proves disease.” |
| “The U-Net predicts a whole-tumour mask.” | “The system confirms a brain-tumour diagnosis.” |
| “EfficientNet-B0 is the frozen selected model among the original three.” | “ViT is the official selected model.” |
| “ViT is a validation-leading experimental comparator.” | “ViT definitively outperforms every model.” |

## 10. Backup plan

1. Do not spend several minutes troubleshooting in front of the professor.
2. Switch to prepared screenshots or a recorded demo.
3. Show X-ray input, probabilities, Grad-CAM, model-comparison evidence, MRI overlay, and Dice/IoU.
4. Continue the same story: input, preprocessing, model, output, metric, limitation.
5. State honestly that captured outputs are being used to keep within the presentation time.
6. Keep the repository, release asset bundle, and screenshots available offline.

## 11. Where to find evidence in the repository

| Location | Purpose |
|---|---|
| app/streamlit_app.py | Streamlit application entry point |
| src/classification/ | X-ray preprocessing, model, training, and metrics |
| src/segmentation/ | MRI segmentation model and metrics |
| configs/ | Experiment settings |
| notebooks/ | Workflow notebooks |
| outputs/classification/ | Saved comparison artifacts and metrics |
| outputs/segmentation/ | Segmentation artifacts and metrics |
| docs/METRICS.md | Correct metric interpretation |
| docs/REPRODUCIBILITY.md | Seeds, stable splits, and reproducibility limits |
| docs/ETHICS.md | Safety and non-clinical-use boundaries |
| docs/TEAM_SETUP.md | Windows setup and troubleshooting |
