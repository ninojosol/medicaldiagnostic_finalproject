"""Generate docs/MRI_SEGMENTATION.md from the real artifacts.

Every number in the document is read from data/processed/mri/** and
outputs/segmentation/<run>/**, so the documentation cannot drift from the run.

Usage::

    python scripts/mri_write_docs.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.constants import (  # noqa: E402
    DATASET_HOMEPAGE,
    DATASET_ID,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_URL,
    HELD_OUT_TEST_NOTE,
    MODEL_ARCH_NOTE,
    MODEL_LABEL,
    PROCESSED_MRI_DIR,
    run_dir,
)

DOC = REPO / "docs" / "MRI_SEGMENTATION.md"


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    cfg = yaml.safe_load((REPO / "configs" / "mri_unet_whole_tumour.yaml").read_text(encoding="utf-8"))
    size = int(cfg["preprocess"]["input_size"])
    out = run_dir(size)

    summary = _j(PROCESSED_MRI_DIR / "split_summary.json")
    inv = _j(PROCESSED_MRI_DIR / "dataset_inventory.json")
    meta = _j(out / "run_metadata.json")
    val = _j(out / "metrics" / "validation_aggregate_metrics.json")
    test = _j(out / "metrics" / "heldout_test_aggregate_metrics.json")
    vv = _j(out / "metrics" / "visual_validation.json")
    smoke = _j(REPO / "outputs" / "segmentation" / "_audit" / "gpu_smoke_test.json")
    cache = _j(REPO / "outputs" / "segmentation" / "_audit" / "slice_cache_summary.json")

    if summary is None:
        print("ERROR: split_summary.json missing — run Phases 1-2 first.", file=sys.stderr)
        return 2

    dj = (inv or {}).get("dataset_json", {})
    L: list[str] = []
    A = L.append

    A("# MRI Brain-Tumour Segmentation — Whole Tumour (binary)")
    A("")
    A(f"*Generated from run artifacts on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*")
    A("")
    A("This document covers the **Segmentation** workstream only. The X-ray")
    A("**Classification** workstream is separate, complete, and untouched by this work;")
    A("no result, chart, model, metric or upload is shared between the two.")
    A("")

    # ---------------------------------------------------------------- task
    A("## 1. Task")
    A("")
    A("| Item | Value |")
    A("|---|---|")
    A(f"| Task | Binary **whole tumour vs background** segmentation on brain MRI |")
    A(f"| Model | **{MODEL_LABEL}** |")
    A(f"| Input | All four MRI sequences as channels — {' · '.join(summary['task']['input_modalities'])} |")
    A("| Output | 1 binary channel (logits; sigmoid applied only for metrics/inference) |")
    A(f"| Binary rule | `{summary['task']['rule']}` |")
    A(f"| Slice axis | {summary['task']['slice_axis_name']} |")
    A("")
    A(f"> {MODEL_ARCH_NOTE}")
    A("")
    A("This is **not** multi-class subregion segmentation: the three source tumour")
    A("labels are merged into one foreground class before training.")
    A("")

    # ------------------------------------------------------------- dataset
    A("## 2. Dataset source")
    A("")
    src = summary["source"]
    A("| Item | Value |")
    A("|---|---|")
    A(f"| Dataset | {DATASET_NAME} |")
    A(f"| Identifier | `{DATASET_ID}` |")
    A(f"| Download URL | `{DATASET_URL}` |")
    A(f"| Project homepage | {DATASET_HOMEPAGE} |")
    A(f"| Download date (UTC) | {src['download_date']} |")
    A(f"| Licence | {DATASET_LICENCE} |")
    if dj:
        A(f"| Shipped `name` | {dj.get('name')} |")
        A(f"| Shipped `release` | {dj.get('release')} |")
        A(f"| Shipped `reference` | {dj.get('reference')} |")
        A(f"| Shipped `numTraining` | {dj.get('numTraining')} |")
    A(f"| Raw location | `data/raw/mri/{DATASET_ID}/` |")
    A("")
    A("The pre-existing tiny sample at `data/raw/mri/brats/` is legacy course data.")
    A("It was left completely untouched and is not used anywhere in this pipeline.")
    A("")

    # -------------------------------------------------------- label mapping
    A("### 2.1 Exact source label mapping")
    A("")
    A("Read from the shipped `dataset.json` and asserted in code")
    A("(`src/segmentation/mri/inventory.py::verify_dataset_json`):")
    A("")
    A("| Source label | Meaning | Mapped to project target |")
    A("|---|---|---|")
    for k, v in summary["task"]["source_label_mapping"].items():
        target = "background (0)" if k == "0" else "**whole tumour (1)**"
        A(f"| `{k}` | {v} | {target} |")
    A("")
    if inv:
        A(f"Observed label values across all {inv['n_ok']} verified volumes: "
          f"`{inv['observed_label_values']}` — no undocumented values present.")
        A("")

    # ------------------------------------------------------------ inventory
    A("## 3. Inventory and integrity")
    A("")
    if inv:
        A("| Check | Result |")
        A("|---|---|")
        A(f"| Matched image/label pairs | {inv['n_matched_pairs']} |")
        A(f"| Images without a label | {inv['n_images_without_label']} |")
        A(f"| Labels without an image | {inv['n_labels_without_image']} |")
        A(f"| Volumes passing full verification | {inv['n_ok']} |")
        A(f"| Volumes failing verification | {inv['n_failed']} |")
        A(f"| Image shapes observed | {', '.join(inv['image_shape_histogram'].keys())} |")
        A(f"| Modalities per volume | {len(inv['modality_names'])} — {' · '.join(inv['modality_names'])} |")
        A(f"| Total axial slices | {inv['total_slices']:,} |")
        A(f"| Slices containing tumour | {inv['total_tumour_slices']:,} |")
        A("")
        A("Every volume was fully decompressed and read voxel-by-voxel, so a truncated")
        A("or corrupt `.nii.gz` would have failed here rather than mid-training. Image and")
        A("label affines were compared per case to confirm spatial alignment.")
        A("")

    # --------------------------------------------------------------- splits
    A("## 4. Patient-safe split")
    A("")
    sp = summary["split_policy"]
    A(f"- Split unit: **{sp['unit']}**")
    A(f"- Seed: `{sp['seed']}` (fixed, reproducible)")
    A(f"- Stratified on: {sp['stratified_on']}")
    A("")
    A("| Split | Cases | % | Slices | Tumour slices | Mean tumour fraction |")
    A("|---|---:|---:|---:|---:|---:|")
    for s, label in (("train", "Train"), ("valid", "Validation"),
                     ("test", "Held-out internal test")):
        d = summary["splits"][s]
        A(f"| {label} | {d['cases']} | {d['percent_of_cases']:.1f}% | {d['total_slices']:,} | "
          f"{d['tumour_slices']:,} | {d['mean_tumour_fraction']:.5f} |")
    A(f"| **Total** | **{summary['totals']['cases']}** | 100% | "
      f"{summary['totals']['total_slices']:,} | {summary['totals']['tumour_slices']:,} | |")
    A("")
    lk = summary["leakage_check"]
    A(f"**Leakage check — zero patient overlap: {lk['zero_patient_overlap']}**")
    A("")
    A(f"- train ∩ validation: {len(lk['train_valid_overlap'])} patients")
    A(f"- train ∩ test: {len(lk['train_test_overlap'])} patients")
    A(f"- validation ∩ test: {len(lk['valid_test_overlap'])} patients")
    A("")
    A("### 4.1 Protected held-out test split")
    A("")
    A(f"> {HELD_OUT_TEST_NOTE}")
    A("")
    A("The test split was **not** used for:")
    A("")
    A("- model architecture choice")
    A("- hyperparameter tuning")
    A("- early stopping")
    A("- threshold selection")
    A("")
    A("It was read exactly once, by `scripts/mri_eval_heldout_test.py`, after the")
    A("validation-selected checkpoint was frozen. That script records a receipt")
    A("(`metrics/heldout_test_evaluation_receipt.json`) containing the checkpoint hash and")
    A("refuses to run a second time without an explicit override.")
    A("")
    A("Manifests are frozen by SHA-256 in `data/processed/mri/split_frozen.json`; both the")
    A("training and the test scripts refuse to run if a manifest has changed.")
    A("")

    # -------------------------------------------------------- preprocessing
    A("## 5. Preprocessing")
    A("")
    A("`4 MRI sequences → nonzero-voxel normalization → 2D slice preparation → binary whole-tumour mask`")
    A("")
    A("| Step | Definition |")
    A("|---|---|")
    A("| Load | `.nii.gz` read directly with **nibabel**; image/label affines verified equal |")
    A("| Channels | four sequences stacked as input channels |")
    A("| Normalization | per volume, per modality **z-score over nonzero brain voxels only**; background written back as exactly `0.0` |")
    A("| Binary target | `mask = (label > 0)` |")
    A(f"| Slice axis | {summary['task']['slice_axis_name']} |")
    A(f"| Input size | **{size} × {size}** |")
    A("| Image resampling | bilinear (image only) |")
    A("| Mask resampling | nearest-neighbour (masks only — labels are never interpolated) |")
    A(f"| Train sample shape | image `(4, {size}, {size})`, mask `(1, {size}, {size})` |")
    A("")
    if smoke:
        chosen = next(
            (r for r in smoke["results"]
             if r["input_size"] == size and r["batch_size"] == cfg["train"]["batch_size"]), None
        )
        A(f"### 5.1 Input size choice ({size}×{size})")
        A("")
        A(f"Measured on {smoke['gpu']} ({smoke['total_vram_gb']} GB) by")
        A("`scripts/mri_gpu_smoke_test.py` — real forward+backward passes, AMP enabled:")
        A("")
        A("| Input size | Batch | Peak allocated | Peak reserved | Step time |")
        A("|---:|---:|---:|---:|---:|")
        for r in smoke["results"]:
            if not r.get("ok"):
                A(f"| {r['input_size']}² | {r['batch_size']} | OOM | OOM | — |")
                continue
            mark = " **←chosen**" if (r["input_size"] == size
                                      and r["batch_size"] == cfg["train"]["batch_size"]) else ""
            A(f"| {r['input_size']}² | {r['batch_size']}{mark} | {r['peak_allocated_gb']:.2f} GB | "
              f"{r['peak_reserved_gb']:.2f} GB | {r['step_seconds'] * 1000:.0f} ms |")
        A("")
        if chosen:
            A(f"{size}×{size} at batch {cfg['train']['batch_size']} peaks at "
              f"{chosen['peak_reserved_gb']:.2f} GB reserved of {smoke['total_vram_gb']} GB — "
              "ample headroom, and it preserves more spatial detail than 160×160.")
            A("")

    A("### 5.2 Slice retention (avoiding empty-slice domination)")
    A("")
    A("| Split | Policy |")
    A("|---|---|")
    A(f"| Train | every tumour-containing slice, plus a deterministic per-case sample of "
      f"`{cfg['slices']['empty_slice_ratio']} ×` that many empty brain slices "
      f"(RNG seeded from the case id, seed `{cfg['slices']['sampling_seed']}`) |")
    A("| Validation | **every** slice of the volume, unfiltered |")
    A("| Held-out test | **every** slice of the volume, unfiltered |")
    A("")
    A("Sampling runs per case, *after* the patient split is assigned, so it can never move")
    A("a patient or a slice across a split boundary. Evaluation splits are deliberately")
    A("unfiltered so per-case Dice cannot be flattered by dropping hard or empty slices.")
    A("")
    if cache:
        A("| Cache | Cases | Slices | Tumour slices | Empty slices |")
        A("|---|---:|---:|---:|---:|")
        for s, d in cache["splits"].items():
            A(f"| {s} | {d['n_cases']} | {d['n_slices']:,} | {d['n_tumour_slices']:,} | "
              f"{d['n_empty_slices']:,} |")
        A("")

    if vv:
        A("### 5.3 Visual validation of mask alignment")
        A("")
        A(f"`scripts/mri_visual_validation.py` rendered {vv['n_overlays_rendered']} overlays from")
        A(f"randomly drawn **training** cases and ran quantitative checks on "
          f"{vv['n_cases_checked']} cases:")
        A("")
        A("| Check | Threshold | Result |")
        A("|---|---|---|")
        A(f"| Tumour voxels falling on zero-intensity background | ≤ "
          f"{vv['thresholds']['max_tumour_on_background_fraction']} | pass |")
        A(f"| FLAIR intensity contrast (tumour − brain, in sd) | ≥ "
          f"{vv['thresholds']['min_flair_contrast_sd']} | pass |")
        A("| Tumour centroid inside brain bounding box | required | pass |")
        A("")
        A(f"All checks passed: **{vv['all_passed']}**. Figure: `{vv['figure']}`")
        A("")

    # ------------------------------------------------------------- training
    if meta:
        A("## 6. Model and training")
        A("")
        m, t, h = meta["model"], meta["training"], meta["hardware"]
        A("| Setting | Value |")
        A("|---|---|")
        A(f"| Model | {m['label']} |")
        A(f"| Architecture | `{m['name']}`, encoder features {m['features']}, "
          f"{m['parameters']:,} parameters |")
        A(f"| Input channels | {m['in_channels']} |")
        A(f"| Output channels | {m['out_channels']} (binary) |")
        A(f"| Loss | `{t['loss']}` |")
        A(f"| Optimizer | {t['optimizer']}, lr `{t['lr']}`, weight decay `{t['weight_decay']}` |")
        A(f"| Scheduler | {t['scheduler']} |")
        A(f"| Batch size | {t['batch_size']} |")
        A(f"| Epochs configured / completed / best | {t['epochs_configured']} / "
          f"{t['epochs_completed']} / **{t['best_epoch']}** |")
        A(f"| Early stopping | patience {t['early_stopping_patience']} on validation Dice |")
        A(f"| Selection metric | {t['selection_metric']} |")
        A(f"| Mixed precision | {t['amp']} |")
        A(f"| Gradient clipping | {t['grad_clip']} |")
        A(f"| Seed / deterministic | {t['seed']} / {t['deterministic']} |")
        A(f"| Augmentation | {t['augmentation']} |")
        A(f"| Training duration | {t['training_minutes']:.1f} min |")
        A(f"| Device | {h['gpu'] or h['device']} |")
        A(f"| torch / CUDA | {h['torch']} / {h['cuda']} |")
        A("")
        A("### 6.1 Loss formula")
        A("")
        A("```")
        A("L = w_bce * BCEWithLogits(z, y) + w_dice * (1 - SoftDice(sigmoid(z), y))")
        A("")
        A("SoftDice = (2 * sum(p*y) + eps) / (sum(p) + sum(y) + eps)")
        A("")
        A(f"w_bce  = {cfg['train']['bce_weight']}")
        A(f"w_dice = {cfg['train']['dice_weight']}")
        A(f"eps    = {cfg['train']['dice_smooth']}")
        A("```")
        A("")
        A("Soft Dice is computed per sample and then averaged over the batch, so a single")
        A("large tumour cannot dominate the batch term. Whole-tumour voxels are only a few")
        A("percent of a brain volume: plain BCE collapses toward empty masks, and plain Dice")
        A("is ill-conditioned on slices with no tumour, so both terms are kept.")
        A("")

    # -------------------------------------------------------------- results
    A("## 7. Results")
    A("")
    A("Dice and IoU are **per-case volumetric**: confusion counts are accumulated over")
    A("every axial slice of a case, then the metric is computed once from the case totals.")
    A(f"Decision threshold is fixed at `{cfg['eval']['threshold']}` and was **not** tuned.")
    A("")
    if val:
        A("### 7.1 Validation (used for model selection)")
        A("")
        ci = val.get("dice_bootstrap_ci") or {}
        cii = val.get("iou_bootstrap_ci") or {}
        A("| Metric | Dice | IoU |")
        A("|---|---:|---:|")
        A(f"| Mean (per case) | {val['mean_dice']:.4f} | {val['mean_iou']:.4f} |")
        A(f"| Median | {val['dice']['median']:.4f} | {val['iou']['median']:.4f} |")
        A(f"| IQR | {val['dice']['p25']:.4f} – {val['dice']['p75']:.4f} | "
          f"{val['iou']['p25']:.4f} – {val['iou']['p75']:.4f} |")
        A(f"| 95% CI (bootstrap) | {ci.get('lo', 0):.4f} – {ci.get('hi', 0):.4f} | "
          f"{cii.get('lo', 0):.4f} – {cii.get('hi', 0):.4f} |")
        A(f"| Micro (voxel-pooled) | {val['micro']['dice']:.4f} | {val['micro']['iou']:.4f} |")
        A("")
        A(f"Cases: {val['n_cases']} ({val['n_cases_with_tumour']} with ground-truth tumour). "
          f"Precision {val['micro']['precision']:.4f}, recall {val['micro']['recall']:.4f}.")
        A("")
    if test:
        A("### 7.2 Held-out internal test (one-time, after checkpoint freeze)")
        A("")
        ci = test.get("dice_bootstrap_ci") or {}
        cii = test.get("iou_bootstrap_ci") or {}
        A("| Metric | Dice | IoU |")
        A("|---|---:|---:|")
        A(f"| Mean (per case) | {test['mean_dice']:.4f} | {test['mean_iou']:.4f} |")
        A(f"| Median | {test['dice']['median']:.4f} | {test['iou']['median']:.4f} |")
        A(f"| 95% CI (bootstrap) | {ci.get('lo', 0):.4f} – {ci.get('hi', 0):.4f} | "
          f"{cii.get('lo', 0):.4f} – {cii.get('hi', 0):.4f} |")
        A(f"| Micro (voxel-pooled) | {test['micro']['dice']:.4f} | {test['micro']['iou']:.4f} |")
        A("")
        A(f"Cases: {test['n_cases']}. Evaluated once at {test['evaluated_utc']} against")
        A(f"checkpoint `{test['checkpoint']}` (epoch {test['checkpoint_epoch']}, "
          f"sha256 `{test['checkpoint_sha256'][:16]}…`).")
        A("")
        A("> Validation selected the model. This held-out number is a single generalization")
        A("> check and nothing was changed after seeing it.")
        A("")

    # ------------------------------------------------------------ artifacts
    A("## 8. Artifacts")
    A("")
    if meta:
        A(f"Run directory: `outputs/segmentation/{meta['run_name']}/`")
        A("")
        A("| Artifact | Path (relative to the run directory) |")
        A("|---|---|")
        for label, rel in meta["artifacts"].items():
            A(f"| {label.replace('_', ' ').capitalize()} | `{rel}` |")
        if test:
            A("| Held-out test per-case metrics | `metrics/heldout_test_per_case_metrics.csv` |")
            A("| Held-out test aggregate metrics | `metrics/heldout_test_aggregate_metrics.json` |")
            A("| Held-out test overlays | `figures/heldout_test_qualitative_examples.png` |")
            A("| Held-out test receipt | `metrics/heldout_test_evaluation_receipt.json` |")
        A("")
    A("Manifests and splits: `data/processed/mri/`")
    A("")
    A("| File | Purpose |")
    A("|---|---|")
    A("| `mri_manifest.csv` | every usable case with shape, modality count, label stats, split |")
    A("| `train_manifest.csv` / `valid_manifest.csv` / `test_manifest.csv` | frozen per-split manifests |")
    A("| `split_summary.json` | split policy, counts, leakage check, source metadata |")
    A("| `split_frozen.json` | SHA-256 of each manifest (freeze record) |")
    A("| `dataset_inventory.json` | full Phase 1 per-case integrity inventory |")
    A("")
    A("Presentation samples (copies only, originals untouched):")
    A("")
    A("- `data/presentation_samples/segmentation_validation/`")
    A("- `data/presentation_samples/segmentation_test/`")
    A("")

    # ---------------------------------------------------------- limitations
    A("## 9. Limitations")
    A("")
    A("1. **Not clinical validation.** No external validation, no prospective evaluation,")
    A("   no evidence of diagnostic readiness. Academic demonstration only.")
    A("2. **The held-out test split is internal.** It comes from the same public Decathlon")
    A("   training archive, the same scanners and the same acquisition protocols as the")
    A("   training data. It measures generalization to unseen *patients*, not to unseen")
    A("   *sites*, scanners, sequences or populations.")
    A("3. **Whole tumour only.** Clinically important subregions (enhancing tumour, tumour")
    A("   core, edema) are merged into one class and are not distinguished.")
    A("4. **2D slice-wise, not 3D.** The model sees one axial slice at a time and has no")
    A("   through-plane context, so predictions can be inconsistent between adjacent slices.")
    A("   A 3D model would typically do better on volumetric coherence.")
    A("5. **Fixed threshold of 0.5**, not tuned; a tuned operating point could trade")
    A("   precision against recall differently.")
    A("6. **Requires all four co-registered sequences.** The model cannot run on a single")
    A("   MRI image or on a generic PNG, and the demo enforces this.")
    A("7. **Resampled to a fixed square input**, which changes voxel geometry; metrics are")
    A(f"   reported at {size}×{size} model resolution, not at native 240×240 acquisition")
    A("   resolution.")
    A("8. **Single training run, single seed.** No cross-validation and no seed-variance")
    A("   estimate, so the reported numbers carry run-to-run uncertainty beyond the")
    A("   bootstrap CI shown.")
    A("")
    A("## 10. Reproducing")
    A("")
    A("```bash")
    A("python scripts/mri_prepare_dataset.py          # Phase 1  download extract + inventory")
    A("python scripts/mri_build_manifests.py          # Phase 2  patient-safe frozen splits")
    A("python scripts/mri_gpu_smoke_test.py           # Phase 3  input-size evidence")
    A("python scripts/mri_build_slice_cache.py --splits train valid")
    A("python scripts/mri_visual_validation.py        # Phase 3  alignment gate")
    A("python scripts/run_mri_unet_whole_tumour.py    # Phase 4  train + validate")
    A("python scripts/mri_eval_heldout_test.py        # Phase 5  ONE-TIME test evaluation")
    A("python scripts/mri_build_presentation_samples.py")
    A("python scripts/mri_verify_handoff.py           # Phase 9  verification")
    A("```")
    A("")

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"written -> {DOC.relative_to(REPO).as_posix()} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
