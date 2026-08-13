# Reproducibility

## What the project does to be reproducible

**Seeding.** `seed_everything(seed)` (in `src/common/seed.py`) seeds Python's
`random`, NumPy, PyTorch CPU and PyTorch CUDA, sets `PYTHONHASHSEED`, and —
when `deterministic: true` — requests deterministic cuDNN kernels and calls
`torch.use_deterministic_algorithms(True, warn_only=True)`. Every notebook seeds
immediately after loading its config.

**DataLoader determinism.** Shuffling uses an explicitly seeded
`torch.Generator`, and each worker process is re-seeded through `seed_worker`,
so batch order and augmentation draws repeat between runs.

**Stable splits.** `patient_level_split` assigns patients by hashing
`f"{seed}:{patient_id}"` rather than by shuffling the row order. Adding or
removing images therefore does not reshuffle unrelated patients into different
splits. The resulting assignment is written to
`data/processed/*.csv` and reused by every downstream notebook, so training and
evaluation cannot disagree about the test set.

**Config snapshots.** Every training run writes `config_used.yaml` and
`run_metadata.json` (timestamp, Python/PyTorch/CUDA versions, GPU name, epochs,
best epoch and score) next to its outputs. A result can always be traced back to
the exact configuration and environment that produced it.

**No hidden state.** All tunable values live in `configs/*.yaml`. Notebooks
contain orchestration only; the logic is in `src/`.

---

## What is *not* guaranteed — state these limits in the report

**Bit-exact results across machines.** Different GPU architectures, CUDA/cuDNN
versions, and PyTorch builds select different kernels. Same seed on a different
machine gives close but not identical numbers.

**Determinism with `deterministic: false`.** Setting this enables cuDNN
autotuning, which selects algorithms based on runtime benchmarking. It is
noticeably faster and slightly non-deterministic.

**Some operations have no deterministic implementation.** `warn_only=True` means
those fall back to non-deterministic kernels with a warning rather than crashing
mid-training. Atomic-add-based backward passes are the usual culprit.

**Mixed precision (`train.amp: true`).** FP16 accumulation order varies, so
results differ slightly from an FP32 run. Deterministic across identical runs on
the same hardware, but not identical to CPU or FP32 output.

**Pretrained weights.** `pretrained: true` downloads whatever torchvision serves
as `DEFAULT` for that architecture. Record the torchvision version — a future
version could ship different default weights.

**Multi-worker data loading.** `num_workers: 0` is the default (required on
Windows inside Jupyter anyway) and is the most reproducible setting. Higher
values are seeded per worker but introduce ordering effects at process level.

---

## How to reproduce a specific reported result

1. Find the run directory: `outputs/classification/<run_name>/` or
   `outputs/segmentation/<run_name>/`.
2. Read `run_metadata.json` for the environment (Python, PyTorch, CUDA, GPU) and
   the epoch that produced the best score.
3. Copy `config_used.yaml` over the matching file in `configs/` (or load it
   directly: `load_config("outputs/.../config_used.yaml")`).
4. Re-run notebook 01 (or 04) to rebuild the manifest with the same seed, then
   the training notebook.
5. Expect metrics within a small tolerance, not identical digits, unless you are
   on the exact same hardware and library versions.

---

## Recommended practice for the report

Include a short table like this, filled in from `run_metadata.json` — do **not**
copy the values below, they are placeholders:

| Item | Value |
|---|---|
| Python | *(from run_metadata.json)* |
| PyTorch | *(from run_metadata.json)* |
| CUDA / GPU | *(from run_metadata.json)* |
| Seed | *(from the config)* |
| `deterministic` | *(from the config)* |
| Mixed precision | *(from the config)* |
| Epochs run / best epoch | *(from run_metadata.json)* |

Then add one honest sentence, e.g.:

> Results were produced with a fixed seed and deterministic cuDNN settings.
> Re-running on the same machine reproduces the reported metrics; results on
> different hardware or library versions may differ slightly due to
> non-deterministic GPU kernels and mixed-precision accumulation order.

---

## Which pipeline produced the reported X-ray results

Two pipelines exist in this repository. Only one produced the reported numbers.

| | Reported results | Superseded |
|---|---|---|
| Framework | **PyTorch** | TensorFlow / Keras |
| Code | `src/classification/`, `scripts/run_xray_*.py`, `run_xray_finetune_efficientnet_b0.py` | `notebooks/02_xray_model_baseline.ipynb`, `notebooks/03_xray_model_finetuning.ipynb` |
| Artifacts | `outputs/classification/<run_name>/` | `outputs/models/xray/*.keras` |

Notebooks 02 and 03 are an earlier TensorFlow/Keras exploration, kept for the
record and clearly retitled *"Superseded exploratory experiment — not used for
reported results"*. **They are not the source of any reported metric, figure or
threshold.** Nothing in the Streamlit app reads their outputs.

### Authoritative per-run configuration

For any completed run, the authoritative configuration is the snapshot written
next to its artifacts:

```
outputs/classification/<run_name>/config_used.yaml    # every hyper-parameter, as used
outputs/classification/<run_name>/run_metadata.json   # + environment, best epoch, best score
```

The files in `configs/` are **templates for launching new experiments**. They are
kept aligned with the reported runs, but if the two ever disagree, the snapshot in
`outputs/` is correct — it was written by the run itself.

### The three reported runs

| Model | Run name | Script |
|---|---|---|
| Baseline CNN | `xray_baseline_cnn_from_scratch_multilabel_320` | `scripts/run_xray_baseline_cnn.py` |
| Fine-tuned DenseNet | `xray_finetuned_densenet_multilabel_320` (+ `_stage1_head_only`) | `scripts/run_xray_finetune_densenet.py` |
| Fine-tuned EfficientNet-B0 | `xray_finetuned_efficientnet_b0_multilabel_320` (+ `_stage1_head_only`) | `run_xray_finetune_efficientnet_b0.py` |

The two `scripts/run_xray_*.py` files were **reconstructed** from each run's own
`config_used.yaml` after the originals were lost; they are not byte-verified
replays of the saved checkpoints (see the header of each script). All three
scripts refuse to overwrite an existing run directory unless `--force` is passed,
so the reported artifacts cannot be clobbered by accident.

All three runs are **validation-only**: they build `train` and `val` loaders and
no `test` loader, so `evaluate_classifier` stops after its validation stage and
the official test set is never scored.

### Where the validation curves and confusion matrices come from

`outputs/classification/<run_name>/predictions/validation_predictions.csv` is
written by `evaluate_classifier` from the **validation loader only**. It is the
single source the Streamlit "Train & Validate" page reads for the aggregated
one-vs-rest confusion matrices and the micro-averaged ROC / PR curves.

Schema: `image_path`, `image`, `patient_id`, then one
`true_<label>` / `prob_<label>` / `pred_<label>` triple per finding, in the frozen
NIH-14 label order. `prob_*` is rounded to 6 decimals for readability while
`pred_*` is derived from the full-precision probability, so recompute hard
decisions from `pred_*`, never from the rounded `prob_*`.
