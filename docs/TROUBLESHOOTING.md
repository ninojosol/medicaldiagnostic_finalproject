# Troubleshooting

---

## Environment

### `ModuleNotFoundError: No module named 'src'`

The bootstrap cell at the top of every notebook adds the project root to
`sys.path`. If it fails:

- run it first — it is always the second cell;
- launch Jupyter from the project root (`jupyter lab` in
  `medicaldiagnostic_finalproject/`), not from a parent directory;
- confirm the kernel is *Python 3 (medicaldiagnostic)* — a different kernel means
  a different environment.

### `ModuleNotFoundError: No module named 'torch'` (or any other package)

The kernel is not the virtual environment. Check with:

```python
import sys; print(sys.executable)
```

It must point inside `.venv`. If not, re-register the kernel:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ipykernel install --user --name medicaldiagnostic --display-name "Python 3 (medicaldiagnostic)"
```

then restart Jupyter and re-select the kernel.

### `torch.cuda.is_available()` is `False` on a machine with an NVIDIA GPU

Check which build you have:

```powershell
python -c "import torch; print(torch.__version__)"
```

`2.x.x+cpu` means the CPU-only wheel. `2.x.x+cu128` is what you want.

**The usual cause:** a `pip install` pulled `torch` from PyPI, whose Windows
wheel is CPU-only, and it replaced your CUDA build. Anything that lists `torch`
as a dependency will do this — which is exactly why `requirements.txt` in this
project does **not** list it. Do not add it back.

Fix:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check the driver separately with `nvidia-smi`.

### `OSError: [WinError 5] Access is denied: ...\torch\_C.pyd` while installing

A process is holding the PyTorch DLL open, so pip cannot replace it. Close every
Jupyter kernel, notebook server and Python REPL that has imported torch — on
Windows a running kernel locks the file even when idle — then retry. If it
persists, delete `.venv\Lib\site-packages\torch` manually and reinstall.

### `SSLError(SSLEOFError(...))` during the PyTorch download

The connection dropped part-way through the ~3 GB wheel. Retry with more
patience:

```powershell
pip install --retries 10 --timeout 180 torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### `CUDA error: no kernel image is available for execution on the device`

The PyTorch build does not support your GPU's compute capability. RTX 50-series
(Blackwell, `sm_120`) needs the **cu128** wheels or newer. Verify:

```python
import torch
print(torch.version.cuda, torch.cuda.get_device_capability())
```

---

## Data loading

### `DataNotFoundError: ... directory not found`

The message prints the path it looked for, the nearest existing directory and
what that directory contains. Either move the data, or change `data.root` in the
config. Paths in configs are relative to the **project root**, not the notebook.

### `DataLayoutError: No images matched the configured labels [...]`

`data.labels` must match your class folder names (matching is
case-insensitive). The error lists the folder names it actually found — copy them
into `data.labels`.

### `DataLayoutError: Could not build labels from the CSV`

For `layout: csv`, either:

- the CSV has one 0/1 column per entry in `data.labels`, **or**
- `data.label_column` names a column holding the class name or
  separator-joined findings (`"Effusion|Infiltration"`), with
  `data.label_separator` set correctly.

The error prints every column in your CSV.

### `DataLayoutError: could not pair ANY of them with a mask`

The mask naming does not match `data.mask_suffix`. The error shows an example
image name and the files in its folder. Common causes:

- masks use a different suffix (`_seg`, `-mask`, `_label`) → set
  `data.mask_suffix`;
- masks live in a separate folder → switch to `layout: parallel` and set
  `data.image_dir` / `data.mask_dir`;
- masks have a different extension from the images — handled automatically, but
  the stem must still match.

### Only 1 patient, or patients == number of images

No patient ID was extracted. Set `data.patient_id_regex` with **one capturing
group** (see `data/README.md`). If the dataset genuinely has no patient
information, say so in the report: the split is per-image and leakage cannot be
ruled out.

### `PATIENT LEAKAGE DETECTED between splits`

This is the check working. It usually means a manually edited manifest, or two
different patients sharing an ID. Rebuild the split by re-running notebook 01 or
04 from the top.

---

## Training

### `CUDA out of memory`

In this order:

1. lower `train.batch_size` (32 → 16 → 8 → 4);
2. lower `data.image_size` (256 → 128);
3. for segmentation, halve `model.features` → `[32, 64, 128, 256]`;
4. restart the kernel — a previous run's model may still hold GPU memory.

Note whatever you changed in the report; it affects the results.

### Training loss is `nan`

- lower `train.lr` by 10×;
- confirm `train.grad_clip: 1.0` is set (it is, in `base.yaml`);
- for classification, check `pos_weight` is not enormous — the cap is 20× by
  default but a label with a single positive example is still a problem;
- disable mixed precision (`train.amp: false`) to see whether FP16 is the cause.

### Validation ROC-AUC is `nan`

The validation split has only one class for that label. Either the label is too
rare or the split is too small. Increase `data.val_size`, or drop labels with
almost no positives from `data.labels`.

### Dice stays near 0 for many epochs

Check in this order:

1. **pairing** — re-run the validation cell in notebook 04; a mis-paired dataset
   trains "successfully" with Dice ≈ 0;
2. **augmentation alignment** — the check cell in notebook 05 must show the mask
   following the image through every transform;
3. **mask values** — `peek_seg_batch` must report masks with unique values
   `[0.0, 1.0]`;
4. **learning rate** — segmentation typically needs `1e-3`, higher than
   classification fine-tuning;
5. give it more epochs — Dice often sits near 0 for several epochs before the
   model finds the foreground.

### Training is extremely slow

- confirm section 3 of notebook 00 reports CUDA available;
- on Windows, keep `train.num_workers: 0` — worker processes usually make things
  *slower* under Jupyter on Windows, not faster;
- reduce `data.image_size` for a first pass;
- use the `SMOKE_TEST` switch while developing.

### Model overfits immediately (train loss ↓, validation loss ↑)

Expected for the from-scratch baseline on a small dataset — it is part of what
the comparison is meant to demonstrate. To reduce it: increase augmentation
strength in the config, raise `model.dropout`, raise `train.weight_decay`, or use
the pretrained model. Early stopping already prevents the over-fitted epoch from
being reported.

---

## Evaluation

### `DataNotFoundError: Checkpoint not found`

Notebook 03 and 06 load `outputs/<task>/<run_name>/models/<run_name>_best.pt`.
If `run_name` in the config changed after training, the path no longer matches —
either revert it or point to the correct checkpoint.

A `_best.pt` is written only when validation improved at least once. If training
was a single epoch with no improvement recorded, use `_last.pt`.

### Grad-CAM heat maps are blank or uniform

- The class had no positive evidence: ReLU zeroed the map. Try a different image
  or `gradcam.class_index`.
- The target layer is wrong for a custom model — see
  `get_gradcam_target_layer` in `src/classification/models.py`.
- The model was in `train()` mode; `GradCAM.__call__` sets `eval()` itself, so
  this should not happen with the provided code.

### Bootstrap CI comes back as `n/a`

Too few valid resamples — usually a test set with very few positives, where most
resamples contain only one class and AUC is undefined. Report the point estimate
and state that the test set was too small for a reliable interval.

### Metrics look suspiciously high (ROC-AUC > 0.99, Dice > 0.95)

Treat this as a bug until proven otherwise:

- re-check the leakage output in notebook 01 / 04;
- confirm `data.patient_id_regex` extracts real patient IDs, not per-image ones;
- check for duplicate images across splits (`check_manifest` reports duplicate
  paths, but not visually identical files with different names);
- confirm the threshold was chosen on validation, not test (it is, in the
  provided code — but check any code you added).

---

## Notebooks

### `NameError` after editing a file in `src/`

`%autoreload 2` picks up most edits, but not new top-level definitions in a
module already imported, and not changes to class hierarchies. Restart the kernel
and re-run.

### Figures do not appear

`save_figure(..., close=False)` keeps the figure open for display; with
`close=True` (the default in some helpers) it is saved and closed. Everything is
written to `outputs/<task>/<run>/figures/` regardless.

### `tqdm` progress bars render as raw text in JupyterLab

```powershell
pip install ipywidgets
```

then restart the kernel. Or set `train.progress: false` in the config.
