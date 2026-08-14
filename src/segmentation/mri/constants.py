"""Canonical facts, paths and label mapping for the MRI segmentation workstream.

Everything the UI, the training script and the docs quote about the dataset is
defined here once, so no number can drift between code, artifacts and report.

The source label mapping is *not* hardcoded blindly: `scripts/mri_prepare_dataset.py`
reads the shipped `dataset.json` and asserts it matches `SOURCE_LABELS` below.
"""

from __future__ import annotations

from pathlib import Path

# src/segmentation/mri/constants.py -> mri -> segmentation -> src -> repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# --- dataset identity ------------------------------------------------------
DATASET_ID = "decathlon_task01_brain_tumour"
DATASET_NAME = "Medical Segmentation Decathlon — Task01_BrainTumour (BraTS)"
DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
DATASET_HOMEPAGE = "http://medicaldecathlon.com/"
DATASET_LICENCE = "CC BY-SA 4.0 (Medical Segmentation Decathlon)"

# --- filesystem ------------------------------------------------------------
RAW_MRI_DIR = PROJECT_ROOT / "data" / "raw" / "mri" / DATASET_ID
RAW_DOWNLOAD_DIR = RAW_MRI_DIR / "_download"
IMAGES_TR = RAW_MRI_DIR / "imagesTr"
LABELS_TR = RAW_MRI_DIR / "labelsTr"
DATASET_JSON = RAW_MRI_DIR / "dataset.json"

PROCESSED_MRI_DIR = PROJECT_ROOT / "data" / "processed" / "mri"
MANIFEST_CSV = PROCESSED_MRI_DIR / "mri_manifest.csv"
TRAIN_MANIFEST_CSV = PROCESSED_MRI_DIR / "train_manifest.csv"
VALID_MANIFEST_CSV = PROCESSED_MRI_DIR / "valid_manifest.csv"
TEST_MANIFEST_CSV = PROCESSED_MRI_DIR / "test_manifest.csv"
SPLIT_SUMMARY_JSON = PROCESSED_MRI_DIR / "split_summary.json"
INVENTORY_JSON = PROCESSED_MRI_DIR / "dataset_inventory.json"

SEG_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "segmentation"
PRESENTATION_ROOT = PROJECT_ROOT / "data" / "presentation_samples"
PRESENTATION_VALID_DIR = PRESENTATION_ROOT / "segmentation_validation"
PRESENTATION_TEST_DIR = PRESENTATION_ROOT / "segmentation_test"

# --- modalities (channel order as stored in the 4th axis of imagesTr) ------
# Display spellings. The shipped dataset.json writes channel 2 as lowercase
# "t1gd"; SOURCE_MODALITY_NAMES keeps the source strings verbatim for the report.
MODALITY_NAMES: tuple[str, ...] = ("FLAIR", "T1w", "T1gd", "T2w")
SOURCE_MODALITY_NAMES: tuple[str, ...] = ("FLAIR", "T1w", "t1gd", "T2w")
N_MODALITIES = len(MODALITY_NAMES)

# --- source label mapping, verbatim from the shipped dataset.json ----------
# Reproduced exactly as the source writes it, including the source's own
# inconsistent spelling ("tumor" for label 2, "tumour" for label 3).
SOURCE_LABELS: dict[int, str] = {
    0: "background",
    1: "edema",
    2: "non-enhancing tumor",
    3: "enhancing tumour",
}

# --- project target --------------------------------------------------------
# Binary whole tumour vs background. Every non-background source label is merged.
WHOLE_TUMOUR_RULE = "mask = (label > 0)"
TUMOUR_SOURCE_LABELS: tuple[int, ...] = (1, 2, 3)
TARGET_CLASSES: tuple[str, ...] = ("background", "whole tumour")

# --- model identity (must read identically everywhere) ---------------------
MODEL_LABEL = "2D slice-wise U-Net using four MRI sequences"
MODEL_ARCH_NOTE = (
    "Slice-wise 2D U-Net — 4 input channels (one per MRI sequence), "
    "1 binary output channel (whole tumour). This is not a 3D U-Net."
)

# --- slicing ---------------------------------------------------------------
# Volumes are (H=240, W=240, D=155, C=4) in RAS-ish index space. Axis 2 is the
# axial/slice axis: each 2D sample is one axial slice of shape (4, 240, 240).
SLICE_AXIS = 2
SLICE_AXIS_NAME = "axial (index axis 2 of the 240x240x155 volume)"

EXPECTED_VOLUME_SHAPE = (240, 240, 155, 4)
EXPECTED_LABEL_SHAPE = (240, 240, 155)

# --- split policy ----------------------------------------------------------
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "valid": 0.15, "test": 0.15}

HELD_OUT_TEST_NOTE = (
    "Project-created internal held-out test split drawn from the same public "
    "Decathlon training archive. It is NOT an external or clinical test set."
)


def run_dir_name(input_size: int) -> str:
    """Canonical run directory name for a given square input size."""
    return f"mri_unet_whole_tumour_2d_{input_size}"


def run_dir(input_size: int) -> Path:
    return SEG_OUTPUT_ROOT / run_dir_name(input_size)
