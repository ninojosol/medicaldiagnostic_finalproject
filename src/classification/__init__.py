"""Chest X-ray disease classification module.

Typical flow (see notebooks 01-03)::

    manifest -> patient-level split -> DataLoaders -> model -> train -> evaluate -> Grad-CAM
"""

from .manifest import build_manifest, check_manifest, image_size_report
from .splits import patient_level_split, use_existing_split, assert_no_patient_leakage, split_summary
from .dataset import XrayDataset, build_dataloaders, compute_pos_weights, peek_batch
from .transforms import build_transforms
from .models import SimpleCNN, build_model, build_transfer_model, get_gradcam_target_layer
from .losses import build_loss
from .train import train_model, predict, load_checkpoint
from .evaluate import evaluate_classifier, compare_runs
from .gradcam import GradCAM, gradcam_panel, pick_examples

__all__ = [
    "build_manifest", "check_manifest", "image_size_report",
    "patient_level_split", "use_existing_split", "assert_no_patient_leakage", "split_summary",
    "XrayDataset", "build_dataloaders", "compute_pos_weights", "peek_batch",
    "build_transforms",
    "SimpleCNN", "build_model", "build_transfer_model", "get_gradcam_target_layer",
    "build_loss",
    "train_model", "predict", "load_checkpoint",
    "evaluate_classifier", "compare_runs",
    "GradCAM", "gradcam_panel", "pick_examples",
]
