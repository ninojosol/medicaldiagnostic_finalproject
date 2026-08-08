"""Brain MRI tumour segmentation module.

Typical flow (see notebooks 04-06)::

    pair discovery -> pair validation -> patient split -> DataLoaders
      -> U-Net -> Dice+BCE training -> Dice/IoU evaluation -> overlays
"""

from .pairing import build_pairs, validate_pairs
from .splits import patient_level_split_seg, assert_no_patient_leakage, seg_split_summary
from .dataset import MRISegmentationDataset, build_seg_dataloaders, build_seg_transforms, peek_seg_batch
from .unet import UNet, ResNetUNet, build_segmentation_model
from .losses import DiceLoss, DiceBCELoss, TverskyLoss, build_seg_loss, compute_foreground_pos_weight
from .metrics import dice_coefficient, iou_score, per_slice_metrics, summarize_segmentation, bootstrap_dice_ci
from .train import train_segmentation, validate, load_checkpoint
from .evaluate import evaluate_segmentation, predict_masks, plot_prediction_overlays, pick_best_worst

__all__ = [
    "build_pairs", "validate_pairs",
    "patient_level_split_seg", "assert_no_patient_leakage", "seg_split_summary",
    "MRISegmentationDataset", "build_seg_dataloaders", "build_seg_transforms", "peek_seg_batch",
    "UNet", "ResNetUNet", "build_segmentation_model",
    "DiceLoss", "DiceBCELoss", "TverskyLoss", "build_seg_loss", "compute_foreground_pos_weight",
    "dice_coefficient", "iou_score", "per_slice_metrics", "summarize_segmentation", "bootstrap_dice_ci",
    "train_segmentation", "validate", "load_checkpoint",
    "evaluate_segmentation", "predict_masks", "plot_prediction_overlays", "pick_best_worst",
]
