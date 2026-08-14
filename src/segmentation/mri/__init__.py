"""MRI whole-tumour segmentation (Medical Segmentation Decathlon Task01_BrainTumour).

This subpackage is **new and self-contained**. It does not import, modify, or
depend on the legacy PNG/TIFF segmentation modules in the parent package, and it
shares nothing with the X-ray classification workstream.

Model: 2D slice-wise U-Net using four MRI sequences (FLAIR, T1w, T1gd, T2w),
four input channels, one binary output channel (whole tumour vs background).

Flow::

    inventory -> patient-level manifests/splits -> nonzero-voxel z-score
      -> 2D slice sampling -> U-Net (Dice+BCE) -> validation selection
      -> one-time held-out internal test
"""

from .constants import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_URL,
    MODALITY_NAMES,
    SOURCE_LABELS,
    WHOLE_TUMOUR_RULE,
    MODEL_LABEL,
    SLICE_AXIS,
    SLICE_AXIS_NAME,
)

__all__ = [
    "DATASET_ID",
    "DATASET_NAME",
    "DATASET_URL",
    "MODALITY_NAMES",
    "SOURCE_LABELS",
    "WHOLE_TUMOUR_RULE",
    "MODEL_LABEL",
    "SLICE_AXIS",
    "SLICE_AXIS_NAME",
]
