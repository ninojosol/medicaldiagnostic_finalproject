"""Grad-CAM for chest X-ray classifiers - implemented directly with hooks.

How it works
------------
1. Forward the image and keep the activations ``A`` of the last convolutional layer.
2. Backward from the logit of the class of interest and keep the gradients ``dY/dA``.
3. Average those gradients over space to get one importance weight per channel.
4. Weighted sum of channels, ReLU, upsample to image size -> the heat map.

Interpretation warning (put this in the report)
-----------------------------------------------
Grad-CAM shows which image regions most influenced the score. It is **not** a
lesion segmentation and **not** an explanation of clinical reasoning. Highlighted
regions are frequently non-anatomical (text markers, tubes, image borders,
scanner artefacts) - which is exactly why the visualisation is useful: it exposes
shortcut learning. Never present a heat map as evidence that the model "found the
disease".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..common.viz import denormalize
from .transforms import get_norm_stats


class GradCAM:
    """Grad-CAM for a single target layer.

    Use as a context manager (or call :meth:`remove`) so the forward/backward
    hooks do not stay attached to the model afterwards.

    Examples
    --------
    >>> with GradCAM(model, target_layer) as cam:          # doctest: +SKIP
    ...     heatmap = cam(image_tensor, class_index=0)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, image: torch.Tensor, class_index: int = 0) -> np.ndarray:
        """Return an ``(H, W)`` heat map in [0, 1] for a single image tensor ``(1,C,H,W)``."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[0] != 1:
            raise ValueError("GradCAM expects one image at a time, got a batch.")

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Grad-CAM needs gradients, so no torch.no_grad() here.
        image = image.clone().requires_grad_(True)
        logits = self.model(image)
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM captured no activations/gradients.\n"
                "  Fix: check that the target layer is really used in forward() "
                "(see models.get_gradcam_target_layer)."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0,1]; an all-zero map means the class had no positive evidence.
        span = cam.max() - cam.min()
        cam = (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

        if was_training:
            self.model.train()
        return cam


def overlay_heatmap(image_hw3: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4,
                    cmap: str = "jet") -> np.ndarray:
    """Blend a [0,1] heat map over a [0,1] RGB image."""
    import matplotlib

    # matplotlib.colormaps replaced the removed matplotlib.cm.get_cmap in 3.9.
    colored = matplotlib.colormaps[cmap](np.clip(heatmap, 0, 1))[..., :3]
    base = image_hw3 if image_hw3.ndim == 3 else np.stack([image_hw3] * 3, axis=-1)
    return np.clip((1 - alpha) * base + alpha * colored, 0, 1)


def gradcam_panel(model, dataset, indices, target_layer, labels, device,
                  class_index: int = 0, pretrained: bool = True, alpha: float = 0.4,
                  thresholds: dict | float = 0.5):
    """Figure with one row per image: original | heat map | overlay, with true/predicted labels.

    ``dataset`` must be an :class:`~src.classification.dataset.XrayDataset`.
    Returns the matplotlib Figure so the caller can save it.
    """
    import matplotlib.pyplot as plt

    mean, std = get_norm_stats(pretrained)
    label_name = labels[class_index]
    thr = thresholds.get(label_name, 0.5) if isinstance(thresholds, dict) else float(thresholds)

    indices = list(indices)
    fig, axes = plt.subplots(len(indices), 3, figsize=(10.5, 3.5 * len(indices)), squeeze=False)

    with GradCAM(model, target_layer) as cam:
        for row, idx in enumerate(indices):
            item = dataset[idx]
            image_tensor, target = item[0], item[1]
            path = item[2] if len(item) > 2 else dataset.paths[idx]

            batch = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                prob = float(torch.sigmoid(model(batch))[0, class_index])
            heatmap = cam(batch, class_index=class_index)

            display = denormalize(image_tensor, mean, std)
            if display.ndim == 2:
                display = np.stack([display] * 3, axis=-1)

            truth = int(target[class_index].item())
            predicted = int(prob >= thr)
            verdict = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(truth, predicted)]

            axes[row][0].imshow(display)
            axes[row][0].set_title(f"{Path(path).name}\ntrue={truth}  p={prob:.3f}  [{verdict}]", fontsize=9)
            axes[row][1].imshow(heatmap, cmap="jet")
            axes[row][1].set_title(f"Grad-CAM: {label_name}", fontsize=10)
            axes[row][2].imshow(overlay_heatmap(display, heatmap, alpha=alpha))
            axes[row][2].set_title("Overlay", fontsize=10)
            for ax in axes[row]:
                ax.axis("off")

    fig.suptitle(f"Grad-CAM - {label_name} (threshold {thr:.3f})\n"
                 "Region attribution only; NOT a lesion segmentation and NOT a diagnosis.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def pick_examples(y_true: np.ndarray, y_prob: np.ndarray, class_index: int = 0,
                  threshold: float = 0.5, per_group: int = 2) -> dict[str, list[int]]:
    """Select confident TP / TN / FP / FN indices for a balanced qualitative panel.

    Showing failures next to successes is what makes the Grad-CAM section
    honest - a panel of hand-picked correct predictions proves nothing.
    """
    y_true = np.asarray(y_true).reshape(len(y_true), -1)[:, class_index].astype(int)
    y_prob = np.asarray(y_prob).reshape(len(y_prob), -1)[:, class_index]
    y_pred = (y_prob >= threshold).astype(int)

    groups = {
        "TP": np.where((y_true == 1) & (y_pred == 1))[0],
        "TN": np.where((y_true == 0) & (y_pred == 0))[0],
        "FP": np.where((y_true == 0) & (y_pred == 1))[0],
        "FN": np.where((y_true == 1) & (y_pred == 0))[0],
    }
    out = {}
    for name, idx in groups.items():
        if len(idx) == 0:
            out[name] = []
            continue
        # Most confident examples first: highest prob for predicted-positive groups.
        confidence = y_prob[idx] if name in {"TP", "FP"} else 1 - y_prob[idx]
        out[name] = idx[np.argsort(-confidence)][:per_group].tolist()
    return out
