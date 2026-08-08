"""Device selection and a short human-readable environment description."""

from __future__ import annotations

import platform
import sys

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return the torch device to train on.

    ``prefer`` may be ``"auto"``, ``"cuda"`` or ``"cpu"``. ``"auto"`` picks CUDA
    when available and silently falls back to CPU otherwise, so the notebooks run
    end-to-end on a laptop without a GPU (just slower).
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "config requested device='cuda' but torch.cuda.is_available() is False.\n"
                "  Fix: install a CUDA build of PyTorch, or set device: auto / cpu in the config."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device | None = None) -> dict:
    """Collect environment facts worth recording alongside every experiment."""
    device = device or get_device()
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        idx = device.index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(idx)))
        info["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(idx).total_memory / 1024**3, 2
        )
    return info
