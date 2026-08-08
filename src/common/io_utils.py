"""Saving metrics, figures and run metadata in a consistent, report-ready way."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .paths import ensure_dir, resolve_path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _jsonable(obj: Any) -> Any:
    """Make numpy / pathlib / torch scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:  # 0-dim torch tensor
        return obj.item()
    return obj


def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    out = resolve_path(path)
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(dict(data)), fh, indent=2)
    print(f"[saved] {out}")
    return out


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    out = resolve_path(path)
    ensure_dir(out.parent)
    df.to_csv(out, index=index)
    print(f"[saved] {out}  ({len(df)} rows)")
    return out


def save_figure(fig, path: str | Path, dpi: int = 150, close: bool = True) -> Path:
    """Save a matplotlib figure at report quality with tight bounding box."""
    out = resolve_path(path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return out


def save_run_metadata(path: str | Path, config: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> Path:
    """Record config + environment + timestamp so a run can be traced later."""
    from .device import describe_device

    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": describe_device(),
        "config": dict(config),
    }
    if extra:
        payload.update(dict(extra))
    return save_json(payload, path)


def append_history_row(history: list[dict], **kwargs) -> list[dict]:
    """Append one epoch of training metrics to an in-memory history list."""
    history.append({k: _jsonable(v) for k, v in kwargs.items()})
    return history


def history_to_csv(history: Sequence[Mapping[str, Any]], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(list(history))
    save_csv(df, path)
    return df


def list_images(directory: str | Path, recursive: bool = True) -> list[Path]:
    """List image files under a directory, sorted for deterministic ordering."""
    root = Path(directory)
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
