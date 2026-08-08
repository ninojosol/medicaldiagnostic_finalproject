"""Project path resolution.

Every path in the config files may be written relative to the project root, so
notebooks behave the same whether they are launched from ``notebooks/`` or from
the repository root.
"""

from __future__ import annotations

from pathlib import Path

# src/common/paths.py -> src/common -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLS_OUTPUT_DIR = OUTPUTS_DIR / "classification"
SEG_OUTPUT_DIR = OUTPUTS_DIR / "segmentation"


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Return an absolute path, resolving relative paths against the project root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return ((base or PROJECT_ROOT) / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it as an absolute Path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dirs(root: str | Path, run_name: str) -> dict[str, Path]:
    """Create and return the standard output sub-directories for one experiment run.

    Layout::

        <root>/<run_name>/models      trained weights (git-ignored)
        <root>/<run_name>/metrics     CSV / JSON metric files
        <root>/<run_name>/figures     PNG figures for the report
        <root>/<run_name>/predictions example predictions / predicted masks
    """
    base = ensure_dir(Path(root) / run_name)
    dirs = {
        "root": base,
        "models": ensure_dir(base / "models"),
        "metrics": ensure_dir(base / "metrics"),
        "figures": ensure_dir(base / "figures"),
        "predictions": ensure_dir(base / "predictions"),
    }
    return dirs
