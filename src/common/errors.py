"""Explicit, actionable errors for missing or mis-structured datasets.

The course dataset is added by the student after the code is written, so the most
common failure mode is "folder not where the config says it is". These helpers
turn that into a readable message instead of a deep stack trace.
"""

from __future__ import annotations

from pathlib import Path


class ProjectError(Exception):
    """Base class for all project-level configuration/data errors."""


class DataNotFoundError(ProjectError):
    """A required dataset file or directory does not exist."""


class DataLayoutError(ProjectError):
    """The dataset exists but its structure does not match what the code expects."""


def _preview(path: Path, limit: int = 10) -> str:
    """Show what actually is in the nearest existing parent directory."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        return "  (no existing parent directory found)"
    try:
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in probe.iterdir())
    except OSError as exc:  # pragma: no cover - permission issues
        return f"  (could not list {probe}: {exc})"
    if not entries:
        return f"  {probe} exists but is empty."
    shown = entries[:limit]
    more = "" if len(entries) <= limit else f"  ... and {len(entries) - limit} more"
    return f"  Nearest existing directory: {probe}\n  Contains: {', '.join(shown)}{more}"


def require_dir(path: str | Path, what: str, hint: str = "") -> Path:
    """Return ``path`` as a Path, or raise a helpful DataNotFoundError."""
    p = Path(path)
    if p.is_dir():
        return p
    msg = [f"{what} directory not found: {p}"]
    if hint:
        msg.append(f"  Expected: {hint}")
    msg.append(_preview(p))
    msg.append("  Fix: place the dataset there, or update the path in your config file (configs/*.yaml).")
    raise DataNotFoundError("\n".join(msg))


def require_file(path: str | Path, what: str, hint: str = "") -> Path:
    """Return ``path`` as a Path, or raise a helpful DataNotFoundError."""
    p = Path(path)
    if p.is_file():
        return p
    msg = [f"{what} file not found: {p}"]
    if hint:
        msg.append(f"  Expected: {hint}")
    msg.append(_preview(p))
    msg.append("  Fix: place the file there, or update the path in your config file (configs/*.yaml).")
    raise DataNotFoundError("\n".join(msg))


def require_nonempty(items, what: str, where: str | Path, hint: str = ""):
    """Raise DataLayoutError when a scan produced nothing."""
    if len(items) == 0:
        msg = [f"No {what} found under: {where}"]
        if hint:
            msg.append(f"  Expected: {hint}")
        msg.append(_preview(Path(where)))
        raise DataLayoutError("\n".join(msg))
    return items
