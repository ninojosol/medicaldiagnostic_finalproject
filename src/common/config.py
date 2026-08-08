"""YAML configuration loading with dot-access and light validation.

All tunable values (paths, image size, batch size, epochs, learning rate, seed,
model name, labels) live in ``configs/*.yaml`` so notebooks stay free of magic
numbers and every experiment can be reproduced from its config file alone.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from .errors import ProjectError, require_file
from .paths import CONFIG_DIR, PROJECT_ROOT, resolve_path


class Config(Mapping):
    """Read-mostly nested config with attribute and dict access.

    Examples
    --------
    >>> cfg = load_config("xray_transfer.yaml")           # doctest: +SKIP
    >>> cfg.train.epochs                                  # doctest: +SKIP
    >>> cfg["data"]["image_size"]                         # doctest: +SKIP
    >>> cfg.get("train.lr", 1e-4)                         # doctest: +SKIP
    """

    def __init__(self, data: dict[str, Any] | None = None, source: Path | None = None):
        self._data: dict[str, Any] = deepcopy(data or {})
        self._source = source

    # -- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key '{key}'. Available keys: {sorted(self._data)}"
            ) from exc

    def __repr__(self) -> str:
        where = f" from {self._source.name}" if self._source else ""
        return f"Config{where}({json.dumps(self._data, indent=2, default=str)})"

    # -- convenience ------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Fetch a value with a dotted path, e.g. ``cfg.get('train.lr', 1e-4)``."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def path(self, key: str) -> Path:
        """Resolve a config value as a project-relative path."""
        value = self.get(key)
        if value is None:
            raise ProjectError(f"Config key '{key}' is missing but a path was requested.")
        return resolve_path(value)

    def save(self, path: str | Path) -> Path:
        """Snapshot the exact config used by a run, next to its outputs."""
        out = resolve_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)
        return out


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(name_or_path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config by file name (looked up in ``configs/``) or by path.

    A config may declare ``inherits: other.yaml`` to reuse a shared base file;
    the child's values win. ``overrides`` is a plain dict merged in last, which
    is how notebooks make one-off changes (e.g. a 2-epoch smoke test).
    """
    candidate = Path(name_or_path)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = CONFIG_DIR / candidate.name
    path = require_file(
        candidate,
        what="Config",
        hint=f"a YAML file in {CONFIG_DIR.relative_to(PROJECT_ROOT)}/ (e.g. xray_transfer.yaml)",
    )

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ProjectError(f"Config {path} must contain a YAML mapping at the top level.")

    parent_name = data.pop("inherits", None)
    if parent_name:
        parent = load_config(parent_name).to_dict()
        data = _deep_merge(parent, data)

    if overrides:
        data = _deep_merge(data, overrides)

    data.setdefault("_config_file", str(path))
    return Config(data, source=path)


def validate_config(cfg: Config, required: list[str]) -> None:
    """Fail fast with one message listing every missing key, not one at a time."""
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ProjectError(
            "Config is missing required keys: "
            + ", ".join(missing)
            + f"\n  Config file: {cfg.get('_config_file', '<in-memory>')}"
        )
