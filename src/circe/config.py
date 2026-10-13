from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    root = project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    roi_file = config.get("roi_file")
    if roi_file:
        roi_path = resolve_path(root, roi_file)
        if roi_path.exists():
            with roi_path.open("r", encoding="utf-8") as handle:
                stored_rois = yaml.safe_load(handle) or {}
            for group in ("references", "queries"):
                config.setdefault("roi", {}).setdefault(group, {}).update(
                    stored_rois.get("roi", {}).get(group, {})
                )
    return config, root


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
