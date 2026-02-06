from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(path: str, project_root: Path, config_dir: Optional[Path] = None) -> Optional[Path]:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    candidate = project_root / raw
    if candidate.exists():
        return candidate
    if config_dir:
        alt = config_dir / raw
        if alt.exists():
            return alt
    return None
