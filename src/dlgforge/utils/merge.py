"""Dictionary merge and path resolution helpers.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge.
    
    Args:
        base (Dict[str, Any]): Dict[str, Any] value used by this operation.
        override (Dict[str, Any]): Dict[str, Any] value used by this operation.
    
    Returns:
        Dict[str, Any]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.merge import deep_merge
        >>> deep_merge(...)
    
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def resolve_path(path: str, project_root: Path, config_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve path from configuration.
    
    Args:
        path (str): Filesystem path used by this operation.
        project_root (Path): Resolved project directory context.
        config_dir (Optional[Path]): Resolved project directory context.
    
    Returns:
        Optional[Path]: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.merge import resolve_path
        >>> resolve_path(...)
    
    """
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
