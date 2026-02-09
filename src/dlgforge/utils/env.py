"""Environment variable parsing and dotenv loading helpers.

"""

from __future__ import annotations

import os
from pathlib import Path

def env_flag(name: str, default: bool = False) -> bool:
    """Env flag.
    
    Args:
        name (str): str value used by this operation.
        default (bool): bool value used by this operation.
    
    Returns:
        bool: Boolean indicator describing the evaluated condition.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.env import env_flag
        >>> env_flag(...)
    
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

def env_int(name: str, default: int) -> int:
    """Env int.
    
    Args:
        name (str): str value used by this operation.
        default (int): int value used by this operation.
    
    Returns:
        int: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.env import env_int
        >>> env_int(...)
    
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

def env_float(name: str, default: float) -> float:
    """Env float.
    
    Args:
        name (str): str value used by this operation.
        default (float): float value used by this operation.
    
    Returns:
        float: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.env import env_float
        >>> env_float(...)
    
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default

def load_dotenv_files(project_root: Path) -> None:
    """Load dotenv files.
    
    Args:
        project_root (Path): Resolved project directory context.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.env import load_dotenv_files
        >>> load_dotenv_files(...)
    
    """
    for path in [project_root / ".env", project_root / "src" / "dlgforge" / ".env"]:
        _load_dotenv_file(path)

def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
