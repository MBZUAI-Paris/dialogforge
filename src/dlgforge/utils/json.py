"""JSON extraction and parsing helpers for model outputs.

"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

def strip_code_fences(text: str) -> str:
    """Strip code fences.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.json import strip_code_fences
        >>> strip_code_fences(...)
    
    """
    kept = []
    for line in (text or "").splitlines():
        if line.strip().startswith("```"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract json object.
    
    Args:
        text (str): Input text.
    
    Returns:
        Optional[Dict[str, Any]]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.json import extract_json_object
        >>> extract_json_object(...)
    
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None

def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse json object.
    
    Args:
        text (str): Input text.
    
    Returns:
        Dict[str, Any]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.json import parse_json_object
        >>> parse_json_object(...)
    
    """
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        cleaned = strip_code_fences(text)
        extracted = extract_json_object(cleaned)
        return extracted or {}
