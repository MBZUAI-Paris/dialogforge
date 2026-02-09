"""Text hashing and template rendering helpers.

"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

def hash_text(text: str) -> str:
    """Hash text.
    
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
        >>> from dlgforge.utils.text import hash_text
        >>> hash_text(...)
    
    """
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def render_template(template: str, values: Dict[str, Any]) -> str:
    """Render template.
    
    Args:
        template (str): Input text.
        values (Dict[str, Any]): Dict[str, Any] value used by this operation.
    
    Returns:
        str: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.utils.text import render_template
        >>> render_template(...)
    
    """
    pattern = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        value = values.get(key, "")
        if value is None:
            return ""
        return str(value)

    return pattern.sub(repl, template)
