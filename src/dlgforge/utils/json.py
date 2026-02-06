from __future__ import annotations

import json
from typing import Any, Dict, Optional


def strip_code_fences(text: str) -> str:
    kept = []
    for line in (text or "").splitlines():
        if line.strip().startswith("```"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
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
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        cleaned = strip_code_fences(text)
        extracted = extract_json_object(cleaned)
        return extracted or {}
