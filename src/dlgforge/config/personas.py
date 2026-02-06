from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from dlgforge.utils import resolve_path


def select_personas(cfg: Dict[str, Any], project_root: Path, config_path: Path) -> Tuple[str, str, Dict[str, str]]:
    if not resolve_personas_enabled(cfg):
        return "", "", {}

    personas = load_personas(cfg, project_root, config_path)
    user_personas = personas.get("user", [])
    assistant_personas = personas.get("assistant", [])

    rng = build_persona_rng(cfg)
    user_choice = rng.choice(user_personas) if user_personas else {}
    assistant_choice = rng.choice(assistant_personas) if assistant_personas else {}

    return (
        format_persona(user_choice),
        format_persona(assistant_choice),
        {
            "user_id": str(user_choice.get("id", "")),
            "assistant_id": str(assistant_choice.get("id", "")),
        },
    )


def resolve_personas_enabled(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("personas", {}) or {}).get("enabled", True))


def resolve_personas_path(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("personas", {}) or {}).get("path", "") or "")


def resolve_question_seed(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("question_seed", "") or "")


def build_persona_rng(cfg: Dict[str, Any]) -> random.Random:
    seed = resolve_question_seed(cfg) or datetime.utcnow().isoformat()
    return random.Random(f"persona-{seed}")


def load_personas(cfg: Dict[str, Any], project_root: Path, config_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    path = resolve_personas_path(cfg)
    if not path:
        return default_personas()
    resolved = resolve_path(path, project_root=project_root, config_dir=config_path.parent)
    if not resolved or not resolved.exists():
        return default_personas()

    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except Exception:
        return default_personas()

    personas = data.get("personas", {}) if isinstance(data, dict) else {}
    user = personas.get("user", []) if isinstance(personas, dict) else []
    assistant = personas.get("assistant", []) if isinstance(personas, dict) else []
    return {
        "user": user if isinstance(user, list) else [],
        "assistant": assistant if isinstance(assistant, list) else [],
    }


def default_personas() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "user": [
            {
                "id": "curious_professional",
                "name": "Curious Professional",
                "traits": ["practical", "detail-oriented", "polite"],
                "style": "Asks clear, goal-oriented questions with real-world constraints.",
            }
        ],
        "assistant": [
            {
                "id": "helpful_tutor",
                "name": "Helpful Tutor",
                "traits": ["patient", "structured", "encouraging"],
                "style": "Explains clearly, uses step-by-step reasoning and examples.",
            }
        ],
    }


def format_persona(persona: Dict[str, Any]) -> str:
    if not persona:
        return ""
    name = persona.get("name", "")
    traits = ", ".join(persona.get("traits", []) or [])
    style = persona.get("style", "")
    parts = [part for part in [name, traits, style] if part]
    return " | ".join(parts)
