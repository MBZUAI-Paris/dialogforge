"""Persona loading, formatting, and sampling utilities."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from dlgforge.utils import resolve_path

class UniformPersonaSampler:
    """Cycle-based sampler for user/assistant personas.

    The sampler shuffles each persona pool and pops items until exhausted, then
    reshuffles. This prevents early repetition while keeping distribution near
    uniform over long runs.

    Args:
        user_personas (List[Dict[str, Any]]): User persona records.
        assistant_personas (List[Dict[str, Any]]): Assistant persona records.
        rng (random.Random): Random generator used for deterministic shuffling.

    Side Effects / I/O:
        - Mutates internal in-memory cycles as samples are drawn.

    Preconditions / Invariants:
        - Persona entries are expected to be mapping-like objects.
        - Empty pools are supported and return empty persona text/ids.

    Examples:
        >>> from dlgforge.config.personas import UniformPersonaSampler
        >>> import random
        >>> sampler = UniformPersonaSampler([{"id": "u1"}], [{"id": "a1"}], random.Random(7))
        >>> sampler.sample()[2]["user_id"]
        'u1'
    """
    def __init__(
        self,
        user_personas: List[Dict[str, Any]],
        assistant_personas: List[Dict[str, Any]],
        rng: random.Random,
    ) -> None:
        self._user_personas = [item for item in user_personas if isinstance(item, dict)]
        self._assistant_personas = [item for item in assistant_personas if isinstance(item, dict)]
        self._rng = rng
        self._user_cycle: List[Dict[str, Any]] = []
        self._assistant_cycle: List[Dict[str, Any]] = []

    def _next_choice(self, pool: List[Dict[str, Any]], cycle: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not pool:
            return {}
        if not cycle:
            cycle.extend(pool)
            self._rng.shuffle(cycle)
        return cycle.pop()

    def sample(self) -> Tuple[str, str, Dict[str, str]]:
        """Sample one user persona and one assistant persona.

        Returns:
            Tuple[str, str, Dict[str, str]]: `(user_text, assistant_text,
            metadata)` where `metadata` includes `user_id` and `assistant_id`.

        Side Effects / I/O:
            - Advances in-memory sampling cycles.

        Examples:
            >>> from dlgforge.config.personas import UniformPersonaSampler
            >>> import random
            >>> sampler = UniformPersonaSampler([{"id": "u1"}], [{"id": "a1"}], random.Random(0))
            >>> sample = sampler.sample()
            >>> sample[2]["assistant_id"]
            'a1'
        """
        user_choice = self._next_choice(self._user_personas, self._user_cycle)
        assistant_choice = self._next_choice(self._assistant_personas, self._assistant_cycle)
        return (
            format_persona(user_choice),
            format_persona(assistant_choice),
            {
                "user_id": str(user_choice.get("id", "")),
                "assistant_id": str(assistant_choice.get("id", "")),
            },
        )

def select_personas(cfg: Dict[str, Any], project_root: Path, config_path: Path) -> Tuple[str, str, Dict[str, str]]:
    """Select one user persona and one assistant persona for a conversation.

    Returns empty values when personas are disabled.

    Args:
        cfg (Dict[str, Any]): Loaded runtime configuration.
        project_root (Path): Project root used for resolving relative paths.
        config_path (Path): Path to the active config file.

    Returns:
        Tuple[str, str, Dict[str, str]]: `(user_text, assistant_text,
        metadata)` with selected persona ids in metadata.

    Side Effects / I/O:
        - Reads persona files when persona loading is enabled.

    Examples:
        >>> from dlgforge.config.personas import select_personas
        >>> select_personas(...)
    """
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
    """Return whether persona sampling is enabled in configuration.

    Args:
        cfg (Dict[str, Any]): Loaded runtime configuration.

    Returns:
        bool: `True` when `personas.enabled` is truthy, otherwise `False`.

    Examples:
        >>> from dlgforge.config.personas import resolve_personas_enabled
        >>> resolve_personas_enabled({"personas": {"enabled": True}})
        True
    """
    return bool((cfg.get("personas", {}) or {}).get("enabled", True))

def resolve_personas_path(cfg: Dict[str, Any]) -> str:
    """Return the configured personas file path.

    Args:
        cfg (Dict[str, Any]): Loaded runtime configuration.

    Returns:
        str: Raw value of `personas.path`, or an empty string when unset.

    Examples:
        >>> from dlgforge.config.personas import resolve_personas_path
        >>> resolve_personas_path({"personas": {"path": "data/personas.yaml"}})
        'data/personas.yaml'
    """
    return str((cfg.get("personas", {}) or {}).get("path", "") or "")

def resolve_question_seed(cfg: Dict[str, Any]) -> str:
    """Return the run-level question seed used for persona RNG initialization.

    Args:
        cfg (Dict[str, Any]): Loaded runtime configuration.

    Returns:
        str: Value of `run.data.seeding.question`, or an empty string when unset.

    Examples:
        >>> from dlgforge.config.personas import resolve_question_seed
        >>> resolve_question_seed({"run": {"data": {"seeding": {"question": "seed-1"}}}})
        'seed-1'
    """
    run_cfg = cfg.get("run", {}) or {}
    data_cfg = run_cfg.get("data", {}) if isinstance(run_cfg.get("data"), dict) else {}
    seeding_cfg = data_cfg.get("seeding", {}) if isinstance(data_cfg.get("seeding"), dict) else {}
    canonical = str(seeding_cfg.get("question", "") or "").strip()
    if canonical:
        return canonical
    legacy = str(run_cfg.get("question_seed", "") or "").strip()
    if legacy:
        return legacy
    return str(run_cfg.get("seed_question", "") or "")

def build_persona_rng(cfg: Dict[str, Any]) -> random.Random:
    """Build a deterministic RNG for persona selection.

    Uses `run.data.seeding.question` when present; otherwise falls back to
    current UTC timestamp for non-deterministic runs.

    Args:
        cfg (Dict[str, Any]): Loaded runtime configuration.

    Returns:
        random.Random: RNG seeded with a persona-specific seed namespace.

    Examples:
        >>> from dlgforge.config.personas import build_persona_rng
        >>> build_persona_rng(...)
    """
    seed = resolve_question_seed(cfg) or datetime.utcnow().isoformat()
    return random.Random(f"persona-{seed}")

def build_uniform_persona_sampler(
    cfg: Dict[str, Any],
    project_root: Path,
    config_path: Path,
) -> UniformPersonaSampler:
    """Build a `UniformPersonaSampler` from config and persona assets.

    Args:
        cfg (Dict[str, Any]): Loaded runtime configuration.
        project_root (Path): Project root used for resolving relative paths.
        config_path (Path): Path to the active config file.

    Returns:
        UniformPersonaSampler: Initialized sampler with user/assistant pools.

    Side Effects / I/O:
        - Reads persona files when enabled.

    Examples:
        >>> from dlgforge.config.personas import build_uniform_persona_sampler
        >>> build_uniform_persona_sampler(...)
    """
    personas = load_personas(cfg, project_root, config_path) if resolve_personas_enabled(cfg) else {"user": [], "assistant": []}
    return UniformPersonaSampler(
        user_personas=personas.get("user", []),
        assistant_personas=personas.get("assistant", []),
        rng=build_persona_rng(cfg),
    )

def load_personas(cfg: Dict[str, Any], project_root: Path, config_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load user and assistant personas from configuration.

    Falls back to built-in defaults when the configured file is missing,
    unreadable, or malformed.

    Args:
        cfg (Dict[str, Any]): Loaded runtime configuration.
        project_root (Path): Project root used for resolving relative paths.
        config_path (Path): Path to the active config file.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Mapping with `user` and `assistant`
        persona lists.

    Side Effects / I/O:
        - Reads YAML from disk when a personas file is configured.

    Examples:
        >>> from dlgforge.config.personas import load_personas
        >>> load_personas(...)
    """
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
    """Return the built-in fallback persona set.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Mapping with one default user persona
        and one default assistant persona.

    Examples:
        >>> from dlgforge.config.personas import default_personas
        >>> sorted(default_personas().keys())
        ['assistant', 'user']
    """
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
    """Format a persona record into a compact display string.

    Args:
        persona (Dict[str, Any]): Persona mapping with optional `name`,
            `traits`, and `style` keys.

    Returns:
        str: Human-readable persona string. Returns an empty string for empty
        persona input.

    Examples:
        >>> from dlgforge.config.personas import format_persona
        >>> format_persona({'name': 'Tutor', 'traits': ['patient'], 'style': 'Explains step-by-step'})
        'Tutor | patient | Explains step-by-step'
    """
    if not persona:
        return ""
    name = persona.get("name", "")
    traits = ", ".join(persona.get("traits", []) or [])
    style = persona.get("style", "")
    parts = [part for part in [name, traits, style] if part]
    return " | ".join(parts)
