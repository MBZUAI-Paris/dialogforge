from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from dlgforge.utils import render_template


_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=1)
def load_agents_config() -> Dict[str, Any]:
    path = _PROMPTS_DIR / "agents.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


@lru_cache(maxsize=1)
def load_tasks_config() -> Dict[str, Any]:
    path = _PROMPTS_DIR / "tasks.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def build_agent_system_prompt(agent_key: str) -> str:
    cfg = load_agents_config().get(agent_key, {})
    if not isinstance(cfg, dict):
        return ""

    role = str(cfg.get("role", "")).strip()
    goal = str(cfg.get("goal", "")).strip()
    backstory = str(cfg.get("backstory", "")).strip()

    sections = []
    if role:
        sections.append(f"Role:\n{role}")
    if goal:
        sections.append(f"Goal:\n{goal}")
    if backstory:
        sections.append(f"Backstory:\n{backstory}")
    sections.append("Return valid JSON only. No markdown.")
    return "\n\n".join(sections)


def build_task_prompt(task_key: str, values: Dict[str, Any]) -> str:
    task_cfg = load_tasks_config().get(task_key, {})
    if not isinstance(task_cfg, dict):
        return ""

    description = str(task_cfg.get("description", ""))
    expected_output = str(task_cfg.get("expected_output", ""))

    rendered = render_template(description, values)
    rendered_output = render_template(expected_output, values)
    if rendered_output:
        rendered = f"{rendered}\n\nExpected output format:\n{rendered_output}"
    return rendered
