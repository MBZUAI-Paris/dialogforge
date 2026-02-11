"""
Tests for strict LLM YAML validation rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from dlgforge.config.loader import load_config


def _write_config(tmp_path: Path, payload: Dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_rejects_forbidden_global_llm_model(tmp_path: Path) -> None:
    payload = {
        "llm": {
            "mode": "api",
            "model": "gpt-5",
            "agents": {
                "user": {"model": "gpt-5"},
                "assistant": {"model": "gpt-5"},
                "judge": {"model": "gpt-5"},
            },
        }
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="llm.model"):
        load_config(path)


@pytest.mark.parametrize("forbidden_key", ["api_key", "api_key_env"])
def test_load_config_rejects_forbidden_global_llm_credential_fields(
    tmp_path: Path, forbidden_key: str
) -> None:
    payload = {
        "llm": {
            "mode": "api",
            forbidden_key: "OPENAI_API_KEY" if forbidden_key.endswith("_env") else "sk-123",
            "agents": {
                "user": {"model": "gpt-5"},
                "assistant": {"model": "gpt-5"},
                "judge": {"model": "gpt-5"},
            },
        }
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=rf"llm.{forbidden_key}"):
        load_config(path)


def test_load_config_rejects_agent_level_api_key_env_field(tmp_path: Path) -> None:
    payload = {
        "llm": {
            "mode": "api",
            "agents": {
                "user": {
                    "model": "gpt-5",
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
        }
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="llm.agents.user.api_key_env"):
        load_config(path)


def test_load_config_rejects_legacy_stage_agent_api_key_field(tmp_path: Path) -> None:
    payload = {
        "llm": {
            "mode": "api",
            "agents": {
                "qa_generator": {
                    "model": "gpt-5",
                    "provider": "openai",
                    "api_key": "sk-123",
                }
            },
        }
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="llm.agents.qa_generator.api_key"):
        load_config(path)


def test_load_config_normalizes_legacy_llm_backend_openai_to_api(tmp_path: Path) -> None:
    payload = {
        "llm": {
            "backend": "openai",
            "agents": {
                "user": {"model": "gpt-5"},
                "assistant": {"model": "gpt-5"},
                "judge": {"model": "gpt-5"},
            },
        }
    }
    path = _write_config(tmp_path, payload)
    cfg, _, _ = load_config(path)

    assert cfg["llm"]["mode"] == "api"


def test_load_config_maps_legacy_stage_agents_to_canonical_roles(tmp_path: Path) -> None:
    payload = {
        "llm": {
            "mode": "api",
            "agents": {
                "qa_generator": {"model": "legacy-user"},
                "kb_responder": {"model": "legacy-assistant"},
                "qa_judge": {"model": "legacy-judge"},
            },
        }
    }
    path = _write_config(tmp_path, payload)
    cfg, _, _ = load_config(path)

    agents = cfg["llm"]["agents"]
    assert agents["user"]["model"] == "legacy-user"
    assert agents["assistant"]["model"] == "legacy-assistant"
    assert agents["judge"]["model"] == "legacy-judge"
