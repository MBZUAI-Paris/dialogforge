"""
Tests for environment-only agent credential resolution.
"""

from __future__ import annotations

import os

import pytest

from dlgforge.llm.settings import resolve_llm_settings


def _cfg() -> dict:
    return {
        "llm": {
            "mode": "api",
            "agents": {
                "user": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "base_url": "https://api.openai.com/v1",
                }
            },
        }
    }


def _cfg_all_roles() -> dict:
    return {
        "llm": {
            "mode": "api",
            "agents": {
                "user": {"provider": "openai", "model": "gpt-5"},
                "assistant": {"provider": "gemini", "model": "gemini/gemini-2.0-flash"},
                "judge": {"provider": "anthropic", "model": "claude-3-5-sonnet"},
            },
        }
    }


def test_resolve_llm_settings_reads_api_key_from_mapping_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_USER_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "token-123")

    settings = resolve_llm_settings(_cfg(), "qa_generator")
    assert settings["api_key"] == "token-123"
    assert settings["api_key_env"] == "OPENAI_API_KEY"


def test_resolve_llm_settings_fails_when_mapping_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_USER_API_KEY_ENV", raising=False)
    monkeypatch.delenv("LLM_QA_GENERATOR_API_KEY_ENV", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "token-123")

    with pytest.raises(RuntimeError, match="LLM_USER_API_KEY_ENV"):
        resolve_llm_settings(_cfg(), "qa_generator")


def test_resolve_llm_settings_fails_when_mapped_secret_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_USER_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        resolve_llm_settings(_cfg(), "qa_generator")


def test_resolve_llm_settings_rejects_agent_api_key_fields_in_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_USER_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "token-123")
    cfg = _cfg()
    cfg["llm"]["agents"]["user"]["api_key_env"] = "OPENAI_API_KEY"

    with pytest.raises(ValueError, match="llm.agents.user"):
        resolve_llm_settings(cfg, "qa_generator")


@pytest.mark.parametrize(
    ("agent_key", "mapping_env", "legacy_mapping_env", "secret_env"),
    [
        ("qa_generator", "LLM_USER_API_KEY_ENV", "LLM_QA_GENERATOR_API_KEY_ENV", "OPENAI_API_KEY"),
        ("kb_responder", "LLM_ASSISTANT_API_KEY_ENV", "LLM_KB_RESPONDER_API_KEY_ENV", "GEMINI_API_KEY"),
        ("qa_judge", "LLM_JUDGE_API_KEY_ENV", "LLM_QA_JUDGE_API_KEY_ENV", "ANTHROPIC_API_KEY"),
    ],
)
def test_resolve_llm_settings_uses_role_specific_mapping_vars(
    monkeypatch: pytest.MonkeyPatch,
    agent_key: str,
    mapping_env: str,
    legacy_mapping_env: str,
    secret_env: str,
) -> None:
    monkeypatch.delenv("LLM_QA_GENERATOR_API_KEY_ENV", raising=False)
    monkeypatch.delenv("LLM_KB_RESPONDER_API_KEY_ENV", raising=False)
    monkeypatch.delenv("LLM_QA_JUDGE_API_KEY_ENV", raising=False)
    monkeypatch.setenv("LLM_USER_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "token-openai")
    monkeypatch.setenv("LLM_ASSISTANT_API_KEY_ENV", "GEMINI_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "token-gemini")
    monkeypatch.setenv("LLM_JUDGE_API_KEY_ENV", "ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "token-anthropic")

    settings = resolve_llm_settings(_cfg_all_roles(), agent_key)
    assert settings["api_key_env"] == secret_env
    assert settings["api_key"] == str(os.getenv(secret_env))

    # Remove only the expected mapping var and verify hard-fail mentions that variable.
    monkeypatch.delenv(mapping_env, raising=False)
    monkeypatch.delenv(legacy_mapping_env, raising=False)
    with pytest.raises(RuntimeError, match=mapping_env):
        resolve_llm_settings(_cfg_all_roles(), agent_key)


def test_resolve_llm_settings_supports_legacy_mapping_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_USER_API_KEY_ENV", raising=False)
    monkeypatch.setenv("LLM_QA_GENERATOR_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "token-legacy")

    settings = resolve_llm_settings(_cfg(), "qa_generator")
    assert settings["api_key"] == "token-legacy"
    assert settings["api_key_env"] == "OPENAI_API_KEY"
