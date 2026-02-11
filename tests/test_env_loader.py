"""Tests for dotenv loading policy."""

from __future__ import annotations

import os
from pathlib import Path

from dlgforge.utils.env import _is_allowed_dotenv_key, load_dotenv_files


def test_is_allowed_dotenv_key_supports_flexible_provider_secrets() -> None:
    assert _is_allowed_dotenv_key("OPENAI_API_KEY")
    assert _is_allowed_dotenv_key("FIREWORKS_API_KEY")
    assert _is_allowed_dotenv_key("GROQ_TOKEN")
    assert _is_allowed_dotenv_key("LLM_USER_API_KEY_ENV")
    assert _is_allowed_dotenv_key("LLM_QA_GENERATOR_API_KEY_ENV")
    assert not _is_allowed_dotenv_key("OPENAI_BASE_URL")
    assert not _is_allowed_dotenv_key("RUN_ID")


def test_load_dotenv_files_loads_only_secret_and_mapping_keys(
    tmp_path: Path, monkeypatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "FIREWORKS_API_KEY=fw-secret",
                "GROQ_TOKEN=groq-secret",
                "LLM_USER_API_KEY_ENV=FIREWORKS_API_KEY",
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "RUN_ID=test-run-id",
            ]
        ),
        encoding="utf-8",
    )

    for key in [
        "FIREWORKS_API_KEY",
        "GROQ_TOKEN",
        "LLM_USER_API_KEY_ENV",
        "OPENAI_BASE_URL",
        "RUN_ID",
    ]:
        monkeypatch.delenv(key, raising=False)

    load_dotenv_files(tmp_path)

    assert os.getenv("FIREWORKS_API_KEY") == "fw-secret"
    assert os.getenv("GROQ_TOKEN") == "groq-secret"
    assert os.getenv("LLM_USER_API_KEY_ENV") == "FIREWORKS_API_KEY"
    assert os.getenv("OPENAI_BASE_URL") is None
    assert os.getenv("RUN_ID") is None
