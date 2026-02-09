"""
Tests for LiteLLM-backed client behavior.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import patch

from dlgforge.llm.client import OpenAIModelClient


@dataclass
class _DummyFunction:
    name: str = "tool"
    arguments: str = "{}"


@dataclass
class _DummyToolCall:
    id: str = "call-1"
    function: _DummyFunction = field(default_factory=_DummyFunction)


@dataclass
class _DummyMessage:
    content: str
    tool_calls: List[_DummyToolCall] | None = None


@dataclass
class _DummyChoice:
    message: _DummyMessage


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_DummyChoice(message=_DummyMessage(content=content, tool_calls=[]))]

    def model_dump(self) -> dict[str, Any]:
        return {"ok": True}


def test_complete_maps_settings_to_litellm_kwargs() -> None:
    client = OpenAIModelClient()
    settings = {
        "model": "openai/gpt-oss-20b",
        "base_url": "http://localhost:1234/v1",
        "api_key": "EMPTY",
        "max_retries": 3,
        "timeout": 12,
    }
    message = {"role": "user", "content": "hello"}
    captured: dict[str, Any] = {}

    def _fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _DummyResponse('{"ok": 1}')

    with patch("dlgforge.llm.client.completion", side_effect=_fake_completion):
        result = client.complete(settings, [message], response_format={"type": "json_object"})

    assert result.content == '{"ok": 1}'
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["messages"] == [message]
    assert captured["api_base"] == "http://localhost:1234/v1"
    assert captured["api_key"] == "EMPTY"
    assert captured["num_retries"] == 3
    assert captured["timeout"] == 12
    assert captured["custom_llm_provider"] == "hosted_vllm"


def test_acomplete_maps_settings_to_litellm_kwargs() -> None:
    client = OpenAIModelClient()
    settings = {
        "model": "openai/gpt-oss-20b",
        "base_url": "http://localhost:1234/v1",
        "api_key": "EMPTY",
        "max_retries": 2,
    }
    message = {"role": "user", "content": "hello"}
    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _DummyResponse('{"ok": 2}')

    async def _run() -> None:
        with patch("dlgforge.llm.client.acompletion", side_effect=_fake_acompletion):
            result = await client.acomplete(settings, [message], response_format={"type": "json_object"})
        assert result.content == '{"ok": 2}'

    asyncio.run(_run())
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["messages"] == [message]
    assert captured["api_base"] == "http://localhost:1234/v1"
    assert captured["api_key"] == "EMPTY"
    assert captured["num_retries"] == 2
    assert captured["custom_llm_provider"] == "hosted_vllm"


def test_complete_sets_passthrough_for_openai_provider_on_custom_base_url() -> None:
    client = OpenAIModelClient()
    settings = {
        "provider": "openai",
        "model": "openai/gpt-oss-20b",
        "base_url": "http://localhost:1234/v1",
        "api_key": "EMPTY",
    }
    captured: dict[str, Any] = {}

    def _fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _DummyResponse("ok")

    with patch("dlgforge.llm.client.completion", side_effect=_fake_completion):
        client.complete(settings, [{"role": "user", "content": "hello"}])

    assert captured["custom_llm_provider"] == "hosted_vllm"


def test_complete_does_not_force_passthrough_on_openai_official_base_url() -> None:
    client = OpenAIModelClient()
    settings = {
        "provider": "openai",
        "model": "openai/gpt-5",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
    }
    captured: dict[str, Any] = {}

    def _fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _DummyResponse("ok")

    with patch("dlgforge.llm.client.completion", side_effect=_fake_completion):
        client.complete(settings, [{"role": "user", "content": "hello"}])

    assert "custom_llm_provider" not in captured


def test_complete_honors_explicit_non_openai_provider() -> None:
    client = OpenAIModelClient()
    settings = {
        "provider": "lm_studio",
        "model": "openai/gpt-oss-20b",
        "base_url": "http://localhost:1234/v1",
        "api_key": "EMPTY",
    }
    captured: dict[str, Any] = {}

    def _fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _DummyResponse("ok")

    with patch("dlgforge.llm.client.completion", side_effect=_fake_completion):
        client.complete(settings, [{"role": "user", "content": "hello"}])

    assert captured["custom_llm_provider"] == "lm_studio"


def test_complete_normalizes_dict_response_shape() -> None:
    client = OpenAIModelClient()
    settings = {"model": "openai/gpt-oss-20b"}
    message = {"role": "user", "content": "hello"}
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "vector_db_search",
                                "arguments": {"query": "abc"},
                            },
                        }
                    ],
                }
            }
        ]
    }

    with patch("dlgforge.llm.client.completion", return_value=response):
        result = client.complete(settings, [message])

    assert result.content == "hello world"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "call-1"
    assert result.tool_calls[0]["function"]["name"] == "vector_db_search"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"query": "abc"}
    assert result.raw == response
