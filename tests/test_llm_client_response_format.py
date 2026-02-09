"""
Test module for DialogForge behavior validation.

Main flows:
- Defines unit/integration test cases that assert expected runtime behavior.

Expected usage:
- Run with `pytest`; import helpers from production modules as needed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import patch

import pytest

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


class _DummyCompletions:
    def __init__(self, effects: List[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._effects = list(effects)

    def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _DummyAsyncCompletions:
    def __init__(self, effects: List[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._effects = list(effects)

    async def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


@dataclass
class _DummyChat:
    completions: Any


@dataclass
class _DummyClient:
    chat: _DummyChat


def test_complete_falls_back_to_text_response_format_for_lmstudio_style_error() -> None:
    """
    Test complete falls back to text response format for lmstudio style error.

    Args:
        None.

    Returns:
        None: Return value produced by `test_complete_falls_back_to_text_response_format_for_lmstudio_style_error`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_llm_client_response_format.py` for concrete usage of `test_complete_falls_back_to_text_response_format_for_lmstudio_style_error`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    client = OpenAIModelClient()
    settings = {"model": "openai/gpt-oss-20b", "base_url": "http://localhost:1234/v1", "api_key": "EMPTY"}
    message = {"role": "user", "content": "hello"}

    unsupported_error = RuntimeError(
        "Error code: 400 - {'error': \"'response_format.type' must be 'json_schema' or 'text'\"}"
    )
    completions = _DummyCompletions(
        effects=[unsupported_error, _DummyResponse('{"ok": 1}'), _DummyResponse('{"ok": 2}')],
    )
    dummy_client = _DummyClient(chat=_DummyChat(completions=completions))

    with patch.object(client, "_get_client", return_value=dummy_client):
        first = client.complete(settings, [message], response_format={"type": "json_object"})
        second = client.complete(settings, [message], response_format={"type": "json_object"})

    assert first.content == '{"ok": 1}'
    assert second.content == '{"ok": 2}'
    assert len(completions.calls) == 3
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert completions.calls[1]["response_format"] == {"type": "text"}
    assert completions.calls[2]["response_format"] == {"type": "text"}


def test_acomplete_falls_back_to_text_response_format_for_lmstudio_style_error() -> None:
    """
    Test acomplete falls back to text response format for lmstudio style error.

    Args:
        None.

    Returns:
        None: Return value produced by `test_acomplete_falls_back_to_text_response_format_for_lmstudio_style_error`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_llm_client_response_format.py` for concrete usage of `test_acomplete_falls_back_to_text_response_format_for_lmstudio_style_error`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    client = OpenAIModelClient()
    settings = {"model": "openai/gpt-oss-20b", "base_url": "http://localhost:1234/v1", "api_key": "EMPTY"}
    message = {"role": "user", "content": "hello"}

    unsupported_error = RuntimeError(
        "Error code: 400 - {'error': \"'response_format.type' must be 'json_schema' or 'text'\"}"
    )
    completions = _DummyAsyncCompletions(
        effects=[unsupported_error, _DummyResponse('{"ok": 1}'), _DummyResponse('{"ok": 2}')],
    )
    dummy_client = _DummyClient(chat=_DummyChat(completions=completions))

    async def _run() -> None:
        with patch.object(client, "_get_async_client", return_value=dummy_client):
            first = await client.acomplete(settings, [message], response_format={"type": "json_object"})
            second = await client.acomplete(settings, [message], response_format={"type": "json_object"})

        assert first.content == '{"ok": 1}'
        assert second.content == '{"ok": 2}'
        assert len(completions.calls) == 3
        assert completions.calls[0]["response_format"] == {"type": "json_object"}
        assert completions.calls[1]["response_format"] == {"type": "text"}
        assert completions.calls[2]["response_format"] == {"type": "text"}

    asyncio.run(_run())


def test_complete_does_not_retry_for_non_response_format_errors() -> None:
    """
    Test complete does not retry for non response format errors.

    Args:
        None.

    Returns:
        None: Return value produced by `test_complete_does_not_retry_for_non_response_format_errors`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_llm_client_response_format.py` for concrete usage of `test_complete_does_not_retry_for_non_response_format_errors`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    client = OpenAIModelClient()
    settings = {"model": "openai/gpt-oss-20b", "base_url": "http://localhost:1234/v1", "api_key": "EMPTY"}
    message = {"role": "user", "content": "hello"}

    completions = _DummyCompletions(effects=[RuntimeError("Error code: 500 - {'error': 'internal'}")])
    dummy_client = _DummyClient(chat=_DummyChat(completions=completions))

    with patch.object(client, "_get_client", return_value=dummy_client):
        with pytest.raises(RuntimeError, match="Error code: 500"):
            client.complete(settings, [message], response_format={"type": "json_object"})

    assert len(completions.calls) == 1
