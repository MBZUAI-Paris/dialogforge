from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

LOGGER = logging.getLogger("dlgforge.llm")


@dataclass
class ChatResult:
    content: str
    tool_calls: List[Dict[str, Any]]
    raw: Dict[str, Any]


class OpenAIModelClient:
    def __init__(self) -> None:
        self._client_cache: Dict[Tuple[str, str, str], OpenAI] = {}
        self._async_client_cache: Dict[Tuple[str, str, str], AsyncOpenAI] = {}

    def complete(
        self,
        settings: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        model = (settings.get("model") or "").strip()
        if not model:
            raise ValueError("LLM model is required for each active agent.")

        client = self._get_client(settings)
        payload = self._build_payload(
            settings=settings,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        start = time.perf_counter()
        LOGGER.info(
            f"[llm] request model={model} messages={len(messages)} tools={len(tools) if tools else 0}"
        )
        try:
            response = client.chat.completions.create(**payload)
        except Exception as err:
            LOGGER.error(f"[llm] request failed model={model}: {err}")
            raise
        return self._to_chat_result(model=model, response=response, start=start)

    async def acomplete(
        self,
        settings: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        model = (settings.get("model") or "").strip()
        if not model:
            raise ValueError("LLM model is required for each active agent.")

        client = self._get_async_client(settings)
        payload = self._build_payload(
            settings=settings,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        start = time.perf_counter()
        LOGGER.info(
            f"[llm] request model={model} messages={len(messages)} tools={len(tools) if tools else 0}"
        )
        try:
            response = await client.chat.completions.create(**payload)
        except Exception as err:
            LOGGER.error(f"[llm] request failed model={model}: {err}")
            raise
        return self._to_chat_result(model=model, response=response, start=start)

    def _build_payload(
        self,
        settings: Dict[str, Any],
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str | Dict[str, Any]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "messages": messages}

        if settings.get("temperature") is not None:
            payload["temperature"] = settings["temperature"]
        if settings.get("max_tokens") is not None:
            payload["max_tokens"] = settings["max_tokens"]
        if settings.get("top_p") is not None:
            payload["top_p"] = settings["top_p"]
        if settings.get("timeout") is not None:
            payload["timeout"] = settings["timeout"]

        extra = settings.get("extra") or {}
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload.setdefault(key, value)

        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        if response_format:
            payload["response_format"] = response_format
        return payload

    def _to_chat_result(self, model: str, response: Any, start: float) -> ChatResult:
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls: List[Dict[str, Any]] = []
        for call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
            )

        raw = response.model_dump() if hasattr(response, "model_dump") else {"repr": repr(response)}
        LOGGER.info(
            f"[llm] response model={model} tool_calls={len(tool_calls)} "
            f"content_chars={len(content)} elapsed={time.perf_counter() - start:.2f}s"
        )
        return ChatResult(content=content, tool_calls=tool_calls, raw=raw)

    def _get_client(self, settings: Dict[str, Any]) -> OpenAI:
        base_url = str(settings.get("base_url") or "").strip()
        api_key = str(settings.get("api_key") or "").strip() or "EMPTY"
        max_retries = str(settings.get("max_retries") if settings.get("max_retries") is not None else "")
        cache_key = (base_url, api_key, max_retries)
        client = self._client_cache.get(cache_key)
        if client is not None:
            return client

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if settings.get("max_retries") is not None:
            kwargs["max_retries"] = settings["max_retries"]

        client = OpenAI(**kwargs)
        self._client_cache[cache_key] = client
        return client

    def _get_async_client(self, settings: Dict[str, Any]) -> AsyncOpenAI:
        base_url = str(settings.get("base_url") or "").strip()
        api_key = str(settings.get("api_key") or "").strip() or "EMPTY"
        max_retries = str(settings.get("max_retries") if settings.get("max_retries") is not None else "")
        cache_key = (base_url, api_key, max_retries)
        client = self._async_client_cache.get(cache_key)
        if client is not None:
            return client

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if settings.get("max_retries") is not None:
            kwargs["max_retries"] = settings["max_retries"]

        client = AsyncOpenAI(**kwargs)
        self._async_client_cache[cache_key] = client
        return client
