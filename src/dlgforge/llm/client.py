from __future__ import annotations

import json
import logging
import os
import random
import threading
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
        self._routing_lock = threading.Lock()
        self._routing_state: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._response_format_lock = threading.Lock()
        self._json_object_unsupported: set[str] = set()

    def complete(
        self,
        settings: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        routed_settings, endpoint_name, endpoint_key = self._settings_with_routed_endpoint(settings)
        model = (routed_settings.get("model") or "").strip()
        if not model:
            raise ValueError("LLM model is required for each active agent.")

        client = self._get_client(routed_settings)
        payload = self._build_payload(
            settings=routed_settings,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        capability_key = self._response_format_capability_key(routed_settings, endpoint_key)
        payload = self._apply_response_format_compatibility(payload, capability_key)

        start = time.perf_counter()
        LOGGER.info(
            f"[llm] request model={model} endpoint={endpoint_name or '-'} "
            f"messages={len(messages)} tools={len(tools) if tools else 0}"
        )
        self._on_endpoint_start(endpoint_key)
        try:
            response = client.chat.completions.create(**payload)
        except Exception as err:
            retried_response = self._retry_with_text_response_format_if_needed(
                err=err,
                payload=payload,
                settings=routed_settings,
                model=model,
                endpoint_name=endpoint_name,
                capability_key=capability_key,
                call=lambda p: client.chat.completions.create(**p),
            )
            if retried_response is None:
                LOGGER.error(f"[llm] request failed model={model}: {err}")
                raise
            response = retried_response
        finally:
            self._on_endpoint_done(endpoint_key, time.perf_counter() - start)
        return self._to_chat_result(model=model, response=response, start=start)

    async def acomplete(
        self,
        settings: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        routed_settings, endpoint_name, endpoint_key = self._settings_with_routed_endpoint(settings)
        model = (routed_settings.get("model") or "").strip()
        if not model:
            raise ValueError("LLM model is required for each active agent.")

        client = self._get_async_client(routed_settings)
        payload = self._build_payload(
            settings=routed_settings,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        capability_key = self._response_format_capability_key(routed_settings, endpoint_key)
        payload = self._apply_response_format_compatibility(payload, capability_key)

        start = time.perf_counter()
        LOGGER.info(
            f"[llm] request model={model} endpoint={endpoint_name or '-'} "
            f"messages={len(messages)} tools={len(tools) if tools else 0}"
        )
        self._on_endpoint_start(endpoint_key)
        try:
            response = await client.chat.completions.create(**payload)
        except Exception as err:
            retried_response = await self._aretry_with_text_response_format_if_needed(
                err=err,
                payload=payload,
                settings=routed_settings,
                model=model,
                endpoint_name=endpoint_name,
                capability_key=capability_key,
                call=lambda p: client.chat.completions.create(**p),
            )
            if retried_response is None:
                LOGGER.error(f"[llm] request failed model={model}: {err}")
                raise
            response = retried_response
        finally:
            self._on_endpoint_done(endpoint_key, time.perf_counter() - start)
        return self._to_chat_result(model=model, response=response, start=start)

    def _settings_with_routed_endpoint(self, settings: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
        routing = settings.get("routing")
        if not isinstance(routing, dict):
            return settings, "", ""
        endpoints = routing.get("endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            return settings, "", ""

        selected = self._pick_endpoint(endpoints)
        if not isinstance(selected, dict):
            return settings, "", ""

        routed = dict(settings)
        base_url = str(selected.get("base_url") or "").strip()
        if base_url:
            routed["base_url"] = base_url

        api_key = str(selected.get("api_key") or "").strip()
        api_key_env = str(selected.get("api_key_env") or "").strip()
        if not api_key and api_key_env:
            api_key = str(os.getenv(api_key_env, "") or "")
        if api_key:
            routed["api_key"] = api_key
        elif not str(routed.get("api_key") or "").strip() and base_url:
            routed["api_key"] = "EMPTY"

        timeout = selected.get("timeout")
        if timeout is not None:
            routed["timeout"] = timeout
        model = str(selected.get("model") or "").strip()
        if model:
            routed["model"] = model

        endpoint_name = str(selected.get("name") or base_url or "endpoint")
        endpoint_key = self._endpoint_state_key(selected)
        return routed, endpoint_name, endpoint_key

    def _pick_endpoint(self, endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        state_key = self._routing_state_key(endpoints)
        with self._routing_lock:
            state = self._routing_state.setdefault(state_key, {})
            scored: List[Tuple[float, float, Dict[str, Any]]] = []
            for endpoint in endpoints:
                if not isinstance(endpoint, dict):
                    continue
                key = self._endpoint_state_key(endpoint)
                endpoint_state = state.setdefault(key, {"inflight": 0.0, "latency_ewma": 0.5})
                inflight = endpoint_state["inflight"]
                latency = max(endpoint_state["latency_ewma"], 1e-3)
                weight = max(float(endpoint.get("weight") or 1.0), 1e-6)
                max_in_flight = float(endpoint.get("max_in_flight") or 0.0)
                pressure = inflight + 1.0
                if max_in_flight > 0 and inflight >= max_in_flight:
                    pressure *= 1000.0
                score = (pressure / weight) * latency
                scored.append((score, random.random(), endpoint))
            if not scored:
                return endpoints[0]
            scored.sort(key=lambda row: (row[0], row[1]))
            return scored[0][2]

    def _routing_state_key(self, endpoints: List[Dict[str, Any]]) -> str:
        try:
            return json.dumps(endpoints, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(endpoints)

    def _endpoint_state_key(self, endpoint: Dict[str, Any]) -> str:
        base_url = str(endpoint.get("base_url") or "")
        name = str(endpoint.get("name") or "")
        return f"{name}|{base_url}"

    def _on_endpoint_start(self, endpoint_key: str) -> None:
        if not endpoint_key:
            return
        with self._routing_lock:
            for state in self._routing_state.values():
                if endpoint_key in state:
                    state[endpoint_key]["inflight"] = state[endpoint_key].get("inflight", 0.0) + 1.0
                    return

    def _on_endpoint_done(self, endpoint_key: str, elapsed: float) -> None:
        if not endpoint_key:
            return
        with self._routing_lock:
            for state in self._routing_state.values():
                if endpoint_key not in state:
                    continue
                inflight = state[endpoint_key].get("inflight", 0.0)
                state[endpoint_key]["inflight"] = max(0.0, inflight - 1.0)
                prev = state[endpoint_key].get("latency_ewma", elapsed)
                state[endpoint_key]["latency_ewma"] = 0.8 * prev + 0.2 * max(elapsed, 1e-6)
                return

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

    def _response_format_capability_key(self, settings: Dict[str, Any], endpoint_key: str) -> str:
        if endpoint_key:
            return f"endpoint:{endpoint_key}"
        base_url = str(settings.get("base_url") or "").strip().rstrip("/")
        if base_url:
            return f"base_url:{base_url}"
        return "default"

    def _apply_response_format_compatibility(self, payload: Dict[str, Any], capability_key: str) -> Dict[str, Any]:
        response_format = payload.get("response_format")
        if not isinstance(response_format, dict):
            return payload
        rf_type = str(response_format.get("type") or "").strip().lower()
        if rf_type != "json_object":
            return payload
        if self._json_object_supported(capability_key):
            return payload

        patched = dict(payload)
        patched["response_format"] = {"type": "text"}
        return patched

    def _json_object_supported(self, capability_key: str) -> bool:
        with self._response_format_lock:
            return capability_key not in self._json_object_unsupported

    def _mark_json_object_unsupported(self, capability_key: str) -> None:
        with self._response_format_lock:
            self._json_object_unsupported.add(capability_key)

    def _is_json_object_response_format_error(self, err: Exception) -> bool:
        message = str(err).lower()
        if "response_format" not in message:
            return False
        if "json_schema" in message and "text" in message:
            return True
        return "json_object" in message and "must" in message

    def _retry_with_text_response_format_if_needed(
        self,
        *,
        err: Exception,
        payload: Dict[str, Any],
        settings: Dict[str, Any],
        model: str,
        endpoint_name: str,
        capability_key: str,
        call: Any,
    ) -> Any:
        response_format = payload.get("response_format")
        if not isinstance(response_format, dict):
            return None
        if str(response_format.get("type") or "").strip().lower() != "json_object":
            return None
        if not self._is_json_object_response_format_error(err):
            return None

        self._mark_json_object_unsupported(capability_key)
        fallback_payload = dict(payload)
        fallback_payload["response_format"] = {"type": "text"}
        LOGGER.warning(
            "[llm] backend does not support response_format=json_object model=%s endpoint=%s; retrying with response_format=text",
            model,
            endpoint_name or settings.get("base_url") or "-",
        )
        return call(fallback_payload)

    async def _aretry_with_text_response_format_if_needed(
        self,
        *,
        err: Exception,
        payload: Dict[str, Any],
        settings: Dict[str, Any],
        model: str,
        endpoint_name: str,
        capability_key: str,
        call: Any,
    ) -> Any:
        response_format = payload.get("response_format")
        if not isinstance(response_format, dict):
            return None
        if str(response_format.get("type") or "").strip().lower() != "json_object":
            return None
        if not self._is_json_object_response_format_error(err):
            return None

        self._mark_json_object_unsupported(capability_key)
        fallback_payload = dict(payload)
        fallback_payload["response_format"] = {"type": "text"}
        LOGGER.warning(
            "[llm] backend does not support response_format=json_object model=%s endpoint=%s; retrying with response_format=text",
            model,
            endpoint_name or settings.get("base_url") or "-",
        )
        return await call(fallback_payload)

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
