from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def resolve_llm_settings(cfg: Dict[str, Any], agent_key: str) -> Dict[str, Any]:
    llm_cfg = cfg.get("llm", {}) or {}
    merged = {
        "backend": llm_cfg.get("backend"),
        "provider": llm_cfg.get("provider"),
        "model": llm_cfg.get("model"),
        "base_url": llm_cfg.get("base_url"),
        "api_key": llm_cfg.get("api_key"),
        "api_key_env": llm_cfg.get("api_key_env"),
        "temperature": llm_cfg.get("temperature"),
        "max_tokens": llm_cfg.get("max_tokens"),
        "top_p": llm_cfg.get("top_p"),
        "timeout": llm_cfg.get("timeout"),
        "max_retries": llm_cfg.get("max_retries"),
        "extra": llm_cfg.get("extra") or {},
        "routing": llm_cfg.get("routing") or {},
    }

    per_agent = (llm_cfg.get("agents", {}) or {}).get(agent_key, {}) or {}
    merged.update({k: v for k, v in per_agent.items() if v is not None})

    prefix = f"LLM_{agent_key.upper()}_"
    env_keys = [
        "BACKEND",
        "PROVIDER",
        "MODEL",
        "BASE_URL",
        "API_KEY",
        "API_KEY_ENV",
        "TEMPERATURE",
        "MAX_TOKENS",
        "TOP_P",
        "TIMEOUT",
        "MAX_RETRIES",
    ]
    for key in env_keys:
        agent_value = os.getenv(prefix + key)
        global_value = os.getenv("LLM_" + key)
        value = agent_value if agent_value not in {None, ""} else global_value
        if value in {None, ""}:
            continue
        merged[key.lower()] = value

    routing = merged.get("routing")
    if not isinstance(routing, dict):
        routing = {}
    routing_strategy = os.getenv(prefix + "ROUTING_STRATEGY") or os.getenv("LLM_ROUTING_STRATEGY")
    if routing_strategy:
        routing["strategy"] = routing_strategy
    endpoints_raw = os.getenv(prefix + "ROUTING_ENDPOINTS_JSON") or os.getenv("LLM_ROUTING_ENDPOINTS_JSON")
    if endpoints_raw:
        try:
            endpoints = json.loads(endpoints_raw)
            if isinstance(endpoints, list):
                routing["endpoints"] = endpoints
        except json.JSONDecodeError:
            pass
    merged["routing"] = routing

    merged["temperature"] = _as_optional_float(merged.get("temperature"))
    merged["max_tokens"] = _as_optional_int(merged.get("max_tokens"))
    merged["top_p"] = _as_optional_float(merged.get("top_p"))
    merged["timeout"] = _as_optional_float(merged.get("timeout"))
    merged["max_retries"] = _as_optional_int(merged.get("max_retries"))

    if not merged.get("api_key"):
        api_key_env = merged.get("api_key_env")
        if api_key_env:
            merged["api_key"] = os.getenv(str(api_key_env), "")

    if not merged.get("api_key"):
        merged["api_key"] = os.getenv("OPENAI_API_KEY", "")

    if not (merged.get("base_url") or "").strip():
        merged["base_url"] = os.getenv("OPENAI_BASE_URL", "")

    model = (merged.get("model") or "").strip()
    if not model:
        merged["model"] = _fallback_model_from_env()

    if not merged.get("api_key") and merged.get("base_url"):
        merged["api_key"] = os.getenv("OPENAI_API_KEY", "EMPTY")

    return merged


def resolve_agent_used_name(cfg: Dict[str, Any], agent_key: str) -> str:
    settings = resolve_llm_settings(cfg, agent_key)
    model = (settings.get("model") or "").strip()
    provider = (settings.get("provider") or "").strip()
    if not model:
        return agent_key
    if provider and "/" not in model:
        return f"{provider}/{model}"
    return model


def required_agents(cfg: Dict[str, Any]) -> List[str]:
    agents = ["qa_generator", "kb_responder"]
    judge_cfg = cfg.get("judge", {}) or {}
    mode = str(judge_cfg.get("mode", "offline") or "offline").strip().lower()
    enabled = bool(judge_cfg.get("enabled", True))
    if mode == "online" and enabled:
        agents.append("qa_judge")
    return agents


def missing_models(cfg: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for agent in required_agents(cfg):
        settings = resolve_llm_settings(cfg, agent)
        if not (settings.get("model") or "").strip():
            missing.append(agent)
    return missing


def _fallback_model_from_env() -> str:
    for key in ("LLM_MODEL", "OPENAI_MODEL"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def _as_optional_int(value: Any) -> Any:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Any:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
