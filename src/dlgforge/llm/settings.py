"""LLM settings resolution and agent model requirements.

"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

_AGENT_ALIAS_TO_ROLE: Dict[str, str] = {
    "qa_generator": "user",
    "kb_responder": "assistant",
    "qa_judge": "judge",
    "user": "user",
    "assistant": "assistant",
    "judge": "judge",
}
_ROLE_TO_LEGACY_AGENT: Dict[str, str] = {
    "user": "qa_generator",
    "assistant": "kb_responder",
    "judge": "qa_judge",
}
_ROLE_TO_STAGE_ENV: Dict[str, str] = {
    "user": "QA_GENERATOR",
    "assistant": "KB_RESPONDER",
    "judge": "QA_JUDGE",
}
_ROLE_TO_API_KEY_MAPPING_ENV: Dict[str, str] = {
    "user": "LLM_USER_API_KEY_ENV",
    "assistant": "LLM_ASSISTANT_API_KEY_ENV",
    "judge": "LLM_JUDGE_API_KEY_ENV",
}
_ROLE_TO_LEGACY_API_KEY_MAPPING_ENV: Dict[str, str] = {
    "user": "LLM_QA_GENERATOR_API_KEY_ENV",
    "assistant": "LLM_KB_RESPONDER_API_KEY_ENV",
    "judge": "LLM_QA_JUDGE_API_KEY_ENV",
}
_FORBIDDEN_AGENT_CREDENTIAL_KEYS: set[str] = {"api_key", "api_key_env"}
_ALLOWED_MODES: set[str] = {"api", "vllm_attach", "vllm_managed"}
LOGGER = logging.getLogger("dlgforge.llm.settings")

def resolve_llm_settings(cfg: Dict[str, Any], agent_key: str) -> Dict[str, Any]:
    """Resolve llm settings from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        agent_key (str): str value used by this operation.
    
    Returns:
        Dict[str, Any]: Resolved value after applying defaults and normalization rules.
    
    Raises:
        RuntimeError: Raised when required environment variables are missing.
        ValueError: Raised when forbidden credential fields are present in YAML config.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.llm.settings import resolve_llm_settings
        >>> resolve_llm_settings(...)
    
    """
    role = _resolve_agent_role(agent_key)
    llm_cfg = cfg.get("llm", {}) or {}
    llm_mode = _normalize_llm_mode(llm_cfg.get("mode") or llm_cfg.get("backend") or "api")

    per_agent = _resolve_per_agent_cfg(llm_cfg, role=role)
    _validate_forbidden_agent_credential_fields(per_agent, role=role)

    merged = {
        "mode": llm_mode,
        "provider": per_agent.get("provider"),
        "model": per_agent.get("model"),
        "base_url": per_agent.get("base_url"),
        "temperature": per_agent.get("temperature"),
        "max_tokens": per_agent.get("max_tokens"),
        "top_p": per_agent.get("top_p"),
        "timeout": per_agent.get("timeout"),
        "max_retries": per_agent.get("max_retries"),
        "extra": per_agent.get("extra") or {},
        "routing": llm_cfg.get("routing") if isinstance(llm_cfg.get("routing"), dict) else {},
    }

    stage_env = _ROLE_TO_STAGE_ENV[role]
    role_env = role.upper()
    env_keys = [
        "PROVIDER",
        "MODEL",
        "BASE_URL",
        "TEMPERATURE",
        "MAX_TOKENS",
        "TOP_P",
        "TIMEOUT",
        "MAX_RETRIES",
    ]
    for key in env_keys:
        env_value = _first_non_empty_env(f"LLM_{stage_env}_{key}", f"LLM_{role_env}_{key}")
        if env_value is not None:
            merged[key.lower()] = env_value

    routing = merged.get("routing")
    if not isinstance(routing, dict):
        routing = {}
    routing_strategy = _first_non_empty_env(f"LLM_{stage_env}_ROUTING_STRATEGY", f"LLM_{role_env}_ROUTING_STRATEGY")
    if routing_strategy:
        routing["strategy"] = routing_strategy
    endpoints_raw = _first_non_empty_env(
        f"LLM_{stage_env}_ROUTING_ENDPOINTS_JSON",
        f"LLM_{role_env}_ROUTING_ENDPOINTS_JSON",
        "LLM_ROUTING_ENDPOINTS_JSON",
    )
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

    mapping_env_name, target_secret_env = _resolve_api_key_mapping_env(role)
    if not target_secret_env:
        raise RuntimeError(
            "Missing required API key mapping environment variable "
            f"`{mapping_env_name}` for agent `{role}`. "
            "Set it to the name of the provider secret env var (for example `OPENAI_API_KEY`)."
        )

    api_key = str(os.getenv(target_secret_env) or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing or empty provider API key environment variable "
            f"`{target_secret_env}` (configured by `{mapping_env_name}`) for agent `{role}`."
        )
    merged["api_key"] = api_key
    merged["api_key_env"] = target_secret_env
    return merged

def resolve_agent_used_name(cfg: Dict[str, Any], agent_key: str) -> str:
    """Resolve agent used name.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        agent_key (str): str value used by this operation.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.llm.settings import resolve_agent_used_name
        >>> resolve_agent_used_name(...)
    
    """
    settings = resolve_llm_settings(cfg, agent_key)
    model = (settings.get("model") or "").strip()
    provider = (settings.get("provider") or "").strip()
    if not model:
        return agent_key
    if provider and "/" not in model:
        return f"{provider}/{model}"
    return model

def required_agents(cfg: Dict[str, Any]) -> List[str]:
    """Return required agents.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        List[str]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.llm.settings import required_agents
        >>> required_agents(...)
    
    """
    agents = ["qa_generator", "kb_responder"]
    judge_cfg = cfg.get("judge", {}) or {}
    mode = str(judge_cfg.get("mode", "offline") or "offline").strip().lower()
    enabled = bool(judge_cfg.get("enabled", True))
    if mode == "online" and enabled:
        agents.append("qa_judge")
    return agents

def missing_models(cfg: Dict[str, Any]) -> List[str]:
    """Return missing models.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        List[str]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.llm.settings import missing_models
        >>> missing_models(...)
    
    """
    missing: List[str] = []
    for agent in required_agents(cfg):
        settings = resolve_llm_settings(cfg, agent)
        if not (settings.get("model") or "").strip():
            missing.append(agent)
    return missing

def _resolve_agent_role(agent_key: str) -> str:
    raw = str(agent_key or "").strip().lower()
    return _AGENT_ALIAS_TO_ROLE.get(raw, raw if raw in {"user", "assistant", "judge"} else "user")

def _resolve_per_agent_cfg(llm_cfg: Dict[str, Any], role: str) -> Dict[str, Any]:
    agents_cfg = llm_cfg.get("agents", {})
    if not isinstance(agents_cfg, dict):
        return {}

    merged: Dict[str, Any] = {}
    legacy_agent = _ROLE_TO_LEGACY_AGENT[role]
    legacy_cfg = agents_cfg.get(legacy_agent)
    role_cfg = agents_cfg.get(role)
    if isinstance(legacy_cfg, dict):
        merged.update(legacy_cfg)
    if isinstance(role_cfg, dict):
        merged.update(role_cfg)
    return merged

def _validate_forbidden_agent_credential_fields(per_agent_cfg: Dict[str, Any], role: str) -> None:
    forbidden_keys = [key for key in _FORBIDDEN_AGENT_CREDENTIAL_KEYS if key in per_agent_cfg]
    if not forbidden_keys:
        return
    joined = ", ".join(f"`{key}`" for key in sorted(forbidden_keys))
    raise ValueError(
        "Forbidden agent credential field(s) in YAML for "
        f"`llm.agents.{role}`: {joined}. "
        "Credentials must be sourced only from environment variables."
    )

def _normalize_llm_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "api").strip().lower()
    if mode == "openai":
        mode = "api"
    return mode if mode in _ALLOWED_MODES else "api"

def _first_non_empty_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value not in {None, ""}:
            return value
    return None

def _resolve_api_key_mapping_env(role: str) -> tuple[str, str]:
    canonical_mapping_env = _ROLE_TO_API_KEY_MAPPING_ENV[role]
    legacy_mapping_env = _ROLE_TO_LEGACY_API_KEY_MAPPING_ENV[role]
    canonical_value = str(os.getenv(canonical_mapping_env) or "").strip()
    legacy_value = str(os.getenv(legacy_mapping_env) or "").strip()

    if canonical_value:
        if legacy_value and legacy_value != canonical_value:
            LOGGER.warning(
                "Both `%s` and deprecated `%s` are set for role `%s`; using `%s`.",
                canonical_mapping_env,
                legacy_mapping_env,
                role,
                canonical_mapping_env,
            )
        return canonical_mapping_env, canonical_value

    if legacy_value:
        LOGGER.warning(
            "DEPRECATED: `%s` is legacy; use `%s` for role `%s`.",
            legacy_mapping_env,
            canonical_mapping_env,
            role,
        )
        return legacy_mapping_env, legacy_value

    return canonical_mapping_env, ""

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
