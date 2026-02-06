from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from dlgforge.config.defaults import DEFAULT_CONFIG
from dlgforge.config.personas import select_personas
from dlgforge.llm.settings import resolve_agent_used_name
from dlgforge.utils import deep_merge


_OUTPUT_COLUMN_DEFAULTS: Dict[str, str] = {
    "messages": "messages",
    "messages_with_tools": "messages_with_tools",
    "metadata": "metadata",
    "user_reasoning": "user_reasoning",
    "assistant_reasoning": "assistant_reasoning",
    "judge": "judge",
}
_OUTPUT_COLUMN_ALIASES: Dict[str, str] = {
    "message_with_tools": "messages_with_tools",
}


def load_config(config_path: str | Path) -> Tuple[Dict[str, Any], Path, Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML object.")

    cfg = deep_merge(deepcopy(DEFAULT_CONFIG), loaded)
    _apply_env_overrides(cfg)

    project_root = path.parent.resolve()
    return cfg, path, project_root


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    run_cfg = cfg.setdefault("run", {})
    models_cfg = cfg.setdefault("models", {})
    retrieval_cfg = cfg.setdefault("retrieval", {})
    coverage_cfg = cfg.setdefault("coverage", {})
    tools_cfg = cfg.setdefault("tools", {})
    judge_cfg = cfg.setdefault("judge", {})
    saving_cfg = cfg.setdefault("saving", {})
    hf_push_cfg = saving_cfg.setdefault("hf_push", {})

    env_map = {
        "N_TURNS": (run_cfg, "n_turns", int),
        "BATCH_SIZE": (run_cfg, "batch_size", int),
        "TOTAL_SAMPLES": (run_cfg, "total_samples", int),
        "MIN_TURNS": (run_cfg, "min_turns", int),
        "MAX_TURNS": (run_cfg, "max_turns", int),
        "TURN_COUNT_DISTRIBUTION": (run_cfg, "turn_count_distribution", str),
        "TURN_COUNT_MEAN": (run_cfg, "turn_count_mean", float),
        "QUESTION": (run_cfg, "seed_question", str),
        "QUESTION_SEED": (run_cfg, "question_seed", str),
        "RUN_ID": (run_cfg, "run_id", str),
        "RESUME_RUN_ID": (run_cfg, "resume_run_id", str),
        "SEED_TOPICS_PATH": (run_cfg, "seed_topics_path", str),
        "SEED_TOPICS_VARIANT": (run_cfg, "seed_topics_variant", str),
        "SEED_TOPICS_PROBABILITY": (run_cfg, "seed_topics_probability", float),
        "SEED_TOPICS_ENABLED": (run_cfg, "seed_topics_enabled", _as_bool),
        "KB_CHUNK_SIZE": (retrieval_cfg, "chunk_size", int),
        "KB_CHUNK_OVERLAP": (retrieval_cfg, "overlap", int),
        "KB_DEFAULT_K": (retrieval_cfg, "default_k", int),
        "KB_PERSIST_DIR": (retrieval_cfg, "persist_dir", str),
        "KB_REBUILD_INDEX": (retrieval_cfg, "rebuild_index", _as_bool),
        "KB_SKIP_IF_UNCHANGED": (retrieval_cfg, "skip_if_unchanged", _as_bool),
        "KB_EMBEDDING_BACKEND": (retrieval_cfg, "embedding_backend", str),
        "KB_EMBEDDING_DEVICE": (retrieval_cfg, "embedding_device", str),
        "KB_FALLBACK_ON_CPU": (retrieval_cfg, "fallback_on_cpu", _as_bool),
        "DOC_COVERAGE_MODE": (coverage_cfg, "doc_coverage_mode", str),
        "DOC_COVERAGE_EPSILON": (coverage_cfg, "doc_coverage_epsilon", float),
        "DOC_COVERAGE_FRACTION": (coverage_cfg, "doc_coverage_fraction", float),
        "QUESTION_DEDUP_RETRIES": (coverage_cfg, "question_dedup_retries", int),
        "ENABLE_WEB_TOOLS": (tools_cfg, "web_search_enabled", _as_bool),
        "JUDGE_MODE": (judge_cfg, "mode", str),
        "JUDGE_GRANULARITY": (judge_cfg, "granularity", str),
        "JUDGE_ENABLED": (judge_cfg, "enabled", _as_bool),
        "OUTPUT_DIR": (saving_cfg, "output_dir", str),
        "HF_PUSH_ENABLED": (hf_push_cfg, "enabled", _as_bool),
        "HF_AUTO_PUSH_ON_RUN": (hf_push_cfg, "auto_push_on_run", _as_bool),
        "HF_PUSH_REPO_ID": (hf_push_cfg, "repo_id", str),
        "HF_PUSH_REPO_TYPE": (hf_push_cfg, "repo_type", str),
        "HF_PUSH_EXPORT_DIR": (hf_push_cfg, "export_dir", str),
        "HF_PUSH_INCLUDE_RUN_STATE": (hf_push_cfg, "include_run_state", _as_bool),
        "HF_PUSH_PRIVATE": (hf_push_cfg, "private", _as_bool),
        "HF_PUSH_COMMIT_MESSAGE": (hf_push_cfg, "commit_message", str),
        "HF_PUSH_SOURCE_FILE": (hf_push_cfg, "source_file", str),
        "HF_PUSH_GENERATE_STATS": (hf_push_cfg, "generate_stats", _as_bool),
        "HF_PUSH_STATS_FILE": (hf_push_cfg, "stats_file", str),
        "HF_PUSH_GENERATE_PLOTS": (hf_push_cfg, "generate_plots", _as_bool),
        "HF_PUSH_PLOTS_DIR": (hf_push_cfg, "plots_dir", str),
    }

    for env_name, (target, key, caster) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        try:
            target[key] = caster(raw)
        except Exception:
            continue

    fallback_model = os.getenv("FALLBACK_EMBEDDING_MODEL")
    if fallback_model is not None and fallback_model.strip() != "":
        models_cfg["fallback_embedding_model"] = fallback_model.strip()

    model_kwargs_json = _parse_json_env("KB_EMBEDDING_MODEL_KWARGS_JSON")
    if model_kwargs_json is not None:
        retrieval_cfg["embedding_model_kwargs"] = model_kwargs_json

    tokenizer_kwargs_json = _parse_json_env("KB_EMBEDDING_TOKENIZER_KWARGS_JSON")
    if tokenizer_kwargs_json is not None:
        retrieval_cfg["embedding_tokenizer_kwargs"] = tokenizer_kwargs_json

    encode_kwargs_json = _parse_json_env("KB_EMBEDDING_ENCODE_KWARGS_JSON")
    if encode_kwargs_json is not None:
        retrieval_cfg["embedding_encode_kwargs"] = encode_kwargs_json

    target_languages = _parse_list_env("TARGET_LANGUAGES")
    if target_languages is not None:
        run_cfg["target_languages"] = target_languages
    else:
        # Backward compatibility for legacy single-language env override.
        legacy_target_language = (os.getenv("TARGET_LANGUAGE") or "").strip()
        if legacy_target_language:
            run_cfg["target_languages"] = [legacy_target_language]


def _parse_json_env(env_name: str) -> Dict[str, Any] | None:
    raw = os.getenv(env_name)
    if raw is None or raw.strip() == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(f"{env_name} must be valid JSON object. Error: {err.msg}") from err
    if not isinstance(parsed, dict):
        raise ValueError(f"{env_name} must be a JSON object.")
    return parsed


def _parse_list_env(env_name: str) -> List[str] | None:
    raw = os.getenv(env_name)
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _as_bool(raw: str) -> bool:
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def resolve_output_dir(cfg: Dict[str, Any], project_root: Path) -> Path:
    output_dir = (cfg.get("saving", {}) or {}).get("output_dir", "outputs")
    raw = Path(str(output_dir))
    return raw if raw.is_absolute() else (project_root / raw)


def resolve_output_columns(cfg: Dict[str, Any]) -> Dict[str, str]:
    configured = dict(_OUTPUT_COLUMN_DEFAULTS)
    columns_cfg = (cfg.get("saving", {}) or {}).get("output_columns", {})
    if not isinstance(columns_cfg, dict):
        return configured

    for key, raw_value in columns_cfg.items():
        source_key = str(key).strip()
        target_key = _OUTPUT_COLUMN_ALIASES.get(source_key, source_key)
        if target_key not in configured:
            continue
        value = str(raw_value or "").strip()
        if value:
            configured[target_key] = value
    return configured


def resolve_question(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("seed_question", "") or "")


def resolve_target_languages(cfg: Dict[str, Any]) -> List[str]:
    run_cfg = cfg.get("run", {}) or {}
    raw = run_cfg.get("target_languages", [])
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list):
        raw = []
    cleaned: List[str] = []
    for item in raw:
        value = str(item or "").strip()
        if value:
            cleaned.append(value)
    deduped = list(dict.fromkeys(cleaned))
    if deduped:
        return deduped

    # Backward compatibility with older configs that still define `target_language`.
    legacy_target_language = str(run_cfg.get("target_language", "") or "").strip()
    if legacy_target_language:
        return [legacy_target_language]
    return ["en"]


def resolve_question_seed(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("question_seed", "") or "")


def resolve_run_id(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("run_id", "") or "")


def resolve_resume_run_id(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("resume_run_id", "") or "")


def resolve_n_turns(cfg: Dict[str, Any], fallback: int = 1) -> int:
    raw = (cfg.get("run", {}) or {}).get("n_turns", fallback)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else 1


def resolve_batch_size(cfg: Dict[str, Any], fallback: int = 1) -> int:
    raw = (cfg.get("run", {}) or {}).get("batch_size", fallback)
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return fallback
    return size if size > 0 else fallback


def resolve_total_samples(cfg: Dict[str, Any], fallback: int = 0) -> int:
    raw = (cfg.get("run", {}) or {}).get("total_samples", fallback)
    try:
        total = int(raw)
    except (TypeError, ValueError):
        return fallback
    return total if total >= 0 else fallback


def resolve_turn_range(cfg: Dict[str, Any], fallback: int = 1) -> Tuple[int, int]:
    default_turns = resolve_n_turns(cfg, fallback=fallback)
    run_cfg = cfg.get("run", {}) or {}

    min_raw = run_cfg.get("min_turns", 0)
    max_raw = run_cfg.get("max_turns", 0)
    try:
        min_turns = int(min_raw)
    except (TypeError, ValueError):
        min_turns = 0
    try:
        max_turns = int(max_raw)
    except (TypeError, ValueError):
        max_turns = 0

    if min_turns <= 0 and max_turns <= 0:
        return default_turns, default_turns
    if min_turns <= 0:
        min_turns = max_turns
    if max_turns <= 0:
        max_turns = min_turns
    if min_turns <= 0:
        return default_turns, default_turns
    if min_turns > max_turns:
        raise ValueError(
            f"Invalid turn range: run.min_turns ({min_turns}) must be <= run.max_turns ({max_turns})."
        )
    return min_turns, max_turns


def resolve_turn_count_distribution(cfg: Dict[str, Any], fallback: str = "poisson") -> str:
    run_cfg = cfg.get("run", {}) or {}
    raw = str(run_cfg.get("turn_count_distribution", fallback) or fallback).strip().lower()
    aliases = {"exp": "exponential", "pois": "poisson"}
    normalized = aliases.get(raw, raw)
    if normalized in {"uniform", "poisson", "exponential"}:
        return normalized
    return fallback


def resolve_turn_count_mean(cfg: Dict[str, Any], fallback: float = 0.0) -> float:
    run_cfg = cfg.get("run", {}) or {}
    raw = run_cfg.get("turn_count_mean", fallback)
    try:
        mean = float(raw)
    except (TypeError, ValueError):
        return fallback
    if mean <= 0:
        return fallback
    return mean


def resolve_retrieval_default_k(cfg: Dict[str, Any], fallback: int = 4) -> int:
    raw = (cfg.get("retrieval", {}) or {}).get("default_k", fallback)
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return fallback
    return k if k > 0 else fallback


def resolve_seed_topics_path(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("seed_topics_path", "") or "")


def resolve_seed_topics_variant(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("seed_topics_variant", "") or "")


def resolve_seed_topics_probability(cfg: Dict[str, Any]) -> float:
    raw = (cfg.get("run", {}) or {}).get("seed_topics_probability", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def resolve_seed_topics_enabled(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("run", {}) or {}).get("seed_topics_enabled", True))


def resolve_judge_mode(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("judge", {}) or {}).get("mode", "offline") or "offline").strip().lower()


def resolve_judge_enabled(cfg: Dict[str, Any]) -> bool:
    enabled = bool((cfg.get("judge", {}) or {}).get("enabled", True))
    if resolve_judge_mode(cfg) == "offline":
        return False
    return enabled


def resolve_judge_granularity(cfg: Dict[str, Any], fallback: str = "turn") -> str:
    judge_cfg = cfg.get("judge", {}) or {}
    raw = str(judge_cfg.get("granularity", fallback) or fallback).strip().lower()
    aliases = {
        "per_turn": "turn",
        "turns": "turn",
        "per_conversation": "conversation",
        "conv": "conversation",
        "conversation_level": "conversation",
    }
    normalized = aliases.get(raw, raw)
    if normalized in {"turn", "conversation"}:
        return normalized
    return fallback


def resolve_judge_reasons(cfg: Dict[str, Any]) -> List[str]:
    reasons = (cfg.get("judge", {}) or {}).get("reasons", [])
    if isinstance(reasons, list):
        return [str(item) for item in reasons if str(item).strip()]
    return []


def build_base_inputs(cfg: Dict[str, Any], project_root: Path, config_path: Path) -> Dict[str, Any]:
    user_persona, assistant_persona, persona_meta = select_personas(cfg, project_root, config_path)
    user_agent_used_name = resolve_agent_used_name(cfg, "qa_generator")
    assistant_agent_used_name = resolve_agent_used_name(cfg, "kb_responder")
    target_languages = resolve_target_languages(cfg)
    primary_language = target_languages[0]
    return {
        "question": resolve_question(cfg),
        "target_language": primary_language,
        "target_languages": target_languages,
        "question_seed": resolve_question_seed(cfg),
        "run_id": resolve_run_id(cfg),
        "resume_run_id": resolve_resume_run_id(cfg),
        "total_samples": resolve_total_samples(cfg),
        "min_turns": (cfg.get("run", {}) or {}).get("min_turns", 0),
        "max_turns": (cfg.get("run", {}) or {}).get("max_turns", 0),
        "turn_count_distribution": resolve_turn_count_distribution(cfg),
        "turn_count_mean": resolve_turn_count_mean(cfg),
        "seed_topics_path": resolve_seed_topics_path(cfg),
        "seed_topics_variant": resolve_seed_topics_variant(cfg),
        "seed_topics_probability": resolve_seed_topics_probability(cfg),
        "seed_topics_enabled": resolve_seed_topics_enabled(cfg),
        "retrieval_default_k": resolve_retrieval_default_k(cfg),
        "user_persona": user_persona,
        "assistant_persona": assistant_persona,
        "user_persona_id": persona_meta.get("user_id", ""),
        "assistant_persona_id": persona_meta.get("assistant_id", ""),
        "user_agent_used_name": user_agent_used_name,
        "assistant_agent_used_name": assistant_agent_used_name,
        "judge_enabled": resolve_judge_enabled(cfg),
        "judge_granularity": resolve_judge_granularity(cfg),
        "judge_reasons": resolve_judge_reasons(cfg),
    }
