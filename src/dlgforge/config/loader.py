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
        "TARGET_LANGUAGE": (run_cfg, "target_language", str),
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


def _as_bool(raw: str) -> bool:
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def resolve_output_dir(cfg: Dict[str, Any], project_root: Path) -> Path:
    output_dir = (cfg.get("saving", {}) or {}).get("output_dir", "outputs")
    raw = Path(str(output_dir))
    return raw if raw.is_absolute() else (project_root / raw)


def resolve_question(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("seed_question", "") or "")


def resolve_target_language(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("run", {}) or {}).get("target_language", "en") or "en")


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


def resolve_judge_reasons(cfg: Dict[str, Any]) -> List[str]:
    reasons = (cfg.get("judge", {}) or {}).get("reasons", [])
    if isinstance(reasons, list):
        return [str(item) for item in reasons if str(item).strip()]
    return []


def build_base_inputs(cfg: Dict[str, Any], project_root: Path, config_path: Path) -> Dict[str, Any]:
    user_persona, assistant_persona, persona_meta = select_personas(cfg, project_root, config_path)
    user_agent_used_name = resolve_agent_used_name(cfg, "qa_generator")
    assistant_agent_used_name = resolve_agent_used_name(cfg, "kb_responder")
    return {
        "question": resolve_question(cfg),
        "target_language": resolve_target_language(cfg),
        "question_seed": resolve_question_seed(cfg),
        "run_id": resolve_run_id(cfg),
        "resume_run_id": resolve_resume_run_id(cfg),
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
        "judge_reasons": resolve_judge_reasons(cfg),
    }
