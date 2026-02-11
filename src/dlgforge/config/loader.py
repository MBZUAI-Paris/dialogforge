"""Configuration loading and resolver helpers.

"""

from __future__ import annotations

import json
import logging
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
_LEGACY_AGENT_TO_ROLE: Dict[str, str] = {
    "qa_generator": "user",
    "kb_responder": "assistant",
    "qa_judge": "judge",
}
_FORBIDDEN_LLM_GLOBAL_KEYS: set[str] = {
    "provider",
    "model",
    "base_url",
    "api_key",
    "api_key_env",
    "temperature",
    "max_tokens",
    "top_p",
    "timeout",
    "max_retries",
    "extra",
}
_FORBIDDEN_AGENT_CREDENTIAL_KEYS: set[str] = {"api_key", "api_key_env"}
LOGGER = logging.getLogger("dlgforge.config")

def load_config(config_path: str | Path) -> Tuple[Dict[str, Any], Path, Path]:
    """Load config.
    
    Args:
        config_path (str | Path): Path to a configuration file.
    
    Returns:
        Tuple[Dict[str, Any], Path, Path]: Loaded value parsed from upstream sources.
    
    Raises:
        FileNotFoundError: Raised when validation or runtime requirements are not met.
        ValueError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import load_config
        >>> load_config(...)
    
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML object.")
    _validate_forbidden_llm_yaml_keys(loaded)

    cfg = deep_merge(deepcopy(DEFAULT_CONFIG), loaded)
    _normalize_non_llm_config(cfg)
    _apply_env_overrides(cfg)
    _normalize_non_llm_config(cfg)
    _normalize_llm_config(cfg)

    project_root = path.parent.resolve()
    return cfg, path, project_root

def _normalize_non_llm_config(cfg: Dict[str, Any]) -> None:
    run_cfg = cfg.setdefault("run", {})
    _normalize_turns_config(run_cfg)
    _normalize_run_data_config(run_cfg)
    _normalize_distributed_config(cfg, run_cfg)
    _normalize_tools_config(cfg)
    _normalize_retrieval_config(cfg)


def _normalize_tools_config(cfg: Dict[str, Any]) -> None:
    tools_cfg = cfg.setdefault("tools", {})
    if not isinstance(tools_cfg, dict):
        tools_cfg = {}
        cfg["tools"] = tools_cfg

    web_search_cfg = tools_cfg.get("web_search", {})
    if not isinstance(web_search_cfg, dict):
        web_search_cfg = {}
    default_tools_cfg = DEFAULT_CONFIG.get("tools", {}) if isinstance(DEFAULT_CONFIG.get("tools"), dict) else {}
    default_web_search_cfg = (
        default_tools_cfg.get("web_search", {}) if isinstance(default_tools_cfg.get("web_search"), dict) else {}
    )
    if "web_search_enabled" in tools_cfg and web_search_cfg.get("enabled") == default_web_search_cfg.get("enabled", False):
        web_search_cfg["enabled"] = tools_cfg.get("web_search_enabled")
    if (
        "serper_num_results" in tools_cfg
        and web_search_cfg.get("serper_num_results") == default_web_search_cfg.get("serper_num_results", 5)
    ):
        web_search_cfg["serper_num_results"] = tools_cfg.get("serper_num_results")
    if "serper_timeout" in tools_cfg and web_search_cfg.get("serper_timeout") == default_web_search_cfg.get("serper_timeout", 30):
        web_search_cfg["serper_timeout"] = tools_cfg.get("serper_timeout")
    web_search_cfg.setdefault("enabled", False)
    web_search_cfg.setdefault("serper_num_results", 5)
    web_search_cfg.setdefault("serper_timeout", 30)
    try:
        web_search_cfg["serper_num_results"] = int(web_search_cfg.get("serper_num_results", 5))
    except (TypeError, ValueError):
        web_search_cfg["serper_num_results"] = 5
    try:
        web_search_cfg["serper_timeout"] = int(web_search_cfg.get("serper_timeout", 30))
    except (TypeError, ValueError):
        web_search_cfg["serper_timeout"] = 30
    web_search_cfg["enabled"] = _as_bool(web_search_cfg.get("enabled", False))
    tools_cfg["web_search"] = web_search_cfg

    top_level_retrieval = cfg.get("retrieval")
    tools_retrieval = tools_cfg.get("retrieval")
    default_retrieval_cfg = (
        default_tools_cfg.get("retrieval", {}) if isinstance(default_tools_cfg.get("retrieval"), dict) else {}
    )
    if isinstance(top_level_retrieval, dict) and top_level_retrieval is not tools_retrieval:
        if isinstance(tools_retrieval, dict) and tools_retrieval != default_retrieval_cfg:
            tools_cfg["retrieval"] = deep_merge(dict(top_level_retrieval), tools_retrieval)
        else:
            tools_cfg["retrieval"] = dict(top_level_retrieval)
        LOGGER.warning(
            "DEPRECATED: top-level `retrieval` is legacy; use `tools.retrieval`.",
        )
    elif not isinstance(tools_retrieval, dict):
        tools_cfg["retrieval"] = {}

    # Legacy mirrors kept for compatibility with unchanged code paths.
    tools_cfg["web_search_enabled"] = bool(web_search_cfg["enabled"])
    tools_cfg["serper_num_results"] = int(web_search_cfg["serper_num_results"])
    tools_cfg["serper_timeout"] = int(web_search_cfg["serper_timeout"])
    cfg["retrieval"] = tools_cfg["retrieval"]

def _normalize_turns_config(run_cfg: Dict[str, Any]) -> None:
    turns_cfg = run_cfg.setdefault("turns", {})
    if not isinstance(turns_cfg, dict):
        turns_cfg = {}
        run_cfg["turns"] = turns_cfg

    legacy_n_turns_set = "n_turns" in run_cfg and str(run_cfg.get("n_turns") or "").strip() != ""
    legacy_min_turns_set = "min_turns" in run_cfg and str(run_cfg.get("min_turns") or "").strip() != ""
    legacy_max_turns_set = "max_turns" in run_cfg and str(run_cfg.get("max_turns") or "").strip() != ""

    def _as_int_or_none(raw: Any) -> int | None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value

    mirrored_legacy_values = False
    turns_mode_existing = str(turns_cfg.get("mode") or "").strip().lower()
    if turns_mode_existing in {"exact", "range"}:
        exact_existing = _as_int_or_none(turns_cfg.get("exact"))
        min_existing = _as_int_or_none(turns_cfg.get("min"))
        max_existing = _as_int_or_none(turns_cfg.get("max"))
        legacy_n_existing = _as_int_or_none(run_cfg.get("n_turns"))
        legacy_min_existing = _as_int_or_none(run_cfg.get("min_turns"))
        legacy_max_existing = _as_int_or_none(run_cfg.get("max_turns"))
        if turns_mode_existing == "exact":
            expected = exact_existing if exact_existing is not None else max_existing
            mirrored_legacy_values = (
                expected is not None
                and legacy_n_existing == expected
                and (legacy_min_existing in {None, expected})
                and (legacy_max_existing in {None, expected})
            )
        else:
            mirrored_legacy_values = (
                min_existing is not None
                and max_existing is not None
                and legacy_n_existing in {None, max_existing}
                and legacy_min_existing in {None, min_existing}
                and legacy_max_existing in {None, max_existing}
            )

    mode_raw = str(turns_cfg.get("mode") or "").strip().lower()
    if legacy_n_turns_set and (legacy_min_turns_set or legacy_max_turns_set) and not mirrored_legacy_values:
        raise ValueError("Both `run.n_turns` and `run.min_turns`/`run.max_turns` are set; use only one turn mode.")
    if not mode_raw:
        mode_raw = "exact" if legacy_n_turns_set else "range"
    if legacy_n_turns_set and not (legacy_min_turns_set or legacy_max_turns_set) and mode_raw == "range":
        mode_raw = "exact"
    if mode_raw not in {"exact", "range"}:
        raise ValueError("`run.turns.mode` must be either `exact` or `range`.")
    turns_cfg["mode"] = mode_raw

    if legacy_n_turns_set:
        turns_cfg["exact"] = run_cfg.get("n_turns")
    if legacy_min_turns_set:
        turns_cfg["min"] = run_cfg.get("min_turns")
    if legacy_max_turns_set:
        turns_cfg["max"] = run_cfg.get("max_turns")
    if "turn_count_distribution" in run_cfg and str(run_cfg.get("turn_count_distribution") or "").strip():
        turns_cfg["distribution"] = run_cfg.get("turn_count_distribution")
    if "turn_count_mean" in run_cfg and str(run_cfg.get("turn_count_mean") or "").strip():
        turns_cfg["mean"] = run_cfg.get("turn_count_mean")

    distribution_raw = str(turns_cfg.get("distribution") or "poisson").strip().lower()
    aliases = {"exp": "exponential", "pois": "poisson"}
    turns_cfg["distribution"] = aliases.get(distribution_raw, distribution_raw)
    if turns_cfg["distribution"] not in {"uniform", "poisson", "exponential"}:
        turns_cfg["distribution"] = "poisson"

    mean_raw = turns_cfg.get("mean")
    if mean_raw in {None, ""}:
        turns_cfg["mean"] = None
    else:
        try:
            mean_value = float(mean_raw)
        except (TypeError, ValueError):
            mean_value = None
        turns_cfg["mean"] = mean_value if (mean_value is not None and mean_value > 0) else None

    def _positive_int(raw: Any, fallback: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = fallback
        return value if value > 0 else fallback

    if turns_cfg["mode"] == "exact":
        exact_turns = _positive_int(turns_cfg.get("exact"), fallback=1)
        turns_cfg["exact"] = exact_turns
        turns_cfg["min"] = exact_turns
        turns_cfg["max"] = exact_turns
    else:
        min_turns = _positive_int(turns_cfg.get("min"), fallback=1)
        max_turns = _positive_int(turns_cfg.get("max"), fallback=min_turns)
        if min_turns > max_turns:
            raise ValueError(
                f"Invalid turn range: run.turns.min ({min_turns}) must be <= run.turns.max ({max_turns})."
            )
        turns_cfg["exact"] = None
        turns_cfg["min"] = min_turns
        turns_cfg["max"] = max_turns

    # Legacy mirrors kept for compatibility with unchanged code paths.
    run_cfg["n_turns"] = int(turns_cfg["max"] if turns_cfg["mode"] == "range" else turns_cfg["exact"] or 1)
    run_cfg["min_turns"] = int(turns_cfg["min"])
    run_cfg["max_turns"] = int(turns_cfg["max"])
    run_cfg["turn_count_distribution"] = str(turns_cfg["distribution"])
    run_cfg["turn_count_mean"] = float(turns_cfg["mean"] or 0.0)

def _normalize_run_data_config(run_cfg: Dict[str, Any]) -> None:
    data_cfg = run_cfg.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
        run_cfg["data"] = data_cfg
    seeding_cfg = data_cfg.setdefault("seeding", {})
    if not isinstance(seeding_cfg, dict):
        seeding_cfg = {}
        data_cfg["seeding"] = seeding_cfg
    topics_cfg = seeding_cfg.setdefault("topics", {})
    if not isinstance(topics_cfg, dict):
        topics_cfg = {}
        seeding_cfg["topics"] = topics_cfg

    legacy_seed_question = str(run_cfg.get("seed_question") or "").strip()
    legacy_question_seed = str(run_cfg.get("question_seed") or "").strip()
    if legacy_seed_question and legacy_question_seed and legacy_seed_question != legacy_question_seed:
        raise ValueError(
            "Both `run.seed_question` and `run.question_seed` are set with different values; "
            "use only one or migrate to `run.data.seeding.question`."
        )

    canonical_question = str(seeding_cfg.get("question") or "").strip()
    if not canonical_question:
        canonical_question = legacy_seed_question or legacy_question_seed
    seeding_cfg["question"] = canonical_question

    if "seed_topics_path" in run_cfg and str(run_cfg.get("seed_topics_path") or "").strip():
        topics_cfg["path"] = run_cfg.get("seed_topics_path")
    if "seed_topics_variant" in run_cfg and str(run_cfg.get("seed_topics_variant") or "").strip():
        topics_cfg["variant"] = run_cfg.get("seed_topics_variant")
    if "seed_topics_probability" in run_cfg and str(run_cfg.get("seed_topics_probability") or "").strip():
        topics_cfg["probability"] = run_cfg.get("seed_topics_probability")
    if "seed_topics_enabled" in run_cfg:
        topics_cfg["enabled"] = run_cfg.get("seed_topics_enabled")

    topics_cfg.setdefault("path", "data/seeds/topics.yaml")
    topics_cfg.setdefault("variant", "")
    try:
        topics_cfg["probability"] = float(topics_cfg.get("probability", 0.35))
    except (TypeError, ValueError):
        topics_cfg["probability"] = 0.35
    topics_cfg["enabled"] = bool(topics_cfg.get("enabled", True))

    # Legacy mirrors kept for compatibility with unchanged code paths.
    run_cfg["seed_question"] = canonical_question
    run_cfg["question_seed"] = canonical_question
    run_cfg["seed_topics_path"] = str(topics_cfg.get("path") or "")
    run_cfg["seed_topics_variant"] = str(topics_cfg.get("variant") or "")
    run_cfg["seed_topics_probability"] = float(topics_cfg.get("probability") or 0.0)
    run_cfg["seed_topics_enabled"] = bool(topics_cfg.get("enabled", True))

def _normalize_distributed_config(cfg: Dict[str, Any], run_cfg: Dict[str, Any]) -> None:
    distributed_cfg = run_cfg.setdefault("distributed", {})
    if not isinstance(distributed_cfg, dict):
        distributed_cfg = {}
        run_cfg["distributed"] = distributed_cfg

    backend = str(distributed_cfg.get("backend") or distributed_cfg.get("executor") or "ray").strip().lower() or "ray"
    distributed_cfg["backend"] = backend
    distributed_cfg["executor"] = backend  # compatibility alias

    spawn_cfg = distributed_cfg.setdefault("spawn", {})
    if not isinstance(spawn_cfg, dict):
        spawn_cfg = {}
        distributed_cfg["spawn"] = spawn_cfg
    spawn_cfg.setdefault("coordinator", True)
    spawn_cfg.setdefault("workers", True)

    legacy_ray_cfg = cfg.get("ray", {})
    if not isinstance(legacy_ray_cfg, dict):
        legacy_ray_cfg = {}

    ray_cfg = distributed_cfg.get("ray", {})
    if not isinstance(ray_cfg, dict):
        ray_cfg = {}
    if legacy_ray_cfg:
        ray_cfg = deep_merge(ray_cfg, legacy_ray_cfg)
    actor_cfg = ray_cfg.setdefault("actor", {})
    if not isinstance(actor_cfg, dict):
        actor_cfg = {}
        ray_cfg["actor"] = actor_cfg
    ray_cfg.setdefault("address", "auto")
    ray_cfg.setdefault("auto_start_local", True)
    ray_cfg.setdefault("namespace", "dlgforge")
    actor_cfg.setdefault("num_cpus", 1.0)
    actor_cfg.setdefault("num_gpus", 0.0)
    actor_cfg.setdefault("coordinator_num_cpus", 1.0)
    actor_cfg.setdefault("replicas_qa", 1)
    actor_cfg.setdefault("replicas_complete", 1)
    distributed_cfg["ray"] = ray_cfg

    # Legacy mirror kept for compatibility with unchanged code paths.
    cfg["ray"] = ray_cfg

def _normalize_retrieval_config(cfg: Dict[str, Any]) -> None:
    tools_cfg = cfg.setdefault("tools", {})
    if not isinstance(tools_cfg, dict):
        tools_cfg = {}
        cfg["tools"] = tools_cfg
    retrieval_cfg = tools_cfg.get("retrieval", {})
    if not isinstance(retrieval_cfg, dict):
        retrieval_cfg = {}
        tools_cfg["retrieval"] = retrieval_cfg
    cfg["retrieval"] = retrieval_cfg
    models_cfg = cfg.setdefault("models", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}
        cfg["models"] = models_cfg

    default_tools_cfg = DEFAULT_CONFIG.get("tools", {}) if isinstance(DEFAULT_CONFIG.get("tools"), dict) else {}
    default_retrieval_cfg = (
        default_tools_cfg.get("retrieval", {}) if isinstance(default_tools_cfg.get("retrieval"), dict) else {}
    )
    default_top_k = int(default_retrieval_cfg.get("top_k", 4) or 4)
    top_k_raw = retrieval_cfg.get("top_k", default_top_k)
    if "default_k" in retrieval_cfg and top_k_raw == default_top_k:
        top_k_raw = retrieval_cfg.get("default_k")
    try:
        top_k = int(top_k_raw)
    except (TypeError, ValueError):
        top_k = 4
    if top_k <= 0:
        raise ValueError("`tools.retrieval.top_k` must be > 0.")
    retrieval_cfg["top_k"] = top_k

    chunking_cfg = retrieval_cfg.setdefault("chunking", {})
    if not isinstance(chunking_cfg, dict):
        chunking_cfg = {}
        retrieval_cfg["chunking"] = chunking_cfg
    default_chunking_cfg = default_retrieval_cfg.get("chunking", {}) if isinstance(default_retrieval_cfg.get("chunking"), dict) else {}
    if "chunk_size" in retrieval_cfg and chunking_cfg.get("chunk_size") == default_chunking_cfg.get("chunk_size", 750):
        chunking_cfg["chunk_size"] = retrieval_cfg.get("chunk_size")
    elif chunking_cfg.get("chunk_size") in {None, ""}:
        chunking_cfg["chunk_size"] = retrieval_cfg.get("chunk_size", 750)
    if "overlap" in retrieval_cfg and chunking_cfg.get("chunk_overlap") == default_chunking_cfg.get("chunk_overlap", 150):
        chunking_cfg["chunk_overlap"] = retrieval_cfg.get("overlap")
    elif chunking_cfg.get("chunk_overlap") in {None, ""}:
        chunking_cfg["chunk_overlap"] = retrieval_cfg.get("overlap", 150)
    try:
        chunk_size = int(chunking_cfg.get("chunk_size", 750))
    except (TypeError, ValueError):
        chunk_size = 750
    try:
        chunk_overlap = int(chunking_cfg.get("chunk_overlap", 150))
    except (TypeError, ValueError):
        chunk_overlap = 150
    if chunk_size <= 0:
        raise ValueError("`tools.retrieval.chunking.chunk_size` must be > 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "`tools.retrieval.chunking.chunk_overlap` must be >= 0 and < `tools.retrieval.chunking.chunk_size`."
        )
    chunking_cfg["chunk_size"] = chunk_size
    chunking_cfg["chunk_overlap"] = chunk_overlap

    index_cfg = retrieval_cfg.setdefault("index", {})
    if not isinstance(index_cfg, dict):
        index_cfg = {}
        retrieval_cfg["index"] = index_cfg
    default_index_cfg = default_retrieval_cfg.get("index", {}) if isinstance(default_retrieval_cfg.get("index"), dict) else {}
    if "persist_dir" in retrieval_cfg and index_cfg.get("persist_dir") == default_index_cfg.get("persist_dir", "knowledge_index"):
        index_cfg["persist_dir"] = retrieval_cfg.get("persist_dir")
    elif index_cfg.get("persist_dir") in {None, ""}:
        index_cfg["persist_dir"] = retrieval_cfg.get("persist_dir", "knowledge_index")
    if "rebuild_index" in retrieval_cfg and index_cfg.get("rebuild") == default_index_cfg.get("rebuild", False):
        index_cfg["rebuild"] = retrieval_cfg.get("rebuild_index")
    elif index_cfg.get("rebuild") in {None, ""}:
        index_cfg["rebuild"] = retrieval_cfg.get("rebuild_index", False)
    if "skip_if_unchanged" in retrieval_cfg and index_cfg.get("skip_if_unchanged") == default_index_cfg.get("skip_if_unchanged", True):
        index_cfg["skip_if_unchanged"] = retrieval_cfg.get("skip_if_unchanged")
    elif index_cfg.get("skip_if_unchanged") in {None, ""}:
        index_cfg["skip_if_unchanged"] = retrieval_cfg.get("skip_if_unchanged", True)
    index_cfg["persist_dir"] = str(index_cfg.get("persist_dir") or "knowledge_index")
    index_cfg["rebuild"] = bool(index_cfg.get("rebuild", False))
    index_cfg["skip_if_unchanged"] = bool(index_cfg.get("skip_if_unchanged", True))

    embeddings_cfg = retrieval_cfg.setdefault("embeddings", {})
    if not isinstance(embeddings_cfg, dict):
        embeddings_cfg = {}
        retrieval_cfg["embeddings"] = embeddings_cfg
    default_embeddings_cfg = (
        default_retrieval_cfg.get("embeddings", {}) if isinstance(default_retrieval_cfg.get("embeddings"), dict) else {}
    )
    embeddings_cfg.setdefault("backend", retrieval_cfg.get("embedding_backend", "sentence_transformers"))
    if "embedding_backend" in retrieval_cfg and embeddings_cfg.get("backend") == default_embeddings_cfg.get("backend"):
        embeddings_cfg["backend"] = retrieval_cfg.get("embedding_backend")
    if "embedding_model" in retrieval_cfg and embeddings_cfg.get("model") == default_embeddings_cfg.get("model"):
        embeddings_cfg["model"] = retrieval_cfg.get("embedding_model")
    elif "embedding_model" in models_cfg and embeddings_cfg.get("model") == default_embeddings_cfg.get("model"):
        embeddings_cfg["model"] = models_cfg.get("embedding_model")
    else:
        embeddings_cfg.setdefault(
            "model",
            retrieval_cfg.get("embedding_model")
            or models_cfg.get("embedding_model")
            or "sentence-transformers/all-MiniLM-L6-v2",
        )
    if "fallback_embedding_model" in models_cfg and embeddings_cfg.get("fallback_model") == default_embeddings_cfg.get("fallback_model"):
        embeddings_cfg["fallback_model"] = models_cfg.get("fallback_embedding_model")
    else:
        embeddings_cfg.setdefault(
            "fallback_model",
            models_cfg.get("fallback_embedding_model")
            or embeddings_cfg.get("model")
            or "sentence-transformers/all-MiniLM-L6-v2",
        )
    if "embedding_device" in retrieval_cfg and embeddings_cfg.get("device") == default_embeddings_cfg.get("device"):
        embeddings_cfg["device"] = retrieval_cfg.get("embedding_device")
    else:
        embeddings_cfg.setdefault("device", retrieval_cfg.get("embedding_device", "auto"))
    if "fallback_on_cpu" in retrieval_cfg and embeddings_cfg.get("fallback_on_cpu") == default_embeddings_cfg.get("fallback_on_cpu"):
        embeddings_cfg["fallback_on_cpu"] = retrieval_cfg.get("fallback_on_cpu")
    else:
        embeddings_cfg.setdefault("fallback_on_cpu", retrieval_cfg.get("fallback_on_cpu", True))
    if "embedding_model_kwargs" in retrieval_cfg and embeddings_cfg.get("model_kwargs") == default_embeddings_cfg.get("model_kwargs"):
        embeddings_cfg["model_kwargs"] = retrieval_cfg.get("embedding_model_kwargs", {})
    else:
        embeddings_cfg.setdefault("model_kwargs", retrieval_cfg.get("embedding_model_kwargs", {}))
    if (
        "embedding_tokenizer_kwargs" in retrieval_cfg
        and embeddings_cfg.get("tokenizer_kwargs") == default_embeddings_cfg.get("tokenizer_kwargs")
    ):
        embeddings_cfg["tokenizer_kwargs"] = retrieval_cfg.get("embedding_tokenizer_kwargs", {})
    else:
        embeddings_cfg.setdefault("tokenizer_kwargs", retrieval_cfg.get("embedding_tokenizer_kwargs", {}))
    if "embedding_encode_kwargs" in retrieval_cfg and embeddings_cfg.get("encode_kwargs") == default_embeddings_cfg.get("encode_kwargs"):
        embeddings_cfg["encode_kwargs"] = retrieval_cfg.get("embedding_encode_kwargs", {})
    else:
        embeddings_cfg.setdefault("encode_kwargs", retrieval_cfg.get("embedding_encode_kwargs", {}))

    reranker_cfg = retrieval_cfg.setdefault("reranker", {})
    if not isinstance(reranker_cfg, dict):
        reranker_cfg = {}
        retrieval_cfg["reranker"] = reranker_cfg
    default_reranker_cfg = (
        default_retrieval_cfg.get("reranker", {}) if isinstance(default_retrieval_cfg.get("reranker"), dict) else {}
    )
    if "use_reranker" in models_cfg and reranker_cfg.get("enabled") == default_reranker_cfg.get("enabled", False):
        reranker_cfg["enabled"] = models_cfg.get("use_reranker")
    else:
        reranker_cfg.setdefault("enabled", models_cfg.get("use_reranker", False))
    if "reranker_model" in models_cfg and reranker_cfg.get("model") == default_reranker_cfg.get("model"):
        reranker_cfg["model"] = models_cfg.get("reranker_model")
    else:
        reranker_cfg.setdefault("model", models_cfg.get("reranker_model", "Qwen/Qwen3-Reranker-4B"))
    if "reranker_backend" in models_cfg and reranker_cfg.get("backend") == default_reranker_cfg.get("backend"):
        reranker_cfg["backend"] = models_cfg.get("reranker_backend")
    else:
        reranker_cfg.setdefault("backend", models_cfg.get("reranker_backend", "qwen3"))
    if "reranker_instruction" in models_cfg and reranker_cfg.get("instruction") == default_reranker_cfg.get("instruction"):
        reranker_cfg["instruction"] = models_cfg.get("reranker_instruction")
    else:
        reranker_cfg.setdefault("instruction", models_cfg.get("reranker_instruction", ""))
    if "reranker_max_length" in models_cfg and reranker_cfg.get("max_length") == default_reranker_cfg.get("max_length"):
        reranker_cfg["max_length"] = models_cfg.get("reranker_max_length")
    else:
        reranker_cfg.setdefault("max_length", models_cfg.get("reranker_max_length", 8192))
    if "reranker_candidates" in models_cfg and reranker_cfg.get("candidates") == default_reranker_cfg.get("candidates"):
        reranker_cfg["candidates"] = models_cfg.get("reranker_candidates")
    else:
        reranker_cfg.setdefault("candidates", models_cfg.get("reranker_candidates", 12))
    if "reranker_batch_size" in models_cfg and reranker_cfg.get("batch_size") == default_reranker_cfg.get("batch_size"):
        reranker_cfg["batch_size"] = models_cfg.get("reranker_batch_size")
    else:
        reranker_cfg.setdefault("batch_size", models_cfg.get("reranker_batch_size", 16))
    if "qwen3_reranker_cmd" in models_cfg and reranker_cfg.get("cmd") == default_reranker_cfg.get("cmd"):
        reranker_cfg["cmd"] = models_cfg.get("qwen3_reranker_cmd") or None
    else:
        reranker_cfg.setdefault("cmd", models_cfg.get("qwen3_reranker_cmd") or None)

    # Legacy mirrors kept for compatibility with unchanged code paths.
    retrieval_cfg["default_k"] = retrieval_cfg["top_k"]
    retrieval_cfg["chunk_size"] = chunking_cfg["chunk_size"]
    retrieval_cfg["overlap"] = chunking_cfg["chunk_overlap"]
    retrieval_cfg["persist_dir"] = index_cfg["persist_dir"]
    retrieval_cfg["rebuild_index"] = index_cfg["rebuild"]
    retrieval_cfg["skip_if_unchanged"] = index_cfg["skip_if_unchanged"]
    retrieval_cfg["embedding_backend"] = embeddings_cfg.get("backend")
    retrieval_cfg["embedding_model"] = embeddings_cfg.get("model")
    retrieval_cfg["embedding_device"] = embeddings_cfg.get("device")
    retrieval_cfg["fallback_on_cpu"] = bool(embeddings_cfg.get("fallback_on_cpu", True))
    retrieval_cfg["embedding_model_kwargs"] = embeddings_cfg.get("model_kwargs", {})
    retrieval_cfg["embedding_tokenizer_kwargs"] = embeddings_cfg.get("tokenizer_kwargs", {})
    retrieval_cfg["embedding_encode_kwargs"] = embeddings_cfg.get("encode_kwargs", {})

    models_cfg["embedding_model"] = str(embeddings_cfg.get("model") or "")
    models_cfg["fallback_embedding_model"] = str(embeddings_cfg.get("fallback_model") or "")
    models_cfg["use_reranker"] = bool(reranker_cfg.get("enabled", False))
    models_cfg["reranker_model"] = str(reranker_cfg.get("model") or "")
    models_cfg["reranker_backend"] = str(reranker_cfg.get("backend") or "")
    models_cfg["reranker_instruction"] = str(reranker_cfg.get("instruction") or "")
    models_cfg["reranker_max_length"] = int(reranker_cfg.get("max_length") or 8192)
    models_cfg["reranker_candidates"] = int(reranker_cfg.get("candidates") or 12)
    models_cfg["reranker_batch_size"] = int(reranker_cfg.get("batch_size") or 16)
    models_cfg["qwen3_reranker_cmd"] = reranker_cfg.get("cmd") or ""

def _validate_forbidden_llm_yaml_keys(loaded: Dict[str, Any]) -> None:
    llm_cfg = loaded.get("llm", {})
    if not isinstance(llm_cfg, dict):
        return

    forbidden_global = [key for key in _FORBIDDEN_LLM_GLOBAL_KEYS if key in llm_cfg]
    if forbidden_global:
        joined = ", ".join(f"`llm.{key}`" for key in sorted(forbidden_global))
        raise ValueError(
            "Forbidden LLM global key(s) in YAML: "
            f"{joined}. Define model/provider/base_url/sampling only under `llm.agents.<role>.*`."
        )

    agents_cfg = llm_cfg.get("agents", {})
    if not isinstance(agents_cfg, dict):
        return

    for agent_key, agent_cfg in agents_cfg.items():
        if not isinstance(agent_cfg, dict):
            continue
        forbidden_agent_keys = [key for key in _FORBIDDEN_AGENT_CREDENTIAL_KEYS if key in agent_cfg]
        if forbidden_agent_keys:
            joined = ", ".join(f"`llm.agents.{agent_key}.{key}`" for key in sorted(forbidden_agent_keys))
            raise ValueError(
                "Forbidden credential key(s) in YAML: "
                f"{joined}. Agent credentials must come only from environment variables."
            )

def _normalize_llm_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "api").strip().lower()
    if mode == "openai":
        return "api"
    return mode or "api"

def _normalize_llm_config(cfg: Dict[str, Any]) -> None:
    llm_cfg = cfg.setdefault("llm", {})
    llm_cfg["mode"] = _normalize_llm_mode(llm_cfg.get("mode") or llm_cfg.get("backend") or "api")
    llm_cfg.setdefault("routing", {"strategy": "weighted_least_inflight", "endpoints": []})
    llm_cfg.setdefault("vllm", {})
    default_agents_cfg = (
        ((DEFAULT_CONFIG.get("llm", {}) or {}).get("agents", {}) or {})
        if isinstance((DEFAULT_CONFIG.get("llm", {}) or {}).get("agents", {}), dict)
        else {}
    )

    agents_cfg_raw = llm_cfg.get("agents")
    if not isinstance(agents_cfg_raw, dict):
        agents_cfg_raw = {}

    for legacy_key, canonical_role in _LEGACY_AGENT_TO_ROLE.items():
        legacy_cfg = agents_cfg_raw.get(legacy_key)
        canonical_cfg = agents_cfg_raw.get(canonical_role)
        if not isinstance(legacy_cfg, dict):
            continue
        if isinstance(canonical_cfg, dict):
            merged_agent_cfg = dict(legacy_cfg)
            default_role_cfg = default_agents_cfg.get(canonical_role, {})
            if not isinstance(default_role_cfg, dict):
                default_role_cfg = {}
            for key, value in canonical_cfg.items():
                if key not in default_role_cfg or value != default_role_cfg[key]:
                    merged_agent_cfg[key] = value
            agents_cfg_raw[canonical_role] = merged_agent_cfg
        else:
            agents_cfg_raw[canonical_role] = dict(legacy_cfg)
        LOGGER.warning(
            "DEPRECATED: `llm.agents.%s` is legacy; use `llm.agents.%s`.",
            legacy_key,
            canonical_role,
        )

    agents_cfg_raw.setdefault("user", {})
    agents_cfg_raw.setdefault("assistant", {})
    agents_cfg_raw.setdefault("judge", {})
    llm_cfg["agents"] = agents_cfg_raw

def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    run_cfg = cfg.setdefault("run", {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}
        cfg["run"] = run_cfg
    turns_cfg = run_cfg.setdefault("turns", {})
    if not isinstance(turns_cfg, dict):
        turns_cfg = {}
        run_cfg["turns"] = turns_cfg
    run_data_cfg = run_cfg.setdefault("data", {})
    if not isinstance(run_data_cfg, dict):
        run_data_cfg = {}
        run_cfg["data"] = run_data_cfg
    run_seeding_cfg = run_data_cfg.setdefault("seeding", {})
    if not isinstance(run_seeding_cfg, dict):
        run_seeding_cfg = {}
        run_data_cfg["seeding"] = run_seeding_cfg
    run_topics_cfg = run_seeding_cfg.setdefault("topics", {})
    if not isinstance(run_topics_cfg, dict):
        run_topics_cfg = {}
        run_seeding_cfg["topics"] = run_topics_cfg
    distributed_cfg = run_cfg.setdefault("distributed", {})
    if not isinstance(distributed_cfg, dict):
        distributed_cfg = {}
        run_cfg["distributed"] = distributed_cfg
    distributed_ray_cfg = distributed_cfg.setdefault("ray", {})
    if not isinstance(distributed_ray_cfg, dict):
        distributed_ray_cfg = {}
        distributed_cfg["ray"] = distributed_ray_cfg
    distributed_spawn_cfg = distributed_cfg.setdefault("spawn", {})
    if not isinstance(distributed_spawn_cfg, dict):
        distributed_spawn_cfg = {}
        distributed_cfg["spawn"] = distributed_spawn_cfg
    ray_actor_cfg = distributed_ray_cfg.setdefault("actor", {})
    if not isinstance(ray_actor_cfg, dict):
        ray_actor_cfg = {}
        distributed_ray_cfg["actor"] = ray_actor_cfg
    store_cfg = cfg.setdefault("store", {})
    if not isinstance(store_cfg, dict):
        store_cfg = {}
        cfg["store"] = store_cfg
    postgres_cfg = store_cfg.setdefault("postgres", {})
    if not isinstance(postgres_cfg, dict):
        postgres_cfg = {}
        store_cfg["postgres"] = postgres_cfg
    tools_cfg = cfg.setdefault("tools", {})
    if not isinstance(tools_cfg, dict):
        tools_cfg = {}
        cfg["tools"] = tools_cfg
    web_search_cfg = tools_cfg.setdefault("web_search", {})
    if not isinstance(web_search_cfg, dict):
        web_search_cfg = {}
        tools_cfg["web_search"] = web_search_cfg
    retrieval_cfg = tools_cfg.setdefault("retrieval", {})
    if not isinstance(retrieval_cfg, dict):
        retrieval_cfg = {}
        tools_cfg["retrieval"] = retrieval_cfg
    cfg["retrieval"] = retrieval_cfg
    retrieval_chunking_cfg = retrieval_cfg.setdefault("chunking", {})
    if not isinstance(retrieval_chunking_cfg, dict):
        retrieval_chunking_cfg = {}
        retrieval_cfg["chunking"] = retrieval_chunking_cfg
    retrieval_index_cfg = retrieval_cfg.setdefault("index", {})
    if not isinstance(retrieval_index_cfg, dict):
        retrieval_index_cfg = {}
        retrieval_cfg["index"] = retrieval_index_cfg
    retrieval_embeddings_cfg = retrieval_cfg.setdefault("embeddings", {})
    if not isinstance(retrieval_embeddings_cfg, dict):
        retrieval_embeddings_cfg = {}
        retrieval_cfg["embeddings"] = retrieval_embeddings_cfg
    retrieval_cfg.setdefault("reranker", {})
    coverage_cfg = cfg.setdefault("coverage", {})
    judge_cfg = cfg.setdefault("judge", {})
    saving_cfg = cfg.setdefault("saving", {})
    hf_push_cfg = saving_cfg.setdefault("hf_push", {})
    llm_cfg = cfg.setdefault("llm", {})
    llm_routing_cfg = llm_cfg.setdefault("routing", {})

    env_map = {
        "N_TURNS": (turns_cfg, "exact", int),
        "BATCH_SIZE": (run_cfg, "batch_size", int),
        "TOTAL_SAMPLES": (run_cfg, "total_samples", int),
        "MIN_TURNS": (turns_cfg, "min", int),
        "MAX_TURNS": (turns_cfg, "max", int),
        "TURN_COUNT_DISTRIBUTION": (turns_cfg, "distribution", str),
        "TURN_COUNT_MEAN": (turns_cfg, "mean", float),
        "QUESTION": (run_seeding_cfg, "question", str),
        "QUESTION_SEED": (run_seeding_cfg, "question", str),
        "RUN_ID": (run_cfg, "run_id", str),
        "RESUME_RUN_ID": (run_cfg, "resume_run_id", str),
        "SEED_TOPICS_PATH": (run_topics_cfg, "path", str),
        "SEED_TOPICS_VARIANT": (run_topics_cfg, "variant", str),
        "SEED_TOPICS_PROBABILITY": (run_topics_cfg, "probability", float),
        "SEED_TOPICS_ENABLED": (run_topics_cfg, "enabled", _as_bool),
        "DISTRIBUTED_ENABLED": (distributed_cfg, "enabled", _as_bool),
        "DISTRIBUTED_BACKEND": (distributed_cfg, "backend", str),
        # Backward compatibility for legacy env name.
        "DISTRIBUTED_EXECUTOR": (distributed_cfg, "backend", str),
        "DISTRIBUTED_SPAWN_COORDINATOR": (distributed_spawn_cfg, "coordinator", _as_bool),
        "DISTRIBUTED_SPAWN_WORKERS": (distributed_spawn_cfg, "workers", _as_bool),
        "RAY_ADDRESS": (distributed_ray_cfg, "address", str),
        "RAY_AUTO_START_LOCAL": (distributed_ray_cfg, "auto_start_local", _as_bool),
        "RAY_NAMESPACE": (distributed_ray_cfg, "namespace", str),
        "RAY_ACTOR_NUM_CPUS": (ray_actor_cfg, "num_cpus", float),
        "RAY_ACTOR_NUM_GPUS": (ray_actor_cfg, "num_gpus", float),
        "RAY_COORDINATOR_NUM_CPUS": (ray_actor_cfg, "coordinator_num_cpus", float),
        "RAY_REPLICAS_QA": (ray_actor_cfg, "replicas_qa", int),
        "RAY_REPLICAS_COMPLETE": (ray_actor_cfg, "replicas_complete", int),
        "STORE_BACKEND": (store_cfg, "backend", str),
        "POSTGRES_DSN": (postgres_cfg, "dsn", str),
        "KB_CHUNK_SIZE": (retrieval_chunking_cfg, "chunk_size", int),
        "KB_CHUNK_OVERLAP": (retrieval_chunking_cfg, "chunk_overlap", int),
        "KB_DEFAULT_K": (retrieval_cfg, "top_k", int),
        "KB_PERSIST_DIR": (retrieval_index_cfg, "persist_dir", str),
        "KB_REBUILD_INDEX": (retrieval_index_cfg, "rebuild", _as_bool),
        "KB_SKIP_IF_UNCHANGED": (retrieval_index_cfg, "skip_if_unchanged", _as_bool),
        "KB_EMBEDDING_BACKEND": (retrieval_embeddings_cfg, "backend", str),
        "KB_EMBEDDING_DEVICE": (retrieval_embeddings_cfg, "device", str),
        "KB_FALLBACK_ON_CPU": (retrieval_embeddings_cfg, "fallback_on_cpu", _as_bool),
        "DOC_COVERAGE_MODE": (coverage_cfg, "doc_coverage_mode", str),
        "DOC_COVERAGE_EPSILON": (coverage_cfg, "doc_coverage_epsilon", float),
        "DOC_COVERAGE_FRACTION": (coverage_cfg, "doc_coverage_fraction", float),
        "QUESTION_DEDUP_RETRIES": (coverage_cfg, "question_dedup_retries", int),
        "ENABLE_WEB_TOOLS": (web_search_cfg, "enabled", _as_bool),
        "SERPER_NUM_RESULTS": (web_search_cfg, "serper_num_results", int),
        "SERPER_TIMEOUT": (web_search_cfg, "serper_timeout", int),
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
        "LLM_MODE": (llm_cfg, "mode", str),
        # Backward compatibility for legacy env name.
        "LLM_BACKEND": (llm_cfg, "mode", str),
        "LLM_ROUTING_STRATEGY": (llm_routing_cfg, "strategy", str),
    }

    for env_name, (target, key, caster) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        try:
            target[key] = caster(raw)
            if env_name == "N_TURNS":
                turns_cfg["mode"] = "exact"
            if env_name in {"MIN_TURNS", "MAX_TURNS"} and str(turns_cfg.get("mode") or "").strip().lower() != "exact":
                turns_cfg["mode"] = "range"
        except Exception:
            continue

    fallback_model = os.getenv("FALLBACK_EMBEDDING_MODEL")
    if fallback_model is not None and fallback_model.strip() != "":
        retrieval_embeddings_cfg["fallback_model"] = fallback_model.strip()

    model_kwargs_json = _parse_json_env("KB_EMBEDDING_MODEL_KWARGS_JSON")
    if model_kwargs_json is not None:
        retrieval_embeddings_cfg["model_kwargs"] = model_kwargs_json

    tokenizer_kwargs_json = _parse_json_env("KB_EMBEDDING_TOKENIZER_KWARGS_JSON")
    if tokenizer_kwargs_json is not None:
        retrieval_embeddings_cfg["tokenizer_kwargs"] = tokenizer_kwargs_json

    encode_kwargs_json = _parse_json_env("KB_EMBEDDING_ENCODE_KWARGS_JSON")
    if encode_kwargs_json is not None:
        retrieval_embeddings_cfg["encode_kwargs"] = encode_kwargs_json

    routing_endpoints_json = _parse_json_list_env("LLM_ROUTING_ENDPOINTS_JSON")
    if routing_endpoints_json is not None:
        llm_routing_cfg["endpoints"] = routing_endpoints_json

    target_languages = _parse_list_env("TARGET_LANGUAGES")
    if target_languages is not None:
        run_cfg["target_languages"] = target_languages
    else:
        # Backward compatibility for legacy single-language env override.
        legacy_target_language = (os.getenv("TARGET_LANGUAGE") or "").strip()
        if legacy_target_language:
            run_cfg["target_languages"] = [legacy_target_language]

    llm_cfg["mode"] = _normalize_llm_mode(llm_cfg.get("mode") or llm_cfg.get("backend") or "api")
    tools_cfg["web_search_enabled"] = _as_bool(web_search_cfg.get("enabled", False))
    try:
        tools_cfg["serper_num_results"] = int(web_search_cfg.get("serper_num_results", 5) or 5)
    except (TypeError, ValueError):
        tools_cfg["serper_num_results"] = 5
    try:
        tools_cfg["serper_timeout"] = int(web_search_cfg.get("serper_timeout", 30) or 30)
    except (TypeError, ValueError):
        tools_cfg["serper_timeout"] = 30

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

def _parse_json_list_env(env_name: str) -> List[Any] | None:
    raw = os.getenv(env_name)
    if raw is None or raw.strip() == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(f"{env_name} must be valid JSON list. Error: {err.msg}") from err
    if not isinstance(parsed, list):
        raise ValueError(f"{env_name} must be a JSON list.")
    return parsed

def _as_bool(raw: str) -> bool:
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}

def resolve_output_dir(cfg: Dict[str, Any], project_root: Path) -> Path:
    """Resolve output dir from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        project_root (Path): Resolved project directory context.
    
    Returns:
        Path: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_output_dir
        >>> resolve_output_dir(...)
    
    """
    output_dir = (cfg.get("saving", {}) or {}).get("output_dir", "outputs")
    raw = Path(str(output_dir))
    return raw if raw.is_absolute() else (project_root / raw)

def resolve_output_columns(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Resolve output columns from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        Dict[str, str]: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_output_columns
        >>> resolve_output_columns(...)
    
    """
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
    """Resolve question from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_question
        >>> resolve_question(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    data_cfg = run_cfg.get("data", {}) if isinstance(run_cfg.get("data"), dict) else {}
    seeding_cfg = data_cfg.get("seeding", {}) if isinstance(data_cfg.get("seeding"), dict) else {}
    canonical = str(seeding_cfg.get("question", "") or "").strip()
    if canonical:
        return canonical
    legacy_seed_question = str(run_cfg.get("seed_question", "") or "").strip()
    if legacy_seed_question:
        return legacy_seed_question
    return str(run_cfg.get("question_seed", "") or "")

def resolve_target_languages(cfg: Dict[str, Any]) -> List[str]:
    """Resolve target languages from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        List[str]: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_target_languages
        >>> resolve_target_languages(...)
    
    """
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
    """Resolve question seed from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_question_seed
        >>> resolve_question_seed(...)
    
    """
    return resolve_question(cfg)

def resolve_run_id(cfg: Dict[str, Any]) -> str:
    """Resolve run id from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_run_id
        >>> resolve_run_id(...)
    
    """
    return str((cfg.get("run", {}) or {}).get("run_id", "") or "")

def resolve_resume_run_id(cfg: Dict[str, Any]) -> str:
    """Resolve resume run id.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_resume_run_id
        >>> resolve_resume_run_id(...)
    
    """
    return str((cfg.get("run", {}) or {}).get("resume_run_id", "") or "")

def resolve_n_turns(cfg: Dict[str, Any], fallback: int = 1) -> int:
    """Resolve n turns from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (int): int value used by this operation.
    
    Returns:
        int: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_n_turns
        >>> resolve_n_turns(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    turns_cfg = run_cfg.get("turns", {}) if isinstance(run_cfg.get("turns"), dict) else {}
    mode = str(turns_cfg.get("mode") or "").strip().lower()
    if mode == "exact":
        raw = turns_cfg.get("exact", fallback)
    elif mode == "range":
        raw = turns_cfg.get("max", fallback)
    else:
        raw = run_cfg.get("n_turns", fallback)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return fallback if fallback > 0 else 1
    return n if n > 0 else (fallback if fallback > 0 else 1)

def resolve_batch_size(cfg: Dict[str, Any], fallback: int = 1) -> int:
    """Resolve batch size from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (int): int value used by this operation.
    
    Returns:
        int: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_batch_size
        >>> resolve_batch_size(...)
    
    """
    raw = (cfg.get("run", {}) or {}).get("batch_size", fallback)
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return fallback
    return size if size > 0 else fallback

def resolve_total_samples(cfg: Dict[str, Any], fallback: int = 0) -> int:
    """Resolve total samples from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (int): int value used by this operation.
    
    Returns:
        int: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_total_samples
        >>> resolve_total_samples(...)
    
    """
    raw = (cfg.get("run", {}) or {}).get("total_samples", fallback)
    try:
        total = int(raw)
    except (TypeError, ValueError):
        return fallback
    return total if total >= 0 else fallback

def resolve_distributed_enabled(cfg: Dict[str, Any]) -> bool:
    """Resolve distributed enabled from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        bool: Boolean indicator describing the evaluated condition.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_distributed_enabled
        >>> resolve_distributed_enabled(...)
    
    """
    distributed_cfg = ((cfg.get("run", {}) or {}).get("distributed", {}) or {})
    return bool(distributed_cfg.get("enabled", False))

def resolve_turn_range(cfg: Dict[str, Any], fallback: int = 1) -> Tuple[int, int]:
    """Resolve turn range from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (int): int value used by this operation.
    
    Returns:
        Tuple[int, int]: Resolved value after applying defaults and normalization rules.
    
    Raises:
        ValueError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_turn_range
        >>> resolve_turn_range(...)
    
    """
    default_turns = resolve_n_turns(cfg, fallback=fallback)
    run_cfg = cfg.get("run", {}) or {}
    turns_cfg = run_cfg.get("turns", {}) if isinstance(run_cfg.get("turns"), dict) else {}
    mode = str(turns_cfg.get("mode") or "").strip().lower()

    if mode == "exact":
        try:
            exact = int(turns_cfg.get("exact", default_turns))
        except (TypeError, ValueError):
            exact = default_turns
        exact = exact if exact > 0 else default_turns
        return exact, exact

    min_raw = turns_cfg.get("min", run_cfg.get("min_turns", 0))
    max_raw = turns_cfg.get("max", run_cfg.get("max_turns", 0))
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
            f"Invalid turn range: run.turns.min ({min_turns}) must be <= run.turns.max ({max_turns})."
        )
    return min_turns, max_turns

def resolve_turn_count_distribution(cfg: Dict[str, Any], fallback: str = "poisson") -> str:
    """Resolve turn count distribution.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (str): str value used by this operation.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_turn_count_distribution
        >>> resolve_turn_count_distribution(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    turns_cfg = run_cfg.get("turns", {}) if isinstance(run_cfg.get("turns"), dict) else {}
    raw = str(turns_cfg.get("distribution", run_cfg.get("turn_count_distribution", fallback)) or fallback).strip().lower()
    aliases = {"exp": "exponential", "pois": "poisson"}
    normalized = aliases.get(raw, raw)
    if normalized in {"uniform", "poisson", "exponential"}:
        return normalized
    return fallback

def resolve_turn_count_mean(cfg: Dict[str, Any], fallback: float = 0.0) -> float:
    """Resolve turn count mean.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (float): float value used by this operation.
    
    Returns:
        float: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_turn_count_mean
        >>> resolve_turn_count_mean(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    turns_cfg = run_cfg.get("turns", {}) if isinstance(run_cfg.get("turns"), dict) else {}
    raw = turns_cfg.get("mean", run_cfg.get("turn_count_mean", fallback))
    try:
        mean = float(raw)
    except (TypeError, ValueError):
        return fallback
    if mean <= 0:
        return fallback
    return mean

def resolve_retrieval_default_k(cfg: Dict[str, Any], fallback: int = 4) -> int:
    """Resolve retrieval default k.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (int): int value used by this operation.
    
    Returns:
        int: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_retrieval_default_k
        >>> resolve_retrieval_default_k(...)
    
    """
    tools_cfg = cfg.get("tools", {}) if isinstance(cfg.get("tools"), dict) else {}
    retrieval_cfg = tools_cfg.get("retrieval", {}) if isinstance(tools_cfg.get("retrieval"), dict) else {}
    if not retrieval_cfg:
        retrieval_cfg = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval"), dict) else {}
    raw = retrieval_cfg.get("top_k", retrieval_cfg.get("default_k", fallback))
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return fallback
    return k if k > 0 else fallback

def resolve_seed_topics_path(cfg: Dict[str, Any]) -> str:
    """Resolve seed topics path.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_seed_topics_path
        >>> resolve_seed_topics_path(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    data_cfg = run_cfg.get("data", {}) if isinstance(run_cfg.get("data"), dict) else {}
    seeding_cfg = data_cfg.get("seeding", {}) if isinstance(data_cfg.get("seeding"), dict) else {}
    topics_cfg = seeding_cfg.get("topics", {}) if isinstance(seeding_cfg.get("topics"), dict) else {}
    value = str(topics_cfg.get("path", "") or "").strip()
    if value:
        return value
    return str(run_cfg.get("seed_topics_path", "") or "")

def resolve_seed_topics_variant(cfg: Dict[str, Any]) -> str:
    """Resolve seed topics variant.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_seed_topics_variant
        >>> resolve_seed_topics_variant(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    data_cfg = run_cfg.get("data", {}) if isinstance(run_cfg.get("data"), dict) else {}
    seeding_cfg = data_cfg.get("seeding", {}) if isinstance(data_cfg.get("seeding"), dict) else {}
    topics_cfg = seeding_cfg.get("topics", {}) if isinstance(seeding_cfg.get("topics"), dict) else {}
    value = str(topics_cfg.get("variant", "") or "").strip()
    if value:
        return value
    return str(run_cfg.get("seed_topics_variant", "") or "")

def resolve_seed_topics_probability(cfg: Dict[str, Any]) -> float:
    """Resolve seed topics probability.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        float: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_seed_topics_probability
        >>> resolve_seed_topics_probability(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    data_cfg = run_cfg.get("data", {}) if isinstance(run_cfg.get("data"), dict) else {}
    seeding_cfg = data_cfg.get("seeding", {}) if isinstance(data_cfg.get("seeding"), dict) else {}
    topics_cfg = seeding_cfg.get("topics", {}) if isinstance(seeding_cfg.get("topics"), dict) else {}
    raw = topics_cfg.get("probability", run_cfg.get("seed_topics_probability", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0

def resolve_seed_topics_enabled(cfg: Dict[str, Any]) -> bool:
    """Resolve seed topics enabled.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        bool: Boolean indicator describing the evaluated condition.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_seed_topics_enabled
        >>> resolve_seed_topics_enabled(...)
    
    """
    run_cfg = cfg.get("run", {}) or {}
    data_cfg = run_cfg.get("data", {}) if isinstance(run_cfg.get("data"), dict) else {}
    seeding_cfg = data_cfg.get("seeding", {}) if isinstance(data_cfg.get("seeding"), dict) else {}
    topics_cfg = seeding_cfg.get("topics", {}) if isinstance(seeding_cfg.get("topics"), dict) else {}
    if "enabled" in topics_cfg:
        return bool(topics_cfg.get("enabled"))
    return bool(run_cfg.get("seed_topics_enabled", True))

def resolve_judge_mode(cfg: Dict[str, Any]) -> str:
    """Resolve judge mode from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_judge_mode
        >>> resolve_judge_mode(...)
    
    """
    return str((cfg.get("judge", {}) or {}).get("mode", "offline") or "offline").strip().lower()

def resolve_judge_enabled(cfg: Dict[str, Any]) -> bool:
    """Resolve judge enabled from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        bool: Boolean indicator describing the evaluated condition.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_judge_enabled
        >>> resolve_judge_enabled(...)
    
    """
    enabled = bool((cfg.get("judge", {}) or {}).get("enabled", True))
    if resolve_judge_mode(cfg) == "offline":
        return False
    return enabled

def resolve_judge_granularity(cfg: Dict[str, Any], fallback: str = "turn") -> str:
    """Resolve judge granularity from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        fallback (str): str value used by this operation.
    
    Returns:
        str: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_judge_granularity
        >>> resolve_judge_granularity(...)
    
    """
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
    """Resolve judge reasons from configuration.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
    
    Returns:
        List[str]: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import resolve_judge_reasons
        >>> resolve_judge_reasons(...)
    
    """
    reasons = (cfg.get("judge", {}) or {}).get("reasons", [])
    if isinstance(reasons, list):
        return [str(item) for item in reasons if str(item).strip()]
    return []

def build_base_inputs(cfg: Dict[str, Any], project_root: Path, config_path: Path) -> Dict[str, Any]:
    """Build base inputs.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        project_root (Path): Resolved project directory context.
        config_path (Path): Path to a configuration file.
    
    Returns:
        Dict[str, Any]: Constructed value derived from the provided inputs.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May read environment variables or mutate process-level runtime state.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.config.loader import build_base_inputs
        >>> build_base_inputs(...)
    
    """
    user_persona, assistant_persona, persona_meta = select_personas(cfg, project_root, config_path)
    user_agent_used_name = resolve_agent_used_name(cfg, "qa_generator")
    assistant_agent_used_name = resolve_agent_used_name(cfg, "kb_responder")
    target_languages = resolve_target_languages(cfg)
    primary_language = target_languages[0]
    min_turns, max_turns = resolve_turn_range(cfg, fallback=1)
    return {
        "question": resolve_question(cfg),
        "target_language": primary_language,
        "target_languages": target_languages,
        "question_seed": resolve_question_seed(cfg),
        "run_id": resolve_run_id(cfg),
        "resume_run_id": resolve_resume_run_id(cfg),
        "total_samples": resolve_total_samples(cfg),
        "min_turns": min_turns,
        "max_turns": max_turns,
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
