"""Main generation and judge orchestration flows.

"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dlgforge.config import (
    build_base_inputs,
    load_config,
    resolve_batch_size,
    resolve_distributed_enabled,
    resolve_judge_enabled,
    resolve_judge_granularity,
    resolve_n_turns,
    resolve_output_columns,
    resolve_output_dir,
    resolve_retrieval_default_k,
    resolve_target_languages,
    resolve_turn_count_distribution,
    resolve_turn_count_mean,
    resolve_total_samples,
    resolve_turn_range,
)
from dlgforge.config.personas import UniformPersonaSampler, build_uniform_persona_sampler
from dlgforge.io import (
    OutputPaths,
    append_sharegpt_judged_record,
    configure_output_columns,
    ensure_output_layout,
    load_coverage_ledger,
    save_training_sample,
)
from dlgforge.llm import OpenAIModelClient, missing_models, resolve_llm_settings
from dlgforge.pipeline.history import build_conversation_history, build_public_history, format_history
from dlgforge.pipeline.dedup import RunQuestionRegistry
from dlgforge.pipeline.hf_push import maybe_auto_push_after_run
from dlgforge.pipeline.prompts import build_agent_system_prompt, build_task_prompt
from dlgforge.pipeline.sampling import (
    build_doc_chunk_counts,
    build_doc_question_hashes,
    build_doc_recent_questions,
    build_doc_usage,
    build_question_inputs,
    build_seed_topic_usage,
    build_used_seed_hashes,
    is_duplicate_question,
    update_coverage_ledger,
    update_doc_question_memory,
    update_seed_memory,
)
from dlgforge.pipeline.state import checkpoint_run_state, init_run_state
from dlgforge.pipeline.state import (
    build_initial_batched_conversations,
    checkpoint_batched_run_state,
    init_batched_run_state,
    load_batched_conversations_from_state,
)
from dlgforge.tools import SerperWebSearchClient, configure_retrieval, vector_db_search
from dlgforge.utils import env_int, load_dotenv_files, parse_json_object, setup_logging

LOGGER = logging.getLogger("dlgforge.pipeline")
RETRIEVAL_LOGGER = logging.getLogger("dlgforge.retrieval")
JUDGE_LOGGER = logging.getLogger("dlgforge.judge")
TOOLS_LOGGER = logging.getLogger("dlgforge.tools")

def run(config_path: str) -> None:
    """Run synthetic conversation generation from a config file.
    
    Args:
        config_path (str): Path to a configuration file.
    
    Returns:
        None: No value is returned.
    
    Raises:
        FileNotFoundError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.runner import run
        >>> run(...)
    
    """
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists() or not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    project_root = config_file.parent
    setup_logging(project_root / "logs")
    load_dotenv_files(project_root)

    cfg, resolved_config_path, project_root = load_config(config_file)
    if resolve_distributed_enabled(cfg) and os.getenv("DLGFORGE_RUN_BOOTSTRAPPED") != "1":
        LOGGER.info("[distributed] Bootstrapping one-command distributed runtime")
        from dlgforge.distributed import run_bootstrap

        run_bootstrap(config_file, cfg)
        return

    configure_output_columns(resolve_output_columns(cfg))
    _preflight_checks(cfg, resolved_config_path, project_root)
    _apply_runtime_env(cfg)
    retrieval_default_k = resolve_retrieval_default_k(cfg)
    use_reranker = _as_bool((cfg.get("models", {}) or {}).get("use_reranker", False), default=False)
    RETRIEVAL_LOGGER.info(f"[retrieval] effective default_k={retrieval_default_k}, use_reranker={use_reranker}")

    output_dir = resolve_output_dir(cfg, project_root)
    output_paths = OutputPaths(project_root=project_root, output_dir=output_dir)
    ensure_output_layout(output_paths)

    configure_retrieval(cfg, project_root)
    base_inputs = build_base_inputs(cfg, project_root=project_root, config_path=resolved_config_path)
    persona_sampler = build_uniform_persona_sampler(
        cfg=cfg,
        project_root=project_root,
        config_path=resolved_config_path,
    )
    base_inputs["project_root"] = str(project_root)
    base_inputs["config_dir"] = str(resolved_config_path.parent)

    n_turns = resolve_n_turns(cfg)
    min_turns, max_turns = resolve_turn_range(cfg, fallback=n_turns)
    turn_count_distribution = resolve_turn_count_distribution(cfg)
    turn_count_mean = resolve_turn_count_mean(cfg)
    batch_size = resolve_batch_size(cfg)
    total_samples_cfg = resolve_total_samples(cfg)
    language_list = resolve_target_languages(cfg)
    total_samples = _resolve_total_samples_target(total_samples_cfg, batch_size)
    base_run_id = str(base_inputs.get("run_id") or "").strip()
    multi_language_seed = base_run_id or uuid.uuid4().hex
    total_generated = 0
    for language_index, language in enumerate(language_list):
        language_inputs = dict(base_inputs)
        language_inputs["target_language"] = language
        language_inputs["target_languages"] = language_list
        language_inputs["resume_run_id"] = str(base_inputs.get("resume_run_id") or "") if language_index == 0 else ""
        if len(language_list) > 1:
            language_suffix = _sanitize_language_tag(language)
            language_inputs["run_id"] = f"{multi_language_seed}-{language_suffix}"

        LOGGER.info(
            "[language] Starting generation for language=%s target_samples=%s (%s/%s)",
            language,
            total_samples,
            language_index + 1,
            len(language_list),
        )
        generated = _run_until_total_samples(
            cfg=cfg,
            output_paths=output_paths,
            base_inputs=language_inputs,
            min_turns=min_turns,
            max_turns=max_turns,
            turn_count_distribution=turn_count_distribution,
            turn_count_mean=turn_count_mean,
            batch_size=batch_size,
            total_samples=total_samples,
            persona_sampler=persona_sampler,
        )
        total_generated += generated
        LOGGER.info(
            "[language] Completed language=%s generated=%s/%s",
            language,
            generated,
            total_samples,
        )

    LOGGER.info(
        "[samples] Completed generation languages=%s target_per_language=%s total_generated=%s",
        len(language_list),
        total_samples,
        total_generated,
    )
    maybe_auto_push_after_run(cfg, output_paths)

def run_judge_only(config_path: str) -> None:
    """Run judge evaluation on existing conversation artifacts.
    
    Args:
        config_path (str): Path to a configuration file.
    
    Returns:
        None: No value is returned.
    
    Raises:
        FileNotFoundError: Raised when validation or runtime requirements are not met.
        RuntimeError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.runner import run_judge_only
        >>> run_judge_only(...)
    
    """
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists() or not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    project_root = config_file.parent
    setup_logging(project_root / "logs")
    load_dotenv_files(project_root)

    cfg, resolved_config_path, project_root = load_config(config_file)
    configure_output_columns(resolve_output_columns(cfg))
    _preflight_judge_only(cfg, resolved_config_path)

    output_dir = resolve_output_dir(cfg, project_root)
    output_paths = OutputPaths(project_root=project_root, output_dir=output_dir)
    ensure_output_layout(output_paths)

    conversation_files = sorted(output_paths.conversation_dir.glob("*.json"))
    if not conversation_files:
        raise RuntimeError(f"No conversation files found under: {output_paths.conversation_dir}")

    judged_output = output_paths.conversation_sft_judged_file
    judged_tmp_output = judged_output.with_name(judged_output.name + ".tmp")
    if judged_tmp_output.exists():
        judged_tmp_output.unlink()
    output_paths.conversation_sft_judged_file = judged_tmp_output

    model_client = OpenAIModelClient()
    judged_count = 0
    judged_turns_total = 0
    judge_granularity = resolve_judge_granularity(cfg)

    for file_path in conversation_files:
        payload = _read_json_dict(file_path)
        if not payload:
            JUDGE_LOGGER.warning(f"[judge] Skipping unreadable file: {file_path.name}")
            continue

        turns_raw = payload.get("turns")
        if not isinstance(turns_raw, list) or not turns_raw:
            JUDGE_LOGGER.warning(f"[judge] Skipping file with no turns: {file_path.name}")
            continue

        inputs = payload.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}

        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = []
        conversation_judge: Dict[str, Any] = {}
        if judge_granularity == "conversation":
            judged_turns = [turn if isinstance(turn, dict) else {} for turn in turns_raw]
            conversation_judge, _ = _generate_conversation_judge_turn(
                cfg=cfg,
                model_client=model_client,
                inputs=inputs,
                turns=judged_turns,
            )
        else:
            judged_turns = _judge_existing_turns(cfg, model_client, inputs, turns_raw)

        payload["turns"] = judged_turns
        payload["conversation_judge"] = conversation_judge
        if conversation_judge:
            payload_inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
            payload_inputs["conversation_judge"] = conversation_judge
            payload["inputs"] = payload_inputs
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        conversation_id = str(payload.get("conversation_id") or file_path.stem)
        timestamp = str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat())
        export_inputs = dict(inputs)
        export_inputs["judge_granularity"] = judge_granularity
        if conversation_judge:
            export_inputs["conversation_judge"] = conversation_judge
        append_sharegpt_judged_record(
            paths=output_paths,
            conversation_id=conversation_id,
            timestamp=timestamp,
            inputs=export_inputs,
            turns=judged_turns,
            messages=messages,
            conversation_judge=conversation_judge,
        )

        judged_count += 1
        judged_turns_total += len(judged_turns)
        JUDGE_LOGGER.info(
            "[judge] Judged %s (%s turns, granularity=%s)",
            file_path.name,
            len(judged_turns),
            judge_granularity,
        )

    if judged_count == 0:
        raise RuntimeError("No valid conversation files were judged.")

    judged_tmp_output.replace(judged_output)
    output_paths.conversation_sft_judged_file = judged_output

    JUDGE_LOGGER.info(
        "[judge] Completed. "
        f"conversations={judged_count}, turns={judged_turns_total}, "
        f"output={judged_output}"
    )

def _read_json_dict(path: Path) -> Dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}

def _judge_existing_turns(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    conversation_inputs: Dict[str, Any],
    turns_raw: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    judged_turns: List[Dict[str, Any]] = []
    judge_reasons = _resolve_judge_reasons(cfg)

    for raw_turn in turns_raw:
        turn = raw_turn if isinstance(raw_turn, dict) else {}
        qa_output = turn.get("qa_generation") or {}
        kb_output = turn.get("kb_answer") or {}

        public_history = build_public_history(judged_turns)
        formatted_history = format_history(public_history)
        turn_inputs = {
            **conversation_inputs,
            "public_conversation_history": formatted_history,
            "formatted_conversation_history": formatted_history,
            "judge_enabled": True,
            "judge_reasons": judge_reasons,
        }

        judge_output, _ = _generate_judge_turn(
            cfg=cfg,
            model_client=model_client,
            inputs=turn_inputs,
            qa_output=qa_output if isinstance(qa_output, dict) else {},
            kb_output=kb_output if isinstance(kb_output, dict) else {},
        )

        updated_turn = dict(turn)
        updated_turn["qa_judge"] = judge_output
        judged_turns.append(updated_turn)

    return judged_turns

def _resolve_judge_reasons(cfg: Dict[str, Any]) -> List[str]:
    reasons = (cfg.get("judge", {}) or {}).get("reasons", [])
    if not isinstance(reasons, list):
        return []
    return [str(item) for item in reasons if str(item).strip()]

def _resolve_judge_granularity(cfg: Dict[str, Any]) -> str:
    return resolve_judge_granularity(cfg)

def _judge_per_turn_enabled(cfg: Dict[str, Any]) -> bool:
    return resolve_judge_enabled(cfg) and _resolve_judge_granularity(cfg) == "turn"

def _preflight_judge_only(cfg: Dict[str, Any], config_path: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    if not resolve_judge_enabled(cfg):
        raise RuntimeError("Judge-only mode requires `judge.mode: online` and `judge.enabled: true`.")

    settings = resolve_llm_settings(cfg, "qa_judge")
    model = (settings.get("model") or "").strip()
    if not model:
        raise RuntimeError(
            "Missing qa_judge model. Set `llm.agents.qa_judge.model`, `llm.model`, `LLM_MODEL`, or `OPENAI_MODEL`."
        )

def _preflight_checks(cfg: Dict[str, Any], config_path: Path, project_root: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    knowledge_dir = project_root / "knowledge"
    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        raise RuntimeError(f"Missing knowledge directory: {knowledge_dir}")

    supported = [p for p in knowledge_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}]
    if not supported:
        raise RuntimeError(
            "No supported knowledge files found. Add at least one .txt, .md, or .pdf under knowledge/."
        )

    missing = missing_models(cfg)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing LLM model for required agent(s): "
            f"{joined}. Set per-agent model in config, or set llm.model, or LLM_MODEL, or OPENAI_MODEL in .env."
        )

    tools_cfg = cfg.get("tools", {}) or {}
    web_enabled = bool(tools_cfg.get("web_search_enabled", True))
    if web_enabled and not (os.getenv("SERPER_API_KEY") or "").strip():
        TOOLS_LOGGER.warning(
            "[preflight] Web search is enabled but SERPER_API_KEY is missing; "
            "run continues unless web_search is called."
        )

def _apply_runtime_env(cfg: Dict[str, Any]) -> None:
    coverage_cfg = cfg.get("coverage", {}) or {}
    os.environ.setdefault("DOC_COVERAGE_MODE", str(coverage_cfg.get("doc_coverage_mode", "balanced")))
    os.environ.setdefault("DOC_COVERAGE_EPSILON", str(coverage_cfg.get("doc_coverage_epsilon", 0.15)))
    os.environ.setdefault("DOC_COVERAGE_FRACTION", str(coverage_cfg.get("doc_coverage_fraction", 0.2)))
    os.environ.setdefault("QUESTION_DEDUP_RETRIES", str(coverage_cfg.get("question_dedup_retries", 3)))

def _sanitize_language_tag(language: str) -> str:
    text = str(language or "").strip().lower()
    if not text:
        return "lang"
    chars = [ch if ch.isalnum() else "-" for ch in text]
    token = "".join(chars).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "lang"

def _resolve_total_samples_target(total_samples_cfg: int, batch_size: int) -> int:
    if total_samples_cfg > 0:
        return total_samples_cfg
    return batch_size if batch_size > 1 else 1

def _sample_turn_count(
    min_turns: int,
    max_turns: int,
    distribution: str = "poisson",
    mean: float = 0.0,
) -> int:
    low = min(min_turns, max_turns)
    high = max(min_turns, max_turns)
    if low <= 0:
        low = 1
    if high <= 0:
        high = low
    if low == high:
        return low

    dist = str(distribution or "poisson").strip().lower()
    if mean <= 0:
        mean = float(low + high) / 2.0

    if dist == "uniform":
        sampled = random.randint(low, high)
    elif dist == "exponential":
        sampled = _sample_exponential(mean)
    else:
        sampled = _sample_poisson(mean)

    if sampled < low:
        return low
    if sampled > high:
        return high
    return sampled

def _sample_turn_targets(
    batch_size: int,
    min_turns: int,
    max_turns: int,
    distribution: str,
    mean: float,
) -> List[int]:
    return [
        _sample_turn_count(min_turns, max_turns, distribution=distribution, mean=mean)
        for _ in range(max(batch_size, 0))
    ]

def _sample_poisson(mean: float) -> int:
    lam = max(float(mean), 1e-6)
    if lam < 30.0:
        limit = math.exp(-lam)
        k = 0
        product = 1.0
        while product > limit:
            k += 1
            product *= random.random()
        return max(0, k - 1)

    # Normal approximation for large lambda to avoid long multiplicative loops.
    return max(0, int(round(random.gauss(lam, math.sqrt(lam)))))

def _sample_exponential(mean: float) -> int:
    scale = max(float(mean), 1e-6)
    value = random.expovariate(1.0 / scale)
    return max(0, int(round(value)))

def _slot_target_n_turns(slot: Dict[str, Any], fallback: int = 1) -> int:
    raw = slot.get("target_n_turns", fallback)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback if fallback > 0 else 1
    if value <= 0:
        return fallback if fallback > 0 else 1
    return value

def _with_sampled_persona(inputs: Dict[str, Any], persona_sampler: UniformPersonaSampler) -> Dict[str, Any]:
    sampled_inputs = dict(inputs)
    user_persona, assistant_persona, persona_meta = persona_sampler.sample()
    sampled_inputs["user_persona"] = user_persona
    sampled_inputs["assistant_persona"] = assistant_persona
    sampled_inputs["user_persona_id"] = persona_meta.get("user_id", "")
    sampled_inputs["assistant_persona_id"] = persona_meta.get("assistant_id", "")
    return sampled_inputs

def _run_until_total_samples(
    cfg: Dict[str, Any],
    output_paths: OutputPaths,
    base_inputs: Dict[str, Any],
    min_turns: int,
    max_turns: int,
    turn_count_distribution: str,
    turn_count_mean: float,
    batch_size: int,
    total_samples: int,
    persona_sampler: UniformPersonaSampler,
) -> int:
    generated = 0
    wave_index = 0
    configured_run_id = str(base_inputs.get("run_id") or "").strip()
    configured_resume_id = str(base_inputs.get("resume_run_id") or "").strip()
    run_id_seed = configured_run_id or configured_resume_id or uuid.uuid4().hex

    while generated < total_samples:
        remaining = total_samples - generated
        current_batch_size = min(batch_size, remaining)

        wave_inputs = dict(base_inputs)
        wave_inputs["resume_run_id"] = configured_resume_id if wave_index == 0 else ""
        if wave_index == 0:
            wave_inputs["run_id"] = configured_run_id or run_id_seed
        else:
            wave_inputs["run_id"] = f"{run_id_seed}-{wave_index:04d}"

        persisted_this_wave = 0
        if batch_size == 1:
            conversation_inputs = _with_sampled_persona(wave_inputs, persona_sampler)
            turns, raw_results, last_result = run_multi_turn(
                cfg,
                output_paths,
                conversation_inputs,
                min_turns,
                max_turns,
                turn_count_distribution,
                turn_count_mean,
            )
            if turns:
                persist_training_sample(output_paths, conversation_inputs, last_result, turns, raw_results)
                persisted_this_wave = 1
        else:
            conversation_inputs_by_index = [
                _with_sampled_persona(wave_inputs, persona_sampler) for _ in range(max(current_batch_size, 0))
            ]
            batch_inputs, conversations = asyncio.run(
                run_multi_turn_batched_async(
                    cfg=cfg,
                    output_paths=output_paths,
                    base_inputs=wave_inputs,
                    max_turns=max_turns,
                    batch_size=current_batch_size,
                    min_turns=min_turns,
                    turn_count_distribution=turn_count_distribution,
                    turn_count_mean=turn_count_mean,
                    conversation_inputs_by_index=conversation_inputs_by_index,
                )
            )
            persisted_this_wave = persist_batched_training_samples(output_paths, batch_inputs, conversations)

        generated += persisted_this_wave
        LOGGER.info(
            "[samples] wave=%s requested_batch=%s persisted=%s generated=%s/%s",
            wave_index + 1,
            current_batch_size,
            persisted_this_wave,
            generated,
            total_samples,
        )

        if persisted_this_wave <= 0:
            raise RuntimeError(
                "No samples were persisted in the last batch wave. "
                "Stopping early to avoid an infinite loop."
            )

        wave_index += 1

    return generated

def run_multi_turn(
    cfg: Dict[str, Any],
    output_paths: OutputPaths,
    base_inputs: Dict[str, Any],
    min_turns: int,
    max_turns: int,
    turn_count_distribution: str,
    turn_count_mean: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run multi turn.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        output_paths (OutputPaths): Filesystem path used by this operation.
        base_inputs (Dict[str, Any]): Mapping payload for this operation.
        min_turns (int): int value used by this operation.
        max_turns (int): int value used by this operation.
        turn_count_distribution (str): str value used by this operation.
        turn_count_mean (float): float value used by this operation.
    
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.runner import run_multi_turn
        >>> run_multi_turn(...)
    
    """
    shared_inputs = base_inputs
    default_turns = max(min_turns, 1)
    run_id, resume_state = init_run_state(output_paths, base_inputs, default_turns)
    target_n_turns = _sample_turn_count(
        min_turns=min_turns,
        max_turns=max_turns,
        distribution=turn_count_distribution,
        mean=turn_count_mean,
    )
    if resume_state:
        stored_inputs = resume_state.get("inputs") or {}
        if stored_inputs:
            base_inputs = {**stored_inputs, **base_inputs}
        target_n_turns = int(resume_state.get("n_turns") or target_n_turns)
        LOGGER.info(
            f"[run-state] Resuming run_id={run_id} at turn {len(resume_state.get('turns', [])) + 1} "
            f"(target_turns={target_n_turns})"
        )
    else:
        LOGGER.info(f"[run-state] Starting run_id={run_id} (target_turns={target_n_turns})")

    base_inputs = {
        **base_inputs,
        "run_id": run_id,
        "n_turns": target_n_turns,
        "min_turns": min_turns,
        "max_turns": max_turns,
        "turn_count_distribution": turn_count_distribution,
        "turn_count_mean": turn_count_mean,
    }

    turns: List[Dict[str, Any]] = resume_state.get("turns", []) if resume_state else []
    raw_results: List[Dict[str, Any]] = resume_state.get("raw_results", []) if resume_state else []
    last_result: Dict[str, Any] = {}

    ledger_entries = load_coverage_ledger(output_paths)
    used_topic_ids = {entry.get("topic_id") for entry in ledger_entries if entry.get("topic_id")}
    used_question_hashes = {entry.get("question_hash") for entry in ledger_entries if entry.get("question_hash")}
    used_seed_hashes = build_used_seed_hashes(ledger_entries)
    seed_topic_usage = build_seed_topic_usage(ledger_entries)
    recent_ledger_questions = [entry.get("question", "") for entry in ledger_entries if entry.get("question")][-8:]
    doc_usage = build_doc_usage(ledger_entries)
    doc_chunk_counts = build_doc_chunk_counts()
    doc_question_hashes = build_doc_question_hashes(ledger_entries)
    doc_recent_questions = build_doc_recent_questions(ledger_entries)

    checkpoint_run_state(
        paths=output_paths,
        run_id=run_id,
        status="started" if not resume_state else "resumed",
        base_inputs=base_inputs,
        n_turns=target_n_turns,
        turns=turns,
        raw_results=raw_results,
    )

    model_client = OpenAIModelClient()
    tools_cfg = cfg.get("tools", {}) or {}
    web_enabled = bool(tools_cfg.get("web_search_enabled", True))
    web_client = SerperWebSearchClient(
        num_results=int(tools_cfg.get("serper_num_results", 5) or 5),
        timeout=int(tools_cfg.get("serper_timeout", 30) or 30),
    )

    run_started_at = time.perf_counter()
    start_turn = len(turns) + 1
    for turn_index in range(start_turn, target_n_turns + 1):
        turn_started_at = time.perf_counter()
        LOGGER.info(f"[turn] {turn_index}/{target_n_turns} started")
        history = build_conversation_history(turns)
        public_history = build_public_history(turns)
        formatted_history = format_history(public_history)

        max_retries = env_int("QUESTION_DEDUP_RETRIES", default=3)
        avoid_sources: set[str] = set()
        forced_mode = ""

        for attempt in range(max_retries + 1):
            LOGGER.info(
                f"[turn] {turn_index}/{target_n_turns} attempt {attempt + 1}/{max_retries + 1} generating question"
            )
            question_inputs = build_question_inputs(
                base_inputs=base_inputs,
                turn_index=turn_index,
                n_turns=target_n_turns,
                public_history=public_history,
                used_topic_ids=used_topic_ids,
                recent_ledger_questions=recent_ledger_questions,
                doc_usage=doc_usage,
                doc_chunk_counts=doc_chunk_counts,
                doc_recent_questions=doc_recent_questions,
                avoid_sources=avoid_sources,
                forced_mode=forced_mode,
                used_seed_hashes=used_seed_hashes,
                seed_topic_usage=seed_topic_usage,
            )
            iteration_inputs = {
                **base_inputs,
                "turn_index": turn_index,
                "conversation_history": history,
                "formatted_conversation_history": formatted_history,
                "public_conversation_history": formatted_history,
                **question_inputs,
            }

            qa_started_at = time.perf_counter()
            qa_output, qa_raw = _generate_user_turn(cfg, model_client, iteration_inputs)
            LOGGER.info(
                f"[turn] {turn_index}/{target_n_turns} qa_generator done in {time.perf_counter() - qa_started_at:.2f}s"
            )

            kb_started_at = time.perf_counter()
            kb_output, tool_events, kb_raw = _generate_assistant_turn(
                cfg=cfg,
                model_client=model_client,
                inputs=iteration_inputs,
                qa_output=qa_output,
                web_enabled=web_enabled,
                web_client=web_client,
            )
            LOGGER.info(
                f"[turn] {turn_index}/{target_n_turns} kb_responder done in {time.perf_counter() - kb_started_at:.2f}s "
                f"(tool_calls={len(tool_events)})"
            )
            _sanitize_retrieval_queries(kb_output)

            judge_output: Dict[str, Any] = {}
            judge_raw: Dict[str, Any] = {}
            if _judge_per_turn_enabled(cfg):
                judge_started_at = time.perf_counter()
                judge_output, judge_raw = _generate_judge_turn(
                    cfg=cfg,
                    model_client=model_client,
                    inputs=iteration_inputs,
                    qa_output=qa_output,
                    kb_output=kb_output,
                )
                LOGGER.info(
                    f"[turn] {turn_index}/{target_n_turns} judge step done in {time.perf_counter() - judge_started_at:.2f}s"
                )

            raw_results.append(
                {
                    "turn_index": turn_index,
                    "attempt": attempt,
                    "qa_raw": qa_raw,
                    "kb_raw": kb_raw,
                    "judge_raw": judge_raw,
                    "qa_output": qa_output,
                    "kb_output": kb_output,
                    "qa_judge": judge_output,
                }
            )

            duplicate, source_path = is_duplicate_question(
                qa_output=qa_output,
                question_inputs=question_inputs,
                doc_question_hashes=doc_question_hashes,
            )
            if duplicate and attempt < max_retries:
                avoid_sources.add(source_path)
                forced_mode = "fresh"
                LOGGER.info(
                    f"[dedup] Duplicate question for source '{source_path}'. Retrying ({attempt + 1}/{max_retries})"
                )
                continue

            update_coverage_ledger(
                paths=output_paths,
                qa_output=qa_output,
                question_inputs=question_inputs,
                used_topic_ids=used_topic_ids,
                used_question_hashes=used_question_hashes,
            )
            update_seed_memory(
                qa_output=qa_output,
                question_inputs=question_inputs,
                used_seed_hashes=used_seed_hashes,
                seed_topic_usage=seed_topic_usage,
            )
            update_doc_question_memory(
                qa_output=qa_output,
                question_inputs=question_inputs,
                doc_question_hashes=doc_question_hashes,
                doc_recent_questions=doc_recent_questions,
            )

            turn_payload = {
                "turn_index": turn_index,
                "qa_generation": qa_output,
                "kb_answer": kb_output,
                "qa_judge": judge_output,
                "qa_agent_used_name": base_inputs.get("user_agent_used_name"),
                "kb_agent_used_name": base_inputs.get("assistant_agent_used_name"),
                "seed_topic": question_inputs.get("seed_topic"),
                "seed_question": question_inputs.get("seed_question"),
                "question_mode": question_inputs.get("question_mode"),
                "tool_events": tool_events,
            }
            turns.append(turn_payload)
            last_result = kb_output

            checkpoint_run_state(
                paths=output_paths,
                run_id=run_id,
                status="in_progress",
                base_inputs=base_inputs,
                n_turns=target_n_turns,
                turns=turns,
                raw_results=raw_results,
            )
            LOGGER.info(
                f"[turn] {turn_index}/{target_n_turns} completed in {time.perf_counter() - turn_started_at:.2f}s"
            )
            break

    if resolve_judge_enabled(cfg) and _resolve_judge_granularity(cfg) == "conversation" and turns:
        conversation_judge, conversation_judge_raw = _generate_conversation_judge_turn(
            cfg=cfg,
            model_client=model_client,
            inputs=base_inputs,
            turns=turns,
        )
        base_inputs["conversation_judge"] = conversation_judge
        shared_inputs["conversation_judge"] = conversation_judge
        raw_results.append(
            {
                "conversation_judge_raw": conversation_judge_raw,
                "conversation_judge": conversation_judge,
            }
        )

    checkpoint_run_state(
        paths=output_paths,
        run_id=run_id,
        status="completed",
        base_inputs=base_inputs,
        n_turns=target_n_turns,
        turns=turns,
        raw_results=raw_results,
    )
    LOGGER.info(
        f"[run-state] Completed {len(turns)}/{target_n_turns} turns in {time.perf_counter() - run_started_at:.2f}s"
    )

    return turns, raw_results, last_result

async def run_multi_turn_batched_async(
    cfg: Dict[str, Any],
    output_paths: OutputPaths,
    base_inputs: Dict[str, Any],
    max_turns: int,
    batch_size: int,
    min_turns: int = 1,
    turn_count_distribution: str = "poisson",
    turn_count_mean: float = 0.0,
    conversation_inputs_by_index: List[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run multi turn batched async.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        output_paths (OutputPaths): Filesystem path used by this operation.
        base_inputs (Dict[str, Any]): Mapping payload for this operation.
        max_turns (int): int value used by this operation.
        batch_size (int): Numeric control value for processing behavior.
        min_turns (int): int value used by this operation.
        turn_count_distribution (str): str value used by this operation.
        turn_count_mean (float): float value used by this operation.
        conversation_inputs_by_index (List[Dict[str, Any]] | None): List[Dict[str, Any]] | None value used by this operation.
    
    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]: Value produced by this API.
    
    Raises:
        RuntimeError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.runner import run_multi_turn_batched_async
        >>> run_multi_turn_batched_async(...)
    
    """
    run_id, resume_state = init_batched_run_state(output_paths, base_inputs, max_turns, batch_size)
    planned_turns = _sample_turn_targets(
        batch_size=batch_size,
        min_turns=min_turns,
        max_turns=max_turns,
        distribution=turn_count_distribution,
        mean=turn_count_mean,
    )
    if resume_state:
        state_batch_size = int(resume_state.get("batch_size") or 0)
        if state_batch_size and state_batch_size != batch_size:
            raise RuntimeError(
                f"Batch size mismatch for resume: requested={batch_size}, state={state_batch_size}."
            )
        if not isinstance(resume_state.get("conversations"), list):
            raise RuntimeError(
                "Cannot resume batched run: run_state is missing `conversations`."
            )
        stored_inputs = resume_state.get("inputs") or {}
        if stored_inputs:
            base_inputs = {**stored_inputs, **base_inputs}
        state_turns = int(resume_state.get("n_turns") or max_turns)
        if state_turns > 0:
            max_turns = state_turns
        resume_defaults = [max_turns for _ in range(max(batch_size, 0))]
        conversations = load_batched_conversations_from_state(run_id, resume_state, resume_defaults)
        LOGGER.info(f"[run-state] Resuming batched run_id={run_id} batch_size={batch_size}")
    else:
        conversations = build_initial_batched_conversations(run_id, planned_turns)
        LOGGER.info(f"[run-state] Starting batched run_id={run_id} batch_size={batch_size}")

    for index, slot in enumerate(conversations):
        existing_inputs = slot.get("inputs")
        if isinstance(existing_inputs, dict) and existing_inputs:
            continue
        if conversation_inputs_by_index and index < len(conversation_inputs_by_index):
            slot["inputs"] = dict(conversation_inputs_by_index[index])
        else:
            slot["inputs"] = dict(base_inputs)

    max_turns = max((_slot_target_n_turns(slot, fallback=max_turns) for slot in conversations), default=max_turns)
    base_inputs = {
        **base_inputs,
        "run_id": run_id,
        "n_turns": max_turns,
        "batch_size": batch_size,
        "min_turns": min_turns,
        "max_turns": max_turns,
        "turn_count_distribution": turn_count_distribution,
        "turn_count_mean": turn_count_mean,
    }

    ledger_entries = load_coverage_ledger(output_paths)
    used_topic_ids = {entry.get("topic_id") for entry in ledger_entries if entry.get("topic_id")}
    used_question_hashes = {entry.get("question_hash") for entry in ledger_entries if entry.get("question_hash")}
    used_seed_hashes = build_used_seed_hashes(ledger_entries)
    seed_topic_usage = build_seed_topic_usage(ledger_entries)
    recent_ledger_questions = [entry.get("question", "") for entry in ledger_entries if entry.get("question")][-8:]
    doc_usage = build_doc_usage(ledger_entries)
    doc_chunk_counts = build_doc_chunk_counts()
    doc_question_hashes = build_doc_question_hashes(ledger_entries)
    doc_recent_questions = build_doc_recent_questions(ledger_entries)

    seen_questions = _collect_batched_user_questions(conversations)
    seen_questions.extend(
        str(entry.get("question") or "").strip()
        for entry in ledger_entries
        if str(entry.get("question") or "").strip()
    )
    dedup_registry = RunQuestionRegistry(seen_questions)

    checkpoint_batched_run_state(
        paths=output_paths,
        run_id=run_id,
        status="started" if not resume_state else "resumed",
        base_inputs=base_inputs,
        n_turns=max_turns,
        batch_size=batch_size,
        conversations=conversations,
    )

    model_client = OpenAIModelClient()
    tools_cfg = cfg.get("tools", {}) or {}
    web_enabled = bool(tools_cfg.get("web_search_enabled", True))
    web_client = SerperWebSearchClient(
        num_results=int(tools_cfg.get("serper_num_results", 5) or 5),
        timeout=int(tools_cfg.get("serper_timeout", 30) or 30),
    )

    run_started_at = time.perf_counter()
    max_retries = env_int("QUESTION_DEDUP_RETRIES", default=3)
    judge_per_turn = _judge_per_turn_enabled(cfg)

    for turn_index in range(1, max_turns + 1):
        turn_started_at = time.perf_counter()
        active_indexes = [
            index
            for index, slot in enumerate(conversations)
            if (
                slot.get("status") == "active"
                and len(slot.get("turns") or []) == turn_index - 1
                and len(slot.get("turns") or []) < _slot_target_n_turns(slot, fallback=max_turns)
            )
        ]
        if not active_indexes:
            continue

        pending = set(active_indexes)
        accepted_candidates: Dict[int, Dict[str, Any]] = {}
        avoid_sources: Dict[int, set[str]] = {index: set() for index in active_indexes}
        forced_mode: Dict[int, str] = {index: "" for index in active_indexes}

        for attempt in range(max_retries + 1):
            if not pending:
                break

            sorted_pending = sorted(pending)
            qa_tasks = []
            slot_inputs: Dict[int, Dict[str, Any]] = {}
            for index in sorted_pending:
                slot = conversations[index]
                slot_base_inputs = slot.get("inputs") if isinstance(slot.get("inputs"), dict) else base_inputs
                slot_target_turns = _slot_target_n_turns(slot, fallback=max_turns)
                slot_turns = slot.get("turns") if isinstance(slot.get("turns"), list) else []
                history = build_conversation_history(slot_turns)
                public_history = build_public_history(slot_turns)
                formatted_history = format_history(public_history)
                question_inputs = build_question_inputs(
                    base_inputs=slot_base_inputs,
                    turn_index=turn_index,
                    n_turns=slot_target_turns,
                    public_history=public_history,
                    used_topic_ids=used_topic_ids,
                    recent_ledger_questions=recent_ledger_questions,
                    doc_usage=doc_usage,
                    doc_chunk_counts=doc_chunk_counts,
                    doc_recent_questions=doc_recent_questions,
                    avoid_sources=avoid_sources[index],
                    forced_mode=forced_mode[index],
                    used_seed_hashes=used_seed_hashes,
                    seed_topic_usage=seed_topic_usage,
                )
                iteration_inputs = {
                    **slot_base_inputs,
                    "turn_index": turn_index,
                    "n_turns": slot_target_turns,
                    "conversation_index": index,
                    "conversation_history": history,
                    "formatted_conversation_history": formatted_history,
                    "public_conversation_history": formatted_history,
                    **question_inputs,
                }
                slot_inputs[index] = {
                    "question_inputs": question_inputs,
                    "iteration_inputs": iteration_inputs,
                }
                qa_tasks.append(_generate_user_turn_async(cfg, model_client, iteration_inputs))

            qa_results = await asyncio.gather(*qa_tasks)

            dedup_candidates: List[Tuple[int, str]] = []
            attempt_candidates: Dict[int, Dict[str, Any]] = {}
            doc_rejected: set[int] = set()
            for index, (qa_output, qa_raw) in zip(sorted_pending, qa_results):
                question_inputs = slot_inputs[index]["question_inputs"]
                iteration_inputs = slot_inputs[index]["iteration_inputs"]
                duplicate, source_path = is_duplicate_question(
                    qa_output=qa_output,
                    question_inputs=question_inputs,
                    doc_question_hashes=doc_question_hashes,
                )
                candidate = {
                    "turn_index": turn_index,
                    "attempt": attempt,
                    "qa_output": qa_output,
                    "qa_raw": qa_raw,
                    "question_inputs": question_inputs,
                    "iteration_inputs": iteration_inputs,
                    "source_path": source_path,
                }
                attempt_candidates[index] = candidate
                if duplicate:
                    doc_rejected.add(index)
                    if source_path:
                        avoid_sources[index].add(source_path)
                    forced_mode[index] = "fresh"
                    continue

                dedup_candidates.append((index, str(qa_output.get("user_message") or "")))

            accepted_global, rejected_global = await dedup_registry.filter_and_commit(dedup_candidates)
            for index in rejected_global:
                forced_mode[index] = "fresh"

            for index in sorted_pending:
                if index in doc_rejected:
                    continue
                if index in accepted_global:
                    accepted_candidates[index] = attempt_candidates[index]
                    pending.discard(index)

            if pending and attempt == max_retries:
                for index in sorted(pending):
                    conversations[index]["status"] = "dropped"
                    conversations[index]["drop_reason"] = "dedup_exhausted"
                    LOGGER.warning(
                        f"[turn] {turn_index}/{max_turns} conversation_index={index} dropped (dedup exhausted)."
                    )
                pending.clear()

        accepted_indexes = sorted(accepted_candidates.keys())
        completion_tasks = [
            _complete_turn_async(
                cfg=cfg,
                model_client=model_client,
                iteration_inputs=accepted_candidates[index]["iteration_inputs"],
                qa_output=accepted_candidates[index]["qa_output"],
                web_enabled=web_enabled,
                web_client=web_client,
                judge_per_turn=judge_per_turn,
            )
            for index in accepted_indexes
        ]
        completion_results = await asyncio.gather(*completion_tasks) if completion_tasks else []

        for index, completion in zip(accepted_indexes, completion_results):
            slot = conversations[index]
            candidate = accepted_candidates[index]
            qa_output = candidate["qa_output"]
            qa_raw = candidate["qa_raw"]
            question_inputs = candidate["question_inputs"]
            kb_output = completion["kb_output"]
            kb_raw = completion["kb_raw"]
            judge_output = completion["judge_output"]
            judge_raw = completion["judge_raw"]
            tool_events = completion["tool_events"]

            raw_results = slot.get("raw_results") if isinstance(slot.get("raw_results"), list) else []
            raw_results.append(
                {
                    "turn_index": turn_index,
                    "attempt": candidate["attempt"],
                    "qa_raw": qa_raw,
                    "kb_raw": kb_raw,
                    "judge_raw": judge_raw,
                    "qa_output": qa_output,
                    "kb_output": kb_output,
                    "qa_judge": judge_output,
                }
            )
            slot["raw_results"] = raw_results

            update_coverage_ledger(
                paths=output_paths,
                qa_output=qa_output,
                question_inputs=question_inputs,
                used_topic_ids=used_topic_ids,
                used_question_hashes=used_question_hashes,
            )
            update_seed_memory(
                qa_output=qa_output,
                question_inputs=question_inputs,
                used_seed_hashes=used_seed_hashes,
                seed_topic_usage=seed_topic_usage,
            )
            update_doc_question_memory(
                qa_output=qa_output,
                question_inputs=question_inputs,
                doc_question_hashes=doc_question_hashes,
                doc_recent_questions=doc_recent_questions,
            )

            turn_payload = {
                "turn_index": turn_index,
                "qa_generation": qa_output,
                "kb_answer": kb_output,
                "qa_judge": judge_output,
                "qa_agent_used_name": candidate["iteration_inputs"].get("user_agent_used_name"),
                "kb_agent_used_name": candidate["iteration_inputs"].get("assistant_agent_used_name"),
                "seed_topic": question_inputs.get("seed_topic"),
                "seed_question": question_inputs.get("seed_question"),
                "question_mode": question_inputs.get("question_mode"),
                "tool_events": tool_events,
            }
            turns = slot.get("turns") if isinstance(slot.get("turns"), list) else []
            turns.append(turn_payload)
            slot["turns"] = turns

            slot_target_turns = _slot_target_n_turns(slot, fallback=max_turns)
            if len(turns) >= slot_target_turns:
                slot["status"] = "completed"

        checkpoint_batched_run_state(
            paths=output_paths,
            run_id=run_id,
            status="in_progress",
            base_inputs=base_inputs,
            n_turns=max_turns,
            batch_size=batch_size,
            conversations=conversations,
        )
        LOGGER.info(
            f"[turn] {turn_index}/{max_turns} batched completed in {time.perf_counter() - turn_started_at:.2f}s "
            f"(accepted={len(accepted_indexes)})"
        )

    for slot in conversations:
        turns = slot.get("turns") if isinstance(slot.get("turns"), list) else []
        slot_target_turns = _slot_target_n_turns(slot, fallback=max_turns)
        if slot.get("status") == "active" and len(turns) >= slot_target_turns:
            slot["status"] = "completed"

    if resolve_judge_enabled(cfg) and _resolve_judge_granularity(cfg) == "conversation":
        judge_indexes = [
            index
            for index, slot in enumerate(conversations)
            if isinstance(slot.get("turns"), list) and len(slot.get("turns") or []) > 0
        ]
        if judge_indexes:
            judge_tasks = []
            for index in judge_indexes:
                slot = conversations[index]
                slot_inputs = slot.get("inputs") if isinstance(slot.get("inputs"), dict) else {}
                turn_list = slot.get("turns") if isinstance(slot.get("turns"), list) else []
                judge_tasks.append(
                    _generate_conversation_judge_turn_async(
                        cfg=cfg,
                        model_client=model_client,
                        inputs={
                            **slot_inputs,
                            "conversation_index": index,
                        },
                        turns=turn_list,
                    )
                )
            judge_results = await asyncio.gather(*judge_tasks)
            for index, (conversation_judge, conversation_judge_raw) in zip(judge_indexes, judge_results):
                slot = conversations[index]
                slot_inputs = slot.get("inputs") if isinstance(slot.get("inputs"), dict) else {}
                slot_inputs["conversation_judge"] = conversation_judge
                slot["inputs"] = slot_inputs
                slot_raw_results = slot.get("raw_results") if isinstance(slot.get("raw_results"), list) else []
                slot_raw_results.append(
                    {
                        "conversation_judge_raw": conversation_judge_raw,
                        "conversation_judge": conversation_judge,
                    }
                )
                slot["raw_results"] = slot_raw_results

    checkpoint_batched_run_state(
        paths=output_paths,
        run_id=run_id,
        status="completed",
        base_inputs=base_inputs,
        n_turns=max_turns,
        batch_size=batch_size,
        conversations=conversations,
    )

    completed = sum(1 for slot in conversations if slot.get("status") == "completed")
    dropped = sum(1 for slot in conversations if slot.get("status") == "dropped")
    LOGGER.info(
        "[run-state] Completed batched run "
        f"batch_size={batch_size} completed={completed} dropped={dropped} "
        f"elapsed={time.perf_counter() - run_started_at:.2f}s"
    )
    return base_inputs, conversations

def _collect_batched_user_questions(conversations: List[Dict[str, Any]]) -> List[str]:
    questions: List[str] = []
    for slot in conversations:
        turns = slot.get("turns")
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            qa_output = turn.get("qa_generation") or {}
            if not isinstance(qa_output, dict):
                continue
            question = str(qa_output.get("user_message") or "").strip()
            if question:
                questions.append(question)
    return questions

async def _complete_turn_async(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    iteration_inputs: Dict[str, Any],
    qa_output: Dict[str, Any],
    web_enabled: bool,
    web_client: SerperWebSearchClient,
    judge_per_turn: bool,
) -> Dict[str, Any]:
    kb_output, tool_events, kb_raw = await _generate_assistant_turn_async(
        cfg=cfg,
        model_client=model_client,
        inputs=iteration_inputs,
        qa_output=qa_output,
        web_enabled=web_enabled,
        web_client=web_client,
    )
    _sanitize_retrieval_queries(kb_output)

    judge_output: Dict[str, Any] = {}
    judge_raw: Dict[str, Any] = {}
    if judge_per_turn:
        judge_output, judge_raw = await _generate_judge_turn_async(
            cfg=cfg,
            model_client=model_client,
            inputs=iteration_inputs,
            qa_output=qa_output,
            kb_output=kb_output,
        )
    return {
        "kb_output": kb_output,
        "tool_events": tool_events,
        "kb_raw": kb_raw,
        "judge_output": judge_output,
        "judge_raw": judge_raw,
    }

async def _generate_user_turn_async(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    settings = resolve_llm_settings(cfg, "qa_generator")
    system_prompt = build_agent_system_prompt("qa_generator")
    task_prompt = build_task_prompt("qa_generation_task", inputs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    result = await model_client.acomplete(settings, messages, response_format={"type": "json_object"})
    qa_output = parse_json_object(result.content)
    if not qa_output:
        repair_messages = messages + [
            {
                "role": "user",
                "content": "Return only a valid JSON object following the expected schema.",
            }
        ]
        repaired = await model_client.acomplete(settings, repair_messages, response_format={"type": "json_object"})
        qa_output = parse_json_object(repaired.content)
        return qa_output, repaired.raw

    return qa_output, result.raw

def _generate_user_turn(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    settings = resolve_llm_settings(cfg, "qa_generator")
    system_prompt = build_agent_system_prompt("qa_generator")
    task_prompt = build_task_prompt("qa_generation_task", inputs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    result = model_client.complete(settings, messages, response_format={"type": "json_object"})
    qa_output = parse_json_object(result.content)
    if not qa_output:
        repair_messages = messages + [
            {
                "role": "user",
                "content": "Return only a valid JSON object following the expected schema.",
            }
        ]
        repaired = model_client.complete(settings, repair_messages, response_format={"type": "json_object"})
        qa_output = parse_json_object(repaired.content)
        return qa_output, repaired.raw

    return qa_output, result.raw

def _assistant_tools_schema(web_enabled: bool) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "vector_db_search",
                "description": "Retrieve relevant passages from the local knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    if web_enabled:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search recent information from the public web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        )

    return tools

async def _generate_assistant_turn_async(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
    qa_output: Dict[str, Any],
    web_enabled: bool,
    web_client: SerperWebSearchClient,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    settings = resolve_llm_settings(cfg, "kb_responder")
    system_prompt = build_agent_system_prompt("kb_responder")
    task_prompt = build_task_prompt("kb_answer_task", inputs)

    qa_json = json.dumps(qa_output, ensure_ascii=False)
    user_content = f"{task_prompt}\n\nPrevious task output (user JSON):\n{qa_json}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    tools = _assistant_tools_schema(web_enabled=web_enabled)
    tool_events: List[Dict[str, Any]] = []

    raw_response: Dict[str, Any] = {}
    for _ in range(8):
        result = await model_client.acomplete(
            settings,
            messages,
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"},
        )
        raw_response = result.raw

        if result.tool_calls:
            messages.append({"role": "assistant", "content": result.content or "", "tool_calls": result.tool_calls})
            for call in result.tool_calls:
                name = ((call.get("function") or {}).get("name") or "").strip()
                args_raw = ((call.get("function") or {}).get("arguments") or "{}")
                args = parse_json_object(args_raw) if isinstance(args_raw, str) else {}
                payload = _execute_tool(name=name, args=args, cfg=cfg, web_client=web_client)

                event = {
                    "id": call.get("id"),
                    "name": name,
                    "arguments": args,
                    "result": payload,
                }
                tool_events.append(event)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
            continue

        kb_output = parse_json_object(result.content)
        if kb_output:
            if "did_web_search" not in kb_output:
                kb_output["did_web_search"] = any(item.get("name") == "web_search" for item in tool_events)
            return kb_output, tool_events, raw_response

        messages.append(
            {
                "role": "user",
                "content": "Return a valid JSON object only, following the expected output schema.",
            }
        )

    return {"assistant_message": "", "reasoning_trace": {}, "did_web_search": False}, tool_events, raw_response

def _generate_assistant_turn(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
    qa_output: Dict[str, Any],
    web_enabled: bool,
    web_client: SerperWebSearchClient,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    settings = resolve_llm_settings(cfg, "kb_responder")
    system_prompt = build_agent_system_prompt("kb_responder")
    task_prompt = build_task_prompt("kb_answer_task", inputs)

    qa_json = json.dumps(qa_output, ensure_ascii=False)
    user_content = f"{task_prompt}\n\nPrevious task output (user JSON):\n{qa_json}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    tools = _assistant_tools_schema(web_enabled=web_enabled)
    tool_events: List[Dict[str, Any]] = []

    raw_response: Dict[str, Any] = {}
    for _ in range(8):
        result = model_client.complete(
            settings,
            messages,
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"},
        )
        raw_response = result.raw

        if result.tool_calls:
            messages.append({"role": "assistant", "content": result.content or "", "tool_calls": result.tool_calls})
            for call in result.tool_calls:
                name = ((call.get("function") or {}).get("name") or "").strip()
                args_raw = ((call.get("function") or {}).get("arguments") or "{}")
                args = parse_json_object(args_raw) if isinstance(args_raw, str) else {}
                payload = _execute_tool(name=name, args=args, cfg=cfg, web_client=web_client)

                event = {
                    "id": call.get("id"),
                    "name": name,
                    "arguments": args,
                    "result": payload,
                }
                tool_events.append(event)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
            continue

        kb_output = parse_json_object(result.content)
        if kb_output:
            if "did_web_search" not in kb_output:
                kb_output["did_web_search"] = any(item.get("name") == "web_search" for item in tool_events)
            return kb_output, tool_events, raw_response

        messages.append(
            {
                "role": "user",
                "content": "Return a valid JSON object only, following the expected output schema.",
            }
        )

    return {"assistant_message": "", "reasoning_trace": {}, "did_web_search": False}, tool_events, raw_response

def _execute_tool(name: str, args: Dict[str, Any], cfg: Dict[str, Any], web_client: SerperWebSearchClient) -> Dict[str, Any]:
    if name == "vector_db_search":
        query = str(args.get("query") or "").strip()
        k = args.get("k")
        use_reranker = _as_bool((cfg.get("models", {}) or {}).get("use_reranker", False), default=False)
        retrieval_cfg = cfg.get("retrieval", {}) or {}
        default_k_raw = retrieval_cfg.get("default_k", 4)
        try:
            default_k = int(default_k_raw)
        except (TypeError, ValueError):
            default_k = 4
        default_k = default_k if default_k > 0 else 4
        effective_k = k if isinstance(k, int) and k > 0 else default_k
        TOOLS_LOGGER.info(f"[retrieval] vector_db_search k={effective_k} use_reranker={use_reranker}")
        return vector_db_search(query=query, k=k if isinstance(k, int) else None, use_reranker=use_reranker)

    if name == "web_search":
        query = str(args.get("query") or "").strip()
        try:
            return web_client.search(query)
        except Exception as err:
            return {
                "query": query,
                "results": [],
                "rendered": f"Web search failed: {err}",
            }

    return {"error": f"Unknown tool: {name}"}

def _generate_conversation_judge_turn(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
    turns: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not resolve_judge_enabled(cfg):
        return (
            {
                "score": 0,
                "reasons": ["other"],
                "notes": "disabled",
                "question_ok": False,
                "answer_ok": False,
            },
            {},
        )

    settings = resolve_llm_settings(cfg, "qa_judge")
    public_history = build_public_history(turns)
    formatted_history = format_history(public_history)
    judge_inputs = {
        **inputs,
        "public_conversation_history": formatted_history,
        "formatted_conversation_history": formatted_history,
        "judge_conversation_turns": json.dumps(_conversation_turns_for_judge(turns), ensure_ascii=False),
        "judge_conversation_evidence": json.dumps(_collect_conversation_evidence(turns), ensure_ascii=False),
        "judge_turn_count": len(turns),
    }

    system_prompt = build_agent_system_prompt("qa_judge")
    task_prompt = build_task_prompt("qa_conversation_judge_task", judge_inputs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    result = model_client.complete(settings, messages, response_format={"type": "json_object"})
    judge_output = parse_json_object(result.content)
    if not judge_output:
        judge_output = {
            "score": 0,
            "reasons": ["other"],
            "notes": "conversation judge parse failure",
            "question_ok": False,
            "answer_ok": False,
        }
    _log_conversation_judge_result(inputs, judge_output)
    return judge_output, result.raw

async def _generate_conversation_judge_turn_async(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
    turns: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not resolve_judge_enabled(cfg):
        return (
            {
                "score": 0,
                "reasons": ["other"],
                "notes": "disabled",
                "question_ok": False,
                "answer_ok": False,
            },
            {},
        )

    settings = resolve_llm_settings(cfg, "qa_judge")
    public_history = build_public_history(turns)
    formatted_history = format_history(public_history)
    judge_inputs = {
        **inputs,
        "public_conversation_history": formatted_history,
        "formatted_conversation_history": formatted_history,
        "judge_conversation_turns": json.dumps(_conversation_turns_for_judge(turns), ensure_ascii=False),
        "judge_conversation_evidence": json.dumps(_collect_conversation_evidence(turns), ensure_ascii=False),
        "judge_turn_count": len(turns),
    }

    system_prompt = build_agent_system_prompt("qa_judge")
    task_prompt = build_task_prompt("qa_conversation_judge_task", judge_inputs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    result = await model_client.acomplete(settings, messages, response_format={"type": "json_object"})
    judge_output = parse_json_object(result.content)
    if not judge_output:
        judge_output = {
            "score": 0,
            "reasons": ["other"],
            "notes": "conversation judge parse failure",
            "question_ok": False,
            "answer_ok": False,
        }
    _log_conversation_judge_result(inputs, judge_output)
    return judge_output, result.raw

def _generate_judge_turn(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
    qa_output: Dict[str, Any],
    kb_output: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not resolve_judge_enabled(cfg):
        return (
            {
                "score": 0,
                "reasons": ["other"],
                "notes": "disabled",
                "question_ok": False,
                "answer_ok": False,
            },
            {},
        )

    settings = resolve_llm_settings(cfg, "qa_judge")
    judge_inputs = {
        **inputs,
        "judge_user_message": qa_output.get("user_message", ""),
        "judge_assistant_message": kb_output.get("assistant_message", "") or "MISSING_ASSISTANT_MESSAGE",
        "judge_evidence": json.dumps(_collect_evidence(qa_output, kb_output), ensure_ascii=False),
    }

    system_prompt = build_agent_system_prompt("qa_judge")
    task_prompt = build_task_prompt("qa_judge_task", judge_inputs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    result = model_client.complete(settings, messages, response_format={"type": "json_object"})
    judge_output = parse_json_object(result.content)
    if not judge_output:
        judge_output = {
            "score": 0,
            "reasons": ["other"],
            "notes": "judge parse failure",
            "question_ok": False,
            "answer_ok": False,
        }
    _log_judge_result(inputs, judge_output)
    return judge_output, result.raw

async def _generate_judge_turn_async(
    cfg: Dict[str, Any],
    model_client: OpenAIModelClient,
    inputs: Dict[str, Any],
    qa_output: Dict[str, Any],
    kb_output: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not resolve_judge_enabled(cfg):
        return (
            {
                "score": 0,
                "reasons": ["other"],
                "notes": "disabled",
                "question_ok": False,
                "answer_ok": False,
            },
            {},
        )

    settings = resolve_llm_settings(cfg, "qa_judge")
    judge_inputs = {
        **inputs,
        "judge_user_message": qa_output.get("user_message", ""),
        "judge_assistant_message": kb_output.get("assistant_message", "") or "MISSING_ASSISTANT_MESSAGE",
        "judge_evidence": json.dumps(_collect_evidence(qa_output, kb_output), ensure_ascii=False),
    }

    system_prompt = build_agent_system_prompt("qa_judge")
    task_prompt = build_task_prompt("qa_judge_task", judge_inputs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    result = await model_client.acomplete(settings, messages, response_format={"type": "json_object"})
    judge_output = parse_json_object(result.content)
    if not judge_output:
        judge_output = {
            "score": 0,
            "reasons": ["other"],
            "notes": "judge parse failure",
            "question_ok": False,
            "answer_ok": False,
        }
    _log_judge_result(inputs, judge_output)
    return judge_output, result.raw

def _log_judge_result(inputs: Dict[str, Any], judge_output: Dict[str, Any]) -> None:
    turn_index = inputs.get("turn_index")
    conversation_index = inputs.get("conversation_index")
    JUDGE_LOGGER.info(
        "[judge-online] turn=%s conversation_index=%s score=%s question_ok=%s answer_ok=%s reasons=%s",
        turn_index if turn_index is not None else "-",
        conversation_index if conversation_index is not None else "-",
        judge_output.get("score"),
        judge_output.get("question_ok"),
        judge_output.get("answer_ok"),
        judge_output.get("reasons"),
    )

def _log_conversation_judge_result(inputs: Dict[str, Any], judge_output: Dict[str, Any]) -> None:
    conversation_index = inputs.get("conversation_index")
    JUDGE_LOGGER.info(
        "[judge-online-conversation] conversation_index=%s score=%s question_ok=%s answer_ok=%s reasons=%s",
        conversation_index if conversation_index is not None else "-",
        judge_output.get("score"),
        judge_output.get("question_ok"),
        judge_output.get("answer_ok"),
        judge_output.get("reasons"),
    )

def _conversation_turns_for_judge(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for turn in turns:
        qa = turn.get("qa_generation") or {}
        kb = turn.get("kb_answer") or {}
        rows.append(
            {
                "turn_index": turn.get("turn_index"),
                "user_message": qa.get("user_message"),
                "assistant_message": kb.get("assistant_message"),
            }
        )
    return rows

def _collect_conversation_evidence(turns: List[Dict[str, Any]], max_items: int = 120) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for turn in turns:
        qa = turn.get("qa_generation") if isinstance(turn.get("qa_generation"), dict) else {}
        kb = turn.get("kb_answer") if isinstance(turn.get("kb_answer"), dict) else {}
        collected.extend(_collect_evidence(qa, kb))
        if len(collected) >= max_items:
            return collected[:max_items]
    return collected

def _collect_evidence(qa: Dict[str, Any], kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for item in qa.get("grounding_facts", []) or []:
        if isinstance(item, dict):
            evidence.append({"source": item.get("source_descriptor"), "content": item.get("fact")})

    trace = kb.get("reasoning_trace") or {}
    for item in trace.get("evidence", []) or []:
        if isinstance(item, dict):
            evidence.append({"source": item.get("cue") or item.get("id"), "content": item.get("content")})
    return evidence

def persist_training_sample(
    output_paths: OutputPaths,
    inputs: Dict[str, Any],
    result: Any,
    turns: List[Dict[str, Any]],
    raw_results: List[Dict[str, Any]],
) -> None:
    """Persist training sample.
    
    Args:
        output_paths (OutputPaths): Filesystem path used by this operation.
        inputs (Dict[str, Any]): Mapping payload for this operation.
        result (Any): Input value for this operation.
        turns (List[Dict[str, Any]]): Conversation or message data used during processing.
        raw_results (List[Dict[str, Any]]): Conversation or message data used during processing.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.runner import persist_training_sample
        >>> persist_training_sample(...)
    
    """
    question = inputs.get("question") or ""
    inputs_with_turns = {**inputs, "n_turns": inputs.get("n_turns") or len(turns)}

    try:
        public_history = build_public_history(turns)
        run_id = inputs.get("run_id")
        dataset_path = save_training_sample(
            paths=output_paths,
            question=question,
            inputs=inputs_with_turns,
            result=result,
            turns=turns,
            conversation_history=build_conversation_history(turns),
            public_history=public_history,
            raw_results=raw_results,
            conversation_id=run_id,
        )
        LOGGER.info(f"[training-data] Sample appended to {dataset_path}")
    except Exception as err:
        LOGGER.error(f"[training-data] Failed to save sample: {err}")

def persist_batched_training_samples(
    output_paths: OutputPaths,
    base_inputs: Dict[str, Any],
    conversations: List[Dict[str, Any]],
) -> int:
    """Persist batched training samples.
    
    Args:
        output_paths (OutputPaths): Filesystem path used by this operation.
        base_inputs (Dict[str, Any]): Mapping payload for this operation.
        conversations (List[Dict[str, Any]]): List[Dict[str, Any]] value used by this operation.
    
    Returns:
        int: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.runner import persist_batched_training_samples
        >>> persist_batched_training_samples(...)
    
    """
    saved = 0
    for slot in sorted(conversations, key=lambda item: int(item.get("conversation_index", 0))):
        turns = slot.get("turns") if isinstance(slot.get("turns"), list) else []
        if not turns:
            continue

        raw_results = slot.get("raw_results") if isinstance(slot.get("raw_results"), list) else []
        last_result = turns[-1].get("kb_answer") if isinstance(turns[-1], dict) else {}
        conversation_id = str(slot.get("conversation_id") or "")
        slot_target_turns = _slot_target_n_turns(slot, fallback=len(turns) or 1)
        slot_inputs = slot.get("inputs") if isinstance(slot.get("inputs"), dict) else {}
        run_inputs = {
            **base_inputs,
            **slot_inputs,
            "run_id": conversation_id or base_inputs.get("run_id"),
            "conversation_index": slot.get("conversation_index"),
            "conversation_status": slot.get("status"),
            "drop_reason": slot.get("drop_reason") or "",
            # Persist per-conversation turn target (not batch-level max turns).
            "n_turns": slot_target_turns,
        }
        question = run_inputs.get("question") or ""
        inputs_with_turns = dict(run_inputs)

        try:
            public_history = build_public_history(turns)
            dataset_path = save_training_sample(
                paths=output_paths,
                question=question,
                inputs=inputs_with_turns,
                result=last_result,
                turns=turns,
                conversation_history=build_conversation_history(turns),
                public_history=public_history,
                raw_results=raw_results,
                conversation_id=conversation_id or None,
            )
            LOGGER.info(
                "[training-data] Batched sample appended to %s (conversation_index=%s, status=%s)",
                dataset_path,
                slot.get("conversation_index"),
                slot.get("status"),
            )
            saved += 1
        except Exception as err:
            LOGGER.error(
                "[training-data] Failed to save batched sample for conversation_index=%s: %s",
                slot.get("conversation_index"),
                err,
            )
    return saved

_LANGUAGE_PLACEHOLDERS = {
    "ar",
    "en",
    "fr",
    "english",
    "french",
    "arabic",
    "morocco",
    "moroccan",
    "egypt",
    "egyptian",
    "levantine",
    "uae",
    "gulf",
    "saudi",
}

def _sanitize_retrieval_queries(kb_output: Dict[str, Any]) -> None:
    trace = kb_output.get("reasoning_trace")
    if not isinstance(trace, dict):
        return
    retrieval = trace.get("retrieval_queries")
    if not isinstance(retrieval, dict):
        return
    for key in ("vector_db_search", "web_search"):
        retrieval[key] = _clean_query_list(retrieval.get(key))

def _clean_query_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = [str(item) for item in raw]
    else:
        items = [str(raw)]

    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if " " not in lowered and lowered in _LANGUAGE_PLACEHOLDERS:
            continue
        if " " not in lowered and len(lowered) <= 2:
            continue
        cleaned.append(text)
    return cleaned

def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default
