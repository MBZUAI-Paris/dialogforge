from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dlgforge.config import (
    build_base_inputs,
    load_config,
    resolve_judge_enabled,
    resolve_n_turns,
    resolve_output_dir,
    resolve_retrieval_default_k,
)
from dlgforge.io import (
    OutputPaths,
    append_sharegpt_judged_record,
    ensure_output_layout,
    load_coverage_ledger,
    save_training_sample,
)
from dlgforge.llm import OpenAIModelClient, missing_models, required_agents, resolve_llm_settings
from dlgforge.pipeline.history import build_conversation_history, build_public_history, format_history
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
from dlgforge.tools import SerperWebSearchClient, configure_retrieval, vector_db_search
from dlgforge.utils import env_int, load_dotenv_files, parse_json_object, setup_logging


LOGGER = logging.getLogger("dlgforge.pipeline")
RETRIEVAL_LOGGER = logging.getLogger("dlgforge.retrieval")
JUDGE_LOGGER = logging.getLogger("dlgforge.judge")
TOOLS_LOGGER = logging.getLogger("dlgforge.tools")


def run(config_path: str) -> None:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists() or not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    project_root = config_file.parent
    setup_logging(project_root / "logs")
    load_dotenv_files(project_root)

    cfg, resolved_config_path, project_root = load_config(config_file)
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
    base_inputs["project_root"] = str(project_root)
    base_inputs["config_dir"] = str(resolved_config_path.parent)

    n_turns = resolve_n_turns(cfg)
    turns, raw_results, last_result = run_multi_turn(cfg, output_paths, base_inputs, n_turns)
    persist_training_sample(output_paths, base_inputs, last_result, turns, raw_results)
    maybe_auto_push_after_run(cfg, output_paths)


def run_judge_only(config_path: str) -> None:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists() or not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    project_root = config_file.parent
    setup_logging(project_root / "logs")
    load_dotenv_files(project_root)

    cfg, resolved_config_path, project_root = load_config(config_file)
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

        judged_turns = _judge_existing_turns(cfg, model_client, inputs, turns_raw)

        payload["turns"] = judged_turns
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        conversation_id = str(payload.get("conversation_id") or file_path.stem)
        timestamp = str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat())
        append_sharegpt_judged_record(
            paths=output_paths,
            conversation_id=conversation_id,
            timestamp=timestamp,
            inputs=inputs,
            turns=judged_turns,
            messages=messages,
        )

        judged_count += 1
        judged_turns_total += len(judged_turns)
        JUDGE_LOGGER.info(f"[judge] Judged {file_path.name} ({len(judged_turns)} turns)")

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

    base_url = (settings.get("base_url") or "").strip()
    api_key = (settings.get("api_key") or "").strip()
    if not api_key and not base_url:
        raise RuntimeError(
            "Missing qa_judge API key. Set OPENAI_API_KEY (or LLM_QA_JUDGE_API_KEY) in .env."
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

    for agent in required_agents(cfg):
        settings = resolve_llm_settings(cfg, agent)
        model = (settings.get("model") or "").strip()
        base_url = (settings.get("base_url") or "").strip()
        api_key = (settings.get("api_key") or "").strip()
        if model and not api_key and not base_url:
            raise RuntimeError(
                f"Missing API key for agent '{agent}'. Set OPENAI_API_KEY (or LLM_{agent.upper()}_API_KEY) in .env."
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


def run_multi_turn(
    cfg: Dict[str, Any],
    output_paths: OutputPaths,
    base_inputs: Dict[str, Any],
    n_turns: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    run_id, resume_state = init_run_state(output_paths, base_inputs, n_turns)
    if resume_state:
        stored_inputs = resume_state.get("inputs") or {}
        if stored_inputs:
            base_inputs = {**stored_inputs, **base_inputs}
        n_turns = int(resume_state.get("n_turns") or n_turns)
        LOGGER.info(f"[run-state] Resuming run_id={run_id} at turn {len(resume_state.get('turns', [])) + 1}")
    else:
        LOGGER.info(f"[run-state] Starting run_id={run_id}")

    base_inputs = {**base_inputs, "run_id": run_id, "n_turns": n_turns}

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
        n_turns=n_turns,
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
    for turn_index in range(start_turn, n_turns + 1):
        turn_started_at = time.perf_counter()
        LOGGER.info(f"[turn] {turn_index}/{n_turns} started")
        history = build_conversation_history(turns)
        public_history = build_public_history(turns)
        formatted_history = format_history(public_history)

        max_retries = env_int("QUESTION_DEDUP_RETRIES", default=3)
        avoid_sources: set[str] = set()
        forced_mode = ""

        for attempt in range(max_retries + 1):
            LOGGER.info(f"[turn] {turn_index}/{n_turns} attempt {attempt + 1}/{max_retries + 1} generating question")
            question_inputs = build_question_inputs(
                base_inputs=base_inputs,
                turn_index=turn_index,
                n_turns=n_turns,
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
            LOGGER.info(f"[turn] {turn_index}/{n_turns} qa_generator done in {time.perf_counter() - qa_started_at:.2f}s")

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
                f"[turn] {turn_index}/{n_turns} kb_responder done in {time.perf_counter() - kb_started_at:.2f}s "
                f"(tool_calls={len(tool_events)})"
            )
            _sanitize_retrieval_queries(kb_output)

            judge_started_at = time.perf_counter()
            judge_output, judge_raw = _generate_judge_turn(
                cfg=cfg,
                model_client=model_client,
                inputs=iteration_inputs,
                qa_output=qa_output,
                kb_output=kb_output,
            )
            LOGGER.info(f"[turn] {turn_index}/{n_turns} judge step done in {time.perf_counter() - judge_started_at:.2f}s")

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
                n_turns=n_turns,
                turns=turns,
                raw_results=raw_results,
            )
            LOGGER.info(f"[turn] {turn_index}/{n_turns} completed in {time.perf_counter() - turn_started_at:.2f}s")
            break

    checkpoint_run_state(
        paths=output_paths,
        run_id=run_id,
        status="completed",
        base_inputs=base_inputs,
        n_turns=n_turns,
        turns=turns,
        raw_results=raw_results,
    )
    LOGGER.info(f"[run-state] Completed {len(turns)}/{n_turns} turns in {time.perf_counter() - run_started_at:.2f}s")

    return turns, raw_results, last_result


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
    return judge_output, result.raw


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
