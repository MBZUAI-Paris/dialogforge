from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OutputPaths:
    project_root: Path
    output_dir: Path

    def __post_init__(self) -> None:
        self.dataset_file = self.output_dir / "synthetic_qa.jsonl"
        self.coverage_ledger_path = self.output_dir / "coverage_ledger.jsonl"
        self.conversation_dir = self.output_dir / "conversations"
        self.conversation_index_file = self.output_dir / "conversations_index.jsonl"
        self.turn_dataset_file = self.output_dir / "turns.jsonl"
        self.conversation_sft_file = self.output_dir / "conversations_sharegpt.jsonl"
        self.conversation_sft_judged_file = self.output_dir / "conversations_sharegpt_judged.jsonl"
        self.run_state_dir = self.output_dir / "run_state"
        self.last_run_id_file = self.run_state_dir / "last_run_id.txt"


def ensure_output_layout(paths: OutputPaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.conversation_dir.mkdir(parents=True, exist_ok=True)
    paths.run_state_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        paths.dataset_file,
        paths.coverage_ledger_path,
        paths.conversation_index_file,
        paths.turn_dataset_file,
        paths.conversation_sft_file,
        paths.conversation_sft_judged_file,
    ]:
        if not path.exists():
            path.touch()


def _serialize_result(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    for attr in ("model_dump", "dict", "to_dict"):
        if hasattr(result, attr):
            try:
                data = getattr(result, attr)()
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {"repr": repr(result)}


def load_coverage_ledger(paths: OutputPaths, max_entries: int = 2000) -> List[Dict[str, Any]]:
    if not paths.coverage_ledger_path.exists():
        return []
    lines = paths.coverage_ledger_path.read_text(encoding="utf-8").splitlines()
    if max_entries > 0:
        lines = lines[-max_entries:]

    entries: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def append_coverage_ledger(paths: OutputPaths, entry: Dict[str, Any]) -> None:
    ensure_output_layout(paths)
    with paths.coverage_ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def save_run_state(paths: OutputPaths, run_id: str, payload: Dict[str, Any]) -> Path:
    ensure_output_layout(paths)
    run_file = paths.run_state_dir / f"{run_id}.json"
    run_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.last_run_id_file.write_text(run_id, encoding="utf-8")
    return run_file


def load_run_state(paths: OutputPaths, run_id: str) -> Optional[Dict[str, Any]]:
    run_file = paths.run_state_dir / f"{run_id}.json"
    if not run_file.exists():
        return None
    try:
        parsed = json.loads(run_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def save_training_sample(
    paths: OutputPaths,
    question: str,
    inputs: Dict[str, Any],
    result: Any,
    turns: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    public_history: Optional[List[Dict[str, Any]]] = None,
    raw_results: Optional[List[Dict[str, Any]]] = None,
    conversation_id: Optional[str] = None,
) -> Path:
    ensure_output_layout(paths)

    turns = turns or []
    conversation_history = conversation_history or []
    public_history = public_history or []
    raw_results = raw_results or []

    last_turn = turns[-1] if turns else {}
    conversation_id = conversation_id or uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    messages = _build_public_messages(public_history or conversation_history)

    entry = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "question": question,
        "inputs": inputs,
        "qa_generation_plan": last_turn.get("qa_generation") or {},
        "kb_final_answer": last_turn.get("kb_answer") or {},
        "qa_judge": last_turn.get("qa_judge") or {},
        "turns": turns,
        "conversation_history": conversation_history,
        "messages": messages,
        "raw_result": _serialize_result(result),
        "raw_results": raw_results,
    }

    with paths.dataset_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    _write_conversation_artifacts(
        paths=paths,
        conversation_id=conversation_id,
        timestamp=timestamp,
        question=question,
        inputs=inputs,
        turns=turns,
        conversation_history=conversation_history,
        messages=messages,
        raw_results=raw_results,
    )
    return paths.dataset_file


def append_sharegpt_judged_record(
    paths: OutputPaths,
    conversation_id: str,
    timestamp: str,
    inputs: Dict[str, Any],
    turns: List[Dict[str, Any]],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> None:
    ensure_output_layout(paths)
    sft_messages = _sharegpt_messages_from_turns(turns, fallback=messages or [])
    sft_messages_with_tools = _sharegpt_messages_with_tools_from_turns(turns, fallback=messages or [])
    metadata = _build_sft_metadata(conversation_id, timestamp, inputs, turns)
    judge = _build_judge_payload(turns)
    record = {
        "messages": sft_messages,
        "messages_with_tools": sft_messages_with_tools,
        "metadata": metadata,
        "user_reasoning": _build_user_reasoning(turns),
        "assistant_reasoning": _build_assistant_reasoning(turns),
        "judge": judge,
    }
    with paths.conversation_sft_judged_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _build_public_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for entry in history:
        role = entry.get("role")
        content = entry.get("message")
        if role and content:
            message = {
                "role": role,
                "content": content,
                "turn_index": entry.get("turn_index"),
            }
            agent = entry.get("agent")
            if agent:
                message["agent"] = agent
            messages.append(message)
    return messages


def _write_conversation_artifacts(
    paths: OutputPaths,
    conversation_id: str,
    timestamp: str,
    question: str,
    inputs: Dict[str, Any],
    turns: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    raw_results: List[Dict[str, Any]],
) -> None:
    conversation_file = paths.conversation_dir / f"{conversation_id}.json"
    messages_with_tools = _sharegpt_messages_with_tools_from_turns(turns, fallback=messages)

    payload = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "seed_question": question,
        "inputs": inputs,
        "n_turns": inputs.get("n_turns") or len(turns),
        "language": inputs.get("target_language"),
        "personas": {
            "user": inputs.get("user_persona"),
            "assistant": inputs.get("assistant_persona"),
            "user_id": inputs.get("user_persona_id"),
            "assistant_id": inputs.get("assistant_persona_id"),
        },
        "messages": messages,
        "messages_with_tools": messages_with_tools,
        "user_reasoning": _build_user_reasoning(turns),
        "assistant_reasoning": _build_assistant_reasoning(turns),
        "turns": turns,
        "conversation_history": conversation_history,
        "raw_results": raw_results,
    }
    conversation_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with paths.conversation_index_file.open("a", encoding="utf-8") as handle:
        index_entry = {
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "n_turns": inputs.get("n_turns") or len(turns),
            "seed_question": question,
            "language": inputs.get("target_language"),
            "conversation_file": str(conversation_file.relative_to(paths.project_root)),
        }
        handle.write(json.dumps(index_entry, ensure_ascii=False, default=str) + "\n")

    _append_sharegpt_record(paths, conversation_id, timestamp, inputs, turns, messages)

    with paths.turn_dataset_file.open("a", encoding="utf-8") as handle:
        for turn in turns:
            qa = turn.get("qa_generation") or {}
            kb = turn.get("kb_answer") or {}
            judge = turn.get("qa_judge") or {}
            record = {
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "turn_index": turn.get("turn_index"),
                "user_message": qa.get("user_message"),
                "assistant_message": kb.get("assistant_message"),
                "question_mode": turn.get("question_mode"),
                "seed_topic": turn.get("seed_topic"),
                "seed_question": turn.get("seed_question"),
                "intent": qa.get("intent"),
                "difficulty": qa.get("difficulty"),
                "coverage_target": qa.get("coverage_target"),
                "grounding_facts": qa.get("grounding_facts", []),
                "notes_for_assistant": qa.get("notes_for_assistant", []),
                "reasoning_trace": _normalize_reasoning_trace(kb.get("reasoning_trace")),
                "did_web_search": kb.get("did_web_search"),
                "judge_score": judge.get("score"),
                "judge_reasons": judge.get("reasons", []),
                "judge_notes": judge.get("notes"),
                "judge_question_ok": judge.get("question_ok"),
                "judge_answer_ok": judge.get("answer_ok"),
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _append_sharegpt_record(
    paths: OutputPaths,
    conversation_id: str,
    timestamp: str,
    inputs: Dict[str, Any],
    turns: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> None:
    sft_messages = _sharegpt_messages_from_turns(turns, fallback=messages)
    sft_messages_with_tools = _sharegpt_messages_with_tools_from_turns(turns, fallback=messages)
    metadata = _build_sft_metadata(conversation_id, timestamp, inputs, turns)
    record = {
        "messages": sft_messages,
        "messages_with_tools": sft_messages_with_tools,
        "metadata": metadata,
        "user_reasoning": _build_user_reasoning(turns),
        "assistant_reasoning": _build_assistant_reasoning(turns),
    }
    with paths.conversation_sft_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _sharegpt_messages_from_turns(turns: List[Dict[str, Any]], fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for turn in turns:
        qa = turn.get("qa_generation") or {}
        kb = turn.get("kb_answer") or {}

        user_message = qa.get("user_message")
        if user_message:
            row = {"role": "user", "content": user_message}
            agent = turn.get("qa_agent_used_name")
            if agent:
                row["agent"] = agent
            messages.append(row)

        assistant_message = kb.get("assistant_message")
        if assistant_message:
            row = {"role": "assistant", "content": assistant_message}
            agent = turn.get("kb_agent_used_name")
            if agent:
                row["agent"] = agent
            messages.append(row)

    if messages:
        return messages

    return [
        {
            "role": msg.get("role"),
            "content": msg.get("content"),
            **({"agent": msg.get("agent")} if msg.get("agent") else {}),
        }
        for msg in fallback
        if msg.get("role") and msg.get("content")
    ]


def _sharegpt_messages_with_tools_from_turns(
    turns: List[Dict[str, Any]],
    fallback: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    call_index = 0

    for turn in turns:
        qa = turn.get("qa_generation") or {}
        kb = turn.get("kb_answer") or {}

        user_message = qa.get("user_message")
        user_agent = turn.get("qa_agent_used_name")
        assistant_agent = turn.get("kb_agent_used_name")
        if user_message:
            row = {"role": "user", "content": user_message}
            if user_agent:
                row["agent"] = user_agent
            messages.append(row)

        tool_events = turn.get("tool_events") or []
        if isinstance(tool_events, list) and tool_events:
            tool_calls = []
            for event in tool_events:
                call_id = event.get("id") or f"call_{call_index}"
                call_index += 1
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": event.get("name", "unknown_tool"),
                            "arguments": json.dumps(event.get("arguments", {}), ensure_ascii=False),
                        },
                    }
                )
            assistant_tool_message = {"role": "assistant", "content": "", "tool_calls": tool_calls}
            if assistant_agent:
                assistant_tool_message["agent"] = assistant_agent
            messages.append(assistant_tool_message)

            for event in tool_events:
                call_id = event.get("id") or f"call_{call_index}"
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": event.get("name", "unknown_tool"),
                    "content": json.dumps(event.get("result", {}), ensure_ascii=False),
                }
                messages.append(tool_message)

        assistant_message = kb.get("assistant_message")
        if assistant_message:
            row = {"role": "assistant", "content": assistant_message}
            if assistant_agent:
                row["agent"] = assistant_agent
            messages.append(row)

    if messages:
        return messages

    return [
        {
            "role": msg.get("role"),
            "content": msg.get("content"),
            **({"agent": msg.get("agent")} if msg.get("agent") else {}),
        }
        for msg in fallback
        if msg.get("role") and msg.get("content")
    ]


def _build_sft_metadata(
    conversation_id: str,
    timestamp: str,
    inputs: Dict[str, Any],
    turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    difficulties = [((turn.get("qa_generation") or {}).get("difficulty") or "").strip() for turn in turns]
    intents = [((turn.get("qa_generation") or {}).get("intent") or "").strip() for turn in turns]
    coverage_targets = [
        ((turn.get("qa_generation") or {}).get("coverage_target") or "").strip() for turn in turns
    ]
    question_modes = [turn.get("question_mode") for turn in turns if turn.get("question_mode")]
    seed_topics = [turn.get("seed_topic") for turn in turns if turn.get("seed_topic")]
    seed_questions = [turn.get("seed_question") for turn in turns if turn.get("seed_question")]

    return {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "n_turns": inputs.get("n_turns") or len(turns),
        "language": inputs.get("target_language"),
        "personas": {
            "user": inputs.get("user_persona"),
            "assistant": inputs.get("assistant_persona"),
            "user_id": inputs.get("user_persona_id"),
            "assistant_id": inputs.get("assistant_persona_id"),
        },
        "difficulty_sequence": [d for d in difficulties if d],
        "difficulty_max": _max_difficulty(difficulties),
        "intent_sequence": [i for i in intents if i],
        "coverage_targets": [c for c in coverage_targets if c],
        "question_modes": question_modes,
        "seed_topics": seed_topics,
        "seed_questions": seed_questions,
    }


def _build_user_reasoning(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for turn in turns:
        qa = turn.get("qa_generation") or {}
        rows.append(
            {
                "turn_index": turn.get("turn_index"),
                "user_message": qa.get("user_message"),
                "intent": qa.get("intent"),
                "difficulty": qa.get("difficulty"),
                "coverage_target": qa.get("coverage_target"),
                "grounding_facts": qa.get("grounding_facts", []),
                "notes_for_assistant": qa.get("notes_for_assistant", []),
                "seed_topic": turn.get("seed_topic"),
                "seed_question": turn.get("seed_question"),
                "question_mode": turn.get("question_mode"),
            }
        )
    return rows


def _normalize_reasoning_trace(trace: Any) -> Dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    normalized = dict(trace)
    thinking = normalized.get("thinking")
    if thinking is None:
        return normalized
    if isinstance(thinking, str):
        normalized["thinking"] = [{"text": thinking}]
        return normalized
    if isinstance(thinking, list):
        cleaned: List[Dict[str, Any]] = []
        for item in thinking:
            if isinstance(item, dict):
                entry = dict(item)
                text = entry.get("text")
                if not isinstance(text, str):
                    entry["text"] = "" if text is None else str(text)
                cleaned.append(entry)
            else:
                cleaned.append({"text": "" if item is None else str(item)})
        normalized["thinking"] = cleaned
        return normalized
    normalized["thinking"] = [{"text": str(thinking)}]
    return normalized


def _build_assistant_reasoning(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for turn in turns:
        kb = turn.get("kb_answer") or {}
        rows.append(
            {
                "turn_index": turn.get("turn_index"),
                "assistant_message": kb.get("assistant_message"),
                "reasoning_trace": _normalize_reasoning_trace(kb.get("reasoning_trace")),
                "did_web_search": kb.get("did_web_search"),
            }
        )
    return rows


def _build_judge_payload(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_turn: List[Dict[str, Any]] = []
    for turn in turns:
        judge = turn.get("qa_judge") or {}
        if not judge:
            continue
        qa = turn.get("qa_generation") or {}
        kb = turn.get("kb_answer") or {}
        per_turn.append(
            {
                "turn_index": turn.get("turn_index"),
                "user_message": qa.get("user_message"),
                "assistant_message": kb.get("assistant_message"),
                "score": judge.get("score"),
                "reasons": judge.get("reasons", []),
                "notes": judge.get("notes"),
                "question_ok": judge.get("question_ok"),
                "answer_ok": judge.get("answer_ok"),
            }
        )
    return {
        "per_turn": per_turn,
        "avg_score": _avg_score([item.get("score") for item in per_turn]),
    }


def _max_difficulty(difficulties: List[str]) -> str:
    ranking = {"easy": 1, "medium": 2, "hard": 3}
    best = ""
    best_score = 0
    for diff in difficulties:
        score = ranking.get(diff, 0)
        if score > best_score:
            best_score = score
            best = diff
    return best


def _avg_score(values: List[Any]) -> Optional[float]:
    numeric = [x for x in values if isinstance(x, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)
