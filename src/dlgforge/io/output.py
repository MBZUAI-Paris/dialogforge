"""Output path definitions and artifact persistence helpers.

"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SFT_COLUMN_DEFAULTS: Dict[str, str] = {
    "messages": "messages",
    "messages_with_tools": "messages_with_tools",
    "metadata": "metadata",
    "user_reasoning": "user_reasoning",
    "assistant_reasoning": "assistant_reasoning",
    "judge": "judge",
}
_SFT_COLUMN_ALIASES: Dict[str, str] = {
    "message_with_tools": "messages_with_tools",
}
_SFT_COLUMNS: Dict[str, str] = dict(_SFT_COLUMN_DEFAULTS)

def configure_output_columns(columns: Optional[Dict[str, Any]]) -> None:
    """Configure output columns.
    
    Args:
        columns (Optional[Dict[str, Any]]): Optional[Dict[str, Any]] value used by this operation.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import configure_output_columns
        >>> configure_output_columns(...)
    
    """
    configured = dict(_SFT_COLUMN_DEFAULTS)
    if isinstance(columns, dict):
        for key, raw_value in columns.items():
            source_key = str(key).strip()
            target_key = _SFT_COLUMN_ALIASES.get(source_key, source_key)
            if target_key not in configured:
                continue
            value = str(raw_value or "").strip()
            if value:
                configured[target_key] = value

    _SFT_COLUMNS.clear()
    _SFT_COLUMNS.update(configured)

def _sft_columns() -> Dict[str, str]:
    return _SFT_COLUMNS

@dataclass
class OutputPaths:
    """Resolved filesystem paths for all generated output artifacts.
    
    Args:
        project_root (Path): Resolved project directory context.
        output_dir (Path): Path value used by this operation.
    
    Raises:
        Exception: Construction may raise when required dependencies or inputs are invalid.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Instantiate and use through documented public methods.
    
    Examples:
        >>> from dlgforge.io.output import OutputPaths
        >>> OutputPaths(...)
    
    """
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
    """Ensure output layout.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import ensure_output_layout
        >>> ensure_output_layout(...)
    
    """
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
    """Load coverage ledger.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        max_entries (int): Numeric control value for processing behavior.
    
    Returns:
        List[Dict[str, Any]]: Loaded value parsed from upstream sources.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import load_coverage_ledger
        >>> load_coverage_ledger(...)
    
    """
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
    """Append coverage ledger.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        entry (Dict[str, Any]): Mapping payload for this operation.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import append_coverage_ledger
        >>> append_coverage_ledger(...)
    
    """
    ensure_output_layout(paths)
    with paths.coverage_ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

def save_run_state(paths: OutputPaths, run_id: str, payload: Dict[str, Any]) -> Path:
    """Save run state.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        run_id (str): Identifier for run state tracking.
        payload (Dict[str, Any]): Mapping payload for this operation.
    
    Returns:
        Path: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import save_run_state
        >>> save_run_state(...)
    
    """
    ensure_output_layout(paths)
    run_file = paths.run_state_dir / f"{run_id}.json"
    run_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.last_run_id_file.write_text(run_id, encoding="utf-8")
    return run_file

def load_run_state(paths: OutputPaths, run_id: str) -> Optional[Dict[str, Any]]:
    """Load run state.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        run_id (str): Identifier for run state tracking.
    
    Returns:
        Optional[Dict[str, Any]]: Loaded value parsed from upstream sources.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import load_run_state
        >>> load_run_state(...)
    
    """
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
    """Save training sample.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        question (str): str value used by this operation.
        inputs (Dict[str, Any]): Mapping payload for this operation.
        result (Any): Input value for this operation.
        turns (Optional[List[Dict[str, Any]]]): Conversation or message data used during processing.
        conversation_history (Optional[List[Dict[str, Any]]]): Optional[List[Dict[str, Any]]] value used by this operation.
        public_history (Optional[List[Dict[str, Any]]]): Conversation or message data used during processing.
        raw_results (Optional[List[Dict[str, Any]]]): Conversation or message data used during processing.
        conversation_id (Optional[str]): Identifier for a conversation artifact.
    
    Returns:
        Path: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import save_training_sample
        >>> save_training_sample(...)
    
    """
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
        "conversation_judge": inputs.get("conversation_judge") or {},
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
    conversation_judge: Optional[Dict[str, Any]] = None,
) -> None:
    """Append sharegpt judged record.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        conversation_id (str): Identifier for a conversation artifact.
        timestamp (str): str value used by this operation.
        inputs (Dict[str, Any]): Mapping payload for this operation.
        turns (List[Dict[str, Any]]): Conversation or message data used during processing.
        messages (Optional[List[Dict[str, Any]]]): Conversation or message data used during processing.
        conversation_judge (Optional[Dict[str, Any]]): Optional[Dict[str, Any]] value used by this operation.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.io.output import append_sharegpt_judged_record
        >>> append_sharegpt_judged_record(...)
    
    """
    ensure_output_layout(paths)
    columns = _sft_columns()
    sft_messages = _sharegpt_messages_from_turns(turns, fallback=messages or [])
    sft_messages_with_tools = _sharegpt_messages_with_tools_from_turns(turns, fallback=messages or [])
    metadata = _build_sft_metadata(conversation_id, timestamp, inputs, turns)
    judge = _build_judge_payload(turns, conversation_judge=conversation_judge or inputs.get("conversation_judge"))
    record = {
        columns["messages"]: sft_messages,
        columns["messages_with_tools"]: sft_messages_with_tools,
        columns["metadata"]: metadata,
        columns["user_reasoning"]: _build_user_reasoning(turns),
        columns["assistant_reasoning"]: _build_assistant_reasoning(turns),
        columns["judge"]: judge,
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
        "conversation_judge": inputs.get("conversation_judge") or {},
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
    if _should_append_judged_record(inputs, turns):
        append_sharegpt_judged_record(
            paths=paths,
            conversation_id=conversation_id,
            timestamp=timestamp,
            inputs=inputs,
            turns=turns,
            messages=messages,
            conversation_judge=inputs.get("conversation_judge"),
        )

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
                "judge_granularity": inputs.get("judge_granularity"),
                "judge_score": judge.get("score"),
                "judge_reasons": judge.get("reasons", []),
                "judge_notes": judge.get("notes"),
                "judge_question_ok": judge.get("question_ok"),
                "judge_answer_ok": judge.get("answer_ok"),
                "conversation_judge_score": ((inputs.get("conversation_judge") or {}).get("score")),
                "conversation_judge_reasons": ((inputs.get("conversation_judge") or {}).get("reasons", [])),
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
    columns = _sft_columns()
    sft_messages = _sharegpt_messages_from_turns(turns, fallback=messages)
    sft_messages_with_tools = _sharegpt_messages_with_tools_from_turns(turns, fallback=messages)
    metadata = _build_sft_metadata(conversation_id, timestamp, inputs, turns)
    record = {
        columns["messages"]: sft_messages,
        columns["messages_with_tools"]: sft_messages_with_tools,
        columns["metadata"]: metadata,
        columns["user_reasoning"]: _build_user_reasoning(turns),
        columns["assistant_reasoning"]: _build_assistant_reasoning(turns),
    }
    with paths.conversation_sft_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

def _should_append_judged_record(inputs: Dict[str, Any], turns: List[Dict[str, Any]]) -> bool:
    conversation_judge = inputs.get("conversation_judge")
    if isinstance(conversation_judge, dict) and conversation_judge:
        notes = str(conversation_judge.get("notes") or "").strip().lower()
        if notes != "disabled":
            return True

    if bool(inputs.get("judge_enabled")):
        return True

    for turn in turns:
        judge = turn.get("qa_judge")
        if not isinstance(judge, dict) or not judge:
            continue
        notes = str(judge.get("notes") or "").strip().lower()
        if notes != "disabled":
            return True
    return False

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
        "judge_granularity": inputs.get("judge_granularity"),
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
    if isinstance(thinking, str):
        normalized["thinking"] = [{"text": thinking}]
    elif isinstance(thinking, list):
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
    elif thinking is not None:
        normalized["thinking"] = [{"text": str(thinking)}]

    premises = normalized.get("premises")
    if isinstance(premises, list):
        cleaned_premises: List[Dict[str, Any]] = []
        for item in premises:
            if isinstance(item, dict):
                premise_id = item.get("id")
                premise_text = item.get("text")
                evidence_refs_raw = item.get("evidence_refs", [])
                if isinstance(evidence_refs_raw, list):
                    evidence_refs = [str(ref) for ref in evidence_refs_raw if str(ref).strip()]
                elif evidence_refs_raw is None:
                    evidence_refs = []
                else:
                    evidence_ref = str(evidence_refs_raw).strip()
                    evidence_refs = [evidence_ref] if evidence_ref else []

                assumption_raw = item.get("assumption")
                if assumption_raw is None:
                    assumption_raw = item.get("note")
                if assumption_raw is None:
                    assumption_raw = item.get("text_note")
                if isinstance(assumption_raw, bool):
                    assumption = "true" if assumption_raw else "false"
                elif assumption_raw is None:
                    assumption = ""
                else:
                    assumption = str(assumption_raw)

                cleaned_premises.append(
                    {
                        "id": "" if premise_id is None else str(premise_id),
                        "text": "" if premise_text is None else str(premise_text),
                        "evidence_refs": evidence_refs,
                        "assumption": assumption,
                    }
                )
            else:
                cleaned_premises.append(
                    {
                        "id": "",
                        "text": "" if item is None else str(item),
                        "evidence_refs": [],
                        "assumption": "",
                    }
                )
        normalized["premises"] = cleaned_premises

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

def _build_judge_payload(
    turns: List[Dict[str, Any]],
    conversation_judge: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        "conversation": conversation_judge if isinstance(conversation_judge, dict) else {},
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
