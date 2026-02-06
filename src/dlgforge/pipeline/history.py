from __future__ import annotations

from typing import Any, Dict, List


def format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No prior turns."
    lines: List[str] = []
    for entry in history:
        role = entry.get("role", "unknown").capitalize()
        turn = entry.get("turn_index")
        message = entry.get("message", "")
        lines.append(f"Turn {turn} - {role}: {message}")
    return "\n".join(lines)


def build_conversation_history(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for turn in turns:
        turn_index = turn.get("turn_index")
        qa = turn.get("qa_generation") or {}
        kb = turn.get("kb_answer") or {}

        user_message = qa.get("user_message")
        if user_message:
            history.append(
                {
                    "turn_index": turn_index,
                    "role": "user",
                    "message": user_message,
                    "agent": turn.get("qa_agent_used_name"),
                    "intent": qa.get("intent"),
                    "difficulty": qa.get("difficulty"),
                    "coverage_target": qa.get("coverage_target"),
                }
            )

        assistant_message = kb.get("assistant_message")
        if assistant_message:
            history.append(
                {
                    "turn_index": turn_index,
                    "role": "assistant",
                    "message": assistant_message,
                    "agent": turn.get("kb_agent_used_name"),
                    "confidence": (kb.get("reasoning_trace") or {}).get("confidence"),
                }
            )
    return history


def build_public_history(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for turn in turns:
        turn_index = turn.get("turn_index")
        qa = turn.get("qa_generation") or {}
        kb = turn.get("kb_answer") or {}

        user_message = qa.get("user_message")
        if user_message:
            history.append(
                {
                    "turn_index": turn_index,
                    "role": "user",
                    "message": user_message,
                    "agent": turn.get("qa_agent_used_name"),
                }
            )

        assistant_message = kb.get("assistant_message")
        if assistant_message:
            history.append(
                {
                    "turn_index": turn_index,
                    "role": "assistant",
                    "message": assistant_message,
                    "agent": turn.get("kb_agent_used_name"),
                }
            )
    return history


def messages_up_to_turn(messages: List[Dict[str, Any]], turn_index: int | None) -> List[Dict[str, Any]]:
    if turn_index is None:
        return messages
    return [entry for entry in messages if entry.get("turn_index") and entry["turn_index"] <= turn_index]
