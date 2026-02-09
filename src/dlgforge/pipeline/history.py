"""Conversation history formatting helpers.

"""

from __future__ import annotations

from typing import Any, Dict, List

def format_history(history: List[Dict[str, Any]]) -> str:
    """Format history.
    
    Args:
        history (List[Dict[str, Any]]): Conversation or message data used during processing.
    
    Returns:
        str: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.history import format_history
        >>> format_history(...)
    
    """
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
    """Build conversation history.
    
    Args:
        turns (List[Dict[str, Any]]): Conversation or message data used during processing.
    
    Returns:
        List[Dict[str, Any]]: Constructed value derived from the provided inputs.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.history import build_conversation_history
        >>> build_conversation_history(...)
    
    """
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
    """Build public history.
    
    Args:
        turns (List[Dict[str, Any]]): Conversation or message data used during processing.
    
    Returns:
        List[Dict[str, Any]]: Constructed value derived from the provided inputs.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.history import build_public_history
        >>> build_public_history(...)
    
    """
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
    """Messages up to turn.
    
    Args:
        messages (List[Dict[str, Any]]): Conversation or message data used during processing.
        turn_index (int | None): Numeric control value for processing behavior.
    
    Returns:
        List[Dict[str, Any]]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.history import messages_up_to_turn
        >>> messages_up_to_turn(...)
    
    """
    if turn_index is None:
        return messages
    return [entry for entry in messages if entry.get("turn_index") and entry["turn_index"] <= turn_index]
