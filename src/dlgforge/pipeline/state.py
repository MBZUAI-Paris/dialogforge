"""Run-state initialization and checkpoint helpers.

"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dlgforge.io.output import OutputPaths, load_run_state, save_run_state

LOGGER = logging.getLogger("dlgforge.pipeline")

def init_run_state(paths: OutputPaths, base_inputs: Dict[str, Any], n_turns: int) -> Tuple[str, Dict[str, Any]]:
    """Initialize run state.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        base_inputs (Dict[str, Any]): Mapping payload for this operation.
        n_turns (int): Numeric control value for processing behavior.
    
    Returns:
        Tuple[str, Dict[str, Any]]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.state import init_run_state
        >>> init_run_state(...)
    
    """
    _ = n_turns
    resume_id = (base_inputs.get("resume_run_id") or "").strip()
    if resume_id:
        state = load_run_state(paths, resume_id)
        if state:
            return resume_id, state
        LOGGER.warning(f"[run-state] Warning: resume id {resume_id} not found; starting new run.")

    run_id = (base_inputs.get("run_id") or "").strip() or uuid.uuid4().hex
    return run_id, {}

def checkpoint_run_state(
    paths: OutputPaths,
    run_id: str,
    status: str,
    base_inputs: Dict[str, Any],
    n_turns: int,
    turns: List[Dict[str, Any]],
    raw_results: List[Dict[str, Any]],
) -> None:
    """Checkpoint run state.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        run_id (str): Identifier for run state tracking.
        status (str): str value used by this operation.
        base_inputs (Dict[str, Any]): Mapping payload for this operation.
        n_turns (int): Numeric control value for processing behavior.
        turns (List[Dict[str, Any]]): Conversation or message data used during processing.
        raw_results (List[Dict[str, Any]]): Conversation or message data used during processing.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.state import checkpoint_run_state
        >>> checkpoint_run_state(...)
    
    """
    payload = {
        "run_id": run_id,
        "status": status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "n_turns": n_turns,
        "inputs": base_inputs,
        "turns": turns,
        "raw_results": raw_results,
    }
    save_run_state(paths, run_id, payload)

def init_batched_run_state(
    paths: OutputPaths,
    base_inputs: Dict[str, Any],
    n_turns: int,
    batch_size: int,
) -> Tuple[str, Dict[str, Any]]:
    """Initialize batched run state.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        base_inputs (Dict[str, Any]): Mapping payload for this operation.
        n_turns (int): Numeric control value for processing behavior.
        batch_size (int): Numeric control value for processing behavior.
    
    Returns:
        Tuple[str, Dict[str, Any]]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.state import init_batched_run_state
        >>> init_batched_run_state(...)
    
    """
    _ = batch_size
    return init_run_state(paths, base_inputs, n_turns)

def build_initial_batched_conversations(run_id: str, target_turns: List[int]) -> List[Dict[str, Any]]:
    """Build initial batched conversations.
    
    Args:
        run_id (str): Identifier for run state tracking.
        target_turns (List[int]): List[int] value used by this operation.
    
    Returns:
        List[Dict[str, Any]]: Constructed value derived from the provided inputs.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.state import build_initial_batched_conversations
        >>> build_initial_batched_conversations(...)
    
    """
    conversations: List[Dict[str, Any]] = []
    for index, target in enumerate(target_turns):
        try:
            target_n_turns = int(target)
        except (TypeError, ValueError):
            target_n_turns = 1
        if target_n_turns <= 0:
            target_n_turns = 1
        conversations.append(
            {
                "conversation_index": index,
                "conversation_id": f"{run_id}-{index:04d}",
                "target_n_turns": target_n_turns,
                "status": "active",
                "drop_reason": "",
                "inputs": {},
                "turns": [],
                "raw_results": [],
            }
        )
    return conversations

def load_batched_conversations_from_state(
    run_id: str,
    state: Dict[str, Any],
    target_turns: List[int],
) -> List[Dict[str, Any]]:
    """Load batched conversations from state.
    
    Args:
        run_id (str): Identifier for run state tracking.
        state (Dict[str, Any]): Dict[str, Any] value used by this operation.
        target_turns (List[int]): List[int] value used by this operation.
    
    Returns:
        List[Dict[str, Any]]: Loaded value parsed from upstream sources.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.state import load_batched_conversations_from_state
        >>> load_batched_conversations_from_state(...)
    
    """
    conversations_raw = state.get("conversations")
    batch_size = len(target_turns)
    if not isinstance(conversations_raw, list):
        return build_initial_batched_conversations(run_id, target_turns)

    conversations: List[Dict[str, Any]] = []
    for index in range(batch_size):
        try:
            default_target = int(target_turns[index])
        except (TypeError, ValueError):
            default_target = 1
        if default_target <= 0:
            default_target = 1
        raw = conversations_raw[index] if index < len(conversations_raw) and isinstance(conversations_raw[index], dict) else {}
        status = str(raw.get("status") or "active")
        if status not in {"active", "completed", "dropped"}:
            status = "active"
        raw_target = raw.get("target_n_turns", default_target)
        try:
            target_n_turns = int(raw_target)
        except (TypeError, ValueError):
            target_n_turns = default_target
        if target_n_turns <= 0:
            target_n_turns = default_target
        slot_inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
        turns = raw.get("turns") if isinstance(raw.get("turns"), list) else []
        raw_results = raw.get("raw_results") if isinstance(raw.get("raw_results"), list) else []
        conversations.append(
            {
                "conversation_index": index,
                "conversation_id": str(raw.get("conversation_id") or f"{run_id}-{index:04d}"),
                "target_n_turns": target_n_turns,
                "status": status,
                "drop_reason": str(raw.get("drop_reason") or ""),
                "inputs": slot_inputs,
                "turns": turns,
                "raw_results": raw_results,
            }
        )
    return conversations

def checkpoint_batched_run_state(
    paths: OutputPaths,
    run_id: str,
    status: str,
    base_inputs: Dict[str, Any],
    n_turns: int,
    batch_size: int,
    conversations: List[Dict[str, Any]],
) -> None:
    """Checkpoint batched run state.
    
    Args:
        paths (OutputPaths): Filesystem path used by this operation.
        run_id (str): Identifier for run state tracking.
        status (str): str value used by this operation.
        base_inputs (Dict[str, Any]): Mapping payload for this operation.
        n_turns (int): Numeric control value for processing behavior.
        batch_size (int): Numeric control value for processing behavior.
        conversations (List[Dict[str, Any]]): List[Dict[str, Any]] value used by this operation.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.state import checkpoint_batched_run_state
        >>> checkpoint_batched_run_state(...)
    
    """
    payload = {
        "run_id": run_id,
        "status": status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "n_turns": n_turns,
        "batch_size": batch_size,
        "inputs": base_inputs,
        "conversations": conversations,
    }
    save_run_state(paths, run_id, payload)
