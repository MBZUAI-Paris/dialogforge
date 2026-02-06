from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dlgforge.io.output import OutputPaths, load_run_state, save_run_state

LOGGER = logging.getLogger("dlgforge.pipeline")


def init_run_state(paths: OutputPaths, base_inputs: Dict[str, Any], n_turns: int) -> Tuple[str, Dict[str, Any]]:
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
