from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import Dict


_LOGGER_FILES: Dict[str, str] = {
    "dlgforge.retrieval": "retrieval.log",
    "dlgforge.tools": "tools.log",
    "dlgforge.llm": "llm.log",
    "dlgforge.judge": "judge.log",
}


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("dlgforge")
    configured_dir = getattr(root, "_dlgforge_logs_dir", None)
    configured_level = getattr(root, "_dlgforge_logs_level", None)
    if configured_dir == str(logs_dir) and configured_level == level:
        return

    _close_handlers(root)
    root.setLevel(level)
    root.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console_handler)

    run_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
    run_handler.setLevel(level)
    run_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(run_handler)

    for logger_name, filename in _LOGGER_FILES.items():
        logger = logging.getLogger(logger_name)
        _close_handlers(logger)
        logger.setLevel(level)
        logger.propagate = True
        file_handler = logging.FileHandler(logs_dir / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(file_handler)

    root._dlgforge_logs_dir = str(logs_dir)  # type: ignore[attr-defined]
    root._dlgforge_logs_level = level  # type: ignore[attr-defined]


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        with suppress(Exception):
            handler.close()
