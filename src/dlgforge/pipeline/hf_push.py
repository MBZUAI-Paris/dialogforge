"""Hugging Face export preparation and push helpers.

"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dlgforge.config import load_config, resolve_output_columns, resolve_output_dir
from dlgforge.io import OutputPaths
from dlgforge.utils import load_dotenv_files, setup_logging

LOGGER = logging.getLogger("dlgforge.pipeline")

@dataclass
class HFPushSettings:
    """Resolved Hugging Face export/push settings.
    
    Args:
        enabled (bool): Boolean flag that controls optional behavior.
        auto_push_on_run (bool): bool value used by this operation.
        repo_id (str): str value used by this operation.
        repo_type (str): str value used by this operation.
        export_dir (Path): Path value used by this operation.
        include_run_state (bool): bool value used by this operation.
        private (bool): bool value used by this operation.
        commit_message (str): str value used by this operation.
        source_file (str): Filesystem path used by this operation.
        clean_remote (bool): Boolean flag that controls optional behavior.
        generate_stats (bool): bool value used by this operation.
        stats_file (str): str value used by this operation.
        generate_plots (bool): bool value used by this operation.
        plots_dir (str): str value used by this operation.
        output_columns (Dict[str, str]): Dict[str, str] value used by this operation.
    
    Raises:
        Exception: Construction may raise when required dependencies or inputs are invalid.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Instantiate and use through documented public methods.
    
    Examples:
        >>> from dlgforge.pipeline.hf_push import HFPushSettings
        >>> HFPushSettings(...)
    
    """
    enabled: bool
    auto_push_on_run: bool
    repo_id: str
    repo_type: str
    export_dir: Path
    include_run_state: bool
    private: bool
    commit_message: str
    source_file: str
    clean_remote: bool
    generate_stats: bool
    stats_file: str
    generate_plots: bool
    plots_dir: str
    output_columns: Dict[str, str]

@dataclass
class HFPushOptions:
    """CLI override options for export/push operations.
    
    Args:
        repo_id (str): str value used by this operation.
        repo_type (str): str value used by this operation.
        source_dir (str): str value used by this operation.
        export_dir (str): str value used by this operation.
        include_run_state (bool): bool value used by this operation.
        token (Optional[str]): Optional[str] value used by this operation.
        commit_message (str): str value used by this operation.
        prepare_export (bool): Boolean flag that controls optional behavior.
        push (bool): Boolean flag that controls optional behavior.
        clean_remote (bool): Boolean flag that controls optional behavior.
    
    Raises:
        Exception: Construction may raise when required dependencies or inputs are invalid.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Instantiate and use through documented public methods.
    
    Examples:
        >>> from dlgforge.pipeline.hf_push import HFPushOptions
        >>> HFPushOptions(...)
    
    """
    repo_id: str = ""
    repo_type: str = ""
    source_dir: str = ""
    export_dir: str = ""
    include_run_state: bool = False
    token: Optional[str] = None
    commit_message: str = ""
    prepare_export: bool = True
    push: bool = True
    clean_remote: bool = False

def run_push(config_path: str, options: HFPushOptions) -> None:
    """Prepare export artifacts and optionally push them to the Hub.
    
    Args:
        config_path (str): Path to a configuration file.
        options (HFPushOptions): Configuration mapping that controls runtime behavior.
    
    Returns:
        None: No value is returned.
    
    Raises:
        FileNotFoundError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.hf_push import run_push
        >>> run_push(...)
    
    """
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists() or not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    project_root = config_file.parent
    setup_logging(project_root / "logs")
    load_dotenv_files(project_root)

    cfg, _, resolved_project_root = load_config(config_file)
    settings = resolve_hf_push_settings(cfg, resolved_project_root)

    source_dir = _resolve_source_dir(options.source_dir, cfg, resolved_project_root)
    export_dir = _resolve_export_dir(options.export_dir, settings)

    repo_id = (options.repo_id or settings.repo_id).strip()
    repo_type = (options.repo_type or settings.repo_type).strip().lower() or "dataset"
    include_run_state = bool(settings.include_run_state or options.include_run_state)
    commit_message = (options.commit_message or settings.commit_message).strip() or "Update synthetic dataset export"
    clean_remote = bool(settings.clean_remote or options.clean_remote)

    if options.prepare_export:
        export_dir = prepare_export(
            source_dir=source_dir,
            export_dir=export_dir,
            source_file=settings.source_file,
            include_run_state=include_run_state,
            repo_id=repo_id,
            generate_stats=settings.generate_stats,
            stats_file=settings.stats_file,
            generate_plots=settings.generate_plots,
            plots_dir=settings.plots_dir,
            output_columns=settings.output_columns,
        )

    if options.push:
        push_to_hub(
            export_dir=export_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            token=options.token,
            commit_message=commit_message,
            private=settings.private,
            clean_remote=clean_remote,
        )
        LOGGER.info(f"[hf-push] Upload completed to {repo_id} ({repo_type}) from {export_dir}")

def maybe_auto_push_after_run(cfg: Dict[str, Any], output_paths: OutputPaths) -> None:
    """Conditionally execute auto push after run.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        output_paths (OutputPaths): Filesystem path used by this operation.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.hf_push import maybe_auto_push_after_run
        >>> maybe_auto_push_after_run(...)
    
    """
    settings = resolve_hf_push_settings(cfg, output_paths.project_root)
    if not settings.enabled or not settings.auto_push_on_run:
        return

    source_dir = output_paths.output_dir
    effective_source_file = _resolve_effective_source_file(source_dir, settings.source_file)
    source_path = source_dir / effective_source_file
    if not source_path.exists() or _count_lines(source_path) == 0:
        LOGGER.warning(
            "[hf-push] Auto-push skipped: "
            f"{effective_source_file} is missing or empty in {source_dir}."
        )
        return

    if not settings.repo_id.strip():
        LOGGER.warning("[hf-push] Auto-push skipped: saving.hf_push.repo_id is empty.")
        return

    try:
        export_dir = prepare_export(
            source_dir=source_dir,
            export_dir=settings.export_dir,
            source_file=settings.source_file,
            include_run_state=settings.include_run_state,
            repo_id=settings.repo_id,
            generate_stats=settings.generate_stats,
            stats_file=settings.stats_file,
            generate_plots=settings.generate_plots,
            plots_dir=settings.plots_dir,
            output_columns=settings.output_columns,
        )
        push_to_hub(
            export_dir=export_dir,
            repo_id=settings.repo_id,
            repo_type=settings.repo_type,
            token=None,
            commit_message=settings.commit_message,
            private=settings.private,
            clean_remote=settings.clean_remote,
        )
        LOGGER.info(f"[hf-push] Auto-push completed to {settings.repo_id} ({settings.repo_type}).")
    except Exception as err:
        LOGGER.warning(f"[hf-push] Auto-push skipped due to error: {err}")

def resolve_hf_push_settings(cfg: Dict[str, Any], project_root: Path) -> HFPushSettings:
    """Resolve hf push settings.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        project_root (Path): Resolved project directory context.
    
    Returns:
        HFPushSettings: Resolved value after applying defaults and normalization rules.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.hf_push import resolve_hf_push_settings
        >>> resolve_hf_push_settings(...)
    
    """
    saving_cfg = cfg.get("saving", {}) or {}
    hf_cfg = saving_cfg.get("hf_push", {}) or {}
    output_columns = resolve_output_columns(cfg)

    output_dir = resolve_output_dir(cfg, project_root)
    export_dir_raw = str(hf_cfg.get("export_dir", "hf_export") or "hf_export")
    export_dir = Path(export_dir_raw)
    if not export_dir.is_absolute():
        export_dir = output_dir / export_dir

    repo_type = str(hf_cfg.get("repo_type", "dataset") or "dataset").strip().lower()
    if repo_type not in {"dataset", "model", "space"}:
        repo_type = "dataset"

    return HFPushSettings(
        enabled=bool(hf_cfg.get("enabled", False)),
        auto_push_on_run=bool(hf_cfg.get("auto_push_on_run", False)),
        repo_id=str(hf_cfg.get("repo_id", "") or "").strip(),
        repo_type=repo_type,
        export_dir=export_dir,
        include_run_state=bool(hf_cfg.get("include_run_state", False)),
        private=bool(hf_cfg.get("private", True)),
        commit_message=str(hf_cfg.get("commit_message", "Update synthetic dataset export") or "").strip()
        or "Update synthetic dataset export",
        source_file=str(hf_cfg.get("source_file", "conversations_sharegpt_judged.jsonl") or "").strip()
        or "conversations_sharegpt_judged.jsonl",
        clean_remote=bool(hf_cfg.get("clean_remote", False)),
        generate_stats=bool(hf_cfg.get("generate_stats", False)),
        stats_file=str(hf_cfg.get("stats_file", "dataset_stats.json") or "").strip() or "dataset_stats.json",
        generate_plots=bool(hf_cfg.get("generate_plots", False)),
        plots_dir=str(hf_cfg.get("plots_dir", "plots") or "").strip() or "plots",
        output_columns=output_columns,
    )

def prepare_export(
    source_dir: Path,
    export_dir: Path,
    source_file: str,
    include_run_state: bool,
    repo_id: str | None,
    generate_stats: bool = False,
    stats_file: str = "dataset_stats.json",
    generate_plots: bool = False,
    plots_dir: str = "plots",
    output_columns: Optional[Dict[str, str]] = None,
) -> Path:
    """Prepare export.
    
    Args:
        source_dir (Path): Path value used by this operation.
        export_dir (Path): Path value used by this operation.
        source_file (str): Filesystem path used by this operation.
        include_run_state (bool): bool value used by this operation.
        repo_id (str | None): str | None value used by this operation.
        generate_stats (bool): bool value used by this operation.
        stats_file (str): str value used by this operation.
        generate_plots (bool): bool value used by this operation.
        plots_dir (str): str value used by this operation.
        output_columns (Optional[Dict[str, str]]): Optional[Dict[str, str]] value used by this operation.
    
    Returns:
        Path: Value produced by this API.
    
    Raises:
        FileNotFoundError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.hf_push import prepare_export
        >>> prepare_export(...)
    
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    effective_source_file = _resolve_effective_source_file(source_dir, source_file)
    source_path = source_dir / effective_source_file
    if not source_path.exists():
        raise FileNotFoundError(f"{effective_source_file} not found in {source_dir}")
    export_source_path = export_dir / effective_source_file
    shutil.copy2(source_path, export_source_path)
    if export_source_path.suffix.lower() == ".jsonl":
        if _sanitize_jsonl_for_hf(export_source_path):
            LOGGER.info(f"[hf-push] Applied JSONL schema sanitization: {export_source_path.name}")

    dataset_stats: Dict[str, Any] = {}
    plot_files: List[str] = []
    if (generate_stats or generate_plots) and export_source_path.suffix.lower() == ".jsonl":
        dataset_stats = _build_dataset_stats(export_source_path, output_columns or {})
    if generate_stats and dataset_stats:
        stats_name = (stats_file or "dataset_stats.json").strip() or "dataset_stats.json"
        stats_path = export_dir / stats_name
        stats_path.write_text(json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info(f"[hf-push] Wrote dataset stats to {stats_path}")
    if generate_plots and dataset_stats:
        plots_folder_name = (plots_dir or "plots").strip() or "plots"
        plots_path = export_dir / plots_folder_name
        plots_path.mkdir(parents=True, exist_ok=True)
        plot_files = _write_dataset_plots(plots_path, dataset_stats, plots_folder_name)
        LOGGER.info(f"[hf-push] Wrote {len(plot_files)} dataset plot(s) to {plots_path}")

    if include_run_state:
        run_state = source_dir / "run_state"
        if run_state.exists() and run_state.is_dir():
            shutil.copytree(run_state, export_dir / "run_state")

    _write_dataset_card(
        export_dir=export_dir,
        source_dir=source_dir,
        source_file=effective_source_file,
        repo_id=repo_id,
        dataset_stats=dataset_stats if dataset_stats else None,
        stats_file=stats_file,
        plot_files=plot_files,
    )
    return export_dir

def _resolve_effective_source_file(source_dir: Path, source_file: str) -> str:
    requested = source_file.strip()
    if requested != "conversations_sharegpt.jsonl":
        return requested

    judged = source_dir / "conversations_sharegpt_judged.jsonl"
    if judged.exists() and judged.is_file() and _count_lines(judged) > 0:
        LOGGER.info(
            "[hf-push] Using conversations_sharegpt_judged.jsonl for export "
            "(configured source_file=conversations_sharegpt.jsonl)."
        )
        return "conversations_sharegpt_judged.jsonl"
    return requested

def _sanitize_jsonl_for_hf(path: Path) -> bool:
    tmp_path = path.with_name(path.name + ".tmp")
    changed = False
    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                dst.write(line)
                continue
            try:
                row = json.loads(stripped)
            except Exception:
                dst.write(line)
                continue

            sanitized = _sanitize_hf_row(row)
            if sanitized != row:
                changed = True
            dst.write(json.dumps(sanitized, ensure_ascii=False, default=str) + "\n")

    if changed:
        tmp_path.replace(path)
    else:
        tmp_path.unlink(missing_ok=True)
    return changed

def _sanitize_hf_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    assistant_reasoning = row.get("assistant_reasoning")
    if not isinstance(assistant_reasoning, list):
        return row

    changed = False
    normalized_rows = []
    for reasoning_item in assistant_reasoning:
        if not isinstance(reasoning_item, dict):
            normalized_rows.append(reasoning_item)
            continue

        reasoning_entry = dict(reasoning_item)
        trace = reasoning_entry.get("reasoning_trace")
        if isinstance(trace, dict):
            normalized_trace = _sanitize_reasoning_trace_for_hf(trace)
            if normalized_trace != trace:
                reasoning_entry["reasoning_trace"] = normalized_trace
                changed = True
        normalized_rows.append(reasoning_entry)

    if not changed:
        return row
    out = dict(row)
    out["assistant_reasoning"] = normalized_rows
    return out

def _sanitize_reasoning_trace_for_hf(trace: Dict[str, Any]) -> Dict[str, Any]:
    changed = False
    out = dict(trace)

    question = _as_text(trace.get("question"))
    if ("question" not in trace) or question != trace.get("question"):
        changed = True
    out["question"] = question

    retrieval_queries = trace.get("retrieval_queries")
    canonical_retrieval_queries = _sanitize_retrieval_queries_for_hf(retrieval_queries)
    if retrieval_queries != canonical_retrieval_queries:
        changed = True
    out["retrieval_queries"] = canonical_retrieval_queries

    evidence = trace.get("evidence")
    canonical_evidence = _sanitize_evidence_for_hf(evidence)
    if evidence != canonical_evidence:
        changed = True
    out["evidence"] = canonical_evidence

    premises = trace.get("premises")
    canonical_premises = _sanitize_premises_for_hf(premises)
    if premises != canonical_premises:
        changed = True
    out["premises"] = canonical_premises

    thinking = trace.get("thinking")
    canonical_thinking = _sanitize_thinking_for_hf(thinking)
    if thinking != canonical_thinking:
        changed = True
    out["thinking"] = canonical_thinking

    confidence = _as_text(trace.get("confidence"))
    if ("confidence" not in trace) or confidence != trace.get("confidence"):
        changed = True
    out["confidence"] = confidence

    known_limits = _as_text_list(trace.get("known_limits"))
    if ("known_limits" not in trace) or known_limits != trace.get("known_limits"):
        changed = True
    out["known_limits"] = known_limits

    if not changed:
        return trace
    return out

def _sanitize_retrieval_queries_for_hf(raw: Any) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        return {"vector_db_search": [], "web_search": []}
    return {
        "vector_db_search": _as_text_list(raw.get("vector_db_search")),
        "web_search": _as_text_list(raw.get("web_search")),
    }

def _sanitize_evidence_for_hf(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "id": _as_text(item.get("id")),
                    "cue": _as_text(item.get("cue")),
                    "content": _as_text(item.get("content")),
                }
            )
        else:
            out.append({"id": "", "cue": "", "content": _as_text(item)})
    return out

def _sanitize_premises_for_hf(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []

    cleaned_premises: List[Dict[str, Any]] = []
    for premise in raw:
        if isinstance(premise, dict):
            assumption_raw = premise.get("assumption")
            if assumption_raw is None:
                assumption_raw = premise.get("note")
            if assumption_raw is None:
                assumption_raw = premise.get("text_note")
            assumption = _as_text(assumption_raw)

            cleaned_premises.append(
                {
                    "id": _as_text(premise.get("id")),
                    "text": _as_text(premise.get("text")),
                    "evidence_refs": _as_text_list(premise.get("evidence_refs")),
                    "assumption": assumption,
                }
            )
        else:
            cleaned_premises.append(
                {
                    "id": "",
                    "text": _as_text(premise),
                    "evidence_refs": [],
                    "assumption": "",
                }
            )
    return cleaned_premises

def _sanitize_thinking_for_hf(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []

    cleaned_thinking: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            text_raw = item.get("text")
            if text_raw is None:
                for key in ("assumption", "text_is_assumption", "note", "content", "thought"):
                    candidate = item.get(key)
                    if candidate is not None and str(candidate).strip():
                        text_raw = candidate
                        break
            cleaned_thinking.append({"text": _as_text(text_raw)})
        else:
            cleaned_thinking.append({"text": _as_text(item)})
    return cleaned_thinking

def _as_text(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw)

def _as_text_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        values: List[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item)
            if text.strip():
                values.append(text)
        return values
    text = str(raw)
    return [text] if text.strip() else []

def push_to_hub(
    export_dir: Path,
    repo_id: str,
    repo_type: str,
    token: Optional[str],
    commit_message: str,
    private: bool = True,
    clean_remote: bool = False,
) -> None:
    """Push to hub.
    
    Args:
        export_dir (Path): Path value used by this operation.
        repo_id (str): str value used by this operation.
        repo_type (str): str value used by this operation.
        token (Optional[str]): Optional[str] value used by this operation.
        commit_message (str): str value used by this operation.
        private (bool): bool value used by this operation.
        clean_remote (bool): Boolean flag that controls optional behavior.
    
    Returns:
        None: No value is returned.
    
    Raises:
        ValueError: Raised when validation or runtime requirements are not met.
        ImportError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.hf_push import push_to_hub
        >>> push_to_hub(...)
    
    """
    if not repo_id.strip():
        raise ValueError("HF repo_id is required. Set saving.hf_push.repo_id or pass --repo-id.")

    token = token or _env_token()
    if not token:
        raise ValueError("HF token not found. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, or pass --token.")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("huggingface_hub is required. Add it to dependencies.") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=private)

    upload_kwargs = {
        "folder_path": str(export_dir),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "commit_message": commit_message,
    }
    if clean_remote:
        upload_kwargs["delete_patterns"] = ["*", "**/*"]

    try:
        api.upload_folder(**upload_kwargs)
    except TypeError:
        # Fallback for older huggingface_hub versions lacking delete_patterns.
        if clean_remote:
            _clean_remote_repo(api, repo_id=repo_id, repo_type=repo_type, commit_message=commit_message)
            upload_kwargs.pop("delete_patterns", None)
            api.upload_folder(**upload_kwargs)
        else:
            raise

def _clean_remote_repo(api: Any, repo_id: str, repo_type: str, commit_message: str) -> None:
    keep_files = {".gitattributes"}
    files = [path for path in api.list_repo_files(repo_id=repo_id, repo_type=repo_type) if path not in keep_files]
    if not files:
        return

    try:
        from huggingface_hub import CommitOperationDelete

        operations = [CommitOperationDelete(path_in_repo=path) for path in files]
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=f"{commit_message} (clean remote)",
        )
        return
    except Exception:
        pass

    for path in files:
        api.delete_file(
            path_in_repo=path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"{commit_message} (clean remote)",
        )

def _resolve_source_dir(source_dir_override: str, cfg: Dict[str, Any], project_root: Path) -> Path:
    if source_dir_override.strip():
        return Path(source_dir_override).expanduser().resolve()
    return resolve_output_dir(cfg, project_root)

def _resolve_export_dir(export_dir_override: str, settings: HFPushSettings) -> Path:
    if export_dir_override.strip():
        return Path(export_dir_override).expanduser().resolve()
    return settings.export_dir

def _env_token() -> str:
    import os

    return (
        (os.getenv("HF_TOKEN") or "").strip()
        or (os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    )

def _write_dataset_card(
    export_dir: Path,
    source_dir: Path,
    source_file: str,
    repo_id: str | None,
    dataset_stats: Optional[Dict[str, Any]] = None,
    stats_file: str = "dataset_stats.json",
    plot_files: Optional[List[str]] = None,
) -> None:
    minimal_stats = _collect_minimal_stats(source_dir, source_file)
    stats = dataset_stats if isinstance(dataset_stats, dict) and dataset_stats else minimal_stats
    dataset_name = _infer_dataset_name(repo_id)
    lines = [
        "---",
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        f"        path: {source_file}",
        "language:",
        "  - en",
        "  - ar",
        "task_categories:",
        "  - question-answering",
        "tags:",
        "  - synthetic",
        "  - rag",
        "  - multi-turn",
        "---",
        "",
        f"# {dataset_name}",
        "",
        "Synthetic multi-turn QA conversations generated with dlgforge.",
        "",
        "## Contents",
        f"- `{source_file}`",
    ]
    if dataset_stats:
        lines.append(f"- `{stats_file}`")
        for plot in plot_files or []:
            lines.append(f"- `{plot}`")
    lines.extend(
        [
            "",
            "## Stats",
            f"- records: {stats.get('records', minimal_stats.get('records', 0))}",
        ]
    )

    turn_stats = stats.get("turn_count") if isinstance(stats.get("turn_count"), dict) else {}
    if turn_stats:
        lines.append(f"- turn_count.avg: {turn_stats.get('avg')}")
        lines.append(f"- turn_count.min: {turn_stats.get('min')}")
        lines.append(f"- turn_count.max: {turn_stats.get('max')}")

    judge_stats = stats.get("judge") if isinstance(stats.get("judge"), dict) else {}
    if judge_stats:
        lines.append(f"- judge.conversations_with_scores: {judge_stats.get('conversations_with_scores', 0)}")
        lines.append(f"- judge.avg_score_mean: {judge_stats.get('avg_score_mean')}")

    text_stats = stats.get("text") if isinstance(stats.get("text"), dict) else {}
    if text_stats:
        lines.append(f"- text.avg_words_per_conversation: {text_stats.get('avg_words_per_conversation')}")
        lines.append(
            f"- text.avg_tokens_per_conversation_estimate: {text_stats.get('avg_tokens_per_conversation_estimate')}"
        )

    if dataset_stats:
        lines.extend(
            [
                "",
                "## Full Stats JSON",
                f"See `{stats_file}` for full distributions and aggregates.",
            ]
        )

    if plot_files:
        lines.extend(
            [
                "",
                "## Plots",
            ]
        )
        for rel_path in plot_files:
            title = Path(rel_path).stem.replace("_", " ")
            lines.append(f"### {title.title()}")
            lines.append(f"![{title}]({rel_path})")
            lines.append("")

    (export_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def _collect_minimal_stats(source_dir: Path, source_file: str) -> Dict[str, int]:
    return {
        "records": _count_lines(source_dir / source_file),
    }

def _build_dataset_stats(source_path: Path, output_columns: Dict[str, str]) -> Dict[str, Any]:
    messages_key = (output_columns.get("messages") or "messages").strip() or "messages"
    metadata_key = (output_columns.get("metadata") or "metadata").strip() or "metadata"
    judge_key = (output_columns.get("judge") or "judge").strip() or "judge"

    records = 0
    language_counts: Counter[str] = Counter()
    turn_counts: List[int] = []
    convo_word_counts: List[int] = []
    convo_token_counts: List[int] = []
    message_words_total = 0
    message_tokens_total = 0
    message_count = 0
    judge_avg_scores: List[float] = []
    judge_per_turn_scores: List[float] = []
    judge_conversation_scores: List[float] = []
    judged_conversations = 0

    for row in _iter_jsonl_rows(source_path):
        if not isinstance(row, dict):
            continue
        records += 1
        metadata = row.get(metadata_key) if isinstance(row.get(metadata_key), dict) else {}
        if not metadata and metadata_key != "metadata":
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        language = str(metadata.get("language") or "").strip()
        if language:
            language_counts[language] += 1

        messages = row.get(messages_key)
        if not isinstance(messages, list) and messages_key != "messages":
            messages = row.get("messages")
        if not isinstance(messages, list):
            messages = []

        convo_words = 0
        convo_tokens = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = str(message.get("content") or "")
            words = _count_words(text)
            tokens = _estimate_tokens(text)
            convo_words += words
            convo_tokens += tokens
            message_words_total += words
            message_tokens_total += tokens
            message_count += 1
        if convo_words > 0:
            convo_word_counts.append(convo_words)
        if convo_tokens > 0:
            convo_token_counts.append(convo_tokens)

        judge_payload = row.get(judge_key)
        if not isinstance(judge_payload, dict) and judge_key != "judge":
            judge_payload = row.get("judge")
        if not isinstance(judge_payload, dict):
            judge_payload = {}

        turn_count = _extract_turn_count(metadata, messages, judge_payload)
        if turn_count is not None and turn_count >= 0:
            turn_counts.append(turn_count)

        has_judge = False
        avg_score = _as_float(judge_payload.get("avg_score"))
        conversation_payload = judge_payload.get("conversation")
        conversation_score = (
            _as_float(conversation_payload.get("score")) if isinstance(conversation_payload, dict) else None
        )
        if avg_score is None and conversation_score is not None:
            avg_score = conversation_score
        if avg_score is not None:
            judge_avg_scores.append(avg_score)
            has_judge = True
        if conversation_score is not None:
            judge_conversation_scores.append(conversation_score)
            has_judge = True

        per_turn = judge_payload.get("per_turn")
        if isinstance(per_turn, list):
            for item in per_turn:
                if not isinstance(item, dict):
                    continue
                score = _as_float(item.get("score"))
                if score is None:
                    continue
                judge_per_turn_scores.append(score)
                has_judge = True

        if has_judge:
            judged_conversations += 1

    return {
        "source_file": source_path.name,
        "records": records,
        "languages": {k: language_counts[k] for k in sorted(language_counts)},
        "turn_count": {
            "min": min(turn_counts) if turn_counts else None,
            "max": max(turn_counts) if turn_counts else None,
            "avg": _round_or_none(_mean(turn_counts)),
            "distribution": _count_distribution(turn_counts),
        },
        "judge": {
            "conversations_with_scores": judged_conversations,
            "avg_score_mean": _round_or_none(_mean(judge_avg_scores)),
            "avg_score_distribution": _score_distribution(judge_avg_scores),
            "per_turn_score_mean": _round_or_none(_mean(judge_per_turn_scores)),
            "per_turn_score_distribution": _score_distribution(judge_per_turn_scores),
            "conversation_score_mean": _round_or_none(_mean(judge_conversation_scores)),
            "conversation_score_distribution": _score_distribution(judge_conversation_scores),
        },
        "text": {
            "avg_words_per_conversation": _round_or_none(_mean(convo_word_counts)),
            "avg_tokens_per_conversation_estimate": _round_or_none(_mean(convo_token_counts)),
            "avg_words_per_message": _round_or_none(message_words_total / message_count if message_count else None),
            "avg_tokens_per_message_estimate": _round_or_none(
                message_tokens_total / message_count if message_count else None
            ),
            "word_count_distribution": _histogram_distribution(convo_word_counts, bins=12),
            "token_estimate_distribution": _histogram_distribution(convo_token_counts, bins=12),
        },
    }

def _write_dataset_plots(plots_dir: Path, dataset_stats: Dict[str, Any], relative_prefix: str) -> List[str]:
    plot_specs: List[Tuple[str, str, Dict[str, int]]] = []

    turn_distribution = (
        ((dataset_stats.get("turn_count") or {}).get("distribution") or {})
        if isinstance(dataset_stats.get("turn_count"), dict)
        else {}
    )
    if isinstance(turn_distribution, dict) and turn_distribution:
        plot_specs.append(("turn_count_distribution.svg", "Turn Count Distribution", turn_distribution))

    judge_distribution = (
        ((dataset_stats.get("judge") or {}).get("avg_score_distribution") or {})
        if isinstance(dataset_stats.get("judge"), dict)
        else {}
    )
    if isinstance(judge_distribution, dict) and judge_distribution:
        plot_specs.append(("judge_avg_score_distribution.svg", "Judge Avg Score Distribution", judge_distribution))

    word_distribution = (
        ((dataset_stats.get("text") or {}).get("word_count_distribution") or {})
        if isinstance(dataset_stats.get("text"), dict)
        else {}
    )
    if isinstance(word_distribution, dict) and word_distribution:
        plot_specs.append(("word_count_distribution.svg", "Word Count Distribution", word_distribution))

    token_distribution = (
        ((dataset_stats.get("text") or {}).get("token_estimate_distribution") or {})
        if isinstance(dataset_stats.get("text"), dict)
        else {}
    )
    if isinstance(token_distribution, dict) and token_distribution:
        plot_specs.append(("token_estimate_distribution.svg", "Token Estimate Distribution", token_distribution))

    written_files: List[str] = []
    for file_name, title, distribution in plot_specs:
        path = plots_dir / file_name
        items = [(str(k), int(v)) for k, v in distribution.items() if int(v) >= 0]
        _write_bar_chart_svg(path, title, items)
        prefix = relative_prefix.strip().strip("/\\")
        written_files.append(f"{prefix}/{file_name}" if prefix else file_name)
    return written_files

def _write_bar_chart_svg(path: Path, title: str, items: List[Tuple[str, int]]) -> None:
    safe_title = _xml_escape(title)
    if not items:
        empty_svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='160'>"
            f"<text x='20' y='40' font-size='20' font-family='sans-serif'>{safe_title}</text>"
            "<text x='20' y='90' font-size='14' font-family='sans-serif'>No data</text>"
            "</svg>"
        )
        path.write_text(empty_svg, encoding="utf-8")
        return

    labels = [label for label, _ in items]
    values = [max(0, int(value)) for _, value in items]
    max_value = max(values) if values else 1
    left = min(360, max(140, max(len(label) for label in labels) * 7 + 24))
    top = 64
    row_height = 28
    bar_area_width = 700
    width = left + bar_area_width + 140
    height = top + row_height * len(items) + 36

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<text x='16' y='34' font-size='22' font-weight='bold' font-family='sans-serif'>{safe_title}</text>",
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * row_height
        bar_width = 0 if max_value <= 0 else int((value / max_value) * bar_area_width)
        lines.append(
            f"<text x='16' y='{y + 18}' font-size='13' font-family='sans-serif'>{_xml_escape(label)}</text>"
        )
        lines.append(
            f"<rect x='{left}' y='{y + 4}' width='{bar_width}' height='18' fill='#2f6fed' rx='3' ry='3' />"
        )
        lines.append(
            f"<text x='{left + bar_width + 8}' y='{y + 18}' font-size='12' font-family='sans-serif'>{value}</text>"
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")

def _iter_jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except Exception:
                continue
            if isinstance(parsed, dict):
                yield parsed

def _extract_turn_count(metadata: Dict[str, Any], messages: List[Any], judge_payload: Dict[str, Any]) -> Optional[int]:
    n_turns_raw = metadata.get("n_turns")
    if isinstance(n_turns_raw, (int, float)):
        return int(n_turns_raw)
    if isinstance(n_turns_raw, str):
        try:
            return int(float(n_turns_raw))
        except Exception:
            pass

    per_turn = judge_payload.get("per_turn")
    if isinstance(per_turn, list) and per_turn:
        return len(per_turn)

    user_messages = 0
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "user":
            user_messages += 1
    return user_messages if user_messages > 0 else None

def _count_words(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())

def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, int(math.ceil(len(stripped) / 4.0)))

def _mean(values: List[float] | List[int]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))

def _round_or_none(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)

def _count_distribution(values: List[int]) -> Dict[str, int]:
    if not values:
        return {}
    counter = Counter(int(value) for value in values)
    return {str(key): counter[key] for key in sorted(counter)}

def _score_distribution(scores: List[float]) -> Dict[str, int]:
    if not scores:
        return {}
    counter: Counter[int] = Counter()
    for score in scores:
        bucket = int(round(float(score)))
        bucket = max(0, min(10, bucket))
        counter[bucket] += 1
    return {str(key): counter[key] for key in range(0, 11) if counter[key] > 0}

def _histogram_distribution(values: List[int], bins: int = 12) -> Dict[str, int]:
    if not values:
        return {}
    numeric = [int(value) for value in values]
    minimum = min(numeric)
    maximum = max(numeric)
    if minimum == maximum:
        label = f"{minimum}-{maximum}"
        return {label: len(numeric)}

    bin_count = max(1, int(bins))
    width = max(1, int(math.ceil((maximum - minimum + 1) / bin_count)))
    ordered_labels: List[str] = []
    counter: Counter[str] = Counter()
    for start in range(minimum, maximum + 1, width):
        end = min(start + width - 1, maximum)
        label = f"{start}-{end}"
        ordered_labels.append(label)
    for value in numeric:
        start = minimum + ((value - minimum) // width) * width
        end = min(start + width - 1, maximum)
        label = f"{start}-{end}"
        counter[label] += 1
    return {label: counter[label] for label in ordered_labels if counter[label] > 0}

def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except Exception:
            return None
    return None

def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)

def _infer_dataset_name(repo_id: str | None) -> str:
    if not repo_id:
        return "Synthetic QA Dataset"
    return repo_id.split("/")[-1]
