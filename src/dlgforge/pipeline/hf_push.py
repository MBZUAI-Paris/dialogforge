from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dlgforge.config import load_config, resolve_output_dir
from dlgforge.io import OutputPaths
from dlgforge.utils import load_dotenv_files, setup_logging


LOGGER = logging.getLogger("dlgforge.pipeline")


@dataclass
class HFPushSettings:
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


@dataclass
class HFPushOptions:
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
    settings = resolve_hf_push_settings(cfg, output_paths.project_root)
    if not settings.enabled or not settings.auto_push_on_run:
        return

    source_dir = output_paths.output_dir
    source_path = source_dir / settings.source_file
    if not source_path.exists() or _count_lines(source_path) == 0:
        LOGGER.warning(
            "[hf-push] Auto-push skipped: "
            f"{settings.source_file} is missing or empty in {source_dir}."
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
    saving_cfg = cfg.get("saving", {}) or {}
    hf_cfg = saving_cfg.get("hf_push", {}) or {}

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
    )


def prepare_export(
    source_dir: Path,
    export_dir: Path,
    source_file: str,
    include_run_state: bool,
    repo_id: str | None,
) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / source_file
    if not source_path.exists():
        raise FileNotFoundError(f"{source_file} not found in {source_dir}")
    shutil.copy2(source_path, export_dir / source_file)

    if include_run_state:
        run_state = source_dir / "run_state"
        if run_state.exists() and run_state.is_dir():
            shutil.copytree(run_state, export_dir / "run_state")

    _write_dataset_card(
        export_dir=export_dir,
        source_dir=source_dir,
        source_file=source_file,
        repo_id=repo_id,
    )
    return export_dir


def push_to_hub(
    export_dir: Path,
    repo_id: str,
    repo_type: str,
    token: Optional[str],
    commit_message: str,
    private: bool = True,
    clean_remote: bool = False,
) -> None:
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


def _write_dataset_card(export_dir: Path, source_dir: Path, source_file: str, repo_id: str | None) -> None:
    stats = _collect_stats(source_dir, source_file)
    dataset_name = _infer_dataset_name(repo_id)
    lines = [
        "---",
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
        "",
        "## Stats",
    ]
    lines.extend([f"- {k}: {v}" for k, v in stats.items()])
    (export_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _collect_stats(source_dir: Path, source_file: str) -> Dict[str, int]:
    return {
        source_file.replace(".", "_") + "_records": _count_lines(source_dir / source_file),
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _infer_dataset_name(repo_id: str | None) -> str:
    if not repo_id:
        return "Synthetic QA Dataset"
    return repo_id.split("/")[-1]
