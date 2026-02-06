from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from dlgforge.config import load_config
from dlgforge.utils import resolve_path, setup_logging


LOGGER = logging.getLogger("dlgforge.pipeline")

_DEFAULT_VARIANT_BY_TARGET_LANGUAGE = {
    "ar": "msa",
    "en": "english",
    "fr": "france",
}


def run_seeds_migrate(
    config_path: str,
    source_file: str = "",
    dest_file: str = "",
    overwrite: bool = False,
) -> Dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists() or not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    project_root = config_file.parent
    setup_logging(project_root / "logs")

    cfg, _, resolved_project_root = load_config(config_file)

    run_seed_path = str((cfg.get("run", {}) or {}).get("seed_topics_path", "") or "")

    if source_file.strip():
        source_path = _resolve_source_path(source_file, resolved_project_root)
    else:
        fallback_legacy = (resolved_project_root / "seed_topics.json").resolve()
        configured = _resolve_source_path(run_seed_path, resolved_project_root) if run_seed_path else None
        if configured and configured.exists() and configured.is_file() and configured.suffix.lower() == ".json":
            source_path = configured
        elif fallback_legacy.exists():
            source_path = fallback_legacy
        elif configured and configured.exists() and configured.is_file():
            source_path = configured
        else:
            source_path = fallback_legacy

    if not source_path.exists():
        raise FileNotFoundError(f"Seed topics source file not found: {source_path}")
    if source_path.is_dir():
        raise ValueError(
            f"Expected seed topics FILE for migration, got directory: {source_path}. "
            "Use --source-file to point to a legacy seed topics file."
        )

    if dest_file.strip():
        destination = Path(dest_file).expanduser()
        if not destination.is_absolute():
            destination = (resolved_project_root / destination).resolve()
    else:
        configured_dest = _resolve_source_path(run_seed_path, resolved_project_root) if run_seed_path else None
        if configured_dest and configured_dest.suffix.lower() in {".yaml", ".yml"}:
            destination = configured_dest
        else:
            destination = (resolved_project_root / "data" / "seeds" / "topics.yaml").resolve()

    result = migrate_seed_topics_file(source_path, destination_file=destination, overwrite=overwrite)
    LOGGER.info(
        "[seeds] Migration completed: source=%s dest_file=%s variants=%s topics=%s",
        result["source_file"],
        result["dest_file"],
        len(result["variants"]),
        result["topic_count"],
    )
    return result


def migrate_seed_topics_file(source_file: Path, destination_file: Path, overwrite: bool = False) -> Dict[str, Any]:
    raw = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Seed topics file must be a YAML/JSON object.")

    by_variant = _split_by_variant(raw)
    if not by_variant:
        raise ValueError("No valid seed topics found in source file.")

    payload = {
        "version": 1,
        "defaults": {
            "by_target_language": _DEFAULT_VARIANT_BY_TARGET_LANGUAGE,
            "final_fallback": "english",
        },
        "aliases": _default_aliases(by_variant),
        "variants": {key: by_variant[key] for key in sorted(by_variant.keys())},
    }

    destination_file.parent.mkdir(parents=True, exist_ok=True)
    if destination_file.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination_file}")
    destination_file.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    unique_topics = {topic for topics_map in by_variant.values() for topic in topics_map.keys()}
    return {
        "source_file": str(source_file),
        "dest_file": str(destination_file),
        "variants": sorted(by_variant.keys()),
        "topic_count": len(unique_topics),
        "files_written": [str(destination_file)],
    }


def _split_by_variant(data: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    structured = _try_extract_structured_variants(data)
    if structured:
        return structured

    by_variant: Dict[str, Dict[str, List[str]]] = {}

    for topic, payload in data.items():
        topic_name = str(topic).strip()
        if not topic_name:
            continue

        if isinstance(payload, list):
            cleaned = _clean_questions(payload)
            if cleaned:
                by_variant.setdefault("english", {})[topic_name] = cleaned
            continue

        if not isinstance(payload, dict):
            continue

        for variant_key, questions in payload.items():
            variant = str(variant_key).strip().lower()
            if not variant:
                continue
            cleaned = _clean_questions(questions if isinstance(questions, list) else [])
            if cleaned:
                by_variant.setdefault(variant, {})[topic_name] = cleaned

    return by_variant


def _try_extract_structured_variants(data: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    variants_raw = data.get("variants")
    if not isinstance(variants_raw, dict):
        variants_raw = data.get("topics")
    if not isinstance(variants_raw, dict):
        return {}

    by_variant: Dict[str, Dict[str, List[str]]] = {}
    for variant_key, topics_map in variants_raw.items():
        variant = str(variant_key).strip().lower()
        if not variant or not isinstance(topics_map, dict):
            continue
        for topic, questions in topics_map.items():
            topic_name = str(topic).strip()
            if not topic_name:
                continue
            cleaned = _clean_questions(questions if isinstance(questions, list) else [])
            if cleaned:
                by_variant.setdefault(variant, {})[topic_name] = cleaned
    return by_variant


def _clean_questions(values: List[Any]) -> List[str]:
    return [str(value).strip() for value in values if isinstance(value, str) and str(value).strip()]


def _default_aliases(by_variant: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for variant in by_variant.keys():
        aliases[variant] = variant

    aliases.update(
        {
            "arabic": "msa",
            "french": "france",
            "ar": "msa",
            "en": "english",
            "fr": "france",
        }
    )
    return aliases


def _resolve_source_path(raw: str, project_root: Path) -> Path:
    if not raw.strip():
        return project_root / "seed_topics.json"

    raw_path = Path(raw).expanduser()
    if raw_path.is_absolute():
        return raw_path

    resolved = resolve_path(raw, project_root=project_root, config_dir=project_root)
    if resolved:
        return resolved
    return (project_root / raw_path).resolve()
