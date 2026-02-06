from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from dlgforge.io.output import OutputPaths, append_coverage_ledger
from dlgforge.tools.retrieval import get_vector_store
from dlgforge.utils import hash_text, resolve_path

LOGGER = logging.getLogger("dlgforge.pipeline")

_DEFAULT_VARIANT_BY_TARGET_LANGUAGE = {
    "ar": "msa",
    "en": "english",
    "fr": "france",
}
_FINAL_VARIANT_FALLBACK = "english"
_KNOWN_VARIANT_LABELS = {
    "english",
    "france",
    "msa",
    "morocco",
    "egypt",
    "gulf",
    "levantine",
    "saudi",
    "ar",
    "en",
    "fr",
    "arabic",
    "french",
}
_STRUCTURED_SEED_META_KEYS = {
    "version",
    "aliases",
    "defaults",
    "variants",
    "topics",
    "final_fallback",
    "default_variant_by_target_language",
}


def build_question_inputs(
    base_inputs: Dict[str, Any],
    turn_index: int,
    n_turns: int,
    public_history: List[Dict[str, Any]],
    used_topic_ids: set,
    recent_ledger_questions: List[str],
    doc_usage: Dict[str, int],
    doc_chunk_counts: Dict[str, int],
    doc_recent_questions: Dict[str, List[str]],
    avoid_sources: set[str],
    forced_mode: str,
    used_seed_hashes: set[str],
    seed_topic_usage: Dict[str, int],
) -> Dict[str, Any]:
    seed_query = (base_inputs.get("question") or "").strip()
    last_assistant_message = ""
    recent_questions: List[str] = []

    for entry in public_history:
        if entry.get("role") == "assistant":
            last_assistant_message = entry.get("message", "")
        if entry.get("role") == "user":
            recent_questions.append(entry.get("message", ""))

    merged_recent = (recent_ledger_questions + recent_questions)[-10:]

    rng = build_rng(base_inputs, turn_index)
    mode = forced_mode or select_question_mode(
        turn_index=turn_index,
        n_turns=n_turns,
        has_assistant=bool(last_assistant_message),
        rng=rng,
        seed_query=seed_query,
    )

    seed_topic = ""
    seed_question = ""
    if not forced_mode:
        seed_topic, seed_question = maybe_select_seed_question(
            base_inputs=base_inputs,
            turn_index=turn_index,
            rng=rng,
            used_seed_hashes=used_seed_hashes,
            seed_topic_usage=seed_topic_usage,
        )
        if seed_question:
            mode = "seeded"

    topic_snippets = sample_topic_snippets(
        mode=mode,
        seed_query=seed_query,
        last_assistant_message=last_assistant_message,
        used_topic_ids=used_topic_ids,
        doc_usage=doc_usage,
        doc_chunk_counts=doc_chunk_counts,
        doc_recent_questions=doc_recent_questions,
        avoid_sources=avoid_sources,
        rng=rng,
    )

    return {
        "question_mode": mode,
        "last_assistant_message": last_assistant_message or "No prior assistant answer.",
        "topic_snippets": json.dumps(topic_snippets, ensure_ascii=False),
        "recent_user_questions": json.dumps(merged_recent, ensure_ascii=False),
        "seed_topic": seed_topic,
        "seed_question": seed_question,
    }


def build_rng(base_inputs: Dict[str, Any], turn_index: int) -> random.Random:
    seed_value = base_inputs.get("question_seed") or ""
    seed = f"{seed_value}-{turn_index}" if seed_value else f"{datetime.utcnow().isoformat()}-{turn_index}"
    return random.Random(seed)


def select_question_mode(
    turn_index: int,
    n_turns: int,
    has_assistant: bool,
    rng: random.Random,
    seed_query: str,
) -> str:
    _ = n_turns
    if turn_index == 1 and seed_query:
        return "seeded"
    if not has_assistant:
        return "fresh"

    if turn_index <= 2:
        weights = {
            "followup": 0.70,
            "adjacent": 0.20,
            "fresh": 0.08,
            "off_topic": 0.02,
        }
    else:
        weights = {
            "followup": 0.45,
            "adjacent": 0.30,
            "fresh": 0.23,
            "off_topic": 0.02,
        }

    roll = rng.random()
    cumulative = 0.0
    for mode, weight in weights.items():
        cumulative += weight
        if roll <= cumulative:
            return mode
    return "fresh"


def sample_topic_snippets(
    mode: str,
    seed_query: str,
    last_assistant_message: str,
    used_topic_ids: set,
    doc_usage: Dict[str, int],
    doc_chunk_counts: Dict[str, int],
    doc_recent_questions: Dict[str, List[str]],
    avoid_sources: set[str],
    rng: random.Random,
) -> List[Dict[str, str]]:
    store = get_vector_store()
    snippets: List[Dict[str, str]] = []

    exclude_ids = {topic_id for topic_id in used_topic_ids if topic_id}
    all_sources = store.list_sources()
    preferred_sources = select_doc_pool(doc_usage, doc_chunk_counts, all_sources, rng)

    if avoid_sources:
        preferred_sources = set(preferred_sources) - set(avoid_sources)
        if not preferred_sources:
            preferred_sources = set(all_sources) - set(avoid_sources)

    query = ""
    if mode == "seeded":
        query = seed_query
    elif mode in {"followup", "adjacent"}:
        query = last_assistant_message or seed_query

    results: List[tuple] = []
    if query:
        results = store.similarity_search_with_ids(query, k=12)
        results = [item for item in results if item[2] not in exclude_ids]
        results = filter_results_by_sources(
            results=results,
            preferred_sources=preferred_sources,
            allow_fallback=(mode in {"followup", "adjacent", "seeded"}),
        )

    if not results:
        if preferred_sources:
            results = store.sample_by_sources(preferred_sources, n=8, exclude_ids=exclude_ids, rng=rng)
        if not results:
            results = store.random_samples(n=8, exclude_ids=exclude_ids, rng=rng)

    for passage, metadata, chunk_id in results[:6]:
        source_path = metadata.get("source", "unknown")
        snippets.append(
            {
                "topic_id": chunk_id,
                "source_path": source_path,
                "source_descriptor": format_source_descriptor(metadata),
                "text": passage,
                "recent_questions": doc_recent_questions.get(source_path, [])[-5:],
            }
        )

    return snippets


def format_source_descriptor(metadata: Dict[str, Any]) -> str:
    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index")
    base = source.split("/")[-1]
    if chunk_index is None:
        return base
    return f"{base}#chunk{chunk_index}"


def build_doc_usage(ledger_entries: List[Dict[str, Any]]) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for entry in ledger_entries:
        source = entry.get("source_path") or entry.get("source")
        if not source:
            continue
        usage[source] = usage.get(source, 0) + 1
    return usage


def build_doc_chunk_counts() -> Dict[str, int]:
    return get_vector_store().source_chunk_counts()


def build_doc_question_hashes(ledger_entries: List[Dict[str, Any]]) -> Dict[str, set[str]]:
    hashes: Dict[str, set[str]] = {}
    for entry in ledger_entries:
        source = entry.get("source_path") or entry.get("source")
        question_hash = entry.get("question_hash")
        if not source or not question_hash:
            continue
        hashes.setdefault(source, set()).add(question_hash)
    return hashes


def build_doc_recent_questions(
    ledger_entries: List[Dict[str, Any]],
    max_per_doc: int = 8,
) -> Dict[str, List[str]]:
    history: Dict[str, List[str]] = {}
    for entry in ledger_entries:
        source = entry.get("source_path") or entry.get("source")
        question = entry.get("question")
        if not source or not question:
            continue
        history.setdefault(source, []).append(question)

    if max_per_doc > 0:
        for source, items in history.items():
            history[source] = items[-max_per_doc:]

    return history


def select_doc_pool(
    doc_usage: Dict[str, int],
    doc_chunk_counts: Dict[str, int],
    all_sources: List[str],
    rng: random.Random,
) -> set[str]:
    mode = str((os.getenv("DOC_COVERAGE_MODE") or "balanced")).strip().lower()
    if mode == "off" or not all_sources:
        return set(all_sources)

    epsilon = clamp_float(os.getenv("DOC_COVERAGE_EPSILON", "0.15"), default=0.15)
    if rng.random() < epsilon:
        return set(all_sources)

    unused = [source for source in all_sources if doc_usage.get(source, 0) == 0]
    if unused:
        return set(unused)

    fraction = clamp_float(os.getenv("DOC_COVERAGE_FRACTION", "0.2"), default=0.2)
    fraction = max(min(fraction, 1.0), 0.05)
    sorted_sources = sorted(all_sources, key=lambda src: coverage_ratio(src, doc_usage, doc_chunk_counts))
    k = max(1, int(len(sorted_sources) * fraction))
    return set(sorted_sources[:k])


def coverage_ratio(source: str, doc_usage: Dict[str, int], doc_chunk_counts: Dict[str, int]) -> float:
    usage = doc_usage.get(source, 0)
    chunks = doc_chunk_counts.get(source, 1)
    return usage / max(chunks, 1)


def filter_results_by_sources(
    results: List[tuple],
    preferred_sources: set[str],
    allow_fallback: bool,
) -> List[tuple]:
    if not preferred_sources:
        return results
    filtered = [item for item in results if item[1].get("source") in preferred_sources]
    if filtered:
        return filtered
    return results if allow_fallback else []


def clamp_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def update_coverage_ledger(
    paths: OutputPaths,
    qa_output: Dict[str, Any],
    question_inputs: Dict[str, Any],
    used_topic_ids: set,
    used_question_hashes: set,
) -> None:
    user_message = (qa_output.get("user_message") or "").strip()
    if not user_message:
        return

    topic_id = (qa_output.get("coverage_target") or "").strip()
    question_hash = hash_text(user_message)
    source_path = lookup_source_path(question_inputs, topic_id)
    seed_topic = question_inputs.get("seed_topic") or ""
    seed_question = question_inputs.get("seed_question") or ""
    seed_question_hash = hash_text(seed_question) if seed_question else ""

    if topic_id:
        used_topic_ids.add(topic_id)
    used_question_hashes.add(question_hash)

    append_coverage_ledger(
        paths,
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "topic_id": topic_id,
            "source_path": source_path,
            "question": user_message,
            "question_hash": question_hash,
            "question_mode": question_inputs.get("question_mode"),
            "seed_topic": seed_topic,
            "seed_question": seed_question,
            "seed_question_hash": seed_question_hash,
        },
    )


def is_duplicate_question(
    qa_output: Dict[str, Any],
    question_inputs: Dict[str, Any],
    doc_question_hashes: Dict[str, set[str]],
) -> tuple[bool, str]:
    question = (qa_output.get("user_message") or "").strip()
    if not question:
        return False, ""

    topic_id = (qa_output.get("coverage_target") or "").strip()
    source_path = lookup_source_path(question_inputs, topic_id)
    if not source_path:
        return False, ""

    question_hash = hash_text(question)
    return question_hash in doc_question_hashes.get(source_path, set()), source_path


def update_doc_question_memory(
    qa_output: Dict[str, Any],
    question_inputs: Dict[str, Any],
    doc_question_hashes: Dict[str, set[str]],
    doc_recent_questions: Dict[str, List[str]],
) -> None:
    question = (qa_output.get("user_message") or "").strip()
    if not question:
        return

    topic_id = (qa_output.get("coverage_target") or "").strip()
    source_path = lookup_source_path(question_inputs, topic_id)
    if not source_path:
        return

    question_hash = hash_text(question)
    doc_question_hashes.setdefault(source_path, set()).add(question_hash)
    doc_recent_questions.setdefault(source_path, []).append(question)


def lookup_source_path(question_inputs: Dict[str, Any], topic_id: str) -> str:
    if not topic_id:
        return ""

    snippets_raw = question_inputs.get("topic_snippets")
    if not snippets_raw:
        return ""

    try:
        snippets = json.loads(snippets_raw)
    except json.JSONDecodeError:
        return ""

    for snippet in snippets:
        if snippet.get("topic_id") == topic_id:
            return snippet.get("source_path") or snippet.get("source_descriptor") or ""
    return ""


def maybe_select_seed_question(
    base_inputs: Dict[str, Any],
    turn_index: int,
    rng: random.Random,
    used_seed_hashes: set[str],
    seed_topic_usage: Dict[str, int],
) -> tuple[str, str]:
    if turn_index != 1:
        return "", ""
    if (base_inputs.get("question") or "").strip():
        return "", ""
    if not base_inputs.get("seed_topics_enabled", True):
        return "", ""

    probability = float(base_inputs.get("seed_topics_probability") or 0.0)
    if probability <= 0 or rng.random() > probability:
        return "", ""

    seed_topics_path = base_inputs.get("seed_topics_path") or ""
    seed_topics = load_seed_topics(
        path=seed_topics_path,
        project_root=Path(base_inputs.get("project_root", ".")),
        config_dir=Path(base_inputs.get("config_dir", ".")),
        target_language=str(base_inputs.get("target_language", "") or ""),
        seed_topics_variant=str(base_inputs.get("seed_topics_variant", "") or ""),
    )
    if not seed_topics:
        return "", ""

    return select_seed_candidate(seed_topics, used_seed_hashes, seed_topic_usage, rng)


def load_seed_topics(
    path: str,
    project_root: Path,
    config_dir: Path,
    target_language: str = "",
    seed_topics_variant: str = "",
) -> Dict[str, List[str]]:
    if not path:
        return {}

    resolved = resolve_path(path, project_root=project_root, config_dir=config_dir)
    if not resolved or not resolved.exists():
        return {}

    if resolved.is_dir():
        for name in ("topics.yaml", "topics.yml"):
            candidate = resolved / name
            if candidate.exists():
                return _load_seed_topics_from_structured_file(
                    file_path=candidate,
                    target_language=target_language,
                    seed_topics_variant=seed_topics_variant,
                )
        return _load_seed_topics_from_directory(
            seed_dir=resolved,
            target_language=target_language,
            seed_topics_variant=seed_topics_variant,
        )

    return _load_seed_topics_from_any_file(
        file_path=resolved,
        target_language=target_language,
        seed_topics_variant=seed_topics_variant,
    )


def _load_seed_topics_from_any_file(
    file_path: Path,
    target_language: str,
    seed_topics_variant: str,
) -> Dict[str, List[str]]:
    data = _load_mapping_file(file_path)
    if not data:
        return {}

    if _is_legacy_nested_seed_topics(data):
        LOGGER.warning(
            "[seeds] Legacy nested seed topics file detected at %s. "
            "Please migrate to YAML layout (data/seeds/topics.yaml).",
            file_path,
        )
        legacy_variant = _resolve_variant_for_legacy_file(
            target_language=target_language,
            seed_topics_variant=seed_topics_variant,
        )
        return _extract_seed_topics_from_legacy_nested(data, variant=legacy_variant)

    structured = _parse_structured_seed_payload(data)
    if structured:
        return _select_seed_topics_from_structured(
            structured=structured,
            target_language=target_language,
            seed_topics_variant=seed_topics_variant,
        )

    return _clean_seed_topics_map(data)


def _load_seed_topics_from_directory(
    seed_dir: Path,
    target_language: str,
    seed_topics_variant: str,
) -> Dict[str, List[str]]:
    index_data = _load_seed_index(seed_dir / "index.yaml")
    if not index_data:
        index_data = _load_seed_index(seed_dir / "index.yml")
    if not index_data:
        index_data = _load_seed_index(seed_dir / "index.json")
    variant = _resolve_variant_for_directory(
        seed_dir=seed_dir,
        target_language=target_language,
        seed_topics_variant=seed_topics_variant,
        index_data=index_data,
    )
    if not variant:
        return {}

    variant_path = _find_variant_file(seed_dir, variant)
    if not variant_path:
        return {}

    data = _load_mapping_file(variant_path)
    return _clean_seed_topics_map(data) if data else {}


def _load_seed_topics_from_structured_file(
    file_path: Path,
    target_language: str,
    seed_topics_variant: str,
) -> Dict[str, List[str]]:
    data = _load_mapping_file(file_path)
    if not data:
        return {}
    structured = _parse_structured_seed_payload(data)
    if not structured:
        return {}
    return _select_seed_topics_from_structured(
        structured=structured,
        target_language=target_language,
        seed_topics_variant=seed_topics_variant,
    )


def _clean_seed_topics_map(data: Dict[str, Any]) -> Dict[str, List[str]]:
    seed_topics: Dict[str, List[str]] = {}
    for topic, questions in data.items():
        if not isinstance(questions, list):
            continue
        cleaned = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        if cleaned:
            seed_topics[str(topic)] = cleaned
    return seed_topics


def _load_seed_index(index_path: Path) -> Dict[str, Any]:
    return _load_mapping_file(index_path)


def _resolve_variant_for_directory(
    seed_dir: Path,
    target_language: str,
    seed_topics_variant: str,
    index_data: Dict[str, Any],
) -> str:
    explicit_variant = seed_topics_variant.strip().lower()
    target = target_language.strip().lower()

    alias_map = _as_str_dict(index_data.get("aliases"))
    by_target = _as_str_dict(index_data.get("default_variant_by_target_language"))

    candidates: List[str] = []
    if explicit_variant:
        candidates.append(explicit_variant)
    if target:
        mapped = by_target.get(target) or alias_map.get(target) or ""
        if mapped:
            candidates.append(mapped.strip().lower())
    hard_default = _DEFAULT_VARIANT_BY_TARGET_LANGUAGE.get(target)
    if hard_default:
        candidates.append(hard_default)
    candidates.append(_FINAL_VARIANT_FALLBACK)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = _find_variant_file(seed_dir, candidate)
        if path:
            return candidate
    return ""


def _resolve_variant_for_legacy_file(target_language: str, seed_topics_variant: str) -> str:
    explicit_variant = seed_topics_variant.strip().lower()
    if explicit_variant:
        return explicit_variant
    target = target_language.strip().lower()
    return _DEFAULT_VARIANT_BY_TARGET_LANGUAGE.get(target, _FINAL_VARIANT_FALLBACK)


def _is_legacy_nested_seed_topics(data: Dict[str, Any]) -> bool:
    if not data:
        return False
    if any(str(key).strip().lower() in _STRUCTURED_SEED_META_KEYS for key in data.keys()):
        return False
    if not all(isinstance(value, dict) for value in data.values()):
        return False

    keys = [str(key).strip().lower() for key in data.keys() if str(key).strip()]
    if not keys:
        return False
    known_variant_like = sum(1 for key in keys if key in _KNOWN_VARIANT_LABELS)
    if known_variant_like / len(keys) >= 0.5:
        return False

    for value in data.values():
        if not any(isinstance(item, list) for item in value.values()):
            return False
    return True


def _extract_seed_topics_from_legacy_nested(data: Dict[str, Any], variant: str) -> Dict[str, List[str]]:
    # Keep backward compatibility with old schema:
    # {topic: {language_variant: [questions]}}
    seed_topics: Dict[str, List[str]] = {}
    fallback_variant = _FINAL_VARIANT_FALLBACK
    for topic, per_lang in data.items():
        if not isinstance(per_lang, dict):
            continue

        questions = per_lang.get(variant)
        if not isinstance(questions, list):
            questions = per_lang.get(fallback_variant)
        if not isinstance(questions, list):
            # Last fallback: first available list in this topic block.
            for _, value in per_lang.items():
                if isinstance(value, list):
                    questions = value
                    break

        if not isinstance(questions, list):
            continue

        cleaned = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        if cleaned:
            seed_topics[str(topic)] = cleaned
    return seed_topics


def _parse_structured_seed_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    variants_raw = data.get("variants")
    if not isinstance(variants_raw, dict):
        variants_raw = data.get("topics")

    if not isinstance(variants_raw, dict) and _looks_like_variant_map(data):
        variants_raw = {
            key: value for key, value in data.items() if str(key).strip().lower() not in _STRUCTURED_SEED_META_KEYS
        }

    if not isinstance(variants_raw, dict):
        return {}

    variants_clean: Dict[str, Dict[str, List[str]]] = {}
    for variant, topics in variants_raw.items():
        if not isinstance(topics, dict):
            continue
        cleaned_topics = _clean_seed_topics_map(topics)
        if cleaned_topics:
            variants_clean[str(variant).strip().lower()] = cleaned_topics

    if not variants_clean:
        return {}

    defaults_raw = data.get("defaults")
    defaults_map = {}
    final_fallback = _FINAL_VARIANT_FALLBACK
    if isinstance(defaults_raw, dict):
        defaults_map = _as_str_dict(defaults_raw.get("by_target_language"))
        fallback_raw = defaults_raw.get("final_fallback")
        if isinstance(fallback_raw, str) and fallback_raw.strip():
            final_fallback = fallback_raw.strip().lower()

    if not defaults_map:
        defaults_map = _as_str_dict(data.get("default_variant_by_target_language"))

    top_level_fallback = data.get("final_fallback")
    if isinstance(top_level_fallback, str) and top_level_fallback.strip():
        final_fallback = top_level_fallback.strip().lower()

    return {
        "variants": variants_clean,
        "aliases": _as_str_dict(data.get("aliases")),
        "default_variant_by_target_language": defaults_map,
        "final_fallback": final_fallback,
    }


def _select_seed_topics_from_structured(
    structured: Dict[str, Any],
    target_language: str,
    seed_topics_variant: str,
) -> Dict[str, List[str]]:
    variants = structured.get("variants") or {}
    if not isinstance(variants, dict):
        return {}

    variant = _resolve_variant_from_structured(
        structured=structured,
        target_language=target_language,
        seed_topics_variant=seed_topics_variant,
    )
    if not variant:
        return {}

    topics = variants.get(variant)
    if not isinstance(topics, dict):
        return {}
    return topics


def _resolve_variant_from_structured(
    structured: Dict[str, Any],
    target_language: str,
    seed_topics_variant: str,
) -> str:
    aliases = structured.get("aliases") or {}
    by_target = structured.get("default_variant_by_target_language") or {}
    final_fallback = str(structured.get("final_fallback") or _FINAL_VARIANT_FALLBACK).strip().lower()
    variants = structured.get("variants") or {}
    if not isinstance(variants, dict):
        return ""

    explicit = seed_topics_variant.strip().lower()
    target = target_language.strip().lower()

    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
        mapped_explicit = aliases.get(explicit) if isinstance(aliases, dict) else ""
        if mapped_explicit:
            candidates.append(str(mapped_explicit).strip().lower())

    if target:
        mapped = ""
        if isinstance(by_target, dict):
            mapped = by_target.get(target) or ""
        if not mapped and isinstance(aliases, dict):
            mapped = aliases.get(target) or ""
        if mapped:
            candidates.append(str(mapped).strip().lower())

    hard_default = _DEFAULT_VARIANT_BY_TARGET_LANGUAGE.get(target)
    if hard_default:
        candidates.append(hard_default)
    if final_fallback:
        candidates.append(final_fallback)
    candidates.append(_FINAL_VARIANT_FALLBACK)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in variants:
            return candidate
    return ""


def _looks_like_variant_map(data: Dict[str, Any]) -> bool:
    candidate_items = []
    for key, value in data.items():
        if str(key).strip().lower() in _STRUCTURED_SEED_META_KEYS:
            continue
        candidate_items.append((key, value))
    if not candidate_items:
        return False
    for _, value in candidate_items:
        if not isinstance(value, dict):
            return False
    return True


def _load_mapping_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _find_variant_file(seed_dir: Path, variant: str) -> Path | None:
    for suffix in (".yaml", ".yml", ".json"):
        candidate = seed_dir / f"{variant}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _as_str_dict(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in raw.items():
        key_text = str(key).strip().lower()
        value_text = str(value).strip().lower()
        if key_text and value_text:
            normalized[key_text] = value_text
    return normalized


def select_seed_candidate(
    seed_topics: Dict[str, List[str]],
    used_seed_hashes: set[str],
    seed_topic_usage: Dict[str, int],
    rng: random.Random,
) -> tuple[str, str]:
    candidates_by_topic: Dict[str, List[str]] = {}
    for topic, questions in seed_topics.items():
        available = [q for q in questions if hash_text(q) not in used_seed_hashes]
        if available:
            candidates_by_topic[topic] = available

    if not candidates_by_topic:
        return "", ""

    min_usage = min(seed_topic_usage.get(topic, 0) for topic in candidates_by_topic.keys())
    eligible_topics = [topic for topic in candidates_by_topic if seed_topic_usage.get(topic, 0) == min_usage]
    chosen_topic = rng.choice(eligible_topics)
    chosen_question = rng.choice(candidates_by_topic[chosen_topic])
    return chosen_topic, chosen_question


def build_used_seed_hashes(ledger_entries: List[Dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for entry in ledger_entries:
        seed_hash = entry.get("seed_question_hash")
        if seed_hash:
            hashes.add(seed_hash)
    return hashes


def build_seed_topic_usage(ledger_entries: List[Dict[str, Any]]) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for entry in ledger_entries:
        topic = entry.get("seed_topic")
        if topic:
            usage[topic] = usage.get(topic, 0) + 1
    return usage


def update_seed_memory(
    qa_output: Dict[str, Any],
    question_inputs: Dict[str, Any],
    used_seed_hashes: set[str],
    seed_topic_usage: Dict[str, int],
) -> None:
    _ = qa_output
    seed_question = question_inputs.get("seed_question") or ""
    seed_topic = question_inputs.get("seed_topic") or ""
    if not seed_question:
        return

    seed_hash = hash_text(seed_question)
    used_seed_hashes.add(seed_hash)
    if seed_topic:
        seed_topic_usage[seed_topic] = seed_topic_usage.get(seed_topic, 0) + 1
