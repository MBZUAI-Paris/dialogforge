"""Tests for canonical v3 config schema and legacy migration paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from dlgforge.config.loader import (
    load_config,
    resolve_question,
    resolve_retrieval_default_k,
    resolve_seed_topics_enabled,
    resolve_seed_topics_path,
    resolve_turn_count_distribution,
    resolve_turn_range,
)


def _write_config(tmp_path: Path, payload: Dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_supports_canonical_turns_and_data_schema(tmp_path: Path) -> None:
    payload = {
        "run": {
            "turns": {
                "mode": "range",
                "exact": None,
                "min": 2,
                "max": 5,
                "distribution": "poisson",
                "mean": None,
            },
            "data": {
                "seeding": {
                    "question": "seed-question",
                    "topics": {
                        "path": "data/seeds/topics.yaml",
                        "enabled": True,
                        "variant": "v1",
                        "probability": 0.35,
                    },
                },
            },
        }
    }
    cfg, _, _ = load_config(_write_config(tmp_path, payload))

    assert resolve_turn_range(cfg) == (2, 5)
    assert resolve_turn_count_distribution(cfg) == "poisson"
    assert resolve_question(cfg) == "seed-question"
    assert resolve_seed_topics_path(cfg) == "data/seeds/topics.yaml"
    assert resolve_seed_topics_enabled(cfg) is True


def test_load_config_migrates_legacy_turn_and_seed_keys(tmp_path: Path) -> None:
    payload = {
        "run": {
            "n_turns": 4,
            "seed_question": "legacy-seed",
            "seed_topics_path": "legacy/topics.yaml",
            "seed_topics_enabled": False,
        }
    }
    cfg, _, _ = load_config(_write_config(tmp_path, payload))

    assert cfg["run"]["turns"]["mode"] == "exact"
    assert cfg["run"]["turns"]["exact"] == 4
    assert cfg["run"]["data"]["seeding"]["question"] == "legacy-seed"
    assert cfg["run"]["data"]["seeding"]["topics"]["path"] == "legacy/topics.yaml"
    assert cfg["run"]["data"]["seeding"]["topics"]["enabled"] is False


def test_load_config_rejects_ambiguous_legacy_turns_config(tmp_path: Path) -> None:
    payload = {
        "run": {
            "n_turns": 3,
            "min_turns": 2,
            "max_turns": 4,
        }
    }

    with pytest.raises(ValueError, match="run.n_turns"):
        load_config(_write_config(tmp_path, payload))


def test_load_config_migrates_legacy_retrieval_and_models_keys(tmp_path: Path) -> None:
    payload = {
        "retrieval": {
            "default_k": 7,
            "chunk_size": 900,
            "overlap": 120,
            "persist_dir": "legacy_index",
            "rebuild_index": True,
        },
        "models": {
            "embedding_model": "legacy/embedding",
            "fallback_embedding_model": "legacy/fallback",
            "use_reranker": True,
            "reranker_model": "legacy/reranker",
            "reranker_candidates": 9,
        },
    }
    cfg, _, _ = load_config(_write_config(tmp_path, payload))

    assert resolve_retrieval_default_k(cfg) == 7
    retrieval_cfg = cfg["tools"]["retrieval"]
    assert retrieval_cfg["chunking"]["chunk_size"] == 900
    assert retrieval_cfg["chunking"]["chunk_overlap"] == 120
    assert retrieval_cfg["index"]["persist_dir"] == "legacy_index"
    assert retrieval_cfg["index"]["rebuild"] is True
    assert retrieval_cfg["embeddings"]["model"] == "legacy/embedding"
    assert retrieval_cfg["embeddings"]["fallback_model"] == "legacy/fallback"
    assert retrieval_cfg["reranker"]["enabled"] is True
    assert retrieval_cfg["reranker"]["model"] == "legacy/reranker"
    assert retrieval_cfg["reranker"]["candidates"] == 9


def test_load_config_supports_canonical_tools_web_search_and_retrieval(tmp_path: Path) -> None:
    payload = {
        "tools": {
            "web_search": {
                "enabled": True,
                "serper_num_results": 8,
                "serper_timeout": 45,
            },
            "retrieval": {
                "top_k": 6,
            },
        }
    }
    cfg, _, _ = load_config(_write_config(tmp_path, payload))

    assert cfg["tools"]["web_search"]["enabled"] is True
    assert cfg["tools"]["web_search"]["serper_num_results"] == 8
    assert cfg["tools"]["web_search"]["serper_timeout"] == 45
    assert cfg["tools"]["retrieval"]["top_k"] == 6
    assert cfg["retrieval"]["top_k"] == 6


def test_load_config_migrates_legacy_flat_web_search_keys(tmp_path: Path) -> None:
    payload = {
        "tools": {
            "web_search_enabled": True,
            "serper_num_results": 9,
            "serper_timeout": 33,
        }
    }
    cfg, _, _ = load_config(_write_config(tmp_path, payload))

    assert cfg["tools"]["web_search"]["enabled"] is True
    assert cfg["tools"]["web_search"]["serper_num_results"] == 9
    assert cfg["tools"]["web_search"]["serper_timeout"] == 33
