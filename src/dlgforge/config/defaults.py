"""Default configuration schema for DialogForge.

"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "batch_size": 1,
        "total_samples": 0,
        "target_languages": ["en"],
        "run_id": "",
        "resume_run_id": "",
        "turns": {
            "mode": "range",
            "exact": None,
            "min": 2,
            "max": 2,
            "distribution": "poisson",
            "mean": None,
        },
        "data": {
            "seeding": {
                "question": "",
                "topics": {
                    "path": "data/seeds/topics.yaml",
                    "enabled": True,
                    "variant": "",
                    "probability": 0.35,
                },
            },
        },
        "distributed": {
            "enabled": False,
            "backend": "ray",
            "spawn": {
                "coordinator": True,
                "workers": True,
            },
            "ray": {
                "address": "auto",
                "auto_start_local": True,
                "namespace": "dlgforge",
                "actor": {
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "coordinator_num_cpus": 1.0,
                    "replicas_qa": 1,
                    "replicas_complete": 1,
                },
            },
        },
    },
    "store": {
        "backend": "postgres",
        "postgres": {
            "dsn": "",
            "min_pool_size": 5,
            "max_pool_size": 30,
            "statement_timeout_ms": 30000,
        },
    },
    "llm": {
        "mode": "api",
        "routing": {
            "strategy": "weighted_least_inflight",
            "endpoints": [],
        },
        "vllm": {
            "model": "",
            "served_model_name": "",
            "replicas": 1,
            "num_gpus_per_replica": 1,
            "host": "0.0.0.0",
            "advertise_host": "127.0.0.1",
            "port_start": 18000,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 256,
            "health_timeout_s": 180,
            "auto_stop_on_exit": True,
        },
        "agents": {
            "user": {
                "provider": "openai",
                "model": "",
                "base_url": "",
                "temperature": None,
                "max_tokens": None,
                "top_p": None,
                "timeout": None,
                "max_retries": None,
                "extra": {},
            },
            "assistant": {
                "provider": "",
                "model": "",
                "base_url": "",
                "temperature": None,
                "max_tokens": None,
                "top_p": None,
                "timeout": None,
                "max_retries": None,
                "extra": {},
            },
            "judge": {
                "provider": "",
                "model": "",
                "base_url": "",
                "temperature": None,
                "max_tokens": None,
                "top_p": None,
                "timeout": None,
                "max_retries": None,
                "extra": {},
            },
        },
    },
    "coverage": {
        "doc_coverage_mode": "balanced",
        "doc_coverage_epsilon": 0.15,
        "doc_coverage_fraction": 0.2,
        "question_dedup_retries": 3,
    },
    "tools": {
        "web_search": {
            "enabled": False,
            "serper_num_results": 5,
            "serper_timeout": 30,
        },
        "retrieval": {
            "top_k": 4,
            "chunking": {
                "chunk_size": 750,
                "chunk_overlap": 150,
            },
            "index": {
                "persist_dir": "knowledge_index",
                "rebuild": False,
                "skip_if_unchanged": True,
            },
            "embeddings": {
                "backend": "sentence_transformers",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "fallback_model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "auto",
                "fallback_on_cpu": True,
                "model_kwargs": {},
                "tokenizer_kwargs": {},
                "encode_kwargs": {},
            },
            "reranker": {
                "enabled": False,
                "model": "Qwen/Qwen3-Reranker-4B",
                "backend": "qwen3",
                "instruction": (
                    "Given a user question and a candidate passage from the knowledge base, "
                    "decide whether the passage contains information needed to answer the question."
                ),
                "max_length": 8192,
                "candidates": 12,
                "batch_size": 16,
                "cmd": None,
            },
        },
    },
    "personas": {
        "enabled": True,
        "path": "personas.yaml",
    },
    "judge": {
        "enabled": True,
        "mode": "offline",
        "granularity": "turn",
        "reasons": [
            "irrelevant",
            "incorrect",
            "hallucinated",
            "weak_grounding",
            "vague",
            "incomplete",
            "unsafe",
            "other",
        ],
    },
    "saving": {
        "output_dir": "outputs",
        "output_columns": {
            "messages": "messages",
            "messages_with_tools": "messages_with_tools",
            "metadata": "metadata",
            "user_reasoning": "user_reasoning",
            "assistant_reasoning": "assistant_reasoning",
            "judge": "judge",
        },
        "hf_push": {
            "enabled": False,
            "auto_push_on_run": False,
            "repo_id": "",
            "repo_type": "dataset",
            "export_dir": "hf_export",
            "include_run_state": False,
            "private": True,
            "commit_message": "Update synthetic dataset export",
            "source_file": "conversations_sharegpt_judged.jsonl",
            "generate_stats": False,
            "stats_file": "dataset_stats.json",
            "generate_plots": False,
            "plots_dir": "plots",
        },
    },
}
