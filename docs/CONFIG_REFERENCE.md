# Configuration Reference

## Resolution and precedence
Config values are resolved in this order:
1. `DEFAULT_CONFIG` (`src/dlgforge/config/defaults.py`).
2. User YAML (`config.yaml`) deep-merged over defaults.
3. Environment overrides applied in `load_config()`.
4. Runtime-specific environment overrides applied later by distributed bootstrap (for routed endpoints and managed model wiring).
5. LLM per-agent env resolution inside `resolve_llm_settings()` at call time.

## `run`
- `n_turns`: fixed turn count fallback.
- `batch_size`: number of conversations advanced concurrently.
- `total_samples`: target conversations per language.
- `target_languages`: primary language loop. If missing, falls back to legacy `target_language`, then `en`.
- `min_turns`/`max_turns`: sampled range. If both are unset/invalid, fixed `n_turns` is used.
- `turn_count_distribution`: `uniform|poisson|exponential` (aliases accepted for some values).
- `turn_count_mean`: positive float used by `poisson`/`exponential`.
- `seed_question` and `question_seed`: deterministic sampling context.
- `run_id` and `resume_run_id`: explicit run identity and resume checkpoint.
- `seed_topics_*`: seed topic path/variant/probability/enable flags.
- `distributed.enabled`: toggles bootstrap path.

Validation details:
- Invalid `min_turns > max_turns` raises `ValueError`.
- `total_samples < 0` falls back to configured default.

## `llm`
- `mode`: `api|vllm_attach|vllm_managed`.
- all provider/model/base_url/sampling knobs are configured per role under `llm.agents.<role>.*`.
- `routing`: endpoint list and strategy for routed execution.
- `vllm`: managed-server settings used when mode is `vllm_managed`.

Credential policy:
- YAML must not contain `api_key` or `api_key_env` (global or per-agent).
- agent credentials are environment-only via:
  - `LLM_USER_API_KEY_ENV`
  - `LLM_ASSISTANT_API_KEY_ENV`
  - `LLM_JUDGE_API_KEY_ENV`
- legacy aliases (`LLM_QA_GENERATOR_API_KEY_ENV`, `LLM_KB_RESPONDER_API_KEY_ENV`, `LLM_QA_JUDGE_API_KEY_ENV`) are still accepted with deprecation warnings.
- each mapping variable must point to a non-empty provider secret env var (name is flexible; common patterns are `*_API_KEY` and `*_TOKEN`).

## `ray`
- `address`: Ray address, commonly `auto`.
- `auto_start_local`: permits local fallback when `address=auto` and no cluster found.
- `namespace`: Ray namespace.
- `actor.*`: CPU/GPU and replica settings for coordinator/worker actors.

## `store`
- `backend`: currently distributed flow expects `postgres`.
- `postgres.dsn`: required for distributed run.
- pool and timeout keys tune DB client behavior.

## `coverage`
- controls document balancing and dedup retry budget.
- `question_dedup_retries` directly impacts drop probability in high-collision datasets.

## `tools`
- `web_search.enabled`: toggles tool availability in assistant stage.
- `web_search.serper_num_results`, `web_search.serper_timeout`: Serper query controls.
- `retrieval.*`: retrieval depth, chunking, index, embeddings, and reranker controls.

## `personas`
- `enabled`: persona injection on/off.
- `path`: YAML persona source path.
- if unavailable, built-in fallback personas are used.

## `judge`
- `mode`: `online|offline`.
- `enabled`: additional gate.
- `granularity`: `turn|conversation`.
- `reasons`: allowed labels for judge output.

Important behavior:
- `resolve_judge_enabled()` returns `False` when mode is `offline` even if `enabled=true`.

## `saving`
- `output_dir`: artifact root.
- `output_columns`: ShareGPT judged export column mapping.
- alias `message_with_tools` is normalized to `messages_with_tools`.
- `hf_push`: export/push toggles and metadata.

## Environment override keys (selected)
Common high-impact env keys handled in `load_config()`:
- run: `N_TURNS`, `BATCH_SIZE`, `TOTAL_SAMPLES`, `MIN_TURNS`, `MAX_TURNS`, `TARGET_LANGUAGES`, `RUN_ID`, `RESUME_RUN_ID`
- distributed/ray: `DISTRIBUTED_ENABLED`, `RAY_ADDRESS`, `RAY_AUTO_START_LOCAL`
- tools.retrieval: `KB_DEFAULT_K`, `KB_CHUNK_SIZE`, `KB_PERSIST_DIR`, embedding kwargs JSON keys
- judge: `JUDGE_MODE`, `JUDGE_GRANULARITY`, `JUDGE_ENABLED`
- output/export: `OUTPUT_DIR`, `HF_PUSH_*`
- llm routing/mode: `LLM_MODE`, `LLM_BACKEND` (legacy alias), `LLM_ROUTING_STRATEGY`, `LLM_ROUTING_ENDPOINTS_JSON`

For the full source of truth, use `src/dlgforge/config/loader.py` and `src/dlgforge/llm/settings.py`.
