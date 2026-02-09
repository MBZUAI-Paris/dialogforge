# Configuration Reference

## Resolution and precedence
Config values are resolved in this order:
1. `DEFAULT_CONFIG` (`src/dlgforge/config/defaults.py`).
2. User YAML (`config.yaml`) deep-merged over defaults.
3. Environment overrides applied in `load_config()`.
4. Runtime-specific environment overrides applied later by distributed bootstrap (for routed endpoints and managed model wiring).
5. LLM per-agent/global env resolution inside `resolve_llm_settings()` at call time.

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
- `backend`: `openai|vllm_attach|vllm_managed`.
- Shared keys: `provider`, `model`, `base_url`, `api_key`, `api_key_env`, `temperature`, `max_tokens`, `top_p`, `timeout`, `max_retries`, `extra`.
- `agents.<agent>` overrides shared keys per logical agent.
- `routing`: endpoint list and strategy for routed execution.
- `vllm`: managed-server settings used when backend is managed.

Per-agent model selection uses:
- agent-specific env `LLM_<AGENT>_*`
- global env `LLM_*`
- `OPENAI_*` fallback for key/base URL/model

## `ray`
- `address`: Ray address, commonly `auto`.
- `auto_start_local`: permits local fallback when `address=auto` and no cluster found.
- `namespace`: Ray namespace.
- `actor.*`: CPU/GPU and replica settings for coordinator/worker actors.

## `store`
- `backend`: currently distributed flow expects `postgres`.
- `postgres.dsn`: required for distributed run.
- pool and timeout keys tune DB client behavior.

## `retrieval`
- `default_k`: retrieval depth.
- `chunk_size`, `overlap`: chunking controls.
- `persist_dir`, `rebuild_index`, `skip_if_unchanged`: index persistence policy.
- embedding backend/model kwargs are configurable and can be env-overridden.

## `coverage`
- controls document balancing and dedup retry budget.
- `question_dedup_retries` directly impacts drop probability in high-collision datasets.

## `tools`
- `web_search_enabled`: toggles tool availability in assistant stage.
- `serper_num_results`, `serper_timeout`: Serper query controls.

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
- retrieval: `KB_DEFAULT_K`, `KB_CHUNK_SIZE`, `KB_PERSIST_DIR`, embedding kwargs JSON keys
- judge: `JUDGE_MODE`, `JUDGE_GRANULARITY`, `JUDGE_ENABLED`
- output/export: `OUTPUT_DIR`, `HF_PUSH_*`
- llm routing: `LLM_BACKEND`, `LLM_ROUTING_STRATEGY`, `LLM_ROUTING_ENDPOINTS_JSON`

For the full source of truth, use `src/dlgforge/config/loader.py` and `src/dlgforge/llm/settings.py`.
