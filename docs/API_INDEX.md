# DialogForge API Index (Curated)

## Purpose
This index documents the stable, practical API surface for operators and contributors. It focuses on module responsibilities, public entrypoints, side effects, and call relationships.

## External entrypoints

### CLI (`src/dlgforge/cli.py`)
- `build_parser() -> argparse.ArgumentParser`
- `main() -> None`

Commands routed by `main`:
- `run` -> `pipeline.runner.run`
- `judge` -> `pipeline.runner.run_judge_only`
- `push` -> `pipeline.hf_push.run_push`
- `seeds-migrate` -> `pipeline.seed_topics_migration.run_seeds_migrate`

### Package exports (`src/dlgforge/__init__.py`)
- `run`
- `run_judge_only`
- `run_push`
- `run_seeds_migrate`
- `HFPushOptions`

## Configuration API

### `src/dlgforge/config/defaults.py`
- `DEFAULT_CONFIG`: canonical default schema baseline.

### `src/dlgforge/config/loader.py`
Public resolver contract:
- `load_config`
- `build_base_inputs`
- `resolve_output_dir`
- `resolve_output_columns`
- `resolve_target_languages`
- `resolve_n_turns`, `resolve_turn_range`, `resolve_turn_count_distribution`, `resolve_turn_count_mean`
- `resolve_total_samples`, `resolve_batch_size`
- `resolve_distributed_enabled`
- `resolve_retrieval_default_k`
- `resolve_seed_topics_path`, `resolve_seed_topics_variant`, `resolve_seed_topics_probability`, `resolve_seed_topics_enabled`
- `resolve_judge_mode`, `resolve_judge_enabled`, `resolve_judge_granularity`, `resolve_judge_reasons`

Side effects:
- file read (`config.yaml`)
- environment override application into in-memory config object

### `src/dlgforge/config/personas.py`
Public contract:
- `UniformPersonaSampler.sample`
- `select_personas`
- `build_uniform_persona_sampler`
- `load_personas`
- persona resolver helpers (`resolve_personas_enabled`, `resolve_personas_path`, `resolve_question_seed`)

Side effects:
- persona YAML file reads with fallback path handling

## Pipeline API

### `src/dlgforge/pipeline/runner.py`
Primary orchestration entrypoints:
- `run(config_path)`
- `run_judge_only(config_path)`

Advanced orchestration helpers used internally and by tests:
- `run_multi_turn`
- `run_multi_turn_batched_async`
- `persist_training_sample`
- `persist_batched_training_samples`

Side effects:
- network/model calls
- retrieval and optional web tool invocations
- output artifact writes
- run-state checkpoint writes
- optional distributed bootstrap handoff

### `src/dlgforge/pipeline/sampling.py`
Sampling and memory contracts:
- `build_question_inputs`
- `select_question_mode`
- `sample_topic_snippets`
- `update_coverage_ledger`
- `is_duplicate_question`
- `update_doc_question_memory`
- seed-topic helpers (`load_seed_topics`, `select_seed_candidate`, `update_seed_memory`)

Side effects:
- coverage ledger writes through IO layer
- seed-topic file reads

### `src/dlgforge/pipeline/state.py`
Run-state contract:
- `init_run_state`, `checkpoint_run_state`
- `init_batched_run_state`, `checkpoint_batched_run_state`
- `build_initial_batched_conversations`, `load_batched_conversations_from_state`

Side effects:
- run-state JSON writes/reads

### `src/dlgforge/pipeline/history.py`
History formatting contract:
- `format_history`
- `build_conversation_history`
- `build_public_history`
- `messages_up_to_turn`

### `src/dlgforge/pipeline/prompts.py`
Prompt loading/rendering contract:
- `load_agents_config`
- `load_tasks_config`
- `build_agent_system_prompt`
- `build_task_prompt`

Side effects:
- prompt YAML file reads with LRU caching

### `src/dlgforge/pipeline/dedup.py`
Dedup contract:
- `normalize_question`
- `RunQuestionRegistry.filter_and_commit`

### `src/dlgforge/pipeline/seed_topics_migration.py`
Seed migration contract:
- `run_seeds_migrate`
- `migrate_seed_topics_file`

Side effects:
- seed file read/write and destination creation

### `src/dlgforge/pipeline/hf_push.py`
Export/push contract:
- dataclasses: `HFPushSettings`, `HFPushOptions`
- `run_push`
- `maybe_auto_push_after_run`
- `resolve_hf_push_settings`
- `prepare_export`
- `push_to_hub`

Side effects:
- export directory file writes
- optional Hugging Face network push

## LLM API

### `src/dlgforge/llm/settings.py`
- `resolve_llm_settings`
- `resolve_agent_used_name`
- `required_agents`
- `missing_models`

### `src/dlgforge/llm/client.py`
- dataclass `ChatResult`
- client `OpenAIModelClient.complete`
- client `OpenAIModelClient.acomplete`

Side effects:
- outbound API calls
- routing state mutation for weighted least-inflight logic

## Tooling API

### `src/dlgforge/tools/retrieval.py`
- class `KnowledgeVectorStore` with retrieval/listing/sample methods
- `configure_retrieval`
- `get_vector_store`
- `vector_db_search`

Side effects:
- index build/read on filesystem
- embedding model initialization

### `src/dlgforge/tools/web_search.py`
- `SerperWebSearchClient.search`

Side effects:
- outbound HTTP requests to Serper endpoint

## IO and utility API

### `src/dlgforge/io/output.py`
Public output contract:
- `OutputPaths`
- `configure_output_columns`
- `ensure_output_layout`
- `load_coverage_ledger`, `append_coverage_ledger`
- `save_run_state`, `load_run_state`
- `save_training_sample`
- `append_sharegpt_judged_record`

### `src/dlgforge/utils/*.py`
- env parsing: `env_flag`, `env_int`, `env_float`, `load_dotenv_files`
- JSON parsing: `strip_code_fences`, `extract_json_object`, `parse_json_object`
- logging setup: `setup_logging`
- config/path merges: `deep_merge`, `resolve_path`
- text helpers: `hash_text`, `render_template`

## Distributed runtime API

### `src/dlgforge/distributed/bootstrap.py`
- class `RunBootstrap.run`
- function `run_bootstrap`

### `src/dlgforge/distributed/provisioning.py`
- protocol `VLLMProvisioner`
- implementations: `NoopProvisioner`, `AttachProvisioner`, `ManagedRayVLLMProvisioner`

### `src/dlgforge/distributed/ray_runtime.py`
- `import_ray`
- `create_worker_actor`
- `create_coordinator_actor`
- `create_vllm_server_actor`

### `src/dlgforge/distributed/types.py`
- dataclass `EndpointSpec`
- `EndpointSpec.to_routing_dict`

## Cross-module call relationships
- `cli` -> `pipeline.runner` / `pipeline.hf_push` / `pipeline.seed_topics_migration`
- `pipeline.runner` -> `config`, `pipeline.sampling`, `pipeline.prompts`, `llm`, `tools`, `io`, `pipeline.state`, `pipeline.hf_push`
- `pipeline.sampling` -> `tools.retrieval`, `io.output`
- `distributed.bootstrap` -> `distributed.provisioning` + `distributed.ray_runtime` -> `pipeline.runner`
- `pipeline.hf_push` -> `config` + `io` + optional Hub API

## Package export map (`__all__`)

### `dlgforge`
- `run`, `run_judge_only`, `run_push`, `run_seeds_migrate`, `HFPushOptions`

### `dlgforge.config`
- `DEFAULT_CONFIG`, `load_config`, `build_base_inputs`, resolver helpers

### `dlgforge.pipeline`
- `run`, `run_judge_only`, `run_push`, `run_seeds_migrate`, `HFPushOptions`

### `dlgforge.llm`
- `ChatResult`, `OpenAIModelClient`, settings helpers

### `dlgforge.io`
- `OutputPaths` and output persistence helpers

### `dlgforge.tools`
- retrieval config/search helpers and `SerperWebSearchClient`

### `dlgforge.distributed`
- bootstrap/provisioning types and helpers

### `dlgforge.utils`
- env/json/logging/merge/text helper functions
