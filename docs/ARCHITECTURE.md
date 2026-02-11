# Architecture

## System purpose
DialogForge generates grounded synthetic multi-turn conversations with deterministic dedup behavior, resumable run state, and export-ready outputs. The system is built around a small CLI contract and a modular pipeline that can run locally or through distributed bootstrap.

## End-to-end execution flow
```mermaid
flowchart TD
  A["dlgforge run config.yaml"] --> B["Load config (defaults + YAML + env overrides)"]
  B --> C{"run.distributed.enabled"}
  C -->|false| D["Initialize local pipeline runtime"]
  C -->|true| E["RunBootstrap (Ray + Postgres + mode checks)"]
  E --> F["Coordinator actor executes generation run"]
  D --> G["Configure retrieval + base inputs"]
  F --> G
  G --> H["Run generation waves per target language"]
  H --> I["Persist outputs + run_state"]
  I --> J["Optional HF auto-push/export"]
```

## Turn pipeline
```mermaid
flowchart LR
  Q["qa_generator"] --> R["kb_responder"]
  R --> T{"judge.enabled"}
  T -->|false| U["Persist turn/conversation"]
  T -->|true + turn| V["qa_judge per turn"]
  T -->|true + conversation| W["conversation-level judge after final turn"]
  V --> U
  W --> U
```

## Distributed bootstrap sequence
```mermaid
flowchart TD
  A["RunBootstrap"] --> B["Initialize Ray runtime"]
  B --> C["Validate Postgres DSN and connectivity"]
  C --> D{"llm.mode"}
  D -->|api| E["No vLLM provisioning"]
  D -->|vllm_attach| F["Validate configured endpoint health"]
  D -->|vllm_managed| G["Provision managed vLLM server actors"]
  E --> H["Spawn coordinator + workers"]
  F --> H
  G --> H
  H --> I["Execute generation workflow"]
```

## Core module boundaries
- `src/dlgforge/cli.py`: external command surface and dispatch.
- `src/dlgforge/config`: config defaults, loader, and resolver layer.
- `src/dlgforge/pipeline/runner.py`: top-level generation orchestration.
- `src/dlgforge/pipeline/sampling.py`: question selection, coverage memory, and seed-topic mechanics.
- `src/dlgforge/tools/retrieval.py`: vector index lifecycle and retrieval operations.
- `src/dlgforge/io/output.py`: output paths and artifact writing.
- `src/dlgforge/pipeline/state.py`: resume/checkpoint state handling.
- `src/dlgforge/distributed`: bootstrap and backend provisioning abstractions.
- `src/dlgforge/pipeline/hf_push.py`: export packaging and hub push flow.

## Data and state model
- Conversation and turn artifacts are written under `saving.output_dir`.
- Run progress is checkpointed in `run_state` files keyed by `run_id`.
- Dedup/coverage memory is tracked in append-oriented ledgers.
- Resume uses persisted run-state and memory artifacts to continue without replaying accepted outputs.

## Stability model
- Stable operator-facing contracts in `v0.1.x`: CLI commands, documented config surfaces, and output layouts.
- Internal module structure under `src/dlgforge` is not a stability contract.
