# dlgforge

Lightweight synthetic multi-turn dialogue generation with an OpenAI-compatible API.

`dlgforge` generates grounded user-assistant conversations with:
- async batched generation
- deterministic dedup of generated user questions
- optional online judging during generation
- resumable run state
- export-ready JSONL artifacts
- optional one-command distributed bootstrap (Ray + Postgres + vLLM backends)

No CrewAI, LiteLLM, FastAPI, or multiprocessing.

## Table of contents
- [1) What this project does](#1-what-this-project-does)
- [2) Quick start (5 minutes)](#2-quick-start-5-minutes)
- [3) How generation works](#3-how-generation-works)
- [4) Configuration guide](#4-configuration-guide)
- [5) Judge modes and budget control](#5-judge-modes-and-budget-control)
- [6) Async batch + dedup semantics](#6-async-batch--dedup-semantics)
- [7) Persona sampling behavior](#7-persona-sampling-behavior)
- [8) Outputs and inspection](#8-outputs-and-inspection)
- [9) Resume and run state](#9-resume-and-run-state)
- [10) CLI commands](#10-cli-commands)
- [11) Troubleshooting playbook](#11-troubleshooting-playbook)

## 1) What this project does
The pipeline runs up to three logical stages per turn:
1. `qa_generator`: produces the next user message.
2. `kb_responder`: answers using KB retrieval (and optional web search tool).
3. `qa_judge`: evaluates quality/grounding (configurable granularity).

It supports:
- fixed turns (`run.n_turns`) or sampled turns per conversation (`min_turns/max_turns` + distribution)
- batched concurrent generation (`run.batch_size`)
- language loops (`run.target_languages`) with `total_samples` generated per language
- deterministic exact-normalized question dedup across the full run

## 2) Quick start (5 minutes)
### 2.1 Create environment and install
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you want managed local/cluster vLLM autostart mode (`llm.backend: vllm_managed`, Linux GPU nodes):
```bash
python -m pip install -e ".[vllm]"
```

### 2.2 Configure env vars
```bash
cp .env.example .env
```

Minimum required:
- `OPENAI_API_KEY`
- model source from one of:
  - `llm.agents.<agent>.model`
  - `llm.model`
  - `LLM_MODEL`
  - `OPENAI_MODEL`

### 2.3 Run generation
```bash
dlgforge run config.yaml
```

Equivalent:
```bash
uv run dlgforge run config.yaml
python -m dlgforge run config.yaml
```

### 2.4 Verify first outputs
- `outputs/conversations/*.json`
- `outputs/conversations_sharegpt.jsonl`
- `outputs/turns.jsonl`
- `outputs/run_state/*.json`
- `logs/run.log`, `logs/llm.log`, `logs/judge.log`

### 2.5 Recommended mode by setup
- macOS laptop + LM Studio: `run.distributed.enabled: false` + `llm.backend: openai` + `llm.base_url: http://127.0.0.1:1234/v1`
- macOS laptop + distributed orchestrator: `run.distributed.enabled: true` + `llm.backend: vllm_attach` + Postgres DSN
- Linux GPU nodes and self-managed cluster: `run.distributed.enabled: true` + `llm.backend: vllm_managed`

### 2.6 Run modes (copy-paste)
#### A) LM Studio local, non-distributed (recommended on macOS)
```yaml
run:
  distributed:
    enabled: false
llm:
  backend: openai
  base_url: http://127.0.0.1:1234/v1
  api_key: EMPTY
  agents:
    qa_generator:
      model: openai/gpt-oss-20b
    kb_responder:
      model: openai/gpt-oss-20b
    qa_judge:
      model: openai/gpt-oss-20b
```
```bash
dlgforge run config.yaml
```

#### B) OpenAI, non-distributed (no Ray/Postgres required)
```yaml
run:
  distributed:
    enabled: false
llm:
  backend: openai
```
```bash
dlgforge run config.yaml
```

#### C) OpenAI, one-command distributed (Ray + Postgres)
```yaml
run:
  distributed:
    enabled: true
ray:
  address: auto
  auto_start_local: true
llm:
  backend: openai
```
```bash
export DLGFORGE_POSTGRES_DSN='postgresql://USER:PASS@HOST:5432/DB'
dlgforge run config.yaml
```

If no Ray cluster is running, `ray.auto_start_local: true` lets `dlgforge` start a local Ray runtime automatically.

#### D) vLLM attach (you already started vLLM endpoints)
```yaml
run:
  distributed:
    enabled: true
llm:
  backend: vllm_attach
  routing:
    endpoints:
      - name: gpu-node-1
        base_url: http://10.0.0.11:8000/v1
        api_key: EMPTY
```
```bash
export DLGFORGE_POSTGRES_DSN='postgresql://USER:PASS@HOST:5432/DB'
dlgforge run config.yaml
```

#### E) vLLM managed autostart (dlgforge starts/stops vLLM)
```yaml
run:
  distributed:
    enabled: true
llm:
  backend: vllm_managed
  vllm:
    model: Qwen/Qwen2.5-7B-Instruct
    served_model_name: qwen
```
```bash
python -m pip install -e ".[vllm]"
export DLGFORGE_POSTGRES_DSN='postgresql://USER:PASS@HOST:5432/DB'
dlgforge run config.yaml
```

Notes:
- managed mode is Linux-oriented (`vllm` extra is Linux-only in this project).
- on macOS, use LM Studio local mode (`A`) or distributed attach mode (`D`).
- if using managed mode, align your `llm.agents.*.model` values with `llm.vllm.served_model_name`.

### 2.7 Start Postgres quickly (required for distributed modes)
Distributed modes (`C`, `D`, `E`) require Postgres.

Start a local Postgres with Docker:
```bash
docker run -d \
  --name dlgforge-postgres \
  -e POSTGRES_USER=dlgforge \
  -e POSTGRES_PASSWORD=dlgforge \
  -e POSTGRES_DB=dlgforge \
  -p 5432:5432 \
  postgres:16
```

Set DSN:
```bash
export DLGFORGE_POSTGRES_DSN='postgresql://dlgforge:dlgforge@127.0.0.1:5432/dlgforge'
```

Health check:
```bash
docker exec dlgforge-postgres pg_isready -U dlgforge -d dlgforge
```

Reuse existing container:
```bash
docker start dlgforge-postgres
```

Stop when done:
```bash
docker stop dlgforge-postgres
```

## 3) How generation works
At runtime:
1. Load config + env overrides.
2. Build base inputs and runtime settings.
3. For each target language:
   - run one or more waves until `total_samples` for that language is reached
   - each wave runs `batch_size` conversations (or remaining count)
4. Persist artifacts and optional HF auto-push.

Important:
- `total_samples` is per language, not global.
- if `target_languages` has 5 values and `total_samples=200`, target is 1000 conversations total.

## 4) Configuration guide
### 4.1 `run`
Core run controls:
- `run.n_turns`: fixed turns when no range sampling is used.
- `run.batch_size`: number of conversations advanced concurrently.
- `run.total_samples`: number of conversations to persist per language.
- `run.target_languages`: list of languages.
- `run.run_id`: optional explicit run id.
- `run.resume_run_id`: resume checkpoint.

Turn count sampling:
- `run.min_turns`
- `run.max_turns`
- `run.turn_count_distribution`: `uniform`, `poisson`, or `exponential`
- `run.turn_count_mean`: mean for `poisson`/`exponential`

Behavior:
- sampled turns are clamped to `[min_turns, max_turns]`
- each conversation samples independently

### 4.2 `llm`
OpenAI-compatible settings:
- `llm.provider`
- `llm.base_url`
- `llm.api_key` / env
- per-agent overrides under `llm.agents`
- `llm.backend`: `openai`, `vllm_attach`, or `vllm_managed`
- `llm.routing.*`: multi-endpoint routing (used by attach/managed vLLM modes)
- `llm.vllm.*`: managed vLLM replica settings when `llm.backend: vllm_managed`

Agents:
- `qa_generator`
- `kb_responder`
- `qa_judge`

### 4.3 `retrieval`
KB search defaults:
- `retrieval.default_k`
- chunking and index options

### 4.4 `coverage`
Dedup and coverage behavior:
- `coverage.question_dedup_retries`
- coverage balancing parameters

### 4.5 `personas`
Persona controls:
- `personas.enabled`
- `personas.path`

Current recommended path:
```yaml
personas:
  enabled: true
  path: src/dlgforge/prompts/personas.yaml
```

If path is missing/unreadable, built-in fallback personas are used.

### 4.6 `judge`
Judge controls:
- `judge.enabled`
- `judge.mode`: `online` or `offline`
- `judge.granularity`: `turn` or `conversation`
- `judge.reasons`: allowed labels

### 4.7 `saving`
Output layout and export:
- `saving.output_dir`
- `saving.output_columns.*` (renamable JSONL columns)
- `saving.hf_push.*`

### 4.8 Distributed one-command runtime
Enable one-command distributed launch from the same CLI entrypoint:
```yaml
run:
  distributed:
    enabled: true
    executor: ray
    spawn:
      coordinator: true
      workers: true

ray:
  address: "auto"
  auto_start_local: true
  namespace: "dlgforge"

store:
  backend: postgres
  postgres:
    dsn: "${DLGFORGE_POSTGRES_DSN}"

llm:
  backend: vllm_attach  # openai | vllm_attach | vllm_managed
  routing:
    strategy: weighted_least_inflight
    endpoints:
      - name: gpu-node-1
        base_url: http://10.0.0.11:8000/v1
        api_key: EMPTY
      - name: gpu-node-2
        base_url: http://10.0.0.12:8000/v1
        api_key: EMPTY
```

Behavior:
- `dlgforge run config.yaml` bootstraps coordinator + workers automatically when `run.distributed.enabled: true`
- Ray init tries `ray.address` first; when `ray.address: auto` has no running cluster and `ray.auto_start_local: true`, it falls back to a local Ray runtime
- `llm.backend: openai` uses hosted API (no vLLM provisioning)
- `llm.backend: vllm_attach` validates configured vLLM endpoints before run
- `llm.backend: vllm_managed` starts/stops vLLM servers on Ray GPU actors
- current execution path runs generation from the coordinator actor while worker replicas are provisioned for lifecycle orchestration hooks

Bootstrap sequence:
```mermaid
flowchart TD
  A["dlgforge run config.yaml"] --> B["RunBootstrap"]
  B --> C["Initialize Ray"]
  C --> D["Validate Postgres DSN and ping"]
  D --> E{"llm.backend"}
  E -->|openai| F["No vLLM provisioning"]
  E -->|vllm_attach| G["Validate configured /v1/models endpoints"]
  E -->|vllm_managed| H["Spawn vLLM server actors and wait healthy"]
  F --> I["Spawn coordinator actor"]
  G --> I
  H --> I
  B --> J["Spawn worker actors"]
  I --> K["Coordinator executes generation run"]
```

Current dispatch/execution model:
```mermaid
flowchart LR
  U["User: dlgforge run"] --> B["RunBootstrap"]
  B --> C["Coordinator actor"]
  B --> W["Worker actors (provisioned)"]
  C --> P["Existing generation loop (turn logic)"]
  P --> R["Endpoint routing"]
  R --> O["OpenAI API"]
  R --> VA["Attached vLLM endpoints"]
  R --> VM["Managed vLLM endpoints"]
```

Mode matrix:
```mermaid
flowchart TD
  S{"run.distributed.enabled"}
  S -->|false| L["Local mode: openai only, no Ray/Postgres requirement"]
  S -->|true| D["Distributed mode: Ray plus Postgres required"]
  D --> B{"llm.backend"}
  B -->|openai| BO["Hosted OpenAI path"]
  B -->|vllm_attach| BA["Use user-provided vLLM endpoints"]
  B -->|vllm_managed| BM["Auto-start vLLM on Ray GPU workers (vllm extra required)"]
```

Useful HF export options:
- `saving.hf_push.source_file`: use `conversations_sharegpt_judged.jsonl` to include judge column.
- `saving.hf_push.generate_stats`: writes dataset stats JSON during export.
- `saving.hf_push.stats_file`: stats JSON filename (default `dataset_stats.json`).
- `saving.hf_push.generate_plots`: writes SVG distribution plots during export.
- `saving.hf_push.plots_dir`: plot output folder inside export dir.

## 5) Judge modes and budget control
Two orthogonal controls:

1. **when** to run judge
- `judge.mode: online` -> judge integrated in `dlgforge run`
- `judge.mode: offline` -> no judge during generation

2. **how often** to run judge
- `judge.granularity: turn` -> judge every turn
- `judge.granularity: conversation` -> judge once per conversation

### 5.1 Recommended config for lower budget
```yaml
judge:
  enabled: true
  mode: online
  granularity: conversation
  reasons:
    - irrelevant
    - incorrect
    - hallucinated
    - weak_grounding
    - vague
    - incomplete
    - unsafe
    - other
```

### 5.2 Tradeoff summary
- `turn` granularity:
  - pros: fine-grained labels and diagnostics
  - cons: most judge-token expensive
- `conversation` granularity:
  - pros: 1 judge call per conversation, cheaper
  - cons: less localized feedback per turn

### 5.3 Verification checklist
For `judge.mode: online` + `granularity: turn`:
1. run `dlgforge run config.yaml`
2. check `logs/judge.log` for `[judge-online] ...`
3. check conversation turns contain `qa_judge`
4. check `outputs/conversations_sharegpt_judged.jsonl` grows

For `judge.mode: online` + `granularity: conversation`:
1. run `dlgforge run config.yaml`
2. check `logs/judge.log` for `[judge-online-conversation] ...`
3. check conversation payload has `conversation_judge`
4. check judged export `judge.conversation` is populated

## 6) Async batch + dedup semantics
### 6.1 Concurrency
- conversations advance independently in slots
- slot ordering is deterministic for acceptance/commit

### 6.2 Dedup scope
Question dedup uses normalized exact match:
- lowercase
- collapsed whitespace

Applied across:
- duplicates within the same batch attempt
- duplicates already accepted in prior attempts/batches

### 6.3 Retry and drop policy
- rejected duplicate slots are regenerated only for missing slots
- retries capped by `coverage.question_dedup_retries`
- on exhaustion: slot marked dropped (`drop_reason=dedup_exhausted`)

## 7) Persona sampling behavior
Personas are sampled per conversation (not per run):
- each conversation gets one user persona and one assistant persona
- sampling is uniform-cycle based across available personas
- this avoids overusing a small subset when generating many samples

In batched mode:
- each slot has its own persona assignment
- assignment is persisted in run_state and reused on resume

## 8) Outputs and inspection
Main files under `outputs/`:
- `synthetic_qa.jsonl`: one record per conversation
- `coverage_ledger.jsonl`: dedup/coverage memory
- `turns.jsonl`: flattened per-turn rows
- `conversations_sharegpt.jsonl`: ShareGPT export
- `conversations_sharegpt_judged.jsonl`: judged ShareGPT export
- `conversations/<conversation_id>.json`: rich conversation artifact
- `run_state/<run_id>.json`: checkpoint state

Judge fields:
- per-turn mode: `turns[].qa_judge`
- conversation mode: top-level `conversation_judge`
- judged ShareGPT column (`judge`) includes:
  - `per_turn`
  - `avg_score`
  - `conversation`

Useful commands:
```bash
# runtime
tail -f logs/run.log

# judge logs
tail -f logs/judge.log

# latest conversation files
ls -lt outputs/conversations | head

# inspect conversation-level judge
rg "conversation_judge" outputs/conversations/*.json

# inspect turn-level judge fields
rg "judge_score|judge_reasons|conversation_judge_score" outputs/turns.jsonl
```

## 9) Resume and run state
Resume from checkpoint:
```yaml
run:
  resume_run_id: "<existing_run_id>"
```

Batched resume rules:
- `run.batch_size` must match saved run_state batch size
- slot states (`active/completed/dropped`) are restored
- per-slot persona inputs are restored
- dedup memory is restored from ledger and existing accepted turns

## 10) CLI commands
Run generation:
```bash
dlgforge run config.yaml
```

Judge-only pass on existing conversations:
```bash
dlgforge judge config.yaml
```

Push/export:
```bash
dlgforge push config.yaml
dlgforge push config.yaml --no-push
```

Seed migration:
```bash
dlgforge seeds-migrate config.yaml --source-file seed_topics.json --dest-file data/seeds/topics.yaml --overwrite
```

## 11) Troubleshooting playbook
### 11.1 Judge not running
Check:
- `judge.enabled: true`
- `judge.mode: online`
- `llm.agents.qa_judge.model` resolves
- API key present

### 11.2 No judged output in conversation mode
Check:
- `judge.granularity: conversation`
- `logs/judge.log` has `[judge-online-conversation]`
- conversation files contain `conversation_judge`

### 11.3 No judged output in turn mode
Check:
- `judge.granularity: turn`
- `logs/judge.log` has `[judge-online]`
- turn payloads contain `qa_judge`

### 11.4 Same persona repeated too often
Check:
- persona file path exists and is readable
- list has enough personas
- `personas.enabled: true`

### 11.5 Batch run appears stalled
Check:
- dedup pressure and retry budget
- model latency in `logs/llm.log`
- dropped slots in run_state (`drop_reason`)

### 11.6 Embedding/index mismatch
If retrieval errors appear after model/backend changes:
- set `retrieval.rebuild_index: true` for one run, then back to `false`
- or remove `knowledge_index/` and regenerate

---

If you need a strict production profile, keep these defaults:
- `judge.mode: online`
- `judge.granularity: conversation` (budget-friendly)
- `batch_size` tuned to provider throughput
- `question_dedup_retries` >= 3
