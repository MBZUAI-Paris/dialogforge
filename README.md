<p align="center">
  <img src="assets/logo.png" alt="DialogForge logo" width="140" />
</p>

<h1 align="center">dlgforge</h1>

<p align="center"><strong>Lightweight synthetic multi-turn dialogue generation with a LiteLLM-routed LLM stack.</strong></p>

<p align="center">
  <a href="https://mbzuai-paris.github.io/dialogforge/">
    <img src="https://img.shields.io/badge/docs-mbzuai--paris.github.io%2Fdialogforge-blue?style=flat-square" alt="Docs">
  </a>
  <a href="https://pypi.org/project/dlgforge/">
    <img src="https://img.shields.io/pypi/v/dlgforge?style=flat-square" alt="PyPI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/MBZUAI-Paris/dialogforge?style=flat-square" alt="License">
  </a>
</p>

`dlgforge` generates grounded user-assistant conversations with:
- async batched generation
- deterministic dedup of generated user questions
- optional online judging during generation
- resumable run state
- export-ready JSONL artifacts
- optional one-command distributed bootstrap (Ray + Postgres + vLLM backends)

<p align="center">
  <img src="assets/dialogforge.jpeg" alt="dlgforge architecture diagram" width="75%" />
</p>

## Documentation hub
- [Docs home](docs/index.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Configuration reference](docs/CONFIG_REFERENCE.md)
- [CLI reference](docs/CLI_REFERENCE.md)
- [API reference (auto-generated)](docs/reference/index.md)

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
- fixed turns (`run.turns.mode: exact` + `run.turns.exact`) or sampled turns per conversation (`run.turns.mode: range` + `run.turns.min/max/distribution`)
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

If you want managed local/cluster vLLM autostart mode (`llm.mode: vllm_managed`, Linux GPU nodes):
```bash
python -m pip install -e ".[vllm]"
```

### 2.2 Configure env vars
```bash
cp .env.example .env
```

Minimum required:
- model per active role:
  - `llm.agents.user.model`
  - `llm.agents.assistant.model`
  - `llm.agents.judge.model` (when `judge.mode: online`)
- provider credential env vars (for example `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`)
  - provider secret names are flexible; common patterns like `*_API_KEY` and `*_TOKEN` are supported
- required agent credential mapping vars:
  - `LLM_USER_API_KEY_ENV`
  - `LLM_ASSISTANT_API_KEY_ENV`
  - `LLM_JUDGE_API_KEY_ENV`
  - legacy aliases (`LLM_QA_GENERATOR_API_KEY_ENV`, `LLM_KB_RESPONDER_API_KEY_ENV`, `LLM_QA_JUDGE_API_KEY_ENV`) are still accepted with deprecation warnings
- do not place `api_key` or `api_key_env` in YAML; credentials are environment-only
- `HF_TOKEN` is optional for generation with the default embedding model (`sentence-transformers/all-MiniLM-L6-v2`); it is needed for gated/private HF models and `dlgforge push`

### 2.3 Prepare knowledge directory
`dlgforge run` requires a `knowledge/` folder at the project root with at least one supported file (`.txt`, `.md`, or `.pdf`).

```bash
mkdir -p knowledge
# add your source docs under knowledge/, for example: knowledge/product_faq.md
```

By default, retrieval embeddings/chunks are cached under `knowledge_index/` (`tools.retrieval.index.persist_dir` in `config.yaml`).

### 2.4 Run generation
```bash
uv run env PYTHONPATH=src python -m dlgforge run config.yaml
```

If the package is installed in the environment (`uv pip install -e .`), these are equivalent:
```bash
uv run dlgforge run config.yaml
dlgforge run config.yaml
```

### 2.5 Verify first outputs
- `outputs/conversations/*.json`
- `outputs/conversations_sharegpt.jsonl`
- `outputs/turns.jsonl`
- `outputs/run_state/*.json`
- `logs/run.log`, `logs/llm.log`, `logs/judge.log`

### 2.6 Recommended mode by setup
- macOS laptop + LM Studio: `run.distributed.enabled: false` + `llm.mode: api` + `llm.agents.*.base_url: http://127.0.0.1:1234/v1`
- macOS laptop + distributed orchestrator: `run.distributed.enabled: true` + `llm.mode: vllm_attach` + Postgres DSN
- Linux GPU nodes and self-managed cluster: `run.distributed.enabled: true` + `llm.mode: vllm_managed`

### 2.7 Run modes (copy-paste)
#### A) LM Studio local, non-distributed (recommended on macOS)
```yaml
run:
  distributed:
    enabled: false
llm:
  mode: api
  agents:
    user:
      model: openai/gpt-oss-20b
      base_url: http://127.0.0.1:1234/v1
    assistant:
      model: openai/gpt-oss-20b
      base_url: http://127.0.0.1:1234/v1
    judge:
      model: openai/gpt-oss-20b
      base_url: http://127.0.0.1:1234/v1
```
Make sure to specify the correct base URLs for each agent via `llm.agents.<role>.base_url` (refer to https://docs.litellm.ai/docs/providers for base_url and api_key formats for different providers). dlgforge auto-uses LiteLLM openai-compatible passthrough so namespaced model IDs like `openai/gpt-oss-20b` are forwarded unchanged.
```bash
dlgforge run config.yaml
```

#### B) OpenAI, non-distributed (no Ray/Postgres required)
```yaml
run:
  distributed:
    enabled: false
llm:
  mode: api
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
  mode: api
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
  mode: vllm_attach
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
  mode: vllm_managed
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

### 2.7 Mixed providers by agent
When mixing providers (for example OpenAI + Gemini + LM Studio), define provider/model/base URL per agent and follow LiteLLM provider conventions:
- LiteLLM providers reference: [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers)

Recommended per-agent setup:
- OpenAI API:
  - `provider: openai`
  - `model: gpt-5.2` (or another OpenAI model)
  - `base_url: https://api.openai.com/v1`
- Gemini (Google AI Studio):
  - `provider: gemini`
  - `model: gemini/gemini-2.0-flash` (or another Gemini model)
  - leave `base_url` empty unless you intentionally set a Gemini-specific proxy/base URL
- LM Studio (OpenAI-compatible local server):
  - `provider: openai` (or `lm_studio`)
  - `model: openai/gpt-oss-20b` (or your served model name format)
  - `base_url: http://localhost:1234/v1`
- vLLM (OpenAI-compatible endpoint):
  - `provider: openai` (or `hosted_vllm`)
  - `base_url: http://localhost:8000/v1`

Be careful with env/config precedence:
- credentials are env-only via `LLM_USER_API_KEY_ENV`, `LLM_ASSISTANT_API_KEY_ENV`, `LLM_JUDGE_API_KEY_ENV`.
- `LLM_<ROLE>_*` env vars override `config.yaml` values for that specific role.
- for mixed providers, prefer setting provider/model/base_url explicitly per agent in `llm.agents.<role>`.

### 2.8 Start Postgres quickly (required for distributed modes)
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
- `run.batch_size`: number of conversations advanced concurrently.
- `run.total_samples`: number of conversations to persist per language.
- `run.target_languages`: list of languages.
- `run.run_id`: optional explicit run id.
- `run.resume_run_id`: resume checkpoint.

Turn count:
- `run.turns.mode`: `exact` or `range`
- `run.turns.exact`: used when `mode: exact`
- `run.turns.min`, `run.turns.max`: used when `mode: range`
- `run.turns.distribution`: `uniform`, `poisson`, or `exponential`
- `run.turns.mean`: mean for `poisson`/`exponential`

Data shaping:
- `run.data.seeding.question`
- `run.data.seeding.topics.path`
- `run.data.seeding.topics.enabled`
- `run.data.seeding.topics.variant`
- `run.data.seeding.topics.probability`

Behavior:
- sampled turns are clamped to `[run.turns.min, run.turns.max]`
- each conversation samples independently

### 4.2 `llm`
LiteLLM-routed settings:
- `llm.mode`: `api`, `vllm_attach`, or `vllm_managed`
- per-agent settings under `llm.agents.<role>`:
  - `provider`, `model`, `base_url`
  - optional sampling params (`temperature`, `max_tokens`, `top_p`, `timeout`, `max_retries`, `extra`)
- credentials are environment-only via:
  - `LLM_USER_API_KEY_ENV`
  - `LLM_ASSISTANT_API_KEY_ENV`
  - `LLM_JUDGE_API_KEY_ENV`
- `llm.routing.*`: multi-endpoint routing (used by attach/managed vLLM modes)
- `llm.vllm.*`: managed vLLM replica settings when `llm.mode: vllm_managed`

Agents:
- `user`
- `assistant`
- `judge`

### 4.3 `tools`
Tool settings:
- `tools.web_search.enabled`
- `tools.web_search.serper_num_results`
- `tools.web_search.serper_timeout`
- `tools.retrieval.top_k`
- `tools.retrieval.chunking.chunk_size`
- `tools.retrieval.chunking.chunk_overlap`
- `tools.retrieval.index.persist_dir`
- `tools.retrieval.index.rebuild`
- `tools.retrieval.embeddings.*`
- `tools.retrieval.reranker.*`

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
    backend: ray
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
  mode: vllm_attach  # api | vllm_attach | vllm_managed
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
- Ray init tries `run.distributed.ray.address` first; when `run.distributed.ray.address: auto` has no running cluster and `run.distributed.ray.auto_start_local: true`, it falls back to a local Ray runtime
- `llm.mode: api` uses hosted API (no vLLM provisioning)
- `llm.mode: vllm_attach` validates configured vLLM endpoints before run
- `llm.mode: vllm_managed` starts/stops vLLM servers on Ray GPU actors
- current execution path runs generation from the coordinator actor while worker replicas are provisioned for lifecycle orchestration hooks

Bootstrap sequence:
```mermaid
flowchart TD
  A["dlgforge run config.yaml"] --> B["RunBootstrap"]
  B --> C["Initialize Ray"]
  C --> D["Validate Postgres DSN and ping"]
  D --> E{"llm.mode"}
  E -->|api| F["No vLLM provisioning"]
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
  S -->|false| L["Local mode: api, no Ray/Postgres requirement"]
  S -->|true| D["Distributed mode: Ray plus Postgres required"]
  D --> B{"llm.mode"}
  B -->|api| BO["Hosted API path"]
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
uv run env PYTHONPATH=src python -m dlgforge run config.yaml
```

If the package is installed in the active environment:
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
- `llm.agents.judge.model` resolves
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
- set `tools.retrieval.index.rebuild: true` for one run, then back to `false`
- or remove `knowledge_index/` and regenerate

### 11.7 Missing `knowledge/` or empty knowledge base
If preflight fails with `Missing knowledge directory` or `No supported knowledge files found`:
- create `knowledge/` at the repository root
- add at least one `.txt`, `.md`, or `.pdf` file under `knowledge/`
- rerun generation (optionally set `tools.retrieval.index.rebuild: true` for one run after major document changes)

### 11.8 Hugging Face auth error while loading embeddings
If you see model download/auth errors:
- keep the default `tools.retrieval.embeddings.model: sentence-transformers/all-MiniLM-L6-v2` (no HF token required in standard setups)
- if you switch to a gated/private HF embedding model, set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in `.env`

---

If you need a strict production profile, keep these defaults:
- `judge.mode: online`
- `judge.granularity: conversation` (budget-friendly)
- `batch_size` tuned to provider throughput
- `question_dedup_retries` >= 3
