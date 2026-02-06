# dlgforge

Minimal Python package for generating multi-turn synthetic dialogues between a User agent and an Assistant agent using local KB retrieval (RAG) and optional web search.

- Direct OpenAI-compatible API orchestration.
- No CrewAI, no LiteLLM, no FastAPI.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
dlgforge run config.yaml
```

Equivalent:

```bash
python -m dlgforge run config.yaml
```

Judge existing generated conversations (no regeneration):

```bash
dlgforge judge config.yaml
```

Export and push dataset artifacts to Hugging Face Hub:

```bash
dlgforge push config.yaml
```

Migrate legacy seed topics into scalable YAML:

```bash
dlgforge seeds-migrate config.yaml --source-file seed_topics.json --dest-file data/seeds/topics.yaml --overwrite
```

For judge-only mode, set:
- `judge.mode: online`
- `judge.enabled: true`

## Python API

```python
import dlgforge

dlgforge.run("config.yaml")
```

## `.env` minimum

Set at least:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (or set `llm.model` in `config.yaml`)

Optional:

- `OPENAI_BASE_URL` for OpenAI-compatible endpoints
- `SERPER_API_KEY` for optional `web_search` calls
- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) for `dlgforge push` / auto-push
- `SEED_TOPICS_VARIANT` to choose a seed question variant file (e.g. `morocco`, `msa`, `english`)
- `KB_EMBEDDING_BACKEND` (`sentence_transformers` or `transformers`)
- `KB_EMBEDDING_MODEL_KWARGS_JSON` (JSON object)
- `KB_EMBEDDING_TOKENIZER_KWARGS_JSON` (JSON object)
- `KB_EMBEDDING_ENCODE_KWARGS_JSON` (JSON object)

## Model fallback chain

If no agent-specific model is set, model resolution is:

1. `llm.agents.<agent>.model`
2. `llm.model`
3. `LLM_MODEL`
4. `OPENAI_MODEL`

## Project layout

```text
src/dlgforge/
  cli.py
  config/
  io/
  llm/
  pipeline/
  tools/
  utils/
  prompts/
```

## Outputs

Artifacts are written under `outputs/`:

- `outputs/synthetic_qa.jsonl`
- `outputs/coverage_ledger.jsonl`
- `outputs/turns.jsonl`
- `outputs/conversations_sharegpt.jsonl`
- `outputs/conversations_sharegpt_judged.jsonl`
- `outputs/conversations/<conversation_id>.json`
- `outputs/run_state/<run_id>.json`
- `outputs/hf_export/` (when pushing/exporting to HF Hub)

## Hugging Face Push

Configure under `saving.hf_push` in `config.yaml`:

```yaml
saving:
  output_dir: outputs
  hf_push:
    enabled: false
    auto_push_on_run: false
    repo_id: MBZUAI-Paris/Synth-QA-Reasoning-Multilingual
    repo_type: dataset
    export_dir: hf_export
    include_run_state: false
    private: true
    commit_message: Update synthetic dataset export
    source_file: conversations_sharegpt_judged.jsonl
```

Manual push command (config defaults + optional overrides):

```bash
dlgforge push config.yaml
dlgforge push config.yaml --no-push
dlgforge push config.yaml --repo-id org/override-dataset --repo-type dataset
```

Auto-push:
- Runs at the end of `dlgforge run` when both:
  - `saving.hf_push.enabled: true`
  - `saving.hf_push.auto_push_on_run: true`
- If `source_file` is missing/empty, push is skipped with a warning.

## Seed Topics Layout

Default seed topics path is now a YAML file:

```yaml
run:
  seed_topics_path: data/seeds/topics.yaml
  seed_topics_variant: ""
```

Scalable YAML structure (single file):

```yaml
version: 1
defaults:
  by_target_language:
    ar: msa
    en: english
    fr: france
  final_fallback: english
aliases:
  ar: msa
  en: english
  fr: france
  arabic: msa
  french: france
variants:
  english:
    Topic Name:
      - question 1
      - question 2
  morocco:
    Topic Name:
      - سؤال 1
```

Variant resolution order:
1. `run.seed_topics_variant` (or `SEED_TOPICS_VARIANT`)
2. `target_language` alias/default from `data/seeds/topics.yaml`
3. hard defaults (`ar -> msa`, `en -> english`, `fr -> france`)
4. final fallback: `english`

Legacy compatibility:
- `run.seed_topics_path` can still point to a legacy JSON/YAML file.
- Nested legacy format (`topic -> language -> [questions]`) is supported with a warning.

## Logs

Runtime logs are written under `logs/`:

- `logs/run.log`
- `logs/retrieval.log`
- `logs/llm.log`
- `logs/tools.log`
- `logs/judge.log`

## Common failures

Missing model:
- set `llm.model`, `LLM_MODEL`, or `OPENAI_MODEL`.

Missing API key:
- set `OPENAI_API_KEY` (or per-agent `LLM_<AGENT>_API_KEY`).

Embedding/model download issues:
- reinstall in a clean uv venv and set a reachable model path/name in:
  - `models.embedding_model`
- default backend is `retrieval.embedding_backend: sentence_transformers`
- optional backend: `retrieval.embedding_backend: transformers`
- there is no automatic fallback to MiniLM; initialization errors fail fast with details

Persisted Chroma embedding config conflict:
- This happens when `knowledge_index` was built with a different embedding-function config.
- Recovery (one-time):
  - set `retrieval.rebuild_index: true` and run once, then set it back to `false`, or
  - delete `knowledge_index/` and rerun.
- Note: `Qwen/Qwen3-Embedding-4B` is inherently heavy on CPU/GPU/memory during load/indexing.

Full installation guide: `INSTALL.md`.
