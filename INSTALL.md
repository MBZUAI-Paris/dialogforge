# Installation (UV)

## 1) Create and activate environment

```bash
uv venv
source .venv/bin/activate
```

## 2) Install dlgforge

```bash
uv pip install -e .
```

## 3) Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (unless you set `llm.model` in `config.yaml`)

Optional:

- `OPENAI_BASE_URL` for OpenAI-compatible servers
- `SERPER_API_KEY` for optional web search
- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) for dataset push to Hugging Face Hub
- `KB_EMBEDDING_BACKEND` (`sentence_transformers` or `transformers`)
- `KB_EMBEDDING_MODEL_KWARGS_JSON` (JSON object)
- `KB_EMBEDDING_TOKENIZER_KWARGS_JSON` (JSON object)
- `KB_EMBEDDING_ENCODE_KWARGS_JSON` (JSON object)

## 4) Run

```bash
dlgforge run config.yaml
```

Alternative without shell activation:

```bash
uv run dlgforge run config.yaml
```

## 5) Alternative module entrypoint

```bash
python -m dlgforge run config.yaml
```

## 6) Judge-only pass (after generation)

Set in `config.yaml`:
- `judge.mode: online`
- `judge.enabled: true`

Then run:

```bash
dlgforge judge config.yaml
```

## 7) Export / push to Hugging Face Hub

Configure `saving.hf_push` in `config.yaml`, then run:

```bash
dlgforge push config.yaml
```

Useful variants:

```bash
# Prepare export bundle only
dlgforge push config.yaml --no-push

# Override destination for this push only
dlgforge push config.yaml --repo-id org/dataset-name --repo-type dataset
```

Optional auto-push at end of generation:
- `saving.hf_push.enabled: true`
- `saving.hf_push.auto_push_on_run: true`

If judged source file is missing/empty, auto-push is skipped with a warning.

## 8) Seed Topics YAML Migration

Seed topics are now YAML-first with single-file layout (`run.seed_topics_path`):

```yaml
run:
  seed_topics_path: data/seeds/topics.yaml
  seed_topics_variant: ""
```

Migrate existing legacy `seed_topics.json` into YAML:

```bash
dlgforge seeds-migrate config.yaml --source-file seed_topics.json --dest-file data/seeds/topics.yaml --overwrite
```

## Troubleshooting

### `dlgforge: command not found`
- Ensure your virtual environment is active.
- Reinstall package: `uv pip install -e .`

### Missing model error
- Set one of:
  - `llm.model` in `config.yaml`
  - `LLM_MODEL` in `.env`
  - `OPENAI_MODEL` in `.env`

### Embedding backend error
- Recreate venv and reinstall dependencies:
  - `rm -rf .venv`
  - `uv venv && source .venv/bin/activate`
  - `uv pip install -e .`
- First run may download embedding models from Hugging Face.
- If your environment has restricted internet, pre-download models or set:
  - `models.embedding_model` to a local model path.
- Default backend is `retrieval.embedding_backend: sentence_transformers`.
- Optional backend is `retrieval.embedding_backend: transformers`.
- No automatic fallback model is used.

### Embedding config conflict in persisted Chroma index
- Symptom: runtime says persisted collection embedding configuration conflicts with current settings.
- Cause: `knowledge_index` was created with a different embedding-function setup.
- One-time recovery:
  - set `retrieval.rebuild_index: true` and run once, then set it back to `false`, or
  - delete `knowledge_index/` and rerun.
- Note: `Qwen/Qwen3-Embedding-4B` is inherently heavy during initial load/indexing.

### Web search key missing
- If `tools.web_search_enabled: true`, web search calls require `SERPER_API_KEY`.
- Runs can still proceed if the model never invokes `web_search`.

### Live progress logs
- Use `tail -f logs/run.log` during generation.
- Detailed components:
  - `logs/retrieval.log`
  - `logs/llm.log`
  - `logs/tools.log`
  - `logs/judge.log`
