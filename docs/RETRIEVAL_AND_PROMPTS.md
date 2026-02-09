# Retrieval and Prompts

## Retrieval subsystem
`src/dlgforge/tools/retrieval.py` manages a Chroma-backed knowledge store.

Key behaviors:
- indexes `.txt`, `.md`, and `.pdf` files recursively from the knowledge directory.
- chunks content by configurable `chunk_size` and `overlap`.
- persists index data when `persist_dir` is configured.
- can skip expensive re-indexing via fingerprint/cache checks when unchanged.

## Query path
Assistant tool calls can invoke:
- `vector_db_search(query, k, use_reranker)`
- optional reranking depending on model settings

Returned payload includes rendered evidence snippets and metadata suitable for assistant grounding traces.

## Coverage-aware sampling
`src/dlgforge/pipeline/sampling.py` uses retrieval metadata and coverage ledger memory to:
- choose question mode (`followup`, `adjacent`, `fresh`, `off_topic`, `seeded`)
- sample topic snippets with source-balancing heuristics
- avoid recently overused documents/questions
- update coverage and seed-topic memories after accepted turns

## Prompt assets
Prompt files under `src/dlgforge/prompts/`:
- `agents.yaml`: role/goal/backstory per logical agent
- `tasks.yaml`: task descriptions and expected JSON output schema
- `personas.yaml`: user/assistant persona pools

Prompt assembly:
- `build_agent_system_prompt(agent_key)`
- `build_task_prompt(task_key, values)`

Templates are rendered with runtime values and enforced as JSON-only contracts in model calls.

## Tool-calling behavior
In assistant stage (`runner.py`):
- tool schema always includes KB retrieval.
- web search tool is added only when `tools.web_search_enabled=true`.
- tool calls are executed and appended back to chat messages before final assistant JSON is parsed.

## Practical tuning knobs
- retrieval depth: `retrieval.default_k`
- dedup retries: `coverage.question_dedup_retries`
- reranker path: `models.use_reranker` and related model/batch settings
- language-aware seed topics: `run.seed_topics_*`
