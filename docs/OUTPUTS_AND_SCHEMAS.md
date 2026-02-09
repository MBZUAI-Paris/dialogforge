# Outputs and Schemas

## Output layout
Artifacts are created under `saving.output_dir` using `OutputPaths`.

Primary files:
- `synthetic_qa.jsonl`
- `coverage_ledger.jsonl`
- `turns.jsonl`
- `conversations_sharegpt.jsonl`
- `conversations_sharegpt_judged.jsonl`
- `conversations/<conversation_id>.json`
- `conversations_index.jsonl`
- `run_state/<run_id>.json`
- `run_state/last_run_id.txt`

## Conversation dataset row (`synthetic_qa.jsonl`)
Typical fields:
- `conversation_id`, `timestamp`
- `question`, `inputs`
- `qa_generation_plan`, `kb_final_answer`, `qa_judge`, `conversation_judge`
- `turns`, `conversation_history`, `messages`
- `raw_result`, `raw_results`

## Conversation file (`conversations/*.json`)
Richer per-conversation artifact containing:
- personas and language metadata
- messages and tool-augmented messages
- user/assistant reasoning slices
- turn payloads and conversation-level judge
- raw per-turn outputs

## Turn dataset row (`turns.jsonl`)
Flattened per-turn row with:
- identifiers (`conversation_id`, `turn_index`, `timestamp`)
- `user_message`, `assistant_message`
- question mode and seed topic context
- judge score/reasons when available

## ShareGPT exports
- `conversations_sharegpt.jsonl`: baseline export.
- `conversations_sharegpt_judged.jsonl`: includes configurable columns for:
  - messages
  - messages with tools
  - metadata
  - user reasoning
  - assistant reasoning
  - judge payload

Column names are remappable via `saving.output_columns`.

## Coverage ledger
`coverage_ledger.jsonl` stores dedup/coverage memory used by sampling:
- question hashes
- topic/document usage
- seed-topic usage

This ledger is read at runtime to preserve diversity and avoid duplicate questions.

## Run state schema
Run state snapshots include:
- `run_id`, `status`, `updated_at`
- `inputs`
- `n_turns`
- per-turn state and raw outputs

Batched state additionally stores:
- `batch_size`
- per-slot conversation status (`active|completed|dropped`)
- `drop_reason` for exhausted dedup slots

## Resume semantics
- `run.resume_run_id` loads matching state file when present.
- Batched resume requires requested `batch_size` to match stored state batch size.
- Slot `inputs` and turn history are restored from state.
- Coverage and dedup memory are rebuilt from existing persisted artifacts.
