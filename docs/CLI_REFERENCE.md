# CLI Reference

## Command surface
`dlgforge` exposes four commands:
- `run`: execute generation pipeline.
- `judge`: run judge-only pass on existing conversations.
- `push`: prepare export and optionally push to Hugging Face Hub.
- `seeds-migrate`: migrate legacy seed topic files into structured YAML format.

## `dlgforge run <config>`
Runs generation from YAML config.

Behavior summary:
- validates config path.
- sets up logging and dotenv loading.
- resolves config and optional distributed bootstrap.
- runs generation and persistence.

Exit behavior:
- exits non-zero with `dlgforge run failed: <error>` on unhandled runtime errors.

## `dlgforge judge <config>`
Runs judge logic over existing conversation artifacts.

Use when:
- conversations were generated without online judge.
- judge prompt/model settings changed and you need a re-pass.

Exit behavior:
- exits non-zero with `dlgforge judge failed: <error>` on failure.

## `dlgforge push <config> [options]`
Exports dataset artifacts and optionally pushes to the hub.

Key options:
- `--repo-id`
- `--repo-type {dataset,model,space}`
- `--source-dir`
- `--export-dir`
- `--include-run-state`
- `--token`
- `--commit-message`
- `--no-export`
- `--no-push`
- `--clean-remote`

Exit behavior:
- exits non-zero with `dlgforge push failed: <error>` on failure.

## `dlgforge seeds-migrate <config> [options]`
Migrates legacy seed topics into scalable YAML schema.

Options:
- `--source-file`
- `--dest-file`
- `--overwrite`

Exit behavior:
- exits non-zero with `dlgforge seeds-migrate failed: <error>` on failure.

## Invocation forms
Equivalent entry forms:
```bash
dlgforge run config.yaml
uv run dlgforge run config.yaml
python -m dlgforge run config.yaml
```

## Operator notes
- In distributed mode, `run` may bootstrap Ray/Postgres/vLLM lifecycle before generation starts.
- `push --no-push` is useful for local export validation in CI/release pipelines.
