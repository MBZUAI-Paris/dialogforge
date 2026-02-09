# Hugging Face Export

## Entry points
Export/push behavior lives in `src/dlgforge/pipeline/hf_push.py`:
- `run_push(config_path, options)` for explicit CLI push flow.
- `maybe_auto_push_after_run(cfg, output_paths)` for automatic post-run export/push.

## Export phases
1. Resolve effective source/export directories.
2. Prepare export bundle from generated artifacts.
3. Sanitize rows for Hub compatibility.
4. Generate dataset card (and optionally stats/plots).
5. Optionally push to Hub via `huggingface_hub` API.

## Key configuration
`saving.hf_push.*` controls:
- enable flags (`enabled`, `auto_push_on_run`)
- destination (`repo_id`, `repo_type`, `private`)
- source/export layout (`source_file`, `export_dir`, `include_run_state`)
- metadata and cleanup (`commit_message`, `clean_remote`)
- analytics outputs (`generate_stats`, `stats_file`, `generate_plots`, `plots_dir`)

## CLI options override
`dlgforge push` options can override config for one-off runs:
- `--repo-id`, `--repo-type`, `--source-dir`, `--export-dir`
- `--include-run-state`, `--token`, `--commit-message`
- `--no-export`, `--no-push`, `--clean-remote`

## Sanitization behavior
Export preparation normalizes content to avoid downstream schema inconsistencies, including:
- reasoning-trace shape normalization
- null/list normalization for retrieval fields
- thinking trace normalization to text-friendly structures

These behaviors are covered by tests in `tests/test_hf_push_sanitization.py`.

## Generated extra assets
When enabled, export can include:
- dataset stats JSON summary
- SVG plots under configured plot directory
- dataset card with source file and metadata

## Operational recommendations
- Use `--no-push` first to validate export bundle locally.
- Keep source file aligned with desired judge granularity (`conversations_sharegpt_judged.jsonl` for judged exports).
- Use explicit token input in CI where environment setup may vary.
