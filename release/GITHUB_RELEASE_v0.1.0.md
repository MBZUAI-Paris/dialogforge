# DialogForge v0.1.0

DialogForge v0.1.0 is the first public release.

This release is maintained by **MBZUAI-IFM Paris Lab**.

## Included

- CLI workflows for generation, judge-only runs, Hugging Face export/push, and seed migration.
- Async multi-turn generation with deterministic dedup and resumable run state.
- HF dataset export with dataset card generation, JSONL sanitization, optional stats, and SVG plots.
- Baseline tests for distributed bootstrap/provisioning, LLM client behavior, packaging, and HF export behavior.

## Compatibility

- Stable in v0.1.x: documented CLI commands and documented config keys.
- Not yet stable: internal Python module paths under `src/dlgforge`.

## Assets

- Source code
- PyPI package
- Sanitized sample dataset on Hugging Face
