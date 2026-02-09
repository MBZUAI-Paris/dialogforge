# Contributing to DialogForge

## Scope
Contributions are welcome for:
- bug fixes and reliability improvements
- documentation and developer experience improvements
- test coverage and CI hardening
- performance and usability improvements

## Development setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Optional managed vLLM extra (Linux):
```bash
python -m pip install -e ".[vllm]"
```

## Local validation
Run before opening a PR:
```bash
uv run env PYTHONPATH=src pytest -q
uv run env PYTHONPATH=src python -m dlgforge --help
uv run env PYTHONPATH=src python -m dlgforge push --help
```

If changing HF export behavior, also run:
```bash
uv run env PYTHONPATH=src pytest -q tests/test_hf_push_dataset_card.py tests/test_hf_push_sanitization.py
```

## Documentation requirements
- Update docs for user-facing behavior changes.
- Keep references consistent with the docs hub in `docs/README.md`.
- Preserve documented CLI/config/output contracts in `v0.1.x` unless intentionally versioned.

## Pull request expectations
- clear problem statement and solution summary
- linked issues (if applicable)
- tests covering behavior changes
- notes on compatibility and operational impact

## Conduct
By contributing, you agree to `CODE_OF_CONDUCT.md`.
