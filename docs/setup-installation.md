# Setup / Installation

Install DialogForge in editable mode and validate the CLI before your first run.

## Standard install
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

## Optional managed vLLM extra (Linux)
```bash
python -m pip install -e ".[vllm]"
```

## CLI smoke checks
```bash
uv run env PYTHONPATH=src python -m dlgforge --help
uv run env PYTHONPATH=src python -m dlgforge push --help
```

## Optional test baseline
```bash
uv run env PYTHONPATH=src pytest -q
```

## Next
- [Getting Started](getting-started.md)
- [Configuration](CONFIG_REFERENCE.md)
- [CLI / Usage](CLI_REFERENCE.md)
