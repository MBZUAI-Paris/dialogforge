# Installation

## Standard install
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

Optional managed vLLM extra (Linux):
```bash
python -m pip install -e ".[vllm]"
```

## CLI smoke checks
```bash
uv run env PYTHONPATH=src python -m dlgforge --help
uv run env PYTHONPATH=src python -m dlgforge push --help
```

## Test baseline
```bash
uv run env PYTHONPATH=src pytest -q
```

## Next steps
- Read the [documentation hub](docs/README.md).
- Start with `dlgforge run config.yaml`.
