# Installation

Use the canonical runbook in `README.md` for the full step-by-step flow:
- environment setup
- configuration
- generation
- online judge verification
- async batch and dedup behavior
- troubleshooting

Quick install summary:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

Then continue with:

```bash
dlgforge run config.yaml
```

For full operational guidance, read `README.md`.
