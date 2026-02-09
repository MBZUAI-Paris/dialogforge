# Getting Started

Launch DialogForge in a few minutes with a local config and one run command.

## 1. Install and set up
- Follow [Setup / Installation](setup-installation.md).
- Copy environment variables:
  ```bash
  cp .env.example .env
  ```

## 2. Run generation
```bash
dlgforge run config.yaml
```

Equivalent commands:
```bash
uv run dlgforge run config.yaml
python -m dlgforge run config.yaml
```

## 3. Check outputs
- `outputs/conversations/*.json`
- `outputs/conversations_sharegpt.jsonl`
- `outputs/turns.jsonl`
- `outputs/run_state/*.json`
- `logs/run.log`
- `logs/llm.log`
- `logs/judge.log`

## Next
- [Setup / Installation](setup-installation.md)
- [Configuration](CONFIG_REFERENCE.md)
- [CLI / Usage](CLI_REFERENCE.md)
