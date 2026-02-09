# DialogForge Documentation

This documentation is organized for two primary audiences:
- Operators running synthetic data generation reliably.
- Contributors extending and maintaining the codebase.

## Start here
- [Architecture](ARCHITECTURE.md): runtime flow, module boundaries, and control paths.
- [Configuration Reference](CONFIG_REFERENCE.md): key-by-key config behavior and env override precedence.
- [CLI Reference](CLI_REFERENCE.md): command contracts and operational usage.

## Runtime operations
- [Distributed Runtime](DISTRIBUTED_RUNTIME.md): Ray bootstrap, Postgres requirements, vLLM attach/managed behavior.
- [Outputs and Schemas](OUTPUTS_AND_SCHEMAS.md): artifact files, record schema expectations, and resume semantics.
- [Hugging Face Export](HF_EXPORT.md): export preparation, sanitization, stats/plots, and push workflow.

## Pipeline internals
- [Retrieval and Prompts](RETRIEVAL_AND_PROMPTS.md): retrieval index lifecycle, tool calls, and prompt assets.
- [Curated API Index](API_INDEX.md): module-level API contracts and call relationships.

## Project entry documents
- [Repository README](../README.md)
- [Install Guide](../INSTALL.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Support Policy](../SUPPORT.md)

## Build docs locally
```bash
python -m pip install mkdocs mkdocs-material "mkdocstrings[python]"
PYTHONPATH=src python -m mkdocs serve
```

Production build:
```bash
PYTHONPATH=src python -m mkdocs build --strict
```
