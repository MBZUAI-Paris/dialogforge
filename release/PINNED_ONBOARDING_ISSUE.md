# Welcome to DialogForge: First Contribution Onboarding

Thanks for checking out DialogForge.

If you are new to the project, start here.

## First steps

1. Read `README.md` and `CONTRIBUTING.md`.
2. Set up a local environment:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```
3. Run baseline checks:
   ```bash
   uv run env PYTHONPATH=src pytest -q
   uv run env PYTHONPATH=src python -m dlgforge --help
   ```

## Good first tasks

- improve docs clarity and examples
- add tests for edge cases in Hugging Face export flow
- improve error messages and troubleshooting guidance

## Contribution expectations

- keep PRs small and focused
- include tests for behavior changes
- update docs for user-facing changes

## Need help?

See `SUPPORT.md` for support boundaries and reporting guidance.
