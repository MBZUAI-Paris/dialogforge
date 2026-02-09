# Contributing, Issues, and Pull Requests

We welcome bug fixes, reliability improvements, docs updates, test coverage improvements, and developer experience work.

## Contribute code or docs
- Read the project contribution guide: [CONTRIBUTING.md](https://github.com/MBZUAI-Paris/dialogforge/blob/main/CONTRIBUTING.md)
- Set up locally:
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install -e .
  ```
- Run validation before opening a PR:
  ```bash
  uv run env PYTHONPATH=src pytest -q
  uv run env PYTHONPATH=src python -m dlgforge --help
  uv run env PYTHONPATH=src python -m dlgforge push --help
  ```

## File an issue
- Bug report: [Open bug report](https://github.com/MBZUAI-Paris/dialogforge/issues/new?template=bug_report.yml)
- Feature request: [Open feature request](https://github.com/MBZUAI-Paris/dialogforge/issues/new?template=feature_request.yml)
- Onboarding/support request: [Open onboarding request](https://github.com/MBZUAI-Paris/dialogforge/issues/new?template=onboarding.md)

For security vulnerabilities, do not open a public issue. Use the private reporting process in the [Security policy](https://github.com/MBZUAI-Paris/dialogforge/security/policy).

## Open a pull request
- Use the project PR template: [PULL_REQUEST_TEMPLATE.md](https://github.com/MBZUAI-Paris/dialogforge/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
- Include:
  - a clear summary of what changed and why
  - linked issues (if applicable)
  - tests for behavior changes
  - notes on compatibility/operational impact

## Community policies
- [Code of Conduct](https://github.com/MBZUAI-Paris/dialogforge/blob/main/CODE_OF_CONDUCT.md)
- [Security policy](https://github.com/MBZUAI-Paris/dialogforge/security/policy)
- [Support policy](https://github.com/MBZUAI-Paris/dialogforge/blob/main/SUPPORT.md)
