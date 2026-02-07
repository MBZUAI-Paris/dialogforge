from __future__ import annotations

from pathlib import Path


def test_pyproject_has_optional_vllm_extra() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "[project.optional-dependencies]" in pyproject
    assert "vllm" in pyproject
    assert "platform_system == 'Linux'" in pyproject
