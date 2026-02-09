from __future__ import annotations

from pathlib import Path


def test_no_generated_docstring_boilerplate_in_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src_root = root / "src" / "dlgforge"

    markers = [
        "Notes/Assumptions:",
        "Parameter consumed by",
        "Return value produced by",
        "Preconditions/Invariant",
        "Main flows:",
        "Expected usage:",
    ]

    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in text:
                offenders.append(f"{path.relative_to(root)} :: {marker}")

    assert not offenders, "Found generated docstring boilerplate markers:\n" + "\n".join(offenders)
