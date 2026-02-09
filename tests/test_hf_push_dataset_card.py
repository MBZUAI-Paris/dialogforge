"""
Test module for DialogForge behavior validation.

Main flows:
- Defines unit/integration test cases that assert expected runtime behavior.

Expected usage:
- Run with `pytest`; import helpers from production modules as needed.
"""

from __future__ import annotations

from pathlib import Path

from dlgforge.pipeline.hf_push import _write_dataset_card


def test_dataset_card_pins_train_split_to_source_file(tmp_path: Path) -> None:
    """
    Test dataset card pins train split to source file.

    Args:
        tmp_path (Path): Parameter consumed by `test_dataset_card_pins_train_split_to_source_file`.

    Returns:
        None: Return value produced by `test_dataset_card_pins_train_split_to_source_file`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_hf_push_dataset_card.py` for concrete usage of `test_dataset_card_pins_train_split_to_source_file`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    export_dir = tmp_path / "export"
    source_dir = tmp_path / "source"
    export_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    source_file = "conversations_sharegpt_judged.jsonl"
    (source_dir / source_file).write_text('{"messages":[]}\n', encoding="utf-8")

    _write_dataset_card(
        export_dir=export_dir,
        source_dir=source_dir,
        source_file=source_file,
        repo_id="org/demo-dataset",
        dataset_stats={"records": 1},
        stats_file="dataset_stats.json",
        plot_files=None,
    )

    readme = (export_dir / "README.md").read_text(encoding="utf-8")
    assert "configs:" in readme
    assert "config_name: default" in readme
    assert "split: train" in readme
    assert f"path: {source_file}" in readme
