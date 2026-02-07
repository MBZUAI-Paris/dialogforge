from __future__ import annotations

from pathlib import Path

from dlgforge.pipeline.hf_push import _write_dataset_card


def test_dataset_card_pins_train_split_to_source_file(tmp_path: Path) -> None:
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
