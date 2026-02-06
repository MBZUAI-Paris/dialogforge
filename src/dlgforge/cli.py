from __future__ import annotations

import argparse

from dlgforge.pipeline.hf_push import HFPushOptions, run_push
from dlgforge.pipeline.runner import run, run_judge_only
from dlgforge.pipeline.seed_topics_migration import run_seeds_migrate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlgforge", description="Synthetic dialogue generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run generation from a YAML config")
    run_parser.add_argument("config", help="Path to config.yaml")

    judge_parser = subparsers.add_parser("judge", help="Run judge-only pass on existing conversations")
    judge_parser.add_argument("config", help="Path to config.yaml")

    push_parser = subparsers.add_parser("push", help="Export and push outputs to Hugging Face Hub")
    push_parser.add_argument("config", help="Path to config.yaml")
    push_parser.add_argument("--repo-id", default="", help="Repo ID to push to (e.g., org/dataset).")
    push_parser.add_argument(
        "--repo-type",
        default="",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type.",
    )
    push_parser.add_argument("--source-dir", default="", help="Override source directory.")
    push_parser.add_argument("--export-dir", default="", help="Override export directory.")
    push_parser.add_argument(
        "--include-run-state",
        action="store_true",
        help="Include run_state checkpoints in the export bundle.",
    )
    push_parser.add_argument(
        "--token",
        default=None,
        help="HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN env vars.",
    )
    push_parser.add_argument(
        "--commit-message",
        default="",
        help="Commit message for the Hub push.",
    )
    push_parser.add_argument(
        "--no-export",
        dest="prepare_export",
        action="store_false",
        help="Skip export preparation (push export-dir as-is).",
    )
    push_parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Skip pushing to the Hub (prepare export only).",
    )
    push_parser.add_argument(
        "--clean-remote",
        action="store_true",
        help="Delete remote repo files before uploading the local export folder.",
    )
    push_parser.set_defaults(prepare_export=True, push=True)

    seeds_parser = subparsers.add_parser(
        "seeds-migrate",
        help="Migrate legacy seed topics into scalable YAML format",
    )
    seeds_parser.add_argument("config", help="Path to config.yaml")
    seeds_parser.add_argument(
        "--source-file",
        default="",
        help="Legacy seed topics file (JSON/YAML). If omitted, falls back to seed_topics.json.",
    )
    seeds_parser.add_argument(
        "--dest-file",
        default="",
        help="Destination YAML file for migrated seed topics (default: data/seeds/topics.yaml).",
    )
    seeds_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        try:
            run(args.config)
        except Exception as err:
            raise SystemExit(f"dlgforge run failed: {err}") from None
        return

    if args.command == "judge":
        try:
            run_judge_only(args.config)
        except Exception as err:
            raise SystemExit(f"dlgforge judge failed: {err}") from None
        return

    if args.command == "push":
        try:
            run_push(
                args.config,
                HFPushOptions(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    source_dir=args.source_dir,
                    export_dir=args.export_dir,
                    include_run_state=args.include_run_state,
                    token=args.token,
                    commit_message=args.commit_message,
                    prepare_export=args.prepare_export,
                    push=args.push,
                    clean_remote=args.clean_remote,
                ),
            )
        except Exception as err:
            raise SystemExit(f"dlgforge push failed: {err}") from None
        return

    if args.command == "seeds-migrate":
        try:
            run_seeds_migrate(
                config_path=args.config,
                source_file=args.source_file,
                dest_file=args.dest_file,
                overwrite=args.overwrite,
            )
        except Exception as err:
            raise SystemExit(f"dlgforge seeds-migrate failed: {err}") from None
        return

    parser.error(f"Unknown command: {args.command}")
