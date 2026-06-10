from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.milestones import list_milestones, run_milestones


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/run_milestone.py")
    parser.add_argument(
        "selection",
        nargs="?",
        default="list",
        help="Milestone to run: m0, m1, ..., m9, m1-m9, m0-m9, all, or list.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where milestone artifacts should be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    selection = args.selection.lower()

    if selection == "list":
        for entry in list_milestones():
            print(f"{entry.milestone_id}\t{entry.status}\t{entry.artifact_dir_name}\t{entry.title}")
        return 0

    results = run_milestones(args.output_dir, selection=selection)
    for result in results:
        print(result.milestone_id)
        print(result.artifact_dir)
        if result.report_path is not None:
            print(result.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
