"""Evaluate Product 4 gates independently for each real-world corpus lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (  # noqa: E402
    GroupingNamespace,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    REAL_WORLD_CORPUS_LANES,
    EpisodeSplitAssignment,
    load_snapshot_episodes,
    load_snapshot_manifest,
    load_source_registry,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio_matrix import (  # noqa: E402
    evaluate_product4_lane_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each Product 4 lane independently, then report whether the full "
            "cross-domain claim passes. Outputs may be written outside the repository."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
    )
    parser.add_argument(
        "--expected-lane",
        action="append",
        choices=REAL_WORLD_CORPUS_LANES,
        help="Restrict the matrix; repeat to evaluate a selected lane set.",
    )
    parser.add_argument(
        "--grouping-namespace",
        action="append",
        choices=tuple(namespace.value for namespace in GroupingNamespace),
        help="Grouping namespace used by every lane leakage audit; repeat for a policy.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return nonzero unless every expected lane passes its composed gate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    registry = load_source_registry(arguments.registry)
    snapshot = load_snapshot_manifest(arguments.snapshot)
    episodes = load_snapshot_episodes(snapshot, arguments.snapshot)
    assignments = tuple(
        EpisodeSplitAssignment.model_validate(row)
        for row in json.loads(arguments.assignments.read_text(encoding="utf-8"))
    )
    expected_lanes = tuple(arguments.expected_lane or REAL_WORLD_CORPUS_LANES)
    grouping_namespaces = tuple(
        GroupingNamespace(value)
        for value in (
            arguments.grouping_namespace
            or [
                GroupingNamespace.PHYSICAL_PLATFORM.value,
                GroupingNamespace.SOURCE_RECORDING.value,
                GroupingNamespace.MISSION_EVENT.value,
            ]
        )
    )
    report = evaluate_product4_lane_matrix(
        registry,
        snapshot=snapshot,
        episodes=episodes,
        assignments=assignments,
        expected_lanes=expected_lanes,
        split_grouping_namespaces=grouping_namespaces,
    )
    payload = report.model_dump_json(indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(arguments.require_pass and not report.all_lanes_pass)


if __name__ == "__main__":
    raise SystemExit(main())
