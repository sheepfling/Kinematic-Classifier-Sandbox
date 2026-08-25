"""Evaluate Product 4 real-world corpus promotion gates."""

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
    evaluate_product4_gates,
    load_snapshot_episodes,
    load_snapshot_manifest,
    load_source_registry,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/audit/evaluate_product4_gates.py",
        description=(
            "Evaluate Product 4 registry, snapshot, quality, rights, leakage, "
            "and classifier gates."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
        help="Source registry YAML path.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Optional snapshot manifest JSON path; episode manifests are loaded beside it.",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        help="Optional JSON list of EpisodeSplitAssignment objects.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return a non-zero status when the composed Product 4 gate is blocked.",
    )
    parser.add_argument(
        "--expected-lane",
        action="append",
        choices=REAL_WORLD_CORPUS_LANES,
        help="Expected lane for a task-scoped gate; repeat to require multiple lanes.",
    )
    parser.add_argument(
        "--grouping-namespace",
        action="append",
        choices=tuple(namespace.value for namespace in GroupingNamespace),
        help="Grouping namespace used by the leakage audit; repeat to declare the policy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = load_source_registry(args.registry)
    snapshot = None
    episodes = ()
    if args.snapshot is not None:
        snapshot = load_snapshot_manifest(args.snapshot)
        episodes = load_snapshot_episodes(snapshot, args.snapshot)

    assignments = ()
    if args.assignments is not None:
        payload = json.loads(args.assignments.read_text(encoding="utf-8"))
        assignments = tuple(EpisodeSplitAssignment.model_validate(row) for row in payload)

    report = evaluate_product4_gates(
        registry,
        snapshot=snapshot,
        episodes=episodes,
        assignments=assignments,
        **(
            {"expected_lanes": tuple(args.expected_lane)}
            if args.expected_lane
            else {}
        ),
        **(
            {
                "split_grouping_namespaces": tuple(
                    GroupingNamespace(value) for value in args.grouping_namespace
                )
            }
            if args.grouping_namespace
            else {}
        ),
    )
    print(report.model_dump_json(indent=2))
    return int(args.require_pass and not report.passes)


if __name__ == "__main__":
    raise SystemExit(main())
