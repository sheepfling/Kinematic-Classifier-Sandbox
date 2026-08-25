"""Evaluate a prepared Product 4 classifier selection through Product 2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.common_experiment.runner import (  # noqa: E402
    analyze_common_trajectory_corpus,
    write_common_experiment_artifacts,
)
from kinematic_classifier_sandbox.corpus.real_world.analysis_products import (  # noqa: E402
    AnalysisProductManifest,
)
from kinematic_classifier_sandbox.corpus.real_world.classifier_bridge import (  # noqa: E402
    RealWorldBridgeConfig,
    build_empirical_product2_hooks,
    build_real_world_bridge_selection,
)
from kinematic_classifier_sandbox.corpus.real_world.common_front_utils import (  # noqa: E402
    sha256_file,
)
from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (  # noqa: E402
    GroupingNamespace,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    REAL_WORLD_CORPUS_LANES,
    EpisodeSplitAssignment,
    SnapshotSplit,
    evaluate_product4_gates,
    load_snapshot_episodes,
    load_snapshot_manifest,
    load_source_registry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one prepared Product 4 classifier-ladder selection through the "
            "existing Product 2 common experiment. Outputs remain external."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
    )
    parser.add_argument(
        "--expected-lane",
        action="append",
        choices=REAL_WORLD_CORPUS_LANES,
        required=True,
    )
    parser.add_argument("--target-label-namespace", default="platform_class")
    parser.add_argument("--pair-id", default="passenger_car_vs_truck")
    parser.add_argument("--class-a", default="passenger_car")
    parser.add_argument("--class-b", default="truck")
    parser.add_argument(
        "--evaluation-split",
        choices=tuple(split.value for split in SnapshotSplit),
        default=SnapshotSplit.TEST.value,
    )
    parser.add_argument(
        "--grouping-namespace",
        action="append",
        choices=tuple(namespace.value for namespace in GroupingNamespace),
        default=None,
    )
    parser.add_argument("--measurement-sigma-floor", type=float, default=0.05)
    parser.add_argument(
        "--max-episodes-per-group",
        type=int,
        default=1,
        help="Maximum Product 2 examples retained per physical platform and split.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "experiments/common_1d_classifier_study/common_experiment_config.yaml",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    registry = load_source_registry(arguments.registry)
    snapshot = load_snapshot_manifest(arguments.snapshot)
    snapshot_root = arguments.snapshot.parent
    episodes = load_snapshot_episodes(snapshot, arguments.snapshot)
    analysis_manifest = AnalysisProductManifest.model_validate_json(
        arguments.analysis_manifest.read_text(encoding="utf-8")
    )
    if analysis_manifest.snapshot_id != snapshot.snapshot_id:
        raise SystemExit("analysis manifest snapshot ID does not match snapshot")
    if analysis_manifest.snapshot_content_sha256 != snapshot.content_sha256():
        raise SystemExit("analysis manifest snapshot content hash does not match snapshot")
    if analysis_manifest.registry_id != registry.registry_id:
        raise SystemExit("analysis manifest registry ID does not match registry")
    assignments = tuple(
        EpisodeSplitAssignment.model_validate(row)
        for row in json.loads(arguments.assignments.read_text(encoding="utf-8"))
    )
    grouping_values = arguments.grouping_namespace or [
        GroupingNamespace.PHYSICAL_PLATFORM.value
    ]
    grouping_namespaces = tuple(GroupingNamespace(value) for value in grouping_values)
    config = RealWorldBridgeConfig(
        pair_id=arguments.pair_id,
        class_a=arguments.class_a,
        class_b=arguments.class_b,
        target_label_namespace=arguments.target_label_namespace,
        evaluation_split=SnapshotSplit(arguments.evaluation_split),
        grouping_namespaces=grouping_namespaces,
        measurement_sigma_floor=arguments.measurement_sigma_floor,
        max_episodes_per_group=arguments.max_episodes_per_group,
    )
    gate_report = evaluate_product4_gates(
        registry,
        snapshot=snapshot,
        episodes=episodes,
        assignments=assignments,
        expected_lanes=tuple(arguments.expected_lane),
        split_grouping_namespaces=grouping_namespaces,
    )
    selection = build_real_world_bridge_selection(
        snapshot_root=snapshot_root,
        analysis_manifest=analysis_manifest,
        episodes=episodes,
        assignments=assignments,
        config=config,
    )
    if not gate_report.passes:
        raise SystemExit(
            "Product 4 task gate is blocked:\n"
            + gate_report.model_dump_json(indent=2)
        )
    if not selection.grouping_audit_passes:
        raise SystemExit(
            "Product 2 bridge grouping audit is blocked:\n"
            + json.dumps(selection.grouping_audit_issues, indent=2)
        )
    reference_builder, measurement_sigma = build_empirical_product2_hooks(selection)
    trajectories = selection.trajectories(config.evaluation_split)
    result = analyze_common_trajectory_corpus(
        pair_specs=(selection.pair_spec,),
        trajectories=trajectories,
        config_path=arguments.experiment_config,
        trajectories_per_case=len(trajectories),
        include_comparison=False,
        reference_builder=reference_builder,
        measurement_sigma=measurement_sigma,
    )
    artifacts = write_common_experiment_artifacts(
        arguments.output_dir,
        result=result,
    )
    bridge_report_path = arguments.output_dir / "bridge-report.json"
    bridge_report_path.write_text(
        json.dumps(
            {
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_content_sha256": snapshot.content_sha256(),
                "analysis_manifest_sha256": sha256_file(arguments.analysis_manifest),
                "target_label_namespace": config.target_label_namespace,
                "pair_id": config.pair_id,
                "evaluation_split": config.evaluation_split.value,
                "grouping_namespaces": [namespace.value for namespace in grouping_namespaces],
                "max_episodes_per_group": config.max_episodes_per_group,
                "grouping_audit_passes": selection.grouping_audit_passes,
                "grouping_audit_issues": list(selection.grouping_audit_issues),
                "class_counts_by_split": {
                    split.value: dict(selection.class_counts(split)) for split in SnapshotSplit
                },
                "episode_count_by_split": {
                    split.value: len(selection.trajectories(split)) for split in SnapshotSplit
                },
                "product4_task_gate": gate_report.model_dump(mode="json"),
                "product2_summary": asdict(result.summary),
                "claim_boundary": (
                    "Bounded held-out evaluation within the selected "
                    f"{', '.join(arguments.expected_lane)} snapshot for target namespace "
                    f"{config.target_label_namespace!r}; no source-shift, "
                    "population-representative, or six-lane claim."
                ),
                "artifact_root": str(artifacts.run_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(bridge_report_path)
    print(artifacts.run_dir)
    print(f"evaluation_split={config.evaluation_split.value}")
    print(f"evaluation_episode_count={len(trajectories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
