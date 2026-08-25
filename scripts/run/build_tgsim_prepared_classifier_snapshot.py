"""Build an external task-scoped prepared TGSIM classifier snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim import (  # noqa: E402
    load_tgsim_foggy_bottom_csv,
)
from kinematic_classifier_sandbox.corpus.real_world.common_front_utils import (  # noqa: E402
    sha256_file,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    SourceEvidenceState,
    load_source_registry,
    write_snapshot_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.prepared_cohort import (  # noqa: E402
    PreparedClassifierCohortConfig,
    build_prepared_classifier_cohort,
)
from kinematic_classifier_sandbox.corpus.real_world.projection import (  # noqa: E402
    ProjectionKind,
)
from kinematic_classifier_sandbox.corpus.real_world.snapshot_builder import (  # noqa: E402
    build_snapshot_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.splits import (  # noqa: E402
    GroupSplitPolicy,
)
from kinematic_classifier_sandbox.corpus.real_world.windowing import (  # noqa: E402
    WindowingPolicy,
)

DEFAULT_DATASET_ID = "fhwa_tgsim_foggy_bottom_balanced_vehicle_cohort_v1"
DEFAULT_ARTIFACT_ID = "fhwa_tgsim_foggy_bottom_balanced_vehicle_cohort_v1_csv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a task-scoped, prepared TGSIM classifier snapshot outside the repository. "
            "The source registry must mark the selected source prepared."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
    )
    parser.add_argument("--source-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--source-artifact-id", default=DEFAULT_ARTIFACT_ID)
    parser.add_argument("--target-label-namespace", default="platform_class")
    parser.add_argument("--pair-id", default="passenger_car_vs_truck")
    parser.add_argument("--class-a", default="passenger_car")
    parser.add_argument("--class-b", default="truck")
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    parser.add_argument("--stride-fraction", type=float, default=0.5)
    parser.add_argument("--nominal-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--split-seed", default="tgsim-foggy-bottom-road-v1")
    parser.add_argument(
        "--projection",
        choices=tuple(kind.value for kind in ProjectionKind),
        default=ProjectionKind.SPEED_PROFILE.value,
    )
    parser.add_argument("--created-at", help="UTC ISO-8601 timestamp; defaults to current UTC time.")
    return parser


def _write_episode(root: Path, episode) -> Path:
    episodes_root = root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    path = episodes_root / f"{episode.episode_id}.json"
    path.write_text(episode.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    return path


def _write_assignments(root: Path, cohort) -> Path:
    path = root / "assignments.json"
    path.write_text(
        json.dumps(
            [assignment.model_dump(mode="json") for assignment in cohort.episode_assignments()],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    arguments = _parser().parse_args()
    registry = load_source_registry(arguments.registry)
    source_entry = registry.source(arguments.source_dataset_id)
    if source_entry.evidence_state is SourceEvidenceState.LEAD_ONLY or source_entry.evidence_state is SourceEvidenceState.ACCESS_VERIFIED:
        raise SystemExit(
            "selected source must be prepared before a classifier snapshot can be built: "
            f"{arguments.source_dataset_id} ({source_entry.evidence_state.value})"
        )
    artifact = next(
        (
            item
            for item in source_entry.artifacts
            if item.artifact_id == arguments.source_artifact_id
        ),
        None,
    )
    if artifact is None:
        raise SystemExit(
            f"source artifact is not declared by registry: {arguments.source_artifact_id}"
        )
    if artifact.sha256 is None:
        raise SystemExit(f"source artifact has no registry SHA-256: {arguments.source_artifact_id}")
    actual_sha256 = sha256_file(arguments.source)
    if actual_sha256 != artifact.sha256:
        raise SystemExit(
            "source SHA-256 does not match registry: "
            f"expected {artifact.sha256}, got {actual_sha256}"
        )

    created_at = (
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if arguments.created_at
        else datetime.now(UTC)
    )
    if created_at.tzinfo is None:
        raise SystemExit("--created-at must include a timezone")
    if not 0.0 < arguments.stride_fraction <= 1.0:
        raise SystemExit("--stride-fraction must be in (0, 1]")

    source_result = load_tgsim_foggy_bottom_csv(arguments.source)
    config = PreparedClassifierCohortConfig(
        corpus_snapshot_id=arguments.snapshot_id,
        dataset_id=arguments.source_dataset_id,
        native_dataset_id=source_result.manifest.dataset_id,
        source_artifact_id=arguments.source_artifact_id,
        target_label_namespace=arguments.target_label_namespace,
        pair_id=arguments.pair_id,
        class_a=arguments.class_a,
        class_b=arguments.class_b,
        projection_kind=ProjectionKind(arguments.projection),
        window_policy=WindowingPolicy(
            window_duration_s=arguments.window_duration_s,
            stride_s=arguments.window_duration_s * arguments.stride_fraction,
            nominal_sample_interval_s=arguments.nominal_sample_interval_s,
        ),
        split_policy=GroupSplitPolicy(seed=arguments.split_seed, stratify_by_class=True),
    )
    root = arguments.snapshot_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cohort = build_prepared_classifier_cohort(
        source_result.tracks,
        output_root=root,
        config=config,
    )
    episode_paths = tuple(_write_episode(root, episode) for episode in cohort.episodes)
    snapshot = build_snapshot_manifest(
        registry,
        episode_paths,
        snapshot_root=root,
        snapshot_id=arguments.snapshot_id,
        created_at=created_at,
        require_prepared_sources=True,
        notes=(
            "Task-scoped prepared classifier snapshot; raw source rows remain outside Git.",
            "Classifier assets contain only relative time and scalar measurements.",
            "The physical-platform holdout is bounded within one source recording; source-shift remains open.",
        ),
    )
    snapshot_path = root / "snapshot.json"
    write_snapshot_manifest(snapshot, snapshot_path)
    assignments_path = _write_assignments(root, cohort)
    cohort_manifest_path = root / "prepared-cohort.json"
    cohort_manifest_path.write_text(
        json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "source_sha256": actual_sha256,
                "track_count": len(cohort.tracks),
                "episode_count": len(cohort.episodes),
                "split_summary": [row.model_dump(mode="json") for row in cohort.split.summary_rows],
                "snapshot_content_sha256": snapshot.content_sha256(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(snapshot_path)
    print(snapshot.content_sha256())
    print(assignments_path)
    print(cohort_manifest_path)
    print(f"track_count={len(cohort.tracks)}")
    print(f"episode_count={len(cohort.episodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
