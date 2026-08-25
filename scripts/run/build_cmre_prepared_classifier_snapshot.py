"""Build an external task-scoped prepared CMRE route-pair snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.corpus.real_world.adapters.cmre_route_tracklets import (  # noqa: E402
    build_fixture,
    parse_tracklets,
    sha256_file,
    write_fixture_index,
)
from kinematic_classifier_sandbox.corpus.real_world.common_front_utils import (  # noqa: E402
    write_json_asset,
)
from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (  # noqa: E402
    GroupingNamespace,
    TrajectoryEpisodeManifest,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    EpisodeSplitAssignment,
    SnapshotSplit,
    SourceEvidenceState,
    assign_grouped_snapshot_splits,
    audit_split_assignments,
    load_source_registry,
    write_snapshot_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.snapshot_builder import (  # noqa: E402
    build_snapshot_manifest,
)

DEFAULT_DATASET_ID = "cmre_brest_maritime_routes_tracklets_v1_0"
DEFAULT_SOURCE_ARTIFACT_ID = "cmre_brest_upstream_tracklets_v2"
DEFAULT_NOMENCLATURE_ARTIFACT_ID = "cmre_brest_external_validation_route_nomenclature_v2"
TASK_PROJECTION_STEP_ID = "cmre-route-pair-speed-profile-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bounded CMRE route-pair Product 4 classifier snapshot outside the "
            "repository. The target namespace remains route, not vessel family."
        )
    )
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--nomenclature", type=Path, required=True)
    parser.add_argument("--identity-key", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
    )
    parser.add_argument("--source-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--source-artifact-id", default=DEFAULT_SOURCE_ARTIFACT_ID)
    parser.add_argument(
        "--nomenclature-artifact-id",
        default=DEFAULT_NOMENCLATURE_ARTIFACT_ID,
    )
    parser.add_argument("--route-a", default="R_06")
    parser.add_argument("--route-b", default="R_14")
    parser.add_argument("--split-seed", default="cmre-route-pair-r06-r14-v1")
    parser.add_argument("--created-at", help="UTC ISO-8601 timestamp; defaults to current UTC time.")
    return parser


def _write_episode(root: Path, episode: TrajectoryEpisodeManifest) -> Path:
    path = root / "episodes" / f"{episode.episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(episode.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    return path


def _registry_artifact(registry, dataset_id: str, artifact_id: str):
    source = registry.source(dataset_id)
    for artifact in source.artifacts:
        if artifact.artifact_id == artifact_id:
            return artifact
    raise SystemExit(f"source artifact is not declared by registry: {artifact_id}")


def _verify_artifact(path: Path, artifact, *, description: str) -> str:
    if artifact.sha256 is None:
        raise SystemExit(f"{description} has no registry SHA-256: {artifact.artifact_id}")
    actual = sha256_file(path)
    if actual != artifact.sha256:
        raise SystemExit(
            f"{description} SHA-256 does not match registry: "
            f"expected {artifact.sha256}, got {actual}"
        )
    return actual


def _route_label(episode: TrajectoryEpisodeManifest) -> str:
    labels = tuple(label for label in episode.labels if label.namespace == "route")
    if len(labels) != 1:
        raise ValueError(f"episode must contain exactly one route label: {episode.episode_id}")
    return labels[0].value


def _project_speed_profile(
    root: Path,
    episode: TrajectoryEpisodeManifest,
) -> TrajectoryEpisodeManifest:
    classifier = episode.classifier_trajectory_view
    if classifier is None:
        raise ValueError(f"CMRE episode has no classifier candidate view: {episode.episode_id}")
    source_path = root / classifier.asset.path
    with np.load(source_path, allow_pickle=False) as arrays:
        required = {
            "elapsed_s",
            "reported_velocity_xy_mps",
            "reported_velocity_valid_xy",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"CMRE classifier asset is missing channels: {missing}")
        timestamps = np.asarray(arrays["elapsed_s"], dtype=np.float64)
        velocity = np.asarray(arrays["reported_velocity_xy_mps"], dtype=np.float64)
        valid = np.asarray(arrays["reported_velocity_valid_xy"], dtype=np.bool_)

    if timestamps.ndim != 1 or velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError(f"CMRE classifier asset has unexpected shapes: {episode.episode_id}")
    if valid.shape != velocity.shape or timestamps.shape[0] != velocity.shape[0]:
        raise ValueError(f"CMRE classifier asset channels are not aligned: {episode.episode_id}")
    if timestamps.size < 2 or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"CMRE classifier timestamps are insufficient: {episode.episode_id}")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError(f"CMRE classifier timestamps are not strictly increasing: {episode.episode_id}")
    if not np.all(valid):
        raise ValueError(f"CMRE route-pair task requires valid horizontal velocity: {episode.episode_id}")
    speed = np.linalg.norm(velocity, axis=1)
    if not np.all(np.isfinite(speed)):
        raise ValueError(f"CMRE speed projection contains non-finite values: {episode.episode_id}")

    relative_path = Path("assets/task_classifier") / f"{episode.episode_id}.json"
    asset = write_json_asset(
        root,
        relative_path,
        {
            "timestamps_s": [float(value) for value in timestamps],
            "measurements": [float(value) for value in speed],
        },
    )
    classifier_update = classifier.model_copy(
        update={
            "asset": asset,
            "sample_count": int(timestamps.size),
            "permitted_extra_channels": (),
            "processing_step_ids": tuple(
                dict.fromkeys((*classifier.processing_step_ids, TASK_PROJECTION_STEP_ID))
            ),
        }
    )
    episode_update = episode.model_copy(
        update={
            "processing_step_ids": tuple(
                dict.fromkeys((*episode.processing_step_ids, TASK_PROJECTION_STEP_ID))
            ),
            "classifier_trajectory_view": classifier_update,
        }
    )
    return TrajectoryEpisodeManifest.model_validate(episode_update.model_dump())


def _write_assignments(root: Path, assignments: tuple[EpisodeSplitAssignment, ...]) -> Path:
    path = root / "assignments.json"
    path.write_text(
        json.dumps([assignment.model_dump(mode="json") for assignment in assignments], indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def _route_counts(
    episodes: tuple[TrajectoryEpisodeManifest, ...],
    assignments: tuple[EpisodeSplitAssignment, ...],
) -> dict[str, dict[str, int]]:
    split_by_episode = {assignment.episode_id: assignment.split for assignment in assignments}
    counts: dict[str, Counter[str]] = {split.value: Counter() for split in SnapshotSplit}
    for episode in episodes:
        counts[split_by_episode[episode.episode_id].value][_route_label(episode)] += 1
    return {split: dict(sorted(values.items())) for split, values in counts.items()}


def _assign_balanced_splits(
    episodes: tuple[TrajectoryEpisodeManifest, ...],
    *,
    route_a: str,
    route_b: str,
    seed: str,
) -> tuple[EpisodeSplitAssignment, ...]:
    expected_routes = {route_a, route_b}
    for attempt in range(1000):
        assignments = assign_grouped_snapshot_splits(
            episodes,
            seed=f"{seed}:{attempt}",
        )
        audit = audit_split_assignments(
            episodes,
            assignments,
            grouping_namespaces=(
                GroupingNamespace.PHYSICAL_PLATFORM,
                GroupingNamespace.SOURCE_RECORDING,
                GroupingNamespace.MISSION_EVENT,
            ),
        )
        if not audit.passes:
            continue
        split_routes: dict[SnapshotSplit, set[str]] = {split: set() for split in SnapshotSplit}
        split_by_episode = {assignment.episode_id: assignment.split for assignment in assignments}
        for episode in episodes:
            split_routes[split_by_episode[episode.episode_id]].add(_route_label(episode))
        if all(routes == expected_routes for routes in split_routes.values()):
            return assignments
    raise SystemExit(
        "could not find grouped train/validation/test splits containing both selected routes "
        "after 1000 deterministic seed attempts"
    )


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.route_a == arguments.route_b:
        raise SystemExit("--route-a and --route-b must be distinct")
    root = arguments.snapshot_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"snapshot root must be empty before building: {root}")
    root.mkdir(parents=True, exist_ok=True)

    created_at = (
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if arguments.created_at
        else datetime.now(UTC)
    )
    if created_at.tzinfo is None:
        raise SystemExit("--created-at must include a timezone")

    registry = load_source_registry(arguments.registry)
    source = registry.source(arguments.source_dataset_id)
    if source.evidence_state not in {SourceEvidenceState.PREPARED, SourceEvidenceState.RELEASED}:
        raise SystemExit(
            "selected source must be prepared before a classifier snapshot can be built: "
            f"{arguments.source_dataset_id} ({source.evidence_state.value})"
        )
    source_artifact = _registry_artifact(
        registry,
        arguments.source_dataset_id,
        arguments.source_artifact_id,
    )
    nomenclature_artifact = _registry_artifact(
        registry,
        arguments.source_dataset_id,
        arguments.nomenclature_artifact_id,
    )
    source_sha256 = _verify_artifact(
        arguments.tracklets,
        source_artifact,
        description="tracklets source",
    )
    nomenclature_sha256 = _verify_artifact(
        arguments.nomenclature,
        nomenclature_artifact,
        description="route nomenclature source",
    )

    requested_routes = {arguments.route_a, arguments.route_b}
    tracklets = parse_tracklets(arguments.tracklets)
    selected_tracklet_ids = {
        tracklet.tracklet_id for tracklet in tracklets if tracklet.route_id in requested_routes
    }
    selected_tracklets = tuple(
        tracklet for tracklet in tracklets if tracklet.tracklet_id in selected_tracklet_ids
    )
    counts = Counter(tracklet.route_id for tracklet in selected_tracklets)
    if set(counts) != requested_routes or any(count == 0 for count in counts.values()):
        raise SystemExit(
            f"selected source does not contain both requested routes: {dict(sorted(counts.items()))}"
        )

    result = build_fixture(
        tracklets_path=arguments.tracklets,
        nomenclature_path=arguments.nomenclature,
        output_root=root,
        source_artifact_id=arguments.source_artifact_id,
        nomenclature_artifact_id=arguments.nomenclature_artifact_id,
        corpus_snapshot_id=arguments.snapshot_id,
        identity_key=arguments.identity_key.read_bytes(),
        selected_tracklet_ids=selected_tracklet_ids,
        dataset_id=arguments.source_dataset_id,
    )
    episodes = tuple(_project_speed_profile(root, episode) for episode in result.manifests)
    episode_paths = tuple(_write_episode(root, episode) for episode in episodes)
    write_fixture_index(output_root=root, result=result)

    assignments = _assign_balanced_splits(
        episodes,
        route_a=arguments.route_a,
        route_b=arguments.route_b,
        seed=arguments.split_seed,
    )
    split_audit = audit_split_assignments(episodes, assignments)
    assignments_path = _write_assignments(root, assignments)
    snapshot = build_snapshot_manifest(
        registry,
        episode_paths,
        snapshot_root=root,
        snapshot_id=arguments.snapshot_id,
        created_at=created_at,
        require_prepared_sources=True,
        notes=(
            "Task-scoped prepared CMRE route-pair snapshot; raw tracklets and identity key remain outside Git.",
            f"Target namespace is route with pair {arguments.route_a} versus {arguments.route_b}; route is not a vessel-family label.",
            "Classifier assets contain only relative time and scalar horizontal speed; labels and grouping remain in episode metadata.",
            "Grouped physical-platform, source-recording, and mission-event split policy is retained; independent-provider source shift remains open.",
        ),
    )
    snapshot_path = root / "snapshot.json"
    write_snapshot_manifest(snapshot, snapshot_path)
    cohort_path = root / "prepared-cohort.json"
    cohort_path.write_text(
        json.dumps(
            {
                "source_dataset_id": arguments.source_dataset_id,
                "source_artifact_id": arguments.source_artifact_id,
                "source_sha256": source_sha256,
                "nomenclature_artifact_id": arguments.nomenclature_artifact_id,
                "nomenclature_sha256": nomenclature_sha256,
                "target_label_namespace": "route",
                "route_pair": [arguments.route_a, arguments.route_b],
                "tracklet_count": len(selected_tracklets),
                "episode_count": len(episodes),
                "route_counts": dict(sorted(counts.items())),
                "physical_platform_group_count": result.physical_platform_group_count,
                "repeated_physical_platform_group_count": len(
                    result.repeated_physical_platform_groups
                ),
                "split_audit_passes": split_audit.passes,
                "split_audit_issues": list(split_audit.issues),
                "route_counts_by_split": _route_counts(episodes, assignments),
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
    print(cohort_path)
    print(f"tracklet_count={len(selected_tracklets)}")
    print(f"episode_count={len(episodes)}")
    print(f"route_counts={dict(sorted(counts.items()))}")
    print(f"route_counts_by_split={_route_counts(episodes, assignments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
