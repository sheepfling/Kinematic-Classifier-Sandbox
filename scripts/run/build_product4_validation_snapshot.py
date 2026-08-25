"""Build the external six-lane Product 4 validation snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.corpus.real_world.adapters.cmre_route_tracklets import (  # noqa: E402
    build_fixture as build_cmre_fixture,
)
from kinematic_classifier_sandbox.corpus.real_world.adapters.cmre_route_tracklets import (
    write_fixture_index as write_cmre_fixture_index,
)
from kinematic_classifier_sandbox.corpus.real_world.air.common_front import (  # noqa: E402
    build_readsb_fixture_episodes,
)
from kinematic_classifier_sandbox.corpus.real_world.land.common_front import (  # noqa: E402
    build_tgsim_fixture_episodes,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    assign_grouped_snapshot_splits,
    load_snapshot_episodes,
    load_source_registry,
    write_snapshot_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.sea_subsurface.common_front import (  # noqa: E402
    build_ioos_anchor_episode,
)
from kinematic_classifier_sandbox.corpus.real_world.snapshot_builder import (  # noqa: E402
    build_snapshot_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.space_near.common_front import (  # noqa: E402
    build_fixture_episode_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.space_near.fixture_adapter import (  # noqa: E402
    load_space_near_fixture_definitions,
)
from kinematic_classifier_sandbox.corpus.real_world.space_orbital.common_front import (  # noqa: E402
    build_nasa_iss_oem_episode,
)

CMRE_SOURCE_ARTIFACT_ID = "cmre_brest_external_validation_packet_v2"
CMRE_NOMENCLATURE_ARTIFACT_ID = "cmre_brest_external_validation_route_nomenclature_v2"
SPACE_NEAR_ARTIFACT_IDS = {
    "nasa-spdf:ENDURANCE_EPHEMERIS_DEF": "nasa_endurance_47_001_synchronized_solution",
    "darts:soundingrockets-s-310-40": "darts_s31040_repository_fixture",
    "darts:soundingrockets-s-310-44": "darts_s31044_repository_fixture",
    "darts:soundingrockets-s-520-26": "darts_s52026_repository_fixture",
    "darts:soundingrockets-s-520-27": "darts_s52027_repository_fixture",
    "darts:soundingrockets-s-520-29": "darts_s52029_repository_fixture",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a six-lane Product 4 validation snapshot outside the repository. "
            "All generated assets remain under --snapshot-root."
        )
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--cmre-tracklets", type=Path, required=True)
    parser.add_argument("--cmre-nomenclature", type=Path, required=True)
    parser.add_argument("--cmre-identity-key", type=Path, required=True)
    parser.add_argument(
        "--land-source",
        type=Path,
        default=ROOT / "tests/corpus/real_world/fixtures/tgsim_foggy_bottom_minimal.csv",
    )
    parser.add_argument(
        "--land-source-dataset-id",
        default="fhwa_tgsim_foggy_bottom",
    )
    parser.add_argument(
        "--land-source-artifact-id",
        default="fhwa_tgsim_foggy_bottom_trajectory_csv",
    )
    parser.add_argument(
        "--sea-sub-source",
        type=Path,
        default=ROOT / "docs/research/product4/sea_subsurface/fixtures/ioos_uaf_unit_191_profile_1709942882.csv",
    )
    parser.add_argument(
        "--air-source",
        type=Path,
        default=ROOT / "tests/corpus/real_world/fixtures/readsb_documented_a320_trace.json",
    )
    parser.add_argument(
        "--space-orbital-source",
        type=Path,
        default=ROOT / "tests/corpus/real_world/fixtures/nasa_iss_oem_20220427_excerpt.kvn",
    )
    parser.add_argument(
        "--space-near-fixture-root",
        type=Path,
        default=ROOT / "tests/corpus/real_world/fixtures/space_near",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
    )
    parser.add_argument("--created-at", help="UTC ISO-8601 timestamp; defaults to current UTC time.")
    return parser


def _write_episode(root: Path, episode) -> None:
    episodes_root = root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    (episodes_root / f"{episode.episode_id}.json").write_text(
        episode.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.snapshot_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    created_at = (
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if arguments.created_at
        else datetime.now(UTC)
    )
    if created_at.tzinfo is None:
        raise SystemExit("--created-at must include a timezone")
    registry = load_source_registry(arguments.registry)

    cmre_result = build_cmre_fixture(
        tracklets_path=arguments.cmre_tracklets,
        nomenclature_path=arguments.cmre_nomenclature,
        output_root=root,
        source_artifact_id=CMRE_SOURCE_ARTIFACT_ID,
        nomenclature_artifact_id=CMRE_NOMENCLATURE_ARTIFACT_ID,
        corpus_snapshot_id=arguments.snapshot_id,
        identity_key=arguments.cmre_identity_key.read_bytes(),
        selected_tracklet_ids=set(range(1, 13)),
    )
    write_cmre_fixture_index(output_root=root, result=cmre_result)

    build_tgsim_fixture_episodes(
        arguments.land_source,
        output_root=root,
        corpus_snapshot_id=arguments.snapshot_id,
        dataset_id=arguments.land_source_dataset_id,
        source_artifact_id=arguments.land_source_artifact_id,
    )
    _write_episode(
        root,
        build_ioos_anchor_episode(
            arguments.sea_sub_source,
            output_root=root,
            corpus_snapshot_id=arguments.snapshot_id,
        ),
    )
    build_readsb_fixture_episodes(
        arguments.air_source,
        output_root=root,
        corpus_snapshot_id=arguments.snapshot_id,
    )
    _write_episode(
        root,
        build_nasa_iss_oem_episode(
            arguments.space_orbital_source,
            output_root=root,
            corpus_snapshot_id=arguments.snapshot_id,
        ),
    )
    for fixture in load_space_near_fixture_definitions(arguments.space_near_fixture_root):
        source_artifact_id = SPACE_NEAR_ARTIFACT_IDS[fixture.source.source_dataset_id]
        _write_episode(
            root,
            build_fixture_episode_manifest(
                fixture,
                output_root=root,
                corpus_snapshot_id=arguments.snapshot_id,
                source_artifact_id=source_artifact_id,
            ),
        )

    episode_paths = tuple(sorted((root / "episodes").glob("*.json")))
    manifest = build_snapshot_manifest(
        registry,
        episode_paths,
        snapshot_root=root,
        snapshot_id=arguments.snapshot_id,
        created_at=created_at,
        notes=(
            "Six-lane validation snapshot assembled from bounded external or repository fixtures.",
            "Classifier projection and prepared-source promotion remain governed gates.",
            "Raw restricted source rows, identity key, and generated assets remain outside Git.",
        ),
    )
    snapshot_path = root / "snapshot.json"
    write_snapshot_manifest(manifest, snapshot_path)
    assignments = assign_grouped_snapshot_splits(
        load_snapshot_episodes(manifest, snapshot_path)
    )
    assignments_path = root / "assignments.json"
    assignments_path.write_text(
        json.dumps(
            [assignment.model_dump(mode="json") for assignment in assignments],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(snapshot_path)
    print(manifest.content_sha256())
    print(assignments_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
