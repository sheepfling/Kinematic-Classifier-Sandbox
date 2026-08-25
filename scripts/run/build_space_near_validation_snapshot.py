"""Build the bounded SPACE-NEAR validation tranche outside the repository."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    load_source_registry,
    write_snapshot_manifest,
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

_ARTIFACT_IDS = {
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
            "Build a common-front SPACE-NEAR validation snapshot from the six "
            "bounded repository fixtures."
        )
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=ROOT / "tests/corpus/real_world/fixtures/space_near",
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
    )
    parser.add_argument("--created-at", help="UTC ISO-8601 timestamp; defaults to current UTC time.")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.snapshot_root.resolve()
    episodes_root = root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    created_at = (
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if arguments.created_at
        else datetime.now(UTC)
    )
    if created_at.tzinfo is None:
        raise SystemExit("--created-at must include a timezone")
    registry = load_source_registry(arguments.registry)
    for fixture in load_space_near_fixture_definitions(arguments.fixture_root):
        try:
            source_artifact_id = _ARTIFACT_IDS[fixture.source.source_dataset_id]
        except KeyError as error:
            raise SystemExit(
                f"no registry artifact mapping for {fixture.source.source_dataset_id}"
            ) from error
        episode = build_fixture_episode_manifest(
            fixture,
            output_root=root,
            corpus_snapshot_id=arguments.snapshot_id,
            source_artifact_id=source_artifact_id,
        )
        (episodes_root / f"{episode.episode_id}.json").write_text(
            episode.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = build_snapshot_manifest(
        registry,
        sorted(episodes_root.glob("*.json")),
        snapshot_root=root,
        snapshot_id=arguments.snapshot_id,
        created_at=created_at,
        notes=(
            "SPACE-NEAR bounded validation tranche; common-front contract validated, "
            "authoritative semantic sign-off and classifier view intentionally blocked.",
        ),
    )
    output = root / "snapshot.json"
    write_snapshot_manifest(manifest, output)
    print(output)
    print(manifest.content_sha256())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
