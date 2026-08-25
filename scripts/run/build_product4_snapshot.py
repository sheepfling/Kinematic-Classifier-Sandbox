"""Build a governed Product 4 snapshot manifest in an external snapshot root."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    load_source_registry,
)
from kinematic_classifier_sandbox.corpus.real_world.snapshot_builder import (  # noqa: E402
    build_snapshot_manifest,
    write_snapshot_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable Product 4 snapshot manifest from episode manifests "
            "stored under an external snapshot root."
        )
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        help="Episode-manifest directory; defaults to <snapshot-root>/episodes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest path; defaults to <snapshot-root>/snapshot.json.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "docs/product4/real_world_source_registry.yaml",
    )
    parser.add_argument("--created-at", help="UTC ISO-8601 timestamp; defaults to current UTC time.")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--require-prepared-sources",
        action="store_true",
        help="Refuse to build when any selected source is below prepared evidence state.",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    snapshot_root = arguments.snapshot_root.resolve()
    episodes_dir = (arguments.episodes_dir or snapshot_root / "episodes").resolve()
    output = (arguments.output or snapshot_root / "snapshot.json").resolve()
    episode_paths = tuple(sorted(episodes_dir.rglob("*.json")))
    created_at = (
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if arguments.created_at
        else datetime.now(UTC)
    )
    if created_at.tzinfo is None:
        raise SystemExit("--created-at must include a timezone")
    registry = load_source_registry(arguments.registry)
    manifest = build_snapshot_manifest(
        registry,
        episode_paths,
        snapshot_root=snapshot_root,
        snapshot_id=arguments.snapshot_id,
        created_at=created_at,
        require_prepared_sources=arguments.require_prepared_sources,
        notes=arguments.note,
    )
    write_snapshot_manifest(manifest, output)
    print(output)
    print(manifest.content_sha256())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
