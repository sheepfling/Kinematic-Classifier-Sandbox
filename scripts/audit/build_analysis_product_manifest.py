"""Build an external Product 4 analysis-product selection manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.corpus.real_world.analysis_products import (  # noqa: E402
    AnalysisProductId,
    build_analysis_product_manifest,
    write_analysis_product_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import (  # noqa: E402
    SnapshotSelectionPolicy,
    load_snapshot_episodes,
    load_snapshot_manifest,
    load_source_registry,
    select_snapshot_episodes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select one explicitly bounded Product 4 analysis product from an external "
            "snapshot. The classifier-ladder profile never falls back to analysis assets."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "docs/product4/real_world_source_registry.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--product",
        choices=tuple(product.value for product in AnalysisProductId),
        required=True,
    )
    parser.add_argument("--lane", action="append", default=[])
    parser.add_argument("--source-dataset-id", action="append", default=[])
    parser.add_argument(
        "--target-label-namespace",
        help="Required for classifier_ladder; explicitly names the out-of-band target task.",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    product_id = AnalysisProductId(arguments.product)
    snapshot = load_snapshot_manifest(arguments.snapshot)
    episodes = load_snapshot_episodes(snapshot, arguments.snapshot)
    selected = select_snapshot_episodes(
        episodes,
        SnapshotSelectionPolicy(
            lanes=tuple(arguments.lane),
            source_dataset_ids=tuple(arguments.source_dataset_id),
            require_classifier_view=product_id is AnalysisProductId.CLASSIFIER_LADDER,
        ),
    )
    registry = load_source_registry(arguments.registry)
    manifest = build_analysis_product_manifest(
        snapshot,
        selected,
        registry,
        product_id=product_id,
        target_label_namespace=arguments.target_label_namespace,
    )
    write_analysis_product_manifest(manifest, arguments.output)
    print(arguments.output)
    print(f"product={manifest.policy.product_id.value}")
    print(f"episode_count={len(manifest.episodes)}")
    print(f"snapshot_content_sha256={manifest.snapshot_content_sha256}")
    print(f"lane_episode_counts={dict(manifest.lane_episode_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
