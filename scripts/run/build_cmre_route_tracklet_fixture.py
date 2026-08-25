from __future__ import annotations

import argparse
from pathlib import Path

from kinematic_classifier_sandbox.corpus.real_world.adapters.cmre_route_tracklets import (
    build_fixture,
    write_fixture_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a governed Product 4 SEA-surface fixture from a local CMRE/Brest "
            "route-tracklet artifact."
        )
    )
    parser.add_argument("tracklets", type=Path)
    parser.add_argument("nomenclature", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-artifact-id", required=True)
    parser.add_argument("--corpus-snapshot-id", required=True)
    parser.add_argument(
        "--identity-key-file",
        type=Path,
        required=True,
        help=(
            "Path to a private key file used for HMAC-based physical-platform grouping. "
            "The key is read locally and is never written to the prepared corpus."
        ),
    )
    parser.add_argument(
        "--tracklet-id",
        type=int,
        action="append",
        default=None,
        help="Repeat to select bounded source tracklets; omit to process all rows.",
    )
    return parser
####


def main() -> None:
    arguments = _parser().parse_args()
    selected = (
        set(arguments.tracklet_id)
        if arguments.tracklet_id is not None
        else None
    )
    result = build_fixture(
        tracklets_path=arguments.tracklets,
        nomenclature_path=arguments.nomenclature,
        output_root=arguments.output_dir,
        source_artifact_id=arguments.source_artifact_id,
        corpus_snapshot_id=arguments.corpus_snapshot_id,
        identity_key=arguments.identity_key_file.read_bytes(),
        selected_tracklet_ids=selected,
    )
    index_path = write_fixture_index(
        output_root=arguments.output_dir,
        result=result,
    )
    print(index_path)
####


if __name__ == "__main__":
    main()
####
