#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.rung_sufficiency.analysis import (
    write_ladder_witness_suite_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the ladder witness corpus suite manifest and schema bundle.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the witness suite bundle should be written.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config defining the witness suite.",
    )
    args = parser.parse_args()
    artifacts = write_ladder_witness_suite_artifacts(Path(args.output_dir), config_path=args.config)
    print(artifacts.run_dir)
    print(artifacts.config_path)
    print(artifacts.schema_path)
    print(artifacts.manifest_path)
    print(artifacts.claim_matrix_path)
    print(artifacts.index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
