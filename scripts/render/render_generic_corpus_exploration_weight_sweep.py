#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.corpus.exploration.generic_corpus_exploration import (
    write_generic_corpus_exploration_weight_sweep_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the generic corpus exploration weight sweep artifact bundle.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the sweep bundle should be written.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for corpus generation.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config defining the baseline and weight variants.",
    )
    args = parser.parse_args()
    artifacts = write_generic_corpus_exploration_weight_sweep_artifacts(
        Path(args.output_dir),
        seed=args.seed,
        config_path=args.config,
    )
    print(artifacts.run_dir)
    print(artifacts.config_path)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.rows_path)
    print(artifacts.overlap_matrix_path)
    print(artifacts.weight_matrix_path)
    print(artifacts.tradeoff_png_path)
    print(artifacts.selected_set_png_path)
    print(artifacts.baseline_manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
