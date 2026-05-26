#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-mpl"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
