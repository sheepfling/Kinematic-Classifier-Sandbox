from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox import write_pca_dimensionality_audit_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/audit/pca_dimensionality_audit.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where the audit bundle should be written.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for feature generation.")
    parser.add_argument(
        "--trajectories-per-class",
        type=int,
        default=5,
        help="Trajectories per class per tier for the generated feature matrix.",
    )
    parser.add_argument("--feature-set", default=None, help="Optional feature-set id to analyze.")
    parser.add_argument(
        "--max-components",
        type=int,
        default=6,
        help="Maximum number of principal components to sweep.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = write_pca_dimensionality_audit_artifacts(
        args.output_dir,
        seed=args.seed,
        trajectories_per_class=args.trajectories_per_class,
        feature_set=args.feature_set,
        max_components=args.max_components,
    )
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
