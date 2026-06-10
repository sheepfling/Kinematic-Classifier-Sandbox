from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.analysis.pca_dimensionality_audit import (
    write_pca_dimensionality_audit_artifacts,
)


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
