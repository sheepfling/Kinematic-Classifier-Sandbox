from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import METHOD_CATALOG
from .coverage_report import write_coverage_report_artifacts
from .pca_dimensionality_audit import write_pca_dimensionality_audit_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kinematic_classifier_sandbox")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("methods", help="List cataloged method families.")

    coverage = subparsers.add_parser(
        "coverage-report",
        help="Run the corpus coverage report and write artifacts.",
    )
    coverage.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the coverage report artifact bundle should be written.",
    )
    coverage.add_argument("--seed", type=int, default=7, help="Random seed for corpus generation.")
    coverage.add_argument(
        "--trajectories-per-class",
        type=int,
        default=5,
        help="Trajectories per class per tier for the generated corpus.",
    )
    coverage.add_argument(
        "--print-report",
        action="store_true",
        help="Also print the markdown report to stdout after writing artifacts.",
    )

    pca_dimensionality = subparsers.add_parser(
        "pca-dimensionality-audit",
        help="Sweep PCA dimensionality and clusterability diagnostics.",
    )
    pca_dimensionality.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the PCA audit artifact bundle should be written.",
    )
    pca_dimensionality.add_argument("--seed", type=int, default=7, help="Random seed for corpus generation.")
    pca_dimensionality.add_argument(
        "--trajectories-per-class",
        type=int,
        default=5,
        help="Trajectories per class per tier for the generated corpus.",
    )
    pca_dimensionality.add_argument(
        "--feature-set",
        default=None,
        help="Optional feature-set id to analyze.",
    )
    pca_dimensionality.add_argument(
        "--max-components",
        type=int,
        default=6,
        help="Maximum number of principal components to sweep.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {None, "methods"}:
        for entry in METHOD_CATALOG:
            print(f"{entry.family}: {entry.name}")
        return 0

    if args.command == "coverage-report":
        artifacts = write_coverage_report_artifacts(
            Path(args.output_dir),
            seed=args.seed,
            trajectories_per_class=args.trajectories_per_class,
        )
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        if args.print_report:
            print(artifacts.report_path.read_text(encoding="utf-8"))
        return 0

    if args.command == "pca-dimensionality-audit":
        artifacts = write_pca_dimensionality_audit_artifacts(
            Path(args.output_dir),
            seed=args.seed,
            trajectories_per_class=args.trajectories_per_class,
            feature_set=args.feature_set,
            max_components=args.max_components,
        )
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
