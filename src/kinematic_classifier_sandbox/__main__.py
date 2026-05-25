from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import METHOD_CATALOG
from .coverage_report import write_coverage_report_artifacts
from .formal_math_registry import write_formal_math_registry_artifacts
from .formal_math_visual_registry import write_formal_math_visual_registry_artifacts
from .functional_surface_catalog import write_functional_surface_catalog_artifacts
from .generic_corpus_exploration import write_generic_corpus_exploration_weight_sweep_artifacts
from .rung_sufficiency import write_ladder_witness_suite_artifacts
from .pca_dimensionality_audit import write_pca_dimensionality_audit_artifacts
from .repo_story import write_repo_story_artifacts
from .strict_equation_audit import write_strict_equation_audit_artifacts


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

    functional_surface_catalog = subparsers.add_parser(
        "functional-surface-catalog",
        help="Render the functional-surface inventory and artifact coverage bundle.",
    )
    functional_surface_catalog.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the catalog bundle should be written.",
    )

    corpus_sweep = subparsers.add_parser(
        "generic-corpus-exploration-weight-sweep",
        help="Run the generic corpus exploration weight sweep.",
    )
    corpus_sweep.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the sweep bundle should be written.",
    )
    corpus_sweep.add_argument("--seed", type=int, default=7, help="Random seed for corpus generation.")
    corpus_sweep.add_argument(
        "--config",
        default=None,
        help="Optional YAML config defining the baseline and weight variants.",
    )

    formal_math_registry = subparsers.add_parser(
        "formal-math-registry",
        help="Render the formal math registry and function-equation crosswalk bundle.",
    )
    formal_math_registry.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the registry bundle should be written.",
    )

    formal_math_visual_registry = subparsers.add_parser(
        "formal-math-visual-registry",
        help="Render the formal math visual gallery bundle.",
    )
    formal_math_visual_registry.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the visual gallery bundle should be written.",
    )

    strict_equation_audit = subparsers.add_parser(
        "strict-equation-audit",
        help="Render the strict equation audit bundle.",
    )
    strict_equation_audit.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the strict audit bundle should be written.",
    )

    ladder_witness_suite = subparsers.add_parser(
        "ladder-witness-suite",
        help="Render the ladder witness corpus suite manifest and schema bundle.",
    )
    ladder_witness_suite.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the witness suite bundle should be written.",
    )
    ladder_witness_suite.add_argument(
        "--config",
        default=None,
        help="Optional YAML config defining the witness suite.",
    )

    repo_story = subparsers.add_parser(
        "repo-story",
        help="Render the canonical repo-story proof-navigation bundle.",
    )
    repo_story.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the repo-story bundle should be written.",
    )
    repo_story.add_argument(
        "--docs-root",
        default="docs",
        help="Docs root where generated story pages should be refreshed.",
    )
    repo_story.add_argument(
        "--no-showcase",
        action="store_true",
        help="Do not refresh showcase/team-packet front doors.",
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

    if args.command == "functional-surface-catalog":
        artifacts = write_functional_surface_catalog_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.catalog_path)
        print(artifacts.plot_path)
        return 0

    if args.command == "generic-corpus-exploration-weight-sweep":
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

    if args.command == "formal-math-registry":
        artifacts = write_formal_math_registry_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.function_registry_path)
        print(artifacts.equation_registry_path)
        print(artifacts.crosswalk_path)
        print(artifacts.plot_path)
        return 0

    if args.command == "formal-math-visual-registry":
        artifacts = write_formal_math_visual_registry_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.gallery_csv_path)
        print(artifacts.provenance_path)
        print(artifacts.runbook_path)
        print(artifacts.visual_coverage_png_path)
        print(artifacts.assets_dir)
        return 0

    if args.command == "strict-equation-audit":
        artifacts = write_strict_equation_audit_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.rows_path)
        print(artifacts.status_plot_path)
        return 0

    if args.command == "ladder-witness-suite":
        artifacts = write_ladder_witness_suite_artifacts(
            Path(args.output_dir),
            config_path=args.config,
        )
        print(artifacts.run_dir)
        print(artifacts.config_path)
        print(artifacts.schema_path)
        print(artifacts.manifest_path)
        print(artifacts.claim_matrix_path)
        print(artifacts.index_path)
        return 0

    if args.command == "repo-story":
        artifacts = write_repo_story_artifacts(
            Path(args.output_dir),
            docs_root=Path(args.docs_root),
            write_showcase=not args.no_showcase,
        )
        print(artifacts.run_dir)
        print(artifacts.claim_matrix_path)
        print(artifacts.artifact_manifest_path)
        print(artifacts.status_report_path)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
