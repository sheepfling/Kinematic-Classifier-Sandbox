from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .analysis.pca_dimensionality_audit import write_pca_dimensionality_audit_artifacts
from .corpus.control_surfaces.backend_sweep import (
    ControlSurfaceBackendSweepConfig,
    write_control_surface_backend_sweep_artifacts,
)
from .corpus.coverage_artifact_io import write_coverage_report_artifacts
from .corpus.exploration.generic_corpus_exploration import (
    write_generic_corpus_exploration_weight_sweep_artifacts,
)
from .corpus.trajectory_exploration.artifact_io import write_trajectory_exploration_artifacts
from .corpus.trajectory_exploration.backend_registry import (
    write_exploration_backend_registry_artifacts,
)
from .corpus.trajectory_exploration.objective_generation import (
    resolve_generated_trajectory_objective,
    write_generated_trajectory_objective_artifacts,
)
from .corpus.trajectory_exploration.ppo_boundary_control import (
    SequentialPpoConfig,
    write_generated_trajectory_objective_ppo_sweep_artifacts,
    write_sequential_ppo_boundary_control_artifacts,
)
from .corpus.trajectory_exploration.sequential_comparison import (
    SequentialCemConfig,
    write_sequential_objective_sweep_comparison_artifacts,
    write_sequential_ppo_vs_cem_comparison_artifacts,
)
from .corpus.trajectory_exploration.sequential_gym import SequentialBoundaryControlConfig
from .corpus.validation import validate_corpus_explorer_packet
from .meta.repo_shape_audit import write_repo_shape_audit_artifacts
from .methodology.latex import (
    write_methodology_latex_artifacts,
    write_methodology_section_symbol_audit_artifacts,
)
from .registry.algorithm_coverage_matrix import write_algorithm_coverage_matrix_artifacts
from .registry.catalog import METHOD_CATALOG
from .registry.corpus_evaluation_gap_matrix import write_corpus_evaluation_gap_matrix_artifacts
from .registry.exported_surface_coverage import write_exported_surface_coverage_artifacts
from .registry.formal_math_registry import write_formal_math_registry_artifacts
from .registry.formal_math_visual_registry import write_formal_math_visual_registry_artifacts
from .registry.functional_surface_catalog import write_functional_surface_catalog_artifacts
from .registry.strict_equation_audit import write_strict_equation_audit_artifacts
from .rung_sufficiency.analysis import write_ladder_witness_suite_artifacts
from .static_admissibility.audit import run_static_admissibility_audit
from .static_admissibility.exemplar_suite import (
    write_static_admissibility_exemplar_suite_packet,
)
from .static_admissibility.io import export_static_admissibility_packet
from .static_admissibility.multi_domain_3d import (
    write_multidomain_3d_static_admissibility_packet,
)
from .static_admissibility.validation import validate_static_admissibility_packet
from .story.repo_story import write_repo_story_artifacts
from .tracing.filter_trace_validation_packet import write_filter_trace_validation_artifacts
from .utils.analysis_cache import clear_analysis_cache, describe_analysis_cache
from .utils.runtime import repo_root
from .validation.correctness import run_correctness_plan
from .validation_packets import (
    validate_v7_anduril_c2_blend_packet,
    write_v7_anduril_c2_blend_packet,
)
from .workbench.mvp import (
    analyze_workbench_run,
    build_epic1_showcase,
    compare_rungs,
    export_presentation_packet,
    export_workbench_packet,
    inspect_run,
    list_runs,
    run_workbench_study,
    search_corpus,
    validate_study_spec,
    validate_workbench_run,
)
from .workbench.revision_replay import (
    change_measurement_association,
    correct_measurement,
    diff_revisions,
    ensure_revision_history,
    inspect_measurement,
    replay_revision,
    restore_measurement,
    revoke_measurement,
    validate_replay,
)


def _init_static_audit_bundle(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    template_dir = repo_root() / "templates"
    template_map = {
        "static_audit_bundle.yaml": "static_audit_bundle.yaml",
        "samples.csv": "static_audit_samples.csv",
        "feature_schema.csv": "static_audit_feature_schema.csv",
        "class_schema.csv": "static_audit_class_schema.csv",
    }
    for output_name, template_name in template_map.items():
        shutil.copyfile(template_dir / template_name, destination / output_name)
    readme_lines = [
        "# Static Audit Bundle",
        "",
        "This directory is a portable Epic 1 static-admissibility bundle.",
        "",
        "Files:",
        "",
        "- `static_audit_bundle.yaml`: study declaration and prior regime",
        "- `samples.csv`: labeled feature rows",
        "- `feature_schema.csv`: feature provenance and online/leakage flags",
        "- `class_schema.csv`: declared class surface",
        "- optional `class_feature_signature.csv`: expected signatures for future classes",
        "  (copy `templates/static_audit_class_feature_signature.csv` and enable it in the YAML)",
        "",
        "Run it with:",
        "",
        "```bash",
        "PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit \\",
        "  --bundle static_audit_bundle.yaml \\",
        "  --output-dir artifacts/runs/my_static_audit",
        "```",
        "",
        "Validate the packet with:",
        "",
        "```bash",
        "PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet \\",
        "  artifacts/runs/my_static_audit",
        "```",
    ]
    (destination / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return destination


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

    exported_surface_coverage = subparsers.add_parser(
        "exported-surface-coverage",
        help="Render the canonical export_artifacts surface coverage audit bundle.",
    )
    exported_surface_coverage.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the exported-surface coverage bundle should be written.",
    )
    exported_surface_coverage.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize the selected surfaces into a temporary output directory and classify observed artifact classes.",
    )
    exported_surface_coverage.add_argument(
        "--surface-id",
        action="append",
        dest="surface_ids",
        default=None,
        help="Optional surface id to audit. Repeat to audit a subset.",
    )

    corpus_evaluation_gap_matrix = subparsers.add_parser(
        "corpus-evaluation-gap-matrix",
        help="Render the canonical corpus-evaluation capability and coherence audit bundle.",
    )
    corpus_evaluation_gap_matrix.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the corpus-evaluation gap-matrix bundle should be written.",
    )
    corpus_evaluation_gap_matrix.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize the selected capabilities into a temporary output directory and classify observed artifact classes.",
    )
    corpus_evaluation_gap_matrix.add_argument(
        "--capability-id",
        action="append",
        dest="capability_ids",
        default=None,
        help="Optional capability id to audit. Repeat to audit a subset.",
    )

    algorithm_coverage_matrix = subparsers.add_parser(
        "algorithm-coverage-matrix",
        help="Render the broader algorithm lane and capability coverage matrix.",
    )
    algorithm_coverage_matrix.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the algorithm-coverage bundle should be written.",
    )

    repo_shape_audit = subparsers.add_parser(
        "repo-shape-audit",
        help="Audit package layout, duplicate scripts, generated cruft, and oversized modules.",
    )
    repo_shape_audit.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the repo-shape audit bundle should be written.",
    )

    filter_trace_validation = subparsers.add_parser(
        "filter-trace-validation",
        help="Render the step-level filter trace validation packet.",
    )
    filter_trace_validation.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the trace-validation bundle should be written.",
    )

    v7_final_packet = subparsers.add_parser(
        "build-v7-final-packet",
        help="Assemble the V7 integrated three-epic validation packet.",
    )
    v7_final_packet.add_argument(
        "--output-dir",
        default="artifacts/validation_packets/v7_anduril_c2_blend",
        help="Directory where the V7 integrated packet should be written.",
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

    methodology_section_symbol_audit = subparsers.add_parser(
        "methodology-section-symbol-audit",
        help="Render the methodology section symbol audit bundle.",
    )
    methodology_section_symbol_audit.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the methodology symbol audit bundle should be written.",
    )

    methodology_latex = subparsers.add_parser(
        "methodology-latex",
        help="Render the methodology LaTeX bundle and optionally build the PDF.",
    )
    methodology_latex.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the methodology LaTeX bundle should be written.",
    )
    methodology_latex.add_argument(
        "--fast",
        action="store_true",
        help="Write the LaTeX bundle without attempting a PDF build.",
    )
    methodology_latex.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write the LaTeX bundle without building the PDF.",
    )

    analysis_cache = subparsers.add_parser(
        "analysis-cache",
        help="Inspect or clear persistent analysis caches.",
    )
    analysis_cache.add_argument(
        "action",
        choices=("summary", "clear"),
        nargs="?",
        default="summary",
        help="Inspect or clear the analysis cache.",
    )
    analysis_cache.add_argument(
        "--namespace",
        default=None,
        help="Restrict the action to one cache namespace.",
    )
    analysis_cache.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    analysis_cache.add_argument(
        "--yes",
        action="store_true",
        help="Required for the destructive `clear` action.",
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

    run_static_audit = subparsers.add_parser(
        "run-static-audit",
        help="Run the static feature/class/prior admissibility packet from a default generator or file-backed study bundle.",
    )
    run_static_audit.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional YAML config for the static audit.",
    )
    run_static_audit.add_argument(
        "--bundle",
        default=None,
        help="Explicit path to a file-backed static-audit bundle YAML. Overrides the positional config if both are provided.",
    )
    run_static_audit.add_argument(
        "--output-dir",
        default="artifacts/packets/static_admissibility_mvp",
        help="Directory where the static admissibility packet should be written.",
    )

    init_static_audit_bundle = subparsers.add_parser(
        "init-static-audit-bundle",
        help="Create a portable static-audit bundle from the repo templates.",
    )
    init_static_audit_bundle.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the starter bundle should be written.",
    )

    run_static_audit_suite = subparsers.add_parser(
        "run-static-audit-suite",
        help="Build the Epic 1 static-admissibility exemplar suite packet.",
    )
    run_static_audit_suite.add_argument(
        "suite_manifest",
        nargs="?",
        default="experiments/static_admissibility/epic1_exemplar_suite.yaml",
        help="YAML manifest declaring the exemplar family suite.",
    )
    run_static_audit_suite.add_argument(
        "--output-dir",
        default="artifacts/validation_packets/01_static_admissibility",
        help="Directory where the Epic 1 validation packet should be written.",
    )

    run_static_audit_multi_domain_3d = subparsers.add_parser(
        "run-static-audit-multi-domain-3d",
        help="Build the Epic 1 multi-domain 3D static-admissibility brief packet.",
    )
    run_static_audit_multi_domain_3d.add_argument(
        "--output-dir",
        default="artifacts/validation_packets/01_static_admissibility_multi_domain_3d",
        help="Directory where the Epic 1 multi-domain 3D packet should be written.",
    )

    validate_correctness = subparsers.add_parser(
        "validate-correctness",
        help="Run the layered algorithm correctness ladder.",
    )
    validate_correctness.add_argument(
        "--level",
        choices=("smoke", "full", "presentation"),
        default="smoke",
        help="Correctness level to run.",
    )

    validate_study = subparsers.add_parser(
        "validate-study",
        help="Validate a workbench study YAML before running it.",
    )
    validate_study.add_argument("study_spec", help="Path to a study YAML.")

    run_study = subparsers.add_parser(
        "run-study",
        help="Run a declared workbench study and emit the standard run directory.",
    )
    run_study.add_argument("study_spec", help="Path to a study YAML.")
    run_study.add_argument("--output-dir", required=True, help="Destination run directory.")
    run_study.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    run_study.add_argument(
        "--trajectories-per-case",
        type=int,
        default=8,
        help="Executable shared scenario count per class.",
    )

    search_corpus_parser = subparsers.add_parser(
        "search-corpus",
        help="Write a governed corpus-search backend packet for CEM/PPO/baseline comparison.",
    )
    search_corpus_parser.add_argument("config", help="Path to a corpus-search config YAML.")
    search_corpus_parser.add_argument("--output-dir", required=True, help="Destination search run directory.")

    analyze_run = subparsers.add_parser(
        "analyze-run",
        help="Refresh a workbench run's decision card and report.",
    )
    analyze_run.add_argument("--run-dir", required=True, help="Workbench run directory.")

    compare_rungs_parser = subparsers.add_parser(
        "compare-rungs",
        help="Refresh the advanced-filter decision surface for a workbench run.",
    )
    compare_rungs_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")

    epic1_showcase = subparsers.add_parser(
        "build-epic1-showcase",
        help="Regenerate the Epic 1 evidence set, workbench packet, governed search lane, and presentation showcase.",
    )
    epic1_showcase.add_argument(
        "--output-dir",
        default="artifacts/epic1_showcase",
        help="Directory where the regenerated Epic 1 evidence set should be written.",
    )
    epic1_showcase.add_argument(
        "--study-spec",
        default="experiments/common_1d_classifier_study/common_experiment_config.yaml",
        help="Study YAML used for the workbench run.",
    )
    epic1_showcase.add_argument(
        "--corpus-search-config",
        default="experiments/templates/corpus_search_study.yaml",
        help="Config used for the governed CEM/PPO corpus-search lane.",
    )
    epic1_showcase.add_argument(
        "--presentation-output-dir",
        default=None,
        help="Optional presentation packet destination, for example artifacts/presentation_hero_charts_v4.",
    )
    epic1_showcase.add_argument("--seed", type=int, default=7, help="Workbench run seed.")
    epic1_showcase.add_argument(
        "--trajectories-per-case",
        type=int,
        default=4,
        help="Executable shared scenario count per class for the sample workbench run.",
    )
    epic1_showcase.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip static-admissibility packet regeneration for a faster smoke run.",
    )
    epic1_showcase.add_argument(
        "--skip-presentation",
        action="store_true",
        help="Skip presentation packet regeneration for a faster smoke run.",
    )

    inspect_run_parser = subparsers.add_parser(
        "inspect-run",
        help="Print a concise status summary for a workbench run.",
    )
    inspect_run_parser.add_argument("run_dir", help="Workbench run directory.")

    inspect_measurement_parser = subparsers.add_parser(
        "inspect-measurement",
        help="Inspect one measurement inside a revision-aware workbench run.",
    )
    inspect_measurement_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    inspect_measurement_parser.add_argument("--measurement-id", required=True, help="Measurement identifier.")
    inspect_measurement_parser.add_argument("--revision-id", default=None, help="Optional revision id.")

    revoke_measurement_parser = subparsers.add_parser(
        "revoke-measurement",
        help="Append a measurement revocation event to a workbench run.",
    )
    revoke_measurement_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    revoke_measurement_parser.add_argument("--measurement-id", required=True, help="Measurement identifier.")
    revoke_measurement_parser.add_argument("--reason", required=True, help="Reason code for the revocation.")
    revoke_measurement_parser.add_argument("--note", default=None, help="Optional operator note.")

    restore_measurement_parser = subparsers.add_parser(
        "restore-measurement",
        help="Append a measurement restore event to a workbench run.",
    )
    restore_measurement_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    restore_measurement_parser.add_argument("--measurement-id", required=True, help="Measurement identifier.")
    restore_measurement_parser.add_argument("--reason", required=True, help="Reason code for the restore.")
    restore_measurement_parser.add_argument("--note", default=None, help="Optional operator note.")

    correct_measurement_parser = subparsers.add_parser(
        "correct-measurement",
        help="Append a measurement correction event to a workbench run.",
    )
    correct_measurement_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    correct_measurement_parser.add_argument("--measurement-id", required=True, help="Measurement identifier.")
    correct_measurement_parser.add_argument("--value", required=True, type=float, help="Corrected measurement value.")
    correct_measurement_parser.add_argument("--reason", required=True, help="Reason code for the correction.")
    correct_measurement_parser.add_argument("--note", default=None, help="Optional operator note.")

    change_association_parser = subparsers.add_parser(
        "change-association",
        help="Append a measurement association-change event to a workbench run.",
    )
    change_association_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    change_association_parser.add_argument("--source-measurement-id", required=True, help="Source measurement identifier.")
    change_association_parser.add_argument("--target-measurement-id", required=True, help="Target measurement identifier.")
    change_association_parser.add_argument("--reason", required=True, help="Reason code for the reassociation.")
    change_association_parser.add_argument("--note", default=None, help="Optional operator note.")

    replay_revision_parser = subparsers.add_parser(
        "replay-revision",
        help="Replay a workbench run from one revision to another.",
    )
    replay_revision_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    replay_revision_parser.add_argument("--from-revision", required=True, help="Source revision id.")
    replay_revision_parser.add_argument("--to-revision", required=True, help="Destination revision id.")

    diff_revisions_parser = subparsers.add_parser(
        "diff-revisions",
        help="Diff two workbench revisions and emit a revision-delta summary.",
    )
    diff_revisions_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    diff_revisions_parser.add_argument("--left", required=True, help="Left revision id.")
    diff_revisions_parser.add_argument("--right", required=True, help="Right revision id.")

    validate_replay_parser = subparsers.add_parser(
        "validate-replay",
        help="Validate replay determinism for one materialized revision.",
    )
    validate_replay_parser.add_argument("--run-dir", required=True, help="Workbench run directory.")
    validate_replay_parser.add_argument("--revision", required=True, help="Revision id to validate.")

    subparsers.add_parser(
        "list-runs",
        help="List runs recorded in the local workbench run registry.",
    )

    export_packet = subparsers.add_parser(
        "export-packet",
        help="Export a generated run into a presentation/review packet profile.",
    )
    export_packet.add_argument(
        "--profile",
        required=True,
        choices=("static_admissibility_mvp", "workbench", "presentation"),
        help="Packet profile to export.",
    )
    export_packet.add_argument(
        "--run-dir",
        required=True,
        help="Source run directory.",
    )
    export_packet.add_argument(
        "--output-dir",
        required=True,
        help="Destination packet directory.",
    )

    validate_packet = subparsers.add_parser(
        "validate-packet",
        help="Validate a generated packet.",
    )
    validate_packet.add_argument("packet_dir", nargs="?", default=None, help="Packet directory to validate.")
    validate_packet.add_argument("--packet-dir", dest="packet_dir_option", default=None, help="Packet directory to validate.")
    validate_packet.add_argument(
        "--profile",
        default="static_admissibility_mvp",
        choices=("static_admissibility_mvp", "corpus_explorer_mvp", "v7_anduril_c2_blend", "workbench", "presentation"),
        help="Packet validation profile.",
    )

    trajectory_exploration = subparsers.add_parser(
        "trajectory-exploration",
        help="Render the unified trajectory-exploration contract and benchmark bundle.",
    )
    trajectory_exploration.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the trajectory-exploration bundle should be written.",
    )
    trajectory_exploration.add_argument("--seed", type=int, default=7, help="Random seed for the benchmark runs.")

    trajectory_exploration_backend_registry = subparsers.add_parser(
        "trajectory-exploration-backend-registry",
        help="Render the trajectory-exploration backend registry bundle.",
    )
    trajectory_exploration_backend_registry.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the backend-registry bundle should be written.",
    )

    trajectory_exploration_objectives = subparsers.add_parser(
        "trajectory-exploration-objectives",
        help="Render the mechanically generated trajectory objective suite.",
    )
    trajectory_exploration_objectives.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the objective-suite bundle should be written.",
    )

    trajectory_control_surface_sweep = subparsers.add_parser(
        "trajectory-control-surface-sweep",
        help="Run the posterior-target objective across multiple 1D control-surface backends.",
    )
    trajectory_control_surface_sweep.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the control-surface sweep bundle should be written.",
    )
    trajectory_control_surface_sweep.add_argument("--seed", type=int, default=7, help="Base seed for backend candidates.")
    trajectory_control_surface_sweep.add_argument(
        "--random-candidates",
        type=int,
        default=24,
        help="Random candidates per backend.",
    )
    trajectory_control_surface_sweep.add_argument(
        "--cem-iterations",
        type=int,
        default=4,
        help="CEM iterations per backend.",
    )
    trajectory_control_surface_sweep.add_argument(
        "--cem-population",
        type=int,
        default=16,
        help="CEM population per iteration.",
    )

    trajectory_exploration_ppo = subparsers.add_parser(
        "trajectory-exploration-ppo",
        help="Render the sequential PPO boundary-control witness bundle.",
    )
    trajectory_exploration_ppo.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the PPO witness bundle should be written.",
    )
    trajectory_exploration_ppo.add_argument("--seed", type=int, default=7, help="Training seed for PPO.")
    trajectory_exploration_ppo.add_argument(
        "--episode-horizon",
        type=int,
        default=16,
        help="Episode horizon for the sequential control witness.",
    )
    trajectory_exploration_ppo.add_argument(
        "--timesteps",
        type=int,
        default=1024,
        help="Total PPO training timesteps.",
    )
    trajectory_exploration_ppo.add_argument(
        "--eval-episodes",
        type=int,
        default=8,
        help="Deterministic evaluation episodes after training.",
    )
    trajectory_exploration_ppo.add_argument(
        "--progress-eval-episodes",
        type=int,
        default=4,
        help="Evaluation episodes for periodic progress snapshots during training.",
    )
    trajectory_exploration_ppo.add_argument(
        "--checkpoint-interval",
        type=int,
        default=256,
        help="Timesteps between persisted PPO checkpoints.",
    )
    trajectory_exploration_ppo.add_argument(
        "--snapshot-interval",
        type=int,
        default=256,
        help="Timesteps between persisted progress snapshots.",
    )
    trajectory_exploration_ppo.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing PPO checkpoint in the target run directory.",
    )
    trajectory_exploration_ppo.add_argument(
        "--objective-id",
        default=None,
        help="Optional generated trajectory objective id to train against instead of the default boundary witness.",
    )

    trajectory_exploration_ppo_vs_cem = subparsers.add_parser(
        "trajectory-exploration-ppo-vs-cem",
        help="Run the matched-budget sequential PPO vs CEM comparison bundle.",
    )
    trajectory_exploration_ppo_vs_cem.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the PPO vs CEM comparison bundle should be written.",
    )
    trajectory_exploration_ppo_vs_cem.add_argument("--seed", type=int, default=7, help="Training seed for PPO.")
    trajectory_exploration_ppo_vs_cem.add_argument("--episode-horizon", type=int, default=16, help="Episode horizon for the sequential control witness.")
    trajectory_exploration_ppo_vs_cem.add_argument("--timesteps", type=int, default=1024, help="Total PPO training timesteps.")
    trajectory_exploration_ppo_vs_cem.add_argument("--eval-episodes", type=int, default=8, help="Deterministic evaluation episodes after training.")
    trajectory_exploration_ppo_vs_cem.add_argument("--progress-eval-episodes", type=int, default=4, help="Evaluation episodes for PPO progress snapshots.")
    trajectory_exploration_ppo_vs_cem.add_argument("--checkpoint-interval", type=int, default=256, help="Timesteps between persisted PPO checkpoints.")
    trajectory_exploration_ppo_vs_cem.add_argument("--snapshot-interval", type=int, default=256, help="Timesteps between persisted PPO progress snapshots.")
    trajectory_exploration_ppo_vs_cem.add_argument("--cem-iterations", type=int, default=10, help="Cross-entropy iterations for the open-loop CEM comparator.")
    trajectory_exploration_ppo_vs_cem.add_argument("--cem-population", type=int, default=24, help="Population size per CEM iteration.")
    trajectory_exploration_ppo_vs_cem.add_argument("--seed-count", type=int, default=1, help="Number of independent seeds to aggregate in the PPO vs CEM study.")
    trajectory_exploration_ppo_vs_cem.add_argument(
        "--objective-id",
        default=None,
        help="Optional generated trajectory objective id to compare against instead of the default boundary witness.",
    )

    trajectory_exploration_ppo_vs_cem_sweep = subparsers.add_parser(
        "trajectory-exploration-ppo-vs-cem-sweep",
        help="Run PPO vs CEM across generated feature/class-space objectives.",
    )
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--output-dir", default="artifacts", help="Directory where the objective-sweep bundle should be written.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--seed", type=int, default=7, help="Base seed for the objective sweep.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--seed-count", type=int, default=1, help="Number of independent seeds per objective.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--objective-limit", type=int, default=None, help="Optional cap on generated objectives to run.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--objective-id", action="append", dest="objective_ids", default=None, help="Optional generated objective id to include. Repeat to select a subset.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--episode-horizon", type=int, default=12, help="Episode horizon for each sequential control objective.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--timesteps", type=int, default=256, help="Total PPO training timesteps per objective and seed.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--eval-episodes", type=int, default=4, help="Deterministic PPO evaluation episodes.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--progress-eval-episodes", type=int, default=2, help="PPO progress snapshot evaluation episodes.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--cem-iterations", type=int, default=5, help="CEM iterations per objective and seed.")
    trajectory_exploration_ppo_vs_cem_sweep.add_argument("--cem-population", type=int, default=12, help="CEM population size per iteration.")

    trajectory_exploration_ppo_sweep = subparsers.add_parser(
        "trajectory-exploration-ppo-sweep-generated",
        help="Run PPO over the mechanically generated objective suite or a selected subset.",
    )
    trajectory_exploration_ppo_sweep.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the generated-objective PPO sweep bundle should be written.",
    )
    trajectory_exploration_ppo_sweep.add_argument("--seed", type=int, default=7, help="Training seed for PPO.")
    trajectory_exploration_ppo_sweep.add_argument("--timesteps", type=int, default=256, help="Total PPO training timesteps per generated objective.")
    trajectory_exploration_ppo_sweep.add_argument("--episode-horizon", type=int, default=12, help="Episode horizon for generated-objective PPO runs.")
    trajectory_exploration_ppo_sweep.add_argument("--eval-episodes", type=int, default=4, help="Deterministic evaluation episodes after training.")
    trajectory_exploration_ppo_sweep.add_argument(
        "--objective-id",
        action="append",
        dest="objective_ids",
        default=None,
        help="Optional generated objective id to include in the sweep. Repeat to select a subset.",
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

    if args.command == "exported-surface-coverage":
        artifacts = write_exported_surface_coverage_artifacts(
            Path(args.output_dir),
            materialize=args.materialize,
            surface_ids=args.surface_ids,
        )
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.coverage_matrix_path)
        print(artifacts.missing_coverage_path)
        print(artifacts.visualization_exemptions_path)
        print(artifacts.rerun_commands_path)
        print(artifacts.category_plot_path)
        print(artifacts.inventory_path)
        return 0

    if args.command == "corpus-evaluation-gap-matrix":
        artifacts = write_corpus_evaluation_gap_matrix_artifacts(
            Path(args.output_dir),
            materialize=args.materialize,
            capability_ids=args.capability_ids,
        )
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.matrix_path)
        print(artifacts.coherence_issues_path)
        print(artifacts.inventory_path)
        print(artifacts.status_plot_path)
        return 0

    if args.command == "algorithm-coverage-matrix":
        artifacts = write_algorithm_coverage_matrix_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.matrix_path)
        print(artifacts.inventory_path)
        print(artifacts.plot_path)
        return 0

    if args.command == "repo-shape-audit":
        artifacts = write_repo_shape_audit_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.issues_path)
        return 0

    if args.command == "filter-trace-validation":
        artifacts = write_filter_trace_validation_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.method_trace_matrix_path)
        print(artifacts.trace_requirement_matrix_path)
        print(artifacts.schema_path)
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

    if args.command == "methodology-section-symbol-audit":
        artifacts = write_methodology_section_symbol_audit_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.rows_path)
        return 0

    if args.command == "methodology-latex":
        artifacts = write_methodology_latex_artifacts(
            Path(args.output_dir),
            build_pdf=not args.no_pdf,
            artifact_mode="fast" if args.fast else "full",
        )
        print(artifacts.run_dir)
        print(artifacts.source_tex_path)
        print(artifacts.artifact_tex_path)
        if artifacts.pdf_path is not None:
            print(artifacts.pdf_path)
        return 0

    if args.command == "analysis-cache":
        if args.action == "clear":
            if not args.yes:
                raise SystemExit("refusing to clear analysis cache without --yes")
            result = clear_analysis_cache(namespace=args.namespace)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print("Analysis Cache Cleared")
                print(f"path: {result['cleared_path']}")
                print(f"namespace: {result['cleared_namespace'] or 'all'}")
                print(f"entries_removed: {result['cleared_entry_count']}")
                print(f"bytes_removed: {result['cleared_bytes']}")
            return 0

        summary = describe_analysis_cache(namespace=args.namespace)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print("Analysis Cache Summary")
            print(f"root: {summary['root']}")
            print(f"namespaces: {summary['namespace_count']}")
            print(f"entries: {summary['entry_count']}")
            print(f"bytes: {summary['bytes']}")
            for row in summary["namespaces"]:
                print(
                    f"- {row['namespace']}: entries={row['entry_count']} metadata={row['metadata_count']} bytes={row['bytes']}"
                )
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

    if args.command == "run-static-audit":
        config_path = args.bundle if args.bundle is not None else args.config
        packet = run_static_admissibility_audit(config_path, Path(args.output_dir))
        print(packet.packet_dir)
        print(packet.decision_card_path)
        print(packet.static_audit_report_path)
        print(packet.figure_manifest_path)
        return 0

    if args.command == "init-static-audit-bundle":
        bundle_dir = _init_static_audit_bundle(Path(args.output_dir))
        print(bundle_dir)
        print(bundle_dir / "static_audit_bundle.yaml")
        print(bundle_dir / "README.md")
        return 0

    if args.command == "run-static-audit-suite":
        packet = write_static_admissibility_exemplar_suite_packet(
            Path(args.output_dir),
            suite_manifest_path=args.suite_manifest,
        )
        print(packet.packet_dir)
        print(packet.decision_card_path)
        print(packet.hero_chart_manifest_path)
        return 0

    if args.command == "run-static-audit-multi-domain-3d":
        packet = write_multidomain_3d_static_admissibility_packet(Path(args.output_dir))
        print(packet.packet_dir)
        print(packet.decision_card_path)
        print(packet.hero_chart_manifest_path)
        return 0

    if args.command == "build-v7-final-packet":
        packet = write_v7_anduril_c2_blend_packet(Path(args.output_dir))
        print(packet.packet_dir)
        print(packet.decision_card_path)
        print(packet.hero_chart_manifest_path)
        return 0

    if args.command == "validate-correctness":
        return run_correctness_plan(args.level)

    if args.command == "validate-study":
        validation = validate_study_spec(args.study_spec)
        if validation.issues:
            for issue in validation.issues:
                print(f"FAIL: {issue}")
            return 1
        print(f"PASS: {validation.path}")
        return 0

    if args.command == "run-study":
        run = run_workbench_study(
            args.study_spec,
            args.output_dir,
            seed=args.seed,
            trajectories_per_case=args.trajectories_per_case,
        )
        print(run.run_dir)
        print(run.manifest_path)
        print(run.decision_card_path)
        print(run.report_path)
        return 0

    if args.command == "search-corpus":
        run_dir = search_corpus(args.config, args.output_dir)
        print(run_dir)
        print(run_dir / "corpus_search_manifest.json")
        print(run_dir / "backend_comparison.csv")
        return 0

    if args.command == "analyze-run":
        run = analyze_workbench_run(args.run_dir)
        print(run.run_dir)
        print(run.decision_card_path)
        print(run.report_path)
        return 0

    if args.command == "compare-rungs":
        decision_path = compare_rungs(args.run_dir)
        print(decision_path)
        return 0

    if args.command == "build-epic1-showcase":
        packet = build_epic1_showcase(
            args.output_dir,
            study_spec=args.study_spec,
            corpus_search_config=args.corpus_search_config,
            presentation_output_dir=args.presentation_output_dir,
            seed=args.seed,
            trajectories_per_case=args.trajectories_per_case,
            include_static=not args.skip_static,
            include_presentation=not args.skip_presentation,
        )
        print(packet.packet_dir)
        print(packet.summary_path)
        print(packet.manifest_path)
        return 0

    if args.command == "inspect-run":
        print(inspect_run(args.run_dir))
        return 0

    if args.command == "inspect-measurement":
        ensure_revision_history(args.run_dir)
        payload = inspect_measurement(args.run_dir, args.measurement_id, revision_id=args.revision_id)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "revoke-measurement":
        ensure_revision_history(args.run_dir)
        revision_id = revoke_measurement(
            args.run_dir,
            args.measurement_id,
            reason=args.reason,
            note=args.note,
        )
        print(revision_id)
        return 0

    if args.command == "restore-measurement":
        ensure_revision_history(args.run_dir)
        revision_id = restore_measurement(
            args.run_dir,
            args.measurement_id,
            reason=args.reason,
            note=args.note,
        )
        print(revision_id)
        return 0

    if args.command == "correct-measurement":
        ensure_revision_history(args.run_dir)
        revision_id = correct_measurement(
            args.run_dir,
            args.measurement_id,
            corrected_value=args.value,
            reason=args.reason,
            note=args.note,
        )
        print(revision_id)
        return 0

    if args.command == "change-association":
        ensure_revision_history(args.run_dir)
        revision_id = change_measurement_association(
            args.run_dir,
            args.source_measurement_id,
            args.target_measurement_id,
            reason=args.reason,
            note=args.note,
        )
        print(revision_id)
        return 0

    if args.command == "replay-revision":
        ensure_revision_history(args.run_dir)
        revision_dir = replay_revision(args.run_dir, args.from_revision, args.to_revision)
        print(revision_dir)
        print(Path(revision_dir) / "revision_delta.md")
        return 0

    if args.command == "diff-revisions":
        ensure_revision_history(args.run_dir)
        payload = diff_revisions(args.run_dir, args.left, args.right)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-replay":
        ensure_revision_history(args.run_dir)
        payload = validate_replay(args.run_dir, args.revision)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "list-runs":
        rows = list_runs()
        for row in rows:
            print(
                f"{row['run_id']}\t{row['study_id']}\t{row['status']}\t"
                f"{row['decision']}\t{row['run_dir']}"
            )
        return 0

    if args.command == "export-packet":
        if args.profile == "static_admissibility_mvp":
            packet = export_static_admissibility_packet(args.run_dir, args.output_dir)
            print(packet.packet_dir)
            print(packet.decision_card_path)
            print(packet.figure_manifest_path)
            return 0
        if args.profile == "workbench":
            packet_dir = export_workbench_packet(args.run_dir, args.output_dir)
            print(packet_dir)
            print(packet_dir / "decision_card.md")
            print(packet_dir / "workbench_report.md")
            return 0
        if args.profile == "presentation":
            packet_dir = export_presentation_packet(args.output_dir, run_dir=args.run_dir)
            print(packet_dir)
            print(packet_dir / "decision_card.md")
            print(packet_dir / "hero_chart_manifest.csv")
            return 0

    if args.command == "validate-packet":
        packet_dir = args.packet_dir_option if args.packet_dir_option is not None else args.packet_dir
        if packet_dir is None:
            raise SystemExit("validate-packet requires a packet directory")
        if args.profile == "static_admissibility_mvp":
            issues = validate_static_admissibility_packet(packet_dir, repo_root=repo_root())
            if issues:
                for issue in issues:
                    print(f"FAIL: {issue}")
                return 1
            print(f"PASS: {packet_dir}")
            return 0
        if args.profile == "corpus_explorer_mvp":
            issues = validate_corpus_explorer_packet(packet_dir)
            if issues:
                for issue in issues:
                    print(f"FAIL: {issue}")
                return 1
            print(f"PASS: {packet_dir}")
            return 0
        if args.profile == "v7_anduril_c2_blend":
            issues = validate_v7_anduril_c2_blend_packet(packet_dir)
            if issues:
                for issue in issues:
                    print(f"FAIL: {issue}")
                return 1
            print(f"PASS: {packet_dir}")
            return 0
        if args.profile == "workbench":
            validation = validate_workbench_run(packet_dir)
            if validation.issues:
                for issue in validation.issues:
                    print(f"FAIL: {issue}")
                return 1
            print(f"PASS: {packet_dir}")
            return 0
        if args.profile == "presentation":
            import subprocess

            result = subprocess.run(
                [
                    "python3",
                    str(repo_root() / "scripts" / "audit" / "validate_presentation_hero_packet.py"),
                    "--packet-dir",
                    packet_dir,
                ],
                cwd=repo_root(),
                text=True,
            )
            return result.returncode

    if args.command == "trajectory-exploration":
        artifacts = write_trajectory_exploration_artifacts(
            Path(args.output_dir),
            seed=args.seed,
        )
        print(artifacts.contract_dir)
        print(artifacts.backend_contract_path)
        print(artifacts.metrics_by_backend_path)
        print(artifacts.rl_decision_report_path)
        print(artifacts.optimizer_trace_path)
        return 0

    if args.command == "trajectory-exploration-backend-registry":
        artifacts = write_exploration_backend_registry_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
        print(artifacts.spec_table_path)
        print(artifacts.family_summary_path)
        print(artifacts.capability_plot_path)
        return 0

    if args.command == "trajectory-exploration-objectives":
        artifacts = write_generated_trajectory_objective_artifacts(Path(args.output_dir))
        print(artifacts.run_dir)
        print(artifacts.spec_path)
        print(artifacts.manifest_path)
        print(artifacts.objectives_path)
        print(artifacts.objective_table_path)
        print(artifacts.report_path)
        if artifacts.proof_artifacts is not None:
            print(artifacts.proof_artifacts.run_dir)
            print(artifacts.proof_artifacts.posterior_target_spec_path)
            print(artifacts.proof_artifacts.proof_ladder_path)
            print(artifacts.proof_artifacts.report_path)
        return 0

    if args.command == "trajectory-control-surface-sweep":
        artifacts = write_control_surface_backend_sweep_artifacts(
            Path(args.output_dir),
            config=ControlSurfaceBackendSweepConfig(
                seed=args.seed,
                random_candidates_per_backend=args.random_candidates,
                cem_iterations=args.cem_iterations,
                cem_population=args.cem_population,
            ),
        )
        print(artifacts.run_dir)
        print(artifacts.config_path)
        print(artifacts.control_surface_manifest_path)
        print(artifacts.backend_capability_matrix_path)
        print(artifacts.backend_objective_achievability_path)
        print(artifacts.posterior_target_backend_sweep_path)
        print(artifacts.target_vs_achieved_posterior_path)
        print(artifacts.generator_identification_probe_path)
        print(artifacts.backend_identification_probe_path)
        print(artifacts.backend_identification_confusion_path)
        print(artifacts.observation_surface_manifest_path)
        print(artifacts.achievability_plot_path)
        print(artifacts.posterior_plot_path)
        print(artifacts.backend_probe_plot_path)
        print(artifacts.report_path)
        return 0

    if args.command == "trajectory-exploration-ppo":
        objective = None if args.objective_id is None else resolve_generated_trajectory_objective(args.objective_id)
        result = write_sequential_ppo_boundary_control_artifacts(
            Path(args.output_dir),
            config=SequentialBoundaryControlConfig(episode_horizon=args.episode_horizon),
            ppo_config=SequentialPpoConfig(
                total_timesteps=args.timesteps,
                train_seed=args.seed,
                eval_seed_start=args.seed + 200,
                eval_episodes=args.eval_episodes,
                progress_eval_episodes=args.progress_eval_episodes,
                checkpoint_interval_timesteps=args.checkpoint_interval,
                snapshot_interval_timesteps=args.snapshot_interval,
                resume_if_possible=not args.no_resume,
            ),
            objective=objective,
        )
        if result.artifacts is None:
            parser.error("trajectory-exploration-ppo did not produce an artifact bundle")
        print(result.artifacts.run_dir)
        print(result.artifacts.checkpoints_dir)
        print(result.artifacts.environment_contract_path)
        print(result.artifacts.control_problem_contract_path)
        print(result.artifacts.transition_report_path)
        print(result.artifacts.checkpoint_manifest_path)
        print(result.artifacts.training_summary_path)
        print(result.artifacts.training_trace_rows_path)
        print(result.artifacts.snapshot_rows_path)
        print(result.artifacts.ppo_vs_heuristics_path)
        print(result.artifacts.utility_progress_path)
        print(result.artifacts.feature_progress_path)
        print(result.artifacts.class_space_progress_path)
        print(result.artifacts.report_path)
        print(result.artifacts.rl_algorithm_decision_report_path)
        return 0

    if args.command == "trajectory-exploration-ppo-sweep-generated":
        artifacts = write_generated_trajectory_objective_ppo_sweep_artifacts(
            Path(args.output_dir),
            config=SequentialBoundaryControlConfig(episode_horizon=args.episode_horizon),
            ppo_config=SequentialPpoConfig(
                total_timesteps=args.timesteps,
                train_seed=args.seed,
                eval_seed_start=args.seed + 200,
                eval_episodes=args.eval_episodes,
            ),
            objective_ids=None if args.objective_ids is None else tuple(args.objective_ids),
        )
        print(artifacts.run_dir)
        print(artifacts.manifest_path)
        print(artifacts.summary_rows_path)
        print(artifacts.report_path)
        return 0

    if args.command == "trajectory-exploration-ppo-vs-cem":
        objective = None if args.objective_id is None else resolve_generated_trajectory_objective(args.objective_id)
        result = write_sequential_ppo_vs_cem_comparison_artifacts(
            Path(args.output_dir),
            config=SequentialBoundaryControlConfig(episode_horizon=args.episode_horizon),
            ppo_config=SequentialPpoConfig(
                total_timesteps=args.timesteps,
                train_seed=args.seed,
                eval_seed_start=args.seed + 200,
                eval_episodes=args.eval_episodes,
                progress_eval_episodes=args.progress_eval_episodes,
                checkpoint_interval_timesteps=args.checkpoint_interval,
                snapshot_interval_timesteps=args.snapshot_interval,
            ),
            cem_config=SequentialCemConfig(
                iterations=args.cem_iterations,
                population_size=args.cem_population,
                eval_seed_start=args.seed + 400,
            ),
            objective=objective,
            seed_count=args.seed_count,
            base_seed=args.seed,
        )
        if result.artifacts is None:
            parser.error("trajectory-exploration-ppo-vs-cem did not produce an artifact bundle")
        print(result.artifacts.run_dir)
        print(result.artifacts.config_path)
        print(result.artifacts.artifact_manifest_path)
        print(result.artifacts.backend_metrics_path)
        print(result.artifacts.aggregate_backend_metrics_path)
        print(result.artifacts.backend_decisions_path)
        print(result.artifacts.seed_runs_path)
        print(result.artifacts.evaluation_rows_path)
        print(result.artifacts.progress_rows_path)
        print(result.artifacts.strengths_limits_path)
        print(result.artifacts.progress_plot_path)
        print(result.artifacts.backend_metrics_plot_path)
        print(result.artifacts.control_gallery_path)
        print(result.artifacts.report_path)
        return 0

    if args.command == "trajectory-exploration-ppo-vs-cem-sweep":
        result = write_sequential_objective_sweep_comparison_artifacts(
            Path(args.output_dir),
            config=SequentialBoundaryControlConfig(episode_horizon=args.episode_horizon),
            ppo_config=SequentialPpoConfig(
                total_timesteps=args.timesteps,
                train_seed=args.seed,
                eval_seed_start=args.seed + 200,
                eval_episodes=args.eval_episodes,
                progress_eval_episodes=args.progress_eval_episodes,
                checkpoint_interval_timesteps=max(args.timesteps, 1),
                snapshot_interval_timesteps=max(args.timesteps, 1),
            ),
            cem_config=SequentialCemConfig(
                iterations=args.cem_iterations,
                population_size=args.cem_population,
                eval_seed_start=args.seed + 400,
            ),
            objective_ids=None if args.objective_ids is None else tuple(args.objective_ids),
            objective_limit=args.objective_limit,
            seed_count=args.seed_count,
            base_seed=args.seed,
        )
        if result.artifacts is None:
            parser.error("trajectory-exploration-ppo-vs-cem-sweep did not produce an artifact bundle")
        print(result.artifacts.run_dir)
        print(result.artifacts.config_path)
        print(result.artifacts.artifact_manifest_path)
        print(result.artifacts.objective_summary_path)
        print(result.artifacts.backend_summary_path)
        print(result.artifacts.decision_summary_path)
        print(result.artifacts.objective_backend_matrix_path)
        print(result.artifacts.objective_backend_heatmap_path)
        print(result.artifacts.report_path)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
