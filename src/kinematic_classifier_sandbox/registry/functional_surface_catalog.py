from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.io import write_csv
from ..utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class FunctionalSurfaceSpec:
    surface_id: str
    category: str
    module: str
    analysis_function: str
    artifact_function: str
    artifact_outputs: tuple[str, ...]
    evaluation_mode: str
    showcase_priority: str
    notes: str


@dataclass(frozen=True, slots=True)
class FunctionalSurfaceRow:
    surface_id: str
    category: str
    module: str
    analysis_function: str
    artifact_function: str
    artifact_outputs: tuple[str, ...]
    evaluation_mode: str
    showcase_priority: str
    analysis_callable: bool
    artifact_callable: bool
    artifact_output_count: int
    notes: str


@dataclass(frozen=True, slots=True)
class FunctionalSurfaceCatalogResult:
    surface_rows: tuple[FunctionalSurfaceRow, ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class FunctionalSurfaceCatalogArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    catalog_path: Path
    plot_path: Path


FUNCTIONAL_SURFACE_REGISTRY: tuple[FunctionalSurfaceSpec, ...] = (
    FunctionalSurfaceSpec(
        surface_id="feature_analysis",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.feature_analysis",
        analysis_function="analyze_feature_datasets",
        artifact_function="write_feature_analysis_artifacts",
        artifact_outputs=("feature_excitation_matrix.csv", "feature_excitation_summary.json", "feature_excitation_heatmap.png"),
        evaluation_mode="pure_analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Best example of a pure analysis function with a clean artifact writer shell.",
    ),
    FunctionalSurfaceSpec(
        surface_id="pca_analysis",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.pca_analysis",
        analysis_function="analyze_feature_pca",
        artifact_function="write_pca_analysis_artifacts",
        artifact_outputs=("coordinates.csv", "loadings.csv", "explained_variance.csv", "scatter.png"),
        evaluation_mode="pure_analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Standard pattern: deterministic analysis result plus markdown/plot artifact writer.",
    ),
    FunctionalSurfaceSpec(
        surface_id="pca_dimensionality_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.pca_dimensionality_audit",
        analysis_function="analyze_pca_dimensionality",
        artifact_function="write_pca_dimensionality_audit_artifacts",
        artifact_outputs=("pca_component_sweep.csv", "pca_clusterability.csv", "variance_vs_error.png", "clusterability.png"),
        evaluation_mode="pure_analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Good example of analysis-driven recommendation with sweep artifacts.",
    ),
    FunctionalSurfaceSpec(
        surface_id="static_feature_class_prior_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit",
        analysis_function="analyze_default_static_feature_class_prior_audit",
        artifact_function="write_static_feature_class_prior_audit_artifacts",
        artifact_outputs=(
            "static_audit_report.md",
            "static_decision_card.md",
            "02b_static_audit_decision_card.png",
            "02c_class_pair_confusability_matrix.png",
            "02d_feature_relevance_rank.png",
            "02e_feature_redundancy_graph.png",
            "02f_feature_synergy_map.png",
            "02g_prior_pathology_surface.png",
            "02h_prior_flip_thresholds.png",
            "02i_static_coverage_feasibility.png",
            "02j_static_leakage_provenance_audit.png",
            "02k_static_audit_to_action_router.png",
            "class_confusability_matrix.csv",
            "feature_relevance_table.csv",
            "feature_redundancy_matrix.csv",
            "feature_synergy_candidates.csv",
            "prior_pathology_report.csv",
            "prior_flip_thresholds.csv",
            "static_coverage_feasibility.csv",
            "static_leakage_provenance_audit.csv",
        ),
        evaluation_mode="pure_analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Front-door admissibility gate for feature relevance, class separability, prior pathology, coverage, leakage, and decisionability.",
    ),
    FunctionalSurfaceSpec(
        surface_id="embedding_baseline_frontier",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.embedding_baseline_frontier",
        analysis_function="analyze_embedding_baseline_frontier",
        artifact_function="write_embedding_baseline_frontier_artifacts",
        artifact_outputs=("view_summary.csv", "embedding_summary.csv", "prediction_summary.csv", "metric_summary.csv", "online_route_summary.csv", "accuracy_bars.png", "online_route_curve.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="First TS2Vec-style embedding witness packet with a shared 1D dynamics corpus, downstream embedding heads, and a prefix-based online route proof.",
    ),
    FunctionalSurfaceSpec(
        surface_id="ts2vec_backend_parity",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.ts2vec_backend_parity",
        analysis_function="analyze_ts2vec_backend_parity",
        artifact_function="write_ts2vec_backend_parity_artifacts",
        artifact_outputs=("prediction_summary.csv", "metric_summary.csv", "summary.csv", "metrics.csv", "ts2vec_backend_parity_report.md", "decision_card.md", "backend_accuracy.png", "backend_gap_summary.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Bounded proxy-versus-external TS2Vec parity witness on the shared 1D corpus so the learned-embedding lane can report backend fidelity honestly.",
    ),
    FunctionalSurfaceSpec(
        surface_id="shapelet_maneuver_motif",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.shapelet_motif_witness",
        analysis_function="analyze_shapelet_maneuver_motif_witness",
        artifact_function="write_shapelet_maneuver_motif_witness_artifacts",
        artifact_outputs=("trajectory_summary.csv", "prediction_summary.csv", "activation_summary.csv", "summary.csv", "metrics.csv", "shapelet_maneuver_motif_report.md", "decision_card.md", "motif_examples.png", "distance_profiles.png", "metric_bars.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Dedicated shapelet motif witness with alignment evidence, per-trajectory activation traces, and a robust decision card.",
    ),
    FunctionalSurfaceSpec(
        surface_id="tsc_archive_baseline_frontier",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.tsc_archive_frontier",
        analysis_function="analyze_tsc_archive_baseline_frontier",
        artifact_function="write_tsc_archive_baseline_frontier_artifacts",
        artifact_outputs=("prediction_summary.csv", "metric_summary.csv", "seed_sweep_summary.csv", "summary.csv", "metrics.csv", "tsc_archive_baseline_frontier_report.md", "decision_card.md", "overall_accuracy.png", "scenario_slice_accuracy.png", "calibration_surface.png", "seed_stability.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="First-class archive-family comparison surface with optional aeon/sktime backends, local fallbacks when absent, and a bounded seed/calibration read that still stops short of witness promotion.",
    ),
    FunctionalSurfaceSpec(
        surface_id="archive_vs_physics_witness",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.archive_vs_physics_witness",
        analysis_function="analyze_archive_vs_physics_witness",
        artifact_function="write_archive_vs_physics_witness_artifacts",
        artifact_outputs=("method_summary.csv", "scenario_winners.csv", "summary.csv", "metrics.csv", "archive_vs_physics_witness_report.md", "decision_card.md", "archive_vs_physics_accuracy.png", "archive_vs_physics_gaps.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Named shared-corpus archive-versus-baseline witness that compares archive family rows against interpretable and physics-aware baselines while keeping the gate closed under fallback-backed execution.",
    ),
    FunctionalSurfaceSpec(
        surface_id="archive_feature_headroom_witness",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.archive_feature_headroom_witness",
        analysis_function="analyze_archive_feature_headroom_witness",
        artifact_function="write_archive_feature_headroom_witness_artifacts",
        artifact_outputs=("method_summary.csv", "seed_sweep_summary.csv", "summary.csv", "metrics.csv", "archive_feature_headroom_witness_report.md", "decision_card.md", "archive_feature_headroom_accuracy.png", "archive_feature_headroom_gaps.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Timing-order archive-family witness against interpretable and engineered-feature baselines on the feature-headroom dataset.",
    ),
    FunctionalSurfaceSpec(
        surface_id="archive_backend_diagnosis",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.archive_backend_diagnosis",
        analysis_function="analyze_archive_backend_diagnosis",
        artifact_function="write_archive_backend_diagnosis_artifacts",
        artifact_outputs=("diagnosis_rows.csv", "summary_rows.csv", "summary.csv", "metrics.csv", "archive_backend_diagnosis_report.md", "decision_card.md", "best_accuracy_by_variant.png", "warning_load_by_method.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Bounded diagnosis packet for archive-family panel variants, resampling lengths, and external warning load across the current 1D witnesses.",
    ),
    FunctionalSurfaceSpec(
        surface_id="archive_family_promotion_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.archive_family_promotion_audit",
        analysis_function="analyze_archive_family_promotion_audit",
        artifact_function="write_archive_family_promotion_audit_artifacts",
        artifact_outputs=("method_summary.csv", "summary.csv", "metrics.csv", "archive_family_promotion_audit_report.md", "decision_card.md", "promotion_deltas.png", "diagnosis_best_accuracy.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Method-level ranking surface for the generic-TSC lane that identifies the closest archive family candidate and states the remaining bounded blockers.",
    ),
    FunctionalSurfaceSpec(
        surface_id="drcif_interval_promotion_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.drcif_interval_promotion_audit",
        analysis_function="analyze_drcif_interval_promotion_audit",
        artifact_function="write_drcif_interval_promotion_audit_artifacts",
        artifact_outputs=("audit_rows.csv", "summary.csv", "metrics.csv", "drcif_interval_promotion_audit_report.md", "decision_card.md", "promotion_deltas.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Narrow method-level audit for the remaining DrCIF holdout so parity-level evidence is separated from true witness support.",
    ),
    FunctionalSurfaceSpec(
        surface_id="gsf_multimodal_promotion_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.gsf_multimodal_promotion_audit",
        analysis_function="analyze_gsf_multimodal_promotion_audit",
        artifact_function="write_gsf_multimodal_promotion_audit_artifacts",
        artifact_outputs=("audit_rows.csv", "summary.csv", "metrics.csv", "gsf_multimodal_promotion_audit_report.md", "decision_card.md", "promotion_metrics.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Narrow method-level audit for the multimodal Gaussian-mixture blocker so GSF can be promoted or held back explicitly before PF escalation.",
    ),
    FunctionalSurfaceSpec(
        surface_id="ukf_nonlinear_promotion_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.ukf_nonlinear_promotion_audit",
        analysis_function="analyze_ukf_nonlinear_promotion_audit",
        artifact_function="write_ukf_nonlinear_promotion_audit_artifacts",
        artifact_outputs=("audit_rows.csv", "summary.csv", "metrics.csv", "ukf_nonlinear_promotion_audit_report.md", "decision_card.md", "promotion_ratios.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Narrow method-level audit for the nonlinear-but-unimodal Gaussian blocker so UKF can be promoted or held back explicitly before mixture or particle escalation.",
    ),
    FunctionalSurfaceSpec(
        surface_id="imm_switching_promotion_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.imm_switching_promotion_audit",
        analysis_function="analyze_imm_switching_promotion_audit",
        artifact_function="write_imm_switching_promotion_audit_artifacts",
        artifact_outputs=("audit_rows.csv", "summary.csv", "metrics.csv", "imm_switching_promotion_audit_report.md", "decision_card.md", "promotion_deltas.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Narrow method-level audit for switching state-mixing so IMM can be promoted or held back explicitly on bounded switch-recovery sweeps.",
    ),
    FunctionalSurfaceSpec(
        surface_id="tsc_archive_backend_smoke",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.tsc_archive_backend_smoke",
        analysis_function="analyze_tsc_archive_backend_smoke",
        artifact_function="write_tsc_archive_backend_smoke_artifacts",
        artifact_outputs=("backend_smoke_rows.csv", "metrics.csv", "tsc_archive_backend_smoke_report.md", "decision_card.md"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Tiny timeout-bounded external backend probe for the archive-family lane so the repo can distinguish unavailable, failed, timed-out, and successful external wrappers.",
    ),
    FunctionalSurfaceSpec(
        surface_id="neural_sequence_vs_physics_frontier",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.neural_sequence_frontier",
        analysis_function="analyze_neural_sequence_vs_physics_frontier",
        artifact_function="write_neural_sequence_vs_physics_frontier_artifacts",
        artifact_outputs=("prediction_summary.csv", "metric_summary.csv", "training_curve.csv", "summary.csv", "metrics.csv", "neural_sequence_vs_physics_frontier_report.md", "decision_card.md", "frontier_test_accuracy.png", "scenario_slice_accuracy.png", "training_curves.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="First-class neural sequence frontier with real local training, calibration, and held-out evaluation surfaces.",
    ),
    FunctionalSurfaceSpec(
        surface_id="neural_sequence_robustness_frontier",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.neural_sequence_robustness",
        analysis_function="analyze_neural_sequence_robustness_frontier",
        artifact_function="write_neural_sequence_robustness_frontier_artifacts",
        artifact_outputs=("seed_summary.csv", "metric_summary.csv", "summary.csv", "metrics.csv", "neural_sequence_robustness_frontier_report.md", "decision_card.md", "mean_accuracy_with_variance.png", "seed_winner_counts.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Bounded multi-seed robustness companion for the trained neural sequence lane so the family is not represented only by a single-seed frontier.",
    ),
    FunctionalSurfaceSpec(
        surface_id="physics_family_promotion_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.analysis.physics_family_promotion_audit",
        analysis_function="analyze_physics_family_promotion_audit",
        artifact_function="write_physics_family_promotion_audit_artifacts",
        artifact_outputs=("method_summary.csv", "summary.csv", "metrics.csv", "physics_family_promotion_audit_report.md", "decision_card.md", "closure_status.png", "blocker_summary.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Bounded family-level closure audit for the physics-aware lane, focused on the remaining witness-supported methods that still block Epic 2 closure.",
    ),
    FunctionalSurfaceSpec(
        surface_id="corpus_adequacy_audit",
        category="evaluation",
        module="kinematic_classifier_sandbox.corpus.adequacy_audit",
        analysis_function="analyze_corpus_adequacy",
        artifact_function="write_corpus_adequacy_artifacts",
        artifact_outputs=("class_pair_coverage.csv", "covariate_leakage.csv", "pair_status_heatmap.png"),
        evaluation_mode="pure_analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Excellent example of gating analysis with compact reportable outputs.",
    ),
    FunctionalSurfaceSpec(
        surface_id="coverage_report",
        category="evaluation",
        module="kinematic_classifier_sandbox.corpus.coverage_report",
        analysis_function="analyze_coverage_report",
        artifact_function="write_coverage_report_artifacts",
        artifact_outputs=("feature_set_summary.csv", "feature_group_summary.csv", "classifier_support.csv"),
        evaluation_mode="pure_analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Bridges corpus adequacy and classifier readiness into one report surface.",
    ),
    FunctionalSurfaceSpec(
        surface_id="corpus_autodevelopment",
        category="corpus",
        module="kinematic_classifier_sandbox.corpus.autodevelopment",
        analysis_function="analyze_corpus_autodevelopment",
        artifact_function="write_corpus_autodevelopment_artifacts",
        artifact_outputs=("candidate_corpus_manifest.csv", "selected_corpus_manifest.json", "feature_excitation_heatmap.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Shows a full candidate-evaluate-select loop with strong artifact coverage.",
    ),
    FunctionalSurfaceSpec(
        surface_id="generated_corpus_features",
        category="corpus",
        module="kinematic_classifier_sandbox.analysis.generated_corpus_features",
        analysis_function="analyze_generated_corpus_features",
        artifact_function="write_generated_corpus_feature_artifacts",
        artifact_outputs=("feature_matrix.csv", "feature_manifest.json", "selected_record_manifest.csv"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="medium",
        notes="Good showcase for connecting selected corpus rows to feature matrices.",
    ),
    FunctionalSurfaceSpec(
        surface_id="corpus_classifier_scoring",
        category="corpus",
        module="kinematic_classifier_sandbox.corpus.classifier_scoring",
        analysis_function="analyze_corpus_classifier_scoring",
        artifact_function="write_corpus_classifier_scoring_artifacts",
        artifact_outputs=("candidate_scores.csv", "posterior_history.csv", "stress_plot.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="medium",
        notes="Demonstrates classifier-in-the-loop scoring and posterior traces.",
    ),
    FunctionalSurfaceSpec(
        surface_id="methodology_latex",
        category="documentation",
        module="kinematic_classifier_sandbox.methodology.latex",
        analysis_function="analyze_methodology_latex",
        artifact_function="write_methodology_latex_artifacts",
        artifact_outputs=("kinematic_classifier_methodology.tex", "algorithm_ladder_table.tex", "methodology_latex.pdf"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Best example of analysis-to-document generation with rendered PDF output.",
    ),
    FunctionalSurfaceSpec(
        surface_id="advanced_filter_decision",
        category="decision_gate",
        module="kinematic_classifier_sandbox.validation.advanced_filter_decision",
        analysis_function="analyze_advanced_filter_decision",
        artifact_function="write_advanced_filter_decision_artifacts",
        artifact_outputs=("advanced_filter_decision_report.md", "advanced_filter_decision_summary.json", "numeric_walkthrough.md"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Great for showing pure decision evidence plus a numeric walkthrough artifact.",
    ),
    FunctionalSurfaceSpec(
        surface_id="generic_inference_contract",
        category="contract",
        module="kinematic_classifier_sandbox.methodology.inference_contract",
        analysis_function="analyze_generic_inference_contract",
        artifact_function="write_generic_inference_contract_artifacts",
        artifact_outputs=("classifier_output_schema.json", "posterior_history_schema.json", "validation_results.json"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="medium",
        notes="Strong schema-and-contract example; mostly pure analysis with validation artifacts.",
    ),
    FunctionalSurfaceSpec(
        surface_id="algorithm_coverage_matrix",
        category="registry",
        module="kinematic_classifier_sandbox.registry.algorithm_coverage_matrix",
        analysis_function="analyze_algorithm_coverage_matrix",
        artifact_function="write_algorithm_coverage_matrix_artifacts",
        artifact_outputs=("algorithm_coverage_matrix_report.md", "algorithm_coverage_matrix.csv", "algorithm_coverage_matrix.png"),
        evaluation_mode="registry_plus_artifact_shell",
        showcase_priority="high",
        notes="Broader algorithm-lane coverage map that keeps the repo explicit about tracked methods beyond the proof ladder.",
    ),
    FunctionalSurfaceSpec(
        surface_id="classifier_family_scorecard",
        category="decision_gate",
        module="kinematic_classifier_sandbox.analysis.classifier_family_scorecard",
        analysis_function="analyze_classifier_family_scorecard",
        artifact_function="write_classifier_family_scorecard_artifacts",
        artifact_outputs=("capability_matrix.csv", "ceiling_efficiency.csv", "family_summary.csv", "classifier_family_atlas.md", "classifier_family_scorecard_report.md", "classifier_efficiency_vs_epic1_proxy_ceiling.png"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="high",
        notes="Epic 2 family scorecard tying capability claims to bounded ceiling-relative evidence and explicit method holdouts.",
    ),
    FunctionalSurfaceSpec(
        surface_id="method_validation_os",
        category="registry",
        module="kinematic_classifier_sandbox.registry.method_validation_os",
        analysis_function="analyze_method_validation_os",
        artifact_function="write_method_validation_os_artifacts",
        artifact_outputs=("method_specs.json", "algorithm_promotion_status_matrix.csv", "witness_to_method_coverage_matrix.csv", "lane_summary.csv", "epic2_family_maturity_matrix.csv"),
        evaluation_mode="registry_plus_artifact_shell",
        showcase_priority="high",
        notes="Method-validation operating system tying lanes, statuses, and witnesses together.",
    ),
    FunctionalSurfaceSpec(
        surface_id="formal_math_registry",
        category="registry",
        module="kinematic_classifier_sandbox.registry.formal_math_registry",
        analysis_function="analyze_formal_math_registry",
        artifact_function="write_formal_math_registry_artifacts",
        artifact_outputs=("function_registry.csv", "equation_registry.csv", "function_equation_crosswalk.csv", "formal_math_registry_role_counts.png"),
        evaluation_mode="source_scanned_registry_plus_artifact_shell",
        showcase_priority="high",
        notes="Formal source-scanned registry linking helper functions to equation implementations and artifact coverage.",
    ),
    FunctionalSurfaceSpec(
        surface_id="formal_math_visual_registry",
        category="registry",
        module="kinematic_classifier_sandbox.registry.formal_math_visual_registry",
        analysis_function="analyze_formal_math_visual_registry",
        artifact_function="write_formal_math_visual_registry_artifacts",
        artifact_outputs=("formal_math_visual_registry.csv", "formal_math_visual_registry_coverage.png", "assets/*.png"),
        evaluation_mode="source_scanned_registry_plus_artifact_shell",
        showcase_priority="high",
        notes="Visual gallery companion to the formal math registry; shows representative plots for each equation.",
    ),
    FunctionalSurfaceSpec(
        surface_id="trajectory_exploration_backend_registry",
        category="corpus",
        module="kinematic_classifier_sandbox.corpus.trajectory_exploration.backend_registry",
        analysis_function="analyze_exploration_backend_registry",
        artifact_function="write_exploration_backend_registry_artifacts",
        artifact_outputs=("report.md", "backend_registry.csv", "family_summary.csv", "capability_matrix.png"),
        evaluation_mode="registry_plus_artifact_shell",
        showcase_priority="medium",
        notes="Registry surface for current and planned exploration/generator backends under one benchmark contract.",
    ),
    FunctionalSurfaceSpec(
        surface_id="generic_feature_taxonomy",
        category="taxonomy",
        module="kinematic_classifier_sandbox.methodology.feature_taxonomy",
        analysis_function="analyze_generic_feature_taxonomy",
        artifact_function="write_generic_feature_taxonomy_artifacts",
        artifact_outputs=("feature_taxonomy_report.md", "feature_taxonomy.json"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="medium",
        notes="Useful for feature-family coverage and structural transfer discussion.",
    ),
    FunctionalSurfaceSpec(
        surface_id="generic_filtering_contract",
        category="contract",
        module="kinematic_classifier_sandbox.methodology.filtering_contract",
        analysis_function="analyze_generic_filtering_contract",
        artifact_function="write_generic_filtering_contract_artifacts",
        artifact_outputs=("filter_backend_contract.json", "filtering_principles_report.md"),
        evaluation_mode="analysis_plus_artifact_shell",
        showcase_priority="medium",
        notes="Good example of filter contract reasoning with generated reports.",
    ),
)


def _load_callable(module_name: str, function_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, function_name, None)


def analyze_functional_surface_catalog() -> FunctionalSurfaceCatalogResult:
    rows: list[FunctionalSurfaceRow] = []
    for spec in FUNCTIONAL_SURFACE_REGISTRY:
        analysis_callable = callable(_load_callable(spec.module, spec.analysis_function))
        artifact_callable = callable(_load_callable(spec.module, spec.artifact_function))
        rows.append(
            FunctionalSurfaceRow(
                surface_id=spec.surface_id,
                category=spec.category,
                module=spec.module,
                analysis_function=spec.analysis_function,
                artifact_function=spec.artifact_function,
                artifact_outputs=spec.artifact_outputs,
                evaluation_mode=spec.evaluation_mode,
                showcase_priority=spec.showcase_priority,
                analysis_callable=analysis_callable,
                artifact_callable=artifact_callable,
                artifact_output_count=len(spec.artifact_outputs),
                notes=spec.notes,
            )
        )

    summary = {
        "surface_count": len(rows),
        "analysis_callable_count": sum(1 for row in rows if row.analysis_callable),
        "artifact_callable_count": sum(1 for row in rows if row.artifact_callable),
        "high_priority_count": sum(1 for row in rows if row.showcase_priority == "high"),
        "medium_priority_count": sum(1 for row in rows if row.showcase_priority == "medium"),
        "categories": sorted({row.category for row in rows}),
        "evaluation_modes": sorted({row.evaluation_mode for row in rows}),
    }

    report_markdown = render_functional_surface_catalog_report(
        FunctionalSurfaceCatalogResult(surface_rows=tuple(rows), summary=summary, report_markdown="")
    )

    return FunctionalSurfaceCatalogResult(
        surface_rows=tuple(rows),
        summary=summary,
        report_markdown=report_markdown,
    )


def _render_priority_plot(result: FunctionalSurfaceCatalogResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = [row.surface_id for row in result.surface_rows]
    values = [row.artifact_output_count for row in result.surface_rows]
    colors = ["#2563eb" if row.showcase_priority == "high" else "#0f766e" for row in result.surface_rows]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("artifact outputs")
    ax.set_title("Functional Surface Artifact Coverage", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def render_functional_surface_catalog_report(result: FunctionalSurfaceCatalogResult) -> str:
    report = MarkdownDocument("Functional Surface Catalog")
    report.paragraph(
        "This catalog is the repo's functional-surface inventory: which modules have a pure analysis entrypoint, "
        "which modules expose a dedicated artifact writer, and what kinds of outputs they produce. "
        "The point is to keep the repo organized around the `analyze -> render -> write artifacts` pattern."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Surface count: `{result.summary['surface_count']}`",
            f"Analysis callables found: `{result.summary['analysis_callable_count']}`",
            f"Artifact callables found: `{result.summary['artifact_callable_count']}`",
            f"High-priority showcases: `{result.summary['high_priority_count']}`",
            f"Categories: `{', '.join(result.summary['categories'])}`",
            f"Evaluation modes: `{', '.join(result.summary['evaluation_modes'])}`",
        ]
    )
    report.heading("Surface Table", level=2)
    report.table(
        ["surface_id", "category", "module", "analysis", "artifact", "outputs", "mode", "priority"],
        [
            (
                f"`{row.surface_id}`",
                row.category,
                f"`{row.module}`",
                "yes" if row.analysis_callable else "no",
                "yes" if row.artifact_callable else "no",
                "<br>".join(f"`{output}`" for output in row.artifact_outputs),
                row.evaluation_mode,
                row.showcase_priority,
            )
            for row in result.surface_rows
        ],
    )
    report.heading("What This Surface Inventory Says", level=2)
    report.bullet_list(
        [
            "The repo already has a strong `analyze -> render -> write artifacts` discipline for most core evaluation modules.",
            "The easiest showcase candidates are the high-priority surfaces with the cleanest callable separation and the richest artifact bundles.",
            "The main remaining work is to keep new modules following the same pattern rather than mixing file I/O into analysis logic.",
        ]
    )
    return report.text()


def write_functional_surface_catalog_artifacts(
    output_dir: str | Path,
    *,
    result: FunctionalSurfaceCatalogResult | None = None,
) -> FunctionalSurfaceCatalogArtifacts:
    payload = result or analyze_functional_surface_catalog()
    output_root = Path(output_dir)
    run_dir = output_root / "functional_surface_catalog_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "functional_surface_catalog_report.md"
    summary_path = run_dir / "functional_surface_catalog_summary.json"
    catalog_path = run_dir / "functional_surface_catalog.csv"
    plot_path = run_dir / "functional_surface_artifact_coverage.png"

    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload.summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(
        catalog_path,
        [
            {
                "surface_id": row.surface_id,
                "category": row.category,
                "module": row.module,
                "analysis_function": row.analysis_function,
                "artifact_function": row.artifact_function,
                "artifact_output_count": row.artifact_output_count,
                "evaluation_mode": row.evaluation_mode,
                "showcase_priority": row.showcase_priority,
                "analysis_callable": row.analysis_callable,
                "artifact_callable": row.artifact_callable,
                "notes": row.notes,
            }
            for row in payload.surface_rows
        ],
        [
            "surface_id",
            "category",
            "module",
            "analysis_function",
            "artifact_function",
            "artifact_output_count",
            "evaluation_mode",
            "showcase_priority",
            "analysis_callable",
            "artifact_callable",
            "notes",
        ],
    )
    plot_path.write_bytes(_figure_to_png(_render_priority_plot(payload)))

    return FunctionalSurfaceCatalogArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        catalog_path=catalog_path,
        plot_path=plot_path,
    )
