from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.markdown_builder import MarkdownDocument
from kinematic_classifier_sandbox.utils.text import markdown_table_preview

from .contracts import (
    ShowcaseAlgorithmReportData,
    ShowcaseCorpusReportData,
    ShowcaseDimensionalLiftReportData,
    ShowcaseFeatureReportData,
    ShowcaseFilteringReportData,
    ShowcaseHeadlineSummary,
    ShowcaseOpenRisksData,
)


def _table_preview(rows: list[dict[str, str]], columns: list[str], limit: int = 8) -> str:
    return markdown_table_preview(rows=rows, columns=columns, limit=limit)


def _render_executive_report(summary: ShowcaseHeadlineSummary) -> str:
    best_classifier = summary.best_common_study_classifier
    best_common_dataset = summary.best_common_dataset_method
    report = MarkdownDocument("Executive Summary")
    report.paragraph("This packet packages the repo as a methodology and evidence suite rather than as a narrow 1D benchmark collection.")
    report.heading("What Is Proven", level=2)
    report.bullet_list(
        [
            f"The best common-study classifier on the current manifest-aligned executable subset is `{best_classifier.identifier}` at `{best_classifier.overall_accuracy:.3f}`.",
            f"The best method on the shared common-dataset comparison is `{best_common_dataset.identifier}` at `{best_common_dataset.overall_accuracy:.3f}`.",
            "Pointwise, windowed, Bayesian accumulator, and Kalman-bank methods now share generic evidence, posterior, and artifact contracts.",
            "Feature metadata, filtering contracts, and dimensional-lift status are all explicitly audited.",
        ]
    )
    report.heading("What Is Still Experimental Or Limited", level=2)
    report.bullet_list(
        [
            f"Corpus adequacy is currently `{summary.corpus_adequacy.overall_status}` rather than fully passing, so some comparisons remain intentionally diagnostic rather than final leaderboard claims.",
            "PCA is used here as a diagnostic, not as a production classifier family.",
            "IMM, PF, and RBPF remain behind explicit decision gates rather than being added prematurely.",
        ]
    )
    report.heading("Core Message", level=2)
    report.bullet_list(
        [
            "The repo is now best understood as a reusable methodology stack: corpus -> features -> evidence -> posterior -> filtering -> analysis -> visualization.",
            "The 1D studies are the current proof surface, but the architecture is now documented and audited for 3D transition work.",
        ]
    )
    return report.text()


def _render_problem_framing_report() -> str:
    report = MarkdownDocument("Problem Framing")
    report.paragraph("The repo is solving a kinematic classification problem under uncertainty.")
    report.paragraph(
        "The engineering question is not only which classifier scores best on one scalar corpus. "
        "The stronger question is whether the repo can support reusable studies across:"
    )
    report.bullet_list(
        [
            "changing feature sets",
            "changing class pairs",
            "changing evidence providers",
            "changing filter backends",
            "changing sensor regimes",
            "changing state dimension from 1D toward 3D",
        ]
    )
    report.paragraph("That is why the showcase is organized around methodology contracts and evidence surfaces rather than only benchmark leaderboards.")
    return report.text()


def _render_methodology_report() -> str:
    report = MarkdownDocument("Methodology Overview")
    report.paragraph("The current repo architecture is intentionally layered:")
    report.ordered_list(
        [
            "corpus generation and study manifests",
            "feature extraction and feature taxonomy",
            "evidence production by classifier family",
            "posterior updating and confidence history",
            "optional filter backends and diagnostics",
            "metrics, identifiability, prior, coverage, and dimensional audits",
            "plots, reports, and packet exports",
        ]
    )
    report.paragraph("Use the generic-methodology proof artifacts for contract details:")
    report.bullet_list(
        [
            "`generic_inference_contract/`",
            "`classification_evidence_proof/`",
            "`filtering_contract/`",
            "`dimensional_lift_audit/`",
        ]
    )
    return report.text()


def _render_algorithm_ladder_report(data: ShowcaseAlgorithmReportData) -> str:
    ladder_preview = _table_preview(
        list(data.metrics_by_classifier),
        ["classifier_id", "overall_accuracy", "num_predictions"],
        limit=len(data.metrics_by_classifier),
    )
    shared_preview = _table_preview(
        list(data.common_dataset_rows),
        [
            "method_name",
            "applicability_status",
            "primary_evaluation_family",
            "overall_accuracy",
            "endpoint_match_accuracy",
            "noisy_accuracy",
            "prior_flip_fraction",
        ],
        limit=len(data.common_dataset_rows),
    )
    report = MarkdownDocument("Algorithm Ladder")
    report.paragraph(
        "The algorithm ladder now spans direct pointwise methods through advanced particle-family witnesses, but the shared-corpus tables are capability-aware rather than naive one-table-for-all leaderboards."
    )
    report.heading("Common Study Metrics", level=2)
    report.paragraph(ladder_preview)
    report.heading("Shared-Corpus Method Comparison", level=2)
    report.paragraph(shared_preview)
    report.heading("Reading Notes", level=2)
    report.bullet_list(
        [
            "`pointwise` is the weak lower bound and uses only instantaneous evidence.",
            "`windowed_*` methods add history-derived features without a state-space model.",
            "`bayes_accumulator` adds sequential evidence accumulation.",
            "`kalman_bank` is the first explicitly model-based evidence provider.",
            "`kalman_bank_velocity_aided` is a stronger sensor regime and should not be treated as a fair same-sensor upgrade over position-only methods.",
            "`particle_filter_bank`, `rbpf`, and `ornstein_uhlenbeck_pf_v1` remain visible in shared tables with `witness_only` applicability when the shared binary corpus is not their valid scoring family.",
        ]
    )
    return report.text()


def _render_feature_report(data: ShowcaseFeatureReportData) -> str:
    taxonomy_preview = _table_preview(
        [{k: str(v) for k, v in row.items()} for row in data.taxonomy_rows],
        ["name", "role", "history_behavior", "geometry_assumption", "dimensional_transfer"],
        limit=8,
    )
    identifiability_preview = _table_preview(
        list(data.identifiability_rows),
        ["class_pair_id", "feature_set_id", "mean_standardized_feature_distance", "overlap_estimate", "identifiability_status"],
        limit=10,
    )
    oracle_preview = _table_preview(
        list(data.oracle_rows),
        ["class_pair_id", "feature_set_id", "oracle_accuracy", "mean_posterior_margin", "is_best_feature_set"],
        limit=10,
    )
    report = MarkdownDocument("Feature Taxonomy And Class-Pair Studies")
    report.paragraph("This section ties together feature metadata, pairwise separability, and feature-only oracle baselines.")
    report.heading("Feature Taxonomy Preview", level=2)
    report.paragraph(taxonomy_preview)
    report.heading("Class-Pair Identifiability Preview", level=2)
    report.paragraph(identifiability_preview)
    report.heading("Oracle Separability Preview", level=2)
    report.paragraph(oracle_preview)
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "Use identifiability rows to decide whether a confusion is fundamentally hard in feature space.",
            "Use oracle rows to decide whether the classifier is underperforming the available feature signal.",
            "Use taxonomy metadata to separate dimension-agnostic features from 1D-specific geometric assumptions.",
        ]
    )
    return report.text()


def _render_filtering_report(data: ShowcaseFilteringReportData) -> str:
    transition_preview = _table_preview(
        list(data.transition_rows),
        ["scenario_name", "static_accuracy", "transition_accuracy", "kalman_accuracy", "transition_post_switch"],
        limit=8,
    )
    advanced_preview = _table_preview(
        [{key: str(value) for key, value in row.items()} for row in data.advanced_summary.method_rows],
        ["method_id", "scenario_family", "primary_metric", "primary_metric_value", "promotion_decision"],
        limit=len(data.advanced_summary.method_rows),
    ) if data.advanced_summary.method_rows else ""
    report = MarkdownDocument("Filtering And Advanced-Method Decision")
    report.bullet_list(
        [
            f"IMM justified now: `{data.advanced_summary.imm_justified}`",
            f"Particle filter justified now: `{data.advanced_summary.particle_filter_justified}`",
            f"RBPF justified now: `{data.advanced_summary.rbpf_justified}`",
        ]
    )
    report.heading("Switching Preview", level=2)
    report.paragraph(transition_preview)
    if advanced_preview:
        report.heading("Advanced Witness Metrics", level=2)
        report.paragraph(advanced_preview)
    report.heading("Decision Notes", level=2)
    report.bullet_list(
        [
            "IMM reaches `witness_supported` when switching-mode witnesses show explicit state mixing helps post-switch inference; broader `study_justified` status remains a separate gate.",
            "PF reaches `witness_supported` or `study_justified` only on nonlinear or non-Gaussian witnesses where the particle model beats the Gaussian baseline and the robustness sweep clears.",
            "RBPF reaches `witness_supported` on latent maneuver-onset witnesses when sampled mode paths and conditional Kalman state estimates help, but it stays distinct from broader complexity justification.",
            "The trace-validation packet adds another explicit layer: `trace_validated` means a method can explain its update steps mechanically, but trace presence alone does not imply promotion.",
            "These are still witness-specific promotions rather than default shared-corpus benchmark winners.",
            "These statuses are intentionally separate so advanced filters do not silently become default shared-corpus winners.",
        ]
    )
    return report.text()


def _render_corpus_report(data: ShowcaseCorpusReportData) -> str:
    pair_preview = _table_preview(
        list(data.class_pair_rows),
        ["class_a", "class_b", "difficulty", "pairwise_auc", "overlap", "status"],
        limit=8,
    )
    leakage_preview = _table_preview(
        list(data.leakage_rows),
        ["covariate", "max_pairwise_auc", "spread_ratio", "worst_pair", "status"],
        limit=8,
    )
    report = MarkdownDocument("Corpus Adequacy And Leakage")
    report.bullet_list(
        [
            f"Overall corpus status: `{data.summary.overall_status}`",
            f"Feature coverage: `{data.summary.feature_status}`",
            f"Class-pair coverage: `{data.summary.class_pair_status}`",
            f"Covariate leakage: `{data.summary.covariate_status}`",
        ]
    )
    report.heading("Class-Pair Boundary Coverage", level=2)
    report.paragraph(pair_preview)
    report.heading("Covariate Leakage Preview", level=2)
    report.paragraph(leakage_preview)
    report.heading("Reading Notes", level=2)
    report.bullet_list(
        [
            "Treat this report as a credibility layer for all downstream classifier comparisons.",
            "A failing corpus result means the benchmark is still useful diagnostically, but not all leaderboard conclusions should be treated as stable.",
        ]
    )
    return report.text()


def _render_study_suite_report() -> str:
    study_rows = [
        ("common_1d_classifier_study", "Manifest-driven pairwise common study", "reports/03_algorithm_ladder.md"),
        ("common_dataset_comparison_v1", "Shared-corpus technique comparison", "reports/03_algorithm_ladder.md"),
        ("feature_analysis_v1", "Feature excitation and identifiability", "reports/04_feature_taxonomy.md"),
        ("pca_analysis_v1", "PCA dimensionality diagnostics", "reports/08_results_summary.md"),
        ("corpus_adequacy_audit_v1", "Corpus adequacy and leakage gates", "reports/08_results_summary.md"),
        ("transition_matrix_accumulator_v1", "Switching benchmark", "reports/05_filtering_taxonomy.md"),
        ("dimensional_lift_audit", "1D-to-3D transition audit", "reports/09_3d_transition_plan.md"),
    ]
    report = MarkdownDocument("Study Suite")
    rows = [{"study": name, "purpose": purpose, "primary_report": report_path} for name, purpose, report_path in study_rows]
    report.paragraph(_table_preview(rows, ["study", "purpose", "primary_report"], limit=len(rows)))
    report.paragraph("These studies are designed to answer different methodology questions rather than all collapsing into a single leaderboard.")
    return report.text()


def _render_gallery_report(plot_entries: list[dict[str, object]]) -> str:
    report = MarkdownDocument("Visualization Gallery")
    report.paragraph("Plots are grouped by the question they answer.")
    for entry in plot_entries:
        plot_id = entry.get("plot_id")
        if entry.get("kind") != "plot" or not plot_id:
            continue
        section_name = str(plot_id)
        report.heading(section_name, level=2)
        report.paragraph(f"![{plot_id}](../plots/{Path(str(entry['relative_path'])).name})")
        report.bullet_list(
            [
                f"Caption: {entry.get('caption', 'n/a')}",
                f"Interpretation: {entry.get('interpretation', 'n/a')}",
                f"Limitations: {entry.get('limitations', 'n/a')}",
                f"Source artifact: `{entry.get('source_path', 'n/a')}`",
            ]
        )
    return report.text()


def _render_results_summary_report(summary: ShowcaseHeadlineSummary) -> str:
    dimensional_counts = summary.dimensional_status_counts
    report = MarkdownDocument("Results Summary")
    report.heading("Headline Findings", level=2)
    report.bullet_list(
        [
            f"Best common-study classifier: `{summary.best_common_study_classifier.identifier}` at `{summary.best_common_study_classifier.overall_accuracy:.3f}`.",
            f"Best shared-corpus method: `{summary.best_common_dataset_method.identifier}` at `{summary.best_common_dataset_method.overall_accuracy:.3f}`.",
            f"Corpus adequacy status: `{summary.corpus_adequacy.overall_status}`.",
            f"IMM justified now: `{summary.advanced_filters.imm_justified}`.",
            f"Particle filter justified now: `{summary.advanced_filters.particle_filter_justified}`.",
            f"RBPF justified now: `{summary.advanced_filters.rbpf_justified}`.",
            f"Dimensional status counts: `{dimensional_counts}`.",
        ]
    )
    report.heading("Takeaway", level=2)
    report.bullet_list(
        [
            "The repo now proves reusable methodology layers more strongly than it proves a final optimized 1D leaderboard.",
            "The strongest next work should therefore be new study families, new sensors, or new 3D-capable adapters rather than more blind 1D tuning.",
        ]
    )
    return report.text()


def _render_3d_transition_report(data: ShowcaseDimensionalLiftReportData) -> str:
    preview = _table_preview(
        list(data.dimension_rows),
        ["module", "layer", "dimensional_status", "required_3d_action"],
        limit=len(data.dimension_rows),
    )
    report = MarkdownDocument("3D Transition Plan")
    report.paragraph(
        "This report shows what remains generic, what is adapter-compatible, and what still needs explicit 3D-specific rewrite work."
    )
    report.paragraph(preview)
    report.heading("Reading Notes", level=2)
    report.bullet_list(
        [
            "`dimension_agnostic` means the current abstraction is already suitable for vector-capable work.",
            "`adapter_compatible` means the contract is good but current study data or helper code is still scalar-scoped.",
            "`rewrite_required` means the implementation itself assumes scalar geometry or scalar dynamics.",
        ]
    )
    return report.text()


def _render_open_risks_report(data: ShowcaseOpenRisksData) -> str:
    report = MarkdownDocument("Open Risks And Next Steps")
    report.heading("Current Risks", level=2)
    report.bullet_list(
        [
            "The current corpus adequacy gate is not fully green, so some comparisons remain deliberately diagnostic.",
            "PCA is documented only as a diagnostic, not as a classifier family.",
            "The current dimensional-lift proof stops at fake vector-compatible methodology flow, not full 3D dynamics or filtering.",
            "Advanced filters remain unjustified on current evidence, so adding them now would lower methodological discipline rather than improve it.",
        ]
    )
    report.heading("Immediate Next Steps", level=2)
    if data.recommendations:
        report.bullet_list([f"{recommendation}" for recommendation in data.recommendations])
    else:
        report.bullet_list(["No open corpus recommendations were emitted."])
    report.bullet_list(
        [
            "Add a true 3D-capable corpus adapter and vector feature registry on top of the dimensional-lift contract.",
            "Decide whether a PCA-classifier study is actually desired; it is not currently part of the proven methodology stack.",
        ]
    )
    return report.text()
