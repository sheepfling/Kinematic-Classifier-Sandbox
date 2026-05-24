from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from .kalman_filter_bank import run_kalman_bank_benchmark


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = ROOT / "artifacts"
SHOWCASE_DOCS_DIR = ROOT / "docs" / "showcase"


@dataclass(frozen=True, slots=True)
class ShowcaseArtifacts:
    showcase_dir: Path
    index_path: Path
    proof_gallery_path: Path
    artifact_manifest_path: Path
    summary_metrics_path: Path
    reports_dir: Path
    plots_dir: Path
    tables_dir: Path
    run_cards_dir: Path
    team_packet_dir: Path
    zip_path: Path | None
    validation_path: Path


@dataclass(frozen=True, slots=True)
class ShowcaseValidationResult:
    overall_status: str
    required_reports_exist: bool
    proof_gallery_complete: bool
    manifest_complete: bool
    metrics_tables_exist: bool
    gallery_references_exist: bool
    proof_gallery_references_exist: bool
    gallery_annotations_complete: bool
    feature_taxonomy_complete: bool
    class_pair_identifiability_complete: bool
    advanced_filter_go_no_go_present: bool
    dimensional_status_present: bool
    errors: tuple[str, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _prepare_matplotlib():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kinematic-classifier-sandbox-mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_plot(fig, path: Path) -> None:
    plt = _prepare_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, format="png", dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


def _table_preview(rows: list[dict[str, str]], columns: list[str], limit: int = 8) -> str:
    if not rows:
        return "_No rows available._"
    visible = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in visible
    ]
    return "\n".join([header, separator, *body])


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _headline_summary() -> dict[str, object]:
    metrics_by_classifier = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_classifier.csv")
    best_classifier = max(metrics_by_classifier, key=lambda row: _float(row, "overall_accuracy"))
    common_dataset_rows = _read_csv(ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_summary.csv")
    best_common_dataset = max(common_dataset_rows, key=lambda row: _float(row, "overall_accuracy"))
    corpus_summary = _read_json(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "corpus_adequacy_summary.json")
    advanced_summary = _read_json(ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_summary.json")
    dimension_rows = _read_csv(ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv")
    dimension_counts: dict[str, int] = {}
    for row in dimension_rows:
        status = row["dimensional_status"]
        dimension_counts[status] = dimension_counts.get(status, 0) + 1
    return {
        "best_common_study_classifier": {
            "classifier_id": best_classifier["classifier_id"],
            "overall_accuracy": _float(best_classifier, "overall_accuracy"),
        },
        "best_common_dataset_method": {
            "method_name": best_common_dataset["method_name"],
            "overall_accuracy": _float(best_common_dataset, "overall_accuracy"),
        },
        "corpus_adequacy": corpus_summary["summary"],
        "advanced_filters": advanced_summary,
        "dimensional_status_counts": dimension_counts,
    }


def _showcase_plot_definitions() -> list[dict[str, str]]:
    return [
        {
            "plot_id": "prior_to_posterior_single_step",
            "source": "artifacts/bayesian_walkthroughs/plots/prior_to_posterior_single_step.png",
            "filename": "prior_to_posterior_single_step.png",
            "section": "bayesian_updates",
            "caption": "Single-step prior-to-posterior walkthrough for a representative Bayesian update.",
            "interpretation": "Use this to verify that the posterior move matches the stated prior and likelihood evidence rather than opaque numerical code paths.",
            "limitations": "This is one witness trajectory and one update step, not a proof that every model family is calibrated equally well.",
        },
        {
            "plot_id": "likelihood_curves_with_feature_value",
            "source": "artifacts/bayesian_walkthroughs/plots/likelihood_curves_with_feature_value.png",
            "filename": "likelihood_curves_with_feature_value.png",
            "section": "bayesian_updates",
            "caption": "Likelihood curves with the observed feature value marked.",
            "interpretation": "Use this to see why a given feature value favored one class over another before any posterior accumulation happened.",
            "limitations": "These curves use the current simplified evidence models and are not a universal likelihood audit across all families.",
        },
        {
            "plot_id": "posterior_timeline",
            "source": "artifacts/bayesian_walkthroughs/plots/posterior_timeline.png",
            "filename": "posterior_timeline.png",
            "section": "bayesian_updates",
            "caption": "Posterior timeline over one representative trajectory.",
            "interpretation": "Use this to see whether evidence accumulated smoothly or only flipped at the end of the track.",
            "limitations": "This is a representative trace rather than a full Monte Carlo posterior distribution.",
        },
        {
            "plot_id": "log_odds_timeline",
            "source": "artifacts/bayesian_walkthroughs/plots/log_odds_timeline.png",
            "filename": "log_odds_timeline.png",
            "section": "bayesian_updates",
            "caption": "Log-odds timeline for the same representative Bayesian walkthrough.",
            "interpretation": "Use this to separate likelihood-driven movement from prior offsets in an additive space.",
            "limitations": "The current walkthough is binary and simplified; multi-class log-odds surfaces are richer than this single example.",
        },
        {
            "plot_id": "prior_sensitivity_curve",
            "source": "artifacts/bayesian_walkthroughs/plots/prior_sensitivity_curve.png",
            "filename": "prior_sensitivity_curve.png",
            "section": "bayesian_updates",
            "caption": "Posterior outcome versus prior sweep for the walkthrough trajectory.",
            "interpretation": "Use this to see where prior changes can and cannot move the decision boundary.",
            "limitations": "This is a local fragility study, not a full prior robustness proof across every study candidate.",
        },
        {
            "plot_id": "bayes_factor_timeline",
            "source": "artifacts/bayesian_walkthroughs/plots/bayes_factor_timeline.png",
            "filename": "bayes_factor_timeline.png",
            "section": "bayesian_updates",
            "caption": "Bayes-factor timeline across the representative trajectory.",
            "interpretation": "Use this to see when evidence was genuinely informative versus merely cumulative.",
            "limitations": "Bayes factors here are driven by the current proxy evidence surfaces for some families.",
        },
        {
            "plot_id": "classifier_pair_accuracy_heatmap",
            "source": "artifacts/common_1d_classifier_study/plots/confusion_matrices/classifier_pair_accuracy_heatmap.png",
            "filename": "classifier_pair_accuracy_heatmap.png",
            "section": "algorithm_comparison",
            "caption": "Classifier-by-class-pair heatmap from the common study.",
            "interpretation": "Use this to see which methods fail on which explicit class boundaries rather than only overall accuracy.",
            "limitations": "This reflects the current 1D executable subset rather than a full multi-dimensional deployment study.",
        },
        {
            "plot_id": "prefix_accuracy_curve",
            "source": "artifacts/common_1d_classifier_study/plots/monte_carlo/prefix_accuracy_curve.png",
            "filename": "prefix_accuracy_curve.png",
            "section": "duration_sensitivity",
            "caption": "Prefix accuracy versus time within the common study.",
            "interpretation": "Use this to separate early-horizon ambiguity from late-horizon classifier failure.",
            "limitations": "The curve is specific to the current synthetic corpus and declared class pairs.",
        },
        {
            "plot_id": "prior_sensitivity",
            "source": "artifacts/common_1d_classifier_study/plots/priors/prior_sensitivity.png",
            "filename": "prior_sensitivity.png",
            "section": "prior_fragility",
            "caption": "Prior-sensitivity view from the common study bundle.",
            "interpretation": "Use this to identify cases where decisions move under prior changes rather than stronger evidence.",
            "limitations": "This summarizes the current binary and pairwise prior studies, not arbitrary class-cardinality priors.",
        },
        {
            "plot_id": "identifiability_summary",
            "source": "artifacts/common_1d_classifier_study/plots/feature_space/identifiability_summary.png",
            "filename": "identifiability_summary.png",
            "section": "identifiability",
            "caption": "Common-study identifiability summary by class pair and feature bundle.",
            "interpretation": "Use this to decide whether a failure is more likely a feature/data limit than a classifier implementation issue.",
            "limitations": "This summarizes engineered features only; it is not a learned representation audit.",
        },
        {
            "plot_id": "feature_space_confusion_map",
            "source": "artifacts/feature_analysis_v1/feature_space_confusion_map.png",
            "filename": "feature_space_confusion_map.png",
            "section": "feature_confusion",
            "caption": "Feature-space scatter emphasizing confusing class pairs.",
            "interpretation": "Use this to see whether classes overlap structurally in feature space.",
            "limitations": "The projection is intentionally simplified and does not prove global separability by itself.",
        },
        {
            "plot_id": "class_confusability_heatmap",
            "source": "artifacts/feature_analysis_v1/class_confusability_heatmap.png",
            "filename": "class_confusability_heatmap.png",
            "section": "feature_confusion",
            "caption": "Class confusability heatmap from pairwise feature analysis.",
            "interpretation": "Use this to rank which class pairs most deserve targeted feature or corpus work.",
            "limitations": "This is feature-space confusability, not end-to-end classifier confusion.",
        },
        {
            "plot_id": "pairwise_overlap_heatmap",
            "source": "artifacts/feature_analysis_v1/pairwise_overlap_heatmap.png",
            "filename": "pairwise_overlap_heatmap.png",
            "section": "feature_confusion",
            "caption": "Pairwise overlap heatmap for the current feature-space class distributions.",
            "interpretation": "Use this to identify class pairs that remain intrinsically close even before classifier choice.",
            "limitations": "Overlap estimates depend on the current engineered-feature representation.",
        },
        {
            "plot_id": "feature_ranking_summary",
            "source": "artifacts/feature_analysis_v1/feature_ranking_summary.png",
            "filename": "feature_ranking_summary.png",
            "section": "feature_confusion",
            "caption": "Feature ranking summary from the separability analysis.",
            "interpretation": "Use this to see which engineered features carry the strongest class evidence on the current corpus.",
            "limitations": "Ranking is corpus-dependent and does not by itself prove causal feature importance.",
        },
        {
            "plot_id": "pc1_pc2_by_class",
            "source": "artifacts/pca_analysis_v1/pc1_pc2_by_class.png",
            "filename": "pc1_pc2_by_class.png",
            "section": "pca",
            "caption": "PC1/PC2 class scatter for the full engineered feature set.",
            "interpretation": "Use this as a dimensionality diagnostic to see whether major separation is already available in a low-dimensional projection.",
            "limitations": "PCA is diagnostic here, not a classifier.",
        },
        {
            "plot_id": "covariate_leakage_audit",
            "source": "artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.png",
            "filename": "covariate_leakage_audit.png",
            "section": "corpus_validity",
            "caption": "Covariate leakage audit across the generated corpus.",
            "interpretation": "Use this to see whether duration, sampling, or noise covariates are class-linked strongly enough to corrupt comparisons.",
            "limitations": "A green result reduces suspicion of leakage; it does not prove the corpus is realistic.",
        },
        {
            "plot_id": "class_pair_coverage_heatmap",
            "source": "artifacts/corpus_adequacy_audit_v1/class_pair_coverage_heatmap.png",
            "filename": "class_pair_coverage_heatmap.png",
            "section": "corpus_validity",
            "caption": "Declared class-pair coverage heatmap for the corpus.",
            "interpretation": "Use this to identify pairs that are still too easy or under-covered in required tiers.",
            "limitations": "Coverage is only as good as the current manifest definitions and tier design.",
        },
        {
            "plot_id": "kalman_bank_diagnostics",
            "source": "artifacts/kalman_filter_bank/kalman_bank_diagnostics.png",
            "filename": "kalman_bank_diagnostics.png",
            "section": "filtering",
            "caption": "Kalman bank diagnostic panel.",
            "interpretation": "Use this to inspect innovation-driven evidence quality and posterior movement for the model-based baseline.",
            "limitations": "This reflects the current position-only Kalman family, not a fully generalized 3D filter stack.",
        },
        {
            "plot_id": "transition_matrix_diagnostics",
            "source": "artifacts/transition_matrix_accumulator_v1/transition_matrix_diagnostics.png",
            "filename": "transition_matrix_diagnostics.png",
            "section": "switching",
            "caption": "Transition benchmark diagnostics over switching scenarios.",
            "interpretation": "Use this to judge whether explicit transition structure buys something before considering IMM.",
            "limitations": "This is a methodology exercise over current synthetic switching cases, not an operational mode tracker.",
        },
    ]


def _copy_showcase_tables(tables_dir: Path) -> list[dict[str, object]]:
    table_sources = [
        ("bayesian_step_tables.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "bayesian_step_tables.csv", "bayesian_updates"),
        ("prior_sweep_examples.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "prior_sweep_examples.csv", "bayesian_updates"),
        ("feature_contribution_examples.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "feature_contribution_examples.csv", "bayesian_updates"),
        ("posterior_flip_thresholds.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "posterior_flip_thresholds.csv", "bayesian_updates"),
        ("metrics_by_classifier.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_classifier.csv", "algorithm_comparison"),
        ("metrics_by_class_pair.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_class_pair.csv", "class_pair_study"),
        ("feature_set_comparison.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "feature_set_comparison.csv", "feature_study"),
        ("prior_sensitivity_by_class_pair.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "prior_sensitivity_by_class_pair.csv", "prior_fragility"),
        ("identifiability_matrix.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "identifiability_matrix.csv", "identifiability"),
        ("oracle_classifier_results.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "oracle_classifier_results.csv", "feature_study"),
        ("covariate_leakage_audit.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "covariate_leakage_audit.csv", "corpus_validity"),
        ("feature_excitation_matrix.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "feature_excitation_matrix.csv", "corpus_validity"),
        ("validation_ladder_decisions.csv", ARTIFACTS_ROOT / "validation_ladder" / "validation_ladder_decisions.csv", "study_candidate_validation"),
        ("validation_ladder_scores.csv", ARTIFACTS_ROOT / "validation_ladder" / "validation_ladder_scores.csv", "study_candidate_validation"),
        ("feature_evidence_table.csv", ARTIFACTS_ROOT / "study_candidate_generation" / "feature_evidence_table.csv", "study_candidate_validation"),
        ("prior_sensitivity_explanation_table.csv", ARTIFACTS_ROOT / "study_candidate_generation" / "prior_sensitivity_explanation_table.csv", "study_candidate_validation"),
        ("method_summary.csv", ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_summary.csv", "algorithm_comparison"),
        ("technique_summary.csv", ARTIFACTS_ROOT / "technique_comparison_v1" / "technique_summary.csv", "algorithm_comparison"),
        ("class_balance.csv", ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "class_balance.csv", "corpus_validity"),
        ("class_pair_coverage.csv", ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "class_pair_coverage.csv", "corpus_validity"),
        ("feature_set_coverage.csv", ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "feature_set_coverage.csv", "corpus_validity"),
        ("classifier_support.csv", ARTIFACTS_ROOT / "coverage_report_v1" / "classifier_support.csv", "corpus_validity"),
        ("feature_taxonomy.json", ARTIFACTS_ROOT / "feature_taxonomy" / "feature_taxonomy.json", "feature_taxonomy"),
        ("feature_separation_scores.csv", ARTIFACTS_ROOT / "feature_analysis_v1" / "feature_separation_scores.csv", "feature_study"),
        ("pairwise_overlap_matrix.csv", ARTIFACTS_ROOT / "feature_analysis_v1" / "pairwise_overlap_matrix.csv", "feature_study"),
        ("pairwise_auc_matrix.csv", ARTIFACTS_ROOT / "feature_analysis_v1" / "pairwise_auc_matrix.csv", "feature_study"),
        ("module_dimension_status.csv", ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv", "dimensional_lift"),
        ("scalar_assumption_inventory.csv", ARTIFACTS_ROOT / "dimensional_lift_audit" / "scalar_assumption_inventory.csv", "dimensional_lift"),
        ("advanced_filter_decision_summary.json", ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_summary.json", "advanced_filters"),
        ("advanced_filter_decision_evidence.json", ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_evidence.json", "advanced_filters"),
        ("transition_matrix_scenario_summary.csv", ARTIFACTS_ROOT / "transition_matrix_accumulator_v1" / "transition_matrix_scenario_summary.csv", "advanced_filters"),
        ("algorithm_ladder_proof.csv", ARTIFACTS_ROOT / "latex" / "algorithm_ladder_proof.csv", "algorithm_comparison"),
        ("toy_problem_summary.csv", ARTIFACTS_ROOT / "latex" / "toy_problem_summary.csv", "study_candidate_validation"),
    ]
    if any(not source.exists() and "study_candidate_generation" in str(source) for _, source, _ in table_sources):
        from .study_candidate_generation import write_study_candidate_generation_artifacts

        write_study_candidate_generation_artifacts(ARTIFACTS_ROOT)
    manifest_entries: list[dict[str, object]] = []
    for filename, source, section in table_sources:
        destination = tables_dir / filename
        _copy_file(source, destination)
        manifest_entries.append(
            {
                "kind": "table",
                "section": section,
                "relative_path": str(destination.relative_to(tables_dir.parents[0])),
                "source_path": str(source.relative_to(ROOT)),
            }
        )
    return manifest_entries


def _proof_gallery_claims() -> list[dict[str, object]]:
    return [
        {
            "heading": "Claim 1: Bayesian update machinery works",
            "claim": "The repo can show a prior, likelihood, Bayes factor, and posterior move explicitly rather than hiding classifier behavior behind final labels only.",
            "evidence": [
                ("Single-step prior to posterior", "plots/prior_to_posterior_single_step.png", "One witness update that can be checked by hand."),
                ("Likelihood curves", "plots/likelihood_curves_with_feature_value.png", "Shows which class the current observation favored before accumulation."),
                ("Posterior timeline", "plots/posterior_timeline.png", "Shows sequential accumulation on one representative trajectory."),
                ("Log-odds timeline", "plots/log_odds_timeline.png", "Shows additive evidence movement and prior offsets."),
                ("Bayes factor timeline", "plots/bayes_factor_timeline.png", "Shows when evidence was informative versus merely cumulative."),
                ("Prior sweep table", "tables/prior_sweep_examples.csv", "Shows decision fragility under controlled prior changes."),
                ("Prior sensitivity explanation table", "tables/prior_sensitivity_explanation_table.csv", "Summarizes which generated studies remain evidence-dominated versus prior-sensitive."),
            ],
        },
        {
            "heading": "Claim 2: History helps",
            "claim": "Sequential accumulation and history-aware methods outperform purely instantaneous evidence on ambiguity that unfolds over time.",
            "evidence": [
                ("Pointwise vs accumulator posterior timelines", "plots/pointwise_vs_accumulator_posterior_timelines.png", "Shows how recursive accumulation differs from a pointwise baseline on the same hard trajectory."),
                ("Prefix accuracy curve", "plots/prefix_accuracy_curve.png", "Shows improvement as more history is available."),
                ("True-class posterior quantiles", "plots/true_class_posterior_quantiles.png", "Shows how true-class posterior mass concentrates across repeated accumulator runs."),
            ],
        },
        {
            "heading": "Claim 3: Features matter",
            "claim": "Feature-set choice changes separability, oracle performance, and posterior behavior in ways that are visible and auditable.",
            "evidence": [
                ("Feature ablation chart", "plots/feature_ablation_posterior.png", "Shows how the witness posterior changes when one feature is removed."),
                ("Feature separation ranking", "plots/feature_ranking_summary.png", "Visual summary of strongest separating engineered features."),
                ("Feature distribution by class", "plots/feature_distribution_by_class.png", "Shows whether separation comes from location, spread, or both."),
                ("Feature correlation heatmap", "plots/feature_correlation_heatmap.png", "Shows where evidence sources are correlated strongly enough to justify caution."),
                ("Feature evidence table", "tables/feature_evidence_table.csv", "Links feature metadata, double-counting risk, transfer status, and best/worst class-pair evidence."),
                ("Feature contribution examples", "tables/feature_contribution_examples.csv", "Walkthrough-level evidence contributions with caveats on correlation."),
            ],
        },
        {
            "heading": "Claim 4: Class pairs have different difficulty",
            "claim": "Not all class boundaries are equally hard, and the packet can show pairwise difficulty explicitly rather than collapsing them into one leaderboard.",
            "evidence": [
                ("Pairwise confusion heatmap", "plots/classifier_pair_accuracy_heatmap.png", "Visual ranking of pair difficulty across methods."),
                ("Pairwise overlap heatmap", "plots/pairwise_overlap_heatmap.png", "Shows pairwise overlap structure directly."),
                ("Oracle classifier results", "tables/oracle_classifier_results.csv", "Feature-only upper-bound separability by pair."),
                ("Duration sensitivity by class pair", "plots/duration_sensitivity_by_class_pair.png", "Shows which class boundaries need more horizon before they separate cleanly."),
            ],
        },
        {
            "heading": "Claim 5: Corpus quality matters",
            "claim": "Classifier comparisons are only credible when coverage, excitation, and leakage are audited explicitly.",
            "evidence": [
                ("Corpus adequacy scorecard", "plots/corpus_adequacy_scorecard.png", "Shows why one corpus candidate outranked another on the adequacy objective."),
                ("Feature excitation matrix", "tables/feature_excitation_matrix.csv", "Shows whether designed scenarios actually exercise the feature families."),
                ("Covariate leakage audit", "tables/covariate_leakage_audit.csv", "Quantifies duration, sample-count, and noise leakage risks."),
                ("Candidate corpus comparison", "plots/candidate_corpus_comparison.png", "Shows the autodevelopment frontier across candidate corpora."),
            ],
        },
        {
            "heading": "Claim 6: Filtering helps when dynamics matter",
            "claim": "Model-based evidence providers add value on dynamics-sensitive cases, but that value is conditional and scenario-specific rather than universal.",
            "evidence": [
                ("Kalman innovation likelihood timeline", "plots/kalman_innovation_likelihood_timeline.png", "Shows which motion hypothesis the measurements favored step by step."),
                ("Kalman vs windowed comparison", "plots/kalman_vs_windowed_comparison.png", "Shows aggregate tradeoffs between filtering and feature-only families."),
                ("State estimate vs truth", "plots/state_estimate_vs_truth.png", "Shows whether the correct model actually tracks the latent motion state."),
                ("Model posterior over time", "plots/model_posterior_over_time.png", "Shows the true-model posterior evolution on a representative Kalman witness trajectory."),
            ],
        },
        {
            "heading": "Claim 7: Advanced filters require evidence",
            "claim": "IMM and particle filtering remain gated by measured failure cases rather than being added only because they are more sophisticated.",
            "evidence": [
                ("Advanced filter decision matrix", "plots/advanced_filter_decision_matrix.png", "Compact view of the current evidence relevant to advanced-filter escalation."),
                ("Transition scenario summary", "tables/transition_matrix_scenario_summary.csv", "Switching benchmark results for transition-aware accumulation."),
                ("PF/RBPF go-no-go table", "tables/pf_rbpf_go_no_go_table.csv", "Explicit current gate status for particle methods."),
            ],
        },
        {
            "heading": "Claim 8: 3D transition is planned",
            "claim": "The repo distinguishes dimension-agnostic contracts from scalar-specific implementations, so 3D transition work is planned explicitly rather than hand-waved.",
            "evidence": [
                ("Dimension-lift audit chart", "plots/dimension_lift_audit_chart.png", "Summarizes how much of the stack already transfers to higher dimensions."),
                ("Feature transfer matrix", "plots/feature_transfer_matrix.png", "Shows per-feature dimensional-transfer status from the taxonomy."),
                ("Generic-vs-1D-specific layer diagram", "plots/generic_vs_1d_specific_layer_diagram.png", "Communicates what transfers directly versus what still needs rewrite work."),
            ],
        },
    ]


def _render_proof_gallery() -> str:
    lines = [
        "# Proof Gallery",
        "",
        "This gallery reorganizes the packet by explicit claim and supporting evidence rather than by artifact folder alone.",
        "",
        "The current 1D studies are used as witness problems: each claim names what the repo can currently prove, the evidence used for that proof, and the scope limit that still remains.",
    ]
    for claim in _proof_gallery_claims():
        lines.extend(
            [
                "",
                f"## {claim['heading']}",
                "",
                str(claim["claim"]),
                "",
                "### Evidence",
                "",
            ]
        )
        for label, relative_path, rationale in claim["evidence"]:
            lines.append(f"- [{label}]({relative_path}): {rationale}")
    return "\n".join(lines)


def _extract_markdown_relative_targets(markdown: str) -> tuple[str, ...]:
    matches = re.findall(r"!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)", markdown)
    targets: list[str] = []
    for image_target, link_target in matches:
        target = image_target or link_target
        if target and "://" not in target and not target.startswith("#"):
            targets.append(target)
    return tuple(targets)


def _copy_showcase_plots(plots_dir: Path) -> list[dict[str, object]]:
    manifest_entries: list[dict[str, object]] = []
    for definition in _showcase_plot_definitions():
        source = ROOT / definition["source"]
        destination = plots_dir / definition["filename"]
        _copy_file(source, destination)
        manifest_entries.append(
            {
                "kind": "plot",
                "plot_id": definition["plot_id"],
                "section": definition["section"],
                "relative_path": str(destination.relative_to(plots_dir.parents[0])),
                "source_path": definition["source"],
                "caption": definition["caption"],
                "interpretation": definition["interpretation"],
                "limitations": definition["limitations"],
            }
        )
    return manifest_entries


def _render_pointwise_vs_accumulator_posterior_timelines(plots_dir: Path) -> dict[str, object]:
    plt = _prepare_matplotlib()
    posterior_rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "unified_posterior_history.csv")
    candidates = [
        row
        for row in posterior_rows
        if row["class_pair_id"] == "constant_velocity_vs_constant_acceleration"
        and row["trajectory_id"] == "constant_velocity_vs_constant_acceleration_short_noisy_constant_acceleration_2"
        and row["classifier_id"] in {"pointwise", "bayes_accumulator"}
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        grouped.setdefault(row["classifier_id"], []).append(row)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = {"pointwise": "#dc2626", "bayes_accumulator": "#2563eb"}
    for classifier_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["time"]))
        true_class = ordered[0]["class_b"]
        ax.plot(
            [float(row["time"]) for row in ordered],
            [float(row["posterior_class_b"]) for row in ordered],
            linewidth=2.2,
            color=colors[classifier_id],
            marker="o",
            label=f"{classifier_id} true-class posterior",
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Pointwise vs Accumulator Posterior Timelines", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("true-class posterior")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plots_dir / "pointwise_vs_accumulator_posterior_timelines.png"
    _write_plot(fig, path)
    return {
        "kind": "plot",
        "plot_id": "pointwise_vs_accumulator_posterior_timelines",
        "section": "history_comparison",
        "relative_path": str(path.relative_to(plots_dir.parents[0])),
        "source_path": "derived:common_1d_classifier_study/unified_posterior_history.csv",
        "caption": "Pointwise versus sequential-accumulator true-class posterior timelines on the same hard trajectory.",
        "interpretation": "Use this to see how recursive evidence accumulation differs from a pointwise baseline on the same short noisy case.",
        "limitations": "This is a representative witness trajectory rather than a full posterior-distribution summary.",
    }


def _render_feature_distribution_by_class(plots_dir: Path) -> dict[str, object]:
    plt = _prepare_matplotlib()
    rows = _read_csv(ARTIFACTS_ROOT / "feature_analysis_v1" / "feature_matrix.csv")
    selected_features = ("position_range", "acceleration_variance", "linear_fit_residual")
    classes = sorted({row["true_class"] for row in rows})
    fig, axes = plt.subplots(1, len(selected_features), figsize=(12.0, 4.2), sharey=False)
    if len(selected_features) == 1:
        axes = [axes]
    for ax, feature_name in zip(axes, selected_features):
        data = [
            [float(row[feature_name]) for row in rows if row["true_class"] == class_name]
            for class_name in classes
        ]
        ax.boxplot(data, tick_labels=classes, patch_artist=True)
        ax.set_title(feature_name, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Feature Distributions by Class", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plots_dir / "feature_distribution_by_class.png"
    _write_plot(fig, path)
    return {
        "kind": "plot",
        "plot_id": "feature_distribution_by_class",
        "section": "feature_confusion",
        "relative_path": str(path.relative_to(plots_dir.parents[0])),
        "source_path": "derived:feature_analysis_v1/feature_matrix.csv",
        "caption": "Representative feature distributions by class for three engineered features.",
        "interpretation": "Use this to inspect whether a feature separates classes through location, spread, or both.",
        "limitations": "The selected features are representative, not exhaustive.",
    }


def _render_feature_correlation_heatmap(plots_dir: Path) -> dict[str, object]:
    plt = _prepare_matplotlib()
    rows = _read_csv(ARTIFACTS_ROOT / "feature_analysis_v1" / "feature_matrix.csv")
    feature_names = (
        "position_range",
        "speed_range",
        "acceleration_range",
        "duration",
        "acceleration_variance",
        "quadratic_fit_residual",
        "linear_fit_residual",
    )
    values = {name: [float(row[name]) for row in rows] for name in feature_names}
    matrix: list[list[float]] = []
    for left in feature_names:
        left_values = values[left]
        left_mean = sum(left_values) / len(left_values)
        left_var = sum((value - left_mean) ** 2 for value in left_values)
        row_values: list[float] = []
        for right in feature_names:
            right_values = values[right]
            right_mean = sum(right_values) / len(right_values)
            right_var = sum((value - right_mean) ** 2 for value in right_values)
            numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_values, right_values))
            denominator = math.sqrt(max(left_var * right_var, 1e-12))
            row_values.append(numerator / denominator if denominator > 0.0 else 0.0)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(feature_names)), feature_names, rotation=35, ha="right")
    ax.set_yticks(range(len(feature_names)), feature_names)
    ax.set_title("Feature Correlation Heatmap", loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = plots_dir / "feature_correlation_heatmap.png"
    _write_plot(fig, path)
    return {
        "kind": "plot",
        "plot_id": "feature_correlation_heatmap",
        "section": "feature_confusion",
        "relative_path": str(path.relative_to(plots_dir.parents[0])),
        "source_path": "derived:feature_analysis_v1/feature_matrix.csv",
        "caption": "Feature correlation heatmap across the common engineered feature matrix.",
        "interpretation": "Use this to see where evidence sources are correlated strongly enough to justify double-counting caution.",
        "limitations": "Linear correlation is only one dependence diagnostic.",
    }


def _render_duration_sensitivity_by_class_pair(plots_dir: Path) -> dict[str, object]:
    plt = _prepare_matplotlib()
    rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "class_pair_duration_study.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["classifier_id"] == "bayes_accumulator":
            grouped.setdefault(row["class_pair_id"], []).append(row)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for class_pair_id, pair_rows in sorted(grouped.items()):
        ordered = sorted(pair_rows, key=lambda row: float(row["time"]))
        ax.plot(
            [float(row["time"]) for row in ordered],
            [float(row["prefix_accuracy"]) for row in ordered],
            linewidth=1.8,
            label=class_pair_id,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Duration Sensitivity by Class Pair", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("prefix accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path = plots_dir / "duration_sensitivity_by_class_pair.png"
    _write_plot(fig, path)
    return {
        "kind": "plot",
        "plot_id": "duration_sensitivity_by_class_pair",
        "section": "duration_sensitivity",
        "relative_path": str(path.relative_to(plots_dir.parents[0])),
        "source_path": "derived:common_1d_classifier_study/class_pair_duration_study.csv",
        "caption": "Prefix-accuracy duration sensitivity by class pair for the sequential accumulator.",
        "interpretation": "Use this to see which class boundaries require more horizon before they separate cleanly.",
        "limitations": "This is one family-specific duration view, not a universal lower bound.",
    }


def _render_corpus_adequacy_scorecard(plots_dir: Path) -> dict[str, object]:
    plt = _prepare_matplotlib()
    rows = _read_csv(ARTIFACTS_ROOT / "corpus_autodevelopment_v1" / "corpus_candidate_scores.csv")
    top_rows = sorted(rows, key=lambda row: float(row["overall_score"]), reverse=True)[:5]
    metrics = ["balance_score", "boundary_coverage_score", "feature_excitation_score", "difficulty_diversity_score"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y_positions = list(range(len(top_rows)))
    for metric, color in zip(metrics, ("#2563eb", "#16a34a", "#f59e0b", "#7c3aed")):
        ax.plot([float(row[metric]) for row in top_rows], y_positions, marker="o", linewidth=1.8, label=metric.replace("_", " "))
    ax.set_yticks(y_positions, [row["candidate_id"] for row in top_rows])
    ax.set_xlim(0.0, 1.05)
    ax.set_title("Corpus Adequacy Scorecard", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("component score")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = plots_dir / "corpus_adequacy_scorecard.png"
    _write_plot(fig, path)
    return {
        "kind": "plot",
        "plot_id": "corpus_adequacy_scorecard",
        "section": "corpus_validity",
        "relative_path": str(path.relative_to(plots_dir.parents[0])),
        "source_path": "derived:corpus_autodevelopment_v1/corpus_candidate_scores.csv",
        "caption": "Scorecard of the leading corpus candidates across adequacy components.",
        "interpretation": "Use this to see why one corpus candidate outranked another rather than only reading the final overall score.",
        "limitations": "The scorecard reflects the current objective weights and does not prove universal corpus realism.",
    }


def _render_kalman_innovation_likelihood_timeline(plots_dir: Path) -> dict[str, object]:
    plt = _prepare_matplotlib()
    rows = _read_csv(ARTIFACTS_ROOT / "kalman_filter_bank" / "innovation_history.csv")
    selected = [row for row in rows if row["trajectory_id"] == "constant_velocity_regular_0"]
    class_names = ("stationary", "constant_velocity", "constant_acceleration")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for class_name, color in zip(class_names, ("#dc2626", "#2563eb", "#16a34a")):
        values = []
        for row in selected:
            innovation = float(row[f"innovation_{class_name}"])
            variance = max(float(row[f"innovation_variance_{class_name}"]), 1e-9)
            values.append(-0.5 * ((innovation * innovation) / variance + math.log(variance)))
        ax.plot([float(row["time"]) for row in selected], values, marker="o", linewidth=2.0, color=color, label=class_name)
    ax.set_title("Kalman Innovation Likelihood Timeline", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("innovation log-likelihood proxy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plots_dir / "kalman_innovation_likelihood_timeline.png"
    _write_plot(fig, path)
    return {
        "kind": "plot",
        "plot_id": "kalman_innovation_likelihood_timeline",
        "section": "filtering",
        "relative_path": str(path.relative_to(plots_dir.parents[0])),
        "source_path": "derived:kalman_filter_bank/innovation_history.csv",
        "caption": "Innovation likelihood timeline for one representative Kalman-bank trajectory.",
        "interpretation": "Use this to see which motion hypothesis the measurements supported at each step.",
        "limitations": "This is one representative trajectory and a proxy likelihood view, not a full posterior audit.",
    }


def _render_kalman_state_estimate_vs_truth(plots_dir: Path) -> list[dict[str, object]]:
    plt = _prepare_matplotlib()
    benchmark = run_kalman_bank_benchmark(seed=7, trajectories_per_class=4)
    run = next(item for item in benchmark.runs if item.scenario_name == "constant_velocity_regular" and item.true_class == "constant_velocity")
    trajectory = next(item for item in benchmark.trajectories if item.trajectory_id == run.trajectory_id)
    posterior_steps = list(run.steps)
    state_history_rows = _read_csv(ARTIFACTS_ROOT / "kalman_filter_bank" / "state_estimate_history.csv")
    selected_rows = [row for row in state_history_rows if row["trajectory_id"] == run.trajectory_id and row["model_name"] == run.true_class]
    fig1, ax1 = plt.subplots(figsize=(8.2, 4.4))
    ax1.plot(trajectory.times, trajectory.true_position, color="#111827", linestyle="--", linewidth=2.0, label="true position")
    ax1.plot(
        [trajectory.times[index] for index in range(len(selected_rows))],
        [float(row["position"]) for row in selected_rows],
        color="#2563eb",
        linewidth=2.0,
        marker="o",
        label="estimated position",
    )
    ax1.plot(trajectory.times, trajectory.measurements, color="#9ca3af", alpha=0.7, marker="x", linestyle=":", label="measurement")
    ax1.set_title("Kalman State Estimate vs Truth", loc="left", fontsize=12, fontweight="bold")
    ax1.set_xlabel("time")
    ax1.set_ylabel("position")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)
    fig1.tight_layout()
    path1 = plots_dir / "state_estimate_vs_truth.png"
    _write_plot(fig1, path1)

    fig2, ax2 = plt.subplots(figsize=(8.2, 4.4))
    ax2.plot(
        trajectory.times,
        [step.posterior_weights[run.true_class] for step in posterior_steps],
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        label=f"{run.true_class} posterior",
    )
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("Kalman Model Posterior Over Time", loc="left", fontsize=12, fontweight="bold")
    ax2.set_xlabel("time")
    ax2.set_ylabel("posterior")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    path2 = plots_dir / "model_posterior_over_time.png"
    _write_plot(fig2, path2)

    return [
        {
            "kind": "plot",
            "plot_id": "state_estimate_vs_truth",
            "section": "filtering",
            "relative_path": str(path1.relative_to(plots_dir.parents[0])),
            "source_path": "derived:run_kalman_bank_benchmark(seed=7)",
            "caption": "Representative Kalman state estimate against truth and measurements.",
            "interpretation": "Use this to see whether the correct model is tracking the latent trajectory rather than only winning a classifier vote.",
            "limitations": "This is a benchmark-witness trajectory from the native Kalman study, not the full common-study corpus.",
        },
        {
            "kind": "plot",
            "plot_id": "model_posterior_over_time",
            "section": "filtering",
            "relative_path": str(path2.relative_to(plots_dir.parents[0])),
            "source_path": "derived:run_kalman_bank_benchmark(seed=7)",
            "caption": "True-model posterior over time for the representative Kalman witness trajectory.",
            "interpretation": "Use this to inspect whether the model-based posterior improves steadily or only after late evidence arrives.",
            "limitations": "This is one trajectory and one family, not a universal filtering calibration study.",
        },
    ]


def _render_kalman_vs_windowed_comparison(plots_dir: Path) -> dict[str, object]:
    source = ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "common_dataset_metric_heatmap.png"
    destination = plots_dir / "kalman_vs_windowed_comparison.png"
    _copy_file(source, destination)
    return {
        "kind": "plot",
        "plot_id": "kalman_vs_windowed_comparison",
        "section": "filtering",
        "relative_path": str(destination.relative_to(plots_dir.parents[0])),
        "source_path": str(source.relative_to(ROOT)),
        "caption": "Shared-dataset metric heatmap highlighting Kalman versus windowed families.",
        "interpretation": "Use this to compare where model-based evidence changes the ranking relative to feature-only windowed baselines.",
        "limitations": "This is an aggregate scenario summary rather than a step-by-step filtering trace.",
    }


def _render_advanced_filter_decision_matrix(plots_dir: Path) -> list[dict[str, object]]:
    plt = _prepare_matplotlib()
    summary = _read_json(ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_summary.json")
    evidence = _read_json(ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_evidence.json")
    rows = [
        ("transition_gain", float(summary["transition_post_switch_gain"])),
        ("transition_vs_kalman", float(summary["transition_vs_kalman_post_switch_gain"])),
        ("velocity_aided_gain", float(summary["velocity_aided_short_noisy_gain"])),
        ("best_kalman_outlier", float(summary["best_kalman_outlier_accuracy"])),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar([name for name, _ in rows], [value for _, value in rows], color="#2563eb")
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_title("Advanced Filter Decision Matrix", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("evidence value")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path1 = plots_dir / "advanced_filter_decision_matrix.png"
    _write_plot(fig, path1)

    pf_rbpf_lines = [
        "gate,status,value,note",
        f"PF,{'defer' if not summary['particle_filter_justified'] else 'promote'},{summary['particle_filter_justified']},Particle filtering remains gated by current evidence.",
        "RBPF,defer,n/a,RBPF remains gated until future vector studies expose conditionally tractable mixed discrete/continuous structure.",
    ]
    path2 = plots_dir.parents[0] / "tables" / "pf_rbpf_go_no_go_table.csv"
    path2.parent.mkdir(parents=True, exist_ok=True)
    path2.write_text("\n".join(pf_rbpf_lines) + "\n", encoding="utf-8")
    return [
        {
            "kind": "plot",
            "plot_id": "advanced_filter_decision_matrix",
            "section": "advanced_filters",
            "relative_path": str(path1.relative_to(plots_dir.parents[0])),
            "source_path": "derived:advanced_filter_decision_summary.json",
            "caption": "Decision matrix summarizing the evidence currently relevant to advanced-filter escalation.",
            "interpretation": "Use this to see why advanced filters remain gated instead of being promoted automatically.",
            "limitations": "The matrix is a compact summary, not a replacement for the full decision report.",
        },
        {
            "kind": "table",
            "section": "advanced_filters",
            "relative_path": str(path2.relative_to(plots_dir.parents[0])),
            "source_path": "derived:advanced_filter_decision_summary.json",
        },
    ]


def _render_dimensional_lift_plots(plots_dir: Path) -> list[dict[str, object]]:
    plt = _prepare_matplotlib()
    rows = _read_csv(ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dimensional_status"]] = counts.get(row["dimensional_status"], 0) + 1
    fig1, ax1 = plt.subplots(figsize=(7.0, 4.2))
    labels = list(counts)
    ax1.bar(labels, [counts[label] for label in labels], color=["#2563eb", "#f59e0b", "#dc2626"][: len(labels)])
    ax1.set_title("Dimension-Lift Audit Chart", loc="left", fontsize=12, fontweight="bold")
    ax1.set_ylabel("module count")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.25)
    fig1.tight_layout()
    path1 = plots_dir / "dimension_lift_audit_chart.png"
    _write_plot(fig1, path1)

    taxonomy_rows = _read_json(ARTIFACTS_ROOT / "feature_taxonomy" / "feature_taxonomy.json")
    transfer_modes = ["dimension_agnostic", "adapter_compatible", "rewrite_required"]
    feature_names = [row["name"] for row in taxonomy_rows]
    matrix = [[1.0 if row["dimensional_transfer"] == mode else 0.0 for mode in transfer_modes] for row in taxonomy_rows]
    fig2, ax2 = plt.subplots(figsize=(7.5, max(4.5, 0.18 * len(feature_names))))
    image = ax2.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(len(transfer_modes)), transfer_modes, rotation=20, ha="right")
    ax2.set_yticks(range(len(feature_names)), feature_names)
    ax2.set_title("Feature Transfer Matrix", loc="left", fontsize=12, fontweight="bold")
    fig2.colorbar(image, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    path2 = plots_dir / "feature_transfer_matrix.png"
    _write_plot(fig2, path2)

    fig3, ax3 = plt.subplots(figsize=(9.0, 4.4))
    ax3.axis("off")
    boxes = [
        (0.10, 0.65, "Dimension-agnostic\ncontracts / evaluation"),
        (0.42, 0.65, "Adapter-compatible\nshared corpus / harness"),
        (0.74, 0.65, "1D-specific\nfeature + filter logic"),
    ]
    for x, y, text in boxes:
        ax3.text(x, y, text, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="#e5e7eb", edgecolor="#6b7280"))
    ax3.annotate("", xy=(0.34, 0.65), xytext=(0.18, 0.65), arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax3.annotate("", xy=(0.66, 0.65), xytext=(0.50, 0.65), arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax3.text(0.26, 0.78, "transfers directly", ha="center", fontsize=9)
    ax3.text(0.58, 0.78, "needs vector adapters", ha="center", fontsize=9)
    ax3.text(0.82, 0.48, "rewrite required for\ntrue 3D dynamics", ha="center", fontsize=9)
    ax3.set_title("Generic vs 1D-Specific Layer Diagram", loc="left", fontsize=12, fontweight="bold")
    path3 = plots_dir / "generic_vs_1d_specific_layer_diagram.png"
    _write_plot(fig3, path3)

    return [
        {
            "kind": "plot",
            "plot_id": "dimension_lift_audit_chart",
            "section": "dimensional_lift",
            "relative_path": str(path1.relative_to(plots_dir.parents[0])),
            "source_path": "derived:dimensional_lift_audit/module_dimension_status.csv",
            "caption": "Count of modules by dimensional-lift status.",
            "interpretation": "Use this to see how much of the stack already transfers to higher dimensions versus needing adapters or rewrites.",
            "limitations": "Counts summarize status classes; they do not capture implementation effort exactly.",
        },
        {
            "kind": "plot",
            "plot_id": "feature_transfer_matrix",
            "section": "dimensional_lift",
            "relative_path": str(path2.relative_to(plots_dir.parents[0])),
            "source_path": "derived:feature_taxonomy/feature_taxonomy.json",
            "caption": "Per-feature dimensional-transfer status matrix.",
            "interpretation": "Use this to see which feature families already generalize cleanly and which still embed scalar assumptions.",
            "limitations": "This reflects the current taxonomy declarations, not yet a vector empirical benchmark.",
        },
        {
            "kind": "plot",
            "plot_id": "generic_vs_1d_specific_layer_diagram",
            "section": "dimensional_lift",
            "relative_path": str(path3.relative_to(plots_dir.parents[0])),
            "source_path": "derived:dimensional_lift_audit/module_dimension_status.csv",
            "caption": "Layer diagram separating generic methodology surfaces from 1D-specific implementation layers.",
            "interpretation": "Use this to communicate what transfers directly to 3D versus what still needs vector adapters or rewrites.",
            "limitations": "This is a conceptual summary diagram rather than a runtime dependency graph.",
        },
    ]


def _generate_showcase_derived_plots(plots_dir: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    entries.append(_render_pointwise_vs_accumulator_posterior_timelines(plots_dir))
    quantile_source = ARTIFACTS_ROOT / "monte_carlo_accumulator" / "true_class_posterior_quantiles.png"
    quantile_destination = plots_dir / "true_class_posterior_quantiles.png"
    _copy_file(quantile_source, quantile_destination)
    entries.append(
        {
            "kind": "plot",
            "plot_id": "true_class_posterior_quantiles",
            "section": "history_comparison",
            "relative_path": str(quantile_destination.relative_to(plots_dir.parents[0])),
            "source_path": str(quantile_source.relative_to(ROOT)),
            "caption": "True-class posterior quantiles from the Monte Carlo accumulator benchmark.",
            "interpretation": "Use this to summarize how quickly true-class posterior mass grows across repeated runs rather than one witness trajectory.",
            "limitations": "This plot comes from the accumulator benchmark, not every classifier family.",
        }
    )
    ablation_source = ARTIFACTS_ROOT / "bayesian_walkthroughs" / "plots" / "feature_ablation_posterior.png"
    ablation_destination = plots_dir / "feature_ablation_posterior.png"
    _copy_file(ablation_source, ablation_destination)
    entries.append(
        {
            "kind": "plot",
            "plot_id": "feature_ablation_posterior",
            "section": "feature_confusion",
            "relative_path": str(ablation_destination.relative_to(plots_dir.parents[0])),
            "source_path": str(ablation_source.relative_to(ROOT)),
            "caption": "Posterior change after feature ablation on a representative feature-only walkthrough.",
            "interpretation": "Use this to see which feature contributed most to the final class preference in the witness case.",
            "limitations": "This remains a feature-only walkthrough, not a universal importance measure.",
        }
    )
    entries.append(_render_feature_distribution_by_class(plots_dir))
    entries.append(_render_feature_correlation_heatmap(plots_dir))
    entries.append(_render_duration_sensitivity_by_class_pair(plots_dir))
    entries.append(_render_corpus_adequacy_scorecard(plots_dir))
    pareto_source = ARTIFACTS_ROOT / "corpus_autodevelopment_v1" / "plots" / "corpus_score_pareto.png"
    pareto_destination = plots_dir / "candidate_corpus_comparison.png"
    _copy_file(pareto_source, pareto_destination)
    entries.append(
        {
            "kind": "plot",
            "plot_id": "candidate_corpus_comparison",
            "section": "corpus_validity",
            "relative_path": str(pareto_destination.relative_to(plots_dir.parents[0])),
            "source_path": str(pareto_source.relative_to(ROOT)),
            "caption": "Pareto-style comparison of candidate corpora under the current adequacy objective.",
            "interpretation": "Use this to see whether corpus improvement came from one clear winner or from tradeoffs between candidates.",
            "limitations": "All current candidates still fail the full adequacy gate, so this is a relative comparison rather than a final green corpus.",
        }
    )
    entries.append(_render_kalman_innovation_likelihood_timeline(plots_dir))
    entries.append(_render_kalman_vs_windowed_comparison(plots_dir))
    entries.extend(_render_kalman_state_estimate_vs_truth(plots_dir))
    entries.extend(_render_advanced_filter_decision_matrix(plots_dir))
    entries.extend(_render_dimensional_lift_plots(plots_dir))
    return entries


def _render_executive_report(summary: dict[str, object], corpus_summary: dict[str, object]) -> str:
    best_classifier = summary["best_common_study_classifier"]
    best_common_dataset = summary["best_common_dataset_method"]
    return "\n".join(
        [
            "# Executive Summary",
            "",
            "This packet packages the repo as a methodology and evidence suite rather than as a narrow 1D benchmark collection.",
            "",
            "## What Is Proven",
            "",
            f"- The best common-study classifier on the current manifest-aligned executable subset is `{best_classifier['classifier_id']}` at `{best_classifier['overall_accuracy']:.3f}`.",
            f"- The best method on the shared common-dataset comparison is `{best_common_dataset['method_name']}` at `{best_common_dataset['overall_accuracy']:.3f}`.",
            "- Pointwise, windowed, Bayesian accumulator, and Kalman-bank methods now share generic evidence, posterior, and artifact contracts.",
            "- Feature metadata, filtering contracts, and dimensional-lift status are all explicitly audited.",
            "",
            "## What Is Still Experimental Or Limited",
            "",
            f"- Corpus adequacy is currently `{corpus_summary['overall_status']}` rather than fully passing, so some comparisons remain intentionally diagnostic rather than final leaderboard claims.",
            "- PCA is used here as a diagnostic, not as a production classifier family.",
            "- IMM, PF, and RBPF remain behind explicit decision gates rather than being added prematurely.",
            "",
            "## Core Message",
            "",
            "- The repo is now best understood as a reusable methodology stack: corpus -> features -> evidence -> posterior -> filtering -> analysis -> visualization.",
            "- The 1D studies are the current proof surface, but the architecture is now documented and audited for 3D transition work.",
        ]
    )


def _render_problem_framing_report() -> str:
    return "\n".join(
        [
            "# Problem Framing",
            "",
            "The repo is solving a kinematic classification problem under uncertainty.",
            "",
            "The engineering question is not only which classifier scores best on one scalar corpus. The stronger question is whether the repo can support reusable studies across:",
            "",
            "- changing feature sets",
            "- changing class pairs",
            "- changing evidence providers",
            "- changing filter backends",
            "- changing sensor regimes",
            "- changing state dimension from 1D toward 3D",
            "",
            "That is why the showcase is organized around methodology contracts and evidence surfaces rather than only benchmark leaderboards.",
        ]
    )


def _render_methodology_report() -> str:
    return "\n".join(
        [
            "# Methodology Overview",
            "",
            "The current repo architecture is intentionally layered:",
            "",
            "1. corpus generation and study manifests",
            "2. feature extraction and feature taxonomy",
            "3. evidence production by classifier family",
            "4. posterior updating and confidence history",
            "5. optional filter backends and diagnostics",
            "6. metrics, identifiability, prior, coverage, and dimensional audits",
            "7. plots, reports, and packet exports",
            "",
            "Use the generic-methodology proof artifacts for contract details:",
            "",
            "- `generic_inference_contract/`",
            "- `classification_evidence_proof/`",
            "- `filtering_contract/`",
            "- `dimensional_lift_audit/`",
        ]
    )


def _render_algorithm_ladder_report(metrics_by_classifier: list[dict[str, str]], common_dataset_rows: list[dict[str, str]]) -> str:
    ladder_preview = _table_preview(
        metrics_by_classifier,
        ["classifier_id", "overall_accuracy", "num_predictions"],
        limit=len(metrics_by_classifier),
    )
    shared_preview = _table_preview(
        common_dataset_rows,
        ["method_name", "overall_accuracy", "endpoint_match_accuracy", "noisy_accuracy", "outlier_accuracy", "prior_flip_fraction"],
        limit=len(common_dataset_rows),
    )
    return "\n".join(
        [
            "# Algorithm Ladder",
            "",
            "The algorithm ladder currently spans direct pointwise methods through model-based Kalman methods.",
            "",
            "## Common Study Metrics",
            "",
            ladder_preview,
            "",
            "## Shared-Corpus Method Comparison",
            "",
            shared_preview,
            "",
            "## Reading Notes",
            "",
            "- `pointwise` is the weak lower bound and uses only instantaneous evidence.",
            "- `windowed_*` methods add history-derived features without a state-space model.",
            "- `bayes_accumulator` adds sequential evidence accumulation.",
            "- `kalman_bank` is the first explicitly model-based evidence provider.",
            "- `kalman_bank_velocity_aided` is a stronger sensor regime and should not be treated as a fair same-sensor upgrade over position-only methods.",
        ]
    )


def _render_feature_report(
    taxonomy_rows: list[dict[str, object]],
    identifiability_rows: list[dict[str, str]],
    oracle_rows: list[dict[str, str]],
) -> str:
    taxonomy_preview = _table_preview(
        [{k: str(v) for k, v in row.items()} for row in taxonomy_rows],
        ["name", "role", "history_behavior", "geometry_assumption", "dimensional_transfer"],
        limit=8,
    )
    identifiability_preview = _table_preview(
        identifiability_rows,
        ["class_pair_id", "feature_set_id", "mean_standardized_feature_distance", "overlap_estimate", "identifiability_status"],
        limit=10,
    )
    oracle_preview = _table_preview(
        oracle_rows,
        ["class_pair_id", "feature_set_id", "oracle_accuracy", "mean_posterior_margin", "is_best_feature_set"],
        limit=10,
    )
    return "\n".join(
        [
            "# Feature Taxonomy And Class-Pair Studies",
            "",
            "This section ties together feature metadata, pairwise separability, and feature-only oracle baselines.",
            "",
            "## Feature Taxonomy Preview",
            "",
            taxonomy_preview,
            "",
            "## Class-Pair Identifiability Preview",
            "",
            identifiability_preview,
            "",
            "## Oracle Separability Preview",
            "",
            oracle_preview,
            "",
            "## Interpretation",
            "",
            "- Use identifiability rows to decide whether a confusion is fundamentally hard in feature space.",
            "- Use oracle rows to decide whether the classifier is underperforming the available feature signal.",
            "- Use taxonomy metadata to separate dimension-agnostic features from 1D-specific geometric assumptions.",
        ]
    )


def _render_filtering_report(advanced_summary: dict[str, object], transition_rows: list[dict[str, str]]) -> str:
    transition_preview = _table_preview(
        transition_rows,
        ["scenario_name", "static_accuracy", "transition_accuracy", "kalman_accuracy", "transition_post_switch"],
        limit=8,
    )
    return "\n".join(
        [
            "# Filtering And Advanced-Method Decision",
            "",
            f"- IMM justified now: `{advanced_summary['imm_justified']}`",
            f"- Particle filter justified now: `{advanced_summary['particle_filter_justified']}`",
            "",
            "## Switching Preview",
            "",
            transition_preview,
            "",
            "## Decision Notes",
            "",
            "- Transition-aware accumulation already improves switching behavior and still outperforms the current switching Kalman mode bank post-switch.",
            "- The strongest short-horizon hard case is still evidence-limited rather than a clean advanced-inference failure case.",
            "- IMM, PF, and RBPF therefore remain explicit no-go items for now.",
        ]
    )


def _render_corpus_report(corpus_summary: dict[str, object], class_pair_rows: list[dict[str, str]], leakage_rows: list[dict[str, str]]) -> str:
    pair_preview = _table_preview(
        class_pair_rows,
        ["class_a", "class_b", "difficulty", "pairwise_auc", "overlap", "status"],
        limit=8,
    )
    leakage_preview = _table_preview(
        leakage_rows,
        ["covariate", "max_pairwise_auc", "spread_ratio", "worst_pair", "status"],
        limit=8,
    )
    return "\n".join(
        [
            "# Corpus Adequacy And Leakage",
            "",
            f"- Overall corpus status: `{corpus_summary['overall_status']}`",
            f"- Feature coverage: `{corpus_summary['feature_status']}`",
            f"- Class-pair coverage: `{corpus_summary['class_pair_status']}`",
            f"- Covariate leakage: `{corpus_summary['covariate_status']}`",
            "",
            "## Class-Pair Boundary Coverage",
            "",
            pair_preview,
            "",
            "## Covariate Leakage Preview",
            "",
            leakage_preview,
            "",
            "## Reading Notes",
            "",
            "- Treat this report as a credibility layer for all downstream classifier comparisons.",
            "- A failing corpus result means the benchmark is still useful diagnostically, but not all leaderboard conclusions should be treated as stable.",
        ]
    )


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
    rows = [{"study": name, "purpose": purpose, "primary_report": report} for name, purpose, report in study_rows]
    return "\n".join(
        [
            "# Study Suite",
            "",
            _table_preview(rows, ["study", "purpose", "primary_report"], limit=len(rows)),
            "",
            "These studies are designed to answer different methodology questions rather than all collapsing into a single leaderboard.",
        ]
    )


def _render_gallery_report(plot_entries: list[dict[str, object]]) -> str:
    sections: list[str] = ["# Visualization Gallery", "", "Plots are grouped by the question they answer."]
    for entry in plot_entries:
        plot_id = entry.get("plot_id")
        if entry.get("kind") != "plot" or not plot_id:
            continue
        sections.extend(
            [
                "",
                f"## {plot_id}",
                "",
                f"![{plot_id}](../plots/{Path(str(entry['relative_path'])).name})",
                "",
                f"- Caption: {entry.get('caption', 'n/a')}",
                f"- Interpretation: {entry.get('interpretation', 'n/a')}",
                f"- Limitations: {entry.get('limitations', 'n/a')}",
                f"- Source artifact: `{entry.get('source_path', 'n/a')}`",
            ]
        )
    return "\n".join(sections)


def _render_results_summary_report(summary: dict[str, object]) -> str:
    dimensional_counts = summary["dimensional_status_counts"]
    return "\n".join(
        [
            "# Results Summary",
            "",
            "## Headline Findings",
            "",
            f"- Best common-study classifier: `{summary['best_common_study_classifier']['classifier_id']}` at `{summary['best_common_study_classifier']['overall_accuracy']:.3f}`.",
            f"- Best shared-corpus method: `{summary['best_common_dataset_method']['method_name']}` at `{summary['best_common_dataset_method']['overall_accuracy']:.3f}`.",
            f"- Corpus adequacy status: `{summary['corpus_adequacy']['overall_status']}`.",
            f"- IMM justified now: `{summary['advanced_filters']['imm_justified']}`.",
            f"- Particle filter justified now: `{summary['advanced_filters']['particle_filter_justified']}`.",
            f"- Dimensional status counts: `{dimensional_counts}`.",
            "",
            "## Takeaway",
            "",
            "- The repo now proves reusable methodology layers more strongly than it proves a final optimized 1D leaderboard.",
            "- The strongest next work should therefore be new study families, new sensors, or new 3D-capable adapters rather than more blind 1D tuning.",
        ]
    )


def _render_3d_transition_report(dimension_rows: list[dict[str, str]]) -> str:
    preview = _table_preview(
        dimension_rows,
        ["module", "layer", "dimensional_status", "required_3d_action"],
        limit=len(dimension_rows),
    )
    return "\n".join(
        [
            "# 3D Transition Plan",
            "",
            "This report shows what remains generic, what is adapter-compatible, and what still needs explicit 3D-specific rewrite work.",
            "",
            preview,
            "",
            "## Reading Notes",
            "",
            "- `dimension_agnostic` means the current abstraction is already suitable for vector-capable work.",
            "- `adapter_compatible` means the contract is good but current study data or helper code is still scalar-scoped.",
            "- `rewrite_required` means the implementation itself assumes scalar geometry or scalar dynamics.",
        ]
    )


def _render_open_risks_report(corpus_payload: dict[str, object]) -> str:
    recommendations = corpus_payload.get("recommendations", [])
    lines = [
        "# Open Risks And Next Steps",
        "",
        "## Current Risks",
        "",
        "- The current corpus adequacy gate is not fully green, so some comparisons remain deliberately diagnostic.",
        "- PCA is documented only as a diagnostic, not as a classifier family.",
        "- The current dimensional-lift proof stops at fake vector-compatible methodology flow, not full 3D dynamics or filtering.",
        "- Advanced filters remain unjustified on current evidence, so adding them now would lower methodological discipline rather than improve it.",
        "",
        "## Immediate Next Steps",
        "",
    ]
    if recommendations:
        lines.extend(f"- {recommendation}" for recommendation in recommendations)
    else:
        lines.append("- No open corpus recommendations were emitted.")
    lines.extend(
        [
            "- Add a true 3D-capable corpus adapter and vector feature registry on top of the dimensional-lift contract.",
            "- Decide whether a PCA-classifier study is actually desired; it is not currently part of the proven methodology stack.",
        ]
    )
    return "\n".join(lines)


def _build_run_cards(run_cards_dir: Path) -> list[dict[str, object]]:
    cards = [
        {
            "filename": "common_study.md",
            "title": "Common Study Run Card",
            "body": "\n".join(
                [
                    "# Common Study Run Card",
                    "",
                    "- Study: `common_1d_classifier_study`",
                    "- Purpose: compare classifier families on manifest-declared class pairs using one unified artifact contract",
                    "- Primary report: `../reports/03_algorithm_ladder.md`",
                    "- Primary source artifact: `../../common_1d_classifier_study/common_experiment_report.md`",
                    "- Rerun: `python3 scripts/run_study.py experiments/common_1d_classifier_study/common_experiment_config.yaml`",
                ]
            ),
        },
        {
            "filename": "common_dataset_comparison.md",
            "title": "Shared-Corpus Comparison Run Card",
            "body": "\n".join(
                [
                    "# Shared-Corpus Comparison Run Card",
                    "",
                    "- Study: `common_dataset_comparison_v1`",
                    "- Purpose: compare methods on the same shared binary dynamics corpus",
                    "- Primary report: `../reports/03_algorithm_ladder.md`",
                    "- Primary source artifact: `../../common_dataset_comparison_v1/common_dataset_comparison_report.md`",
                ]
            ),
        },
        {
            "filename": "feature_analysis.md",
            "title": "Feature Analysis Run Card",
            "body": "\n".join(
                [
                    "# Feature Analysis Run Card",
                    "",
                    "- Study: `feature_analysis_v1`",
                    "- Purpose: identify feature excitation, class confusability, and separability limits",
                    "- Primary report: `../reports/04_feature_taxonomy.md`",
                    "- Primary source artifact: `../../feature_analysis_v1/feature_analysis_report.md`",
                ]
            ),
        },
        {
            "filename": "corpus_adequacy.md",
            "title": "Corpus Adequacy Run Card",
            "body": "\n".join(
                [
                    "# Corpus Adequacy Run Card",
                    "",
                    "- Study: `corpus_adequacy_audit_v1`",
                    "- Purpose: determine whether the synthetic corpus is credible enough to support classifier comparisons",
                    "- Primary report: `../reports/08_results_summary.md`",
                    "- Primary source artifact: `../../corpus_adequacy_audit_v1/corpus_adequacy_report.md`",
                ]
            ),
        },
        {
            "filename": "advanced_filter_decision.md",
            "title": "Advanced Filter Decision Run Card",
            "body": "\n".join(
                [
                    "# Advanced Filter Decision Run Card",
                    "",
                    "- Study: `advanced_filter_decision_v1`",
                    "- Purpose: decide whether IMM, PF, or RBPF is justified by current failure cases",
                    "- Primary report: `../reports/05_filtering_taxonomy.md`",
                    "- Primary source artifact: `../../advanced_filter_decision_v1/advanced_filter_decision_report.md`",
                ]
            ),
        },
        {
            "filename": "dimensional_lift.md",
            "title": "Dimensional Lift Run Card",
            "body": "\n".join(
                [
                    "# Dimensional Lift Run Card",
                    "",
                    "- Study: `dimensional_lift_audit`",
                    "- Purpose: identify which methodology layers already support 3D transition work and which remain scalar-specific",
                    "- Primary report: `../reports/09_3d_transition_plan.md`",
                    "- Primary source artifact: `../../dimensional_lift_audit/dimensional_lift_audit.md`",
                ]
            ),
        },
    ]
    manifest_entries: list[dict[str, object]] = []
    for card in cards:
        path = run_cards_dir / card["filename"]
        _write_text(path, card["body"])
        manifest_entries.append(
            {
                "kind": "run_card",
                "title": card["title"],
                "relative_path": str(path.relative_to(run_cards_dir.parents[0])),
            }
        )
    return manifest_entries


def _copy_showcase_sources(reports_dir: Path) -> list[dict[str, object]]:
    manifest_entries: list[dict[str, object]] = []
    for source in sorted(SHOWCASE_DOCS_DIR.glob("*.md")):
        destination = reports_dir / "source_docs" / source.name
        _copy_file(source, destination)
        manifest_entries.append(
            {
                "kind": "source_doc",
                "relative_path": str(destination.relative_to(reports_dir.parents[0])),
                "source_path": str(source.relative_to(ROOT)),
            }
        )
    return manifest_entries


def _required_report_names() -> tuple[str, ...]:
    return (
        "00_executive_summary.md",
        "01_problem_framing.md",
        "02_methodology_overview.md",
        "03_algorithm_ladder.md",
        "04_feature_taxonomy.md",
        "05_filtering_taxonomy.md",
        "06_study_suite.md",
        "07_visualization_gallery.md",
        "08_results_summary.md",
        "09_3d_transition_plan.md",
        "10_open_risks_and_next_steps.md",
    )


def build_showcase_artifacts(
    output_dir: str | Path = ARTIFACTS_ROOT,
    *,
    refresh: bool = False,
    create_zip: bool = False,
) -> ShowcaseArtifacts:
    if refresh:
        subprocess.run(["python3", "scripts/export_artifacts.py"], cwd=ROOT, check=True)

    output_root = Path(output_dir)
    showcase_dir = output_root / "showcase"
    reports_dir = showcase_dir / "reports"
    plots_dir = showcase_dir / "plots"
    tables_dir = showcase_dir / "tables"
    run_cards_dir = showcase_dir / "run_cards"
    proof_gallery_path = showcase_dir / "proof_gallery.md"
    team_packet_dir = output_root / "team_packet"
    zip_path = output_root / "kinematic_classifier_team_packet.zip" if create_zip else None
    validation_path = showcase_dir / "validation_results.json"

    if showcase_dir.exists():
        shutil.rmtree(showcase_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    run_cards_dir.mkdir(parents=True, exist_ok=True)

    summary = _headline_summary()
    corpus_payload = _read_json(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "corpus_adequacy_summary.json")
    metrics_by_classifier = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_classifier.csv")
    common_dataset_rows = _read_csv(ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_summary.csv")
    taxonomy_rows = _read_json(ARTIFACTS_ROOT / "feature_taxonomy" / "feature_taxonomy.json")
    identifiability_rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "identifiability_matrix.csv")
    oracle_rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "oracle_classifier_results.csv")
    advanced_summary = _read_json(ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_summary.json")
    transition_rows = _read_csv(ARTIFACTS_ROOT / "transition_matrix_accumulator_v1" / "transition_matrix_scenario_summary.csv")
    class_pair_rows = _read_csv(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "class_pair_coverage.csv")
    leakage_rows = _read_csv(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "covariate_leakage_audit.csv")
    dimension_rows = _read_csv(ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv")

    report_payloads = {
        "00_executive_summary.md": _render_executive_report(summary, corpus_payload["summary"]),
        "01_problem_framing.md": _render_problem_framing_report(),
        "02_methodology_overview.md": _render_methodology_report(),
        "03_algorithm_ladder.md": _render_algorithm_ladder_report(metrics_by_classifier, common_dataset_rows),
        "04_feature_taxonomy.md": _render_feature_report(taxonomy_rows, identifiability_rows, oracle_rows),
        "05_filtering_taxonomy.md": _render_filtering_report(advanced_summary, transition_rows),
        "06_study_suite.md": _render_study_suite_report(),
        "07_visualization_gallery.md": "",
        "08_results_summary.md": _render_results_summary_report(summary),
        "09_3d_transition_plan.md": _render_3d_transition_report(dimension_rows),
        "10_open_risks_and_next_steps.md": _render_open_risks_report(corpus_payload),
    }

    manifest_entries: list[dict[str, object]] = []
    manifest_entries.extend(_copy_showcase_sources(reports_dir))
    plot_entries = _copy_showcase_plots(plots_dir)
    plot_entries.extend(_generate_showcase_derived_plots(plots_dir))
    manifest_entries.extend(plot_entries)
    manifest_entries.extend(_copy_showcase_tables(tables_dir))
    manifest_entries.extend(_build_run_cards(run_cards_dir))

    report_payloads["07_visualization_gallery.md"] = _render_gallery_report(plot_entries)

    for filename, body in report_payloads.items():
        path = reports_dir / filename
        _write_text(path, body)
        manifest_entries.append(
            {
                "kind": "report",
                "relative_path": str(path.relative_to(showcase_dir)),
                "title": filename.removesuffix(".md"),
            }
        )

    summary_metrics_path = showcase_dir / "summary_metrics.json"
    _write_json(summary_metrics_path, summary)
    manifest_entries.append(
        {
            "kind": "summary_metrics",
            "relative_path": str(summary_metrics_path.relative_to(showcase_dir)),
        }
    )

    index_lines = [
        "# Team-Facing Methodology Showcase",
        "",
        "This index is the team-facing packet entrypoint for the kinematic-classifier methodology stack.",
        "",
        "## Claim-Oriented Entry Point",
        "",
        "- [proof_gallery.md](proof_gallery.md)",
        "",
        "## Reports",
        "",
    ]
    for report_name in _required_report_names():
        index_lines.append(f"- [reports/{report_name}](reports/{report_name})")
    index_lines.extend(
        [
            "",
            "## Run Cards",
            "",
        ]
    )
    for card_path in sorted(run_cards_dir.glob("*.md")):
        index_lines.append(f"- [run_cards/{card_path.name}](run_cards/{card_path.name})")
    index_lines.extend(
        [
            "",
            "## Tables",
            "",
        ]
    )
    for table_path in sorted(tables_dir.iterdir()):
        if table_path.is_file():
            index_lines.append(f"- [tables/{table_path.name}](tables/{table_path.name})")
    index_lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for plot_path in sorted(plots_dir.iterdir()):
        if plot_path.is_file():
            index_lines.append(f"- [plots/{plot_path.name}](plots/{plot_path.name})")
    index_lines.extend(
        [
            "",
            "## Rerun Flow",
            "",
            "- Refresh source artifacts: `python3 scripts/export_artifacts.py`",
            "- Rebuild showcase: `python3 scripts/build_showcase.py`",
            "- Export team packet: `python3 scripts/export_team_packet.py --zip`",
        ]
    )
    index_path = showcase_dir / "index.md"
    _write_text(index_path, "\n".join(index_lines))
    _write_text(proof_gallery_path, _render_proof_gallery())

    artifact_manifest_path = showcase_dir / "artifact_manifest.json"
    manifest_entries.append(
        {
            "kind": "index",
            "relative_path": str(index_path.relative_to(showcase_dir)),
        }
    )
    manifest_entries.append(
        {
            "kind": "proof_gallery",
            "relative_path": str(proof_gallery_path.relative_to(showcase_dir)),
        }
    )

    _write_json(artifact_manifest_path, {"items": manifest_entries})
    validation = validate_showcase_artifacts(showcase_dir)
    _write_json(validation_path, asdict(validation))
    manifest_entries.append(
        {
            "kind": "validation",
            "relative_path": str(validation_path.relative_to(showcase_dir)),
        }
    )
    _write_json(artifact_manifest_path, {"items": manifest_entries})

    if team_packet_dir.exists():
        shutil.rmtree(team_packet_dir)
    shutil.copytree(showcase_dir, team_packet_dir)

    if create_zip and zip_path is not None:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(team_packet_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(Path("team_packet") / path.relative_to(team_packet_dir)))

    return ShowcaseArtifacts(
        showcase_dir=showcase_dir,
        index_path=index_path,
        proof_gallery_path=proof_gallery_path,
        artifact_manifest_path=artifact_manifest_path,
        summary_metrics_path=summary_metrics_path,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        run_cards_dir=run_cards_dir,
        team_packet_dir=team_packet_dir,
        zip_path=zip_path,
        validation_path=validation_path,
    )


def validate_showcase_artifacts(showcase_dir: str | Path) -> ShowcaseValidationResult:
    root = Path(showcase_dir)
    errors: list[str] = []
    reports_dir = root / "reports"
    plots_dir = root / "plots"
    tables_dir = root / "tables"
    run_cards_dir = root / "run_cards"
    proof_gallery_path = root / "proof_gallery.md"
    manifest_path = root / "artifact_manifest.json"

    required_reports_exist = True
    for report_name in _required_report_names():
        path = reports_dir / report_name
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            required_reports_exist = False
            errors.append(f"missing or empty report: {report_name}")

    proof_gallery_complete = True
    proof_gallery_references_exist = True
    required_claim_headings = tuple(f"## Claim {index}:" for index in range(1, 9))
    if not proof_gallery_path.exists() or not proof_gallery_path.read_text(encoding="utf-8").strip():
        proof_gallery_complete = False
        proof_gallery_references_exist = False
        errors.append("proof_gallery.md is missing or empty")
        proof_gallery_text = ""
    else:
        proof_gallery_text = proof_gallery_path.read_text(encoding="utf-8")
        for heading in required_claim_headings:
            if heading not in proof_gallery_text:
                proof_gallery_complete = False
                errors.append(f"proof gallery missing claim section: {heading}")
        for target in _extract_markdown_relative_targets(proof_gallery_text):
            if not (root / target).exists():
                proof_gallery_references_exist = False
                errors.append(f"proof gallery missing referenced artifact: {target}")

    manifest_complete = manifest_path.exists()
    manifest_items = _read_json(manifest_path)["items"] if manifest_complete else []
    if not manifest_complete:
        errors.append("artifact_manifest.json is missing")

    metrics_tables_exist = True
    for filename in (
        "metrics_by_classifier.csv",
        "metrics_by_class_pair.csv",
        "feature_set_comparison.csv",
        "identifiability_matrix.csv",
    ):
        path = tables_dir / filename
        if not path.exists() or len(path.read_text(encoding="utf-8").splitlines()) <= 1:
            metrics_tables_exist = False
            errors.append(f"missing or empty metrics table: {filename}")

    plot_entries = [entry for entry in manifest_items if entry.get("kind") == "plot"]
    gallery_references_exist = True
    gallery_annotations_complete = True
    for entry in plot_entries:
        relative_path = root / str(entry["relative_path"])
        if not relative_path.exists():
            gallery_references_exist = False
            errors.append(f"missing gallery file: {entry['relative_path']}")
        if not entry.get("caption") or not entry.get("interpretation"):
            gallery_annotations_complete = False
            errors.append(f"incomplete gallery annotation: {entry['relative_path']}")

    feature_taxonomy_complete = True
    taxonomy_path = tables_dir / "feature_taxonomy.json"
    if taxonomy_path.exists():
        taxonomy_rows = _read_json(taxonomy_path)
        required_feature_keys = {
            "name",
            "role",
            "history_behavior",
            "geometry_assumption",
            "dimensional_transfer",
            "dependency_tags",
            "sensitivity_tags",
        }
        for row in taxonomy_rows:
            missing = required_feature_keys.difference(row)
            if missing:
                feature_taxonomy_complete = False
                errors.append(f"feature taxonomy missing keys for {row.get('name', 'unknown')}: {sorted(missing)}")
                break
    else:
        feature_taxonomy_complete = False
        errors.append("feature_taxonomy.json is missing")

    class_pair_identifiability_complete = True
    identifiability_path = tables_dir / "identifiability_matrix.csv"
    if identifiability_path.exists():
        rows = _read_csv(identifiability_path)
        pair_ids = {row["class_pair_id"] for row in rows}
        manifest_pairs = _read_json(ROOT / "experiments" / "common_1d_classifier_study" / "class_pair_manifest.json")["class_pairs"]
        expected = {"_vs_".join(row["pair"]) for row in manifest_pairs}
        if not expected.issubset(pair_ids):
            class_pair_identifiability_complete = False
            errors.append("not every declared class pair has an identifiability row in the packet")
    else:
        class_pair_identifiability_complete = False
        errors.append("identifiability_matrix.csv is missing from packet tables")

    advanced_filter_go_no_go_present = True
    filtering_report = reports_dir / "05_filtering_taxonomy.md"
    if filtering_report.exists():
        text = filtering_report.read_text(encoding="utf-8")
        if "IMM justified now" not in text or "Particle filter justified now" not in text:
            advanced_filter_go_no_go_present = False
            errors.append("advanced-method go/no-go status missing from filtering report")
    else:
        advanced_filter_go_no_go_present = False
        errors.append("filtering report is missing")

    dimensional_status_present = True
    transition_report = reports_dir / "09_3d_transition_plan.md"
    if transition_report.exists():
        text = transition_report.read_text(encoding="utf-8")
        if "dimension_agnostic" not in text or "adapter_compatible" not in text or "rewrite_required" not in text:
            dimensional_status_present = False
            errors.append("3D transition report does not name dimensional status categories")
    else:
        dimensional_status_present = False
        errors.append("3D transition report is missing")

    if not plots_dir.exists():
        gallery_references_exist = False
        errors.append("plots directory is missing")
    if not tables_dir.exists():
        metrics_tables_exist = False
        errors.append("tables directory is missing")
    if not run_cards_dir.exists():
        errors.append("run_cards directory is missing")

    overall_status = "pass" if not errors else "fail"
    return ShowcaseValidationResult(
        overall_status=overall_status,
        required_reports_exist=required_reports_exist,
        proof_gallery_complete=proof_gallery_complete,
        manifest_complete=manifest_complete,
        metrics_tables_exist=metrics_tables_exist,
        gallery_references_exist=gallery_references_exist,
        proof_gallery_references_exist=proof_gallery_references_exist,
        gallery_annotations_complete=gallery_annotations_complete,
        feature_taxonomy_complete=feature_taxonomy_complete,
        class_pair_identifiability_complete=class_pair_identifiability_complete,
        advanced_filter_go_no_go_present=advanced_filter_go_no_go_present,
        dimensional_status_present=dimensional_status_present,
        errors=tuple(errors),
    )
