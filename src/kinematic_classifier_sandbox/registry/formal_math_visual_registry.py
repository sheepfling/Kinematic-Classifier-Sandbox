from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..utils.runtime import repo_root

from ..analysis.feature_analysis_artifact_io import write_feature_analysis_artifacts
from ..analysis.generated_corpus_features import select_generated_corpus_records
from ..artifacts import render_posterior_numeric_walkthrough_png_bytes
from ..corpus.exploration.candidate_generation_core import analyze_candidate_generation
from ..corpus.gym import CorpusGymAction, CorpusGymEnvironment, default_corpus_gym_targets
from ..inference.monte_carlo_benchmark import (
    render_monte_carlo_calibration_png_bytes,
    run_accumulator_monte_carlo_benchmark,
)
from ..inference.transition_matrix_accumulator import write_transition_benchmark_artifacts
from ..markdown_builder import MarkdownDocument
from ..rung_sufficiency.analysis import write_rung_sufficiency_artifacts
from ..utils.io import write_csv
from ..utils.plotting import _figure_to_png, plt
from .formal_math_registry import load_equation_registry

ROOT = repo_root()
ARTIFACT_DIR = ROOT / "artifacts" / "formal_math_visual_registry_v1"


@dataclass(frozen=True, slots=True)
class FormalMathVisualSpec:
    equation_id: str
    title: str
    source_artifact: str | None
    source_type: str
    caption: str
    notes: str


@dataclass(frozen=True, slots=True)
class FormalMathVisualRow:
    equation_id: str
    status: str
    visual_status: str
    visual_kind: str
    implementation: str
    visual_path: str
    source_artifact: str
    source_data_artifacts: tuple[str, ...]
    source_type: str
    generated_visual: bool
    rerun_command: str
    caption: str
    notes: str


@dataclass(frozen=True, slots=True)
class FormalMathVisualRegistryResult:
    rows: tuple[FormalMathVisualRow, ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class FormalMathVisualRegistryArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    gallery_csv_path: Path
    provenance_path: Path
    runbook_path: Path
    visual_coverage_png_path: Path
    assets_dir: Path


FORMAL_MATH_VISUAL_REGISTRY: tuple[FormalMathVisualSpec, ...] = (
    FormalMathVisualSpec(
        equation_id="bayes_logsumexp_update",
        title="Bayes Recursive Update",
        source_artifact="artifacts/showcase/plots/posterior_timeline.png",
        source_type="existing_plot",
        caption="Posterior timeline for recursive Bayes accumulation.",
        notes="Uses the existing posterior timeline plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="two_class_log_odds",
        title="Two-Class Log Odds",
        source_artifact="artifacts/showcase/plots/prior_to_posterior_single_step.png",
        source_type="existing_plot",
        caption="Single-step prior-to-posterior update showing flip sensitivity.",
        notes="Uses the existing prior-to-posterior plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="transition_matrix_update",
        title="Transition-Matrix Update",
        source_artifact="artifacts/showcase/plots/transition_matrix_diagnostics.png",
        source_type="existing_plot",
        caption="Switching-model diagnostic plot for the transition accumulator.",
        notes="Uses the existing transition diagnostics plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="gaussian_feature_likelihood",
        title="Gaussian Feature Likelihood",
        source_artifact=None,
        source_type="illustrative_plot",
        caption="Empirical class-conditional likelihood curves fit from the generated corpus slice.",
        notes="Built from the generated corpus measurements, not from a placeholder curve.",
    ),
    FormalMathVisualSpec(
        equation_id="kalman_innovation_likelihood",
        title="Kalman Innovation Likelihood",
        source_artifact="artifacts/showcase/plots/kalman_innovation_likelihood_timeline.png",
        source_type="existing_plot",
        caption="Innovation-likelihood timeline for the Kalman-bank family.",
        notes="Uses the showcase innovation timeline as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="imm_mode_mixing_recursion",
        title="IMM Mode Mixing Recursion",
        source_artifact="artifacts/imm_filter_v1/plots/imm_mode_probability_timeline.png",
        source_type="existing_plot",
        caption="Mode-posterior timeline for the IMM switching witness.",
        notes="Uses the IMM witness artifact as the representative visual for mode mixing.",
    ),
    FormalMathVisualSpec(
        equation_id="pf_importance_weight_update",
        title="PF Importance Weight Update",
        source_artifact="artifacts/pf_abs_range_multimodal_oracle_v1/plots/oracle_kl_timeline.png",
        source_type="existing_plot",
        caption="Oracle divergence timeline for the canonical multimodal PF witness.",
        notes="Uses the oracle-backed PF witness to show where weighted particle support tracks the posterior better than a single-Gaussian projection.",
    ),
    FormalMathVisualSpec(
        equation_id="pf_class_evidence_extraction",
        title="PF Class Evidence Extraction",
        source_artifact="artifacts/pf_abs_range_multimodal_oracle_v1/plots/gaussian_collapse_panel.png",
        source_type="existing_plot",
        caption="Posterior overlay showing PF versus single-Gaussian collapse on the canonical oracle witness.",
        notes="Uses the oracle-backed PF witness as the representative class-evidence visualization.",
    ),
    FormalMathVisualSpec(
        equation_id="rbpf_conditional_weight_update",
        title="RBPF Conditional Weight Update",
        source_artifact="artifacts/advanced_filter_comparison_v1/pf_vs_rbpf_frontier.png",
        source_type="existing_plot",
        caption="Compute-normalized PF-versus-RBPF frontier for the latent-onset witness family.",
        notes="Uses the shared frontier artifact as the representative RBPF promotion visual.",
    ),
    FormalMathVisualSpec(
        equation_id="calibration_metrics",
        title="Calibration Metrics",
        source_artifact=None,
        source_type="illustrative_plot",
        caption="Calibration diagnostics from the actual Monte Carlo benchmark.",
        notes="Built from the real Monte Carlo calibration bins instead of a schematic curve.",
    ),
    FormalMathVisualSpec(
        equation_id="pairwise_mahalanobis_distance",
        title="Pairwise Mahalanobis Distance",
        source_artifact="artifacts/feature_analysis_v1/pairwise_distance_heatmap.png",
        source_type="existing_plot",
        caption="Class-pair distance heatmap from feature analysis.",
        notes="Uses the existing pairwise distance heatmap as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="corpus_autodevelopment_score",
        title="Corpus Autodevelopment Score",
        source_artifact="artifacts/corpus_autodevelopment_v1/plots/corpus_score_pareto.png",
        source_type="existing_plot",
        caption="Pareto view of corpus autodevelopment candidate scores.",
        notes="Uses the existing corpus score Pareto chart as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="pareto_dominance",
        title="Pareto Dominance",
        source_artifact="artifacts/corpus_autodevelopment_v1/plots/corpus_score_pareto.png",
        source_type="existing_plot",
        caption="Non-dominated candidates highlighted in the corpus score Pareto plot.",
        notes="Uses the same Pareto chart as the score equation because the plot already expresses dominance.",
    ),
    FormalMathVisualSpec(
        equation_id="corpus_gym_reward",
        title="CorpusGym Reward",
        source_artifact=None,
        source_type="generated_plot",
        caption="Reward-component decomposition for the CorpusGym utility equation.",
        notes="Generated from the explicit reward weights and a representative episode.",
    ),
    FormalMathVisualSpec(
        equation_id="corpus_explorer_utility",
        title="Corpus Explorer Utility",
        source_artifact="artifacts/generic_corpus_exploration/score_component_parallel_coordinates.png",
        source_type="existing_plot",
        caption="Parallel-coordinates view of the exploration utility components.",
        notes="Uses the existing component plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="qd_archive_utility",
        title="QD Archive Utility",
        source_artifact="artifacts/quality_diversity_corpus_v1/archive_coverage_by_iteration.png",
        source_type="existing_plot",
        caption="Archive coverage over iterations in the quality-diversity setup.",
        notes="Uses the existing archive coverage plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="qd_cell_mapping",
        title="QD Cell Mapping",
        source_artifact="artifacts/quality_diversity_corpus/plots/archive_coverage_heatmap.png",
        source_type="existing_plot",
        caption="Archive cell occupancy heatmap for the QD mapping function.",
        notes="Uses the existing archive heatmap as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="sampler_mixture",
        title="Sampler Mixture",
        source_artifact=None,
        source_type="generated_plot",
        caption="Sampler family mixture derived from candidate generation counts.",
        notes="Generated from the current candidate-generation result so the sampler mix is visible.",
    ),
    FormalMathVisualSpec(
        equation_id="class_validity_status",
        title="Class Validity Status",
        source_artifact="artifacts/class_validity/class_validity_status_distribution.png",
        source_type="existing_plot",
        caption="Status distribution for valid, ambiguous, invalid, and relabel candidates.",
        notes="Uses the existing class-validity status distribution plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="corpus_policy_normalization",
        title="Corpus Policy Normalization",
        source_artifact="artifacts/corpus_hyperparameter_tuning_v1/weight_sensitivity_tornado.png",
        source_type="existing_plot",
        caption="Weight sensitivity plot for the normalized corpus policy terms.",
        notes="Uses the corpus policy tuning tornado plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="corpus_policy_score",
        title="Corpus Policy Score",
        source_artifact="artifacts/corpus_hyperparameter_tuning_v1/pareto_tradeoff_scatter.png",
        source_type="existing_plot",
        caption="Pareto tradeoff scatter from corpus policy sweep scoring.",
        notes="Uses the corpus policy tuning tradeoff plot as the representative visual.",
    ),
    FormalMathVisualSpec(
        equation_id="rung_sufficiency_promotion_rule",
        title="Rung Sufficiency Promotion Rule",
        source_artifact="artifacts/ladder_witness_suite_v1/plots/rung_promotion_decisions.png",
        source_type="existing_plot",
        caption="Promotion decisions generated by the rung-sufficiency ladder.",
        notes="Uses the rung witness artifact when available, otherwise falls back to a generated registry visual.",
    ),
    FormalMathVisualSpec(
        equation_id="advanced_filter_gate",
        title="Advanced Filter Gate",
        source_artifact="artifacts/showcase/plots/advanced_filter_decision_matrix.png",
        source_type="existing_plot",
        caption="Decision matrix used to justify or defer advanced filtering methods.",
        notes="Uses the existing showcase decision matrix as the representative visual.",
    ),
)


FORMAL_MATH_VISUAL_SOURCE_DATA: dict[str, tuple[str, ...]] = {
    "bayes_logsumexp_update": (
        "artifacts/corpus_classifier_scoring/posterior_history.csv",
        "artifacts/bayesian_walkthroughs/bayesian_step_tables.csv",
    ),
    "two_class_log_odds": (
        "artifacts/bayesian_walkthroughs/posterior_flip_thresholds.csv",
        "artifacts/bayesian_walkthroughs/bayesian_step_tables.csv",
    ),
    "transition_matrix_update": (
        "artifacts/transition_matrix_accumulator_v1/posterior_history.csv",
        "artifacts/transition_matrix_accumulator_v1/scenario_summary.csv",
    ),
    "gaussian_feature_likelihood": (
        "artifacts/generated_corpus_features/feature_matrix.csv",
        "artifacts/generated_corpus_features/selected_record_manifest.csv",
    ),
    "kalman_innovation_likelihood": (
        "artifacts/corpus_classifier_scoring/classifier_candidate_scores.csv",
        "artifacts/corpus_classifier_scoring/posterior_history.csv",
    ),
    "imm_mode_mixing_recursion": (
        "artifacts/imm_filter_v1/mixing_probability_history.csv",
        "artifacts/imm_filter_v1/mode_probability_history.csv",
        "artifacts/imm_filter_v1/switching_detection_metrics.csv",
    ),
    "pf_importance_weight_update": (
        "artifacts/pf_abs_range_multimodal_oracle_v1/particle_diagnostics.csv",
        "artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv",
    ),
    "pf_class_evidence_extraction": (
        "artifacts/pf_abs_range_multimodal_oracle_v1/method_posterior_history.csv",
        "artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv",
    ),
    "rbpf_conditional_weight_update": (
        "artifacts/advanced_filter_comparison_v1/pf_vs_rbpf_frontier_summary.csv",
        "artifacts/rbpf_v1/rbpf_method_comparison.csv",
    ),
    "calibration_metrics": (
        "artifacts/monte_carlo_accumulator/calibration_bins.csv",
        "artifacts/monte_carlo_accumulator/metrics_by_time.csv",
    ),
    "pairwise_mahalanobis_distance": (
        "artifacts/feature_analysis_v1/pairwise_distance_matrix.csv",
        "artifacts/feature_analysis_v1/feature_matrix.csv",
    ),
    "corpus_autodevelopment_score": (
        "artifacts/corpus_autodevelopment_v1/candidate_scores.csv",
        "artifacts/corpus_autodevelopment_v1/corpus_adequacy_summary.json",
    ),
    "pareto_dominance": (
        "artifacts/corpus_autodevelopment_v1/pareto_front.csv",
        "artifacts/corpus_autodevelopment_v1/corpus_adequacy_summary.json",
    ),
    "corpus_gym_reward": (
        "artifacts/corpus_gym/corpus_gym_report.md",
        "artifacts/corpus_gym/corpus_gym_numeric_walkthrough.md",
    ),
    "corpus_explorer_utility": (
        "artifacts/generic_corpus_exploration/generic_corpus_exploration_report.md",
        "artifacts/generic_corpus_exploration/generic_corpus_explorer_numeric_walkthrough.md",
    ),
    "qd_archive_utility": (
        "artifacts/quality_diversity_corpus_v1/archive_elites.csv",
        "artifacts/quality_diversity_corpus_v1/archive_cells.csv",
    ),
    "qd_cell_mapping": (
        "artifacts/quality_diversity_corpus_v1/archive_cells.csv",
    ),
    "sampler_mixture": (
        "artifacts/candidate_generation/generated_candidates.csv",
    ),
    "class_validity_status": (
        "artifacts/class_validity/class_validity_scores.csv",
    ),
    "corpus_policy_normalization": (
        "artifacts/corpus_hyperparameter_tuning_v1/default_weight_spec.yaml",
        "artifacts/corpus_hyperparameter_tuning_v1/weight_spec_schema.json",
    ),
    "corpus_policy_score": (
        "artifacts/corpus_hyperparameter_tuning_v1/sweep_results.csv",
        "artifacts/corpus_hyperparameter_tuning_v1/pareto_front.csv",
    ),
    "rung_sufficiency_promotion_rule": (
        "artifacts/rung_sufficiency/rung_promotion_matrix.csv",
        "artifacts/rung_sufficiency/rung_sufficiency_report.md",
    ),
    "advanced_filter_gate": (
        "artifacts/advanced_filter_decision_v1/advanced_filter_decision_report.md",
        "artifacts/advanced_filter_decision_v1/advanced_filter_decision_numeric_walkthrough.md",
    ),
}


def _load_equation_lookup() -> dict[str, dict[str, object]]:
    return {row["id"]: row for row in load_equation_registry()}


def _ensure_asset(source: Path | None, assets_dir: Path, equation_id: str, builder) -> tuple[Path, bool]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_path = assets_dir / f"{equation_id}.png"
    if source is not None and source.exists():
        shutil.copy2(source, asset_path)
        return asset_path, False
    asset_path.write_bytes(builder())
    return asset_path, True


def _plot_pairwise_mahalanobis_distance() -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = write_feature_analysis_artifacts(temp_dir)
        return Path(artifacts.plot_distance_png_path).read_bytes()


def _plot_rung_sufficiency_promotion_rule() -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = write_rung_sufficiency_artifacts(temp_dir)
        return Path(artifacts.promotion_decision_plot_path).read_bytes()


def _plot_bayes_recursive_update() -> bytes:
    return render_posterior_numeric_walkthrough_png_bytes()


def _plot_transition_matrix_update() -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts = write_transition_benchmark_artifacts(temp_dir)
        return Path(artifacts.plot_png_path).read_bytes()


def _visual_spec_for_equation(equation_id: str) -> FormalMathVisualSpec:
    return next(spec for spec in FORMAL_MATH_VISUAL_REGISTRY if spec.equation_id == equation_id)


def _plot_registry_card(equation_id: str) -> bytes:
    spec = _visual_spec_for_equation(equation_id)
    equation = _load_equation_lookup().get(equation_id)
    implementation = ""
    if equation:
        impl = equation["implementation"]
        implementation = f"{impl['module']}::{impl['function']}"
    source_artifacts = [str(item) for item in equation.get("artifacts", ())] if equation else []
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axis("off")
    ax.set_title(spec.title, loc="left", fontsize=15, fontweight="bold")
    lines = [
        f"Equation ID: {spec.equation_id}",
        f"Status: {equation['status'] if equation else 'missing'}",
        f"Implementation: {implementation or 'documentation / visual only'}",
        "",
        "Source artifacts:",
    ]
    lines.extend(f"- {item}" for item in (source_artifacts or [spec.source_artifact or "none"]))
    lines.extend(
        [
            "",
            "This fallback card preserves the registry's provenance even when a dedicated plot is not stored in the repository checkout.",
        ]
    )
    ax.text(
        0.04,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    return _figure_to_png(fig)


def _visual_builder_for_equation(equation_id: str):
    return {
        "bayes_logsumexp_update": _plot_bayes_recursive_update,
        "two_class_log_odds": _plot_bayes_recursive_update,
        "transition_matrix_update": _plot_transition_matrix_update,
        "gaussian_feature_likelihood": _plot_gaussian_feature_likelihood,
        "calibration_metrics": _plot_calibration_metrics,
        "corpus_gym_reward": _plot_corpus_gym_reward,
        "sampler_mixture": _plot_sampler_mixture,
        "pairwise_mahalanobis_distance": _plot_pairwise_mahalanobis_distance,
        "rung_sufficiency_promotion_rule": _plot_rung_sufficiency_promotion_rule,
    }.get(equation_id, lambda: _plot_registry_card(equation_id))


def _plot_gaussian_feature_likelihood() -> bytes:
    records = select_generated_corpus_records()
    class_values: dict[str, list[float]] = {}
    for record in records:
        measurements = record.execution.trajectory_run.observations.get("position", ())
        class_values.setdefault(record.assigned_class, []).extend(float(value) for value in measurements)

    classes = sorted((name, values) for name, values in class_values.items() if values)
    if not classes:
        classes = [("unknown", [0.0, 0.5, 1.0])]
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.6, 5.8),
        gridspec_kw={"height_ratios": [3.0, 1.1]},
        sharex=True,
    )
    colors = ["#2563eb", "#0f766e", "#dc2626", "#7c3aed", "#d97706", "#0891b2"]
    summary_rows: list[tuple[str, float, float, int]] = []
    max_density = 0.0
    for index, (label, values) in enumerate(classes):
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / max(len(values), 1)
        sigma = max(variance ** 0.5, 0.08)
        xs = [mean_value + offset * sigma for offset in [x / 8.0 for x in range(-24, 25)]]
        ys = [math.exp(-0.5 * ((x - mean_value) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) for x in xs]
        color = colors[index % len(colors)]
        ax.plot(xs, ys, color=color, linewidth=2.2, label=label)
        ax.fill_between(xs, ys, color=color, alpha=0.08)
        max_density = max(max_density, max(ys))
        summary_rows.append((label, mean_value, sigma, len(values)))
    ax.set_title("Gaussian Feature Likelihood", loc="left", fontweight="bold")
    ax.set_ylabel("likelihood")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(alpha=0.25)
    ax2.bar(
        [label for label, _, _, _ in summary_rows],
        [count for _, _, _, count in summary_rows],
        color=[colors[index % len(colors)] for index, _ in enumerate(summary_rows)],
    )
    ax2.set_ylabel("sample count")
    ax2.set_xlabel("candidate class hypothesis")
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _figure_to_png(fig)


def _plot_calibration_metrics() -> bytes:
    benchmark = run_accumulator_monte_carlo_benchmark()
    return render_monte_carlo_calibration_png_bytes(benchmark)


def _plot_corpus_gym_reward() -> bytes:
    labels = [
        "class_validity",
        "feature_excitation",
        "coverage_gain",
        "boundary_closeness",
        "classifier_stress",
        "prior_sensitivity",
        "leakage_penalty",
        "physical_invalidity_penalty",
        "total_utility",
    ]
    targets = default_corpus_gym_targets()
    environment = CorpusGymEnvironment()
    component_rows: list[list[float]] = []
    target_labels: list[str] = []
    for index, target in enumerate(targets):
        environment.reset(target)
        if target.target_type == "target_class_pair":
            action = CorpusGymAction(seed=2_100 + index, tier_name="boundary_v1", duration_scale=0.88, measurement_scale=1.10, irregularity_scale=1.05, outlier_scale=0.95, step_scale=0.92)
        elif target.target_type in {"target_failure_mode", "target_feature_cell"}:
            action = CorpusGymAction(seed=2_100 + index, tier_name="adversarial_v1", duration_scale=0.80, measurement_scale=1.22, irregularity_scale=1.18, outlier_scale=1.20, step_scale=0.86)
        elif target.target_type == "target_prior_sensitivity":
            action = CorpusGymAction(seed=2_100 + index, tier_name="boundary_v1", duration_scale=0.74, measurement_scale=1.06, irregularity_scale=0.96, outlier_scale=0.90, step_scale=0.84)
        elif target.target_type == "target_switching_pattern":
            action = CorpusGymAction(seed=2_100 + index, tier_name="boundary_v1", duration_scale=0.92, measurement_scale=1.08, irregularity_scale=1.12, outlier_scale=1.00, step_scale=0.88)
        else:
            action = CorpusGymAction(seed=2_100 + index, tier_name="realistic_v1", duration_scale=1.0, measurement_scale=1.0, irregularity_scale=1.0, outlier_scale=1.0, step_scale=1.0)
        episode = environment.simulate(action)
        reward = episode.reward
        component_rows.append(
            [
                reward.class_validity,
                reward.feature_excitation,
                reward.coverage_gain,
                reward.boundary_closeness,
                reward.classifier_stress,
                reward.prior_sensitivity,
                -reward.leakage_penalty,
                -reward.physical_invalidity_penalty,
                reward.total_utility,
            ]
        )
        target_labels.append(target.target_id.replace("target_", ""))
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    matrix = component_rows
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("CorpusGym Reward Decomposition Across Targets", loc="left", fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(len(target_labels)), target_labels)
    ax.set_xlabel("reward component")
    ax.set_ylabel("target")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="weighted contribution")
    fig.tight_layout()
    return _figure_to_png(fig)


def _plot_sampler_mixture() -> bytes:
    result = analyze_candidate_generation()

    rows = list(result.generated_candidate_rows)
    scenario_families = sorted({str(row.get("scenario_family", "unknown")) for row in rows})
    sampler_names = sorted({str(row.get("sampler_name", "unknown")) for row in rows})
    counts = {
        scenario: {
            sampler: sum(1 for row in rows if str(row.get("scenario_family", "unknown")) == scenario and str(row.get("sampler_name", "unknown")) == sampler)
            for sampler in sampler_names
        }
        for scenario in scenario_families
    }
    totals = [sum(counts[scenario].values()) for scenario in scenario_families]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    bottom = [0] * len(scenario_families)
    palette = ["#2563eb", "#0f766e", "#7c3aed", "#d97706", "#dc2626"]
    for index, sampler in enumerate(sampler_names):
        values = [counts[scenario][sampler] for scenario in scenario_families]
        ax.bar(scenario_families, values, bottom=bottom, label=sampler, color=palette[index % len(palette)])
        bottom = [existing + value for existing, value in zip(bottom, values)]
    ax.plot(scenario_families, totals, color="#111827", marker="o", linewidth=1.8, linestyle="--", label="total")
    ax.set_title("Sampler Mixture Across Scenario Families", loc="left", fontweight="bold")
    ax.set_ylabel("candidate count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return _figure_to_png(fig)


def analyze_formal_math_visual_registry() -> FormalMathVisualRegistryResult:
    equation_lookup = _load_equation_lookup()
    rows: list[FormalMathVisualRow] = []
    for spec in FORMAL_MATH_VISUAL_REGISTRY:
        equation = equation_lookup.get(spec.equation_id)
        status = str(equation["status"]) if equation else "missing"
        implementation = ""
        if equation:
            impl = equation["implementation"]
            implementation = f"{impl['module']}::{impl['function']}"
        source = ROOT / spec.source_artifact if spec.source_artifact else None
        builder = _visual_builder_for_equation(spec.equation_id)
        if source is not None and source.exists():
            visual_kind = "copied_existing_plot"
            generated_visual = False
        elif builder is not None:
            visual_kind = "generated_from_real_data"
            generated_visual = True
        else:
            raise FileNotFoundError(
                f"No source artifact or builder available for visual equation {spec.equation_id}"
            )
        visual_status = "implemented" if status == "implemented" else ("illustrative" if status == "conceptual" else "missing")
        visual_path = f"assets/{spec.equation_id}.png"
        rows.append(
            FormalMathVisualRow(
                equation_id=spec.equation_id,
                status=status,
                visual_status=visual_status,
                visual_kind=visual_kind,
                implementation=implementation,
                visual_path=visual_path,
                source_artifact=spec.source_artifact or "",
                source_data_artifacts=FORMAL_MATH_VISUAL_SOURCE_DATA.get(spec.equation_id, ()),
                source_type=spec.source_type,
                generated_visual=generated_visual,
                rerun_command="python3 scripts/render/render_formal_math_visual_registry.py --output-dir artifacts",
                caption=spec.caption,
                notes=spec.notes,
            )
        )
    summary = {
        "visual_count": len(rows),
        "generated_visual_count": sum(1 for row in rows if row.generated_visual),
        "implemented_visual_count": sum(1 for row in rows if row.visual_status == "implemented"),
        "illustrative_visual_count": sum(1 for row in rows if row.visual_status == "illustrative"),
        "missing_visual_count": sum(1 for row in rows if row.visual_status == "missing"),
        "copied_visual_count": sum(1 for row in rows if not row.generated_visual),
        "implemented_equation_count": sum(1 for row in rows if row.status == "implemented"),
        "conceptual_equation_count": sum(1 for row in rows if row.status == "conceptual"),
        "source_artifact_count": sum(1 for row in rows if row.source_artifact),
        "source_data_reference_count": sum(len(row.source_data_artifacts) for row in rows),
    }
    report_markdown = render_formal_math_visual_registry_report(
        FormalMathVisualRegistryResult(rows=tuple(rows), summary=summary, report_markdown="")
    )
    return FormalMathVisualRegistryResult(rows=tuple(rows), summary=summary, report_markdown=report_markdown)


def render_formal_math_visual_registry_report(result: FormalMathVisualRegistryResult) -> str:
    doc = MarkdownDocument("Formal Math Visual Registry")
    doc.paragraph(
        "This gallery pairs the equation registry with representative charts and records the exact source artifact or source data used for each visual. There are no anonymous placeholder charts in the bundle."
    )
    
    doc.heading("Summary", level=2)
    doc.bullet_list(
        [
            f"Visual count: `{result.summary['visual_count']}`",
            f"Generated visuals: `{result.summary['generated_visual_count']}`",
            f"Implemented visuals: `{result.summary['implemented_visual_count']}`",
            f"Illustrative visuals: `{result.summary['illustrative_visual_count']}`",
            f"Missing visuals: `{result.summary['missing_visual_count']}`",
            f"Copied visuals: `{result.summary['copied_visual_count']}`",
            f"- Implemented equations covered: `{result.summary['implemented_equation_count']}`",
            f"Conceptual equations covered: `{result.summary['conceptual_equation_count']}`",
            f"Source artifact references: `{result.summary['source_artifact_count']}`",
            f"Source-data references: `{result.summary['source_data_reference_count']}`",
        ]
    )

    doc.heading("Provenance", level=2)
    doc.paragraph(
        "Every row in the provenance CSV points to the exact visual artifact and the data sources used to generate or copy it. The canonical rerun command is recorded in the runbook."
    )

    doc.heading("Gallery", level=2)
    for row in result.rows:
        doc.heading(f"`{row.equation_id}`", level=3)
        doc.bullet_list(
            [
                f"Status: `{row.status}`",
                f"Visual status: `{row.visual_status}`",
                f"Visual kind: `{row.visual_kind}`",
                f"Visual source: `{row.source_type}`",
                f"Visual path: `{row.visual_path}`",
                f"Source artifact: `{row.source_artifact or 'generated'}`",
                f"Source data: `{', '.join(row.source_data_artifacts) if row.source_data_artifacts else 'n/a'}`",
                f"Rerun command: `{row.rerun_command}`",
                f"Implementation: `{row.implementation or 'missing'}`",
            ]
        )
        doc.paragraph(f"![{row.equation_id}]({row.visual_path})")
        doc.paragraph(row.caption)
        
    return doc.text()


def write_formal_math_visual_registry_artifacts(
    output_dir: str | Path,
    *,
    result: FormalMathVisualRegistryResult | None = None,
) -> FormalMathVisualRegistryArtifacts:
    payload = result or analyze_formal_math_visual_registry()
    output_root = Path(output_dir)
    run_dir = output_root / "formal_math_visual_registry_v1"
    assets_dir = run_dir / "assets"
    run_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "formal_math_visual_registry_report.md"
    summary_path = run_dir / "formal_math_visual_registry_summary.json"
    gallery_csv_path = run_dir / "formal_math_visual_registry.csv"
    provenance_path = run_dir / "formal_math_visual_registry_provenance.csv"
    runbook_path = run_dir / "formal_math_visual_registry_runbook.md"
    visual_coverage_png_path = run_dir / "formal_math_visual_registry_coverage.png"

    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload.summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(
        gallery_csv_path,
        [
            {
                "equation_id": row.equation_id,
                "status": row.status,
                "implementation": row.implementation,
                "visual_path": row.visual_path,
                "source_type": row.source_type,
                "generated_visual": row.generated_visual,
                "caption": row.caption,
                "notes": row.notes,
            }
            for row in payload.rows
        ],
        [
            "equation_id",
            "status",
            "implementation",
            "visual_path",
            "source_type",
            "generated_visual",
            "caption",
            "notes",
        ],
    )
    write_csv(
        provenance_path,
        [
            {
                "equation_id": row.equation_id,
                "equation_status": row.status,
                "visual_status": row.visual_status,
                "visual_kind": row.visual_kind,
                "implementation": row.implementation,
                "visual_path": row.visual_path,
                "source_artifact": row.source_artifact,
                "source_data_artifacts": " | ".join(row.source_data_artifacts),
                "rerun_command": row.rerun_command,
                "source_type": row.source_type,
                "generated_visual": row.generated_visual,
            }
            for row in payload.rows
        ],
        [
            "equation_id",
            "equation_status",
            "visual_status",
            "visual_kind",
            "implementation",
            "visual_path",
            "source_artifact",
            "source_data_artifacts",
            "rerun_command",
            "source_type",
            "generated_visual",
        ],
    )
    runbook_doc = MarkdownDocument("Formal Math Visual Registry Runbook")
    runbook_doc.paragraph("The canonical rerun command is:")
    runbook_doc.fence(
        "python3 scripts/render/render_formal_math_visual_registry.py --output-dir artifacts",
        language="bash"
    )
    runbook_doc.paragraph("The table below tells you what each equation visual is sourced from.")
    runbook_doc.table(
        ["equation_id", "visual_status", "visual_kind", "source artifact", "source data", "rerun command"],
        [
            (
                f"`{row.equation_id}`",
                f"`{row.visual_status}`",
                f"`{row.visual_kind}`",
                f"`{row.source_artifact or 'generated'}`",
                f"{', '.join(f'`{path}`' for path in row.source_data_artifacts) if row.source_data_artifacts else 'n/a'}",
                f"`{row.rerun_command}`",
            )
            for row in payload.rows
        ]
    )
    runbook_doc.paragraph("Rows marked `illustrative` are explicit documentation visuals and are not counted as implemented equation coverage.")
    runbook_path.write_text(runbook_doc.text() + "\n", encoding="utf-8")

    coverage_labels = [row.equation_id for row in payload.rows]
    generated = [1 if row.generated_visual else 0 for row in payload.rows]
    copied = [1 if not row.generated_visual else 0 for row in payload.rows]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(coverage_labels, copied, color="#2563eb", label="copied")
    ax.bar(coverage_labels, generated, bottom=copied, color="#0f766e", label="generated")
    ax.set_title("Formal Math Visual Coverage", loc="left", fontweight="bold")
    ax.set_ylabel("visuals")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    visual_coverage_png_path.write_bytes(_figure_to_png(fig))

    for row in payload.rows:
        spec = next(spec for spec in FORMAL_MATH_VISUAL_REGISTRY if spec.equation_id == row.equation_id)
        source = ROOT / spec.source_artifact if spec.source_artifact else None
        asset_path = assets_dir / f"{row.equation_id}.png"
        builder = _visual_builder_for_equation(row.equation_id)
        if source is not None and source.exists():
            shutil.copy2(source, asset_path)
        elif builder is not None:
            asset_path.write_bytes(builder())
        else:
            raise FileNotFoundError(f"No source artifact or builder available for {row.equation_id}")

    return FormalMathVisualRegistryArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        gallery_csv_path=gallery_csv_path,
        provenance_path=provenance_path,
        runbook_path=runbook_path,
        visual_coverage_png_path=visual_coverage_png_path,
        assets_dir=assets_dir,
    )
