from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt
from pathlib import Path
from typing import NamedTuple

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ...analysis.feature_analysis import (
    load_feature_registry,
    load_feature_set_manifest,
    resolve_feature_names,
)
from ...common_experiment.config import load_common_experiment_config, resolve_common_study_adapter
from ...common_experiment.contracts import CommonExperimentResult
from ...common_experiment.pair_evaluation import (
    _gaussian_logpdf,
    _normalize_scores,
)
from ...common_experiment.pair_evaluation import (
    classifier_scores_for_prefix as _classifier_scores_for_prefix,
)
from ...common_experiment.pair_evaluation import (
    feature_set_scores_for_prefix as _feature_set_scores_for_prefix,
)
from ...common_experiment.pair_evaluation import (
    feature_sigma as _feature_sigma,
)
from ...common_experiment.pair_evaluation import (
    pair_priors as _pair_priors,
)
from ...common_experiment.pair_evaluation import (
    reference_trajectory as _reference_trajectory,
)
from ...common_experiment.pair_evaluation import (
    trajectory_features as _trajectory_features,
)
from ...common_experiment.runner import analyze_common_experiment
from ...corpus.coverage_report import load_classifier_manifest
from ...utils.io import write_csv
from ...utils.plotting import plt, write_plot


@dataclass(frozen=True, slots=True)
class BayesianWalkthroughResult:
    selected_walkthrough: dict[str, object]
    feature_example: dict[str, object]
    bayesian_step_rows: tuple[dict[str, object], ...]
    prior_sweep_rows: tuple[dict[str, object], ...]
    feature_contribution_rows: tuple[dict[str, object], ...]
    posterior_flip_threshold_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class BayesianWalkthroughArtifacts:
    run_dir: Path
    report_path: Path
    bayesian_step_tables_path: Path
    prior_sweep_examples_path: Path
    prior_sweep_examples_png_path: Path
    feature_contribution_examples_path: Path
    posterior_flip_thresholds_path: Path
    plots_dir: Path
    prior_to_posterior_single_step_path: Path
    likelihood_curves_with_feature_value_path: Path
    posterior_timeline_path: Path
    log_odds_timeline_path: Path
    bayes_factor_timeline_path: Path
    prior_sensitivity_curve_path: Path
    feature_ablation_posterior_path: Path
    confidence_threshold_crossing_path: Path


class PriorSweepRows(NamedTuple):
    rows: tuple[dict[str, object], ...]
    threshold_rows: tuple[dict[str, object], ...]


class FeatureContributionRows(NamedTuple):
    rows: tuple[dict[str, object], ...]
    feature_example: dict[str, object]


class PreferenceScore(NamedTuple):
    tier_score: float
    confidence_score: float
    margin_delta: float


def _gaussian_pdf(value: float, mean: float, sigma: float) -> float:
    sigma = max(sigma, 1e-9)
    coefficient = 1.0 / (sigma * sqrt(2.0 * pi))
    exponent = -0.5 * ((value - mean) / sigma) ** 2
    return coefficient * exp(exponent)


def _log_odds(prob_a: float, prob_b: float) -> float:
    return log(max(prob_a, 1e-12) / max(prob_b, 1e-12))


def _select_representative_run(
    pair_prediction_rows: tuple[dict[str, object], ...],
    posterior_history_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    preferred_rows = [
        row
        for row in pair_prediction_rows
        if str(row["classifier_id"]) == "bayes_accumulator"
        and str(row["class_pair_id"]) == "constant_velocity_vs_constant_acceleration"
        and str(row["predicted_class"]) == str(row["true_class"])
    ]
    if not preferred_rows:
        preferred_rows = [
            row
            for row in pair_prediction_rows
            if str(row["classifier_id"]) == "bayes_accumulator" and str(row["predicted_class"]) == str(row["true_class"])
        ]
    posterior_lookup: dict[str, list[dict[str, object]]] = {}
    for row in posterior_history_rows:
        posterior_lookup.setdefault(str(row["run_id"]), []).append(dict(row))

    tier_preference = {
        "stress_v1": 5.0,
        "boundary_v1": 4.0,
        "realistic_v1": 3.0,
        "adversarial_v1": 2.0,
        "easy_v1": 1.0,
    }

    def score(row: dict[str, object]) -> PreferenceScore:
        history = sorted(posterior_lookup[str(row["run_id"])], key=lambda item: float(item["time"]))
        first_margin = abs(float(history[0]["posterior_class_a"]) - float(history[0]["posterior_class_b"]))
        final_margin = abs(float(history[-1]["posterior_class_a"]) - float(history[-1]["posterior_class_b"]))
        confidence = float(row["confidence"])
        return PreferenceScore(
            tier_score=tier_preference.get(str(row["dataset_tier"]), 0.0),
            confidence_score=1.0 - abs(confidence - 0.72),
            margin_delta=final_margin - first_margin,
        )

    return max(preferred_rows, key=score)


def _build_bayesian_step_rows(
    *,
    selected_run: dict[str, object],
    posterior_rows: tuple[dict[str, object], ...],
    likelihood_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    run_id = str(selected_run["run_id"])
    class_a = str(selected_run["class_a"])
    class_b = str(selected_run["class_b"])
    selected_posteriors = sorted(
        (dict(row) for row in posterior_rows if str(row["run_id"]) == run_id),
        key=lambda row: float(row["time"]),
    )
    selected_likelihoods = sorted(
        (dict(row) for row in likelihood_rows if str(row["run_id"]) == run_id),
        key=lambda row: float(row["time"]),
    )
    rows: list[dict[str, object]] = [
        {
            "example_id": "hand_checkable_binary_demo",
            "example_type": "hand_checkable_demo",
            "study_id": "hand_checkable_binary_demo",
            "trajectory_id": "n/a",
            "class_pair_id": "A_vs_B",
            "classifier_id": "generic_bayes_demo",
            "step_index": 1,
            "time": 0.0,
            "class_a": "A",
            "class_b": "B",
            "prior_a": 0.5,
            "prior_b": 0.5,
            "log_prior_odds": 0.0,
            "cumulative_log_score_a": log(0.8),
            "cumulative_log_score_b": log(0.2),
            "incremental_log_evidence_a": log(0.8),
            "incremental_log_evidence_b": log(0.2),
            "log_bayes_factor_ab": log(0.8 / 0.2),
            "posterior_a": 0.8,
            "posterior_b": 0.2,
            "log_posterior_odds": log(0.8 / 0.2),
            "predicted_class": "A",
            "true_class": "A",
        }
    ]

    previous_delta: float | None = None
    previous_cumulative_a: float | None = None
    previous_cumulative_b: float | None = None
    previous_posterior_a = 0.5
    previous_posterior_b = 0.5
    for step_index, (posterior_row, likelihood_row) in enumerate(zip(selected_posteriors, selected_likelihoods), start=1):
        cumulative_a = float(likelihood_row["log_likelihood_class_a"])
        cumulative_b = float(likelihood_row["log_likelihood_class_b"])
        delta = cumulative_a - cumulative_b
        if previous_delta is None or previous_cumulative_a is None or previous_cumulative_b is None:
            incremental_a = cumulative_a
            incremental_b = cumulative_b
            log_bayes_factor_ab = delta
        else:
            incremental_a = cumulative_a - previous_cumulative_a
            incremental_b = cumulative_b - previous_cumulative_b
            log_bayes_factor_ab = delta - previous_delta
        posterior_a = float(posterior_row["posterior_class_a"])
        posterior_b = float(posterior_row["posterior_class_b"])
        rows.append(
            {
                "example_id": "representative_common_study",
                "example_type": "trajectory_walkthrough",
                "study_id": f"{selected_run['classifier_id']}_{selected_run['class_pair_id']}",
                "trajectory_id": selected_run["trajectory_id"],
                "class_pair_id": selected_run["class_pair_id"],
                "classifier_id": selected_run["classifier_id"],
                "step_index": step_index,
                "time": posterior_row["time"],
                "class_a": class_a,
                "class_b": class_b,
                "prior_a": previous_posterior_a,
                "prior_b": previous_posterior_b,
                "log_prior_odds": _log_odds(previous_posterior_a, previous_posterior_b),
                "cumulative_log_score_a": cumulative_a,
                "cumulative_log_score_b": cumulative_b,
                "incremental_log_evidence_a": incremental_a,
                "incremental_log_evidence_b": incremental_b,
                "log_bayes_factor_ab": log_bayes_factor_ab,
                "posterior_a": posterior_a,
                "posterior_b": posterior_b,
                "log_posterior_odds": _log_odds(posterior_a, posterior_b),
                "predicted_class": class_a if posterior_a >= posterior_b else class_b,
                "true_class": selected_run["true_class"],
            }
        )
        previous_delta = delta
        previous_cumulative_a = cumulative_a
        previous_cumulative_b = cumulative_b
        previous_posterior_a = posterior_a
        previous_posterior_b = posterior_b
    return tuple(rows)


def _trajectory_context(seed: int, trajectories_per_case: int):
    config = load_common_experiment_config()
    adapter = resolve_common_study_adapter(config)
    pair_specs = adapter.pair_spec_builder(config)
    trajectories = adapter.trajectory_generator(pair_specs, seed, trajectories_per_case)
    pair_lookup = {spec.pair_id: spec for spec in pair_specs}
    trajectory_lookup = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    classifier_lookup = {str(entry["id"]): entry for entry in load_classifier_manifest(config.classifier_manifest_path)}
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    feature_registry = load_feature_registry()
    return config, pair_lookup, trajectory_lookup, classifier_lookup, feature_manifest, feature_registry


def _build_prior_sweep_rows(
    *,
    selected_run: dict[str, object],
    pair_lookup: dict[str, object],
    trajectory_lookup: dict[str, object],
    classifier_lookup: dict[str, dict[str, object]],
    feature_manifest: dict[str, dict[str, object]],
) -> PriorSweepRows:
    pair_spec = pair_lookup[str(selected_run["class_pair_id"])]
    trajectory = trajectory_lookup[str(selected_run["trajectory_id"])]
    classifier_entry = classifier_lookup[str(selected_run["classifier_id"])]
    class_a = str(selected_run["class_a"])
    class_b = str(selected_run["class_b"])
    prior_grid = [round(index / 20.0, 2) for index in range(1, 20)]
    rows: list[dict[str, object]] = []
    threshold_prior_a_for_a = None
    threshold_prior_a_for_b = None

    uniform_prior = {class_a: 0.5, class_b: 0.5}
    uniform_scores = _classifier_scores_for_prefix(
        classifier_entry,
        pair_spec,
        trajectory,
        len(trajectory.times),
        uniform_prior,
        feature_manifest,
    )
    uniform_llr = (uniform_scores[class_a] - uniform_scores[class_b]) - (_log_odds(0.5, 0.5))
    if uniform_llr >= 60.0:
        theoretical_threshold_prior_a = 0.0
    elif uniform_llr <= -60.0:
        theoretical_threshold_prior_a = 1.0
    else:
        theoretical_threshold_prior_a = 1.0 / (1.0 + exp(uniform_llr))

    for prior_a in prior_grid:
        prior = {class_a: prior_a, class_b: 1.0 - prior_a}
        scores = _classifier_scores_for_prefix(
            classifier_entry,
            pair_spec,
            trajectory,
            len(trajectory.times),
            prior,
            feature_manifest,
        )
        weights = _normalize_scores(scores)
        predicted_class = class_a if weights[class_a] >= weights[class_b] else class_b
        if predicted_class == class_a and threshold_prior_a_for_a is None:
            threshold_prior_a_for_a = prior_a
        if predicted_class == class_b:
            threshold_prior_a_for_b = prior_a
        rows.append(
            {
                "study_id": f"{selected_run['classifier_id']}_{selected_run['class_pair_id']}",
                "trajectory_id": selected_run["trajectory_id"],
                "class_pair_id": selected_run["class_pair_id"],
                "classifier_id": selected_run["classifier_id"],
                "class_a": class_a,
                "class_b": class_b,
                "prior_a": prior_a,
                "prior_b": 1.0 - prior_a,
                "log_prior_odds": _log_odds(prior_a, 1.0 - prior_a),
                "posterior_a": weights[class_a],
                "posterior_b": weights[class_b],
                "final_log_posterior_odds": _log_odds(weights[class_a], weights[class_b]),
                "predicted_class": predicted_class,
                "confidence": max(weights[class_a], weights[class_b]),
                "cumulative_log_likelihood_ratio": scores[class_a] - scores[class_b] - _log_odds(prior_a, 1.0 - prior_a),
            }
        )

    threshold_rows = [
        {
            "study_id": f"{selected_run['classifier_id']}_{selected_run['class_pair_id']}",
            "trajectory_id": selected_run["trajectory_id"],
            "class_pair_id": selected_run["class_pair_id"],
            "classifier_id": selected_run["classifier_id"],
            "class_a": class_a,
            "class_b": class_b,
            "uniform_predicted_class": selected_run["predicted_class"],
            "uniform_confidence": selected_run["confidence"],
            "empirical_min_prior_a_for_class_a": threshold_prior_a_for_a,
            "empirical_max_prior_a_for_class_b": threshold_prior_a_for_b,
            "theoretical_flip_threshold_prior_a": theoretical_threshold_prior_a,
        }
    ]
    return PriorSweepRows(tuple(rows), tuple(threshold_rows))


def _select_feature_example_run(
    pair_prediction_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    preferred = [
        row
        for row in pair_prediction_rows
        if str(row["classifier_id"]) == "pointwise"
        and str(row["class_pair_id"]) == "stationary_vs_constant_velocity"
        and str(row["predicted_class"]) == str(row["true_class"])
    ]
    if not preferred:
        preferred = [
            row
            for row in pair_prediction_rows
            if str(row["classifier_id"]) == "pointwise" and str(row["predicted_class"]) == str(row["true_class"])
        ]
    tier_preference = {
        "stress_v1": 5.0,
        "boundary_v1": 4.0,
        "realistic_v1": 3.0,
        "adversarial_v1": 2.0,
        "easy_v1": 1.0,
    }
    return max(
        preferred,
        key=lambda row: (
            tier_preference.get(str(row["dataset_tier"]), 0.0),
            1.0 - abs(float(row["confidence"]) - 0.78),
        ),
    )


def _build_feature_contribution_rows(
    *,
    feature_run: dict[str, object],
    pair_lookup: dict[str, object],
    trajectory_lookup: dict[str, object],
    feature_manifest: dict[str, dict[str, object]],
    feature_registry: dict[str, object],
) -> FeatureContributionRows:
    pair_spec = pair_lookup[str(feature_run["class_pair_id"])]
    trajectory = trajectory_lookup[str(feature_run["trajectory_id"])]
    feature_set_id = str(feature_run["feature_set_id"])
    feature_names = resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
    prior = _pair_priors(pair_spec.class_a, pair_spec.class_b, "uniform")
    robust = feature_set_id == "robust_extrema"
    observed = _trajectory_features(trajectory, robust=robust)
    reference_a = _trajectory_features(_reference_trajectory(pair_spec, pair_spec.class_a, trajectory.scenario_id, trajectory.times), robust=robust)
    reference_b = _trajectory_features(_reference_trajectory(pair_spec, pair_spec.class_b, trajectory.scenario_id, trajectory.times), robust=robust)

    full_scores = _feature_set_scores_for_prefix(
        feature_set_id=feature_set_id,
        feature_entry=feature_manifest[feature_set_id],
        pair_spec=pair_spec,
        trajectory=trajectory,
        prefix_length=len(trajectory.times),
        prior_weights=prior,
    )
    full_weights = _normalize_scores(full_scores)
    true_class = str(feature_run["true_class"])

    rows: list[dict[str, object]] = []
    for feature_name in feature_names:
        sigma = _feature_sigma(feature_name)
        log_a = _gaussian_logpdf(observed[feature_name], reference_a[feature_name], sigma)
        log_b = _gaussian_logpdf(observed[feature_name], reference_b[feature_name], sigma)
        ablated_scores = {
            pair_spec.class_a: full_scores[pair_spec.class_a] - log_a,
            pair_spec.class_b: full_scores[pair_spec.class_b] - log_b,
        }
        ablated_weights = _normalize_scores(ablated_scores)
        spec = feature_registry[feature_name]
        rows.append(
            {
                "study_id": f"pointwise_{feature_run['class_pair_id']}_{feature_set_id}",
                "trajectory_id": feature_run["trajectory_id"],
                "class_pair_id": feature_run["class_pair_id"],
                "classifier_id": feature_run["classifier_id"],
                "feature_set_id": feature_set_id,
                "feature_name": feature_name,
                "feature_group": spec.group,
                "history_behavior": spec.history_behavior,
                "evidence_role": spec.role,
                "double_counting_risk": "high" if spec.history_behavior == "cumulative" else "low",
                "observed_value": observed[feature_name],
                "class_a_reference": reference_a[feature_name],
                "class_b_reference": reference_b[feature_name],
                "feature_sigma": sigma,
                "log_likelihood_class_a": log_a,
                "log_likelihood_class_b": log_b,
                "log_likelihood_ratio_ab": log_a - log_b,
                "posterior_true_with_full_set": full_weights[true_class],
                "posterior_true_without_feature": ablated_weights[true_class],
            }
        )
    top_feature = max(rows, key=lambda row: abs(float(row["log_likelihood_ratio_ab"])))
    feature_example = {
        "study_id": f"pointwise_{feature_run['class_pair_id']}_{feature_set_id}",
        "trajectory_id": feature_run["trajectory_id"],
        "class_pair_id": feature_run["class_pair_id"],
        "classifier_id": feature_run["classifier_id"],
        "feature_set_id": feature_set_id,
        "true_class": true_class,
        "top_feature": top_feature["feature_name"],
        "top_feature_observed_value": top_feature["observed_value"],
        "full_posterior_true": full_weights[true_class],
    }
    return FeatureContributionRows(tuple(rows), feature_example)


def _render_report(
    *,
    selected_walkthrough: dict[str, object],
    feature_example: dict[str, object],
    step_rows: tuple[dict[str, object], ...],
    threshold_rows: tuple[dict[str, object], ...],
) -> str:
    trajectory_steps = [row for row in step_rows if str(row["example_type"]) == "trajectory_walkthrough"]
    final_row = trajectory_steps[-1]
    threshold_row = threshold_rows[0]
    report = MarkdownDocument("Bayesian Evidence Walkthroughs")
    report.paragraph(
        "This artifact grounds Bayesian posterior mechanics in the current common-study outputs rather than in a disconnected "
        "toy-only explainer."
    )
    report.heading("Representative Sequential Walkthrough", level=2)
    study_label = f"{selected_walkthrough['classifier_id']}_{selected_walkthrough['class_pair_id']}"
    confidence_text = f"{float(selected_walkthrough['confidence']):.3f}"
    odds_label = f"{final_row['class_a']}/{final_row['class_b']}"
    odds_value = f"{float(final_row['log_posterior_odds']):.3f}"
    report.bullet_list(
        [
            f"Study: {report.inline_code(study_label)}",
            f"Trajectory: {report.inline_code(selected_walkthrough['trajectory_id'])}",
            f"True class: {report.inline_code(selected_walkthrough['true_class'])}",
            f"Final prediction: {report.inline_code(selected_walkthrough['predicted_class'])} with confidence "
            f"{report.inline_code(confidence_text)}",
            f"Final log posterior odds {report.inline_code(odds_label)}: "
            f"{report.inline_code(odds_value)}",
        ]
    )
    report.heading("Prior Sensitivity", level=2)
    empirical_min_prior = f"{float(threshold_row['empirical_min_prior_a_for_class_a']):.3f}"
    theoretical_flip_threshold = f"{float(threshold_row['theoretical_flip_threshold_prior_a']):.3f}"
    feature_true_posterior = f"{float(feature_example['full_posterior_true']):.3f}"
    report.bullet_list(
        [
            f"Empirical minimum prior on {report.inline_code(threshold_row['class_a'])} that keeps the final decision on "
            f"{report.inline_code(threshold_row['class_a'])}: {report.inline_code(empirical_min_prior)}",
            f"Theoretical flip threshold for {report.inline_code(threshold_row['class_a'])} from the final likelihood ratio: "
            f"{report.inline_code(theoretical_flip_threshold)}",
        ]
    )
    report.heading("Feature Evidence Example", level=2)
    report.bullet_list(
        [
            f"Study: {report.inline_code(feature_example['study_id'])}",
            f"Trajectory: {report.inline_code(feature_example['trajectory_id'])}",
            f"Dominant feature contribution: {report.inline_code(feature_example['top_feature'])}",
            f"True-class posterior under feature-only scoring: {report.inline_code(feature_true_posterior)}",
        ]
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "`bayesian_step_tables.csv` includes both a hand-checkable binary demo and the real common-study trajectory walkthrough.",
            "`prior_sweep_examples.csv` shows how the same trajectory responds to prior odds changes without changing the evidence stream.",
            "`feature_contribution_examples.csv` uses per-feature Gaussian evidence differences and ablations to show why a simple "
            "pointwise study moved the way it did.",
        ]
    )
    return report.text()


def analyze_bayesian_walkthroughs(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
    common_result: CommonExperimentResult | None = None,
) -> BayesianWalkthroughResult:
    common = common_result or analyze_common_experiment(seed=seed, trajectories_per_case=trajectories_per_case)
    _, pair_lookup, trajectory_lookup, classifier_lookup, feature_manifest, feature_registry = _trajectory_context(
        seed,
        trajectories_per_case,
    )

    selected_walkthrough = dict(_select_representative_run(common.pair_prediction_rows, common.posterior_history_rows))
    bayesian_step_rows = _build_bayesian_step_rows(
        selected_run=selected_walkthrough,
        posterior_rows=common.posterior_history_rows,
        likelihood_rows=common.likelihood_history_rows,
    )
    prior_sweep_bundle = _build_prior_sweep_rows(
        selected_run=selected_walkthrough,
        pair_lookup=pair_lookup,
        trajectory_lookup=trajectory_lookup,
        classifier_lookup=classifier_lookup,
        feature_manifest=feature_manifest,
    )
    prior_sweep_rows = prior_sweep_bundle.rows
    posterior_flip_threshold_rows = prior_sweep_bundle.threshold_rows

    feature_run = dict(_select_feature_example_run(common.pair_prediction_rows))
    feature_contribution_bundle = _build_feature_contribution_rows(
        feature_run=feature_run,
        pair_lookup=pair_lookup,
        trajectory_lookup=trajectory_lookup,
        feature_manifest=feature_manifest,
        feature_registry=feature_registry,
    )
    feature_contribution_rows = feature_contribution_bundle.rows
    feature_example = feature_contribution_bundle.feature_example
    report_markdown = _render_report(
        selected_walkthrough=selected_walkthrough,
        feature_example=feature_example,
        step_rows=bayesian_step_rows,
        threshold_rows=posterior_flip_threshold_rows,
    )
    return BayesianWalkthroughResult(
        selected_walkthrough=selected_walkthrough,
        feature_example=feature_example,
        bayesian_step_rows=bayesian_step_rows,
        prior_sweep_rows=prior_sweep_rows,
        feature_contribution_rows=feature_contribution_rows,
        posterior_flip_threshold_rows=posterior_flip_threshold_rows,
        report_markdown=report_markdown,
    )


def _plot_prior_to_posterior_single_step(result: BayesianWalkthroughResult):
    row = next(row for row in result.bayesian_step_rows if str(row["example_type"]) == "trajectory_walkthrough")
    class_names = [str(row["class_a"]), str(row["class_b"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    axes[0].bar(class_names, [float(row["prior_a"]), float(row["prior_b"])], color="#2563eb")
    axes[0].set_title("Prior")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(class_names, [float(row["posterior_a"]), float(row["posterior_b"])], color="#7c3aed")
    axes[1].set_title("Posterior")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Prior to Posterior: Representative Update", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _plot_likelihood_curves_with_feature_value(result: BayesianWalkthroughResult):
    row = max(result.feature_contribution_rows, key=lambda item: abs(float(item["log_likelihood_ratio_ab"])))
    observed = float(row["observed_value"])
    mean_a = float(row["class_a_reference"])
    mean_b = float(row["class_b_reference"])
    sigma = float(row["feature_sigma"])
    lower = min(observed, mean_a, mean_b) - 4.0 * sigma
    upper = max(observed, mean_a, mean_b) + 4.0 * sigma
    xs = [lower + (upper - lower) * index / 200.0 for index in range(201)]
    ys_a = [_gaussian_pdf(value, mean_a, sigma) for value in xs]
    ys_b = [_gaussian_pdf(value, mean_b, sigma) for value in xs]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(xs, ys_a, color="#2563eb", label=f"{row['class_pair_id'].split('_vs_')[0]} likelihood")
    ax.plot(xs, ys_b, color="#dc2626", label=f"{row['class_pair_id'].split('_vs_')[1]} likelihood")
    ax.axvline(observed, color="#111827", linestyle="--", linewidth=1.6, label="observed feature")
    ax.set_title(f"Likelihood Curves for `{row['feature_name']}`", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("feature value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_posterior_timeline(result: BayesianWalkthroughResult):
    rows = [row for row in result.bayesian_step_rows if str(row["example_type"]) == "trajectory_walkthrough"]
    class_a = str(rows[0]["class_a"])
    class_b = str(rows[0]["class_b"])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot([float(row["time"]) for row in rows], [float(row["posterior_a"]) for row in rows], color="#2563eb", linewidth=2.2, label=class_a)
    ax.plot([float(row["time"]) for row in rows], [float(row["posterior_b"]) for row in rows], color="#dc2626", linewidth=2.2, label=class_b)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Sequential Posterior Timeline", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("posterior")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_log_odds_timeline(result: BayesianWalkthroughResult):
    rows = [row for row in result.bayesian_step_rows if str(row["example_type"]) == "trajectory_walkthrough"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot([float(row["time"]) for row in rows], [float(row["log_posterior_odds"]) for row in rows], color="#0f766e", linewidth=2.2)
    ax.axhline(0.0, color="#991b1b", linestyle="--", linewidth=1.0)
    ax.set_title("Log Posterior Odds Over Time", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("log odds")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_bayes_factor_timeline(result: BayesianWalkthroughResult):
    rows = [row for row in result.bayesian_step_rows if str(row["example_type"]) == "trajectory_walkthrough"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar([float(row["time"]) for row in rows], [float(row["log_bayes_factor_ab"]) for row in rows], width=0.09, color="#f59e0b")
    ax.axhline(0.0, color="#991b1b", linestyle="--", linewidth=1.0)
    ax.set_title("Incremental Log Bayes Factor", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("log Bayes factor")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_prior_sensitivity_curve(result: BayesianWalkthroughResult):
    rows = sorted(result.prior_sweep_rows, key=lambda row: float(row["prior_a"]))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot([float(row["prior_a"]) for row in rows], [float(row["posterior_a"]) for row in rows], color="#2563eb", linewidth=2.2, label=str(rows[0]["class_a"]))
    ax.plot([float(row["prior_a"]) for row in rows], [float(row["posterior_b"]) for row in rows], color="#dc2626", linewidth=2.2, label=str(rows[0]["class_b"]))
    threshold = float(result.posterior_flip_threshold_rows[0]["theoretical_flip_threshold_prior_a"])
    ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1.2, label="flip threshold")
    ax.set_title("Prior Sensitivity Sweep", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"prior on {rows[0]['class_a']}")
    ax.set_ylabel("final posterior")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_feature_ablation_posterior(result: BayesianWalkthroughResult):
    rows = sorted(result.feature_contribution_rows, key=lambda row: float(row["posterior_true_without_feature"]))
    labels = ["full_set"] + [str(row["feature_name"]) for row in rows]
    values = [float(result.feature_example["full_posterior_true"])] + [float(row["posterior_true_without_feature"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(labels, values, color=["#2563eb"] + ["#7c3aed"] * len(rows))
    ax.set_title("True-Class Posterior After Feature Ablation", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("posterior on true class")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_confidence_threshold_crossing(result: BayesianWalkthroughResult):
    rows = [row for row in result.bayesian_step_rows if str(row["example_type"]) == "trajectory_walkthrough"]
    confidence = [max(float(row["posterior_a"]), float(row["posterior_b"])) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot([float(row["time"]) for row in rows], confidence, color="#2563eb", linewidth=2.2)
    ax.axhline(0.60, color="#f59e0b", linestyle="--", linewidth=1.0, label="0.60")
    ax.axhline(0.80, color="#dc2626", linestyle="--", linewidth=1.0, label="0.80")
    ax.set_title("Confidence Threshold Crossing", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("MAP confidence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_bayesian_walkthrough_artifacts(
    output_dir: str | Path,
    *,
    result: BayesianWalkthroughResult | None = None,
) -> BayesianWalkthroughArtifacts:
    if result is None:
        result = analyze_bayesian_walkthroughs()

    run_dir = Path(output_dir) / "bayesian_walkthroughs"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "bayesian_evidence_walkthrough_report.md"
    bayesian_step_tables_path = run_dir / "bayesian_step_tables.csv"
    prior_sweep_examples_path = run_dir / "prior_sweep_examples.csv"
    feature_contribution_examples_path = run_dir / "feature_contribution_examples.csv"
    posterior_flip_thresholds_path = run_dir / "posterior_flip_thresholds.csv"

    report_path.write_text(result.report_markdown, encoding="utf-8")
    write_csv(
        bayesian_step_tables_path,
        list(result.bayesian_step_rows),
        [
            "example_id",
            "example_type",
            "study_id",
            "trajectory_id",
            "class_pair_id",
            "classifier_id",
            "step_index",
            "time",
            "class_a",
            "class_b",
            "prior_a",
            "prior_b",
            "log_prior_odds",
            "cumulative_log_score_a",
            "cumulative_log_score_b",
            "incremental_log_evidence_a",
            "incremental_log_evidence_b",
            "log_bayes_factor_ab",
            "posterior_a",
            "posterior_b",
            "log_posterior_odds",
            "predicted_class",
            "true_class",
        ],
    )
    write_csv(
        prior_sweep_examples_path,
        list(result.prior_sweep_rows),
        [
            "study_id",
            "trajectory_id",
            "class_pair_id",
            "classifier_id",
            "class_a",
            "class_b",
            "prior_a",
            "prior_b",
            "log_prior_odds",
            "posterior_a",
            "posterior_b",
            "final_log_posterior_odds",
            "predicted_class",
            "confidence",
            "cumulative_log_likelihood_ratio",
        ],
    )
    write_csv(
        feature_contribution_examples_path,
        list(result.feature_contribution_rows),
        [
            "study_id",
            "trajectory_id",
            "class_pair_id",
            "classifier_id",
            "feature_set_id",
            "feature_name",
            "feature_group",
            "history_behavior",
            "evidence_role",
            "double_counting_risk",
            "observed_value",
            "class_a_reference",
            "class_b_reference",
            "feature_sigma",
            "log_likelihood_class_a",
            "log_likelihood_class_b",
            "log_likelihood_ratio_ab",
            "posterior_true_with_full_set",
            "posterior_true_without_feature",
        ],
    )
    write_csv(
        posterior_flip_thresholds_path,
        list(result.posterior_flip_threshold_rows),
        [
            "study_id",
            "trajectory_id",
            "class_pair_id",
            "classifier_id",
            "class_a",
            "class_b",
            "uniform_predicted_class",
            "uniform_confidence",
            "empirical_min_prior_a_for_class_a",
            "empirical_max_prior_a_for_class_b",
            "theoretical_flip_threshold_prior_a",
        ],
    )

    prior_to_posterior_single_step_path = plots_dir / "prior_to_posterior_single_step.png"
    likelihood_curves_with_feature_value_path = plots_dir / "likelihood_curves_with_feature_value.png"
    posterior_timeline_path = plots_dir / "posterior_timeline.png"
    log_odds_timeline_path = plots_dir / "log_odds_timeline.png"
    bayes_factor_timeline_path = plots_dir / "bayes_factor_timeline.png"
    prior_sensitivity_curve_path = plots_dir / "prior_sensitivity_curve.png"
    prior_sweep_examples_png_path = run_dir / "prior_sweep_examples.png"
    feature_ablation_posterior_path = plots_dir / "feature_ablation_posterior.png"
    confidence_threshold_crossing_path = plots_dir / "confidence_threshold_crossing.png"

    write_plot(_plot_prior_to_posterior_single_step(result), prior_to_posterior_single_step_path)
    write_plot(_plot_likelihood_curves_with_feature_value(result), likelihood_curves_with_feature_value_path)
    write_plot(_plot_posterior_timeline(result), posterior_timeline_path)
    write_plot(_plot_log_odds_timeline(result), log_odds_timeline_path)
    write_plot(_plot_bayes_factor_timeline(result), bayes_factor_timeline_path)
    write_plot(_plot_prior_sensitivity_curve(result), prior_sensitivity_curve_path)
    write_plot(_plot_prior_sensitivity_curve(result), prior_sweep_examples_png_path)
    write_plot(_plot_feature_ablation_posterior(result), feature_ablation_posterior_path)
    write_plot(_plot_confidence_threshold_crossing(result), confidence_threshold_crossing_path)

    return BayesianWalkthroughArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        bayesian_step_tables_path=bayesian_step_tables_path,
        prior_sweep_examples_path=prior_sweep_examples_path,
        feature_contribution_examples_path=feature_contribution_examples_path,
        posterior_flip_thresholds_path=posterior_flip_thresholds_path,
        plots_dir=plots_dir,
        prior_to_posterior_single_step_path=prior_to_posterior_single_step_path,
        likelihood_curves_with_feature_value_path=likelihood_curves_with_feature_value_path,
        posterior_timeline_path=posterior_timeline_path,
        log_odds_timeline_path=log_odds_timeline_path,
        bayes_factor_timeline_path=bayes_factor_timeline_path,
        prior_sensitivity_curve_path=prior_sensitivity_curve_path,
        prior_sweep_examples_png_path=prior_sweep_examples_png_path,
        feature_ablation_posterior_path=feature_ablation_posterior_path,
        confidence_threshold_crossing_path=confidence_threshold_crossing_path,
    )
