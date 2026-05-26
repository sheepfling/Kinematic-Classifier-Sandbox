from __future__ import annotations

from typing import NamedTuple

from ...utils.math import _binary_log_odds, _log_odds_from_prior
from .pointwise_baseline import (
    PointwiseClassSpec,
    PointwiseTrajectory,
    default_pointwise_class_specs,
    generate_pointwise_benchmark_trajectories,
    run_pointwise_classifier,
)
from .sequential_bayes_accumulator import (
    AccumulatorClassSpec,
    AccumulatorTrajectory,
    default_accumulator_class_specs,
    generate_accumulator_trajectories,
    run_accumulator,
)
from .windowed_baseline import (
    WindowedClassSpec,
    WindowedFeatureClassifier,
    WindowedTrajectory,
    default_windowed_class_specs,
    extract_windowed_feature_rows,
    generate_windowed_trajectories,
)
from .prior_sensitivity_contracts import (
    CrossMethodPriorComparisonResult,
    PriorFlipThreshold,
    PriorSensitivityResult,
    PriorSensitivitySummary,
    PriorSweepRow,
)


class PriorSensitivityRunnerResult(NamedTuple):
    final_class: str
    final_confidence: float
    final_weights: dict[str, float]
    cumulative_log_likelihood_ratio: float


def _default_prior_grid(step: float = 0.05) -> tuple[float, ...]:
    values = []
    current = step
    while current < 1.0:
        values.append(round(current, 10))
        current += step
    return tuple(values)


def _trajectory_order_key(trajectory: AccumulatorTrajectory) -> tuple[int, str]:
    order = {"easy": 0, "ambiguous": 1, "late_flip": 2}
    return (order.get(trajectory.scenario_name, 99), trajectory.trajectory_id)


def _generic_trajectory_order_key(trajectory) -> tuple[int, str]:
    order = {
        "easy": 0,
        "overlap": 1,
        "ambiguous": 2,
        "late_flip": 3,
        "low_clean": 4,
        "high_clean": 5,
        "low_spike": 6,
        "high_dip": 7,
        "low_long": 8,
        "high_long": 9,
    }
    return (order.get(getattr(trajectory, "scenario_name", ""), 99), getattr(trajectory, "trajectory_id", ""))


def _analyze_generic_prior_sensitivity(
    *,
    method_name: str,
    class_names: tuple[str, str],
    trajectories: tuple[object, ...],
    runner,
    prior_grid: tuple[float, ...],
    confidence_threshold: float,
    forgetting_factor: float,
) -> PriorSensitivityResult:
    class_a, class_b = class_names
    sweep_rows: list[PriorSweepRow] = []
    rows_by_trajectory: dict[str, list[PriorSweepRow]] = {getattr(trajectory, "trajectory_id"): [] for trajectory in trajectories}
    for trajectory in trajectories:
        trajectory_id = getattr(trajectory, "trajectory_id")
        scenario_name = getattr(trajectory, "scenario_name")
        true_class = getattr(trajectory, "true_class")
        for prior_a in prior_grid:
            prior = {class_a: prior_a, class_b: 1.0 - prior_a}
            final_class, final_confidence, final_weights, cumulative_log_likelihood_ratio = runner(trajectory, prior)
            row = PriorSweepRow(
                trajectory_id=trajectory_id,
                scenario_name=scenario_name,
                true_class=true_class,
                prior_a=prior_a,
                prior_b=1.0 - prior_a,
                log_prior_odds=_log_odds_from_prior(prior_a, 1.0 - prior_a),
                final_class=final_class,
                final_confidence=final_confidence,
                abstained=final_class == "unknown",
                posterior_a=final_weights[class_a],
                posterior_b=final_weights[class_b],
                final_log_posterior_odds=_binary_log_odds(final_weights, class_a, class_b),
                cumulative_log_likelihood_ratio=cumulative_log_likelihood_ratio,
            )
            sweep_rows.append(row)
            rows_by_trajectory[trajectory_id].append(row)

    flip_thresholds: list[PriorFlipThreshold] = []
    flipped_by_small_prior = 0
    prior_shift_values: list[float] = []
    log_prior_shift_values: list[float] = []
    ambiguous_uniform_class = "n/a"
    ambiguous_flip_threshold_for_a = None
    small_prior_delta = 0.25
    uniform_prior = 0.50
    for trajectory in trajectories:
        trajectory_id = getattr(trajectory, "trajectory_id")
        scenario_name = getattr(trajectory, "scenario_name")
        true_class = getattr(trajectory, "true_class")
        rows = sorted(rows_by_trajectory[trajectory_id], key=lambda row: row.prior_a)
        uniform_row = min(rows, key=lambda row: abs(row.prior_a - uniform_prior))
        rows_for_a = [row.prior_a for row in rows if row.final_class == class_a]
        rows_for_b = [row.prior_a for row in rows if row.final_class == class_b]
        min_prior_a_for_a = min(rows_for_a) if rows_for_a else None
        max_prior_a_for_b = max(rows_for_b) if rows_for_b else None
        flip_candidates = [row for row in rows if row.final_class != uniform_row.final_class]
        smallest_prior_shift_to_flip = None
        smallest_log_prior_shift_to_flip = None
        if flip_candidates:
            smallest_prior_shift_to_flip = min(abs(row.prior_a - uniform_row.prior_a) for row in flip_candidates)
            prior_shift_values.append(smallest_prior_shift_to_flip)
            base_log_prior = _log_odds_from_prior(uniform_row.prior_a, 1.0 - uniform_row.prior_a)
            smallest_log_prior_shift_to_flip = min(abs(row.log_prior_odds - base_log_prior) for row in flip_candidates)
            log_prior_shift_values.append(smallest_log_prior_shift_to_flip)
        lower_row = min(rows, key=lambda row: abs(row.prior_a - (uniform_prior - small_prior_delta)))
        upper_row = min(rows, key=lambda row: abs(row.prior_a - (uniform_prior + small_prior_delta)))
        if lower_row.final_class != uniform_row.final_class or upper_row.final_class != uniform_row.final_class:
            flipped_by_small_prior += 1
        flip_thresholds.append(
            PriorFlipThreshold(
                trajectory_id=trajectory_id,
                scenario_name=scenario_name,
                true_class=true_class,
                uniform_prior_class=uniform_row.final_class,
                uniform_prior_confidence=uniform_row.final_confidence,
                min_prior_a_for_a=min_prior_a_for_a,
                max_prior_a_for_b=max_prior_a_for_b,
                smallest_prior_shift_to_flip=smallest_prior_shift_to_flip,
                smallest_log_prior_shift_to_flip=smallest_log_prior_shift_to_flip,
            )
        )
        if scenario_name in {"ambiguous", "overlap", "low_spike", "high_dip"}:
            ambiguous_uniform_class = uniform_row.final_class
            ambiguous_flip_threshold_for_a = min_prior_a_for_a

    summary = PriorSensitivitySummary(
        trajectory_count=len(trajectories),
        sweep_count=len(sweep_rows),
        flipped_by_small_prior_fraction=flipped_by_small_prior / max(len(trajectories), 1),
        median_smallest_prior_shift_to_flip=(sorted(prior_shift_values)[len(prior_shift_values) // 2] if prior_shift_values else None),
        median_smallest_log_prior_shift_to_flip=(sorted(log_prior_shift_values)[len(log_prior_shift_values) // 2] if log_prior_shift_values else None),
        ambiguous_uniform_class=ambiguous_uniform_class,
        ambiguous_flip_threshold_for_a=ambiguous_flip_threshold_for_a,
    )
    metrics = {
        "method_name": method_name,
        "forgetting_factor": forgetting_factor,
        "confidence_threshold": confidence_threshold,
        "binary_classes": [class_a, class_b],
        "small_prior_delta": small_prior_delta,
        "fraction_flipped_by_small_prior_perturbation": summary.flipped_by_small_prior_fraction,
        "median_smallest_prior_shift_to_flip": summary.median_smallest_prior_shift_to_flip,
        "median_smallest_log_prior_shift_to_flip": summary.median_smallest_log_prior_shift_to_flip,
        "ambiguous_uniform_class": summary.ambiguous_uniform_class,
        "ambiguous_flip_threshold_for_a": summary.ambiguous_flip_threshold_for_a,
        "decomposition_note": "For recursive Bayes with forgetting_factor=1.0, final log posterior odds equal cumulative log-likelihood ratio plus log prior odds.",
    }
    return PriorSensitivityResult(
        method_name=method_name,
        class_names=class_names,
        trajectories=trajectories,
        sweep_rows=tuple(sweep_rows),
        flip_thresholds=tuple(flip_thresholds),
        summary=summary,
        prior_dominance_metrics=metrics,
    )


def analyze_prior_sensitivity(
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    trajectories_per_class: int = 3,
    class_specs: tuple[AccumulatorClassSpec, ...] | None = None,
    prior_grid: tuple[float, ...] | None = None,
) -> PriorSensitivityResult:
    specs = class_specs or default_accumulator_class_specs()
    if len(specs) != 2:
        raise ValueError("prior sensitivity analysis currently expects exactly two classes")
    grid = prior_grid or _default_prior_grid()
    trajectories = tuple(sorted(generate_accumulator_trajectories(seed=seed, trajectories_per_class=trajectories_per_class), key=_trajectory_order_key))

    def _runner(trajectory: AccumulatorTrajectory, prior: dict[str, float]) -> PriorSensitivityRunnerResult:
        run = run_accumulator(trajectory, specs, forgetting_factor=forgetting_factor, confidence_threshold=confidence_threshold, prior=prior)
        class_a = specs[0].name
        class_b = specs[1].name
        cumulative_log_likelihood_ratio = sum(step.log_likelihood_terms[class_a] - step.log_likelihood_terms[class_b] for step in run.steps)
        return PriorSensitivityRunnerResult(
            final_class=run.final_predicted_class,
            final_confidence=run.final_confidence,
            final_weights=run.final_weights,
            cumulative_log_likelihood_ratio=cumulative_log_likelihood_ratio,
        )

    return _analyze_generic_prior_sensitivity(
        method_name="accumulator",
        class_names=(specs[0].name, specs[1].name),
        trajectories=trajectories,
        runner=_runner,
        prior_grid=grid,
        confidence_threshold=confidence_threshold,
        forgetting_factor=forgetting_factor,
    )


def analyze_pointwise_prior_sensitivity(
    *,
    seed: int = 7,
    class_specs: tuple[PointwiseClassSpec, ...] | None = None,
    prior_grid: tuple[float, ...] | None = None,
) -> PriorSensitivityResult:
    specs = class_specs or default_pointwise_class_specs()
    if len(specs) != 2:
        raise ValueError("pointwise prior sensitivity currently expects exactly two classes")
    grid = prior_grid or _default_prior_grid()
    trajectories = tuple(sorted(generate_pointwise_benchmark_trajectories(seed=seed), key=_generic_trajectory_order_key))

    def _runner(trajectory: PointwiseTrajectory, prior: dict[str, float]) -> PriorSensitivityRunnerResult:
        run = run_pointwise_classifier(trajectory, specs, prior=prior)
        class_a = specs[0].name
        class_b = specs[1].name
        cumulative_log_likelihood_ratio = sum(step.log_likelihood_terms[class_a] - step.log_likelihood_terms[class_b] for step in run.steps)
        confidence = max(run.final_weights.values())
        return PriorSensitivityRunnerResult(
            final_class=run.final_predicted_class,
            final_confidence=confidence,
            final_weights=run.final_weights,
            cumulative_log_likelihood_ratio=cumulative_log_likelihood_ratio,
        )

    return _analyze_generic_prior_sensitivity(
        method_name="pointwise",
        class_names=(specs[0].name, specs[1].name),
        trajectories=trajectories,
        runner=_runner,
        prior_grid=grid,
        confidence_threshold=0.0,
        forgetting_factor=1.0,
    )


def analyze_windowed_prior_sensitivity(
    *,
    seed: int = 7,
    class_specs: tuple[WindowedClassSpec, ...] | None = None,
    prior_grid: tuple[float, ...] | None = None,
    feature_mode: str = "raw",
    window_size: int = 5,
    trim_fraction: float = 0.2,
) -> PriorSensitivityResult:
    specs = class_specs or default_windowed_class_specs()
    if len(specs) != 2:
        raise ValueError("windowed prior sensitivity currently expects exactly two classes")
    grid = prior_grid or _default_prior_grid()
    trajectories = tuple(sorted(generate_windowed_trajectories(seed=seed), key=_generic_trajectory_order_key))

    def _runner(trajectory: WindowedTrajectory, prior: dict[str, float]) -> PriorSensitivityRunnerResult:
        classifier = WindowedFeatureClassifier(specs, feature_mode=feature_mode, prior=prior)
        classifier.reset(prior)
        feature_rows = extract_windowed_feature_rows(trajectory, window_size=window_size, trim_fraction=trim_fraction)
        for row in feature_rows:
            classifier.update(row)
        history = classifier.history()
        final_weights = classifier.posterior()
        final_class = classifier.predict()
        confidence = max(final_weights.values())
        class_a = specs[0].name
        class_b = specs[1].name
        cumulative_log_likelihood_ratio = sum(step.log_likelihood_terms[class_a] - step.log_likelihood_terms[class_b] for step in history)
        return PriorSensitivityRunnerResult(
            final_class=final_class,
            final_confidence=confidence,
            final_weights=final_weights,
            cumulative_log_likelihood_ratio=cumulative_log_likelihood_ratio,
        )

    return _analyze_generic_prior_sensitivity(
        method_name=f"windowed_{feature_mode}",
        class_names=(specs[0].name, specs[1].name),
        trajectories=trajectories,
        runner=_runner,
        prior_grid=grid,
        confidence_threshold=0.0,
        forgetting_factor=1.0,
    )


def analyze_cross_method_prior_comparison(*, seed: int = 7) -> CrossMethodPriorComparisonResult:
    method_results = (
        analyze_prior_sensitivity(seed=seed),
        analyze_pointwise_prior_sensitivity(seed=seed),
        analyze_windowed_prior_sensitivity(seed=seed, feature_mode="raw"),
        analyze_windowed_prior_sensitivity(seed=seed, feature_mode="robust"),
    )
    scenario_names = ("easy", "boundary", "outlier", "transition", "long_history")
    scenario_family_map = {
        "easy": {"easy", "low_clean", "high_clean"},
        "boundary": {"ambiguous", "overlap"},
        "outlier": {"low_spike", "high_dip"},
        "transition": {"late_flip"},
        "long_history": {"low_long", "high_long"},
    }
    rows: list[dict[str, object]] = []
    for result in method_results:
        row = {"method_name": result.method_name}
        for scenario_name in scenario_names:
            matching = [item for item in result.flip_thresholds if item.scenario_name in scenario_family_map[scenario_name]]
            if not matching:
                row[scenario_name] = None
                row[f"{scenario_name}_status"] = "missing"
                continue
            values = [item.smallest_prior_shift_to_flip for item in matching if item.smallest_prior_shift_to_flip is not None]
            if values:
                row[scenario_name] = min(values)
                row[f"{scenario_name}_status"] = "flips"
            else:
                row[scenario_name] = 0.50
                row[f"{scenario_name}_status"] = "stable"
        row["fraction_flipped_by_small_prior_perturbation"] = result.summary.flipped_by_small_prior_fraction
        rows.append(row)
    return CrossMethodPriorComparisonResult(method_results=method_results, scenario_names=scenario_names, rows=tuple(rows))
