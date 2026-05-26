from __future__ import annotations

from ..analysis.common_dataset_comparison import default_shared_classifier_adapters
from ..inference.kalman_filter_bank import run_kalman_bank_benchmark
from ..inference.pointwise_baseline import run_pointwise_benchmark
from ..inference.prior_sensitivity_analysis import (
    analyze_pointwise_prior_sensitivity,
    analyze_prior_sensitivity,
    analyze_windowed_prior_sensitivity,
)
from ..inference.sequential_bayes_accumulator import run_accumulator_benchmark
from ..inference.velocity_aided_kalman_comparison import analyze_velocity_aided_kalman_comparison
from ..inference.windowed_baseline import run_windowed_benchmark
from .technique_comparison_contracts import (
    TechniqueComparisonResult,
    TechniqueComparisonRow,
    TechniqueDefinition,
)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _pointwise_row(seed: int) -> TechniqueComparisonRow:
    result = run_pointwise_benchmark(seed=seed)
    prior = analyze_pointwise_prior_sensitivity(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name == "easy"]
    overlap_runs = [run for run in result.runs if run.scenario_name == "overlap"]
    return TechniqueComparisonRow(
        method_name="pointwise",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=prior.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        boundary_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in overlap_runs]),
        outlier_accuracy=None,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=0.0,
        model_based=0.0,
        irregular_dt_native=0.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )


def _windowed_rows(seed: int) -> tuple[TechniqueComparisonRow, TechniqueComparisonRow]:
    result = run_windowed_benchmark(seed=seed)
    prior_raw = analyze_windowed_prior_sensitivity(seed=seed, feature_mode="raw")
    prior_robust = analyze_windowed_prior_sensitivity(seed=seed, feature_mode="robust")

    def _scenario_accuracy(runs, matchers: tuple[str, ...]) -> float | None:
        selected = [run for run in runs if any(token in run.scenario_name for token in matchers)]
        return _safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected])

    raw_row = TechniqueComparisonRow(
        method_name="windowed_raw",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.raw_final_accuracy,
        prior_flip_fraction=prior_raw.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior_raw.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_scenario_accuracy(result.raw_runs, ("clean",)),
        boundary_accuracy=None,
        outlier_accuracy=_scenario_accuracy(result.raw_runs, ("spike", "dip")),
        transition_accuracy=None,
        long_history_accuracy=_scenario_accuracy(result.raw_runs, ("long",)),
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=1.0,
        model_based=0.0,
        irregular_dt_native=1.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )
    robust_row = TechniqueComparisonRow(
        method_name="windowed_robust",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.robust_final_accuracy,
        prior_flip_fraction=prior_robust.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior_robust.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_scenario_accuracy(result.robust_runs, ("clean",)),
        boundary_accuracy=None,
        outlier_accuracy=_scenario_accuracy(result.robust_runs, ("spike", "dip")),
        transition_accuracy=None,
        long_history_accuracy=_scenario_accuracy(result.robust_runs, ("long",)),
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=1.0,
        model_based=0.0,
        irregular_dt_native=1.0,
        outlier_aware=1.0,
        stronger_sensor_stream=0.0,
    )
    return raw_row, robust_row


def _accumulator_row(seed: int) -> TechniqueComparisonRow:
    result = run_accumulator_benchmark(seed=seed)
    prior = analyze_prior_sensitivity(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name == "easy"]
    ambiguous_runs = [run for run in result.runs if run.scenario_name == "ambiguous"]
    transition_runs = [run for run in result.runs if run.scenario_name == "late_flip"]
    return TechniqueComparisonRow(
        method_name="accumulator",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=prior.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        boundary_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in ambiguous_runs]),
        outlier_accuracy=None,
        transition_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in transition_runs]),
        long_history_accuracy=None,
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=1.0,
        model_based=0.0,
        irregular_dt_native=0.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )


def _kalman_row(seed: int) -> TechniqueComparisonRow:
    result = run_kalman_bank_benchmark(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name in {"stationary_regular", "constant_velocity_regular"}]
    irregular_runs = [run for run in result.runs if run.scenario_name == "constant_velocity_irregular"]
    acceleration_runs = [run for run in result.runs if run.scenario_name == "constant_acceleration_regular"]
    return TechniqueComparisonRow(
        method_name="kalman_bank",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=0.0,
        median_flip_threshold=None,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        boundary_accuracy=None,
        outlier_accuracy=None,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in irregular_runs]),
        acceleration_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in acceleration_runs]),
        uses_temporal_history=1.0,
        model_based=1.0,
        irregular_dt_native=1.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )


def _kalman_velocity_aided_row(seed: int) -> TechniqueComparisonRow:
    result = analyze_velocity_aided_kalman_comparison(seed=seed)
    position_only = next(row for row in result.rows if row.measurement_mode == "position_only")
    velocity_aided = next(row for row in result.rows if row.measurement_mode == "position_plus_direct_velocity")
    return TechniqueComparisonRow(
        method_name="kalman_bank_velocity_aided",
        sensor_regime_id="position_plus_direct_velocity",
        overall_accuracy=velocity_aided.overall_accuracy,
        prior_flip_fraction=0.0,
        median_flip_threshold=None,
        easy_accuracy=None,
        boundary_accuracy=velocity_aided.short_noisy_accuracy,
        outlier_accuracy=velocity_aided.outlier_accuracy,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=velocity_aided.endpoint_match_accuracy,
        acceleration_accuracy=velocity_aided.short_accuracy,
        uses_temporal_history=1.0,
        model_based=1.0,
        irregular_dt_native=1.0,
        outlier_aware=1.0 if velocity_aided.outlier_accuracy > position_only.outlier_accuracy else 0.0,
        stronger_sensor_stream=1.0,
    )


def default_technique_definitions() -> tuple[TechniqueDefinition, ...]:
    shared_adapters = {adapter.method_name: adapter for adapter in default_shared_classifier_adapters()}
    return (
        TechniqueDefinition(
            method_name="pointwise",
            sensor_regime_id=shared_adapters["pointwise"].sensor_regime_id,
            build_row=_pointwise_row,
        ),
        TechniqueDefinition(
            method_name="windowed_raw",
            sensor_regime_id=shared_adapters["windowed_raw"].sensor_regime_id,
            build_row=lambda seed: _windowed_rows(seed)[0],
        ),
        TechniqueDefinition(
            method_name="windowed_robust",
            sensor_regime_id=shared_adapters["windowed_robust"].sensor_regime_id,
            build_row=lambda seed: _windowed_rows(seed)[1],
        ),
        TechniqueDefinition(
            method_name="accumulator",
            sensor_regime_id=shared_adapters["accumulator"].sensor_regime_id,
            build_row=_accumulator_row,
        ),
        TechniqueDefinition(
            method_name="kalman_bank",
            sensor_regime_id=shared_adapters["kalman_bank"].sensor_regime_id,
            build_row=_kalman_row,
        ),
        TechniqueDefinition(
            method_name="kalman_bank_velocity_aided",
            sensor_regime_id=shared_adapters["kalman_bank_velocity_aided"].sensor_regime_id,
            build_row=_kalman_velocity_aided_row,
        ),
    )


def analyze_technique_comparison(*, seed: int = 7) -> TechniqueComparisonResult:
    definitions = default_technique_definitions()
    windowed_cache: tuple[TechniqueComparisonRow, TechniqueComparisonRow] | None = None
    rows: list[TechniqueComparisonRow] = []
    for definition in definitions:
        if definition.method_name.startswith("windowed_"):
            if windowed_cache is None:
                windowed_cache = _windowed_rows(seed)
            row = windowed_cache[0] if definition.method_name == "windowed_raw" else windowed_cache[1]
        else:
            row = definition.build_row(seed)
        rows.append(row)
    return TechniqueComparisonResult(rows=tuple(rows))


__all__ = [
    "TechniqueComparisonResult",
    "TechniqueComparisonRow",
    "TechniqueDefinition",
    "analyze_technique_comparison",
    "default_technique_definitions",
]
