from __future__ import annotations

import csv
import tempfile
from typing import NamedTuple

from ..advanced_filters.evaluation import (
    write_particle_filter_witness_artifacts,
    write_rbpf_witness_artifacts,
)
from ..advanced_filters.ou_witness import write_ornstein_uhlenbeck_witness_artifacts
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
    TechniqueScenarioSupportRow,
)


class WindowedRows(NamedTuple):
    raw: TechniqueComparisonRow
    robust: TechniqueComparisonRow


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _base_row(adapter_map, method_name: str, **kwargs: object) -> TechniqueComparisonRow:
    method_spec = adapter_map[method_name].method_spec
    values = {
        "method_name": method_spec.method_name,
        "sensor_regime_id": method_spec.sensor_regime_id,
        "applicability_status": "supported",
        "primary_evaluation_family": method_spec.primary_evaluation_family,
        "witness_artifact": method_spec.witness_artifact,
        "overall_accuracy": None,
        "prior_flip_fraction": None,
        "median_flip_threshold": None,
        "easy_accuracy": None,
        "boundary_accuracy": None,
        "outlier_accuracy": None,
        "transition_accuracy": None,
        "long_history_accuracy": None,
        "irregular_dt_accuracy": None,
        "acceleration_accuracy": None,
    }
    values.update(kwargs)
    return TechniqueComparisonRow(**values)


def _pointwise_row(seed: int, adapter_map) -> TechniqueComparisonRow:
    result = run_pointwise_benchmark(seed=seed)
    prior = analyze_pointwise_prior_sensitivity(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name == "easy"]
    overlap_runs = [run for run in result.runs if run.scenario_name == "overlap"]
    return _base_row(
        adapter_map,
        "pointwise",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=prior.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        boundary_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in overlap_runs]),
    )


def _windowed_rows(seed: int, adapter_map) -> WindowedRows:
    result = run_windowed_benchmark(seed=seed)
    prior_raw = analyze_windowed_prior_sensitivity(seed=seed, feature_mode="raw")
    prior_robust = analyze_windowed_prior_sensitivity(seed=seed, feature_mode="robust")

    def _scenario_accuracy(runs, matchers: tuple[str, ...]) -> float | None:
        selected = [run for run in runs if any(token in run.scenario_name for token in matchers)]
        return _safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected])

    return WindowedRows(
        raw=_base_row(
            adapter_map,
            "windowed_raw",
            overall_accuracy=result.summary.raw_final_accuracy,
            prior_flip_fraction=prior_raw.summary.flipped_by_small_prior_fraction,
            median_flip_threshold=prior_raw.summary.median_smallest_prior_shift_to_flip,
            easy_accuracy=_scenario_accuracy(result.raw_runs, ("clean",)),
            outlier_accuracy=_scenario_accuracy(result.raw_runs, ("spike", "dip")),
            long_history_accuracy=_scenario_accuracy(result.raw_runs, ("long",)),
        ),
        robust=_base_row(
            adapter_map,
            "windowed_robust",
            overall_accuracy=result.summary.robust_final_accuracy,
            prior_flip_fraction=prior_robust.summary.flipped_by_small_prior_fraction,
            median_flip_threshold=prior_robust.summary.median_smallest_prior_shift_to_flip,
            easy_accuracy=_scenario_accuracy(result.robust_runs, ("clean",)),
            outlier_accuracy=_scenario_accuracy(result.robust_runs, ("spike", "dip")),
            long_history_accuracy=_scenario_accuracy(result.robust_runs, ("long",)),
        ),
    )


def _accumulator_row(seed: int, adapter_map) -> TechniqueComparisonRow:
    result = run_accumulator_benchmark(seed=seed)
    prior = analyze_prior_sensitivity(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name == "easy"]
    ambiguous_runs = [run for run in result.runs if run.scenario_name == "ambiguous"]
    transition_runs = [run for run in result.runs if run.scenario_name == "late_flip"]
    return _base_row(
        adapter_map,
        "accumulator",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=prior.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        boundary_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in ambiguous_runs]),
        transition_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in transition_runs]),
    )


def _kalman_row(seed: int, adapter_map) -> TechniqueComparisonRow:
    result = run_kalman_bank_benchmark(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name in {"stationary_regular", "constant_velocity_regular"}]
    irregular_runs = [run for run in result.runs if run.scenario_name == "constant_velocity_irregular"]
    acceleration_runs = [run for run in result.runs if run.scenario_name == "constant_acceleration_regular"]
    return _base_row(
        adapter_map,
        "kalman_bank",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=0.0,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        irregular_dt_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in irregular_runs]),
        acceleration_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in acceleration_runs]),
    )


def _kalman_velocity_aided_row(seed: int, adapter_map) -> TechniqueComparisonRow:
    result = analyze_velocity_aided_kalman_comparison(seed=seed)
    velocity_aided = next(row for row in result.rows if row.measurement_mode == "position_plus_direct_velocity")
    return _base_row(
        adapter_map,
        "kalman_bank_velocity_aided",
        overall_accuracy=velocity_aided.overall_accuracy,
        prior_flip_fraction=0.0,
        boundary_accuracy=velocity_aided.short_noisy_accuracy,
        outlier_accuracy=velocity_aided.outlier_accuracy,
        irregular_dt_accuracy=velocity_aided.endpoint_match_accuracy,
        acceleration_accuracy=velocity_aided.short_accuracy,
    )


def _witness_only_row(adapter_map, method_name: str) -> TechniqueComparisonRow:
    method_spec = adapter_map[method_name].method_spec
    return TechniqueComparisonRow(
        method_name=method_spec.method_name,
        sensor_regime_id=method_spec.sensor_regime_id,
        applicability_status="witness_only",
        primary_evaluation_family=method_spec.primary_evaluation_family,
        witness_artifact=method_spec.witness_artifact,
        overall_accuracy=None,
        prior_flip_fraction=None,
        median_flip_threshold=None,
        easy_accuracy=None,
        boundary_accuracy=None,
        outlier_accuracy=None,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
    )


def _read_first_csv_row(path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    return {} if row is None else {str(key): str(value) for key, value in row.items()}


def _advanced_witness_metric_rows() -> dict[str, tuple[str, float | None, str]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        pf_artifacts = write_particle_filter_witness_artifacts(temp_dir)
        rbpf_artifacts = write_rbpf_witness_artifacts(temp_dir)
        ou_artifacts = write_ornstein_uhlenbeck_witness_artifacts(temp_dir)
        pf_metrics = _read_first_csv_row(pf_artifacts.metrics_path)
        rbpf_metrics = _read_first_csv_row(rbpf_artifacts.metrics_path)
        ou_metrics = _read_first_csv_row(ou_artifacts.metrics_path)
    return {
        "particle_filter_bank:nonlinear_drag_outlier": (
            "position_rmse",
            float(pf_metrics["position_rmse"]) if pf_metrics.get("position_rmse") else None,
            "PF witness metric from nonlinear drag/outlier study.",
        ),
        "particle_filter_bank:ou_mean_reversion": (
            "final_mean_reverting_posterior",
            float(ou_metrics["final_mean_reverting_posterior"]) if ou_metrics.get("final_mean_reverting_posterior") else None,
            "PF-family OU witness metric for mean-reverting stochastic dynamics.",
        ),
        "rbpf:latent_maneuver_onset": (
            "post_onset_mode_accuracy",
            float(rbpf_metrics["post_onset_mode_accuracy"]) if rbpf_metrics.get("post_onset_mode_accuracy") else None,
            "RBPF witness metric from latent maneuver onset study.",
        ),
        "ornstein_uhlenbeck_pf_v1:ou_mean_reversion": (
            "final_mean_reverting_posterior",
            float(ou_metrics["final_mean_reverting_posterior"]) if ou_metrics.get("final_mean_reverting_posterior") else None,
            "OU witness metric for PF-family mean-reversion support.",
        ),
    }


def _ou_witness_row() -> TechniqueComparisonRow:
    return TechniqueComparisonRow(
        method_name="ornstein_uhlenbeck_pf_v1",
        sensor_regime_id="position_only",
        applicability_status="witness_only",
        primary_evaluation_family="ou_mean_reversion",
        witness_artifact="artifacts/ornstein_uhlenbeck_witness_v1/ou_method_comparison.csv",
        overall_accuracy=None,
        prior_flip_fraction=None,
        median_flip_threshold=None,
        easy_accuracy=None,
        boundary_accuracy=None,
        outlier_accuracy=None,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
    )


def default_technique_definitions() -> tuple[TechniqueDefinition, ...]:
    shared_adapters = {adapter.method_name: adapter for adapter in default_shared_classifier_adapters()}
    return (
        TechniqueDefinition(method_spec=shared_adapters["pointwise"].method_spec, build_row=lambda seed: _pointwise_row(seed, shared_adapters)),
        TechniqueDefinition(method_spec=shared_adapters["windowed_raw"].method_spec, build_row=lambda seed: _windowed_rows(seed, shared_adapters).raw),
        TechniqueDefinition(method_spec=shared_adapters["windowed_robust"].method_spec, build_row=lambda seed: _windowed_rows(seed, shared_adapters).robust),
        TechniqueDefinition(method_spec=shared_adapters["accumulator"].method_spec, build_row=lambda seed: _accumulator_row(seed, shared_adapters)),
        TechniqueDefinition(method_spec=shared_adapters["kalman_bank"].method_spec, build_row=lambda seed: _kalman_row(seed, shared_adapters)),
        TechniqueDefinition(method_spec=shared_adapters["kalman_bank_velocity_aided"].method_spec, build_row=lambda seed: _kalman_velocity_aided_row(seed, shared_adapters)),
        TechniqueDefinition(method_spec=shared_adapters["particle_filter_bank"].method_spec, build_row=lambda seed: _witness_only_row(shared_adapters, "particle_filter_bank")),
        TechniqueDefinition(method_spec=shared_adapters["rbpf"].method_spec, build_row=lambda seed: _witness_only_row(shared_adapters, "rbpf")),
    )


def _scenario_support_rows(result_rows: tuple[TechniqueComparisonRow, ...], adapter_map) -> tuple[TechniqueScenarioSupportRow, ...]:
    scenario_rows: list[TechniqueScenarioSupportRow] = []
    advanced_metrics = _advanced_witness_metric_rows()
    classic_fields = (
        ("easy", "easy_accuracy"),
        ("boundary", "boundary_accuracy"),
        ("outlier", "outlier_accuracy"),
        ("transition", "transition_accuracy"),
        ("long_history", "long_history_accuracy"),
        ("irregular_dt", "irregular_dt_accuracy"),
        ("acceleration", "acceleration_accuracy"),
    )
    for row in result_rows:
        for scenario_family, field_name in classic_fields:
            metric_value = getattr(row, field_name)
            scenario_rows.append(
                TechniqueScenarioSupportRow(
                    method_name=row.method_name,
                    scenario_family=scenario_family,
                    applicability_status="supported" if metric_value is not None else "not_applicable",
                    metric_name=field_name,
                    metric_value=metric_value,
                    note="Shared benchmark metric." if metric_value is not None else "Method is not scored on this benchmark family.",
                )
            )
        if row.method_name == "ornstein_uhlenbeck_pf_v1":
            supported_families = {"ou_mean_reversion"}
        else:
            supported_families = set(adapter_map.get(row.method_name, None).method_spec.supported_scenario_families if row.method_name in adapter_map else ())
        for scenario_family in ("nonlinear_drag_outlier", "latent_maneuver_onset", "ou_mean_reversion"):
            metric_name, metric_value, note = advanced_metrics.get(
                f"{row.method_name}:{scenario_family}",
                (None, None, "Capability manifest does not cover this advanced family."),
            )
            supported = scenario_family in supported_families
            scenario_rows.append(
                TechniqueScenarioSupportRow(
                    method_name=row.method_name,
                    scenario_family=scenario_family,
                    applicability_status="witness_only" if supported else "not_applicable",
                    metric_name=metric_name if supported else None,
                    metric_value=metric_value if supported else None,
                    note=note if supported else "Capability manifest does not cover this advanced family.",
                )
            )
    return tuple(scenario_rows)


def analyze_technique_comparison(*, seed: int = 7) -> TechniqueComparisonResult:
    definitions = default_technique_definitions()
    adapter_map = {adapter.method_name: adapter for adapter in default_shared_classifier_adapters()}
    rows = [definition.build_row(seed) for definition in definitions]
    rows.append(_ou_witness_row())
    return TechniqueComparisonResult(
        rows=tuple(rows),
        method_specs=tuple(definition.method_spec for definition in definitions),
        scenario_support_rows=_scenario_support_rows(tuple(rows), adapter_map),
    )


__all__ = [
    "TechniqueComparisonResult",
    "TechniqueComparisonRow",
    "TechniqueDefinition",
    "TechniqueScenarioSupportRow",
    "analyze_technique_comparison",
    "default_technique_definitions",
]
