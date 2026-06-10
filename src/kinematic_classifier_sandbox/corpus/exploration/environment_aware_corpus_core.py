from __future__ import annotations

from statistics import mean
from typing import Any

from ..trajectory_backend_contract import default_backend_contract_definitions
from .backend_adapter_proof_core import (
    BackendCandidateSpec,
    EnvironmentAware1DAdapter,
    _environment_candidate,
)


def _environment_adapter() -> EnvironmentAware1DAdapter:
    definition = {
        definition.capabilities.backend_id: definition
        for definition in default_backend_contract_definitions()
    }["environment_aware_1d"]
    return EnvironmentAware1DAdapter(definition)


def _environment_regimes() -> tuple[dict[str, Any], ...]:
    return (
        {
            "environment_id": "dense_calm",
            "density_scale": 1.10,
            "wind_bias": 0.00,
            "drag_coefficient": 0.28,
            "description": "Dense and calm reference regime.",
        },
        {
            "environment_id": "nominal_mixed",
            "density_scale": 1.00,
            "wind_bias": 0.05,
            "drag_coefficient": 0.20,
            "description": "Nominal density with mild wind bias.",
        },
        {
            "environment_id": "thin_windy",
            "density_scale": 0.82,
            "wind_bias": 0.12,
            "drag_coefficient": 0.12,
            "description": "Thin and wind-biased regime.",
        },
    )


def _class_specs() -> tuple[dict[str, Any], ...]:
    return (
        {"target_class": "constant_velocity", "acceleration": 0.04, "initial_velocity": 1.05},
        {"target_class": "constant_acceleration", "acceleration": 0.42, "initial_velocity": 0.78},
    )


def _candidate_rows() -> tuple[BackendCandidateSpec, ...]:
    base = _environment_candidate()
    candidates: list[BackendCandidateSpec] = []
    seed = 500
    for class_index, class_spec in enumerate(_class_specs()):
        for regime_index, regime in enumerate(_environment_regimes()):
            for replicate in range(2):
                candidates.append(
                    BackendCandidateSpec(
                        candidate_id=f"{class_spec['target_class']}_{regime['environment_id']}_{replicate}",
                        scenario_id=f"environment_{class_spec['target_class']}_{regime['environment_id']}",
                        scenario_family="environment_regime_case",
                        target_class=str(class_spec["target_class"]),
                        difficulty_tier="realistic_v1",
                        seed=seed + class_index * 100 + regime_index * 10 + replicate,
                        duration=base.duration,
                        sample_period=base.sample_period,
                        initial_position=base.initial_position,
                        initial_velocity=float(class_spec["initial_velocity"]) + 0.03 * replicate,
                        acceleration=float(class_spec["acceleration"]),
                        measurement_std=base.measurement_std,
                        drag_coefficient=float(regime["drag_coefficient"]),
                        density_scale=float(regime["density_scale"]),
                        wind_bias=float(regime["wind_bias"]),
                        provenance={
                            "search_method": "environment_regime_targeting",
                            "search_iteration": len(candidates),
                            "environment_id": regime["environment_id"],
                        },
                    )
                )
    return tuple(candidates)


def _trajectory_summary_row(candidate: BackendCandidateSpec, run: dict[str, Any]) -> dict[str, Any]:
    density_trace = run["environment_trace"]["density_scale"]
    wind_trace = run["environment_trace"]["wind_bias"]
    velocities = run["truth_state"]["velocity"]
    positions = run["truth_state"]["position"]
    return {
        "trajectory_id": f"{candidate.candidate_id}_trajectory",
        "candidate_id": candidate.candidate_id,
        "true_class": candidate.target_class,
        "environment_id": candidate.provenance["environment_id"],
        "seed": candidate.seed,
        "duration": candidate.duration,
        "num_samples": len(run["times"]),
        "density_mean": mean(density_trace),
        "wind_bias_mean": mean(wind_trace),
        "drag_coefficient": candidate.drag_coefficient,
        "position_final": positions[-1],
        "speed_final": velocities[-1],
        "speed_range": max(velocities) - min(velocities),
        "position_range": max(positions) - min(positions),
        "environment_trace_available": True,
        "environment_feature_view": "available",
        "agnostic_feature_view": "available",
    }


def _coverage_rows(summary_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["environment_id"]), str(row["true_class"]))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (environment_id, true_class), members in sorted(grouped.items()):
        rows.append(
            {
                "environment_id": environment_id,
                "true_class": true_class,
                "trajectory_count": len(members),
                "mean_duration": mean(float(member["duration"]) for member in members),
                "mean_density": mean(float(member["density_mean"]) for member in members),
                "mean_wind_bias": mean(float(member["wind_bias_mean"]) for member in members),
                "mean_speed_range": mean(float(member["speed_range"]) for member in members),
                "mean_position_range": mean(float(member["position_range"]) for member in members),
            }
        )
    return tuple(rows)


def _leakage_rows(summary_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    selected_rows = list(summary_rows)
    biased_control_rows = [
        row for row in summary_rows
        if (row["true_class"] == "constant_velocity" and row["environment_id"] == "dense_calm")
        or (row["true_class"] == "constant_acceleration" and row["environment_id"] == "thin_windy")
    ]
    slices = {
        "selected_corpus": selected_rows,
        "biased_control_slice": biased_control_rows,
    }
    variables = ("density_mean", "wind_bias_mean", "drag_coefficient")
    rows: list[dict[str, Any]] = []
    for slice_id, members in slices.items():
        for variable in variables:
            cv_values = [float(row[variable]) for row in members if row["true_class"] == "constant_velocity"]
            ca_values = [float(row[variable]) for row in members if row["true_class"] == "constant_acceleration"]
            if not cv_values or not ca_values:
                continue
            cv_mean = mean(cv_values)
            ca_mean = mean(ca_values)
            delta_ratio = abs(cv_mean - ca_mean) / max(abs(cv_mean) + abs(ca_mean), 1e-6)
            flagged = delta_ratio >= 0.15
            rows.append(
                {
                    "slice_id": slice_id,
                    "variable_name": variable,
                    "class_a": "constant_velocity",
                    "class_b": "constant_acceleration",
                    "class_a_mean": cv_mean,
                    "class_b_mean": ca_mean,
                    "delta_ratio": delta_ratio,
                    "flagged_class_linkage": flagged,
                }
            )
    return tuple(rows)
