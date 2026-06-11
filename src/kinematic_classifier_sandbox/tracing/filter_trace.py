from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils.io import write_csv


@dataclass(frozen=True, slots=True)
class FilterStepTrace:
    run_id: str
    study_id: str
    trajectory_id: str
    method_id: str
    rung: str
    time_index: int
    time: float
    dt: float
    class_or_model: str
    true_class: str | None
    true_mode: str | None
    prior_probability: float | None
    predicted_probability: float | None
    log_transition_probability: float | None
    measurement: tuple[float, ...]
    predicted_measurement: tuple[float, ...] | None
    innovation: tuple[float, ...] | None
    innovation_covariance_diag: tuple[float, ...] | None
    normalized_innovation_squared: float | None
    log_likelihood: float | None
    incremental_log_evidence: float | None
    posterior_probability: float | None
    posterior_entropy: float | None
    predicted_state_mean: tuple[float, ...] | None
    predicted_state_covariance_diag: tuple[float, ...] | None
    updated_state_mean: tuple[float, ...] | None
    updated_state_covariance_diag: tuple[float, ...] | None
    effective_sample_size: float | None
    is_resampled: bool | None


FILTER_STEP_TRACE_FIELDNAMES = [
    "run_id",
    "study_id",
    "trajectory_id",
    "method_id",
    "rung",
    "time_index",
    "time",
    "dt",
    "class_or_model",
    "true_class",
    "true_mode",
    "prior_probability",
    "predicted_probability",
    "log_transition_probability",
    "measurement",
    "predicted_measurement",
    "innovation",
    "innovation_covariance_diag",
    "normalized_innovation_squared",
    "log_likelihood",
    "incremental_log_evidence",
    "posterior_probability",
    "posterior_entropy",
    "predicted_state_mean",
    "predicted_state_covariance_diag",
    "updated_state_mean",
    "updated_state_covariance_diag",
    "effective_sample_size",
    "is_resampled",
]


def tuple_to_cell(values: tuple[float, ...] | None) -> str:
    if values is None:
        return ""
    return " ".join(f"{float(value):.12g}" for value in values)


def trace_to_row(trace: FilterStepTrace) -> dict[str, Any]:
    row = asdict(trace)
    for key in (
        "measurement",
        "predicted_measurement",
        "innovation",
        "innovation_covariance_diag",
        "predicted_state_mean",
        "predicted_state_covariance_diag",
        "updated_state_mean",
        "updated_state_covariance_diag",
    ):
        row[key] = tuple_to_cell(row[key])
    return row


def write_filter_step_trace_csv(path: str | Path, traces: tuple[FilterStepTrace, ...] | list[FilterStepTrace]) -> Path:
    output_path = Path(path)
    write_csv(output_path, [trace_to_row(trace) for trace in traces], FILTER_STEP_TRACE_FIELDNAMES)
    return output_path


def posterior_entropy(probabilities: dict[str, float]) -> float:
    from math import log

    return -sum(float(value) * log(max(float(value), 1.0e-300)) for value in probabilities.values())
