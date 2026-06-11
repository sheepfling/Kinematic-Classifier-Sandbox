from __future__ import annotations

from collections import defaultdict
from math import isfinite

from .filter_trace import FilterStepTrace


def validate_filter_step_trace(trace: FilterStepTrace) -> None:
    if not trace.run_id:
        raise ValueError("run_id must not be empty")
    if not trace.study_id:
        raise ValueError("study_id must not be empty")
    if not trace.trajectory_id:
        raise ValueError("trajectory_id must not be empty")
    if not trace.method_id:
        raise ValueError("method_id must not be empty")
    if trace.time_index < 0:
        raise ValueError("time_index must be non-negative")
    if trace.dt <= 0.0:
        raise ValueError("dt must be positive")
    for name in ("prior_probability", "predicted_probability", "posterior_probability"):
        value = getattr(trace, name)
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]")
    for name in (
        "time",
        "log_transition_probability",
        "normalized_innovation_squared",
        "log_likelihood",
        "incremental_log_evidence",
        "posterior_entropy",
        "effective_sample_size",
    ):
        value = getattr(trace, name)
        if value is not None and not isfinite(float(value)):
            raise ValueError(f"{name} must be finite when present")
    if trace.effective_sample_size is not None and trace.effective_sample_size <= 0.0:
        raise ValueError("effective_sample_size must be positive when present")


def validate_filter_step_trace_set(traces: tuple[FilterStepTrace, ...] | list[FilterStepTrace], *, atol: float = 1.0e-6) -> None:
    if not traces:
        raise ValueError("trace set must not be empty")
    posterior_by_key: dict[tuple[str, str, str, int], float] = defaultdict(float)
    predicted_by_key: dict[tuple[str, str, str, int], float] = defaultdict(float)
    for trace in traces:
        validate_filter_step_trace(trace)
        key = (trace.run_id, trace.trajectory_id, trace.method_id, trace.time_index)
        if trace.posterior_probability is not None:
            posterior_by_key[key] += trace.posterior_probability
        if trace.predicted_probability is not None:
            predicted_by_key[key] += trace.predicted_probability
    for key, value in posterior_by_key.items():
        if abs(value - 1.0) > atol:
            raise ValueError(f"posterior probabilities must sum to 1.0 for {key}, got {value}")
    for key, value in predicted_by_key.items():
        if abs(value - 1.0) > atol:
            raise ValueError(f"predicted probabilities must sum to 1.0 for {key}, got {value}")
