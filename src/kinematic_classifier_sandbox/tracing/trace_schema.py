from __future__ import annotations

from .filter_trace import FILTER_STEP_TRACE_FIELDNAMES


def filter_step_trace_schema() -> dict[str, object]:
    nullable_float = {"type": ["number", "null"]}
    nullable_vector = {"type": ["array", "null"], "items": {"type": "number"}}
    return {
        "artifact": "filter_step_trace",
        "primary_key": ["run_id", "trajectory_id", "method_id", "time_index", "class_or_model"],
        "fieldnames": FILTER_STEP_TRACE_FIELDNAMES,
        "fields": {
            "run_id": {"type": "string"},
            "study_id": {"type": "string"},
            "trajectory_id": {"type": "string"},
            "method_id": {"type": "string"},
            "rung": {"type": "string"},
            "time_index": {"type": "integer"},
            "time": {"type": "number"},
            "dt": {"type": "number"},
            "class_or_model": {"type": "string"},
            "true_class": {"type": ["string", "null"]},
            "true_mode": {"type": ["string", "null"]},
            "prior_probability": nullable_float,
            "predicted_probability": nullable_float,
            "log_transition_probability": nullable_float,
            "measurement": {"type": "array", "items": {"type": "number"}},
            "predicted_measurement": nullable_vector,
            "innovation": nullable_vector,
            "innovation_covariance_diag": nullable_vector,
            "normalized_innovation_squared": nullable_float,
            "log_likelihood": nullable_float,
            "incremental_log_evidence": nullable_float,
            "posterior_probability": nullable_float,
            "posterior_entropy": nullable_float,
            "predicted_state_mean": nullable_vector,
            "predicted_state_covariance_diag": nullable_vector,
            "updated_state_mean": nullable_vector,
            "updated_state_covariance_diag": nullable_vector,
            "effective_sample_size": nullable_float,
            "is_resampled": {"type": ["boolean", "null"]},
        },
    }
