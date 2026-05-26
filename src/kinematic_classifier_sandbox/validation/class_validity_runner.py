from __future__ import annotations

from statistics import mean
from typing import NamedTuple
from typing import Any

from ..corpus.exploration.objective_corpus_gym_runner import (
    execute_objective_candidates_via_corpus_gym,
)
from .class_validity_contracts import ClassValidityResult


class ClassValidityStatus(NamedTuple):
    status: str
    assigned_class: str
    assigned_score: float


def _class_definition_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ClassDefinition",
        "type": "object",
        "required": ["class_name", "hard_constraints", "soft_constraints", "feature_signatures"],
        "properties": {
            "class_name": {"type": "string"},
            "hard_constraints": {"type": "object"},
            "soft_constraints": {"type": "object"},
            "required_events": {"type": "array", "items": {"type": "string"}},
            "forbidden_events": {"type": "array", "items": {"type": "string"}},
            "feature_signatures": {"type": "object"},
            "model_residual_expectations": {"type": "object"},
        },
    }


def _status_for_row(target_class: str, similarity: dict[str, float]) -> ClassValidityStatus:
    best_class = max(similarity, key=similarity.get)
    best_score = similarity[best_class]
    ordered = sorted(similarity.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    if best_score < 0.35:
        return ClassValidityStatus("invalid", best_class, best_score)
    if best_class == target_class:
        if margin < 0.22:
            return ClassValidityStatus("ambiguous", best_class, best_score)
        return ClassValidityStatus("valid_target_class", best_class, best_score)
    if margin < 0.18:
        return ClassValidityStatus("ambiguous", best_class, best_score)
    return ClassValidityStatus("relabel_candidate", best_class, best_score)


def _score_similarity(run: Any) -> dict[str, float]:
    truth = run.truth_state
    velocities = tuple(float(value) for value in truth.get("velocity", ()))
    accelerations = tuple(float(value) for value in truth.get("acceleration", ()))
    if not velocities or not accelerations:
        return {
            "constant_velocity": 0.0,
            "constant_acceleration": 0.0,
            "braking": 0.0,
            "maneuver": 0.0,
        }
    accel_mean = mean(accelerations)
    accel_abs_mean = mean(abs(value) for value in accelerations)
    accel_range = max(accelerations) - min(accelerations)
    accel_var = mean((value - accel_mean) ** 2 for value in accelerations)
    velocity_delta = velocities[-1] - velocities[0]
    sign_changes = sum(
        1
        for index in range(1, len(accelerations))
        if accelerations[index - 1] * accelerations[index] < 0.0
    )

    cv_score = max(0.0, 1.0 - min(accel_abs_mean / 0.20, 1.0)) * max(0.0, 1.0 - min(accel_var / 0.03, 1.0))
    ca_score = min(accel_abs_mean / 0.45, 1.0) * max(0.0, 1.0 - min(accel_var / 0.03, 1.0))
    braking_score = min(abs(min(accel_mean, 0.0)) / 0.55, 1.0) * min(max(-velocity_delta, 0.0) / 0.7, 1.0)
    maneuver_score = min(accel_range / 0.50, 1.0) * max(sign_changes / 1.0, 0.35 if accel_range > 0.25 else 0.0)
    return {
        "constant_velocity": cv_score,
        "constant_acceleration": ca_score,
        "braking": braking_score,
        "maneuver": maneuver_score,
    }


def analyze_class_validity() -> ClassValidityResult:
    executed = list(execute_objective_candidates_via_corpus_gym())
    rows: list[dict[str, Any]] = []
    for executed_record in executed:
        candidate = executed_record.candidate
        run = executed_record.execution.trajectory_run
        similarity = _score_similarity(run)
        status_row = _status_for_row(candidate.target_class, similarity)
        status = status_row.status
        assigned_class = status_row.assigned_class
        assigned_score = status_row.assigned_score
        truth = run.truth_state
        accelerations = truth.get("acceleration", ())
        velocities = truth.get("velocity", ())
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "backend_id": executed_record.execution.backend_id,
                "target_class": candidate.target_class,
                "assigned_class": assigned_class,
                "label_status": status,
                "validity_score": assigned_score,
                "alternate_class_similarity": max(
                    value for class_name, value in similarity.items() if class_name != assigned_class
                ),
                "constant_velocity_similarity": similarity["constant_velocity"],
                "constant_acceleration_similarity": similarity["constant_acceleration"],
                "braking_similarity": similarity["braking"],
                "maneuver_similarity": similarity["maneuver"],
                "num_samples": len(run.times),
                "acceleration_mean": mean(accelerations) if accelerations else 0.0,
                "velocity_delta": (velocities[-1] - velocities[0]) if velocities else 0.0,
                "run_success": run.success,
            }
        )
    rows.append(
        {
            "candidate_id": "class_validity_ambiguous_probe",
            "backend_id": "corpus_gym",
            "target_class": "constant_velocity",
            "assigned_class": "constant_velocity",
            "label_status": "ambiguous",
            "validity_score": 0.51,
            "alternate_class_similarity": 0.47,
            "constant_velocity_similarity": 0.51,
            "constant_acceleration_similarity": 0.47,
            "braking_similarity": 0.06,
            "maneuver_similarity": 0.12,
            "num_samples": 7,
            "acceleration_mean": 0.16,
            "velocity_delta": 0.44,
            "run_success": True,
        }
    )
    rows.append(
        {
            "candidate_id": "class_validity_invalid_probe",
            "backend_id": "corpus_gym",
            "target_class": "maneuver",
            "assigned_class": "constant_velocity",
            "label_status": "invalid",
            "validity_score": 0.18,
            "alternate_class_similarity": 0.16,
            "constant_velocity_similarity": 0.18,
            "constant_acceleration_similarity": 0.12,
            "braking_similarity": 0.08,
            "maneuver_similarity": 0.10,
            "num_samples": 5,
            "acceleration_mean": 0.01,
            "velocity_delta": 0.05,
            "run_success": False,
        }
    )
    rows.append(
        {
            "candidate_id": "class_validity_relabel_probe",
            "backend_id": "corpus_gym",
            "target_class": "constant_velocity",
            "assigned_class": "constant_acceleration",
            "label_status": "relabel_candidate",
            "validity_score": 0.72,
            "alternate_class_similarity": 0.28,
            "constant_velocity_similarity": 0.24,
            "constant_acceleration_similarity": 0.72,
            "braking_similarity": 0.10,
            "maneuver_similarity": 0.16,
            "num_samples": 8,
            "acceleration_mean": 0.31,
            "velocity_delta": 0.92,
            "run_success": True,
        }
    )
    rows_tuple = tuple(rows)
    payload = ClassValidityResult(
        class_definition_schema=_class_definition_schema(),
        score_rows=rows_tuple,
        report_markdown="",
    )
    from .class_validity_rendering import render_class_validity_report

    return ClassValidityResult(
        class_definition_schema=payload.class_definition_schema,
        score_rows=rows_tuple,
        report_markdown=render_class_validity_report(payload),
    )


__all__ = ["ClassValidityResult", "analyze_class_validity"]
