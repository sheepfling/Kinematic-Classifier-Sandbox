from __future__ import annotations

from typing import Any, NamedTuple

from .objectives import default_corpus_objectives


class CanonicalPair(NamedTuple):
    pair_id: str
    class_a: str
    class_b: str


def _objective_lookup() -> dict[str, Any]:
    return {objective.objective_id: objective for objective in default_corpus_objectives()}


def _canonical_pair(record) -> CanonicalPair:
    objective = _objective_lookup().get(str(record.candidate.provenance.get("objective_id", "")))
    if objective is not None and objective.target_class_pair is not None:
        class_a, class_b = objective.target_class_pair
        return CanonicalPair(pair_id=f"{class_a}_vs_{class_b}", class_a=class_a, class_b=class_b)
    target = record.candidate.target_class
    mapping = {
        "constant_acceleration": ("constant_velocity_vs_constant_acceleration", "constant_velocity", "constant_acceleration"),
        "braking": ("constant_velocity_vs_braking", "constant_velocity", "braking"),
        "maneuver": ("constant_acceleration_vs_maneuver", "constant_acceleration", "maneuver"),
        "constant_velocity": ("stationary_vs_constant_velocity", "stationary", "constant_velocity"),
    }
    pair_id, class_a, class_b = mapping.get(
        target,
        ("constant_velocity_vs_constant_acceleration", "constant_velocity", "constant_acceleration"),
    )
    return CanonicalPair(pair_id=pair_id, class_a=class_a, class_b=class_b)


def _canonical_scenario_id(record) -> str:
    mapping = {
        "boundary_v1": "endpoint_match",
        "stress_v1": "short_noisy",
        "adversarial_v1": "outlier",
        "realistic_v1": "irregular",
        "easy_v1": "easy",
    }
    return mapping.get(record.candidate.difficulty_tier, "easy")


def _record_to_executable(record):
    run = record.execution.trajectory_run
    canonical_pair = _canonical_pair(record)
    pair_id = canonical_pair.pair_id
    class_a = canonical_pair.class_a
    class_b = canonical_pair.class_b
    scenario_id = _canonical_scenario_id(record)
    truth = run.truth_state
    from ..common_experiment.adapters import ExecutableTrajectory

    return ExecutableTrajectory(
        trajectory_id=run.run_id,
        class_pair_id=pair_id,
        class_a=class_a,
        class_b=class_b,
        true_class=record.assigned_class,
        scenario_id=scenario_id,
        seed=run.seed,
        times=tuple(float(value) for value in run.times),
        measurements=tuple(float(value) for value in run.observations.get("position", ())),
        true_position=tuple(float(value) for value in truth.get("position", ())),
        true_velocity=tuple(float(value) for value in truth.get("velocity", ())),
        true_acceleration=tuple(float(value) for value in truth.get("acceleration", ())),
        measurement_dim=1,
        coordinate_frame="scalar_line",
    )
