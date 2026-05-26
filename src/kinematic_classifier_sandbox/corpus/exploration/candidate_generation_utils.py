from __future__ import annotations

import random
from typing import Any

from ..objectives import CorpusObjectiveSpec
from .backend_adapter_proof import BackendCandidateSpec
from .candidate_generation_types import CandidateGenerationRow


def _backend_constraints_for_objective(objective: CorpusObjectiveSpec) -> tuple[str, ...]:
    if objective.backend_constraints:
        return objective.backend_constraints
    return ("parameter_only_1d",)


def _base_class_parameters(class_name: str) -> dict[str, float]:
    defaults = {
        "constant_velocity": {"initial_velocity": 1.0, "acceleration": 0.05},
        "constant_acceleration": {"initial_velocity": 0.8, "acceleration": 0.40},
        "braking": {"initial_velocity": 1.35, "acceleration": 0.0},
        "maneuver": {"initial_velocity": 0.9, "acceleration": 0.28},
    }
    return defaults.get(class_name, {"initial_velocity": 1.0, "acceleration": 0.10})


def _objective_primary_class(objective: CorpusObjectiveSpec) -> str:
    if objective.target_class is not None:
        return objective.target_class
    assert objective.target_class_pair is not None
    return objective.target_class_pair[0]


def _base_candidate(objective: CorpusObjectiveSpec, backend_id: str, index: int) -> BackendCandidateSpec:
    class_name = _objective_primary_class(objective)
    base = _base_class_parameters(class_name)
    scenario_family = (
        "environment_regime_case"
        if objective.target_environment_regimes
        else "switching_case"
        if objective.target_class == "braking" and "controlled_1d" in backend_id
        else "shared_boundary_case"
    )
    candidate_id = f"{objective.objective_id}_{backend_id}_{index}"
    environment_id = objective.target_environment_regimes[index % max(len(objective.target_environment_regimes), 1)] if objective.target_environment_regimes else ""
    return BackendCandidateSpec(
        candidate_id=candidate_id,
        scenario_id=objective.objective_id,
        scenario_family=scenario_family,
        target_class=class_name,
        difficulty_tier=objective.target_difficulty,
        seed=1_000 + index,
        duration=2.0,
        sample_period=0.5,
        initial_position=0.0,
        initial_velocity=base["initial_velocity"],
        acceleration=base["acceleration"],
        measurement_std=0.03,
        switch_time=1.0 if scenario_family == "switching_case" else None,
        acceleration_after_switch=-0.7 if scenario_family == "switching_case" else None,
        drag_coefficient=0.20 if scenario_family == "environment_regime_case" else None,
        density_scale=1.0 if environment_id == "nominal_mixed" else 1.1 if environment_id == "dense_calm" else 0.82 if environment_id else None,
        wind_bias=0.05 if environment_id == "nominal_mixed" else 0.0 if environment_id == "dense_calm" else 0.12 if environment_id else None,
        input_deck_hash=f"{candidate_id}_deck" if "mock_file_backend_1d" in backend_id else None,
        longitudinal_command=(0.0, 0.0, -0.7, -0.7, -0.7) if scenario_family == "switching_case" else (),
        provenance={"objective_id": objective.objective_id, "sampler_name": "", "backend_id": backend_id, "parent_candidate_id": "", "environment_id": environment_id},
    )


def _candidate_row(candidate: BackendCandidateSpec, sampler_name: str, parent_candidate_id: str) -> CandidateGenerationRow:
    target_type = (
        "target_feature_cell"
        if candidate.provenance.get("objective_id", "").endswith("_feature")
        else "target_class_pair"
        if candidate.target_class in {"braking", "maneuver"}
        else "target_class"
    )
    feature_excitation = max(0.0, min(1.0, 1.0 - abs(candidate.acceleration) * 0.65 + (0.08 if candidate.provenance.get("environment_id") else 0.0)))
    coverage_gain = max(0.0, min(1.0, 0.35 + 0.15 * len(candidate.provenance.get("environment_id", "")) + 0.05 * (candidate.duration - 1.5)))
    boundary_closeness = max(0.0, min(1.0, 1.0 - abs(candidate.acceleration - 0.28) * 1.4))
    total_utility = 0.50 * feature_excitation + 0.30 * coverage_gain + 0.20 * boundary_closeness
    return {
        "candidate_id": candidate.candidate_id,
        "objective_id": candidate.provenance["objective_id"],
        "target_type": target_type,
        "sampler_name": sampler_name,
        "search_method": sampler_name,
        "backend_id": candidate.provenance["backend_id"],
        "scenario_family": candidate.scenario_family,
        "target_class": candidate.target_class,
        "difficulty_tier": candidate.difficulty_tier,
        "seed": candidate.seed,
        "duration": candidate.duration,
        "sample_period": candidate.sample_period,
        "initial_velocity": candidate.initial_velocity,
        "acceleration": candidate.acceleration,
        "measurement_std": candidate.measurement_std,
        "environment_id": candidate.provenance.get("environment_id", ""),
        "parent_candidate_id": parent_candidate_id,
        "selected": 1,
        "feature_excitation": feature_excitation,
        "coverage_gain": coverage_gain,
        "boundary_closeness": boundary_closeness,
        "total_utility": total_utility,
    }


def _random_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int, rng: random.Random) -> list[BackendCandidateSpec]:
    candidates = []
    for index in range(max(2, budget // 6)):
        base = _base_candidate(objective, backend_id, index)
        candidates.append(
            _replace_candidate(
                base,
                seed=base.seed + 10 * index,
                duration=1.7 + 0.6 * rng.random(),
                measurement_std=0.02 + 0.05 * rng.random(),
                initial_velocity=base.initial_velocity + 0.15 * (rng.random() - 0.5),
                acceleration=base.acceleration + 0.15 * (rng.random() - 0.5),
                sampler_name="random",
                parent_candidate_id="",
            )
        )
    return candidates


def _grid_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    grid = [(1.8, 0.03), (2.0, 0.04), (2.2, 0.05)]
    candidates = []
    for index, (duration, measurement_std) in enumerate(grid[: max(2, budget // 6)]):
        base = _base_candidate(objective, backend_id, 100 + index)
        candidates.append(_replace_candidate(base, duration=duration, measurement_std=measurement_std, sampler_name="grid", parent_candidate_id=""))
    return candidates


def _lhs_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    spans = [(0.15, 0.25), (0.35, 0.50), (0.55, 0.75)]
    candidates = []
    for index, (alpha, beta) in enumerate(spans[: max(2, budget // 6)]):
        base = _base_candidate(objective, backend_id, 200 + index)
        duration = 1.6 + alpha
        measurement_std = 0.02 + 0.04 * beta
        candidates.append(
            _replace_candidate(
                base,
                duration=duration,
                measurement_std=measurement_std,
                initial_velocity=base.initial_velocity + 0.1 * alpha,
                acceleration=base.acceleration + 0.1 * (beta - 0.5),
                sampler_name="lhs",
                parent_candidate_id="",
            )
        )
    return candidates


def _boundary_mutation_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    seeds = _grid_sampler(objective, backend_id, budget)
    candidates = []
    for index, seed_candidate in enumerate(seeds):
        candidates.append(
            _replace_candidate(
                seed_candidate,
                candidate_id=f"{seed_candidate.candidate_id}_boundary_mut",
                duration=max(1.5, seed_candidate.duration * 0.92),
                acceleration=seed_candidate.acceleration * 0.82,
                measurement_std=seed_candidate.measurement_std * 1.10,
                sampler_name="boundary_mutation",
                parent_candidate_id=seed_candidate.candidate_id,
            )
        )
    return candidates


def _archive_mutation_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    seeds = _lhs_sampler(objective, backend_id, budget)
    candidates = []
    for index, seed_candidate in enumerate(seeds):
        candidates.append(
            _replace_candidate(
                seed_candidate,
                candidate_id=f"{seed_candidate.candidate_id}_archive_mut",
                duration=seed_candidate.duration * 1.08,
                measurement_std=seed_candidate.measurement_std * 0.95,
                acceleration=seed_candidate.acceleration + 0.04 * ((index % 2) * 2 - 1),
                sampler_name="archive_mutation",
                parent_candidate_id=seed_candidate.candidate_id,
            )
        )
    return candidates


def _stress_mutation_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    seeds = _random_sampler(objective, backend_id, budget, random.Random(99))
    candidates = []
    for seed_candidate in seeds:
        candidates.append(
            _replace_candidate(
                seed_candidate,
                candidate_id=f"{seed_candidate.candidate_id}_stress_mut",
                duration=max(1.4, seed_candidate.duration * 0.88),
                measurement_std=seed_candidate.measurement_std * 1.35,
                acceleration=seed_candidate.acceleration * 0.9,
                sampler_name="stress_mutation",
                parent_candidate_id=seed_candidate.candidate_id,
            )
        )
    return candidates


def _replace_candidate(
    candidate: BackendCandidateSpec,
    *,
    candidate_id: str | None = None,
    seed: int | None = None,
    duration: float | None = None,
    measurement_std: float | None = None,
    initial_velocity: float | None = None,
    acceleration: float | None = None,
    sampler_name: str,
    parent_candidate_id: str,
) -> BackendCandidateSpec:
    provenance = dict(candidate.provenance)
    provenance["sampler_name"] = sampler_name
    provenance["parent_candidate_id"] = parent_candidate_id
    return BackendCandidateSpec(
        candidate_id=candidate_id or candidate.candidate_id,
        scenario_id=candidate.scenario_id,
        scenario_family=candidate.scenario_family,
        target_class=candidate.target_class,
        difficulty_tier=candidate.difficulty_tier,
        seed=seed if seed is not None else candidate.seed,
        duration=duration if duration is not None else candidate.duration,
        sample_period=candidate.sample_period,
        initial_position=candidate.initial_position,
        initial_velocity=initial_velocity if initial_velocity is not None else candidate.initial_velocity,
        acceleration=acceleration if acceleration is not None else candidate.acceleration,
        measurement_std=measurement_std if measurement_std is not None else candidate.measurement_std,
        switch_time=candidate.switch_time,
        acceleration_after_switch=candidate.acceleration_after_switch,
        drag_coefficient=candidate.drag_coefficient,
        density_scale=candidate.density_scale,
        wind_bias=candidate.wind_bias,
        input_deck_hash=candidate.input_deck_hash,
        longitudinal_command=candidate.longitudinal_command,
        provenance=provenance,
    )
