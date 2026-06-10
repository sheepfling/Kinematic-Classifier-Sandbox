from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from ..gym import CorpusGymAction, CorpusGymTarget
from ..objectives import CorpusObjectiveSpec
from ..trajectory_backend_contract import TrajectoryRun
from ..trajectory_backend_contract_utils import validate_trajectory_run
from .backend_adapter_proof_types import AdapterExecutionRecord, BackendCandidateSpec


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]


def _primary_class(objective: CorpusObjectiveSpec) -> str:
    if objective.target_class is not None:
        return objective.target_class
    assert objective.target_class_pair is not None
    return objective.target_class_pair[0]


def objective_to_corpus_gym_target(objective: CorpusObjectiveSpec) -> CorpusGymTarget:
    if objective.target_feature_excitation:
        return CorpusGymTarget(
            target_id=objective.objective_id,
            target_type="target_feature_cell",
            description=objective.description,
            class_name=_primary_class(objective),
            class_pair=objective.target_class_pair,
            feature_constraints=objective.target_feature_excitation or None,
            target_tier=objective.target_difficulty,
            target_prior_sensitivity=objective.target_posterior_entropy,
        )
    if objective.target_class_pair is not None:
        return CorpusGymTarget(
            target_id=objective.objective_id,
            target_type="target_class_pair",
            description=objective.description,
            class_pair=objective.target_class_pair,
            class_name=_primary_class(objective),
            target_tier=objective.target_difficulty,
            target_prior_sensitivity=objective.target_posterior_entropy,
        )
    return CorpusGymTarget(
        target_id=objective.objective_id,
        target_type="target_class",
        description=objective.description,
        class_name=objective.target_class,
        target_tier=objective.target_difficulty,
        target_prior_sensitivity=objective.target_posterior_entropy,
    )


def candidate_to_corpus_gym_action(candidate: BackendCandidateSpec) -> CorpusGymAction:
    duration_scale = max(candidate.duration / 2.0, 0.5)
    measurement_scale = max(candidate.measurement_std / 0.03, 0.5)
    irregularity_scale = (
        1.35
        if candidate.difficulty_tier in {"stress_v1", "adversarial_v1"}
        else 1.15
        if candidate.difficulty_tier in {"boundary_v1", "realistic_v1"}
        else 1.0
    )
    outlier_scale = 1.35 if candidate.scenario_family in {"outlier_stress", "shared_boundary_case"} else 1.0
    step_scale = max(0.5 / max(candidate.sample_period, 1e-6), 0.6)
    return CorpusGymAction(
        seed=candidate.seed,
        tier_name=candidate.difficulty_tier,
        duration_scale=duration_scale,
        measurement_scale=measurement_scale,
        irregularity_scale=irregularity_scale,
        outlier_scale=outlier_scale,
        step_scale=step_scale,
        metadata={
            "candidate_id": candidate.candidate_id,
            "objective_id": candidate.provenance.get("objective_id", ""),
            "proposed_backend_id": candidate.provenance.get("backend_id", ""),
            "sampler_name": candidate.provenance.get("sampler_name", ""),
            "parent_candidate_id": candidate.provenance.get("parent_candidate_id", ""),
            "environment_id": candidate.provenance.get("environment_id", ""),
            "scenario_family": candidate.scenario_family,
            "target_class": candidate.target_class,
        },
    )


def _episode_to_execution(candidate: BackendCandidateSpec, episode) -> AdapterExecutionRecord:
    trajectory = episode.trajectory
    input_bundle = {
        "target_id": episode.target.target_id,
        "target_type": episode.target.target_type,
        "action": {
            "seed": episode.action.seed,
            "tier_name": episode.action.tier_name,
            "duration_scale": episode.action.duration_scale,
            "measurement_scale": episode.action.measurement_scale,
            "irregularity_scale": episode.action.irregularity_scale,
            "outlier_scale": episode.action.outlier_scale,
            "step_scale": episode.action.step_scale,
            "metadata": dict(episode.action.metadata),
        },
    }
    raw_output = {
        "times": trajectory.times,
        "measurements": trajectory.measurements,
        "true_position": trajectory.true_position,
        "true_velocity": trajectory.true_velocity,
        "true_acceleration": trajectory.true_acceleration,
    }
    trajectory_run = TrajectoryRun(
        run_id=f"corpus_gym_{candidate.candidate_id}",
        backend_id="corpus_gym",
        scenario_id=trajectory.scenario_id,
        seed=trajectory.seed,
        success=True,
        failure_reason=None,
        times=tuple(float(value) for value in trajectory.times),
        truth_state={
            "position": tuple(float(value) for value in trajectory.true_position or ()),
            "velocity": tuple(float(value) for value in trajectory.true_velocity or ()),
            "acceleration": tuple(float(value) for value in trajectory.true_acceleration or ()),
        },
        observations={"position": tuple(float(value) for value in trajectory.measurements)},
        events=(
            {
                "time": trajectory.times[-1] if trajectory.times else 0.0,
                "event_type": "termination",
                "event_value": "corpus_gym_end",
            },
        ),
        metadata={
            "adapter_family": "corpus_gym",
            "candidate_id": candidate.candidate_id,
            "measurement_dim": 1,
            "coordinate_frame": "scalar_line",
            "search_provenance": {
                **candidate.provenance,
                "target_id": episode.target.target_id,
                "target_type": episode.target.target_type,
            },
            "reward": {
                "class_validity": episode.reward.class_validity,
                "feature_excitation": episode.reward.feature_excitation,
                "coverage_gain": episode.reward.coverage_gain,
                "boundary_closeness": episode.reward.boundary_closeness,
                "classifier_stress": episode.reward.classifier_stress,
                "prior_sensitivity": episode.reward.prior_sensitivity,
                "leakage_penalty": episode.reward.leakage_penalty,
                "physical_invalidity_penalty": episode.reward.physical_invalidity_penalty,
                "total_utility": episode.reward.total_utility,
            },
        },
    )
    return AdapterExecutionRecord(
        backend_id="corpus_gym",
        candidate_id=candidate.candidate_id,
        cache_key=_stable_hash(input_bundle),
        cache_hit=False,
        input_bundle=input_bundle,
        raw_output=raw_output,
        trajectory_run=trajectory_run,
        validation_errors=tuple(validate_trajectory_run(trajectory_run)),
    )
