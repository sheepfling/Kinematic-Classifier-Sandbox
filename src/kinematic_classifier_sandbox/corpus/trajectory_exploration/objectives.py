from __future__ import annotations

from dataclasses import asdict

from ..gym import default_corpus_gym_targets
from ..gym_types import CorpusGymEpisode
from .contracts import (
    TrajectoryExplorationEvaluation,
    TrajectoryExplorationObjective,
    TrajectoryExplorationProposal,
)


def _weights(
    *,
    validity: float,
    excitation: float,
    coverage: float,
    geometry: float,
    stress: float,
    prior: float,
    leakage: float,
    physical: float,
) -> dict[str, float]:
    return {
        "validity": validity,
        "feature_excitation": excitation,
        "coverage_gain": coverage,
        "geometry_score": geometry,
        "classifier_stress": stress,
        "prior_sensitivity": prior,
        "leakage_penalty": leakage,
        "physical_invalidity_penalty": physical,
    }


def default_trajectory_exploration_objectives() -> tuple[TrajectoryExplorationObjective, ...]:
    targets = {target.target_id: target for target in default_corpus_gym_targets()}
    return (
        TrajectoryExplorationObjective(
            objective_id="feature_cell_repair",
            mode="corpus_growth",
            geometry_target="fill_underexcited_feature_cells",
            description="Fill weak feature-excitation cells with valid high-acceleration, low-monotonicity trajectories.",
            target=targets["target_feature_cell_high_accel_low_monotonicity"],
            reward_weights=_weights(validity=0.30, excitation=0.35, coverage=0.20, geometry=0.15, stress=0.05, prior=0.02, leakage=0.14, physical=0.14),
            thresholds={"min_class_validity": 0.45, "min_feature_excitation": 0.65, "max_leakage_penalty": 0.35},
            evaluation_budget=18,
        ),
        TrajectoryExplorationObjective(
            objective_id="class_pair_boundary_refinement",
            mode="class_pair_refinement",
            geometry_target="create_ambiguous_boundary_trajectories",
            description="Search CV/CA boundary trajectories that tighten the class-pair witness set.",
            target=targets["target_class_pair_cv_vs_ca_boundary"],
            reward_weights=_weights(validity=0.24, excitation=0.18, coverage=0.16, geometry=0.28, stress=0.12, prior=0.08, leakage=0.12, physical=0.12),
            thresholds={"min_class_validity": 0.45, "min_boundary_closeness": 0.50, "max_leakage_penalty": 0.40},
            evaluation_budget=18,
        ),
        TrajectoryExplorationObjective(
            objective_id="prior_flip_witness_search",
            mode="adversarial",
            geometry_target="maximize_prior_flip_witness",
            description="Target prior-sensitive CV/braking witnesses under a fixed budget.",
            target=targets["target_prior_sensitivity_small_flip"],
            reward_weights=_weights(validity=0.20, excitation=0.10, coverage=0.12, geometry=0.14, stress=0.16, prior=0.28, leakage=0.10, physical=0.12),
            thresholds={"min_class_validity": 0.45, "min_prior_flip_witness_score": 0.45, "max_leakage_penalty": 0.45},
            evaluation_budget=18,
        ),
    )


def evaluate_proposal(
    objective: TrajectoryExplorationObjective,
    proposal: TrajectoryExplorationProposal,
    episode: CorpusGymEpisode,
) -> TrajectoryExplorationEvaluation:
    reward = episode.reward
    diagnostics = episode.diagnostics
    feature_dependency_stress = min(
        1.0,
        0.45 * float(diagnostics.get("sampling_irregularity", 0.0))
        + 0.25 * float(diagnostics.get("outlier_score", 0.0))
        + 0.30 * (1.0 - float(diagnostics.get("monotonicity", 1.0))),
    )
    geometry_terms = {
        "fill_underexcited_feature_cells": reward.feature_excitation,
        "create_ambiguous_boundary_trajectories": reward.boundary_closeness,
        "maximize_prior_flip_witness": reward.prior_sensitivity,
        "sweep_feature_row_novelty": max(0.0, 0.65 * reward.feature_excitation + 0.35 * reward.coverage_gain),
        "discover_novel_feature_class_region": max(0.0, 0.40 * reward.feature_excitation + 0.30 * reward.classifier_stress + 0.30 * reward.coverage_gain),
        "reduce_leakage_shortcuts": max(0.0, 1.0 - reward.leakage_penalty),
        "separate_class_pair": max(0.0, reward.class_validity - reward.classifier_stress),
        "break_classifier_family": reward.classifier_stress,
    }
    geometry_score = geometry_terms.get(objective.geometry_target, reward.coverage_gain)
    feature_cell_coverage_gain = reward.feature_excitation if "feature" in objective.geometry_target else max(0.0, reward.coverage_gain * 0.5)
    class_pair_overlap_reduction = max(0.0, reward.class_validity - reward.boundary_closeness * 0.25 - reward.leakage_penalty * 0.20)
    pairwise_auc_gain = max(0.0, 0.55 * reward.class_validity + 0.30 * reward.feature_excitation + 0.15 * reward.coverage_gain - 0.20 * reward.leakage_penalty)
    pca_margin_gain = max(0.0, 0.40 * float(diagnostics.get("position_range", 0.0)) + 0.35 * float(diagnostics.get("speed_range", 0.0)) + 0.25 * float(diagnostics.get("acceleration_range", 0.0)))
    confusion_witness_score = max(reward.classifier_stress, reward.boundary_closeness)
    prior_flip_witness_score = reward.prior_sensitivity
    weights = objective.reward_weights
    total_utility = (
        weights["validity"] * reward.class_validity
        + weights["feature_excitation"] * reward.feature_excitation
        + weights["coverage_gain"] * reward.coverage_gain
        + weights["geometry_score"] * geometry_score
        + weights["classifier_stress"] * reward.classifier_stress
        + weights["prior_sensitivity"] * reward.prior_sensitivity
        - weights["leakage_penalty"] * reward.leakage_penalty
        - weights["physical_invalidity_penalty"] * reward.physical_invalidity_penalty
    )
    return TrajectoryExplorationEvaluation(
        proposal_id=proposal.proposal_id,
        backend_id=proposal.backend_id,
        objective_id=objective.objective_id,
        iteration=proposal.iteration,
        candidate_index=proposal.candidate_index,
        target_id=objective.target.target_id,
        trajectory_id=episode.trajectory.trajectory_id,
        true_class=episode.trajectory.true_class,
        total_utility=total_utility,
        class_validity=reward.class_validity,
        feature_excitation=reward.feature_excitation,
        coverage_gain=reward.coverage_gain,
        boundary_closeness=reward.boundary_closeness,
        classifier_stress=reward.classifier_stress,
        prior_sensitivity=reward.prior_sensitivity,
        leakage_penalty=reward.leakage_penalty,
        physical_invalidity_penalty=reward.physical_invalidity_penalty,
        feature_cell_coverage_gain=feature_cell_coverage_gain,
        class_pair_overlap_reduction=class_pair_overlap_reduction,
        pairwise_auc_gain=pairwise_auc_gain,
        pca_margin_gain=pca_margin_gain,
        confusion_witness_score=confusion_witness_score,
        feature_dependency_stress=feature_dependency_stress,
        prior_flip_witness_score=prior_flip_witness_score,
        geometry_score=geometry_score,
        diagnostics={
            "tier_name": proposal.action.tier_name,
            "duration_scale": proposal.action.duration_scale,
            "measurement_scale": proposal.action.measurement_scale,
            "irregularity_scale": proposal.action.irregularity_scale,
            "outlier_scale": proposal.action.outlier_scale,
            "step_scale": proposal.action.step_scale,
            "target_type": objective.target.target_type,
            "target_class": objective.target.class_name or "",
            "target_class_pair": " vs ".join(objective.target.class_pair) if objective.target.class_pair else "",
        },
    )


def trajectory_exploration_objective_schema() -> dict[str, object]:
    return {
        "title": "TrajectoryExplorationObjective",
        "type": "object",
        "required": ["objective_id", "mode", "geometry_target", "description", "target", "reward_weights", "thresholds", "evaluation_budget"],
        "properties": {
            "objective_id": {"type": "string"},
            "mode": {"type": "string"},
            "geometry_target": {"type": "string"},
            "description": {"type": "string"},
            "target": {"type": "object"},
            "reward_weights": {"type": "object"},
            "thresholds": {"type": "object"},
            "evaluation_budget": {"type": "integer", "minimum": 1},
            "backend_constraints": {"type": "object"},
            "classifier_family": {"type": ["string", "null"]},
        },
    }


def trajectory_exploration_evaluation_schema() -> dict[str, object]:
    return {
        "title": "TrajectoryExplorationEvaluation",
        "type": "object",
        "required": ["proposal_id", "backend_id", "objective_id", "trajectory_id", "true_class", "total_utility", "selected"],
        "properties": {
            "proposal_id": {"type": "string"},
            "backend_id": {"type": "string"},
            "objective_id": {"type": "string"},
            "trajectory_id": {"type": "string"},
            "true_class": {"type": "string"},
            "total_utility": {"type": "number"},
            "class_validity": {"type": "number"},
            "feature_excitation": {"type": "number"},
            "coverage_gain": {"type": "number"},
            "boundary_closeness": {"type": "number"},
            "classifier_stress": {"type": "number"},
            "prior_sensitivity": {"type": "number"},
            "leakage_penalty": {"type": "number"},
            "physical_invalidity_penalty": {"type": "number"},
            "feature_cell_coverage_gain": {"type": "number"},
            "class_pair_overlap_reduction": {"type": "number"},
            "pairwise_auc_gain": {"type": "number"},
            "pca_margin_gain": {"type": "number"},
            "confusion_witness_score": {"type": "number"},
            "feature_dependency_stress": {"type": "number"},
            "prior_flip_witness_score": {"type": "number"},
            "geometry_score": {"type": "number"},
            "selected": {"type": "boolean"},
        },
    }


def objective_as_row(objective: TrajectoryExplorationObjective) -> dict[str, object]:
    payload = asdict(objective)
    payload["target_id"] = objective.target.target_id
    payload["target_type"] = objective.target.target_type
    return payload
