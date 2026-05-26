from __future__ import annotations

import random

from .gym import CorpusGymAction, CorpusGymEpisode, CorpusGymTarget


def _random_action(rng: random.Random, target: CorpusGymTarget, *, seed: int) -> CorpusGymAction:
    return CorpusGymAction(
        seed=seed,
        tier_name=target.target_tier or "realistic_v1",
        duration_scale=rng.uniform(0.75, 1.25),
        measurement_scale=rng.uniform(0.80, 1.30),
        irregularity_scale=rng.uniform(0.75, 1.35),
        outlier_scale=rng.uniform(0.75, 1.35),
        step_scale=rng.uniform(0.80, 1.20),
    )


def _doe_actions(target: CorpusGymTarget, *, base_seed: int) -> tuple[CorpusGymAction, ...]:
    grid = (
        (0.85, 0.90, 0.90, 0.85, 0.90),
        (0.95, 1.00, 1.10, 1.00, 0.95),
        (1.05, 1.10, 0.90, 1.20, 1.05),
        (1.15, 1.20, 1.25, 1.10, 1.10),
    )
    actions = []
    for index, (duration_scale, measurement_scale, irregularity_scale, outlier_scale, step_scale) in enumerate(grid):
        actions.append(
            CorpusGymAction(
                seed=base_seed + index,
                tier_name=target.target_tier or "realistic_v1",
                duration_scale=duration_scale,
                measurement_scale=measurement_scale,
                irregularity_scale=irregularity_scale,
                outlier_scale=outlier_scale,
                step_scale=step_scale,
            )
        )
    return tuple(actions)


def _episode_row(
    *,
    candidate_id: str,
    target: CorpusGymTarget,
    search_method: str,
    episode: CorpusGymEpisode,
) -> dict[str, object]:
    diagnostics = episode.diagnostics
    reward = episode.reward
    trajectory = episode.trajectory
    return {
        "candidate_id": candidate_id,
        "target_id": target.target_id,
        "target_type": target.target_type,
        "target_tier": target.target_tier or "",
        "target_class": target.class_name or "",
        "target_class_pair": " vs ".join(target.class_pair) if target.class_pair else "",
        "search_method": search_method,
        "seed": episode.action.seed,
        "tier_name": episode.action.tier_name,
        "duration_scale": episode.action.duration_scale,
        "measurement_scale": episode.action.measurement_scale,
        "irregularity_scale": episode.action.irregularity_scale,
        "outlier_scale": episode.action.outlier_scale,
        "step_scale": episode.action.step_scale,
        "trajectory_id": trajectory.trajectory_id,
        "generated_class": trajectory.true_class,
        "generated_tier": str(trajectory.generator_parameters.get("tier", "")),
        "class_validity": reward.class_validity,
        "feature_excitation": reward.feature_excitation,
        "coverage_gain": reward.coverage_gain,
        "boundary_closeness": reward.boundary_closeness,
        "classifier_stress": reward.classifier_stress,
        "prior_sensitivity": reward.prior_sensitivity,
        "leakage_penalty": reward.leakage_penalty,
        "physical_invalidity_penalty": reward.physical_invalidity_penalty,
        "total_utility": reward.total_utility,
        "duration": diagnostics["duration"],
        "position_range": diagnostics["position_range"],
        "speed_range": diagnostics["speed_range"],
        "acceleration_range": diagnostics["acceleration_range"],
        "monotonicity": diagnostics["monotonicity"],
        "sampling_irregularity": diagnostics["sampling_irregularity"],
        "num_samples": diagnostics["num_samples"],
    }
