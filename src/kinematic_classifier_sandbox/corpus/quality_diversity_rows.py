from __future__ import annotations

from typing import NamedTuple

from ..utils.categorical import bucket_thresholds


def _bucket(value: float, thresholds: tuple[float, float]) -> str:
    return bucket_thresholds(value, thresholds)


class ArchiveCellId(NamedTuple):
    generated_class: str
    target_tier: str
    duration_bucket: str
    acceleration_bucket: str
    monotonicity_bucket: str


def _archive_cell_id(row: dict[str, object]) -> ArchiveCellId:
    return ArchiveCellId(
        generated_class=str(row["generated_class"]),
        target_tier=str(row["target_tier"]),
        duration_bucket=_bucket(float(row["duration"]), (6.0, 12.0)),
        acceleration_bucket=_bucket(float(row["acceleration_range"]), (0.35, 0.85)),
        monotonicity_bucket=_bucket(1.0 - float(row["monotonicity"]), (0.08, 0.22)),
    )


def _episode_row(
    *,
    iteration: int,
    candidate_id: str,
    target_id: str,
    target_type: str,
    target_tier: str,
    episode,
) -> dict[str, object]:
    reward = episode.reward
    diagnostics = episode.diagnostics
    return {
        "iteration": iteration,
        "candidate_id": candidate_id,
        "target_id": target_id,
        "target_type": target_type,
        "target_tier": target_tier,
        "trajectory_id": episode.trajectory.trajectory_id,
        "generated_class": episode.trajectory.true_class,
        "seed": episode.action.seed,
        "duration_scale": episode.action.duration_scale,
        "measurement_scale": episode.action.measurement_scale,
        "irregularity_scale": episode.action.irregularity_scale,
        "outlier_scale": episode.action.outlier_scale,
        "step_scale": episode.action.step_scale,
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
        "acceleration_range": diagnostics["acceleration_range"],
        "monotonicity": diagnostics["monotonicity"],
        "sampling_irregularity": diagnostics["sampling_irregularity"],
        "num_samples": diagnostics["num_samples"],
    }
