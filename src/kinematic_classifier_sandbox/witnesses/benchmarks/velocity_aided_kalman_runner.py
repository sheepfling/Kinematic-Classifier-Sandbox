from __future__ import annotations

import random
from typing import NamedTuple

from ...analysis.common_dataset_comparison import (
    SharedDynamicsTrajectory,
    generate_shared_dynamics_dataset,
)
from .kalman_filter_bank import (
    KalmanClassificationRun,
    KalmanModelSpec,
    KalmanTrajectory,
    run_kalman_filter_bank,
)
from .velocity_aided_kalman_contracts import VelocityAidedComparisonResult, VelocityAidedRow, VelocityAidedTrace


class RunModeResult(NamedTuple):
    run: KalmanClassificationRun
    velocity_measurements_used: tuple[float, ...]


def _shared_kalman_trajectory(trajectory: SharedDynamicsTrajectory) -> KalmanTrajectory:
    return KalmanTrajectory(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        seed=trajectory.seed,
        times=trajectory.times,
        measurements=trajectory.measurements,
        true_position=trajectory.true_position,
        true_velocity=trajectory.true_velocity,
        true_acceleration=trajectory.true_acceleration,
    )


def _kalman_shared_model_specs() -> tuple[KalmanModelSpec, ...]:
    return (
        KalmanModelSpec("constant_velocity_quiet", "constant_velocity", 2, 0.14, 0.20, 5.0, 0.375),
        KalmanModelSpec("constant_velocity_rough", "constant_velocity", 2, 0.22, 0.20, 5.5, 0.125),
        KalmanModelSpec("constant_acceleration_quiet", "constant_acceleration", 3, 0.24, 0.20, 6.0, 0.375),
        KalmanModelSpec("constant_acceleration_rough", "constant_acceleration", 3, 0.34, 0.20, 6.5, 0.125),
    )


def _synthesized_velocity_measurements(trajectory: SharedDynamicsTrajectory, *, sigma: float) -> tuple[float, ...]:
    rng = random.Random(trajectory.seed + 9001)
    return tuple(value + rng.gauss(0.0, sigma) for value in trajectory.true_velocity)


def _run_mode(
    trajectory: SharedDynamicsTrajectory,
    *,
    measurement_mode: str,
) -> RunModeResult:
    velocity_sigma = 0.12
    velocity_measurements = _synthesized_velocity_measurements(trajectory, sigma=velocity_sigma)
    kwargs = {
        "robust_measurement_update": True,
        "adaptive_process_noise": True,
        "derived_velocity_observation": False,
        "derived_acceleration_observation": False,
    }
    if measurement_mode == "position_only":
        velocity_measurements_used = tuple(0.0 for _ in trajectory.times)
    elif measurement_mode == "position_plus_direct_velocity":
        kwargs["velocity_measurements"] = velocity_measurements
        kwargs["velocity_measurement_sigma"] = velocity_sigma
        velocity_measurements_used = velocity_measurements
    else:
        raise ValueError(f"Unsupported measurement mode: {measurement_mode}")
    run = run_kalman_filter_bank(_shared_kalman_trajectory(trajectory), _kalman_shared_model_specs(), **kwargs)
    return RunModeResult(run=run, velocity_measurements_used=velocity_measurements_used)


def _mode_accuracy(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    measurement_mode: str,
) -> VelocityAidedRow:
    runs = [_run_mode(trajectory, measurement_mode=measurement_mode).run for trajectory in trajectories]

    def _accuracy(scenario_name: str | None = None) -> float:
        selected = [run for run, trajectory in zip(runs, trajectories) if scenario_name is None or trajectory.scenario_name == scenario_name]
        return sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected) / len(selected)

    return VelocityAidedRow(
        measurement_mode=measurement_mode,
        overall_accuracy=_accuracy(),
        endpoint_match_accuracy=_accuracy("endpoint_match"),
        short_accuracy=_accuracy("short"),
        short_noisy_accuracy=_accuracy("short_noisy"),
        outlier_accuracy=_accuracy("outlier"),
    )


def analyze_velocity_aided_kalman_comparison(*, seed: int = 7, trajectories_per_case: int = 8) -> VelocityAidedComparisonResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    measurement_modes = ("position_only", "position_plus_direct_velocity")
    rows = tuple(_mode_accuracy(trajectories, measurement_mode=mode) for mode in measurement_modes)
    representative_trajectories = []
    for scenario_name, class_name in (("short_noisy", "constant_acceleration"), ("endpoint_match", "constant_acceleration"), ("outlier", "constant_velocity")):
        representative_trajectories.append(next(trajectory for trajectory in trajectories if trajectory.scenario_name == scenario_name and trajectory.true_class == class_name))
    traces: list[VelocityAidedTrace] = []
    for trajectory in representative_trajectories:
        for mode in measurement_modes:
            run_result = _run_mode(trajectory, measurement_mode=mode)
            run = run_result.run
            velocity_measurements = run_result.velocity_measurements_used
            traces.append(
                VelocityAidedTrace(
                    measurement_mode=mode,
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    true_class=trajectory.true_class,
                    final_predicted_class=run.final_predicted_class,
                    final_confidence=run.final_confidence,
                    times=trajectory.times,
                    measurements=trajectory.measurements,
                    velocity_measurements=velocity_measurements,
                    true_class_posterior=tuple(step.posterior_weights[trajectory.true_class] for step in run.steps),
                )
            )
    return VelocityAidedComparisonResult(rows=rows, traces=tuple(traces))
