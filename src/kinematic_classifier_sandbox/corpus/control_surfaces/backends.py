from __future__ import annotations

import random
from math import cos, pi, sin

from ...trajectory_generator import _make_manual_trajectory
from ...utils.math import _clamp
from .contracts import ControlSurfaceBackend, ControlSurfaceMetadata, TrajectoryCandidate


def _metadata(
    *,
    backend_id: str,
    display_name: str,
    control_surface_type: str,
    control_variables: tuple[str, ...],
    best_use: str,
    lift_to_3d: str,
    supports_ppo: bool,
) -> ControlSurfaceMetadata:
    return ControlSurfaceMetadata(
        backend_id=backend_id,
        display_name=display_name,
        control_surface_type=control_surface_type,
        state_variables=("position", "velocity", "acceleration"),
        control_variables=control_variables,
        constraints={"max_abs_velocity": 6.0, "max_abs_acceleration": 1.2, "max_abs_jerk": 2.5},
        supports={
            "random": True,
            "cem": True,
            "ppo": supports_ppo,
            "qd_archive": True,
            "posterior_target": True,
            "feature_target": True,
            "switching_witness": backend_id == "hybrid_mode_schedule",
            "smoothness_constraints": backend_id in {"jerk_sequence", "spline_knots", "stochastic_process"},
            "endpoint_ambiguity": backend_id in {"spline_knots", "hybrid_mode_schedule"},
            "3d_lift": True,
        },
        classifier_allowed_fields=("time", "observed_position", "derived_velocity", "derived_acceleration"),
        hidden_fields=("backend_id", "generator_params", "control_trace", "objective_score"),
        best_use=best_use,
        lift_to_3d=lift_to_3d,
    )


def _times(steps: int, dt: float) -> tuple[float, ...]:
    return tuple(index * dt for index in range(steps))


def _finite_difference(values: tuple[float, ...], dt: float) -> tuple[float, ...]:
    if len(values) < 2:
        return tuple(0.0 for _ in values)
    result = [0.0]
    result.extend((values[index] - values[index - 1]) / dt for index in range(1, len(values)))
    return tuple(result)


def _trajectory_from_acceleration(
    *,
    backend_id: str,
    candidate_id: str,
    seed: int,
    accelerations: tuple[float, ...],
    v0: float,
    x0: float,
    dt: float,
    measurement_std: float,
    params: dict[str, float],
    control_trace: dict[str, tuple[float, ...]],
) -> TrajectoryCandidate:
    rng = random.Random(seed + 991)
    positions: list[float] = [x0]
    velocities: list[float] = [v0]
    for index in range(1, len(accelerations)):
        previous_v = velocities[-1]
        acceleration = accelerations[index - 1]
        velocities.append(previous_v + acceleration * dt)
        positions.append(positions[-1] + previous_v * dt + 0.5 * acceleration * dt * dt)
    times = _times(len(accelerations), dt)
    measurements = tuple(position + rng.gauss(0.0, measurement_std) for position in positions)
    trajectory = _make_manual_trajectory(
        trajectory_id=candidate_id,
        true_class="constant_acceleration",
        tier="boundary_v1",
        scenario_family="control_surface_backend_sweep",
        measurements=measurements,
        times=times,
        true_position=tuple(positions),
        true_velocity=tuple(velocities),
        true_acceleration=accelerations,
        measurement_std=measurement_std,
        outlier_indices=[],
        seed=seed,
        generator_parameters={
            "backend_id": backend_id,
            "control_surface_type": backend_id,
            "classifier_must_ignore": ("backend_id", "generator_params", "control_trace", "objective_score"),
            **params,
        },
    )
    return TrajectoryCandidate(
        candidate_id=candidate_id,
        backend_id=backend_id,
        trajectory=trajectory,
        params=params,
        control_trace=control_trace,
        generation_metadata={"classifier_allowed_fields": ("time", "observed_position", "derived_velocity", "derived_acceleration")},
    )


class _BaseSurface:
    backend_id = "base"

    def metadata(self) -> ControlSurfaceMetadata:
        raise NotImplementedError

    def sample_params(self, seed: int, *, accel_magnitude: float | None = None) -> dict[str, float]:
        rng = random.Random(seed)
        magnitude = rng.uniform(0.0, 0.45) if accel_magnitude is None else _clamp(accel_magnitude, 0.0, 0.45)
        return {
            "accel_magnitude": magnitude,
            "sign": -1.0 if rng.random() < 0.5 else 1.0,
            "v0": rng.uniform(0.45, 1.25),
            "x0": rng.uniform(-0.25, 0.25),
            "dt": rng.uniform(0.18, 0.28),
            "steps": float(rng.randint(14, 22)),
            "measurement_std": rng.uniform(0.005, 0.025),
            "shape": rng.uniform(0.2, 0.8),
        }

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate:
        raise NotImplementedError

    def _common(self, params: dict[str, float]) -> tuple[int, float, float, float, float, float]:
        steps = max(8, int(params["steps"]))
        dt = float(params["dt"])
        accel = float(params["sign"]) * float(params["accel_magnitude"])
        return steps, dt, accel, float(params["v0"]), float(params["x0"]), float(params["measurement_std"])


class DirectKinematicParamsBackend(_BaseSurface):
    backend_id = "direct_kinematic_params"

    def metadata(self) -> ControlSurfaceMetadata:
        return _metadata(
            backend_id=self.backend_id,
            display_name="Direct kinematic parameters",
            control_surface_type="static_parameters",
            control_variables=("x0", "v0", "a0", "duration", "noise_level"),
            best_use="Readable unit witnesses and class-validity probes.",
            lift_to_3d="Lift scalar position, velocity, and acceleration to vectors plus frame metadata.",
            supports_ppo=False,
        )

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate:
        steps, dt, accel, v0, x0, measurement_std = self._common(params)
        accelerations = tuple(accel for _ in range(steps))
        return _trajectory_from_acceleration(
            backend_id=self.backend_id,
            candidate_id=candidate_id,
            seed=seed,
            accelerations=accelerations,
            v0=v0,
            x0=x0,
            dt=dt,
            measurement_std=measurement_std,
            params=params,
            control_trace={"acceleration_command": accelerations},
        )


class AccelerationSequenceBackend(_BaseSurface):
    backend_id = "acceleration_sequence"

    def metadata(self) -> ControlSurfaceMetadata:
        return _metadata(
            backend_id=self.backend_id,
            display_name="Acceleration action sequence",
            control_surface_type="sequential_control",
            control_variables=("acceleration_command",),
            best_use="PPO/CEM control optimization and posterior-boundary witnesses.",
            lift_to_3d="Lift acceleration command to body-frame or inertial acceleration vector.",
            supports_ppo=True,
        )

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate:
        steps, dt, accel, v0, x0, measurement_std = self._common(params)
        shape = float(params["shape"])
        accelerations = tuple(accel * (1.0 + 0.18 * sin(2.0 * pi * index / max(steps - 1, 1)) * shape) for index in range(steps))
        return _trajectory_from_acceleration(
            backend_id=self.backend_id,
            candidate_id=candidate_id,
            seed=seed,
            accelerations=accelerations,
            v0=v0,
            x0=x0,
            dt=dt,
            measurement_std=measurement_std,
            params=params,
            control_trace={"acceleration_command": accelerations},
        )


class JerkSequenceBackend(_BaseSurface):
    backend_id = "jerk_sequence"

    def metadata(self) -> ControlSurfaceMetadata:
        return _metadata(
            backend_id=self.backend_id,
            display_name="Jerk action sequence",
            control_surface_type="sequential_control",
            control_variables=("jerk_command",),
            best_use="Smooth onset and weak-acceleration ambiguity witnesses.",
            lift_to_3d="Lift jerk to vector jerk or control-surface-induced acceleration derivative.",
            supports_ppo=True,
        )

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate:
        steps, dt, accel, v0, x0, measurement_std = self._common(params)
        accelerations = tuple(accel * (0.55 + 0.45 * (index / max(steps - 1, 1))) for index in range(steps))
        jerks = _finite_difference(accelerations, dt)
        return _trajectory_from_acceleration(
            backend_id=self.backend_id,
            candidate_id=candidate_id,
            seed=seed,
            accelerations=accelerations,
            v0=v0,
            x0=x0,
            dt=dt,
            measurement_std=measurement_std,
            params=params,
            control_trace={"jerk_command": jerks, "acceleration_command": accelerations},
        )


class SplineKnotsBackend(_BaseSurface):
    backend_id = "spline_knots"

    def metadata(self) -> ControlSurfaceMetadata:
        return _metadata(
            backend_id=self.backend_id,
            display_name="Spline knots",
            control_surface_type="compact_smooth_parameters",
            control_variables=("position_knot", "velocity_knot", "acceleration_knot"),
            best_use="Compact smooth CEM search, endpoint ambiguity, and feature-region targeting.",
            lift_to_3d="Lift knots to vector-valued splines with frame-aware constraints.",
            supports_ppo=False,
        )

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate:
        steps, dt, accel, v0, x0, measurement_std = self._common(params)
        shape = float(params["shape"])
        times = _times(steps, dt)
        duration = times[-1] if times else dt
        positions = tuple(x0 + v0 * time + 0.5 * accel * time * time + 0.025 * shape * sin(pi * time / max(duration, dt)) for time in times)
        velocities = _finite_difference(positions, dt)
        accelerations = _finite_difference(velocities, dt)
        return _trajectory_from_acceleration(
            backend_id=self.backend_id,
            candidate_id=candidate_id,
            seed=seed,
            accelerations=accelerations,
            v0=v0,
            x0=x0,
            dt=dt,
            measurement_std=measurement_std,
            params=params,
            control_trace={"spline_acceleration": accelerations},
        )


class HybridModeScheduleBackend(_BaseSurface):
    backend_id = "hybrid_mode_schedule"

    def metadata(self) -> ControlSurfaceMetadata:
        return _metadata(
            backend_id=self.backend_id,
            display_name="Hybrid mode schedule",
            control_surface_type="latent_mode_schedule",
            control_variables=("mode_id", "switch_time", "mode_strength"),
            best_use="Switching, transition-matrix, IMM, and future RBPF witnesses.",
            lift_to_3d="Lift mode schedule to vector dynamics modes or aerodynamic regimes.",
            supports_ppo=True,
        )

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate:
        steps, dt, accel, v0, x0, measurement_std = self._common(params)
        switch_index = max(2, min(steps - 3, int(steps * (0.35 + 0.30 * float(params["shape"])))))
        accelerations = tuple(0.0 if index < switch_index else accel * 1.65 for index in range(steps))
        mode_trace = tuple(0.0 if index < switch_index else 1.0 for index in range(steps))
        return _trajectory_from_acceleration(
            backend_id=self.backend_id,
            candidate_id=candidate_id,
            seed=seed,
            accelerations=accelerations,
            v0=v0,
            x0=x0,
            dt=dt,
            measurement_std=measurement_std,
            params={**params, "switch_index": float(switch_index)},
            control_trace={"mode_id": mode_trace, "acceleration_command": accelerations},
        )


class StochasticProcessBackend(_BaseSurface):
    backend_id = "stochastic_process"

    def metadata(self) -> ControlSurfaceMetadata:
        return _metadata(
            backend_id=self.backend_id,
            display_name="Stochastic process",
            control_surface_type="stochastic_dynamics",
            control_variables=("acceleration_process_seed", "mean_reversion", "process_noise"),
            best_use="Generalization, calibration, prior-sensitivity, and anti-template stress.",
            lift_to_3d="Lift process to vector stochastic acceleration or bounded stochastic jerk.",
            supports_ppo=False,
        )

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate:
        steps, dt, accel, v0, x0, measurement_std = self._common(params)
        rng = random.Random(seed + 313)
        value = accel
        accelerations: list[float] = []
        for _ in range(steps):
            value = 0.82 * value + 0.18 * accel + rng.gauss(0.0, 0.018)
            accelerations.append(_clamp(value, -0.65, 0.65))
        return _trajectory_from_acceleration(
            backend_id=self.backend_id,
            candidate_id=candidate_id,
            seed=seed,
            accelerations=tuple(accelerations),
            v0=v0,
            x0=x0,
            dt=dt,
            measurement_std=measurement_std,
            params=params,
            control_trace={"stochastic_acceleration": tuple(accelerations)},
        )


def default_control_surface_backends() -> tuple[ControlSurfaceBackend, ...]:
    return (
        DirectKinematicParamsBackend(),
        AccelerationSequenceBackend(),
        JerkSequenceBackend(),
        SplineKnotsBackend(),
        HybridModeScheduleBackend(),
        StochasticProcessBackend(),
    )

