from __future__ import annotations

import hashlib
import json
import random
from math import sin
from typing import Any

from ...trajectory_series import KinematicSeries
from ..trajectory_backend_contract import (
    BackendContractDefinition,
    TrajectoryRun,
    default_backend_contract_definitions,
)
from ..trajectory_backend_contract_utils import validate_trajectory_run
from .backend_adapter_proof_types import AdapterExecutionRecord, BackendCandidateSpec


class TrajectoryBackendAdapter:
    def __init__(self, definition: BackendContractDefinition) -> None:
        self.definition = definition
        self.backend_id = definition.capabilities.backend_id
        self.family = definition.capabilities.family
        self._cache: dict[str, AdapterExecutionRecord] = {}

    def supports(self, candidate: BackendCandidateSpec) -> bool:
        raise NotImplementedError

    def prepare(self, candidate: BackendCandidateSpec) -> dict[str, Any]:
        raise NotImplementedError

    def run(self, candidate: BackendCandidateSpec) -> AdapterExecutionRecord:
        input_bundle = self.prepare(candidate)
        cache_key = _stable_input_hash(input_bundle)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return AdapterExecutionRecord(
                backend_id=cached.backend_id,
                candidate_id=cached.candidate_id,
                cache_key=cached.cache_key,
                cache_hit=True,
                input_bundle=cached.input_bundle,
                raw_output=cached.raw_output,
                trajectory_run=cached.trajectory_run,
                validation_errors=cached.validation_errors,
            )
        raw_output = self._simulate_raw_output(candidate, input_bundle)
        run = self.normalize_output(candidate, input_bundle, raw_output)
        validation_errors = tuple(validate_trajectory_run(run))
        record = AdapterExecutionRecord(
            backend_id=self.backend_id,
            candidate_id=candidate.candidate_id,
            cache_key=cache_key,
            cache_hit=False,
            input_bundle=input_bundle,
            raw_output=raw_output,
            trajectory_run=run,
            validation_errors=validation_errors,
        )
        self._cache[cache_key] = record
        return record

    def _simulate_raw_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def normalize_output(
        self,
        candidate: BackendCandidateSpec,
        input_bundle: dict[str, Any],
        raw_output: dict[str, Any],
    ) -> TrajectoryRun:
        raise NotImplementedError


def _stable_input_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _times_for_candidate(candidate: BackendCandidateSpec) -> tuple[float, ...]:
    steps = max(4, int(round(candidate.duration / max(candidate.sample_period, 1e-6)))) + 1
    return tuple(round(index * candidate.sample_period, 6) for index in range(steps))


def _deterministic_noise(seed: int, index: int, scale: float) -> float:
    rng = random.Random(seed * 997 + index * 37)
    return scale * (rng.random() - 0.5) * 2.0


def _parameter_truth(candidate: BackendCandidateSpec, times: tuple[float, ...]) -> KinematicSeries:
    positions: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []
    for time in times:
        positions.append(candidate.initial_position + candidate.initial_velocity * time + 0.5 * candidate.acceleration * time * time)
        velocities.append(candidate.initial_velocity + candidate.acceleration * time)
        accelerations.append(candidate.acceleration)
    return KinematicSeries(tuple(positions), tuple(velocities), tuple(accelerations))


def _switching_truth(candidate: BackendCandidateSpec, times: tuple[float, ...]) -> KinematicSeries:
    switch_time = candidate.switch_time if candidate.switch_time is not None else candidate.duration + 1.0
    accel_after = candidate.acceleration_after_switch if candidate.acceleration_after_switch is not None else candidate.acceleration
    positions: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []
    for time in times:
        if time <= switch_time:
            accel = candidate.acceleration
            position = candidate.initial_position + candidate.initial_velocity * time + 0.5 * accel * time * time
            velocity = candidate.initial_velocity + accel * time
        else:
            velocity_at_switch = candidate.initial_velocity + candidate.acceleration * switch_time
            position_at_switch = candidate.initial_position + candidate.initial_velocity * switch_time + 0.5 * candidate.acceleration * switch_time * switch_time
            dt = time - switch_time
            accel = accel_after
            position = position_at_switch + velocity_at_switch * dt + 0.5 * accel * dt * dt
            velocity = velocity_at_switch + accel * dt
        positions.append(position)
        velocities.append(velocity)
        accelerations.append(accel)
    return KinematicSeries(tuple(positions), tuple(velocities), tuple(accelerations))


class ParameterOnly1DAdapter(TrajectoryBackendAdapter):
    def supports(self, candidate: BackendCandidateSpec) -> bool:
        return candidate.scenario_family == "shared_boundary_case"

    def prepare(self, candidate: BackendCandidateSpec) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "scenario_id": candidate.scenario_id,
            "seed": candidate.seed,
            "duration": candidate.duration,
            "sample_period": candidate.sample_period,
            "initial_position": candidate.initial_position,
            "initial_velocity": candidate.initial_velocity,
            "acceleration": candidate.acceleration,
            "measurement_std": candidate.measurement_std,
        }

    def _simulate_raw_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any]) -> dict[str, Any]:
        times = _times_for_candidate(candidate)
        positions, velocities, accelerations = _parameter_truth(candidate, times)
        observations = tuple(position + _deterministic_noise(candidate.seed, index, candidate.measurement_std) for index, position in enumerate(positions))
        return {
            "times": times,
            "truth_state": {"position": positions, "velocity": velocities, "acceleration": accelerations},
            "observations": {"position": observations},
            "events": ({"time": times[-1], "event_type": "termination", "event_value": "nominal_end"},),
        }

    def normalize_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any], raw_output: dict[str, Any]) -> TrajectoryRun:
        return TrajectoryRun(
            run_id=f"{self.backend_id}_{candidate.candidate_id}",
            backend_id=self.backend_id,
            scenario_id=candidate.scenario_id,
            seed=candidate.seed,
            success=True,
            failure_reason=None,
            times=tuple(raw_output["times"]),
            truth_state=raw_output["truth_state"],
            observations=raw_output["observations"],
            events=tuple(raw_output["events"]),
            metadata={
                "adapter_family": self.family,
                "candidate_id": candidate.candidate_id,
                "measurement_dim": 1,
                "coordinate_frame": "scalar_line",
                "search_provenance": candidate.provenance,
            },
        )


class Controlled1DAdapter(TrajectoryBackendAdapter):
    def supports(self, candidate: BackendCandidateSpec) -> bool:
        return candidate.scenario_family in {"switching_case", "shared_boundary_case"}

    def prepare(self, candidate: BackendCandidateSpec) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "scenario_id": candidate.scenario_id,
            "seed": candidate.seed,
            "duration": candidate.duration,
            "sample_period": candidate.sample_period,
            "initial_position": candidate.initial_position,
            "initial_velocity": candidate.initial_velocity,
            "acceleration_before_switch": candidate.acceleration,
            "switch_time": candidate.switch_time,
            "acceleration_after_switch": candidate.acceleration_after_switch,
            "measurement_std": candidate.measurement_std,
        }

    def _simulate_raw_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any]) -> dict[str, Any]:
        times = _times_for_candidate(candidate)
        if candidate.scenario_family == "switching_case" and candidate.switch_time is None:
            return {"failure_reason": "missing_switch_time", "times": times}
        positions, velocities, accelerations = _switching_truth(candidate, times)
        observations = tuple(position + _deterministic_noise(candidate.seed + 100, index, candidate.measurement_std) for index, position in enumerate(positions))
        controls = tuple(
            accelerations[index] if index < len(accelerations) else accelerations[-1]
            for index in range(len(times))
        )
        events: list[dict[str, Any]] = []
        if candidate.switch_time is not None and candidate.switch_time <= times[-1]:
            events.append({"time": candidate.switch_time, "event_type": "mode_switch", "event_value": "control_schedule_change"})
        events.append({"time": times[-1], "event_type": "termination", "event_value": "nominal_end"})
        return {
            "times": times,
            "truth_state": {"position": positions, "velocity": velocities, "acceleration": accelerations},
            "observations": {"position": observations},
            "controls": {"acceleration_command": controls},
            "events": tuple(events),
        }

    def normalize_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any], raw_output: dict[str, Any]) -> TrajectoryRun:
        if "failure_reason" in raw_output:
            return TrajectoryRun(
                run_id=f"{self.backend_id}_{candidate.candidate_id}",
                backend_id=self.backend_id,
                scenario_id=candidate.scenario_id,
                seed=candidate.seed,
                success=False,
                failure_reason=str(raw_output["failure_reason"]),
                times=tuple(raw_output["times"]),
                truth_state={},
                observations={},
                events=(),
                metadata={"adapter_family": self.family, "candidate_id": candidate.candidate_id, "search_provenance": candidate.provenance},
            )
        return TrajectoryRun(
            run_id=f"{self.backend_id}_{candidate.candidate_id}",
            backend_id=self.backend_id,
            scenario_id=candidate.scenario_id,
            seed=candidate.seed,
            success=True,
            failure_reason=None,
            times=tuple(raw_output["times"]),
            truth_state=raw_output["truth_state"],
            observations=raw_output["observations"],
            controls=raw_output["controls"],
            events=tuple(raw_output["events"]),
            metadata={
                "adapter_family": self.family,
                "candidate_id": candidate.candidate_id,
                "measurement_dim": 1,
                "coordinate_frame": "scalar_line",
                "search_provenance": candidate.provenance,
            },
        )


class EnvironmentAware1DAdapter(TrajectoryBackendAdapter):
    def supports(self, candidate: BackendCandidateSpec) -> bool:
        return candidate.scenario_family in {"shared_boundary_case", "environment_regime_case"}

    def prepare(self, candidate: BackendCandidateSpec) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "scenario_id": candidate.scenario_id,
            "seed": candidate.seed,
            "duration": candidate.duration,
            "sample_period": candidate.sample_period,
            "initial_position": candidate.initial_position,
            "initial_velocity": candidate.initial_velocity,
            "acceleration": candidate.acceleration,
            "drag_coefficient": candidate.drag_coefficient if candidate.drag_coefficient is not None else 0.15,
            "density_scale": candidate.density_scale if candidate.density_scale is not None else 1.0,
            "wind_bias": candidate.wind_bias if candidate.wind_bias is not None else 0.0,
            "measurement_std": candidate.measurement_std,
        }

    def _simulate_raw_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any]) -> dict[str, Any]:
        times = _times_for_candidate(candidate)
        drag = float(input_bundle["drag_coefficient"])
        density_scale = float(input_bundle["density_scale"])
        wind_bias = float(input_bundle["wind_bias"])
        positions: list[float] = []
        velocities: list[float] = []
        accelerations: list[float] = []
        density_trace: list[float] = []
        wind_trace: list[float] = []
        current_position = candidate.initial_position
        current_velocity = candidate.initial_velocity
        for index, time in enumerate(times):
            local_density = max(0.6, density_scale - 0.04 * index)
            local_wind = wind_bias + 0.02 * sin(time * 2.0)
            effective_accel = candidate.acceleration - drag * local_density * current_velocity + 0.08 * local_wind
            if index > 0:
                dt = times[index] - times[index - 1]
                current_position = current_position + current_velocity * dt + 0.5 * effective_accel * dt * dt
                current_velocity = current_velocity + effective_accel * dt
            positions.append(current_position)
            velocities.append(current_velocity)
            accelerations.append(effective_accel)
            density_trace.append(local_density)
            wind_trace.append(local_wind)
        observations = tuple(position + _deterministic_noise(candidate.seed + 200, index, candidate.measurement_std) for index, position in enumerate(positions))
        return {
            "times": times,
            "truth_state": {"position": tuple(positions), "velocity": tuple(velocities), "acceleration": tuple(accelerations)},
            "observations": {"position": observations},
            "environment_trace": {"density_scale": tuple(density_trace), "wind_bias": tuple(wind_trace)},
            "events": (
                {"time": times[-2], "event_type": "threshold_crossing", "event_value": "density_regime_shift"},
                {"time": times[-1], "event_type": "termination", "event_value": "nominal_end"},
            ),
        }

    def normalize_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any], raw_output: dict[str, Any]) -> TrajectoryRun:
        return TrajectoryRun(
            run_id=f"{self.backend_id}_{candidate.candidate_id}",
            backend_id=self.backend_id,
            scenario_id=candidate.scenario_id,
            seed=candidate.seed,
            success=True,
            failure_reason=None,
            times=tuple(raw_output["times"]),
            truth_state=raw_output["truth_state"],
            observations=raw_output["observations"],
            environment_trace=raw_output["environment_trace"],
            events=tuple(raw_output["events"]),
            metadata={
                "adapter_family": self.family,
                "candidate_id": candidate.candidate_id,
                "measurement_dim": 1,
                "coordinate_frame": "scalar_line",
                "search_provenance": candidate.provenance,
            },
        )


class MockFileBackend1DAdapter(TrajectoryBackendAdapter):
    def supports(self, candidate: BackendCandidateSpec) -> bool:
        return candidate.scenario_family in {"shared_boundary_case", "switching_case", "file_backend_case"}

    def prepare(self, candidate: BackendCandidateSpec) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "scenario_id": candidate.scenario_id,
            "seed": candidate.seed,
            "duration": candidate.duration,
            "sample_period": candidate.sample_period,
            "initial_position": candidate.initial_position,
            "initial_velocity": candidate.initial_velocity,
            "acceleration": candidate.acceleration,
            "switch_time": candidate.switch_time,
            "acceleration_after_switch": candidate.acceleration_after_switch,
            "input_deck_hash": candidate.input_deck_hash,
            "measurement_std": candidate.measurement_std,
        }

    def _simulate_raw_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any]) -> dict[str, Any]:
        times = _times_for_candidate(candidate)
        if not candidate.input_deck_hash:
            return {"failure_reason": "missing_input_deck_hash", "times": times, "stdout": "", "stderr": "input deck hash missing"}
        if candidate.scenario_family == "switching_case":
            positions, velocities, accelerations = _switching_truth(candidate, times)
        else:
            positions, velocities, accelerations = _parameter_truth(candidate, times)
        observed_positions = tuple(position + _deterministic_noise(1000 + candidate.seed, index, candidate.measurement_std) for index, position in enumerate(positions))
        observed_velocities = tuple(velocity + _deterministic_noise(2000 + candidate.seed, index, candidate.measurement_std * 0.5) for index, velocity in enumerate(velocities))
        controls = tuple(
            candidate.longitudinal_command[index] if index < len(candidate.longitudinal_command) else candidate.acceleration
            for index in range(len(times))
        )
        events: list[dict[str, Any]] = []
        if candidate.switch_time is not None and candidate.switch_time <= times[-1]:
            events.append({"time": candidate.switch_time, "event_type": "phase_change", "event_value": "scheduled_switch"})
        events.append({"time": times[-1], "event_type": "termination", "event_value": "nominal_end"})
        return {
            "times": times,
            "truth_state": {"position": positions, "velocity": velocities, "acceleration": accelerations},
            "observations": {"position": observed_positions, "velocity": observed_velocities},
            "controls": {"longitudinal_command": controls},
            "environment_trace": {"density_scale": tuple(max(0.7, 1.0 - 0.01 * index) for index in range(len(times)))},
            "events": tuple(events),
            "stdout": "mock backend completed",
            "stderr": "",
            "return_code": 0,
        }

    def normalize_output(self, candidate: BackendCandidateSpec, input_bundle: dict[str, Any], raw_output: dict[str, Any]) -> TrajectoryRun:
        if "failure_reason" in raw_output:
            return TrajectoryRun(
                run_id=f"{self.backend_id}_{candidate.candidate_id}",
                backend_id=self.backend_id,
                scenario_id=candidate.scenario_id,
                seed=candidate.seed,
                success=False,
                failure_reason=str(raw_output["failure_reason"]),
                times=tuple(raw_output["times"]),
                truth_state={},
                observations={},
                events=(),
                metadata={
                    "adapter_family": self.family,
                    "candidate_id": candidate.candidate_id,
                    "search_provenance": candidate.provenance,
                    "stdout": raw_output.get("stdout", ""),
                    "stderr": raw_output.get("stderr", ""),
                },
            )
        return TrajectoryRun(
            run_id=f"{self.backend_id}_{candidate.candidate_id}",
            backend_id=self.backend_id,
            scenario_id=candidate.scenario_id,
            seed=candidate.seed,
            success=True,
            failure_reason=None,
            times=tuple(raw_output["times"]),
            truth_state=raw_output["truth_state"],
            observations=raw_output["observations"],
            controls=raw_output["controls"],
            environment_trace=raw_output["environment_trace"],
            events=tuple(raw_output["events"]),
            metadata={
                "adapter_family": self.family,
                "candidate_id": candidate.candidate_id,
                "measurement_dim": 1,
                "coordinate_frame": "scalar_line",
                "search_provenance": candidate.provenance,
                "input_deck_hash": candidate.input_deck_hash,
                "stdout": raw_output["stdout"],
                "stderr": raw_output["stderr"],
            },
        )


def _adapter_map() -> dict[str, TrajectoryBackendAdapter]:
    definitions_by_id = {definition.capabilities.backend_id: definition for definition in default_backend_contract_definitions()}
    return {
        "parameter_only_1d": ParameterOnly1DAdapter(definitions_by_id["parameter_only_1d"]),
        "controlled_1d": Controlled1DAdapter(definitions_by_id["controlled_1d"]),
        "environment_aware_1d": EnvironmentAware1DAdapter(definitions_by_id["environment_aware_1d"]),
        "mock_file_backend_1d": MockFileBackend1DAdapter(definitions_by_id["mock_file_backend_1d"]),
    }


def _shared_boundary_candidate() -> BackendCandidateSpec:
    return BackendCandidateSpec(
        candidate_id="shared_boundary_cv_ca",
        scenario_id="shared_boundary_cv_ca",
        scenario_family="shared_boundary_case",
        target_class="constant_velocity",
        difficulty_tier="boundary_v1",
        seed=31,
        duration=2.0,
        sample_period=0.5,
        initial_position=0.0,
        initial_velocity=1.05,
        acceleration=0.12,
        measurement_std=0.03,
        drag_coefficient=0.12,
        density_scale=1.0,
        wind_bias=0.02,
        input_deck_hash="shared_boundary_input_v1",
        longitudinal_command=(0.15, 0.12, 0.12, 0.10, 0.08),
        provenance={"search_method": "manual_proof", "search_iteration": 0},
    )


def _switching_candidate() -> BackendCandidateSpec:
    return BackendCandidateSpec(
        candidate_id="switching_velocity_to_braking",
        scenario_id="switching_velocity_to_braking",
        scenario_family="switching_case",
        target_class="braking",
        difficulty_tier="stress_v1",
        seed=37,
        duration=2.0,
        sample_period=0.5,
        initial_position=0.0,
        initial_velocity=1.4,
        acceleration=0.0,
        measurement_std=0.04,
        switch_time=1.0,
        acceleration_after_switch=-0.7,
        input_deck_hash="switching_case_input_v1",
        longitudinal_command=(0.0, 0.0, -0.7, -0.7, -0.7),
        provenance={"search_method": "manual_proof", "search_iteration": 1},
    )


def _environment_candidate() -> BackendCandidateSpec:
    return BackendCandidateSpec(
        candidate_id="environment_density_gradient",
        scenario_id="environment_density_gradient",
        scenario_family="environment_regime_case",
        target_class="constant_acceleration",
        difficulty_tier="realistic_v1",
        seed=41,
        duration=2.0,
        sample_period=0.5,
        initial_position=0.0,
        initial_velocity=0.8,
        acceleration=0.45,
        measurement_std=0.03,
        drag_coefficient=0.22,
        density_scale=1.0,
        wind_bias=0.05,
        provenance={"search_method": "manual_proof", "search_iteration": 2},
    )


def _failing_candidates() -> tuple[BackendCandidateSpec, ...]:
    return (
        BackendCandidateSpec(
            candidate_id="controlled_missing_switch",
            scenario_id="switching_velocity_to_braking",
            scenario_family="switching_case",
            target_class="braking",
            difficulty_tier="stress_v1",
            seed=43,
            duration=2.0,
            sample_period=0.5,
            initial_position=0.0,
            initial_velocity=1.2,
            acceleration=0.0,
            measurement_std=0.03,
            switch_time=None,
            acceleration_after_switch=-0.8,
            provenance={"search_method": "failure_probe", "search_iteration": 3},
        ),
        BackendCandidateSpec(
            candidate_id="mock_missing_input_deck",
            scenario_id="file_backend_case",
            scenario_family="file_backend_case",
            target_class="maneuver",
            difficulty_tier="adversarial_v1",
            seed=47,
            duration=2.0,
            sample_period=0.5,
            initial_position=0.0,
            initial_velocity=0.9,
            acceleration=0.25,
            measurement_std=0.03,
            input_deck_hash=None,
            provenance={"search_method": "failure_probe", "search_iteration": 4},
        ),
    )
