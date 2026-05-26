from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from io import BytesIO
from math import sin
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.utils.io import write_csv

from ...markdown_builder import MarkdownDocument, MermaidEdge, MermaidFlow, MermaidNode
from ...runtime_paths import prepare_matplotlib
from ...utils.plotting import plt
from ..trajectory_backend_contract import (
    BackendContractDefinition,
    TrajectoryRun,
    default_backend_contract_definitions,
)
from ..trajectory_backend_contract_utils import validate_trajectory_run


@dataclass(frozen=True, slots=True)
class BackendCandidateSpec:
    candidate_id: str
    scenario_id: str
    scenario_family: str
    target_class: str
    difficulty_tier: str
    seed: int
    duration: float
    sample_period: float
    initial_position: float
    initial_velocity: float
    acceleration: float
    measurement_std: float
    switch_time: float | None = None
    acceleration_after_switch: float | None = None
    drag_coefficient: float | None = None
    density_scale: float | None = None
    wind_bias: float | None = None
    input_deck_hash: str | None = None
    longitudinal_command: tuple[float, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterExecutionRecord:
    backend_id: str
    candidate_id: str
    cache_key: str
    cache_hit: bool
    input_bundle: dict[str, Any]
    raw_output: dict[str, Any]
    trajectory_run: TrajectoryRun
    validation_errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BackendAdapterProofResult:
    backend_manifest: dict[str, Any]
    backend_run_rows: tuple[dict[str, Any], ...]
    equivalence_rows: tuple[dict[str, Any], ...]
    failure_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class BackendAdapterProofArtifacts:
    run_dir: Path
    backend_manifest_path: Path
    backend_run_examples_path: Path
    backend_output_equivalence_report_path: Path
    adapter_failure_cases_path: Path
    telemetry_comparison_png_path: Path
    failure_taxonomy_png_path: Path


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


def _parameter_truth(candidate: BackendCandidateSpec, times: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    positions: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []
    for time in times:
        positions.append(candidate.initial_position + candidate.initial_velocity * time + 0.5 * candidate.acceleration * time * time)
        velocities.append(candidate.initial_velocity + candidate.acceleration * time)
        accelerations.append(candidate.acceleration)
    return tuple(positions), tuple(velocities), tuple(accelerations)


def _switching_truth(candidate: BackendCandidateSpec, times: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
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
    return tuple(positions), tuple(velocities), tuple(accelerations)


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


def _run_row(record: AdapterExecutionRecord) -> dict[str, Any]:
    run = record.trajectory_run
    success = run.success and not record.validation_errors
    position = run.truth_state.get("position", ())
    velocity = run.truth_state.get("velocity", ())
    return {
        "backend_id": record.backend_id,
        "candidate_id": record.candidate_id,
        "scenario_id": run.scenario_id,
        "success": success,
        "failure_reason": run.failure_reason or "",
        "cache_key": record.cache_key,
        "cache_hit": record.cache_hit,
        "num_times": len(run.times),
        "position_final": position[-1] if position else "",
        "velocity_final": velocity[-1] if velocity else "",
        "event_count": len(run.events),
        "observation_fields": ",".join(sorted(run.observations.keys())),
        "environment_fields": ",".join(sorted(run.environment_trace.keys())),
        "validation_error_count": len(record.validation_errors),
    }


def _equivalence_rows(shared_records: tuple[AdapterExecutionRecord, ...]) -> tuple[dict[str, Any], ...]:
    baseline = shared_records[0].trajectory_run
    baseline_position = baseline.truth_state["position"]
    rows: list[dict[str, Any]] = []
    for record in shared_records[1:]:
        position = record.trajectory_run.truth_state["position"]
        velocity = record.trajectory_run.truth_state["velocity"]
        rows.append(
            {
                "baseline_backend_id": baseline.backend_id,
                "comparison_backend_id": record.backend_id,
                "scenario_id": baseline.scenario_id,
                "same_num_samples": len(position) == len(baseline_position),
                "max_position_delta": max(abs(a - b) for a, b in zip(position, baseline_position)),
                "final_position_delta": abs(position[-1] - baseline_position[-1]),
                "final_velocity": velocity[-1],
                "common_truth_fields": "position,velocity,acceleration",
                "common_observation_fields": ",".join(sorted(set(baseline.observations).intersection(record.trajectory_run.observations))),
            }
        )
    return tuple(rows)


def _render_telemetry_comparison_png(shared_records: tuple[AdapterExecutionRecord, ...]) -> bytes:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    for record in shared_records:
        run = record.trajectory_run
        axes[0].plot(run.times, run.truth_state["position"], marker="o", label=record.backend_id)
        axes[1].plot(run.times, run.truth_state["velocity"], marker="o", label=record.backend_id)
    axes[0].set_ylabel("Position")
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("Time")
    axes[0].set_title("Normalized Telemetry Comparison")
    axes[0].legend(fontsize=8)
    fig.tight_layout()


    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_failure_taxonomy_png(failure_rows: tuple[dict[str, Any], ...]) -> bytes:
    counts: dict[str, int] = {}
    for row in failure_rows:
        reason = str(row["failure_reason"] or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.bar(labels, values, color="#ca5b4b")
    ax.set_ylabel("Count")
    ax.set_title("Adapter Failure Taxonomy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()


    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def analyze_backend_adapter_proof() -> BackendAdapterProofResult:
    adapters = _adapter_map()
    shared_candidate = _shared_boundary_candidate()
    switching_candidate = _switching_candidate()
    environment_candidate = _environment_candidate()
    failures = _failing_candidates()

    shared_backend_ids = ("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d")
    shared_records = tuple(adapters[backend_id].run(shared_candidate) for backend_id in shared_backend_ids)
    repeated_cache_record = adapters["parameter_only_1d"].run(shared_candidate)
    switching_records = tuple(
        adapters[backend_id].run(switching_candidate)
        for backend_id in ("controlled_1d", "mock_file_backend_1d")
    )
    environment_record = adapters["environment_aware_1d"].run(environment_candidate)
    failure_records = (
        adapters["controlled_1d"].run(failures[0]),
        adapters["mock_file_backend_1d"].run(failures[1]),
    )

    all_records = list(shared_records) + [repeated_cache_record] + list(switching_records) + [environment_record] + list(failure_records)
    run_rows = [_run_row(record) for record in all_records]
    failure_rows = tuple(row for row in run_rows if not bool(row["success"]))
    equivalence_rows = _equivalence_rows(shared_records)

    cache_probe = {
        "candidate_id": shared_candidate.candidate_id,
        "backend_id": "parameter_only_1d",
        "first_cache_key": shared_records[0].cache_key,
        "second_cache_key": repeated_cache_record.cache_key,
        "second_run_cache_hit": repeated_cache_record.cache_hit,
        "stable_cache_key": shared_records[0].cache_key == repeated_cache_record.cache_key,
    }

    backend_manifest = {
        "proof_version": "m32_v1",
        "adapters": [
            {
                "backend_id": adapter.backend_id,
                "family": adapter.family,
                "runtime_class": adapter.definition.capabilities.runtime_class,
                "supports_environment": adapter.definition.capabilities.supports_environment,
                "supports_sequential_control": adapter.definition.capabilities.supports_sequential_control,
                "supported_scenario_families": [
                    scenario_family
                    for scenario_family in ("shared_boundary_case", "switching_case", "environment_regime_case", "file_backend_case")
                    if adapter.supports(
                        {
                            "shared_boundary_case": shared_candidate,
                            "switching_case": switching_candidate,
                            "environment_regime_case": environment_candidate,
                            "file_backend_case": failures[1],
                        }[scenario_family]
                    )
                ],
            }
            for adapter in adapters.values()
        ],
        "shared_compatible_scenario": {
            "candidate_id": shared_candidate.candidate_id,
            "backend_ids": list(shared_backend_ids),
            "common_truth_fields": ["position", "velocity", "acceleration"],
            "common_observation_fields": ["position"],
        },
        "cache_probe": cache_probe,
        "structured_failure_count": len(failure_rows),
    }

    report = MarkdownDocument()
    report.heading("Backend Adapter Proof", level=1)
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"adapters exercised: `{len(adapters)}`",
            f"shared scenario executed across compatible backends: `{', '.join(shared_backend_ids)}`",
            f"structured failures captured: `{len(failure_rows)}`",
            f"stable cache key proven: `{cache_probe['stable_cache_key']}`",
            f"second repeated execution served from cache: `{cache_probe['second_run_cache_hit']}`",
        ]
    )
    report.heading("Shared Compatible Scenario", level=2)
    report.table(
        ["Backend", "Success", "Samples", "Final Position", "Observation Fields"],
        [
            (
                f"`{record.backend_id}`",
                f"{record.trajectory_run.success}",
                f"{len(record.trajectory_run.times)}",
                f"`{record.trajectory_run.truth_state.get('position', ('',))[-1] if record.trajectory_run.truth_state.get('position') else ''}`",
                f"`{', '.join(sorted(record.trajectory_run.observations))}`",
            )
            for record in shared_records
        ],
    )
    report.heading("Adapter Flow", level=2)
    report.mermaid(
        MermaidFlow(
            nodes=(
                MermaidNode("A", "BackendCandidateSpec"),
                MermaidNode("B", "prepare(input_bundle)"),
                MermaidNode("C", "cache_key / cache lookup"),
                MermaidNode("D", "run simulator or reuse cache"),
                MermaidNode("E", "raw output"),
                MermaidNode("F", "normalize_output()"),
                MermaidNode("G", "TrajectoryRun"),
                MermaidNode("H", "validation + artifact rows"),
            ),
            edges=(
                MermaidEdge("A", "B"),
                MermaidEdge("B", "C"),
                MermaidEdge("C", "D"),
                MermaidEdge("D", "E"),
                MermaidEdge("E", "F"),
                MermaidEdge("F", "G"),
                MermaidEdge("G", "H"),
            ),
        )
    )
    report.heading("Output Equivalence", level=2)
    report.table(
        ["Baseline", "Comparison", "Same Samples", "Max Position Delta", "Common Observations"],
        [
            (
                f"`{row['baseline_backend_id']}`",
                f"`{row['comparison_backend_id']}`",
                f"`{row['same_num_samples']}`",
                f"`{row['max_position_delta']:.4f}`",
                f"`{row['common_observation_fields']}`",
            )
            for row in equivalence_rows
        ],
    )
    report.heading("Failure Cases", level=2)
    report.table(
        ["Backend", "Candidate", "Failure Reason"],
        [
            (f"`{row['backend_id']}`", f"`{row['candidate_id']}`", f"`{row['failure_reason']}`")
            for row in failure_rows
        ],
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "This milestone proves adapter execution, normalization, cache-key stability, and structured failures with 1D backends only.",
            "The mock file backend remains synthetic, but it now follows the same prepare, run, normalize, and failure-capture flow expected of a future external simulator adapter.",
            "The shared boundary scenario is intentionally run across multiple backends so the proof covers execution equivalence rather than only schema compatibility.",
        ]
    )
    report_markdown = report.text()

    return BackendAdapterProofResult(
        backend_manifest=backend_manifest,
        backend_run_rows=tuple(run_rows),
        equivalence_rows=equivalence_rows,
        failure_rows=failure_rows,
        report_markdown=report_markdown,
    )


def write_backend_adapter_proof_artifacts(
    base_dir: str | Path,
    *,
    result: BackendAdapterProofResult | None = None,
) -> BackendAdapterProofArtifacts:
    run_dir = Path(base_dir) / "backend_adapter_proof"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_backend_adapter_proof()

    backend_manifest_path = run_dir / "backend_manifest.json"
    backend_run_examples_path = run_dir / "backend_run_examples.csv"
    backend_output_equivalence_report_path = run_dir / "backend_output_equivalence_report.md"
    adapter_failure_cases_path = run_dir / "adapter_failure_cases.csv"
    telemetry_comparison_png_path = run_dir / "normalized_telemetry_comparison.png"
    failure_taxonomy_png_path = run_dir / "adapter_failure_taxonomy.png"

    backend_manifest_path.write_text(json.dumps(payload.backend_manifest, indent=2), encoding="utf-8")
    backend_output_equivalence_report_path.write_text(payload.report_markdown, encoding="utf-8")

    run_fieldnames = list(payload.backend_run_rows[0].keys()) if payload.backend_run_rows else []
    write_csv(backend_run_examples_path, list(payload.backend_run_rows), run_fieldnames)

    failure_fieldnames = list(payload.failure_rows[0].keys()) if payload.failure_rows else run_fieldnames
    write_csv(adapter_failure_cases_path, list(payload.failure_rows), failure_fieldnames)

    adapters = _adapter_map()
    shared_candidate = _shared_boundary_candidate()
    shared_records = tuple(adapters[backend_id].run(shared_candidate) for backend_id in ("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d"))
    telemetry_comparison_png_path.write_bytes(_render_telemetry_comparison_png(shared_records))
    failure_taxonomy_png_path.write_bytes(_render_failure_taxonomy_png(payload.failure_rows))

    return BackendAdapterProofArtifacts(
        run_dir=run_dir,
        backend_manifest_path=backend_manifest_path,
        backend_run_examples_path=backend_run_examples_path,
        backend_output_equivalence_report_path=backend_output_equivalence_report_path,
        adapter_failure_cases_path=adapter_failure_cases_path,
        telemetry_comparison_png_path=telemetry_comparison_png_path,
        failure_taxonomy_png_path=failure_taxonomy_png_path,
    )
