from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy.linalg as linalg
from numpy import (
    arange,
    array,
    asarray,
    diag,
    divide,
    eye,
    full,
    maximum,
    ndarray,
    outer,
    sqrt,
    tile,
    zeros,
    zeros_like,
)
from numpy import mean as nmean

from kinematic_classifier_sandbox.utils.io import _write_json, write_csv

from ..markdown_builder import MarkdownDocument
from ..runtime_paths import prepare_matplotlib
from ..trajectory_generator import generate_switching_scenarios
from ..utils.plotting import plt
from ..utils.io import _write_yaml_like
from ..utils.math import (
    _as_tuple,
    _as_tuple_matrix,
    _block_diag,
    _gaussian_logpdf,
    _logsumexp,
    _mean,
    _normalize_log_scores,
)
from .transition_matrix_accumulator import run_transition_benchmark


def _kalman_predict(
    mean: ndarray,
    covariance: ndarray,
    transition: ndarray,
    process_noise: ndarray,
    bias: ndarray,
) -> tuple[ndarray, ndarray]:
    predicted_mean = transition @ mean + bias
    predicted_covariance = transition @ covariance @ transition.T + process_noise
    predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
    return predicted_mean, predicted_covariance


def _kalman_update(
    predicted_mean: ndarray,
    predicted_covariance: ndarray,
    measurement: ndarray,
    measurement_matrix: ndarray,
    measurement_covariance: ndarray,
) -> tuple[ndarray, ndarray, ndarray, ndarray, float]:
    innovation = measurement - measurement_matrix @ predicted_mean
    innovation_covariance = measurement_matrix @ predicted_covariance @ measurement_matrix.T + measurement_covariance
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
    kalman_gain = predicted_covariance @ measurement_matrix.T @ linalg.inv(innovation_covariance)
    updated_mean = predicted_mean + kalman_gain @ innovation
    identity = eye(predicted_covariance.shape[0], dtype=float)
    updated_covariance = (identity - kalman_gain @ measurement_matrix) @ predicted_covariance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    log_likelihood = _gaussian_logpdf(innovation, innovation_covariance)
    return updated_mean, updated_covariance, innovation, innovation_covariance, log_likelihood
@dataclass(frozen=True, slots=True)
class AdvancedFilterContract:
    backend_id: str
    state_layout: str
    state_labels: tuple[str, ...]
    measurement_labels: tuple[str, ...]
    interface_methods: tuple[str, ...]
    required_outputs: tuple[str, ...]
    diagnostics: tuple[str, ...]
    supported_dimensions: tuple[str, ...]
    future_extensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StateSpaceModeSpec:
    name: str
    class_name: str
    axes: int
    prior_weight: float
    process_noise_scale: float
    measurement_sigma: float
    velocity_damping: float
    acceleration_damping: float
    acceleration_bias: float

    @property
    def state_dim(self) -> int:
        return self.axes * 3

    def transition_matrix(self, dt: float) -> ndarray:
        safe_dt = max(float(dt), 1e-6)
        base = array(
            [
                [1.0, safe_dt, 0.5 * safe_dt * safe_dt],
                [0.0, self.velocity_damping, safe_dt],
                [0.0, 0.0, self.acceleration_damping],
            ],
            dtype=float,
        )
        return _block_diag(base, self.axes)

    def process_covariance(self, dt: float) -> ndarray:
        safe_dt = max(float(dt), 1e-6)
        q = (self.process_noise_scale * self.measurement_sigma) ** 2
        base = array(
            [
                [q * (safe_dt ** 4) / 4.0, q * (safe_dt ** 3) / 2.0, q * (safe_dt ** 2) / 2.0],
                [q * (safe_dt ** 3) / 2.0, q * (safe_dt ** 2), q * safe_dt],
                [q * (safe_dt ** 2) / 2.0, q * safe_dt, q],
            ],
            dtype=float,
        )
        return _block_diag(base, self.axes)

    def process_bias(self) -> ndarray:
        block = array([0.0, 0.0, self.acceleration_bias], dtype=float)
        return tile(block, self.axes)

    def measurement_matrix(self) -> ndarray:
        matrix = zeros((self.axes, self.state_dim), dtype=float)
        for axis in range(self.axes):
            matrix[axis, axis * 3] = 1.0
        return matrix

    def measurement_covariance(self) -> ndarray:
        return eye(self.axes, dtype=float) * (self.measurement_sigma ** 2)

    def initial_mean(self, measurement: ndarray) -> ndarray:
        state = zeros(self.state_dim, dtype=float)
        observation = asarray(measurement, dtype=float).reshape(-1)
        position_values = observation if observation.size == self.axes else full(self.axes, float(observation[0]))
        for axis in range(self.axes):
            state[axis * 3] = float(position_values[axis])
        return state

    def initial_covariance(self, *, scale: float = 4.0) -> ndarray:
        covariance = zeros((self.state_dim, self.state_dim), dtype=float)
        for axis in range(self.axes):
            start = axis * 3
            covariance[start : start + 3, start : start + 3] = diag([scale, scale, scale])
        return covariance


@dataclass(frozen=True, slots=True)
class SwitchingWitness:
    trajectory_id: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_velocity: tuple[float, ...]
    true_acceleration: tuple[float, ...]
    true_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImmStepResult:
    step: int
    time: float
    measurement: float
    true_mode: str
    true_class: str
    switch_event: bool
    mode_prior: dict[str, float]
    mode_mixing: dict[str, dict[str, float]]
    mode_posteriors: dict[str, float]
    class_posteriors: dict[str, float]
    mode_log_likelihoods: dict[str, float]
    mode_innovations: dict[str, tuple[float, ...]]
    mode_innovation_variances: dict[str, tuple[tuple[float, ...], ...]]
    state_mean_by_mode: dict[str, tuple[float, ...]]
    state_covariance_by_mode: dict[str, tuple[tuple[float, ...], ...]]
    combined_state_mean: tuple[float, ...]
    combined_state_covariance: tuple[tuple[float, ...], ...]
    predicted_mode: str
    predicted_class: str
    confidence: float
    mode_entropy: float
    state_rmse: float


@dataclass(frozen=True, slots=True)
class ImmRun:
    trajectory_id: str
    scenario_name: str
    seed: int
    state_labels: tuple[str, ...]
    mode_names: tuple[str, ...]
    class_names: tuple[str, ...]
    steps: tuple[ImmStepResult, ...]
    final_mode_posteriors: dict[str, float]
    final_class_posteriors: dict[str, float]
    final_predicted_mode: str
    final_predicted_class: str
    final_confidence: float
    accuracy: float
    post_switch_accuracy: float
    switch_detection_delay: float
    mean_state_rmse: float


@dataclass(frozen=True, slots=True)
class AdvancedStateInferenceSummary:
    num_witnesses: int
    imm_accuracy: float
    imm_post_switch_accuracy: float
    transition_post_switch_accuracy: float
    improved_witnesses: int
    mean_switch_detection_delay: float
    median_switch_detection_delay: float
    mean_state_rmse: float


@dataclass(frozen=True, slots=True)
class AdvancedFilterContractResult:
    contract: AdvancedFilterContract
    output_schema: dict[str, object]
    diagnostics_schema: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class AdvancedStateInferenceResult:
    witnesses: tuple[SwitchingWitness, ...]
    mode_specs: tuple[StateSpaceModeSpec, ...]
    runs: tuple[ImmRun, ...]
    transition_result: object
    transition_summary: object
    summary: AdvancedStateInferenceSummary
    report_markdown: str
    contract_result: AdvancedFilterContractResult


@dataclass(frozen=True, slots=True)
class AdvancedFilterContractArtifacts:
    run_dir: Path
    contract_path: Path
    output_schema_path: Path
    diagnostics_schema_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class AdvancedStateInferenceArtifacts:
    run_dir: Path
    report_path: Path
    config_path: Path
    mode_probability_history_path: Path
    mixing_probability_history_path: Path
    mode_likelihood_history_path: Path
    state_estimate_history_path: Path
    posterior_history_path: Path
    diagnostics_history_path: Path
    comparison_path: Path
    mode_probability_plot_path: Path
    mixing_probability_plot_path: Path
    mode_likelihood_plot_path: Path
    state_estimate_plot_path: Path
    switch_delay_plot_path: Path
    comparison_plot_path: Path
    plot_png_path: Path


@dataclass(frozen=True, slots=True)
class AdvancedStatePosteriorRow:
    trajectory_id: str
    scenario_name: str
    step: int
    time: float
    measurement: float
    true_mode: str
    true_class: str
    predicted_mode: str
    predicted_class: str
    confidence: float
    mode_entropy: float
    switch_event: bool
    switch_detection_delay: float
    state_rmse: float
    mode_posteriors: dict[str, float]
    class_posteriors: dict[str, float]


@dataclass(frozen=True, slots=True)
class AdvancedStateModeProbabilityRow:
    trajectory_id: str
    scenario_name: str
    step: int
    time: float
    measurement: float
    true_mode: str
    true_class: str
    predicted_mode: str
    predicted_class: str
    confidence: float
    mode_entropy: float
    switch_event: bool
    switch_detection_delay: float
    state_rmse: float
    mode_posteriors: dict[str, float]
    class_posteriors: dict[str, float]
    mode_priors: dict[str, float]


@dataclass(frozen=True, slots=True)
class AdvancedStateMixingRow:
    trajectory_id: str
    scenario_name: str
    step: int
    time: float
    target_mode: str
    source_mode: str
    mixing_probability: float
    mode_prior: float


@dataclass(frozen=True, slots=True)
class AdvancedStateLikelihoodRow:
    trajectory_id: str
    scenario_name: str
    step: int
    time: float
    mode_name: str
    true_mode: str
    innovation: float
    innovation_variance: float
    log_likelihood: float
    mode_posterior: float
    mode_prior: float
    predicted_mode: str


@dataclass(frozen=True, slots=True)
class AdvancedStateEstimateRow:
    trajectory_id: str
    scenario_name: str
    step: int
    time: float
    estimate_type: str
    model_name: str
    true_mode: str
    true_class: str
    switch_event: bool
    state_rmse: float
    state_mean: tuple[float, ...]
    state_cov_diag: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AdvancedStateDiagnosticsRow:
    trajectory_id: str
    scenario_name: str
    step: int
    time: float
    true_mode: str
    predicted_mode: str
    switch_event: bool
    mode_entropy: float
    state_rmse: float
    switch_detection_delay: float
    innovation_energy: float
    confidence: float


@dataclass(frozen=True, slots=True)
class AdvancedStateComparisonRow:
    trajectory_id: str
    scenario_name: str
    imm_accuracy: float
    imm_post_switch_accuracy: float
    imm_switch_delay: float
    imm_state_rmse: float
    transition_post_switch_accuracy: float | str
    transition_accuracy: float | str


def _flatten_state_row(row: AdvancedStateEstimateRow) -> dict[str, object]:
    flat = asdict(row)
    state_mean = flat.pop("state_mean")
    state_cov_diag = flat.pop("state_cov_diag")
    flat.update({f"state_mean_{index}": value for index, value in enumerate(state_mean)})
    flat.update({f"state_cov_diag_{index}": value for index, value in enumerate(state_cov_diag)})
    return flat


def _flatten_probability_row(row: AdvancedStatePosteriorRow) -> dict[str, object]:
    flat = asdict(row)
    mode_posteriors = flat.pop("mode_posteriors")
    class_posteriors = flat.pop("class_posteriors")
    flat.update({f"posterior_{name}": value for name, value in mode_posteriors.items()})
    flat.update({f"class_posterior_{name}": value for name, value in class_posteriors.items()})
    return flat


def _flatten_mode_probability_row(row: AdvancedStateModeProbabilityRow) -> dict[str, object]:
    flat = _flatten_probability_row(
        AdvancedStatePosteriorRow(
            trajectory_id=row.trajectory_id,
            scenario_name=row.scenario_name,
            step=row.step,
            time=row.time,
            measurement=row.measurement,
            true_mode=row.true_mode,
            true_class=row.true_class,
            predicted_mode=row.predicted_mode,
            predicted_class=row.predicted_class,
            confidence=row.confidence,
            mode_entropy=row.mode_entropy,
            switch_event=row.switch_event,
            switch_detection_delay=row.switch_detection_delay,
            state_rmse=row.state_rmse,
            mode_posteriors=row.mode_posteriors,
            class_posteriors=row.class_posteriors,
        )
    )
    flat.update({f"mode_prior_{name}": value for name, value in row.mode_priors.items()})
    return flat


def default_advanced_filter_contract() -> AdvancedFilterContract:
    return AdvancedFilterContract(
        backend_id="imm_1d_pva_lift_prototype",
        state_layout="pva_block",
        state_labels=("position", "velocity", "acceleration"),
        measurement_labels=("position",),
        interface_methods=("initialize", "predict", "update", "state_summary", "evidence_summary", "diagnostics", "history"),
        required_outputs=(
            "trajectory_id",
            "time",
            "filter_id",
            "model_id",
            "true_class",
            "predicted_class",
            "posterior_<class>",
            "log_likelihood_<class>",
            "confidence",
            "state_mean",
            "state_covariance",
        ),
        diagnostics=(
            "mode_probability",
            "mixing_probability",
            "mode_likelihood",
            "combined_state_mean",
            "combined_state_covariance",
            "mode_entropy",
            "switch_event",
            "switch_detection_delay",
            "state_rmse",
        ),
        supported_dimensions=("1d", "3d-pva"),
        future_extensions=("particle_filter", "rbpf", "external-3d-backends"),
    )


def _contract_output_schema(contract: AdvancedFilterContract) -> dict[str, object]:
    state_index_fields = [f"state_mean_{index}" for index in range(len(contract.state_labels))]
    covariance_index_fields = [f"state_cov_diag_{index}" for index in range(len(contract.state_labels))]
    return {
        "artifact": "advanced_filter_output",
        "contract_backend_id": contract.backend_id,
        "required_rows": {
            "posterior_history": [
                "trajectory_id",
                "scenario_name",
                "step",
                "time",
                "true_mode",
                "predicted_mode",
                "true_class",
                "predicted_class",
                "confidence",
                "mode_entropy",
                "state_rmse",
                "switch_event",
                "switch_detection_delay",
                "posterior_<mode_or_class>",
                "class_posterior_<class>",
            ],
            "state_estimate_history": [
                "trajectory_id",
                "scenario_name",
                "step",
                "time",
                "estimate_type",
                "model_name",
                "true_mode",
                "true_class",
                *state_index_fields,
                *covariance_index_fields,
            ],
            "mode_likelihood_history": [
                "trajectory_id",
                "scenario_name",
                "step",
                "time",
                "mode_name",
                "innovation",
                "innovation_variance",
                "log_likelihood",
            ],
        },
        "state_vector_convention": "position, velocity, acceleration blocks repeated per axis",
        "future_lift": "A 3D PVA filter only changes the state size and measurement matrix; the downstream contract stays the same.",
    }


def _diagnostics_schema(contract: AdvancedFilterContract) -> dict[str, object]:
    return {
        "artifact": "advanced_filter_diagnostics",
        "required_fields": [
            "trajectory_id",
            "scenario_name",
            "step",
            "time",
            "switch_event",
            "mode_entropy",
            "state_rmse",
            "switch_detection_delay",
            "innovation_energy",
        ],
        "mode_specific_fields": [
            "mode_prior",
            "mixing_probability",
            "mode_posterior",
            "mode_likelihood",
        ],
        "reference_backend": contract.backend_id,
    }


def _contract_report(contract: AdvancedFilterContract, output_schema: dict[str, object], diagnostics_schema: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Advanced Filter Contract",
            "",
            "This contract defines the shared output surface for advanced state inference backends.",
            "",
            "## Contract Summary",
            "",
            f"- Backend id: `{contract.backend_id}`",
            f"- State layout: `{contract.state_layout}`",
            f"- State labels: `{', '.join(contract.state_labels)}`",
            f"- Measurement labels: `{', '.join(contract.measurement_labels)}`",
            f"- Interface methods: `{', '.join(contract.interface_methods)}`",
            f"- Supported dimensions: `{', '.join(contract.supported_dimensions)}`",
            "",
            "## Shared Output Contract",
            "",
            f"- Required outputs: `{', '.join(contract.required_outputs)}`",
            f"- Diagnostics: `{', '.join(contract.diagnostics)}`",
            "",
            "## Output Schema",
            "",
            "- Posterior rows include posterior columns for each mode in the witness set.",
            "- State rows use `state_mean_i` and `state_cov_diag_i` fields so the same schema can hold 1D and 3D PVA states.",
            f"- Future extensions: `{', '.join(contract.future_extensions)}`",
            "",
            "## Diagnostics Schema",
            "",
            f"- Required diagnostics: `{', '.join(diagnostics_schema['required_fields'])}`",
            "",
            "## Notes",
            "",
            "- The contract is intentionally mode- and class-compatible rather than tied to one 1D example.",
            "- The downstream evaluation surface stays stable when the state vector grows from 1D PVA to 3D PVA.",
            "- PF and RBPF are future extensions, not required for this proof.",
        ]
    )


def analyze_advanced_filter_contract() -> AdvancedFilterContractResult:
    contract = default_advanced_filter_contract()
    output_schema = _contract_output_schema(contract)
    diagnostics_schema = _diagnostics_schema(contract)
    report_markdown = _contract_report(contract, output_schema, diagnostics_schema)
    return AdvancedFilterContractResult(
        contract=contract,
        output_schema=output_schema,
        diagnostics_schema=diagnostics_schema,
        report_markdown=report_markdown,
    )


def _state_layout_axes(contract: AdvancedFilterContract) -> int:
    return len(contract.state_labels)


def default_imm_mode_specs(*, axes: int = 1) -> tuple[StateSpaceModeSpec, ...]:
    return (
        StateSpaceModeSpec(
            name="stationary",
            class_name="stationary",
            axes=axes,
            prior_weight=0.25,
            process_noise_scale=0.28,
            measurement_sigma=0.20,
            velocity_damping=0.72,
            acceleration_damping=0.50,
            acceleration_bias=0.0,
        ),
        StateSpaceModeSpec(
            name="constant_velocity",
            class_name="constant_velocity",
            axes=axes,
            prior_weight=0.25,
            process_noise_scale=0.38,
            measurement_sigma=0.20,
            velocity_damping=0.98,
            acceleration_damping=0.72,
            acceleration_bias=0.0,
        ),
        StateSpaceModeSpec(
            name="braking",
            class_name="braking",
            axes=axes,
            prior_weight=0.25,
            process_noise_scale=0.42,
            measurement_sigma=0.20,
            velocity_damping=0.95,
            acceleration_damping=0.84,
            acceleration_bias=-0.28,
        ),
        StateSpaceModeSpec(
            name="maneuver",
            class_name="maneuver",
            axes=axes,
            prior_weight=0.25,
            process_noise_scale=1.10,
            measurement_sigma=0.20,
            velocity_damping=0.99,
            acceleration_damping=0.97,
            acceleration_bias=0.0,
        ),
    )


def generate_advanced_state_inference_witnesses(*, seed: int = 7, replicas: int = 6) -> tuple[SwitchingWitness, ...]:
    witnesses: list[SwitchingWitness] = []
    for replica in range(replicas):
        for artifact in generate_switching_scenarios(seed=seed + replica * 31):
            params = artifact.generator_parameters
            segment_modes = list(params["segment_modes"])
            switch_time = float(params["switch_time"])
            true_modes = tuple(segment_modes[0] if time < switch_time else segment_modes[1] for time in artifact.times)
            witnesses.append(
                SwitchingWitness(
                    trajectory_id=f"{artifact.trajectory_id}_{replica}",
                    scenario_name=artifact.scenario_id,
                    seed=artifact.seed,
                    times=tuple(float(time) for time in artifact.times),
                    measurements=tuple(float(value) for value in artifact.measurements),
                    true_position=tuple(float(value) for value in artifact.true_position),
                    true_velocity=tuple(float(value) for value in artifact.true_velocity),
                    true_acceleration=tuple(float(value) for value in artifact.true_acceleration),
                    true_modes=true_modes,
                )
            )
    return tuple(witnesses)


def _mode_transition_template(mode_names: Sequence[str]) -> dict[str, dict[str, float]]:
    if set(mode_names) == {"stationary", "constant_velocity", "braking", "maneuver"}:
        return {
            "stationary": {"stationary": 0.80, "constant_velocity": 0.16, "braking": 0.02, "maneuver": 0.02},
            "constant_velocity": {"stationary": 0.04, "constant_velocity": 0.72, "braking": 0.16, "maneuver": 0.08},
            "braking": {"stationary": 0.03, "constant_velocity": 0.18, "braking": 0.72, "maneuver": 0.07},
            "maneuver": {"stationary": 0.03, "constant_velocity": 0.16, "braking": 0.10, "maneuver": 0.71},
        }
    matrix = {}
    for source in mode_names:
        matrix[source] = {target: (0.7 if target == source else 0.3 / max(len(mode_names) - 1, 1)) for target in mode_names}
    return matrix


def _mixing_weights(
    prior: dict[str, float],
    transition_matrix: dict[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    predicted_prior = {
        target: sum(prior[source] * transition_matrix[source][target] for source in prior)
        for target in prior
    }
    mixing: dict[str, dict[str, float]] = {}
    for target, target_prior in predicted_prior.items():
        mixing[target] = {}
        denominator = max(target_prior, 1e-12)
        for source in prior:
            mixing[target][source] = prior[source] * transition_matrix[source][target] / denominator
    return predicted_prior, mixing


def _initial_states(
    witness: SwitchingWitness,
    mode_specs: tuple[StateSpaceModeSpec, ...],
) -> dict[str, tuple[ndarray, ndarray]]:
    initial_measurement = array([witness.measurements[0]], dtype=float)
    states: dict[str, tuple[ndarray, ndarray]] = {}
    for spec in mode_specs:
        states[spec.name] = (spec.initial_mean(initial_measurement), spec.initial_covariance(scale=4.0))
    return states


def run_imm_filter(
    witness: SwitchingWitness,
    mode_specs: tuple[StateSpaceModeSpec, ...],
    *,
    transition_matrix: dict[str, dict[str, float]] | None = None,
) -> ImmRun:
    mode_names = tuple(spec.name for spec in mode_specs)
    class_names = tuple(dict.fromkeys(spec.class_name for spec in mode_specs))
    total_prior = sum(spec.prior_weight for spec in mode_specs)
    posterior = {spec.name: spec.prior_weight / total_prior for spec in mode_specs}
    transition = transition_matrix or _mode_transition_template(mode_names)
    states = _initial_states(witness, mode_specs)
    steps: list[ImmStepResult] = []

    for step_index, (time, measurement, true_mode, true_position, true_velocity, true_acceleration) in enumerate(
        zip(
            witness.times,
            witness.measurements,
            witness.true_modes,
            witness.true_position,
            witness.true_velocity,
            witness.true_acceleration,
        )
    ):
        dt = 0.0 if step_index == 0 else time - witness.times[step_index - 1]
        mode_prior, mixing = _mixing_weights(posterior, transition)
        mode_log_scores: dict[str, float] = {}
        mode_likelihoods: dict[str, float] = {}
        mode_innovations: dict[str, tuple[float, ...]] = {}
        mode_innovation_variances: dict[str, tuple[tuple[float, ...], ...]] = {}
        updated_states: dict[str, tuple[ndarray, ndarray]] = {}
        for spec in mode_specs:
            mixed_mean = zeros(spec.state_dim, dtype=float)
            for source in mode_names:
                mixed_mean += mixing[spec.name][source] * states[source][0]
            mixed_covariance = zeros((spec.state_dim, spec.state_dim), dtype=float)
            for source in mode_names:
                source_weight = mixing[spec.name][source]
                source_mean, source_covariance = states[source]
                diff = source_mean - mixed_mean
                mixed_covariance += source_weight * (source_covariance + outer(diff, diff))

            transition_matrix_t = spec.transition_matrix(dt)
            process_covariance_t = spec.process_covariance(dt)
            bias = spec.process_bias()
            predicted_mean, predicted_covariance = _kalman_predict(
                mixed_mean,
                mixed_covariance,
                transition_matrix_t,
                process_covariance_t,
                bias,
            )
            measurement_matrix = spec.measurement_matrix()
            measurement_covariance = spec.measurement_covariance()
            updated_mean, updated_covariance, innovation, innovation_covariance, log_likelihood = _kalman_update(
                predicted_mean,
                predicted_covariance,
                array([measurement], dtype=float),
                measurement_matrix,
                measurement_covariance,
            )
            updated_states[spec.name] = (updated_mean, updated_covariance)
            mode_likelihoods[spec.name] = float(log_likelihood)
            mode_log_scores[spec.name] = float(log(max(mode_prior[spec.name], 1e-12)) + log_likelihood)
            mode_innovations[spec.name] = _as_tuple(innovation)
            mode_innovation_variances[spec.name] = _as_tuple_matrix(innovation_covariance)

        posterior = _normalize_log_scores(mode_log_scores)
        class_log_scores = {
            class_name: _logsumexp([mode_log_scores[spec.name] for spec in mode_specs if spec.class_name == class_name])
            for class_name in class_names
        }
        class_posteriors = _normalize_log_scores(class_log_scores)
        predicted_mode = max(posterior, key=posterior.get)
        predicted_class = max(class_posteriors, key=class_posteriors.get)

        combined_state_mean = zeros(mode_specs[0].state_dim, dtype=float)
        for spec in mode_specs:
            combined_state_mean += posterior[spec.name] * updated_states[spec.name][0]
        combined_state_covariance = zeros((mode_specs[0].state_dim, mode_specs[0].state_dim), dtype=float)
        for spec in mode_specs:
            mean_j, covariance_j = updated_states[spec.name]
            diff = mean_j - combined_state_mean
            combined_state_covariance += posterior[spec.name] * (covariance_j + outer(diff, diff))

        truth_vector = array([true_position, true_velocity, true_acceleration], dtype=float)
        state_rmse = float(sqrt(nmean((combined_state_mean[:3] - truth_vector) ** 2)))
        switch_event = step_index > 0 and true_mode != witness.true_modes[step_index - 1]
        mode_entropy = float(-sum(prob * log(max(prob, 1e-12)) for prob in posterior.values()))

        steps.append(
            ImmStepResult(
                step=step_index,
                time=float(time),
                measurement=float(measurement),
                true_mode=true_mode,
                true_class=true_mode,
                switch_event=switch_event,
                mode_prior=dict(mode_prior),
                mode_mixing={target: dict(weights) for target, weights in mixing.items()},
                mode_posteriors=dict(posterior),
                class_posteriors=dict(class_posteriors),
                mode_log_likelihoods=dict(mode_likelihoods),
                mode_innovations=dict(mode_innovations),
                mode_innovation_variances=dict(mode_innovation_variances),
                state_mean_by_mode={name: _as_tuple(state[0]) for name, state in updated_states.items()},
                state_covariance_by_mode={name: _as_tuple_matrix(state[1]) for name, state in updated_states.items()},
                combined_state_mean=_as_tuple(combined_state_mean),
                combined_state_covariance=_as_tuple_matrix(combined_state_covariance),
                predicted_mode=predicted_mode,
                predicted_class=predicted_class,
                confidence=float(posterior[predicted_mode]),
                mode_entropy=mode_entropy,
                state_rmse=state_rmse,
            )
        )

        states = updated_states

    first_switch_index = next((index for index, mode in enumerate(witness.true_modes) if mode != witness.true_modes[0]), len(witness.true_modes))
    if first_switch_index < len(witness.true_modes):
        switched_mode = witness.true_modes[first_switch_index]
        detection_index = next((step.step for step in steps[first_switch_index:] if step.predicted_mode == switched_mode), None)
        if detection_index is None:
            switch_detection_delay = float(witness.times[-1] - witness.times[first_switch_index])
        else:
            switch_detection_delay = float(witness.times[detection_index] - witness.times[first_switch_index])
        post_switch_steps = steps[first_switch_index:]
    else:
        switch_detection_delay = 0.0
        post_switch_steps = steps[-1:]

    accuracy = _mean([1.0 if step.predicted_mode == step.true_mode else 0.0 for step in steps])
    post_switch_accuracy = _mean([1.0 if step.predicted_mode == step.true_mode else 0.0 for step in post_switch_steps])
    mean_state_rmse = _mean([step.state_rmse for step in steps])
    final_mode_posteriors = dict(steps[-1].mode_posteriors)
    final_class_posteriors = dict(steps[-1].class_posteriors)
    final_predicted_mode = steps[-1].predicted_mode
    final_predicted_class = steps[-1].predicted_class
    final_confidence = steps[-1].confidence

    return ImmRun(
        trajectory_id=witness.trajectory_id,
        scenario_name=witness.scenario_name,
        seed=witness.seed,
        state_labels=("position", "velocity", "acceleration"),
        mode_names=mode_names,
        class_names=class_names,
        steps=tuple(steps),
        final_mode_posteriors=final_mode_posteriors,
        final_class_posteriors=final_class_posteriors,
        final_predicted_mode=final_predicted_mode,
        final_predicted_class=final_predicted_class,
        final_confidence=final_confidence,
        accuracy=accuracy,
        post_switch_accuracy=post_switch_accuracy,
        switch_detection_delay=switch_detection_delay,
        mean_state_rmse=mean_state_rmse,
    )


def analyze_advanced_state_inference(*, seed: int = 7, replicas: int = 6) -> AdvancedStateInferenceResult:
    contract_result = analyze_advanced_filter_contract()
    witnesses = generate_advanced_state_inference_witnesses(seed=seed, replicas=replicas)
    mode_specs = default_imm_mode_specs(axes=1)
    runs = tuple(run_imm_filter(witness, mode_specs) for witness in witnesses)
    transition_result = run_transition_benchmark(seed=seed, replicas=replicas)
    transition_by_id = {run.trajectory_id: run for run in transition_result.transition_runs}
    improved_witnesses = sum(
        1
        for run in runs
        if run.trajectory_id in transition_by_id and run.post_switch_accuracy > transition_by_id[run.trajectory_id].post_switch_accuracy
    )
    summary = AdvancedStateInferenceSummary(
        num_witnesses=len(runs),
        imm_accuracy=_mean([run.accuracy for run in runs]),
        imm_post_switch_accuracy=_mean([run.post_switch_accuracy for run in runs]),
        transition_post_switch_accuracy=_mean([transition_by_id[run.trajectory_id].post_switch_accuracy for run in runs]),
        improved_witnesses=improved_witnesses,
        mean_switch_detection_delay=_mean([run.switch_detection_delay for run in runs]),
        median_switch_detection_delay=float(median([run.switch_detection_delay for run in runs])),
        mean_state_rmse=_mean([run.mean_state_rmse for run in runs]),
    )
    report_markdown = render_advanced_state_inference_report(
        result=AdvancedStateInferenceResult(
            witnesses=witnesses,
            mode_specs=mode_specs,
            runs=runs,
            transition_result=transition_result,
            transition_summary=transition_result.summary,
            summary=summary,
            report_markdown="",
            contract_result=contract_result,
        )
    )
    return AdvancedStateInferenceResult(
        witnesses=witnesses,
        mode_specs=mode_specs,
        runs=runs,
        transition_result=transition_result,
        transition_summary=transition_result.summary,
        summary=summary,
        report_markdown=report_markdown,
        contract_result=contract_result,
    )


def _representative_run(result: AdvancedStateInferenceResult) -> ImmRun:
    for run in result.runs:
        if run.post_switch_accuracy >= result.summary.imm_post_switch_accuracy:
            return run
    return result.runs[0]


def render_advanced_state_inference_report(result: AdvancedStateInferenceResult) -> str:
    representative = _representative_run(result)
    transition_summary = result.transition_summary
    report = MarkdownDocument("Advanced State Inference And IMM Lift")
    report.paragraph(
        "This report documents the abstract advanced-filter contract and a 1D IMM witness backend that plugs "
        "into the current classifier ladder. The contract is intentionally generic so the same state/evidence/diagnostic "
        "surface can later lift to 3D PVA trajectories without changing the downstream evaluation path."
    )

    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Witnesses: {result.summary.num_witnesses}",
            f"IMM accuracy: {result.summary.imm_accuracy:.3f}",
            f"IMM post-switch accuracy: {result.summary.imm_post_switch_accuracy:.3f}",
            f"Transition post-switch accuracy: {result.summary.transition_post_switch_accuracy:.3f}",
            f"Improved witnesses over transition baseline: {result.summary.improved_witnesses}",
            f"Mean switch detection delay: {result.summary.mean_switch_detection_delay:.3f}",
            f"Median switch detection delay: {result.summary.median_switch_detection_delay:.3f}",
            f"Mean state RMSE: {result.summary.mean_state_rmse:.3f}",
        ]
    )

    report.heading("Contract", level=2)
    report.bullet_list(
        [
            f"State labels: {report.inline_code(', '.join(result.contract_result.contract.state_labels))}",
            f"Measurement labels: {report.inline_code(', '.join(result.contract_result.contract.measurement_labels))}",
            f"Interface methods: {report.inline_code(', '.join(result.contract_result.contract.interface_methods))}",
            f"Required outputs: {report.inline_code(', '.join(result.contract_result.contract.required_outputs))}",
            f"Diagnostics: {report.inline_code(', '.join(result.contract_result.contract.diagnostics))}",
        ]
    )

    report.heading("Witness Interpretation", level=2)
    report.bullet_list(
        [
            f"Representative witness: {report.inline_code(representative.scenario_name)}",
            f"Representative run: {report.inline_code(representative.trajectory_id)}",
            f"Final IMM mode: {report.inline_code(representative.final_predicted_mode)}",
            f"Final IMM class: {report.inline_code(representative.final_predicted_class)}",
            f"Final confidence: {report.inline_code(f'{representative.final_confidence:.3f}')}",
        ]
    )

    report.heading("Transition Baseline", level=2)
    report.bullet_list(
        [
            f"Transition baseline post-switch accuracy: {report.inline_code(f'{transition_summary.transition_post_switch_accuracy:.3f}')}",
            "Kalman-bank reference remains the simpler model-based rung in the ladder; this IMM proof is the switching-aware extension.",
        ]
    )

    report.heading("3D Lift Note", level=2)
    report.bullet_list(
        [
            "The current proof uses a 1D PVA state vector, but the contract is axis-blocked and can lift to 3D PVA by increasing the axis count from 1 to 3.",
            "The downstream evaluation surface does not change: the backend still emits state summaries, evidence summaries, posterior rows, and diagnostics.",
            "The only structural change in a 3D lift is the state size and measurement matrix shape.",
        ]
    )

    report.heading("Evidence Surface", level=2)
    report.bullet_list(
        [
            "IMM emits mode priors, mixing weights, innovation likelihoods, combined state estimates, and posterior-compatible rows.",
            "The same witness family is used to compare against the transition-matrix rung so the benefit of state mixing is visible in post-switch behavior.",
        ]
    )

    report.heading("Promotion Interpretation", level=2)
    report.bullet_list(
        [
            "This is a 1D proof of the advanced-filter contract, not a final claim that PF or RBPF are required.",
            "PF and RBPF remain future extensions if switching or non-Gaussian failure evidence justifies them.",
        ]
    )
    return report.text()


def _render_diagnostics_figure(result: AdvancedStateInferenceResult):
    representative = _representative_run(result)
    transition_lookup = {run.trajectory_id: run for run in result.transition_result.transition_runs} if hasattr(result.transition_result, "transition_runs") else {}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    mode_names = list(representative.mode_names)
    colors = {name: color for name, color in zip(mode_names, ("#2563eb", "#16a34a", "#d97706", "#dc2626"))}

    axes[0, 0].set_title(f"Mode posterior timeline: {representative.scenario_name}", loc="left", fontsize=12, fontweight="bold")
    for mode_name in mode_names:
        axes[0, 0].plot(
            [step.time for step in representative.steps],
            [step.mode_posteriors[mode_name] for step in representative.steps],
            color=colors[mode_name],
            linewidth=2.0,
            label=mode_name,
        )
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("posterior")
    axes[0, 0].set_xlabel("time")

    axes[0, 1].set_title("Combined state estimate vs truth", loc="left", fontsize=12, fontweight="bold")
    times = [step.time for step in representative.steps]
    axes[0, 1].plot(times, [step.combined_state_mean[0] for step in representative.steps], color="#2563eb", linewidth=2.0, label="estimated position")
    axes[0, 1].plot(times, list(_witness_lookup(result.witnesses)[representative.trajectory_id].true_position), color="#9ca3af", linewidth=1.8, linestyle="--", label="true position")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].set_ylabel("position")
    axes[0, 1].set_xlabel("time")

    axes[1, 0].set_title("IMM vs transition post-switch accuracy", loc="left", fontsize=12, fontweight="bold")
    labels = [run.scenario_name for run in result.runs[:8]]
    imm_values = [run.post_switch_accuracy for run in result.runs[:8]]
    transition_values = [transition_lookup[run.trajectory_id].post_switch_accuracy for run in result.runs[:8]] if transition_lookup else [0.0 for _ in labels]
    x_positions = arange(len(labels))
    width = 0.36
    axes[1, 0].bar(x_positions - width / 2.0, transition_values, width=width, color="#9ca3af", label="transition")
    axes[1, 0].bar(x_positions + width / 2.0, imm_values, width=width, color="#2563eb", label="IMM")
    axes[1, 0].set_xticks(x_positions)
    axes[1, 0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_ylabel("post-switch accuracy")

    axes[1, 1].set_title("Switch detection delay", loc="left", fontsize=12, fontweight="bold")
    axes[1, 1].plot([run.switch_detection_delay for run in result.runs], color="#d97706", linewidth=2.0, marker="o")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].set_ylabel("delay")
    axes[1, 1].set_xlabel("witness index")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.suptitle("Advanced State Inference Diagnostics", fontsize=14, fontweight="bold")
    return fig


def _render_mode_probability_figure(result: AdvancedStateInferenceResult):
    representative = _representative_run(result)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    mode_names = list(representative.mode_names)
    colors = {name: color for name, color in zip(mode_names, ("#2563eb", "#16a34a", "#d97706", "#dc2626"))}
    ax.set_title(f"Mode posterior timeline: {representative.scenario_name}", loc="left", fontsize=12, fontweight="bold")
    for mode_name in mode_names:
        ax.plot(
            [step.time for step in representative.steps],
            [step.mode_posteriors[mode_name] for step in representative.steps],
            color=colors[mode_name],
            linewidth=2.0,
            label=mode_name,
        )
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("posterior")
    ax.set_xlabel("time")
    fig.tight_layout()
    return fig


def _render_mixing_probability_figure(result: AdvancedStateInferenceResult):
    representative = _representative_run(result)
    mode_names = list(representative.mode_names)
    matrix = zeros((len(mode_names), len(mode_names)), dtype=float)
    counts = zeros((len(mode_names), len(mode_names)), dtype=float)
    for step in representative.steps:
        for target_index, target_mode in enumerate(mode_names):
            for source_index, source_mode in enumerate(mode_names):
                matrix[target_index, source_index] += step.mode_mixing[target_mode][source_mode]
                counts[target_index, source_index] += 1.0
    matrix = divide(matrix, maximum(counts, 1.0), out=zeros_like(matrix), where=counts > 0)
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Average mode mixing probability", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels(mode_names, rotation=20, ha="right")
    ax.set_yticks(range(len(mode_names)))
    ax.set_yticklabels(mode_names)
    ax.set_xlabel("source mode")
    ax.set_ylabel("target mode")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_mode_likelihood_figure(result: AdvancedStateInferenceResult):
    representative = _representative_run(result)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    mode_names = list(representative.mode_names)
    colors = {name: color for name, color in zip(mode_names, ("#2563eb", "#16a34a", "#d97706", "#dc2626"))}
    ax.set_title(f"Mode log-likelihood timeline: {representative.scenario_name}", loc="left", fontsize=12, fontweight="bold")
    for mode_name in mode_names:
        ax.plot(
            [step.time for step in representative.steps],
            [step.mode_log_likelihoods[mode_name] for step in representative.steps],
            color=colors[mode_name],
            linewidth=2.0,
            label=mode_name,
        )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("log likelihood")
    ax.set_xlabel("time")
    fig.tight_layout()
    return fig


def _render_state_estimate_figure(result: AdvancedStateInferenceResult):
    representative = _representative_run(result)
    witness = _witness_lookup(result.witnesses)[representative.trajectory_id]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    times = [step.time for step in representative.steps]
    estimated_position = [step.combined_state_mean[0] for step in representative.steps]
    estimated_sigma = [float(sqrt(max(step.combined_state_covariance[0][0], 0.0))) for step in representative.steps]
    ax.set_title(f"Combined state estimate vs truth: {representative.scenario_name}", loc="left", fontsize=12, fontweight="bold")
    ax.plot(times, estimated_position, color="#2563eb", linewidth=2.0, label="estimated position")
    ax.fill_between(
        times,
        [mean - sigma for mean, sigma in zip(estimated_position, estimated_sigma)],
        [mean + sigma for mean, sigma in zip(estimated_position, estimated_sigma)],
        color="#93c5fd",
        alpha=0.35,
        label="1-sigma band",
    )
    ax.plot(times, list(witness.true_position), color="#9ca3af", linewidth=1.8, linestyle="--", label="true position")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("position")
    ax.set_xlabel("time")
    fig.tight_layout()
    return fig


def _render_switch_delay_figure(result: AdvancedStateInferenceResult):
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    delays = [run.switch_detection_delay for run in result.runs]
    ax.set_title("Switch detection delay", loc="left", fontsize=12, fontweight="bold")
    ax.plot(range(len(delays)), delays, color="#d97706", linewidth=2.0, marker="o")
    ax.grid(True, alpha=0.25)
    ax.set_ylabel("delay")
    ax.set_xlabel("witness index")
    fig.tight_layout()
    return fig


def _render_comparison_figure(result: AdvancedStateInferenceResult):
    transition_lookup = {run.trajectory_id: run for run in result.transition_result.transition_runs} if hasattr(result.transition_result, "transition_runs") else {}
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    labels = [run.scenario_name for run in result.runs[:8]]
    imm_values = [run.post_switch_accuracy for run in result.runs[:8]]
    transition_values = [transition_lookup[run.trajectory_id].post_switch_accuracy for run in result.runs[:8]] if transition_lookup else [0.0 for _ in labels]
    x_positions = arange(len(labels))
    width = 0.36
    ax.bar(x_positions - width / 2.0, transition_values, width=width, color="#9ca3af", label="transition")
    ax.bar(x_positions + width / 2.0, imm_values, width=width, color="#2563eb", label="IMM")
    ax.set_title("IMM vs transition post-switch accuracy", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("post-switch accuracy")
    fig.tight_layout()
    return fig


def _witness_lookup(witnesses: tuple[SwitchingWitness, ...]) -> dict[str, SwitchingWitness]:
    return {witness.trajectory_id: witness for witness in witnesses}


def render_advanced_state_inference_png_bytes(result: AdvancedStateInferenceResult) -> bytes:
    fig = _render_diagnostics_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_advanced_state_inference_mode_probability_png_bytes(result: AdvancedStateInferenceResult) -> bytes:
    fig = _render_mode_probability_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_advanced_state_inference_mixing_probability_png_bytes(result: AdvancedStateInferenceResult) -> bytes:
    fig = _render_mixing_probability_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_advanced_state_inference_mode_likelihood_png_bytes(result: AdvancedStateInferenceResult) -> bytes:
    fig = _render_mode_likelihood_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_advanced_state_inference_state_estimate_png_bytes(result: AdvancedStateInferenceResult) -> bytes:
    fig = _render_state_estimate_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_advanced_state_inference_switch_delay_png_bytes(result: AdvancedStateInferenceResult) -> bytes:
    fig = _render_switch_delay_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_advanced_state_inference_comparison_png_bytes(result: AdvancedStateInferenceResult) -> bytes:
    fig = _render_comparison_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_advanced_filter_contract_artifacts(
    output_dir: str | Path,
    *,
    result: AdvancedFilterContractResult | None = None,
) -> AdvancedFilterContractArtifacts:
    contract_result = result or analyze_advanced_filter_contract()
    run_dir = Path(output_dir) / "advanced_state_inference_contract"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "filter_backend_contract.json"
    output_schema_path = run_dir / "advanced_filter_output_schema.json"
    diagnostics_schema_path = run_dir / "diagnostics_schema.json"
    report_path = run_dir / "contract_report.md"
    _write_json(contract_path, asdict(contract_result.contract))
    _write_json(output_schema_path, contract_result.output_schema)
    _write_json(diagnostics_schema_path, contract_result.diagnostics_schema)
    report_path.write_text(contract_result.report_markdown, encoding="utf-8")
    return AdvancedFilterContractArtifacts(
        run_dir=run_dir,
        contract_path=contract_path,
        output_schema_path=output_schema_path,
        diagnostics_schema_path=diagnostics_schema_path,
        report_path=report_path,
    )


def write_advanced_state_inference_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    replicas: int = 6,
    result: AdvancedStateInferenceResult | None = None,
) -> AdvancedStateInferenceArtifacts:
    analysis = result or analyze_advanced_state_inference(seed=seed, replicas=replicas)
    run_dir = Path(output_dir) / "advanced_state_inference_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "imm_report.md"
    config_path = run_dir / "imm_config.yaml"
    mode_probability_history_path = run_dir / "mode_probability_history.csv"
    mixing_probability_history_path = run_dir / "mixing_probability_history.csv"
    mode_likelihood_history_path = run_dir / "mode_likelihood_history.csv"
    state_estimate_history_path = run_dir / "state_estimate_history.csv"
    posterior_history_path = run_dir / "posterior_history.csv"
    diagnostics_history_path = run_dir / "diagnostics_history.csv"
    comparison_path = run_dir / "advanced_state_inference_comparison.csv"
    plot_png_path = run_dir / "imm_diagnostics.png"
    mode_probability_plot_path = run_dir / "mode_probability_timeline.png"
    mixing_probability_plot_path = run_dir / "mixing_probability_heatmap.png"
    mode_likelihood_plot_path = run_dir / "mode_likelihood_timeline.png"
    state_estimate_plot_path = run_dir / "state_estimate_with_truth.png"
    switch_delay_plot_path = run_dir / "switch_detection_delay.png"
    comparison_plot_path = run_dir / "imm_vs_transition_comparison.png"

    report_path.write_text(analysis.report_markdown, encoding="utf-8")
    plot_png_path.write_bytes(render_advanced_state_inference_png_bytes(analysis))
    mode_probability_plot_path.write_bytes(render_advanced_state_inference_mode_probability_png_bytes(analysis))
    mixing_probability_plot_path.write_bytes(render_advanced_state_inference_mixing_probability_png_bytes(analysis))
    mode_likelihood_plot_path.write_bytes(render_advanced_state_inference_mode_likelihood_png_bytes(analysis))
    state_estimate_plot_path.write_bytes(render_advanced_state_inference_state_estimate_png_bytes(analysis))
    switch_delay_plot_path.write_bytes(render_advanced_state_inference_switch_delay_png_bytes(analysis))
    comparison_plot_path.write_bytes(render_advanced_state_inference_comparison_png_bytes(analysis))
    _write_yaml_like(
        config_path,
        [
            "experiment:",
            "  name: advanced_state_inference_imm",
            f"  seed: {seed}",
            f"  replicas: {replicas}",
            "filter:",
            "  backend: imm_1d_pva_lift_prototype",
            "  state_layout: pva_block",
            "  state_labels: [position, velocity, acceleration]",
            "  supported_dimensions: [1d, 3d-pva]",
            "  witness_family: switching_trajectory_generator",
        ],
    )

    mode_names = [spec.name for spec in analysis.mode_specs]
    class_names = list(analysis.runs[0].class_names)
    state_dim = len(analysis.runs[0].state_labels)
    posterior_rows: list[AdvancedStatePosteriorRow] = []
    mode_probability_rows: list[AdvancedStateModeProbabilityRow] = []
    mixing_rows: list[AdvancedStateMixingRow] = []
    likelihood_rows: list[AdvancedStateLikelihoodRow] = []
    state_rows: list[AdvancedStateEstimateRow] = []
    diagnostics_rows: list[AdvancedStateDiagnosticsRow] = []
    comparison_rows: list[AdvancedStateComparisonRow] = []
    transition_lookup = {run.trajectory_id: run for run in analysis.transition_result.transition_runs} if hasattr(analysis.transition_result, "transition_runs") else {}

    for run in analysis.runs:
        transition_run = transition_lookup.get(run.trajectory_id)
        comparison_rows.append(
            AdvancedStateComparisonRow(
                trajectory_id=run.trajectory_id,
                scenario_name=run.scenario_name,
                imm_accuracy=run.accuracy,
                imm_post_switch_accuracy=run.post_switch_accuracy,
                imm_switch_delay=run.switch_detection_delay,
                imm_state_rmse=run.mean_state_rmse,
                transition_post_switch_accuracy=transition_run.post_switch_accuracy if transition_run else "",
                transition_accuracy=transition_run.accuracy if transition_run else "",
            )
        )
        for step in run.steps:
            posterior_row = AdvancedStatePosteriorRow(
                trajectory_id=run.trajectory_id,
                scenario_name=run.scenario_name,
                step=step.step,
                time=step.time,
                measurement=step.measurement,
                true_mode=step.true_mode,
                true_class=step.true_class,
                predicted_mode=step.predicted_mode,
                predicted_class=step.predicted_class,
                confidence=step.confidence,
                mode_entropy=step.mode_entropy,
                switch_event=step.switch_event,
                switch_detection_delay=run.switch_detection_delay,
                state_rmse=step.state_rmse,
                mode_posteriors=step.mode_posteriors,
                class_posteriors=step.class_posteriors,
            )
            posterior_rows.append(posterior_row)
            mode_probability_rows.append(
                AdvancedStateModeProbabilityRow(
                    trajectory_id=run.trajectory_id,
                    scenario_name=run.scenario_name,
                    step=step.step,
                    time=step.time,
                    measurement=step.measurement,
                    true_mode=step.true_mode,
                    true_class=step.true_class,
                    predicted_mode=step.predicted_mode,
                    predicted_class=step.predicted_class,
                    confidence=step.confidence,
                    mode_entropy=step.mode_entropy,
                    switch_event=step.switch_event,
                    switch_detection_delay=run.switch_detection_delay,
                    state_rmse=step.state_rmse,
                    mode_posteriors=step.mode_posteriors,
                    class_posteriors=step.class_posteriors,
                    mode_priors=step.mode_prior,
                )
            )
            for target_mode in mode_names:
                for source_mode in mode_names:
                    mixing_rows.append(
                        AdvancedStateMixingRow(
                            trajectory_id=run.trajectory_id,
                            scenario_name=run.scenario_name,
                            step=step.step,
                            time=step.time,
                            target_mode=target_mode,
                            source_mode=source_mode,
                            mixing_probability=step.mode_mixing[target_mode][source_mode],
                            mode_prior=step.mode_prior[target_mode],
                        )
                    )
                likelihood_rows.append(
                    AdvancedStateLikelihoodRow(
                        trajectory_id=run.trajectory_id,
                        scenario_name=run.scenario_name,
                        step=step.step,
                        time=step.time,
                        mode_name=target_mode,
                        true_mode=step.true_mode,
                        innovation=step.mode_innovations[target_mode][0],
                        innovation_variance=step.mode_innovation_variances[target_mode][0][0],
                        log_likelihood=step.mode_log_likelihoods[target_mode],
                        mode_posterior=step.mode_posteriors[target_mode],
                        mode_prior=step.mode_prior[target_mode],
                        predicted_mode=step.predicted_mode,
                    )
                )
                state_rows.append(
                    AdvancedStateEstimateRow(
                        trajectory_id=run.trajectory_id,
                        scenario_name=run.scenario_name,
                        step=step.step,
                        time=step.time,
                        estimate_type="mode",
                        model_name=target_mode,
                        true_mode=step.true_mode,
                        true_class=step.true_class,
                        switch_event=step.switch_event,
                        state_rmse=step.state_rmse,
                        state_mean=tuple(step.state_mean_by_mode[target_mode]),
                        state_cov_diag=tuple(
                            step.state_covariance_by_mode[target_mode][index][index]
                            for index in range(len(step.state_mean_by_mode[target_mode]))
                        ),
                    )
                )
            state_rows.append(
                AdvancedStateEstimateRow(
                    trajectory_id=run.trajectory_id,
                    scenario_name=run.scenario_name,
                    step=step.step,
                    time=step.time,
                    estimate_type="combined",
                    model_name="imm_combined",
                    true_mode=step.true_mode,
                    true_class=step.true_class,
                    switch_event=step.switch_event,
                    state_rmse=step.state_rmse,
                    state_mean=tuple(step.combined_state_mean),
                    state_cov_diag=tuple(
                        step.combined_state_covariance[index][index]
                        for index in range(len(step.combined_state_mean))
                    ),
                )
            )
            diagnostics_rows.append(
                AdvancedStateDiagnosticsRow(
                    trajectory_id=run.trajectory_id,
                    scenario_name=run.scenario_name,
                    step=step.step,
                    time=step.time,
                    true_mode=step.true_mode,
                    predicted_mode=step.predicted_mode,
                    switch_event=step.switch_event,
                    mode_entropy=step.mode_entropy,
                    state_rmse=step.state_rmse,
                    switch_detection_delay=run.switch_detection_delay,
                    innovation_energy=sum(abs(value[0]) for value in step.mode_innovations.values()),
                    confidence=step.confidence,
                )
            )

    write_csv(
        posterior_history_path,
        [_flatten_probability_row(row) for row in posterior_rows],
        [
            "trajectory_id",
            "scenario_name",
            "step",
            "time",
            "measurement",
            "true_mode",
            "true_class",
            "predicted_mode",
            "predicted_class",
            "confidence",
            "mode_entropy",
            "switch_event",
            "switch_detection_delay",
            "state_rmse",
            *[f"posterior_{name}" for name in mode_names],
            *[f"class_posterior_{name}" for name in class_names],
        ],
    )
    write_csv(
        mode_probability_history_path,
        [_flatten_mode_probability_row(row) for row in mode_probability_rows],
        [
            "trajectory_id",
            "scenario_name",
            "step",
            "time",
            "measurement",
            "true_mode",
            "true_class",
            "predicted_mode",
            "predicted_class",
            "confidence",
            "mode_entropy",
            "switch_event",
            "switch_detection_delay",
            "state_rmse",
            *[f"posterior_{name}" for name in mode_names],
            *[f"class_posterior_{name}" for name in class_names],
            *[f"mode_prior_{name}" for name in mode_names],
        ],
    )
    write_csv(
        mixing_probability_history_path,
        [asdict(row) for row in mixing_rows],
        [
            "trajectory_id",
            "scenario_name",
            "step",
            "time",
            "target_mode",
            "source_mode",
            "mixing_probability",
            "mode_prior",
        ],
    )
    write_csv(
        mode_likelihood_history_path,
        [asdict(row) for row in likelihood_rows],
        [
            "trajectory_id",
            "scenario_name",
            "step",
            "time",
            "mode_name",
            "true_mode",
            "innovation",
            "innovation_variance",
            "log_likelihood",
            "mode_posterior",
            "mode_prior",
            "predicted_mode",
        ],
    )
    write_csv(
        state_estimate_history_path,
        [_flatten_state_row(row) for row in state_rows],
        [
            "trajectory_id",
            "scenario_name",
            "step",
            "time",
            "estimate_type",
            "model_name",
            "true_mode",
            "true_class",
            "switch_event",
            "state_rmse",
            *[f"state_mean_{index}" for index in range(state_dim)],
            *[f"state_cov_diag_{index}" for index in range(state_dim)],
        ],
    )
    write_csv(
        diagnostics_history_path,
        [asdict(row) for row in diagnostics_rows],
        [
            "trajectory_id",
            "scenario_name",
            "step",
            "time",
            "true_mode",
            "predicted_mode",
            "switch_event",
            "mode_entropy",
            "state_rmse",
            "switch_detection_delay",
            "innovation_energy",
            "confidence",
        ],
    )
    write_csv(
        comparison_path,
        [asdict(row) for row in comparison_rows],
        [
            "trajectory_id",
            "scenario_name",
            "imm_accuracy",
            "imm_post_switch_accuracy",
            "imm_switch_delay",
            "imm_state_rmse",
            "transition_post_switch_accuracy",
            "transition_accuracy",
        ],
    )

    return AdvancedStateInferenceArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        config_path=config_path,
        mode_probability_history_path=mode_probability_history_path,
        mixing_probability_history_path=mixing_probability_history_path,
        mode_likelihood_history_path=mode_likelihood_history_path,
        state_estimate_history_path=state_estimate_history_path,
        posterior_history_path=posterior_history_path,
        diagnostics_history_path=diagnostics_history_path,
        comparison_path=comparison_path,
        mode_probability_plot_path=mode_probability_plot_path,
        mixing_probability_plot_path=mixing_probability_plot_path,
        mode_likelihood_plot_path=mode_likelihood_plot_path,
        state_estimate_plot_path=state_estimate_plot_path,
        switch_delay_plot_path=switch_delay_plot_path,
        comparison_plot_path=comparison_plot_path,
        plot_png_path=plot_png_path,
    )
