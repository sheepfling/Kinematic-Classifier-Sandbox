from __future__ import annotations

from dataclasses import asdict

from .contracts import RungCapabilitySpec


_CAPABILITY_SPECS: tuple[RungCapabilitySpec, ...] = (
    RungCapabilitySpec(
        rung_id="pointwise",
        rank=0,
        adds_capability="instantaneous likelihood",
        main_failure_addressed="no temporal memory",
        next_rung_id="windowed",
        complexity_cost=1.0,
        required_artifacts=("prediction_rows", "posterior_rows"),
    ),
    RungCapabilitySpec(
        rung_id="windowed",
        rank=1,
        adds_capability="short-history features",
        main_failure_addressed="noisy local observations",
        next_rung_id="sequential_bayes",
        complexity_cost=1.5,
        required_artifacts=("windowed_features", "posterior_rows"),
    ),
    RungCapabilitySpec(
        rung_id="sequential_bayes",
        rank=2,
        adds_capability="recursive posterior memory",
        main_failure_addressed="evidence needs accumulation",
        next_rung_id="kalman_bank",
        complexity_cost=2.0,
        required_artifacts=("posterior_history", "likelihood_history"),
    ),
    RungCapabilitySpec(
        rung_id="kalman_bank",
        rank=3,
        adds_capability="state-space prediction and innovation likelihood",
        main_failure_addressed="dynamics matter",
        next_rung_id="transition_matrix",
        complexity_cost=2.8,
        required_artifacts=("state_estimate_history", "innovation_history"),
    ),
    RungCapabilitySpec(
        rung_id="transition_matrix",
        rank=4,
        adds_capability="mode persistence and switching prior",
        main_failure_addressed="static class assumption fails",
        next_rung_id="imm",
        complexity_cost=3.5,
        required_artifacts=("mode_probability_history", "mixing_probability_history"),
    ),
    RungCapabilitySpec(
        rung_id="imm",
        rank=5,
        adds_capability="interacting switching state models",
        main_failure_addressed="state mixing across modes needed",
        next_rung_id="particle_filter",
        complexity_cost=4.5,
        required_artifacts=("mode_probability_history", "state_estimate_history", "diagnostics_history"),
    ),
    RungCapabilitySpec(
        rung_id="particle_filter",
        rank=6,
        adds_capability="sampled nonlinear and non-Gaussian posterior",
        main_failure_addressed="linear-Gaussian assumption fails",
        next_rung_id="rbpf",
        complexity_cost=6.0,
        required_artifacts=("particle_summary_history", "ess_history", "resampling_history"),
    ),
    RungCapabilitySpec(
        rung_id="rbpf",
        rank=7,
        adds_capability="sampled latent plus analytic conditional state",
        main_failure_addressed="mixed discrete/continuous latent structure",
        next_rung_id=None,
        complexity_cost=7.5,
        required_artifacts=("rbpf_particle_history", "conditional_filter_history", "ess_history"),
    ),
)


def capability_specs() -> tuple[RungCapabilitySpec, ...]:
    return _CAPABILITY_SPECS


def capability_lookup() -> dict[str, RungCapabilitySpec]:
    return {spec.rung_id: spec for spec in _CAPABILITY_SPECS}


def canonicalize_rung_id(classifier_id: str) -> str:
    value = classifier_id.strip()
    if value == "bayes_accumulator":
        return "sequential_bayes"
    if value.startswith("windowed_"):
        return "windowed"
    if value == "pointwise":
        return "pointwise"
    if value == "kalman_bank":
        return "kalman_bank"
    if value == "transition_matrix":
        return "transition_matrix"
    if value in {"imm", "particle_filter", "rbpf"}:
        return value
    return value


def next_rung_id(current_rung_id: str) -> str | None:
    spec = capability_lookup().get(current_rung_id)
    return spec.next_rung_id if spec is not None else None


def capability_rows() -> tuple[dict[str, object], ...]:
    rows = []
    for spec in _CAPABILITY_SPECS:
        row = asdict(spec)
        row["required_artifacts"] = " | ".join(spec.required_artifacts)
        rows.append(row)
    return tuple(rows)

