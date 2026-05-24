from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .advanced_filter_decision import analyze_advanced_filter_decision
from .common_dataset_comparison import generate_shared_dynamics_dataset
from .kalman_filter_bank import default_kalman_model_specs, run_kalman_bank_benchmark, run_kalman_filter_bank
from .transition_matrix_accumulator import run_transition_benchmark


@dataclass(frozen=True, slots=True)
class GenericFilteringContractResult:
    filter_backend_contract: dict[str, object]
    filter_diagnostics_schema: dict[str, object]
    filtering_principles_report: str
    particle_filter_decision_report: str
    rbpf_decision_report: str


@dataclass(frozen=True, slots=True)
class GenericFilteringContractArtifacts:
    run_dir: Path
    filter_backend_contract_path: Path
    filter_diagnostics_schema_path: Path
    filtering_principles_report_path: Path
    particle_filter_decision_report_path: Path
    rbpf_decision_report_path: Path


def _filter_backend_contract() -> dict[str, object]:
    return {
        "backend_interface": {
            "required_run_fields": [
                "trajectory_id",
                "scenario_name",
                "steps",
                "final_weights",
                "final_predicted_class_or_mode",
                "final_confidence_or_accuracy",
            ],
            "required_step_fields": [
                "time",
                "measurement",
                "posterior_weights",
                "predicted_label",
                "confidence",
            ],
            "optional_step_fields": [
                "log_likelihood_terms",
                "innovations",
                "innovation_variances",
                "prior_weights",
                "estimated_speed",
                "estimated_accel",
            ],
            "optional_backend_outputs": [
                "state means",
                "state covariances",
                "model probabilities",
                "effective sample size",
                "resampling count",
            ],
        },
        "reference_backends": [
            "kalman_bank",
            "kalman_mode_bank",
        ],
    }


def _filter_diagnostics_schema() -> dict[str, object]:
    return {
        "innovation_backend_fields": [
            "innovations",
            "innovation_variances",
            "log_likelihood_terms",
            "posterior_weights",
        ],
        "state_backend_fields": [
            "state mean",
            "state covariance",
        ],
        "switching_backend_fields": [
            "prior_weights",
            "posterior_weights",
            "predicted_mode",
            "true_mode",
        ],
        "future_particle_backend_fields": [
            "effective_sample_size",
            "resampling_count",
            "particle_diversity",
        ],
    }


def _validate_kalman_backend() -> dict[str, object]:
    benchmark = run_kalman_bank_benchmark(seed=7, trajectories_per_class=2)
    run = benchmark.runs[0]
    step_contract_passed = all(
        abs(sum(step.posterior_weights.values()) - 1.0) <= 1e-6
        and set(step.posterior_weights) == set(step.log_likelihood_terms)
        and set(step.posterior_weights) == set(step.innovations)
        and set(step.posterior_weights) == set(step.innovation_variances)
        for step in run.steps
    )
    state_contract_passed = all(
        len(state.mean) == 3
        and len(state.covariance) == 3
        and all(len(row) == 3 for row in state.covariance)
        for state in run.final_states.values()
    )
    return {
        "backend": "kalman_bank",
        "num_runs_checked": len(benchmark.runs),
        "step_contract_passed": step_contract_passed,
        "state_contract_passed": state_contract_passed,
        "final_accuracy": benchmark.summary.final_accuracy,
    }


def _validate_switching_backend() -> dict[str, object]:
    benchmark = run_transition_benchmark(seed=7, replicas=4)
    run = benchmark.kalman_runs[0]
    step_contract_passed = all(
        abs(sum(step.posterior_weights.values()) - 1.0) <= 1e-6
        and set(step.prior_weights) == set(step.posterior_weights)
        and 0.0 <= step.confidence <= 1.0
        for step in run.steps
    )
    return {
        "backend": "kalman_mode_bank",
        "num_runs_checked": len(benchmark.kalman_runs),
        "step_contract_passed": step_contract_passed,
        "post_switch_accuracy": benchmark.summary.kalman_post_switch_accuracy,
    }


def _render_particle_filter_decision_report() -> str:
    decision = analyze_advanced_filter_decision()
    return "\n".join(
        [
            "# Particle Filter Decision Report",
            "",
            f"- Justified now: `{decision.particle_filter_justified}`",
            "",
            "## What Would Be Sampled",
            "",
            "- Nonlinear or event-timing latent variables that cannot be represented well by the current linear-Gaussian backends.",
            "",
            "## What Simpler Method Still Fails",
            "",
            "- No dedicated nonlinear or non-Gaussian benchmark is yet present where transition-aware or robust Kalman-style methods provably fail.",
            "",
            "## Current Evidence",
            "",
            f"- Velocity-aided short-noisy gain: `{decision.velocity_aided_short_noisy_gain:.3f}`",
            f"- Best Kalman outlier accuracy: `{decision.best_kalman_outlier_accuracy:.3f}`",
            "",
            "## Decision Rule",
            "",
            "- Defer particle filtering until a documented nonlinear or non-Gaussian benchmark exists and simpler methods fail for reasons other than sensing limits or identifiability.",
        ]
    )


def _render_rbpf_decision_report() -> str:
    decision = analyze_advanced_filter_decision()
    return "\n".join(
        [
            "# Rao-Blackwell Particle Filter Decision Report",
            "",
            "- Justified now: `False`",
            "",
            "## What Would Be Sampled",
            "",
            "- Discrete or strongly nonlinear latent structure such as maneuver timing, mode sequence, or event-trigger variables.",
            "",
            "## What Would Be Marginalized Analytically",
            "",
            "- Conditional linear-Gaussian continuous state such as position, velocity, and acceleration under a fixed sampled mode path.",
            "",
            "## Why Kalman Bank Or Transition Methods Are Still Enough",
            "",
            f"- Transition vs Kalman post-switch gain is still positive: `{decision.transition_vs_kalman_post_switch_gain:.3f}`",
            "- The repo still lacks a case where sampled latent structure plus conditional analytic filtering is clearly required.",
            "",
            "## Decision Rule",
            "",
            "- Defer RBPF until the repo can name the sampled latent variables, the marginalized conditional state, and a benchmark proving that this split outperforms simpler transition-aware filtering.",
        ]
    )


def render_generic_filtering_principles_report(
    *,
    filter_backend_contract: dict[str, object],
    filter_diagnostics_schema: dict[str, object],
    kalman_validation: dict[str, object],
    switching_validation: dict[str, object],
) -> str:
    return "\n".join(
        [
            "# Generic Filtering Contract",
            "",
            "This artifact proves that current filter backends can be described through a shared state/evidence/diagnostics contract instead of backend-specific ad hoc outputs.",
            "",
            "## Contract Summary",
            "",
            f"- Reference backends: `{', '.join(filter_backend_contract['reference_backends'])}`",
            f"- Required run fields: `{', '.join(filter_backend_contract['backend_interface']['required_run_fields'])}`",
            f"- Required step fields: `{', '.join(filter_backend_contract['backend_interface']['required_step_fields'])}`",
            f"- Optional backend outputs: `{', '.join(filter_backend_contract['backend_interface']['optional_backend_outputs'])}`",
            "",
            "## Validation",
            "",
            f"- Kalman backend step contract passed: `{kalman_validation['step_contract_passed']}`",
            f"- Kalman backend state contract passed: `{kalman_validation['state_contract_passed']}`",
            f"- Kalman benchmark final accuracy: `{kalman_validation['final_accuracy']:.3f}`",
            f"- Switching Kalman-mode backend step contract passed: `{switching_validation['step_contract_passed']}`",
            f"- Switching Kalman-mode post-switch accuracy: `{switching_validation['post_switch_accuracy']:.3f}`",
            "",
            "## Diagnostics Schema",
            "",
            f"- Innovation backend fields: `{', '.join(filter_diagnostics_schema['innovation_backend_fields'])}`",
            f"- State backend fields: `{', '.join(filter_diagnostics_schema['state_backend_fields'])}`",
            f"- Switching backend fields: `{', '.join(filter_diagnostics_schema['switching_backend_fields'])}`",
            f"- Future particle backend fields: `{', '.join(filter_diagnostics_schema['future_particle_backend_fields'])}`",
            "",
            "## Notes",
            "",
            "- Kalman is the first concrete state-estimation backend under this contract.",
            "- The switching Kalman-mode bank shows that a second backend-style surface can still fit the same posterior/diagnostic structure.",
            "- Particle and Rao-Blackwell particle filtering remain contract-aware future backends, not current implementations.",
        ]
    )


def analyze_generic_filtering_contract() -> GenericFilteringContractResult:
    filter_backend_contract = _filter_backend_contract()
    filter_diagnostics_schema = _filter_diagnostics_schema()
    kalman_validation = _validate_kalman_backend()
    switching_validation = _validate_switching_backend()
    filtering_principles_report = render_generic_filtering_principles_report(
        filter_backend_contract=filter_backend_contract,
        filter_diagnostics_schema=filter_diagnostics_schema,
        kalman_validation=kalman_validation,
        switching_validation=switching_validation,
    )
    particle_filter_decision_report = _render_particle_filter_decision_report()
    rbpf_decision_report = _render_rbpf_decision_report()
    return GenericFilteringContractResult(
        filter_backend_contract={
            **filter_backend_contract,
            "validation": {
                "kalman": kalman_validation,
                "switching_kalman_mode_bank": switching_validation,
            },
        },
        filter_diagnostics_schema=filter_diagnostics_schema,
        filtering_principles_report=filtering_principles_report,
        particle_filter_decision_report=particle_filter_decision_report,
        rbpf_decision_report=rbpf_decision_report,
    )


def write_generic_filtering_contract_artifacts(
    output_dir: str | Path,
    *,
    result: GenericFilteringContractResult | None = None,
) -> GenericFilteringContractArtifacts:
    contract = result or analyze_generic_filtering_contract()
    run_dir = Path(output_dir) / "filtering_contract"
    run_dir.mkdir(parents=True, exist_ok=True)

    filter_backend_contract_path = run_dir / "filter_backend_contract.json"
    filter_diagnostics_schema_path = run_dir / "filter_diagnostics_schema.json"
    filtering_principles_report_path = run_dir / "filtering_principles_report.md"
    particle_filter_decision_report_path = run_dir / "particle_filter_decision_report.md"
    rbpf_decision_report_path = run_dir / "rbpf_decision_report.md"

    filter_backend_contract_path.write_text(json.dumps(contract.filter_backend_contract, indent=2), encoding="utf-8")
    filter_diagnostics_schema_path.write_text(json.dumps(contract.filter_diagnostics_schema, indent=2), encoding="utf-8")
    filtering_principles_report_path.write_text(contract.filtering_principles_report, encoding="utf-8")
    particle_filter_decision_report_path.write_text(contract.particle_filter_decision_report, encoding="utf-8")
    rbpf_decision_report_path.write_text(contract.rbpf_decision_report, encoding="utf-8")

    return GenericFilteringContractArtifacts(
        run_dir=run_dir,
        filter_backend_contract_path=filter_backend_contract_path,
        filter_diagnostics_schema_path=filter_diagnostics_schema_path,
        filtering_principles_report_path=filtering_principles_report_path,
        particle_filter_decision_report_path=particle_filter_decision_report_path,
        rbpf_decision_report_path=rbpf_decision_report_path,
    )
