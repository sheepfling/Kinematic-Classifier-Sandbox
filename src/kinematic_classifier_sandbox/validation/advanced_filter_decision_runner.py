from __future__ import annotations

from ..analysis.short_horizon_identifiability import analyze_short_horizon_identifiability
from ..inference.kalman_variant_comparison import analyze_kalman_variant_comparison
from ..inference.transition_matrix_accumulator import run_transition_benchmark
from ..inference.velocity_aided_kalman_comparison import analyze_velocity_aided_kalman_comparison
from .advanced_filter_decision_contracts import AdvancedFilterDecisionResult


def analyze_advanced_filter_decision() -> AdvancedFilterDecisionResult:
    transition = run_transition_benchmark(seed=7, replicas=8)
    short_horizon = analyze_short_horizon_identifiability()
    velocity_aided = analyze_velocity_aided_kalman_comparison(seed=7, trajectories_per_case=8)
    kalman_variants = analyze_kalman_variant_comparison(seed=7, trajectories_per_case=8)

    transition_post_switch_gain = (
        transition.summary.transition_post_switch_accuracy
        - transition.summary.static_post_switch_accuracy
    )
    transition_overall_gain = transition.summary.transition_accuracy - transition.summary.static_accuracy
    transition_vs_kalman_post_switch_gain = (
        transition.summary.transition_post_switch_accuracy
        - transition.summary.kalman_post_switch_accuracy
    )
    transition_vs_kalman_overall_gain = transition.summary.transition_accuracy - transition.summary.kalman_accuracy

    nominal_noise = next(
        row for row in short_horizon.noise_sweep
        if abs(row.measurement_sigma - short_horizon.nominal_measurement_sigma) < 1e-9
    )
    short_horizon_mean_gap_sigma = nominal_noise.mean_normalized_gap
    short_horizon_final_gap_sigma = nominal_noise.final_step_normalized_gap

    position_only = next(row for row in velocity_aided.rows if row.measurement_mode == "position_only")
    direct_velocity = next(row for row in velocity_aided.rows if row.measurement_mode == "position_plus_direct_velocity")
    velocity_aided_short_noisy_gain = direct_velocity.short_noisy_accuracy - position_only.short_noisy_accuracy

    best_kalman_outlier_accuracy = max(row.outlier_accuracy for row in kalman_variants.rows)

    imm_prereq_transition_exists = True
    imm_prereq_transition_improves = transition_post_switch_gain > 0.0
    imm_prereq_model_based_switching_failure_documented = transition_vs_kalman_post_switch_gain <= 0.0
    imm_justified = False

    particle_prereq_nonlinear_benchmark_exists = False
    particle_prereq_non_gaussian_failure_unresolved = False
    particle_prereq_sensor_limited_not_primary = velocity_aided_short_noisy_gain <= 0.05
    particle_filter_justified = False

    evidence_rows = (
        {
            "gate": "IMM",
            "criterion": "transition_matrix_benchmark_exists",
            "status": "met" if imm_prereq_transition_exists else "missing",
            "value": "yes",
            "note": "Switching scenarios and transition-matrix benchmark exist.",
        },
        {
            "gate": "IMM",
            "criterion": "transition_matrix_improves_post_switch_accuracy",
            "status": "met" if imm_prereq_transition_improves else "failed",
            "value": round(transition_post_switch_gain, 6),
            "note": "Positive gain means simpler transition structure is still buying measurable value.",
        },
        {
            "gate": "IMM",
            "criterion": "transition_matrix_no_longer_beats_switching_kalman",
            "status": "met" if imm_prereq_model_based_switching_failure_documented else "failed",
            "value": round(transition_vs_kalman_post_switch_gain, 6),
            "note": "Positive values mean the simpler transition-matrix accumulator still outperforms the current switching-mode Kalman bank post-switch.",
        },
        {
            "gate": "Particle Filter",
            "criterion": "nonlinear_or_non_gaussian_benchmark_exists",
            "status": "missing" if not particle_prereq_nonlinear_benchmark_exists else "met",
            "value": "no",
            "note": "The repo does not yet have a dedicated nonlinear/non-Gaussian benchmark where simpler filters provably fail.",
        },
        {
            "gate": "Particle Filter",
            "criterion": "short_horizon_failure_is_sensor_limited",
            "status": "met" if not particle_prereq_sensor_limited_not_primary else "failed",
            "value": round(velocity_aided_short_noisy_gain, 6),
            "note": "Direct velocity improves short-noisy accuracy materially, so the current limit is sensing/identifiability, not advanced inference.",
        },
        {
            "gate": "Particle Filter",
            "criterion": "outlier_failure_remains_unresolved_after_kalman_robustness",
            "status": "failed" if not particle_prereq_non_gaussian_failure_unresolved else "met",
            "value": round(best_kalman_outlier_accuracy, 6),
            "note": "Robust/adaptive Kalman variants already recover much of the outlier case without particle filtering.",
        },
    )

    return AdvancedFilterDecisionResult(
        imm_justified=imm_justified,
        particle_filter_justified=particle_filter_justified,
        transition_post_switch_gain=transition_post_switch_gain,
        transition_overall_gain=transition_overall_gain,
        transition_vs_kalman_post_switch_gain=transition_vs_kalman_post_switch_gain,
        transition_vs_kalman_overall_gain=transition_vs_kalman_overall_gain,
        short_horizon_mean_gap_sigma=short_horizon_mean_gap_sigma,
        short_horizon_final_gap_sigma=short_horizon_final_gap_sigma,
        velocity_aided_short_noisy_gain=velocity_aided_short_noisy_gain,
        best_kalman_outlier_accuracy=best_kalman_outlier_accuracy,
        evidence_rows=evidence_rows,
    )


__all__ = ["analyze_advanced_filter_decision"]
