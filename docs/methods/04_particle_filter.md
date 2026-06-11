# Particle Filter

Particle filters represent Bayesian state uncertainty with weighted samples
rather than a single Gaussian state. They are appropriate when nonlinear
dynamics or non-Gaussian observation noise make Gaussian filters biased,
overconfident, or structurally mismatched.

## Role In The Classifier

Use PF only when the failure is nonlinear or non-Gaussian inference, not when
the primary bottleneck is missing sensor information. The primary promotion
witness is now `pf_abs_range_multimodal_oracle_v1`, where a non-injective
absolute-range measurement creates a genuinely multimodal posterior that a
single Gaussian cannot represent faithfully. The repo still keeps
`particle_filter_v1` as the shared trace/ESS packet and
`ornstein_uhlenbeck_mean_reversion_1d` as a stochastic-dynamics witness.

## Contract Hook

The canonical PF promotion witness emits:

- `artifacts/pf_abs_range_multimodal_oracle_v1/truth_trajectory.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/measurements.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/grid_oracle_posterior_history.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/method_posterior_history.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/gaussian_baseline_posterior_history.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/state_estimate_history.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/particle_diagnostics.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv`
- `artifacts/pf_abs_range_multimodal_oracle_v1/decision_card.md`
- `artifacts/pf_abs_range_multimodal_oracle_v1/plots/final_posterior_overlay.png`

The legacy supplemental trace packet remains:

- `artifacts/particle_filter_v1/posterior_history.csv`
- `artifacts/particle_filter_v1/state_estimate_history.csv`
- `artifacts/particle_filter_v1/pf_method_comparison.csv`

The shared classifier hook is `run_shared_particle_filter_classifier()`.

The claim boundary is deliberate: the oracle witness can support
`justified_for_study` for the multimodal posterior family once the particle
count and seed robustness sweep passes. It still does not justify PF as a
broad default outside that failure family.

## Research Note

Gordon, Salmond, and Smith introduced the bootstrap particle filter as a
recursive sample-based Bayesian estimator. Arulampalam, Maskell, Gordon, and
Clapp give the standard online nonlinear/non-Gaussian tracking tutorial.
