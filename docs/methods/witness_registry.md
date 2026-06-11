# Witness Registry

Every method lane is justified by named witnesses, not by abstract ambition.

## Contract

Each witness should answer:

1. What simpler method fails?
2. What failure mode is present?
3. What candidate method is designed to represent it?
4. What traces, metrics, and plots would prove improvement?
5. What oracle or negative control should block overclaiming?

## Generated Registry

The generated witness registry bundle lives in:

- `artifacts/method_validation_os_v1/witness_specs.json`
- `artifacts/method_validation_os_v1/witness_to_method_coverage_matrix.csv`

## Core Witnesses

The initial operating-system registry tracks:

- `linear_gaussian_negative_control`
- `pointwise_overlap_prior_flip`
- `windowed_outlier_extrema`
- `shapelet_maneuver_motif`
- `transition_flicker_persistence`
- `duration_limited_maneuver`
- `unknown_maneuver_onset`
- `kalman_endpoint_match`
- `nonlinear_unimodal_sensor`
- `heavy_tail_outlier_tracking`
- `abs_range_multimodal_1d`
- `markov_switching_acceleration`
- `latent_maneuver_onset_duration`
- `learned_model_mismatch`
- `2d_range_bearing_geometry`
- `2d_clutter_association`

## Rule

Negative controls matter as much as positive witnesses. A method-validation
system is not credible unless it can reject unnecessary complexity.

The current `abs_range_multimodal_1d` registry entry is now backed by both PF
and GSF oracle witnesses. The next decision gate is not "PF or GSF by taste";
it is a robustness and complexity comparison on the same witness family.
