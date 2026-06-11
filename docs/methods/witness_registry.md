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
- `continuous_generator_frontier`
- `coverage_archive_diversity_frontier`
- `sequential_control_generator_frontier`
- `sequential_offpolicy_control_frontier`
- `embedding_baseline_frontier`
- `2d_range_bearing_geometry`
- `2d_clutter_association`

## Rule

Negative controls matter as much as positive witnesses. A method-validation
system is not credible unless it can reject unnecessary complexity.

The current `abs_range_multimodal_1d` registry entry is now backed by both PF
and GSF oracle witnesses. The next decision gate is not "PF or GSF by taste";
it is a robustness and complexity comparison on the same witness family.

The `nonlinear_unimodal_sensor` registry entry is now backed by
`ukf_nonlinear_unimodal_oracle_v1`, which fills the nonlinear Gaussian blocker
rung before broader PF claims.

The `heavy_tail_outlier_tracking` registry entry is now backed by
`student_t_heavy_tail_oracle_v1`, which fills the heavy-tail blocker rung
before PF claims are allowed to rely on outliers alone.

The `duration_limited_maneuver` registry entry is now backed by
`hsmm_duration_limited_maneuver_v1`, which fills the explicit dwell-time
blocker rung before BOCPD claims are needed.

The `unknown_maneuver_onset` registry entry is now backed by
`bocpd_unknown_maneuver_onset_v1`, which fills the explicit changepoint
blocker rung before richer latent-state claims.

The `shapelet_maneuver_motif` registry entry is now backed by
`shapelet_maneuver_motif_v1`, which fills the localized motif blocker rung
before stronger modern TSC claims are allowed to lean on short maneuver
signatures.

The `feature_headroom_frontier` registry entry is now backed by
`feature_headroom_frontier_v1`, which fills the low-risk nonlinear
engineered-feature blocker rung before stronger modern TSC claims skip over the
existing feature surface.

The `neural_sequence_vs_physics_frontier` registry entry now has a sequence
proxy packet, `neural_sequence_vs_physics_frontier_v1`, which keeps the neural
baseline lane explicit without overclaiming trained TCN or InceptionTime
fidelity.

The `tsc_archive_baseline_frontier` registry entry now has a modern-TSC proxy
packet, `tsc_archive_baseline_frontier_v1`, which keeps the archive-classifier
lane explicit without overclaiming faithful HIVE-COTE or MiniRocket-family
implementations.

The `confidence_calibration_shift` registry entry is now backed by
`confidence_calibration_shift_v1`, which fills the first calibration-wrapper
rung before broader coverage and abstention claims.

The `coverage_control_under_shift` registry entry is now backed by
`coverage_control_under_shift_v1`, which fills the first conformal coverage
control rung before broader abstention and sequential-conformal claims.

The `continuous_generator_frontier` registry entry is now backed by
`continuous_generator_frontier_v1`, which fills the first continuous-generator
promotion rung before broader search-budget and objective-family claims.

The `coverage_archive_diversity_frontier` registry entry is now backed by
`quality_diversity_corpus_v1`, which fills the first archive-diversity rung
before broader archive-policy and diversity-sweep claims.

The `sequential_control_generator_frontier` registry entry now has a PPO proxy
packet, `sequential_control_generator_frontier_v1`, which keeps the sequential-
control lane explicit without pretending SAC or TD3 has already been trained.

The `sequential_offpolicy_control_frontier` registry entry now has a matched-
budget smoke packet, `sequential_offpolicy_control_frontier_v1`, which gives
SAC and TD3 a concrete comparison surface against PPO and the existing control
baselines. The packet now includes a narrow seed sweep so the off-policy lane
has an explicit stability check, not just a single-run existence proof.

The `embedding_baseline_frontier` registry entry now has a first TS2Vec-style
embedding packet, `embedding_baseline_frontier_v1`, which keeps the
representation-learning lane explicit without pretending the external TS2Vec
library has already been trained in repo.

The `learning_evidence` lane now keeps supervised tabular baselines,
compact sequence learners, and unsupervised discovery visible as audited
evidence providers. Those methods remain subject to split discipline,
calibration checks, and hard failure reporting rather than being treated as a
generic accuracy leaderboard.

The `learned_model_mismatch` registry entry is still missing. The learned
filter lane has coverage rows, but it needs a trained witness packet before
KalmanNet or differentiable PF can be promoted.
