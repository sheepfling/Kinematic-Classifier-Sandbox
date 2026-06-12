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
That comparison is now partially closed for GSF: the narrow method packet
`gsf_multimodal_promotion_audit_v1` promotes GSF as the least-complex
multimodal blocker while still leaving PF as a separate costlier option on the
same witness family.

The `nonlinear_unimodal_sensor` registry entry is now backed by
`ukf_nonlinear_unimodal_oracle_v1`, which fills the nonlinear Gaussian blocker
rung before broader PF claims. That blocker is now partially closed at the
method level too: the narrow packet `ukf_nonlinear_promotion_audit_v1`
promotes UKF as the study-justified nonlinear Gaussian rung on the current
bounded witness family.

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

The `tsc_archive_baseline_frontier` registry entry now has a modern-TSC
execution packet, `tsc_archive_baseline_frontier_v1`, which keeps the
archive-classifier lane explicit without overclaiming faithful or finished
MiniRocket, DrCIF, dictionary-family, or HIVE-COTE implementations.

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

The `shapelet_maneuver_motif` registry entry now has a dedicated localized
motif witness packet, `shapelet_maneuver_motif_v1`, which keeps the short-
pattern lane explicit and makes the shapelet family first-class instead of a
roadmap-only note.

The `tsc_archive_baseline_frontier` registry entry now has a first-class modern
TSC archive frontier packet, `tsc_archive_baseline_frontier_v1`, which keeps
`minirocket_family`, `drcif_interval_forests`, `dictionary_tde_family`, and
`hive_cote` visible as implemented archive families with optional external
backend provenance, local fallback runs, and a bounded seed/calibration read
without claiming witness promotion.

The `archive_vs_physics_witness` registry entry now has a named shared-corpus
comparison packet, `archive_vs_physics_witness_v1`, which puts the archive
family rows directly against `windowed_robust` and `kalman_bank`. That packet
now provides a real positive bounded path: all archive families execute
externally, the bounded robustness/calibration read passes, and MiniRocket is
the current shared-corpus archive champion.

The `archive_feature_headroom_witness` registry entry now has a second named
archive comparison packet, `archive_feature_headroom_witness_v1`, which tests
the archive family on a timing-order witness against `windowed_feature_summary`
and `gradient_boosted_features`. The current bounded external archive rows now
match the engineered timing-order champion, which is enough to keep MiniRocket
credible without pretending the whole archive family is closed.

The `archive_backend_diagnosis` registry entry now has a bounded diagnosis
packet, `archive_backend_diagnosis_v1`, which sweeps panel variants, channel
sets, resample lengths, and warning load across the current archive witnesses.
That packet is now a bounded follow-on tuning surface rather than evidence that
the archive lane is fundamentally broken.

The `archive_family_promotion_audit` registry entry now has a bounded
method-level ranking packet, `archive_family_promotion_audit_v1`, which
identifies MiniRocket as the current archive family candidate and keeps the
family-level claim honest by separating public family proof from broader
member-level closure.

The `drcif_interval_promotion_audit` registry entry now has a narrow
method-level packet, `drcif_interval_promotion_audit_v1`, which explains why
DrCIF still remains below `witness_supported`: the current bounded evidence is
parity-level rather than a positive witness win. DrCIF is therefore now an
integrated explicit holdout, not an untracked generic-TSC family blocker.

The `neural_sequence_vs_physics_frontier` registry entry now has a trained
local neural packet, `neural_sequence_vs_physics_frontier_v1`, which keeps
`tcn` and `inceptiontime` visible as witness-backed sequence learners rather
than leaving them as future research placeholders.

The `neural_sequence_robustness_frontier` registry entry now has a bounded
multi-seed packet, `neural_sequence_robustness_frontier_v1`, which keeps the
learned-sequence lane from depending only on a single trained frontier run.
That packet is still bounded and does not by itself close the learned family.

The `physics_family_promotion_audit` registry entry now has a bounded
family-level packet, `physics_family_promotion_audit_v1`, which now records
that the advanced-filter blocker set is cleared even though the audit itself is
not a new witness. IMM, UKF, GSF, PF, and RBPF are now all study-justified on
their intended witness families, and HMM / transition plus the Kalman bank are
now witness-supported on their named foundation witnesses. The public
physics-aware family gate is therefore witness-backed on the current 1D surface
and no longer partial because of missing foundation-rung promotion.

The `embedding_baseline_frontier` registry entry now has a first TS2Vec-style
embedding packet, `embedding_baseline_frontier_v1`, which keeps the
representation-learning lane explicit and records the backend actually used on
that witness. That packet now also includes a prefix-based online route proof
against the current online-capable baselines.

The `ts2vec_backend_parity` registry entry now has a bounded parity packet,
`ts2vec_backend_parity_v1`, which compares the local proxy route against the
optional external TS2Vec backend on the same shared 1D corpus. That fills the
old external-fidelity gap at the witness level without turning TS2Vec into a
generalized or fully promoted family.

The `learning_evidence` lane now keeps supervised tabular baselines,
compact sequence learners, and unsupervised discovery visible as audited
evidence providers. Those methods remain subject to split discipline,
calibration checks, and hard failure reporting rather than being treated as a
generic accuracy leaderboard.

The `learned_model_mismatch` registry entry is still missing. The learned
filter lane has coverage rows, but it needs a trained witness packet before
KalmanNet or differentiable PF can be promoted.
