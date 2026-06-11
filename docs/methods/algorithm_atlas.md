# Algorithm Atlas

The repo now tracks algorithms as lanes in a shared method-validation operating
system rather than as isolated additions.

## Lanes

| Lane | Role |
| --- | --- |
| transparent_kinematic_classifiers | Interpretable baselines and failure diagnostics |
| modern_time_series_classifiers | Strong classification baselines and accuracy ceilings |
| segmentation_regime_models | Unknown switch, duration, and maneuver-onset reasoning |
| state_space_filters | Physics-aware posterior, state, and uncertainty estimation |
| neural_sequence_models | Neural sequence baselines that stay separate from the proof ladder |
| learned_hybrid_filters | Future learned-model and differentiable filtering lane |
| uncertainty_calibration | Coverage, abstention, and calibration wrappers over evidence providers |
| exploration_generators | Search and control backends that generate witnesses and corpora |
| tracking_2d_plus | Future operational multi-target and clutter lane |

## Current Scope

The generated atlas bundle lives in:

- `artifacts/method_validation_os_v1/method_specs.json`
- `artifacts/method_validation_os_v1/algorithm_promotion_status_matrix.csv`
- `artifacts/method_validation_os_v1/witness_to_method_coverage_matrix.csv`

This atlas is intentionally broader than the current proof ladder. A method can
be researched and tracked here without being promoted in the classifier ladder.

## Immediate Gaps

The highest-value missing blockers before broader PF/RBPF and benchmark claims are:

- `UKF / EKF`
- `Student-t / robust Kalman`
- `HSMM`
- `BOCPD`
- `MiniRocket / MultiRocket / HYDRA`
- `shapelet / motif`
- `TCN / InceptionTime`
- `temperature scaling / conformal wrapper`
- `CMA-ES`

`Gaussian Sum Filter` has now moved from missing to an oracle-backed
`witness_supported` blocker lane, but it still needs robustness and
compute-normalized comparison against PF.

## Generator Registry

Expanded exploration options are tracked in
`src/kinematic_classifier_sandbox/corpus/trajectory_exploration/backend_registry.py`.
That scaffold keeps three things explicit:

- which generator backends are implemented today
- which ones are phase-1 or phase-2 benchmark candidates
- which ones require sequential control and should stay out of fixed-budget parameter-only comparisons
