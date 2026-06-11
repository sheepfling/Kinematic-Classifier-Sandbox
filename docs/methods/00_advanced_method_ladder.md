# Advanced Method Ladder

This workstream promotes advanced filters only through artifact-backed witness
evidence. It does not treat a complex backend as a better default.

The ladder is intentionally narrower than the full algorithm map. Modern
time-series baselines, neural sequence models, uncertainty wrappers, and
generator/search backends are tracked in parallel lanes, but they only interact
with the ladder through shared witnesses, posterior contracts, and decision
cards.

## Ladder

| Rung | Method | Intended failure mode | Witness | Current repo status |
| --- | --- | --- | --- | --- |
| 0 | Pointwise | no temporal context | `pointwise_overlap` | promoted baseline |
| 1 | Windowed | local outliers and extrema | `windowed_outlier_extrema` | case dependent |
| 2 | Sequential Bayes | pointwise ignores history | `sequential_history` | promoted baseline |
| 3 | Kalman bank | similar endpoints, different dynamics | `kalman_endpoint_match` | promoted baseline |
| 4 | Transition matrix / HMM | static class assumption under regime switching | `transition_switching` | promoted baseline |
| 5 | IMM | switching dynamics plus state uncertainty | `imm_switching_v1` | witness-specific `witness_supported` |
| 6 | Particle filter | nonlinear or non-Gaussian state evidence | `pf_abs_range_multimodal_oracle_v1`, `ornstein_uhlenbeck_mean_reversion_1d` | multimodal oracle family `justified_for_study`; other PF witnesses remain witness-specific |
| 7 | RBPF | mixed discrete mode path and continuous state | `latent_maneuver_onset_1d`, `pf_vs_rbpf_frontier` | witness-specific `witness_supported` |

## Status Levels

| Status | Meaning |
| --- | --- |
| `implemented` | Code exists and emits the shared advanced-filter artifact surface. |
| `witness_supported` | A named controlled witness exists, improves on that failure mode, and has an auditable packet. |
| `justified_for_study` | Reserved for witnesses whose improvement also survives robustness sweeps. |
| `generalized` | Reserved for broad evidence. Current advanced-filter rows should remain `no`. |

The canonical generated status table is
`artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv`.

## Architecture Rule

Filters estimate latent state. Evidence providers convert filter updates into
classifier evidence. The posterior updater, decision card, and artifact readers
stay shared.

The implementation surface is:

| Layer | Repo surface | Responsibility |
| --- | --- | --- |
| Filter state | `src/kinematic_classifier_sandbox/advanced_filters/*` | Maintain mode, particle, or Kalman-conditioned state. |
| Evidence row | `AdvancedFilterStep` | Emit posterior-compatible label probabilities, log evidence, confidence, and diagnostics. |
| State summary | `AdvancedStateSummary` | Emit state mean, covariance, and backend diagnostics. |
| Witness runner | `scripts/run/run_*_witness.py` | Materialize named controlled failure cases. |
| Comparison artifact | `write_advanced_filter_comparison_artifacts()` | Join witness metrics into method and gate matrices. |
| Shared evaluation hook | `shared_classifier_methods.py` | Expose PF/RBPF as capability-aware shared classifiers without forcing broad leaderboard claims. |

## Research Backbone

The method choices follow standard filtering literature:

| Method | Reference anchor | Why it belongs |
| --- | --- | --- |
| HMM / transition matrix | Rabiner, 1989 | Canonical forward update over hidden discrete regimes. |
| Kalman bank | Kalman, 1960; tracking literature | Model-specific innovations become dynamics evidence. |
| IMM | Blom and Bar-Shalom, 1988 | Multiple Kalman filters with Markov switching and state mixing. |
| Particle filter | Gordon, Salmond, and Smith, 1993; Arulampalam et al., 2002 | Sequential Monte Carlo for nonlinear or non-Gaussian Bayesian filtering. |
| RBPF | Doucet, de Freitas, Murphy, and Russell, 2000 | Samples only part of the state and marginalizes the rest with exact conditional filters. |

## Claim Boundary

Advanced filters are ambitious in implementation and conservative in claims.
The repo can say IMM, PF, and RBPF are implemented and witness-supported for
their named studies. It should not claim they are globally better than simpler
rungs, or even justified-for-study beyond the named witness family, until
robustness sweeps and broader corpora support that.
