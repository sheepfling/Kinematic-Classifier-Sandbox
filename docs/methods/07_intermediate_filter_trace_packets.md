# Intermediate Filter Trace Packets

Intermediate trace packets make a filter explain each update before the final
decision card or hero chart is trusted.

## Canonical Row

The central row is `FilterStepTrace` in
`src/kinematic_classifier_sandbox/tracing/filter_trace.py`. It records:

- prior or transition-predicted probability before the measurement,
- measurement and optional prediction residuals,
- per-model likelihood or evidence contribution,
- posterior probability and entropy,
- optional state means, covariance diagonals, ESS, and resampling flags.

For pointwise and windowed methods many state fields may remain empty. For
Kalman, IMM, PF, and RBPF those fields should be populated as the backend
provides them.

## M0 Implementation

The M0 implementation adds:

| Surface | Purpose |
| --- | --- |
| `tracing/filter_trace.py` | Dataclass, CSV field order, tuple serialization, entropy helper. |
| `tracing/trace_schema.py` | Machine-readable schema for the trace artifact. |
| `tracing/trace_validation.py` | Per-row checks plus posterior/predicted probability normalization checks. |
| `render/intermediate_plots.py` | Posterior timeline, likelihood strip, and prior-likelihood-posterior waterfall. |
| `render/step_cards.py` | Markdown step-card rendering for selected update times. |

The first hooked backend is IMM:

- `artifacts/imm_filter_v1/traces/filter_step_trace.csv`
- `artifacts/imm_filter_v1/traces/per_method_diagnostics.csv`
- `artifacts/imm_filter_v1/plots/intermediate/posterior_timeline_with_regimes.png`
- `artifacts/imm_filter_v1/plots/intermediate/innovation_likelihood_strip.png`
- `artifacts/imm_filter_v1/plots/intermediate/prior_likelihood_posterior_waterfall.png`
- `artifacts/imm_filter_v1/plots/intermediate/mixing_probability_heatmap.png`
- `artifacts/imm_filter_v1/plots/intermediate/mode_conditioned_state_traces.png`
- `artifacts/imm_filter_v1/plots/intermediate/switch_recovery_panel.png`
- `artifacts/imm_filter_v1/step_cards/t_000.md`
- `artifacts/imm_filter_v1/step_cards/t_switch.md`
- `artifacts/imm_filter_v1/step_cards/t_final.md`

The second hooked benchmark is the transition/HMM rung:

- `artifacts/transition_matrix_accumulator_v1/traces/filter_step_trace.csv`
- `artifacts/transition_matrix_accumulator_v1/traces/per_method_diagnostics.csv`
- `artifacts/transition_matrix_accumulator_v1/plots/intermediate/posterior_timeline_with_regimes.png`
- `artifacts/transition_matrix_accumulator_v1/plots/intermediate/innovation_likelihood_strip.png`
- `artifacts/transition_matrix_accumulator_v1/plots/intermediate/prior_likelihood_posterior_waterfall.png`
- `artifacts/transition_matrix_accumulator_v1/plots/intermediate/static_vs_transition_flicker.png`
- `artifacts/transition_matrix_accumulator_v1/step_cards/t_switch.md`

The third hooked benchmark is the Kalman bank:

- `artifacts/kalman_filter_bank/traces/filter_step_trace.csv`
- `artifacts/kalman_filter_bank/traces/per_method_diagnostics.csv`
- `artifacts/kalman_filter_bank/plots/intermediate/measurement_prediction_timeline.png`
- `artifacts/kalman_filter_bank/plots/intermediate/innovation_likelihood_strip.png`
- `artifacts/kalman_filter_bank/plots/intermediate/uncertainty_diagnostics.png`
- `artifacts/kalman_filter_bank/step_cards/t_mid.md`

## Acceptance Rule

A method should not move past implementation-only status for a new study unless
its trace packet can answer:

- What was the prior?
- What was predicted?
- What was measured?
- How surprising was the measurement?
- How did likelihood update the posterior?
- Did uncertainty behave correctly?
- Where did the simpler method fail?
- Where did the advanced method fix it?

Robustness sweeps remain a separate gate; the trace packet explains one
controlled witness run before broader claims are made.
