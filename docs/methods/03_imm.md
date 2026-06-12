# Interacting Multiple Model Filter

IMM is the switching-aware extension of the Kalman-bank rung. It keeps one
model-conditioned Kalman state per mode, mixes states through a Markov
transition model, and emits mode posterior evidence.

## Role In The Classifier

Use IMM when the witness has real switching dynamics and the simpler transition
or Kalman-bank rungs leave a switching failure unresolved. The repo witness is
`imm_switching_v1`.

## Contract Hook

Implementation lives in `src/kinematic_classifier_sandbox/advanced_filters`.
The emitted artifacts include:

- `artifacts/imm_filter_v1/posterior_history.csv`
- `artifacts/imm_filter_v1/state_estimate_history.csv`
- `artifacts/imm_filter_v1/mixing_probability_history.csv`
- `artifacts/imm_filter_v1/summary.csv`
- `artifacts/imm_filter_v1/advanced_filter_method_comparison.csv`
- `artifacts/imm_filter_v1/plots/intermediate/mixing_probability_heatmap.png`
- `artifacts/imm_filter_v1/plots/intermediate/mode_conditioned_state_traces.png`
- `artifacts/imm_filter_v1/plots/intermediate/switch_recovery_panel.png`

The current IMM witness is intentionally conservative in status semantics: it
now supports `study_justified` on the current switching family because the
switching witness, trace packet, and bounded promotion audit are all present.
The narrow packet
`artifacts/imm_switching_promotion_audit_v1/imm_switching_promotion_audit_report.md`
shows the switch-recovery gain survives a small seed-and-witness-size sweep.

## Research Note

Blom and Bar-Shalom introduced IMM for systems with Markovian switching
coefficients. The repo mirrors that structure at small scale: Markov mode
probabilities, model-conditioned Kalman updates, mixed state estimates, and
posterior-compatible evidence rows.
