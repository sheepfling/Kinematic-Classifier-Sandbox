# Method Selection Gates

Advanced methods are selected by evidence gates, not by aspiration.

| Gate | Required evidence |
| --- | --- |
| Previous rung failure | A simpler rung fails on a named witness and metric. |
| Assumption match | The witness contains the structure the method targets. |
| Controlled comparison | Same corpus, seed policy, prior regime, and evaluation surface. |
| Measurable improvement | Accuracy, NLL, calibration, RMSE, ESS, switch delay, or posterior margin improves. |
| Robustness | Improvement survives seed, noise, and prior sensitivity sweeps. |
| Complexity accounting | Runtime, particle count, or model count is reported. |
| Intermediate trace packet | Step traces, diagnostics, intermediate plots, and selected step cards exist. |
| Decision-card trace | The result appears in generated comparison and gate artifacts. |

The current comparison artifact records robustness as `not_yet` for advanced
filters. That is intentional: a witness win plus trace packet yields
`witness_supported`, not `justified_for_study`. The stronger status is reserved
for methods whose improvement survives robustness sweeps.

## Canonical Artifacts

- `artifacts/advanced_filter_comparison_v1/method_comparison.csv`
- `artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv`
- `artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.json`
- `artifacts/advanced_filter_comparison_v1/advanced_filter_decision_matrix.csv`
- `artifacts/advanced_filter_comparison_v1/particle_filter_robustness_summary.csv`
- `artifacts/advanced_filter_comparison_v1/rbpf_robustness_summary.csv`
- `artifacts/imm_filter_v1/traces/filter_step_trace.csv`
- `artifacts/imm_filter_v1/step_cards/t_switch.md`

## Interpretation Rule

`advanced_filter_decision_v1` is a conservative historical escalation gate for
whether the older ladder needed IMM or PF before the dedicated witnesses
existed. `advanced_filter_comparison_v1` is the current witness-specific
promotion surface for implemented IMM, PF, RBPF, and OU-style PF studies.
