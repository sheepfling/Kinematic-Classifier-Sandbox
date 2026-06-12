# Rao-Blackwellized Particle Filter

RBPF samples the part of the state that needs sampling and marginalizes the rest
with an exact conditional filter, such as a Kalman filter or finite HMM update.

## Role In The Classifier

Use RBPF when the witness has mixed latent discrete structure and continuous
state. The current witness family is `latent_maneuver_onset_1d`, and the
canonical promotion surface is `pf_vs_rbpf_frontier`.

## Contract Hook

The canonical RBPF promotion surface emits:

- `artifacts/advanced_filter_comparison_v1/pf_vs_rbpf_frontier.csv`
- `artifacts/advanced_filter_comparison_v1/pf_vs_rbpf_frontier_summary.csv`
- `artifacts/advanced_filter_comparison_v1/pf_vs_rbpf_frontier.png`
- `artifacts/advanced_filter_comparison_v1/rbpf_robustness_summary.csv`

The concrete RBPF witness packet emits:

- `artifacts/rbpf_v1/latent_mode_posterior.csv`
- `artifacts/rbpf_v1/conditional_filter_history.csv`
- `artifacts/rbpf_v1/summary.csv`
- `artifacts/rbpf_v1/traces/filter_step_trace.csv`
- `artifacts/rbpf_v1/plots/rbpf_mode_posterior.png`

The shared classifier hook is `run_shared_rbpf_classifier()`.

## Research Note

Doucet, de Freitas, Murphy, and Russell describe RBPF as sampling selected
variables while exactly marginalizing the rest using a finite-dimensional
optimal filter. In this repo, particles carry latent mode hypotheses while
continuous PVA state is conditionally Kalman-filtered.

The current claim boundary remains conservative. The frontier comparison against
PF now shows a bounded RBPF-preferred regime on the latent witness after the
shared frontier is scored by true post-onset recovery and includes lower
particle-count comparisons. The smooth witness remains `metric_split`, which is
the intended negative control. On that basis RBPF is now `study_justified` on
the current latent-structure witness family rather than only
`witness_supported`.
