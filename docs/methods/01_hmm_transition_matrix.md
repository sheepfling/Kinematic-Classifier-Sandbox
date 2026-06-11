# HMM And Transition Matrix

The transition-matrix rung is the discrete-regime bridge between sequential
Bayes and advanced switching filters. It propagates class or mode posterior
mass through a Markov transition matrix before applying per-step evidence.

## Role In The Classifier

Use this rung when the failure is posterior flicker or delayed recovery caused
by a static class assumption. The witness is `transition_switching`.

## Contract

- Inputs: prior mode probabilities, per-mode evidence, transition matrix.
- Outputs: posterior history rows under the shared classifier/evidence contract.
- Gate: transition logic must beat the static accumulator on switching metrics
  before IMM is considered.

The intermediate packet for this rung now includes:

- `artifacts/transition_matrix_accumulator_v1/traces/filter_step_trace.csv`
- `artifacts/transition_matrix_accumulator_v1/plots/intermediate/posterior_timeline_with_regimes.png`
- `artifacts/transition_matrix_accumulator_v1/plots/intermediate/prior_likelihood_posterior_waterfall.png`
- `artifacts/transition_matrix_accumulator_v1/step_cards/t_switch.md`

## Research Note

Rabiner's HMM tutorial frames hidden-state sequence inference around likelihood
of observations, best state sequence, and parameter adjustment. This repo uses
the forward-update portion as a disciplined transition-aware classifier rung.
