# Witness: transition_switching

Purpose: Show that mode-transition evidence is justified before considering IMM-style methods.

Class set: switching scenarios from `artifacts/transition_matrix_accumulator_v1/transition_matrix_dataset_manifest.json`.

Feature set: mode posterior and transition evidence history.

Classifier/filter family: transition matrix accumulator.

Prior regime: transition matrix priors in `artifacts/transition_matrix_accumulator_v1/transition_matrix_config.yaml`.

Corpus objective: switching trajectories that violate a static class assumption.

What it proves: transition logic can encode evidence that a static class accumulator cannot.

What it does not prove: full IMM, PF, or RBPF state inference is implemented or unnecessary.

Key equations: `T_ij` transition update and normalized posterior over modes.

Key plots:
- `artifacts/transition_matrix_accumulator_v1/transition_matrix_diagnostics.png`
- `artifacts/showcase/plots/transition_matrix_diagnostics.png`

Key tables:
- `artifacts/transition_matrix_accumulator_v1/transition_matrix_scenario_summary.csv`
- `artifacts/transition_matrix_accumulator_v1/transition_matrix_posterior_history.csv`

Key artifacts:
- `artifacts/transition_matrix_accumulator_v1/transition_matrix_numeric_walkthrough.md`
- `artifacts/advanced_filter_decision_v1/advanced_filter_decision_evidence.json`

Promotion status: pass as the switching-logic witness; advanced filters remain gated.

Next extension toward 3D: connect mode transitions to 3D maneuver states and evaluate IMM only after failure evidence demands it.
