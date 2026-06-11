# CMA-ES

The repo now has a dedicated continuous-generator frontier witness for the
exploration-generator lane:

- study id: `continuous_generator_frontier_v1`
- artifacts: `artifacts/continuous_generator_frontier_v1/`

## What It Proves

This witness uses the shared trajectory-exploration benchmark surface and
compares `cmaes` against `heuristic_search` and `blackbox_optimizer` on the
same fixed-budget objective family. The current packet is enough to show that
covariance-adapting search beats the heuristic baseline on mean total utility
without claiming that CMA-ES is the best backend for every objective.

The current witness is enough to justify:

- `cmaes` moving from `researched` to `witness_supported`

## Claim Boundary

This is not yet a full generator-selection workbench for all continuous search
budgets or all objective families.

What remains open:

- wider seed and budget sweeps
- broader objective-family coverage beyond the current frontier packet
- stronger compute-normalized comparisons against Bayesian optimization and
  sequential-control generators
