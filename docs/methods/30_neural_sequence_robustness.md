# Neural Sequence Robustness Frontier

The repo now has a bounded robustness companion for the learned-sequence lane:

- study id: `neural_sequence_robustness_frontier_v1`
- artifacts: `artifacts/neural_sequence_robustness_frontier_v1/`

## What It Proves

This packet exists to answer a narrower question than the main neural frontier:

Are the trained `tcn` and `inceptiontime` rows only a single-seed accident, or
do they at least show a bounded multi-seed candidate signal on the shared 1D
corpus?

The packet aggregates:

- mean test accuracy
- accuracy variance across seeds
- scenario slices
- seed-winner counts

for:

- `tcn`
- `inceptiontime`
- `windowed_robust`
- `rocket_proxy`
- `kalman_bank`

## Current Read

The current bounded read is positive but still conservative:

- the neural lane now has a multi-seed robustness packet rather than only a
  single-seed frontier
- the bounded packet can show a neural candidate signal
- that still does not count as broad learned-sequence closure for Epic 2

## Claim Boundary

This is a bounded robustness packet, not a broad generalization or promotion
claim.

It is enough to improve the learned-sequence evidence story by proving that the
lane is not represented only by one trained frontier run.

What remains open:

- broader seed and corpus coverage
- stronger comparisons against named physics-aware witnesses
- family-level closure for the learned sequence and embedding lane
