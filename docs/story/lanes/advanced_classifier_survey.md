# Advanced Classifier Survey Lane

## Purpose

The survey lane organizes classifier and filter families by evidence
capability, assumptions, failure modes, and promotion gates.

It keeps the algorithm catalog from becoming a leaderboard. Each method is
framed as a way to construct evidence for the shared posterior contract.

The lane thesis is:

> The ladder is not a ranking. It is a sequence of evidence capabilities.

## Outputs

- `docs/story/algorithm_ladder.md`
- `docs/story/algorithm_map.md`
- `artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv`
- `artifacts/algorithm_coverage_matrix/algorithm_coverage_matrix.csv`

## Claim Boundary

Method coverage is not method promotion. Every rung must consume the same
tracklet surface and emit comparable posterior histories. IMM, PF, and RBPF are
promoted only for named witness regimes and should not be treated as global
defaults.

## Next Work

- Expand method assumption cards and witness cards.
- Add vector PVA versions of the current 1D witness checks.
- Keep advanced filters tied to simpler-rung failure evidence.
- Surface rung sufficiency decisions as the main output, not leaderboard rank.
