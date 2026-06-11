# MAP-Elites

The repo now has a dedicated quality-diversity witness for the exploration
generator lane:

- study id: `quality_diversity_corpus_v1`
- artifacts: `artifacts/quality_diversity_corpus_v1/`

## What It Proves

This witness uses the shared CorpusGym archive surface and retains one elite
per behavior cell. It shows archive coverage growth over iterations and records
whether the archive improves feature excitation over the random-search
baseline.

The current witness is enough to justify:

- `map_elites` moving from `implemented` to `witness_supported`

## Claim Boundary

This is not yet a full quality-diversity research harness.

What remains open:

- broader archive-policy comparisons
- larger seed and iteration sweeps
- direct comparison to more sophisticated diversity search strategies
