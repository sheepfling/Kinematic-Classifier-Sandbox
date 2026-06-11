# Gradient Boosted Features

The repo now has a dedicated engineered-feature witness for the
`gradient_boosted_features` lane:

- study id: `feature_headroom_frontier_v1`
- artifacts: `artifacts/feature_headroom_frontier_v1/`

## What It Proves

This witness targets a case where global windowed summaries leave headroom
inside the engineered feature space. The simpler baseline uses only coarse
track-level summaries. The candidate method trains a small boosted-stump model
over explicit engineered features such as early and late residual means.

The current witness is enough to justify:

- a nonlinear engineered-feature bridge baseline
- `gradient_boosted_features` moving from `researched` to `witness_supported`

## Claim Boundary

This is not yet a claim that the repo has full external gradient-boosting
parity, tuned boosting libraries, or broad superiority over stronger modern
time-series classifiers.

What remains open:

- broader corpus coverage beyond the dedicated headroom witness
- comparison against ROCKET-family and heavier TSC baselines on matched studies
- robustness sweeps over noise, segment timing, and feature-set variants
