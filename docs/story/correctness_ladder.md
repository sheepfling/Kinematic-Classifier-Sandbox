# Correctness Ladder

Correctness in this repo is layered. The goal is not to prove every algorithm is globally optimal. The goal is to prove that each method respects the shared contracts, invariants, witness cases, and claim boundaries that make the methodology workbench defensible.

## Levels

| Level | Purpose | Example |
| --- | --- | --- |
| L0 | Schema correctness | priors sum to 1, matrices are symmetric, packet fields exist |
| L1 | Invariant correctness | posteriors sum to 1, covariance stays PSD, ESS is bounded |
| L2 | Toy oracle correctness | two-class Gaussian separability, named 1D witnesses |
| L3 | Statistical regression | seeded CEM/PPO/PF/IMM runs stay within tolerance |
| L4 | Claim correctness | charts, packets, and decision cards do not overclaim |

## Epics

1. Static admissibility correctness
2. Classifier / filter ladder correctness
3. Corpus explorer correctness
4. Packet and claim correctness

## Command Surface

- `python -m kinematic_classifier_sandbox validate-correctness --level smoke`
- `python -m kinematic_classifier_sandbox validate-correctness --level full`
- `python -m kinematic_classifier_sandbox validate-correctness --level presentation`

## What Each Level Checks

- `smoke`: schemas, invariants, and the smallest admissibility gates.
- `full`: smoke plus toy witnesses and seeded regressions for the ladder and corpus surfaces.
- `presentation`: full plus packet and claim-boundary validators.

## Claim Boundary

The repo claims layered methodological correctness, not universal algorithm supremacy. Each method is evaluated through a shared contract and promoted only when invariants, witnesses, regression fixtures, and claim boundaries justify the added complexity.
