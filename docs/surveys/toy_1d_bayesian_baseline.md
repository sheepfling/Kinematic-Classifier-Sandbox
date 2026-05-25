# 1D Toy Bayesian Baseline

Status: superseded by the 1D witness-problem framing in
`docs/witnesses/index.md`. Keep this document for historical baseline detail,
but use "1D witness problem" in new narrative docs.

This toy baseline exists to validate the recursive Bayesian architecture before
the repo grows into full 3D track classification.

What it proves:

- class-matched filtering over noisy observations
- log-space class-posterior updates
- covariance-aware soft constraint likelihoods
- explicit unknown-class handling for out-of-bank dynamics

What it does not prove:

- 3D kinematics or realistic trajectory geometry
- within-class mode switching such as IMM behavior
- aerodynamic parameter estimation such as ballistic coefficient or `L/D`
- handling correlated upstream state estimates instead of raw observations
