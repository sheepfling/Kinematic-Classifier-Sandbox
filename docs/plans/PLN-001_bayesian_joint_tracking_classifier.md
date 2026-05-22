# PLN-001 Bayesian Joint Tracking Classifier

Title: Bayesian Joint Tracking and Classification with Class-Conditioned Physical Constraints
Plan ID: PLN-001
Status: proposed
Owner: @rick
Priority: P1
Objective: Establish the repo's first implementation plan around Bayesian joint target tracking and classification using class-matched filters, class-conditioned validity likelihoods, optional aerodynamic parameter evidence, and an explicit unknown-class pathway.
Scope:
- Define the baseline probabilistic architecture for online class inference from PVA-style tracks.
- Specify the initial class bank, within-class mode switching, and likelihood composition strategy.
- Identify the first code modules, validation targets, and artifacts needed for implementation.
Out of Scope:
- Full dataset ingestion pipelines.
- Large-scale benchmark orchestration.
- Learned discriminative models beyond optional comparison baselines.
- Production-grade sensor fusion or multi-target association.
Implementation Steps:
1. Formalize the state, class, mode, and optional parameter variables for the sandbox baseline.
2. Implement a class-matched filter-bank interface with per-class dynamic hypotheses and likelihood hooks.
3. Add an IMM-style within-class mode layer for maneuver switching such as climb, coast, turn, glide, or ballistic phases.
4. Implement soft constraint likelihoods for speed, altitude, and acceleration envelopes using state covariance rather than hard thresholds.
5. Add optional parameter-evidence hooks for ballistic coefficient, lift-to-drag ratio, turn behavior, and thrust/drag proxies.
6. Introduce an explicit unknown class with nonzero prior mass and broad fallback dynamics.
7. Add temporal smoothing and guardrails against double-counting correlated upstream track estimates.
8. Create synthetic validation scenarios spanning feasible, borderline, and out-of-envelope tracks.
Validation:
- Unit tests for class-weight normalization, log-space likelihood composition, and unknown-class retention.
- Deterministic scenario tests showing recovery from early misclassification under a mostly diagonal class-transition matrix.
- Constraint-probability tests covering scalar Gaussian CDF cases and nonlinear approximations.
- Synthetic track studies demonstrating separation across ballistic, glide, powered, hover, and high-turn regimes.
Artifacts / Config:
- Survey notes in `docs/surveys/`.
- Method catalog entries in `src/kinematic_classifier_sandbox/catalog.py`.
- Generated markdown summary in `artifacts/method_survey_summary.md`.
- Future implementation modules under `src/kinematic_classifier_sandbox/`.
Dependencies:
- Baseline motion-model definitions for CV, CA, coordinated-turn, glide, and ballistic behaviors.
- Covariance-aware envelope probability utilities.
- Synthetic scenario generator for regression tests.
- Optional atmospheric and gravity helpers for drag-residual feature estimation.
Last Updated: 2026-05-21
