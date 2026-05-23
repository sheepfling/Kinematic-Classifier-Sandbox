# PLN-002 Kinematic Classification Roadmap

Title: Kinematic Classification Lab Roadmap
Plan ID: PLN-002
Status: proposed
Owner: @rick
Priority: P1
Objective: Turn the sandbox into a repeatable kinematic-classification lab with explicit artifact contracts, reproducible synthetic datasets, staged classifiers, identifiability analysis, and decision gates for when to advance to more complex models.
Scope:
- Define shared artifact contracts for trajectories, features, classifier outputs, metrics, and run directories.
- Build a staged classifier ladder: pointwise baseline, windowed features, Bayesian accumulator, Kalman bank, IMM, and particle filter only when justified.
- Add synthetic trajectory generation tiers that exercise easy, boundary, adversarial, stress, and realistic cases.
- Add analysis workstreams for feature utility, pairwise identifiability, Monte Carlo behavior, PCA, and prior sensitivity.
- Standardize visualization and reporting so every run produces comparable plots and a report.
Out of Scope:
- Production data ingestion pipelines.
- Multi-target association or full sensor-fusion systems.
- Learned end-to-end models as the primary solution.
- Particle filters or IMM for cases where simpler methods already explain the data.

Implementation Steps:
1. Define the common experiment contract.
   - Standardize run-directory contents, config persistence, dataset manifests, feature matrices, prediction tables, posterior histories, likelihood histories, metrics, and markdown reports.
   - Require reproducibility from config plus seed.
2. Build the pointwise baseline.
   - Implement a common classifier interface with `reset`, `update`, `posterior`, `predict`, and `history`.
   - Add a simple Gaussian pointwise classifier and a two-class synthetic acceptance test.
   - Emit posterior history, confusion matrices, and a baseline report.
3. Add the windowed-feature classifier.
   - Implement windowed and running features with explicit history declarations.
   - Include raw and robust extrema, slope/curvature, monotonicity, sign-change counts, and irregular-time-aware statistics.
   - Measure duration bias, outlier sensitivity, and fixed-window correctness.
4. Add the sequential Bayesian accumulator.
   - Implement log-domain posterior updates, configurable priors, forgetting, confidence thresholds, and an unknown/abstain path.
   - Add prior-sweep and Bayes-factor diagnostics so prior dominance is visible.
5. Build the Kalman filter bank.
   - Define per-class motion models with irregular-`dt` transitions.
   - Update class posteriors from innovation likelihoods and track state, covariance, and residual diagnostics.
   - Use this as the main model-based sequential classifier.
6. Add IMM only for real switching cases.
   - Introduce mode probabilities, transition matrices, and mixed state estimates only when trajectories switch between valid modes.
   - Validate against known switching synthetic tracks.
7. Add particle filtering only when simpler methods fail.
   - Gate this behind nonlinear, multimodal, censored, or hard-constraint cases that Kalman or IMM cannot represent well.
   - Track particle degeneracy, resampling, and effective sample size.
8. Build the synthetic trajectory generator stack.
   - Define explicit class-generating models for stationary, CV, CA, braking, maneuvering, oscillatory, bounded-acceleration, and similar regimes.
   - Create dataset tiers for easy, boundary, adversarial, stress, and realistic conditions.
   - Add DOE coverage and feature-excitation checks.
9. Add visualization and reporting.
   - Produce single-trajectory plots, confusion matrices, Monte Carlo curves, calibration plots, prior-sensitivity plots, and artifact-linked markdown reports.
   - Ensure every classifier emits the same output shape so plots can be reused.
10. Add identifiability and feature analysis.
   - Build pairwise class distance, overlap, confusion, and confusability graph reports.
   - Rank features by separation utility and test whether classes are fundamentally separable from the current feature set.
11. Add PCA and principal-feature analysis.
   - Run PCA on engineered features, resampled trajectories, basis coefficients, and smoothed latent states where appropriate.
   - Use PCA as a diagnostic, not as a replacement for classifier development.
12. Add prior-sensitivity and bias studies.
   - Sweep priors, measure decision flips, and separate evidence-driven from prior-driven behavior.
   - Report pairwise flip thresholds and class-dominance metrics.
13. Use explicit decision gates.
   - Advance to IMM only when the static Kalman bank fails on switching-mode trajectories.
   - Advance to particle filtering only when nonlinear or non-Gaussian cases remain unresolved.
   - Refine features or trajectory generation before moving to a more complex classifier if the issue is really identifiability or excitation.

Validation:
- Schema checks for trajectory, feature, prediction, and metrics artifacts.
- Deterministic seed-reproducibility tests for dataset generation and metrics.
- Posterior normalization tests: all probabilities sum to 1, remain in range, and match argmax predictions.
- Easy-vs-boundary-vs-adversarial dataset tests showing the expected accuracy and calibration ordering.
- Hand-computed Bayesian update tests and prior-sweep tests for the accumulator.
- Running extrema, fixed-window, and outlier-sensitivity tests for feature extraction.
- Constant-velocity, constant-acceleration, and irregular-`dt` tests for the Kalman bank.
- Switching-mode tests for IMM and nonlinear/non-Gaussian tests for particle filtering before adoption.
- Identifiability tests proving the analyzer flags overlapping classes and separates clearly distinct ones.
- Monte Carlo tests confirming accuracy, confidence, calibration, and abstention behavior evolve sensibly over time.

Artifacts / Config:
- `experiments/*.yaml` for experiment configs.
- `runs/<timestamp>_<name>/config.yaml` for captured configs.
- `dataset_manifest.json`, `class_definitions.json`, `feature_manifest.json`.
- `predictions.parquet`, `posterior_history.parquet`, `likelihood_history.parquet`, `feature_matrix.parquet`.
- `metrics.json`, `confusion_final.csv`, `confusion_by_time.parquet`, `prior_sensitivity.parquet`, `identifiability_matrix.csv`.
- `plots/` subdirectories for trajectories, posteriors, confusion matrices, Monte Carlo, feature space, priors, and PCA.
- `report.md` as the human-readable summary for every run.

Dependencies:
- Shared artifact and schema definitions.
- Synthetic trajectory generator and scenario library.
- Feature extraction utilities with explicit history metadata.
- Common plotting/reporting helpers.
- Baseline Bayesian and Kalman filtering utilities.
- Optional numerical helpers for PCA, overlap metrics, calibration, and pairwise separability.

Last Updated: 2026-05-22
