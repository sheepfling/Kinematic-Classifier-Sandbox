# Algorithm Map

The repo now keeps two distinct method surfaces on purpose:

1. The proof ladder in [algorithm_ladder.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/story/algorithm_ladder.md), which is the disciplined escalation path for methods we actively justify with witnesses.
2. The broader algorithm coverage matrix, which records the method families the repo explicitly tracks so the project does not look blind to modern time-series classification, learned filtering, calibration, or future tracking/fusion extensions.

The generated artifact bundle for the broader surface is:

- `artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md`
- `artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix.csv`
- `artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix.png`

## Lane Structure

- `core_physics_probabilistic`
  - pointwise
  - windowed
  - sequential Bayes
  - model residual Bayes
  - BOCPD / PELT changepoint
  - transition HMM
  - Kalman bank
  - EKF / UKF / CKF
  - Gaussian-sum / switching Kalman / IMM
  - PF / RBPF
- `time_series_baselines`
  - DTW / kNN
  - shapelets
  - DrCIF / interval forests
  - BOSS / WEASEL / TDE
  - ROCKET family
  - HIVE-COTE
  - gradient boosting on engineered features
- `neural_sequence`
  - TCN / InceptionTime
  - LSTM / GRU
  - transformer / PatchTST-style encoder
  - TimesNet / S4 / Mamba watchlist
- `representation_learning`
  - TS2Vec-style contrastive encoders
  - TS-TCC / SoftCLT
  - masked time-series autoencoders
- `learned_filters`
  - KalmanNet / learned-gain filters
  - differentiable Kalman / deep state-space models
  - Koopman / DMD residual classifiers
- `calibration_uncertainty`
  - temperature scaling
  - isotonic / Dirichlet calibration
  - conformal wrappers
  - ensembles / MC dropout / evidential wrappers
  - OOD detection / abstention
- `optimizer_generator`
  - random / grid / Latin hypercube
  - CEM / CMA-ES
  - Bayesian optimization
  - MAP-Elites
  - PPO / SAC / TD3
  - MPC-style adversarial generators
- `tracking_fusion_extension`
  - PDA / JPDA / MHT
  - GLMB / PMBM

The concrete generator-side scaffolding now lives beside the benchmark runner in
`src/kinematic_classifier_sandbox/corpus/trajectory_exploration/backend_registry.py`.
That registry is intentionally broader than the currently implemented backends:
it records which search families are active today and which ones are queued for
phased evaluation.

## Reading Rule

The broader map is intentionally not a flat implementation checklist.

- `implemented` means there is current code and artifact/test coverage.
- `implemented_or_planned` means the lane already has partial footing in the repo and an explicit near-term expansion path.
- `experimental` means the repo has a prototype pathway but not a fully promoted production claim.
- `planned_near_term`, `benchmark_candidate`, `research_candidate`, and `watchlist` mean the family is explicitly tracked and should fit the same methodology surfaces when added.
- `future_extension` means the family matters for the 3D or multi-target roadmap but does not belong in the current 1D proof ladder.

The key discipline is unchanged:

New methods do not get promoted because they are fashionable. They get promoted because a named witness shows the simpler rung fails for a reason the new method is designed to address.
