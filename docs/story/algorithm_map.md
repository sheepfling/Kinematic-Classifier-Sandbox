# Algorithm Map

The repo now keeps two distinct method surfaces on purpose:

1. The proof ladder in [algorithm_ladder.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/story/algorithm_ladder.md), which is the disciplined escalation path for methods we actively justify with witnesses.
2. The broader algorithm coverage matrix, which records the method families the repo explicitly tracks so the project does not look blind to modern time-series classification, learned filtering, calibration, or future tracking/fusion extensions.

The generated artifact bundle for the broader surface is:

- `artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md`
- `artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix.csv`
- `artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix.png`

The RL and exploration packets now follow the same comparison convention:

- `summary.csv`
- `report.md`
- `decision_card.md`

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
  - Student-t / robust Kalman
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
- `learning_evidence`
  - supervised tabular evidence providers
  - compact sequence learners
  - unsupervised discovery and anomaly detection
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
  - random / DOE / structured DOE
  - CEM
  - CMA-ES
  - Bayesian optimization
  - MAP-Elites
  - stateless RL-shaped policy search
  - PPO
  - SAC / TD3
  - sequential-control frontier packet (PPO proxy)
  - sequential off-policy frontier packet (SAC / TD3 smoke run)
  - MPC-style adversarial generators
- `tracking_fusion_extension`
  - PDA / JPDA / MHT
  - GLMB / PMBM

The concrete generator-side scaffolding now lives beside the benchmark runner in
`src/kinematic_classifier_sandbox/corpus/trajectory_exploration/backend_registry.py`.
That registry is intentionally broader than the currently promoted backends and
now maps cleanly onto the public coverage matrix:

- implemented now: random / DOE heuristic search, CEM, stateless RL-shaped policy search, MAP-Elites
- experimental now: PPO sequential control
- planned next: Latin hypercube, CMA-ES, Bayesian optimization
- research candidates: SAC, TD3, MPC-style adversarial generators

## Reading Rule

The broader map is intentionally not a flat implementation checklist.

- `implemented` means there is current code and artifact/test coverage.
- `implemented_or_planned` means the lane already has partial footing in the repo and an explicit near-term expansion path.
- `experimental` means the repo has a prototype pathway but not a fully promoted production claim.
- `planned_near_term`, `benchmark_candidate`, `research_candidate`, and `watchlist` mean the family is explicitly tracked and should fit the same methodology surfaces when added.
- `future_extension` means the family matters for the 3D or multi-target roadmap but does not belong in the current 1D proof ladder.

The key discipline is unchanged:

New methods do not get promoted because they are fashionable. They get promoted because a named witness shows the simpler rung fails for a reason the new method is designed to address.

The `learning_evidence` lane is intentionally separate from the proof ladder. It exists so supervised and unsupervised ML can be audited as evidence providers, not because they replace the kinematic or filter story. The repository uses that lane to keep trajectory-level split discipline, calibration checks, and discovery diagnostics explicit while preserving the main ladder for the physics-aware escalation path.
