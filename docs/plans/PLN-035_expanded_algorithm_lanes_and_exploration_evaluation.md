# PLN-035 Expanded Algorithm Lanes and Exploration Evaluation

Title: Expanded algorithm-map coverage and phased exploration-backend evaluation
Plan ID: PLN-035
Status: active
Owner: @rick
Priority: P1
Last Updated: 2026-06-10

Objective:
Expand the repo's method map so it explicitly covers modern time-series
classification, neural sequence baselines, learned filtering, calibration, and
future tracking/fusion methods, while also adding generator-side scaffolding for
evaluating more exploration backends under one shared contract.

Why This Exists:
- The current proof ladder is strong for a tracking-and-filtering story, but it
  is too easy for a reader to ask where the modern time-series baselines are.
- The repo already has a good exploration contract, but it needs an explicit
  registry for advanced search/generator options so CEM, MAP-Elites, PPO, SAC,
  TD3, CMA-ES, and related methods can be evaluated coherently rather than
  added ad hoc.
- The project should look disciplined rather than trendy:
  advanced methods are visible, but promotion still requires named witnesses and
  shared artifact contracts.

Target Surfaces:
- `docs/story/algorithm_map.md`
- `docs/methods/algorithm_atlas.md`
- `src/kinematic_classifier_sandbox/registry/algorithm_coverage_matrix.py`
- `src/kinematic_classifier_sandbox/registry/method_validation_os.py`
- `src/kinematic_classifier_sandbox/corpus/trajectory_exploration/backend_registry.py`

Lane Structure:
- Core physics / probabilistic ladder
  - pointwise
  - windowed
  - sequential Bayes
  - model residual Bayes
  - changepoint and segmentation
  - Kalman and switching filters
  - nonlinear and particle methods
- Benchmark classifier lane
  - DTW / kNN
  - shapelets
  - interval forests / DrCIF
  - dictionary methods
  - ROCKET family
  - HIVE-COTE
  - boosted engineered features
- Neural sequence lane
  - TCN
  - InceptionTime
  - LSTM / GRU
  - transformer and PatchTST-style encoders
  - TimesNet / S4 / Mamba watchlist
- Representation lane
  - TS2Vec
  - TS-TCC / SoftCLT
  - masked time-series autoencoders
- Learned filter lane
  - KalmanNet
  - differentiable Kalman / deep state-space models
  - Koopman / DMD residual models
- Uncertainty and calibration lane
  - temperature scaling
  - isotonic / Dirichlet calibration
  - conformal prediction
  - deep uncertainty wrappers
  - OOD and abstention
- Generator and optimizer lane
  - random / DOE
  - CEM
  - CMA-ES
  - Bayesian optimization
  - MAP-Elites
  - PPO / SAC / TD3
  - MPC-style adversarial generation
- Future tracking/fusion extension lane
  - PDA / JPDA / MHT
  - LMB / GLMB / PMBM
  - track-before-detect

Phase Plan:
1. Phase 1: high-value, low-risk additions
   - model residual Bayes
   - changepoint witness scaffolds
   - ROCKET-family benchmark lane
   - temperature-scaling lane
   - CMA-ES registry row
   - MAP-Elites kept as the diversity baseline
2. Phase 2: stronger sequence and nonlinear depth
   - UKF
   - Gaussian-sum filter robustness packet
   - switching Kalman / SLDS witness
   - TCN
   - InceptionTime
   - TS2Vec
3. Phase 3: research candidates
   - KalmanNet
   - Koopman classifier
   - conformal prediction
   - SAC / TD3
   - PatchTST / S4 / Mamba watchlist candidates
4. Phase 4: future 3D and multi-target roadmap
   - JPDA / MHT
   - GLMB / PMBM
   - track-before-detect
   - graph or scene-level sequence models

Evaluation Rules:
- The proof ladder stays disciplined:
  simple evidence -> history -> dynamics -> switching -> nonlinear or
  non-Gaussian -> latent mode.
- Benchmark and neural lanes must use the same corpus, score, calibration, and
  decision-card contracts.
- Representation models do not bypass the evidence contract; they feed
  embeddings or evidence-provider adapters.
- Generator backends are compared under matched budgets and shared utility
  decomposition.
- Multi-target tracking methods remain visible but out of scope for current 1D
  claims.

Required New Witness Families:
- `tsc_archive_baseline_frontier`
- `feature_headroom_frontier`
- `switch_cv_ca_regime_split`
- `neural_sequence_vs_physics_frontier`
- `confidence_calibration_shift`
- `coverage_control_under_shift`
- `continuous_generator_frontier`
- `coverage_archive_diversity_frontier`
- `sequential_control_generator_frontier`

Deliverables:
- Expanded algorithm coverage matrix rows across all tracked lanes.
- Expanded method-validation OS lanes and witness registry rows.
- Exploration-backend registry scaffold for current and planned generator
  families.
- Updated documentation that distinguishes promoted ladder rungs from tracked
  method families.

Out of Scope:
- Implementing every new algorithm in this plan.
- Claiming broad superiority for neural or learned methods without witness
  evidence.
- Pulling multi-target tracking into the current single-track 1D ladder.
