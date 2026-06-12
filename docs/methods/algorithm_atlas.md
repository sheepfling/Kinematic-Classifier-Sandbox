# Algorithm Atlas

The repo now tracks algorithms as lanes in a shared method-validation operating
system rather than as isolated additions.

## Epic 2 Family Framing

For Epic 2, the public story is four classifier families:

| Family | Role |
| --- | --- |
| Interpretable kinematic classifiers | Transparent feature, window, and motif evidence |
| Physics-aware inference classifiers | Dynamics, residual, uncertainty, and posterior evidence |
| Generic time-series benchmark classifiers | Strong non-physics TSC benchmark ceilings |
| Learned sequence and embedding classifiers | Neural sequence baselines and reusable learned representations |

That framing is the stable top level. The internal lanes below are the
implementation-facing decomposition that feeds the registry and artifact
surfaces.

## Internal Lanes

| Lane | Role |
| --- | --- |
| transparent_kinematic_classifiers | Interpretable baselines and failure diagnostics |
| modern_time_series_classifiers | Strong classification baselines and accuracy ceilings |
| segmentation_regime_models | Unknown switch, duration, and maneuver-onset reasoning |
| state_space_filters | Physics-aware posterior, state, and uncertainty estimation |
| learning_evidence | Supervised, sequence, and unsupervised learning-based evidence providers |
| neural_sequence_models | Neural sequence baselines that stay separate from the proof ladder |
| representation_learning_models | Reusable learned embeddings and self-supervised representation baselines |
| learned_hybrid_filters | Future learned-model and differentiable filtering lane |
| uncertainty_calibration | Coverage, abstention, and calibration wrappers over evidence providers |
| exploration_generators | Search and control backends that generate witnesses and corpora |
| tracking_2d_plus | Future operational multi-target and clutter lane |

Epic 2 maps these internal lanes into the four public families like this:

- `transparent_kinematic_classifiers` -> Interpretable kinematic classifiers
- `segmentation_regime_models` and `state_space_filters` -> Physics-aware inference classifiers
- `modern_time_series_classifiers` -> Generic time-series benchmark classifiers
- `neural_sequence_models` and `representation_learning_models`
  -> Learned sequence and embedding classifiers

`learning_evidence` and `learned_hybrid_filters` are tracked by the registry
but are intentionally outside the public Epic 2 family gates. They inform the
research backlog without redefining the family-completion read.

The same rule now applies to `switching_kalman_slds` inside the physics-aware
domain: it remains tracked in the registry, but it is treated as a deferred
extension rather than a public Epic 2 family-closure blocker.

The mapping is many-to-one on purpose. The registry needs more granularity than
the epic story.

## Current Scope

The generated atlas bundle lives in:

- `artifacts/method_validation_os_v1/method_specs.json`
- `artifacts/method_validation_os_v1/algorithm_promotion_status_matrix.csv`
- `artifacts/method_validation_os_v1/epic2_family_maturity_matrix.csv`
- `artifacts/method_validation_os_v1/witness_to_method_coverage_matrix.csv`

This atlas is intentionally broader than the current proof ladder. A method can
be researched and tracked here without being promoted in the classifier-family
evaluation story.

## Immediate Gaps

The highest-value remaining blockers before broader PF/RBPF and benchmark
claims are now outside the first-pass transparent, calibration, and continuous
generator lanes.

`shapelet / motif`, `gradient boosting on engineered features`, `BOCPD`,
`HSMM`, `UKF / EKF`, `Student-t / robust Kalman`, and `Gaussian Sum Filter`
now sit in witness-backed blocker lanes, but all still need broader robustness
and comparison work before stronger claims.

`TCN / InceptionTime` now has a trained local neural frontier with held-out
temperature scaling plus a bounded multi-seed robustness companion. It is
witness-backed and first-class, but broader robustness and external benchmark
breadth are still open.

`Shapelets / motif transforms` now has a dedicated localized motif witness in
`shapelet_maneuver_motif_v1`, which makes the short-pattern lane first-class
instead of just a theoretical baseline.

`MiniRocket / MultiRocket / HYDRA`, `DrCIF / interval forests`,
`BOSS / WEASEL / TDE`, and `HIVE-COTE` now all have a modern-TSC execution
frontier with optional external wrapper hooks and explicit fallback reporting.
They are tracked as implemented surfaces, not promoted finished methods. A
wrapper or fallback row is evidence that the lane is being integrated, not
proof that the family is complete. The lane now also has named negative
witnesses plus a bounded diagnosis packet, so the generic-TSC family stays
closed for diagnosed reasons rather than only because the wrappers are new.

`TS2Vec / contrastive trajectory encoder` now has a first embedding witness in
the repository, including a prefix-based online route proof, plus a bounded
proxy-versus-external parity witness on the shared 1D corpus. It is
witness-backed and first-class, but broader external benchmark coverage and
generalized library parity still remain open.

`TCN` and `InceptionTime` sit in the learned-sequence half of the fourth Epic 2
family. `TS2Vec` sits in the learned-embedding half. They stay grouped in the
public story because the point of the family is to test learned evidence, not
to imply those subfamilies are interchangeable.

`learning_evidence` is now an explicit lane for supervised tabular baselines,
compact sequence learners, and unsupervised discovery. The point of the lane
is not to replace the proof ladder, but to keep ML evidence providers audited
with the same split discipline, calibration checks, and decision-card logic.

`MAP-Elites` is now witness-backed for the dedicated quality-diversity corpus
study, but it still needs broader archive-policy and diversity-sweep coverage
before stronger generator claims.

`temperature scaling` is now witness-backed for a dedicated calibration-shift
study, and `conformal wrapper` is now witness-backed for a dedicated
coverage-control study.

`CMA-ES` is now witness-backed for a dedicated continuous-generator frontier
study, but it still needs broader seed, budget, and objective-family sweeps
before stronger generator-selection claims.

`learned_model_mismatch` is still missing. The learned-filter lane has a
coverage entry for `kalmannet_family`, but it needs a real witness packet
before KalmanNet or differentiable PF can move out of the research-candidate
bucket.

`sequential_control_generator_frontier_v1` keeps the sequential-control lane
explicit with a PPO proxy packet.

`sequential_offpolicy_control_frontier_v1` adds the first SAC/TD3 smoke run on
the sequential-control witness surface. It is still small-budget and
experimental, but it now gives the off-policy lane a concrete comparison packet
with a narrow seed sweep instead of only a roadmap note.

## Generator Registry

Expanded exploration options are tracked in
`src/kinematic_classifier_sandbox/corpus/trajectory_exploration/backend_registry.py`.
That scaffold keeps three things explicit:

- which generator backends are implemented today
- which ones are phase-1 or phase-2 benchmark candidates
- which ones require sequential control and should stay out of fixed-budget parameter-only comparisons

The current tracked exploration surface is now explicit rather than grouped:

- baseline search: random / DOE heuristic search, Latin hypercube
- black-box optimization: CEM, CMA-ES, Bayesian optimization
- quality-diversity: MAP-Elites
- RL / learned search: stateless RL-shaped policy search, PPO, SAC, TD3
- control optimization: MPC-style adversarial generator

The current backend registry treats baseline search, black-box optimization,
quality-diversity, and the sequential-control RL lane as first-class
implemented surfaces. PPO / SAC / TD3 still depend on witness-specific control
studies and budgeted smoke runs, so the repo keeps their claim boundary narrow
even though the frontier packet exists.
