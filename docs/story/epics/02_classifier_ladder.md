# Epic 2: 1D Classifier Family Evaluation Framework

Goal: build and prove a reusable 1D time-series classification pipeline that
evaluates multiple classifier families under the same data, metrics, stress
tests, artifact contracts, and reporting surfaces.

Core question: for a 1D trajectory classification problem, which classifier
families provide useful, reliable, and explainable evidence, and when?

Epic 2 is not "simple classifiers first, advanced classifiers later." It is a
proof framework for classifier families. The repo treats classifiers and
filters as evidence providers under a shared evaluation contract and then asks
which family is sufficient for which failure mode.

The architectural claim is:

> Epic 2 proves a real-data-ready 1D classification pipeline by evaluating
> four classifier families - interpretable kinematic, physics-aware
> inference, generic time-series benchmark, and learned sequence / embedding
> methods - under common metrics, robustness tests, and failure-mode analysis.

## The Four Top-Level Lanes

Epic 2 uses four public-facing family lanes:

| Lane | Purpose |
| --- | --- |
| Interpretable kinematic classifiers | Transparent feature, window, and motif evidence; failure diagnostics |
| Physics-aware inference classifiers | Residual, likelihood, uncertainty, state, and posterior evidence |
| Generic time-series benchmark classifiers | Strong non-physics benchmark ceilings and runtime/accuracy pressure tests |
| Learned sequence and embedding classifiers | Supervised neural sequence baselines plus representation learning |

Internally, the fourth lane is split into:

- `4A` neural sequence classifiers such as `TCN` and `InceptionTime`
- `4B` representation-learning / embedding methods such as `TS2Vec`

This split stays internal. The public story remains the four-lane framing.

## Lane Detail

### 1. Interpretable Kinematic Classifiers

Purpose: answer how far transparent, engineered kinematic evidence can take
the study before richer sequence or state-space assumptions are necessary.

Included families:

- pointwise classifiers
- windowed classifiers
- robust-windowed classifiers
- shapelet / motif classifiers
- engineered-feature boosting

Main proof obligations:

- simple global features work on easy classes
- windowed features help when timing matters
- robust windows help under noise and outliers
- shapelets help when localized motifs matter
- failure cases are documented rather than hidden

### 2. Physics-Aware Inference Classifiers

Purpose: answer whether dynamics, residuals, uncertainty, and posterior
evolution provide auditable classification evidence.

Included families:

- HMM / transition matrix
- Kalman bank
- UKF
- robust Kalman
- GSF
- IMM
- PF
- RBPF

Main proof obligations:

- motion assumptions produce class evidence
- posterior probabilities evolve sensibly over time
- model mismatch appears in residuals or likelihood collapse
- online classification can use only past observations
- complexity escalation is witness-driven, not prestige-driven

### 3. Generic Time-Series Benchmark Classifiers

Purpose: answer whether handcrafted and physics-aware methods are leaving raw
classification performance on the table.

Included families:

- ROCKET / MiniRocket / MultiRocket / HYDRA
- CIF / DrCIF
- BOSS / WEASEL / TDE
- HIVE-COTE

Main proof obligations:

- strong benchmark families can run on the shared 1D surface
- runtime and memory are measured alongside accuracy
- optional wrappers and fallbacks are reported honestly
- no archive-style family is promoted as "finished" until faithful
  non-fallback execution and comparison evidence exist

### 4. Learned Sequence and Embedding Classifiers

Purpose: answer whether learned neural filters or learned embeddings justify
their complexity as data scale, noise, or label scarcity increase.

Included families:

- `TCN`
- `InceptionTime`
- `TS2Vec`
- future longer-context sequence models

Main proof obligations:

- training is reproducible
- calibration and runtime are visible
- overfitting and data-scale dependence are documented
- embeddings are evaluated through downstream probes rather than treated as
  self-justifying

## Shared Epic 2 Gates

A family does not count as finished because code exists. Epic 2 uses three
practical maturity gates:

| Gate | Meaning |
| --- | --- |
| Implemented | Class exists, basic fit/predict works, unit tests pass |
| Integrated | Runs through the shared dataset, split, metric, and artifact pipeline |
| Proven | Compared against relevant baselines on controlled 1D studies with runtime, robustness, and documented failure modes |

The repo's finer-grained status ladder remains canonical for detailed method
tracking. These Epic 2 gates are the user-facing simplification:

- `Implemented` maps to at least `implemented`
- `Integrated` maps to at least `trace_validated` and shared-pipeline support
- `Proven` maps to at least `witness_supported`
- stronger complexity claims generally require `study_justified`

Current family read:

| Family | Implemented | Integrated | Proven | Current read |
| --- | --- | --- | --- | --- |
| Interpretable kinematic classifiers | yes | yes | yes | Proven on the current 1D witness set |
| Physics-aware inference classifiers | yes | yes | partial | Strongest family overall, but not fully closed |
| Generic time-series benchmark classifiers | yes | partial | partial | Real external execution exists and MiniRocket now has a bounded promotion path, but the family is still partial |
| Learned sequence and embedding classifiers | yes | yes | partial | Visible and useful on the bounded 1D surface, but not yet strongly ceiling-aligned |

The authoritative family-level completion read lives in
`docs/story/epic2_completion_audit.md` and
`artifacts/method_validation_os_v1/epic2_family_maturity_matrix.csv`.

## Missing Centerpiece

Epic 2 is not complete just because the ladder exists and the charts render.
The central deliverable is the classifier family scorecard:

1. What capability does each family add?
2. When should each family win?
3. How close does each family get to the Epic 1 admissibility ceiling?

The repo now has the first honest version of that surface in
`artifacts/classifier_family_scorecard_v1/`. It makes capability additions and
expected win conditions explicit, but it also shows that ceiling-relative
alignment is still incomplete for several families.

## Required Deliverables

Epic 2 should produce three top-level deliverables:

| Deliverable | Purpose | Current read |
| --- | --- | --- |
| Classifier Family Atlas | One-page family cards covering assumptions, evidence type, strengths, failure modes, complexity, witnesses, and status | Present in bounded form |
| Capability Matrix | Cross-family map of what evidence each family can represent | Present |
| Performance Relative to Ceiling | Family-by-family read of how much admissible Epic 1 signal is actually captured | Partial and still the main blocker |

## Primary Epic 2 Evaluation Tiers

Every family should run against the same 1D study tiers.

| Tier | Purpose | Typical winning families |
| --- | --- | --- |
| Tier A: Sanity / Easy | Prove pipeline, labels, and features are not broken | pointwise, engineered boosting, Kalman bank |
| Tier B: Timing Matters | Show pointwise summaries are insufficient | windowed, HMM, Kalman-bank, MiniRocket, InceptionTime |
| Tier C: Local Motif / Shape | Show localized pattern detection matters | shapelets, dictionary methods, DrCIF, neural sequence models |
| Tier D: Robustness / Realism | Stress noise, outliers, missingness, imbalance, partial observation | robust windows, robust Kalman, PF/RBPF where justified, strong generic TSC baselines |

## Honesty Rule

This repo does not treat proxies, fallbacks, or wrapper-stage execution as a
finished method family.

For archive-style and other external methods:

- executable wrapper surfaces are useful
- provenance about the backend actually used is required
- local fallback rows may appear in comparison packets
- fallback execution does not count as a faithful finished implementation
- such methods stay below promotion until the evidence packet is real

## Primary Artifacts

- `artifacts/common_1d_classifier_study/unified_posterior_history.csv`
- `artifacts/method_validation_os_v1/algorithm_promotion_status_matrix.csv`
- `artifacts/method_validation_os_v1/epic2_family_maturity_matrix.csv`
- `artifacts/method_validation_os_v1/witness_to_method_coverage_matrix.csv`
- `artifacts/classifier_family_scorecard_v1/classifier_family_atlas.md`
- `artifacts/classifier_family_scorecard_v1/capability_matrix.csv`
- `artifacts/classifier_family_scorecard_v1/ceiling_efficiency.csv`
- `artifacts/classifier_family_scorecard_v1/classifier_family_scorecard_report.md`
- `artifacts/rung_sufficiency/rung_promotion_matrix.csv`
- `artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv`

## Main Charts

- `06_posterior_timeline_witness`
- `06c_capability_ladder`
- `07_rung_sufficiency_map`
- `10_advanced_filter_gate_matrix`
- `15_method_atlas_map`
- `16_algorithm_promotion_status_matrix`
- `17_witness_to_method_coverage_matrix`
- `18_classifier_efficiency_vs_epic1_proxy_ceiling`
- `19_calibration_and_confidence_panel`
- `20_particle_count_pareto`
- `21_pf_rbpf_compute_frontier`
- `22_learned_baseline_comparison`

## Decision Language

- `implemented`
- `integrated`
- `proven`
- `simpler_rung_sufficient`
- `witness_supported`
- `study_justified`
- `insufficient_evidence`
- `not_complexity_justified`

The audience should leave believing that Epic 2 is a family-evaluation
framework, not a pile of algorithms. Epic 2 is only fully closed when that
framework can state, for each family, what capability it adds, when it should
win, and how much of the Epic 1 ceiling it captures.
