# PLN-008 M1 Through M10 Generalization Assessment

Title: M1 Through M10 Generalization Assessment
Plan ID: PLN-008
Status: done
Owner: @codex
Priority: P1
Objective: Record an explicit assessment of whether milestones M1 through M10 collectively demonstrate only a working toy classifier problem or a methodology that is plausibly generalizable across kinematic feature sets, class vocabularies, and 1D versus 3D domains.
Scope:
- Assess milestones M1 through M10 against the current repo implementation.
- Distinguish what is actually proven from what is only structurally prepared.
- State the current claim the repo can honestly make about generalization.
- Identify the concrete gaps that still block a credible 1D-to-3D methodology claim.
Out of Scope:
- Implementing the missing multidimensional or second-domain studies.
- Rewriting the roadmap.
- Reclassifying milestone completion status beyond the specific generalization question.
Implementation Steps:
1. Review the current milestone surfaces from M1 through M10.
2. Separate classifier-method validation from reusable harness and diagnostic methodology.
3. Identify which parts are 1D-specific versus dimension-ready.
4. Record an honest repo-level conclusion and the remaining generalization gaps.
Validation:
- Manual review of `src/kinematic_classifier_sandbox/`
- Manual review of `docs/plans/PLN-002_kinematic_classification_roadmap.md`
- Manual review of the current milestone and common-study surfaces
Artifacts / Config:
- `src/kinematic_classifier_sandbox/pointwise_baseline.py`
- `src/kinematic_classifier_sandbox/windowed_baseline.py`
- `src/kinematic_classifier_sandbox/sequential_bayes_accumulator.py`
- `src/kinematic_classifier_sandbox/monte_carlo_benchmark.py`
- `src/kinematic_classifier_sandbox/trajectory_generator.py`
- `src/kinematic_classifier_sandbox/feature_analysis.py`
- `src/kinematic_classifier_sandbox/pca_analysis.py`
- `src/kinematic_classifier_sandbox/common_experiment_harness.py`
- `src/kinematic_classifier_sandbox/contracts.py`
- `experiments/common_1d_classifier_study/common_experiment_config.yaml`
Dependencies:
- `PLN-002` roadmap expectations
- Completed `M1` through `M9` artifact surfaces
- Current `M10` common experiment harness implementation
Last Updated: 2026-05-23

## Conclusion

The repo is no longer just a toy classifier demo, but it also does not yet prove a fully general kinematic-classification framework across arbitrary feature sets, class definitions, and both 1D and 3D motion domains.

The most accurate current claim is:

- the repo is a strong 1D kinematic experimentation and diagnostics sandbox
- several of its analysis and reporting patterns are designed to generalize
- the executable classifier and generator stack is still mostly specialized to the current scalar domain

## What M1 Through M10 Prove

### M1 Through M4

`M1` through `M4` prove method-comparison discipline more than domain generality.

- `M1` proves a reproducible pointwise baseline with clear posterior accounting.
- `M2` proves windowed feature extraction and comparison patterns.
- `M3` proves sequential Bayesian accumulation and prior-sensitive decision behavior.
- `M4` proves Monte Carlo evaluation and calibration-style reporting.

These are valuable, but they still operate on a narrow scalar problem family. They show that the repo can compare techniques cleanly, not that the repo already supports arbitrary kinematic domains.

### M5 Through M9

`M5` through `M9` are the main transition away from a pure toy benchmark.

- `M5` introduces explicit trajectory-class definitions, tiered synthetic corpora, and a reusable generator surface.
- `M6` adds feature excitation, overlap, confusability, and separability diagnostics.
- `M7` expands the method space with model-based state-space classifiers.
- `M8` adds principal-component analysis and lower-dimensional separability analysis.
- `M9` expands the generator stack with short-horizon, perturbation-sweep, and switching scenario families.

These milestones demonstrate a methodology for studying:

- feature usefulness
- class-pair confusability
- corpus quality
- regime sensitivity
- scenario-library coverage

That methodology is more general than the earliest toy benchmarks.

### M10

`M10` is the beginning of the framework transition, but not the end of it.

The current common harness now supports:

- config-driven study execution
- an explicit `study_adapter`
- manifest-aware class-pair selection
- config-driven artifact naming
- a junior-usable study runner surface

This matters because it moves the repo from an implicit one-off execution path toward a reusable study runner. But it still only proves that design on the current `common_1d_classifier_study`.

## What Is Genuinely Generalizable Today

The following parts are credible building blocks for broader generalization:

- artifact and contract discipline in `contracts.py`
- manifest-based feature-set and class-pair surfaces
- feature coverage, adequacy, and separability diagnostics
- milestone and study rerun entrypoints
- the explicit separation between study config and execution in `M10`

These are the parts that should transfer well to:

- new 1D feature libraries
- new class-pair manifests
- new classifier manifests
- future multidimensional studies

## What Is Still Fundamentally 1D-Specialized

The current executable core is still tied to the scalar domain in important ways.

- The active generator stack is still built around 1D motion classes and scalar observables.
- The current feature library is still largely 1D-derived in practical execution, even if the context model now exposes a more generic base.
- The common harness still relies on hardcoded executable pair dynamics for the current runnable study.
- The current study and manifest surfaces are centered on `common_1d_classifier_study`, not on multiple materially different domains.

Because of those facts, the repo does not yet prove:

- interchangeable 1D and 3D study execution
- dimension-agnostic feature computation
- classifier portability across materially different kinematic domains without core-code extension

## Honest Claim the Repo Can Make Now

The repo can honestly claim that it is proving out reusable experimental techniques for kinematic classification, especially around:

- diagnostics
- coverage auditing
- corpus design
- classifier comparison
- manifest-backed study organization

The repo cannot yet honestly claim that it has already demonstrated those techniques across:

- arbitrary class vocabularies
- arbitrary feature families
- both 1D and 3D executable studies

## Remaining Gaps Before a Strong Generalization Claim

The minimum remaining gaps are:

1. Add a second study domain that runs through the `M10` harness without core harness edits.
2. Add a real multidimensional study, even if minimal, to exercise `measurement_dim`, `state_dim`, and coordinate-frame fields in practice.
3. Make feature extraction more explicitly dimension-aware in execution, not only in type structure.
4. Reduce study-specific executable pair logic in the common harness path.
5. Move more classifier execution choices from Python wiring into manifests and config.

## Decision

`M1` through `M10` should be described as:

- more than a toy classifier exercise
- not yet a fully general kinematic-classification framework
- a strong 1D proving ground with several framework-quality abstractions that still need one additional study domain and one multidimensional proof point before a broad generalization claim is justified

## Recommended Next Statement for Repo Positioning

If the repo needs a concise public-facing description right now, the safest accurate wording is:

> A 1D kinematic classification sandbox that is evolving toward a multi-study methodology framework, with generalizable diagnostics and reporting surfaces but not yet a fully proven 1D/3D execution stack.
