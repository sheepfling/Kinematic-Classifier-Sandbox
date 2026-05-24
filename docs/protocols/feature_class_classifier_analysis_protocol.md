# Feature + Class + Classifier Analysis Protocol

This protocol defines the standard evaluation path for a proposed `Feature Set + Class Set + Classifier / Filter` study. The goal is to replace ad hoc judgment with a repeatable ladder that produces a promotion, revision, rejection, or deferral decision.

## Inputs

A study proposal should be representable as a `StudyCandidate`:

```text
StudyCandidate =
    CorpusSpec
    + FeatureSetSpec
    + ClassSetSpec
    + ClassifierSpec
    + PriorSpec
    + optional FilterSpec
    + optional VisualizationSpec
```

The minimum required declarations are:

- study hypothesis
- class set and class-pair claims
- feature sets and taxonomy metadata
- classifier or filter family
- prior choice and prior sensitivity expectations
- corpus or corpus-tier target
- expected failure modes

## Ten-Step Protocol

1. Define the study hypothesis.
   State what the proposal is trying to prove, improve, or falsify.

2. Declare the class set and class-pair claims.
   Make the important pairwise boundaries explicit instead of hiding them inside aggregate accuracy.

3. Declare feature sets and feature taxonomy metadata.
   Record feature groups, history behavior, evidence role, dependency risk, sensitivity tags, and dimensional transfer status.

4. Declare classifier or filter family and assumptions.
   Record whether the proposal is pointwise, windowed, sequential Bayes, state-space, or transition-aware, and state any independence, dynamics, or switching assumptions.

5. Run static compatibility screening.
   Estimate whether the feature set, class set, classifier family, and prior policy are coherent before running expensive studies.

6. Generate or select corpus candidates.
   Choose corpus tiers and scenarios that exercise the declared class-pair claims and expected failure modes.

7. Run corpus adequacy and leakage audits.
   Check class balance, scenario balance, difficulty distribution, feature excitation, and covariate leakage before trusting performance metrics.

8. Run static separability and oracle studies.
   Measure class-pair overlap, feature-space identifiability, PCA structure, and feature-only oracle ceilings to determine whether the classifier is solving a real separability problem.

9. Run the Monte Carlo classifier ladder.
   Execute shared-corpus and study-specific evaluations, including calibration, prior sensitivity, stress slices, and robustness diagnostics.

10. Produce a promotion, revision, rejection, or deferral decision.
    Promotion requires evidence across the ladder. Rejection or revision should cite the exact failure mode, not just a low aggregate score.

## Static Screening Questions

- Is the feature set compatible with the declared classes?
- Does the proposal mix cumulative and summary evidence in a way that risks double-counting?
- Are any key class pairs likely to be weakly identifiable at the declared horizon?
- Does the classifier family match the problem assumptions?
- Is the proposal tied to a scalar-only assumption that blocks 3D transfer?

## Statistical Validation Questions

- Does the corpus actually contain the intended easy, boundary, adversarial, stress, and realistic regimes?
- Is oracle separability materially above classifier performance?
- Is the proposal prior-sensitive on the hardest class pairs?
- Are confident errors concentrated in a particular scenario family?
- Does the method degrade under short horizon, irregular sampling, noise, or outlier corruption?

## Decision Policy

The protocol uses four terminal decisions:

- `promote`: the proposal is strong enough to move forward as a recommended study configuration
- `revise`: the proposal is promising but needs corpus, feature, or classifier changes
- `reject`: the proposal fails static or statistical checks strongly enough that it should not proceed
- `defer`: the proposal is not wrong, but is blocked by missing infrastructure or evidence

## Example Decision

```text
Proposal:
    robust_extrema + shape_window features
    classes: constant_velocity, constant_acceleration, maneuver
    classifier: Bayesian accumulator

Static result:
    feature set is compatible
    cumulative-feature double-counting risk is medium
    constant_acceleration vs maneuver is likely hard

Statistical result:
    good accuracy on easy tier
    weak calibration on boundary tier
    prior sensitivity elevated for acceleration vs maneuver

Decision:
    revise corpus and add model-residual features before promotion
```

## Exit Condition

This protocol is working when a new `Feature + Class + Classifier` proposal can be evaluated with a checklist and a stable artifact set, rather than with informal interpretation alone.
