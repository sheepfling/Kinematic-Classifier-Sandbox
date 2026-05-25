# Repo Story

This repository is a reusable methodology workbench for kinematic classification studies. It is not just a 1D classifier benchmark. The current 1D problems are witness problems: small controlled studies that prove the methodology layers before the framework is lifted to 3D.

The central object is a study candidate:

```text
s = (D, f, C, m, pi, b)
```

where `D` is a corpus or corpus slice, `f` is a feature set, `C` is a class set or class pair, `m` is a classifier or filter family, `pi` is a prior regime, and `b` is an optional backend, filter, or dynamics family.

The repo story is:

1. Declare a study candidate.
2. Generate or select a corpus.
3. Validate class labels and feature excitation.
4. Run classifier or filter evidence providers.
5. Update posteriors under explicit priors.
6. Evaluate separability, calibration, prior sensitivity, and confusion.
7. Promote, revise, reject, or defer the study.
8. Use 1D witness problems to prove each layer before lifting to 3D.

## Pillars

### Study Candidate Evaluator

The Study Candidate Evaluator is the generic engine for `FeatureSet + ClassSet + ClassifierFamily/FilterBackend + PriorRegime`. It asks whether a study is compatible, separable, robust to priors, and ready for promotion. It treats poor results as diagnostic evidence: a weak confusion matrix may indicate a corpus problem, a feature problem, a posterior problem, or a prior fragility problem.

### Corpus Explorer

The Corpus Explorer is the generic engine for `CorpusObjective + Backend + Search/Archive + Adequacy Evaluation`. It generates, searches, validates, scores, and selects corpora. It owns corpus objectives, backend adapters, candidate samplers, quality-diversity archives, class validity, feature excitation, leakage audits, and selected generated corpora.

### Classifier / Filter Ladder

The classifier ladder is the algorithm spine. Each rung is a different answer to the same question: how should class evidence be constructed from the observations, features, or filter residuals available up to time `t`?

The ladder is:

```text
pointwise -> windowed -> sequential Bayes -> Kalman bank -> transition matrix -> IMM -> PF -> RBPF
```

Each rung must add one evidence capability and be justified by a demonstrated failure mode of the previous rung.

### Evaluation And Promotion

Evaluation is the decision layer. It interprets evidence in this order:

1. corpus adequacy
2. class validity
3. feature excitation and separability
4. oracle gap
5. posterior behavior and prior sensitivity
6. confusion localization
7. promotion decision

The goal is not to chase top-line accuracy first. The goal is to understand why a study can or cannot support its claim.

### 1D Witness Suite

The 1D witness suite contains controlled methodology proofs. These are intentionally small so each layer can be isolated before adding 3D geometry, richer dynamics, sensor models, and backend diversity.

The witness problems currently prove:

- pointwise overlap: likelihoods, priors, and posterior flips
- windowed outlier/extrema: raw versus robust feature behavior
- sequential history: history can help beyond pointwise evidence
- Kalman endpoint match: dynamics evidence helps in matched-endpoint cases
- transition switching: transition logic is justified before IMM
- generated corpus stress: the Corpus Explorer can discover hard or fragile examples

## 3D Transition

The repo does not claim to be dynamically complete for 3D deployment. It does claim that the evaluation stack is generic enough that 3D transition is an adapter, feature, and dynamics lift rather than a rewrite of the posterior, artifact, and decision machinery.

3D transition means adding vector-valued trajectory backends, 3D feature families, geometry-aware class definitions, richer dynamics models, and sensor/regime audits while preserving the same study candidate, corpus explorer, evidence contract, posterior updater, and validation ladder.

## What Is Proven

- The methodology stack exists end to end.
- Corpora can be generated, audited, explored, and selected.
- Classifiers can be compared through shared evidence and posterior contracts.
- Priors and posterior fragility are measurable.
- Feature and class separability are inspected before blaming algorithms.
- 1D witness problems prove the ladder layers.
- Advanced filters are implemented and promoted only for demonstrated witness failures.

## What Is Not Yet Proven

- The synthetic corpus is final.
- A single classifier family is generally best.
- The Kalman bank is generally worse or better.
- That IMM/PF/RBPF are generally best outside their promoted witness failures.
- 3D dynamics and geometry are complete.
- Every generated corpus passes every adequacy and leakage gate.

The strongest current claim is architectural: the repo has the machinery to define, generate, evaluate, and decide kinematic classification studies under explicit evidence contracts.
