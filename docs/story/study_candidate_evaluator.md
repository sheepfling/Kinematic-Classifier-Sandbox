# Study Candidate Evaluator

The Study Candidate Evaluator is the front door for testing whether a proposed kinematic classification study is meaningful.

```text
s = (D, f, C, m, pi, b)
```

where:

| Symbol | Meaning |
| --- | --- |
| `D` | corpus or corpus slice |
| `f` | feature set |
| `C` | class set or class pair |
| `m` | classifier or filter family |
| `pi` | prior regime |
| `b` | optional backend, filter, or dynamics family |

The evaluator maps a candidate through the validation ladder:

```text
s -> {l_1, l_2, ..., l_10} -> d
d in {promote, revise, reject, defer}
```

## Required Inputs

- A corpus or selected corpus with provenance.
- A feature set with a manifest and excitation checks.
- A class set or class pair with class definitions.
- A classifier family or filter backend.
- A prior regime and prior-sensitivity sweep.
- Optional backend metadata when dynamics or generator capability matters.

## Static Checks

The first static gate ignores the classifier family and asks whether `(f, C, pi)` is admissible before corpus generation or classifier escalation:

- class separability and class-pair confusability
- feature relevance, redundancy, and synergy candidates
- prior pathology and likelihood-ratio flip thresholds
- coverage feasibility over observed feature samples
- leakage risk from provenance, future dependence, metadata, or label-rule overlap
- decisionability: promote to corpus explorer, revise features/classes/priors, or reject

The canonical static packet is `artifacts/static_feature_class_prior_audit_v1/static_audit_report.md`, with `static_decision_card.md`, `class_confusability_matrix.csv`, `feature_relevance_table.csv`, `feature_redundancy_matrix.csv`, `feature_synergy_candidates.csv`, `prior_pathology_report.csv`, `coverage_static_report.csv`, and `leakage_static_report.csv`.

## Monte Carlo Checks

Monte Carlo checks ask whether the candidate holds under repeated draws, noise, duration variation, stress cases, and class-pair slices. They produce calibration, accuracy over time, confidence crossing, confusion, and oracle-gap artifacts.

## Prior Sensitivity

Priors are explicit inputs, not hidden defaults. A candidate is fragile when posterior decisions flip under plausible prior shifts or when the prior dominates the likelihood evidence.

## Separability

Separability is evaluated before blaming the classifier. If feature oracle or pairwise AUC says the classes are not separable, the right response is to revise the feature set, corpus, or class definitions.

## Classifier Performance

Classifier performance is evaluated after corpus and feature checks. A classifier underperforms when it fails relative to an oracle, another rung with a justified evidence capability, or its own calibration and confusion expectations.

## Promotion Decisions

- `promote`: evidence is adequate and limitations are documented.
- `revise`: the concept is promising, but corpus, features, priors, or algorithm settings need repair.
- `reject`: the study cannot support its claim under current definitions.
- `defer`: evidence is blocked by missing data, implementation gaps, or unresolved dependencies.
