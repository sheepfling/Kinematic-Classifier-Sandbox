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

Static checks run before treating classifier performance as evidence:

- class definition validity
- feature availability and units
- feature excitation by class and sensor regime
- covariate leakage audit
- pairwise overlap and AUC
- artifact provenance and schema compatibility

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
