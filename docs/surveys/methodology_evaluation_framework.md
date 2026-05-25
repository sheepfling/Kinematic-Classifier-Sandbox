# Methodology Evaluation Framework for Classification Studies

This note covers the evaluation side of the repo. The posterior document
explains how recursive inference works. This document explains how the repo
decides whether that inference is:

- informative
- trustworthy
- sufficiently exercised
- generalizable

The goal is not to list artifacts. The goal is to state the evaluation
quantities, the assumptions behind them, and how they connect to code.

## 1. Problem Statement

The central evaluation question is:

```tex
\text{Given a classifier, feature family, and corpus, what evidence shows whether success or failure is real?}
```

The repo addresses that question with multiple layers because one metric is not
enough:

- priors probe decision fragility
- AUC and overlap probe evidence-space separability
- confusion probes end-to-end decision behavior
- adequacy and leakage probe whether the corpus itself is credible

## 2. Global Notation

The main symbols in this document are:

- `p(A)`: prior probability of class `A`
- `L_t(c)`: likelihood term for class `c` at time `t`
- `C_{a,b}`: confusion count for true class `a` predicted as `b`
- `phi_t`: feature vector at time `t`
- `AUC(a,b)`: pairwise area-under-curve for classes `a` and `b`
- `O(a,b)`: overlap estimate for classes `a` and `b`
- `S_k`: corpus or inspection summary score

The common evaluation chain is:

```tex
\text{corpus}
\rightarrow
\text{features or filters}
\rightarrow
\text{evidence}
\rightarrow
\text{posterior}
\rightarrow
\text{metrics}
\rightarrow
\text{artifact bundle}.
```

That layering matters because a failure can happen at any stage.

## 3. Prior Sensitivity

### 3.1 Problem

`prior_sensitivity_analysis.py` asks whether a decision is being driven by data
or by the prior regime.

### 3.2 Derivation

For a binary comparison between classes `A` and `B`, the posterior log-odds can
be written as:

```tex
\log \frac{p(A \mid z)}{p(B \mid z)}
=
\log \frac{p(A)}{p(B)}
+
\sum_t \log \frac{L_A(z_t)}{L_B(z_t)}.
```

This identity is the right interpretation tool:

- if the accumulated likelihood ratio is large, the prior matters less
- if the likelihood ratio is weak, small prior changes can flip the result

### 3.3 Sweep Logic

The studies sweep a binary prior regime:

```tex
p(A) \in [0.05, 0.95], \qquad p(B)=1-p(A),
```

rerun the classifier, and record:

- the smallest prior shift that changes the decision
- the smallest log-prior shift that changes the decision
- the fraction of trajectories that flip under a moderate perturbation

### 3.4 Implementation Mapping

- `prior_sensitivity_analysis.py`
- `inspection_bundle.py` for bundle-level summary surfacing

### 3.5 Methodological Use

Prior sensitivity is not a replacement for accuracy. It is a fragility measure.
It tells us how much posterior confidence is actually coming from evidence.

## 4. Feature Separability and Pairwise Difficulty

### 4.1 Problem

Before blaming a classifier, the repo asks whether the classes are even
separable in the current feature space.

### 4.2 Pairwise AUC

For class pair `(a,b)`,

```tex
\mathrm{AUC}(a,b) \in [0.5, 1.0].
```

Interpretation:

- `1.0` means strong rank separation
- `0.5` means no useful rank separation

But AUC is not enough on its own because a corpus can make a pair trivially
rank-separable for the wrong reasons.

### 4.3 Overlap and Distance

The repo also computes overlap-style and distance-style quantities such as:

- overlap estimate
- Mahalanobis-like distance
- Bhattacharyya-like distance
- identifiability matrices

These numbers answer a different question from confusion: how much of the
feature geometry itself is shared?

### 4.4 Implementation Mapping

- `feature_analysis.py`
- `short_horizon_identifiability.py`
- `pca_analysis.py`
- `generic_feature_taxonomy.py`

### 4.5 Diagnostic Interpretation

The most important methodological use is:

- low AUC and high overlap imply a feature problem
- good AUC with bad final confusion implies a posterior, calibration, or model
  problem
- near-perfect AUC on a pair declared “hard” implies a corpus problem

## 5. Confusion Matrices

### 5.1 Definition

Confusion is an end-to-end decision summary:

```tex
C_{a,b} = \#\{\text{true class } a, \text{ predicted class } b\}.
```

### 5.2 Why Confusion Alone Is Weak

The same off-diagonal confusion can arise from:

- feature insufficiency
- poor likelihood shape
- prior dominance
- weak within-class mode logic
- corpus bias

So confusion is necessary but not sufficient.

### 5.3 Multi-View Confusion

The richer benchmarks already distinguish:

- transient confusion
- terminal confusion
- phase confusion
- feature-vs-class confusion

This is methodologically stronger than a single final confusion matrix because
it localizes where the failure begins.

## 6. Class Validity and Relabel Pressure

`class_validity.py` evaluates whether a generated example still looks like the
class it was meant to instantiate.

This is conceptually upstream of classifier scoring. The main statuses are:

- `valid_target_class`
- `ambiguous`
- `invalid`
- `relabel_candidate`

Methodologically, this matters because some apparent classifier “failures” are
actually failures of class schema or synthetic realization.

## 7. Corpus Adequacy and Leakage

### 7.1 Problem

`corpus_adequacy_audit.py` asks whether the study data is broad, balanced, and
hard in the intended ways.

### 7.2 Evaluation Axes

The audit checks:

- class balance
- scenario balance
- duration and sample-count balance
- noise and irregular-sampling coverage
- feature excitation
- class-pair boundary coverage
- covariate leakage

### 7.3 Leakage Interpretation

The corpus should not allow nuisance variables such as duration, sample count,
or sampling irregularity to predict class too well on their own. If those
covariates become highly class-linked, the classifier may be learning the corpus
rather than the motion class.

### 7.4 Implementation Mapping

- `corpus_adequacy_audit.py`
- `coverage_report.py`
- `generated_corpus_features.py`
- `corpus_classifier_scoring.py`

### 7.5 Current Methodological Use

The adequacy audit is already acting as a real gate. It can say not just “pass”
or “fail,” but **why**:

- over-separated hard pairs
- class-linked irregular-sampling variables
- weak feature excitation for certain feature bundles

That is the right shape for a credible methodology audit.

## 8. PCA and Inspection Bundles

### 8.1 Problem

The repo also needs compact summary surfaces across many feature sets and class
pairs.

### 8.2 PCA Role

`pca_analysis.py` is not a classifier. It is a dimensional diagnostic. It asks
whether a lower-dimensional projection already reveals clean class geometry or
obvious overlap.

### 8.3 Inspection Bundle Role

`inspection_bundle.py` aggregates feature-set and class-pair summaries into one
bundle so the repo can recommend:

- the strongest current feature set
- the hardest current class boundary

This layer is useful because it turns many raw study outputs into one decision
surface.

## 9. Generated Class and Feature Exploration

### 9.1 Problem

The repo now has a second feature-analysis path beyond the hand-authored
synthetic corpus: objective-driven generated corpus records that may be
relabelled, filtered, or promoted after backend execution.

### 9.2 Generated Feature Integration

`generated_corpus_features.py` creates the bridge:

```tex
\text{generated candidate}
\rightarrow
\text{executed trajectory}
\rightarrow
\text{validity-adjusted label}
\rightarrow
\text{feature extraction}
\rightarrow
\text{excitation and separability artifacts}.
```

This matters because feature analysis is no longer tied only to the original
trajectory generator. It can now inspect generated candidates from search and
archive mechanisms using the same feature registry and excitation logic.

### 9.3 Corpus-Conditioned Classifier Scoring

`corpus_classifier_scoring.py` evaluates how the existing classifier ladder
behaves on those generated trajectories. Its per-method stress proxy is
approximately:

```tex
\text{stress}
\approx
0.5 \cdot (1 - \text{margin})
+ 0.5 \cdot \text{entropy}
+ 0.35 \cdot \mathbf{1}\{\text{final prediction wrong}\}.
```

This is not a formal posterior quantity. It is a composite diagnostic that says:

- small margin is stressful
- high entropy is stressful
- a final error is extra stressful

The point is to rank generated trajectories by how much pressure they place on
the current classifier ladder.

### 9.4 Class Validity as a Bridge

`class_validity.py` is the key bridge here. Without it, generated trajectories
would be scored only against their intended class rather than the class they
most resemble after execution and corruption. That would make the entire
generated-corpus evaluation layer much less trustworthy.

## 10. Failure Modes

This evaluation framework can still fail if:

- a metric is interpreted outside its intended layer
- the corpus is too biased for AUC or confusion to mean what they seem to mean
- feature geometry is poor but the blame is assigned to the posterior update
- prior sensitivity is ignored on supposedly “accurate” but fragile cases

That is why the evaluation stack is intentionally multi-view.

## 11. What This Document Proves

This note is complete only if it supports the following claims:

- the repo distinguishes evidence quality, posterior behavior, and corpus
  quality
- priors, AUC, overlap, confusion, and adequacy are defined for different
  methodological purposes
- each major metric family has an implementation surface in the code
- the artifact families are interpretable as measurements of those quantities,
  not just dashboards
