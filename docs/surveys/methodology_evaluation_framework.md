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

In the canonical repo story, this note is the `Evaluation / Promotion` pillar.
It is the part of the repo that decides whether a study candidate deserves
promotion after the Corpus Explorer has selected data and the Classifier Ladder
has emitted evidence.

## Repo Role

This note is the downstream judgment layer for

```tex
s = (D, f, C, m, \pi, b).
```

Its intended interpretation order is:

```tex
\text{corpus adequacy}
\rightarrow
\text{class validity}
\rightarrow
\text{feature excitation / separability}
\rightarrow
\text{oracle gap}
\rightarrow
\text{posterior / prior sensitivity}
\rightarrow
\text{confusion}
\rightarrow
d,
```

with

```tex
d \in \{\text{promote}, \text{revise}, \text{reject}, \text{defer}\}.
```

That ordering is a methodological rule for the repo: top-line accuracy is not
the first thing to trust.

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

`inference/prior_sensitivity_analysis.py` asks whether a decision is being driven by data
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

- `inference/prior_sensitivity_analysis.py`
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
hard in the intended ways without leaking labels through metadata shortcuts.

A corpus is not good merely because classifiers perform well on it. A corpus is
good when it exercises the declared classes, feature families, boundary cases,
priors, and sensor conditions without letting the classifier solve the problem
from duration, sample count, irregular sampling, or other nuisance variables.

The corpus is treated as

```tex
D = \{(\tau_i, c_i, s_i, m_i)\}_{i=1}^{N},
```

where:

- `τ_i`: trajectory `i`
- `c_i`: validated class label
- `s_i`: tier or scenario label
- `m_i`: metadata such as duration, sample count, `mean_dt`, `std_dt`,
  irregularity, noise, and outlier rate

The current code now makes this explicit with a bounded scorecard:

```tex
Q_{\text{corpus}}(D)
=
\frac{
B_{\text{class}}
+
B_{\text{tier}}
+
B_{\text{covariates}}
+
E_{\text{feature}}
+
C_{\text{pair}}
+
V
-
L
-
T
-
G
}{6}.
```

The positive terms should be high. The penalties should be low.

### 7.2 Evaluation Axes

The implemented scorecard terms are:

| term | meaning | desired direction | current artifact |
| --- | --- | --- | --- |
| `B_class` | class balance | high | `class_balance.csv` |
| `B_tier` | tier balance | high | `class_balance.csv` |
| `B_covariates` | metadata balance across classes | high | `covariate_leakage_audit.csv` |
| `E_feature` | feature excitation coverage | high | `feature_set_coverage.csv` |
| `C_pair` | class-pair boundary coverage | high | `class_pair_coverage.csv` |
| `V` | class-validity score | high | `class_validity_audit.csv` |
| `L` | leakage penalty | low | `covariate_leakage_audit.csv` |
| `T` | triviality penalty | low | `class_pair_coverage.csv` |
| `G` | degeneracy penalty | low | `corpus_degeneracy_report.csv` |

For the current common synthetic corpus, the latest audit reports:

- `B_class = 1.000`
- `B_tier = 1.000`
- `B_covariates = 0.704`
- `E_feature = 0.562`
- `C_pair = 0.731`
- `V = 1.000`
- `L = 0.613`
- `T = 0.500`
- `G = 0.023`
- `Q_corpus = 0.644`

So the corpus currently fails not because balance is poor, but because leakage
and hard-pair triviality are still too large.

### 7.3 Class and Tier Balance

Let

```tex
\hat{p}_D(c)
=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[c_i=c]
```

be the empirical class distribution. For a uniform target over class set
`𝒞`, the balance score is

```tex
B_{\text{class}}(D)
=
1-\frac{1}{2}\sum_{c\in\mathcal{C}}|\hat{p}_D(c)-p^\star(c)|.
```

The code uses the same total-variation-style score for tier balance:

```tex
B_{\text{tier}}(D)
=
1-\frac{1}{2}\sum_{s\in\mathcal{S}}|\hat{p}_D(s)-p^\star(s)|.
```

Interpretation:

- `1.0` means the observed distribution matches the target distribution
- values near `0` mean one class or tier is dominating the study

On the current common corpus both scores are `1.000`, which is why class and
tier balance are not the reason for failure.

### 7.4 Covariate Balance and Leakage

For continuous metadata covariates `q` such as duration, sample count,
`mean_dt`, `std_dt`, `sampling_irregularity`, and outlier fraction, the audit
now separates two questions:

1. are class-conditional covariate distributions similar?
2. can the covariate predict class on its own?

The balance side uses a normalized 1D Wasserstein distance:

```tex
I_q(D)
=
\max_{a,b\in\mathcal{C}}
\frac{W_1(F_{q|a},F_{q|b})}{R_q+\epsilon},
\qquad
B_q(D)=1-\operatorname{clip}(I_q(D),0,1).
```

The leakage side uses covariate-only predictability:

```tex
L_q(D)
=
\operatorname{clip}\left(\frac{\mathrm{AUC}_q-0.5}{0.5},0,1\right).
```

The current aggregate scores are:

```tex
B_{\text{covariates}}(D)=\frac{1}{|\mathcal{Q}|}\sum_q \left[1-\max\{L_q(D),B_q^{\text{risk}}(D)\}\right],
\qquad
L(D)=\max_q L_q(D).
```

Operationally:

- `std_dt` is yellow with covariate-only AUC `0.806`
- `sampling_irregularity` is yellow with covariate-only AUC `0.805`
- the aggregate leakage penalty is `L = 0.613`

So the current corpus is still class-linked through irregular-sampling metadata
more than it should be.

### 7.5 Feature Excitation Coverage

The audit no longer treats feature excitation as a checklist item. It scores
whether each declared feature is actually activated across classes and tiers.

For feature `j`, class `c`, and tier `s`, the current implementation uses a
moderate-threshold excitation count:

```tex
E_{jcs}(D)
=
\min\left(
1,
\frac{
\#\{i : c_i=c,\ s_i=s,\ |\phi_{ij}| \ge \tau^{\text{moderate}}_j\}
}{n_{\min}}
\right).
```

Then:

```tex
E_{\text{feature}}(D)
=
\frac{1}{|\mathcal{J}||\mathcal{C}||\mathcal{S}|}
\sum_{j,c,s} E_{jcs}(D).
```

The current audit reports `E_feature = 0.562`. That is not a failure, but it is
also not saturated coverage. It means the corpus is exercising many features,
but not uniformly or strongly enough across every class-tier cell.

### 7.6 Class-Pair Boundary Coverage and Triviality

For each declared important pair `h = (a,b)`, the audit checks both:

1. whether the required tiers are present
2. whether the pair remains hard in the intended way

The current implementation uses a pair boundary score:

```tex
C_h(D)
=
\text{tier\_fraction}(h)
\times
\text{difficulty-window score}(h),
```

where the difficulty-window score depends on the expected pair type:

- easy pairs should have high AUC
- hard pairs should not become too separable
- hard pairs should retain overlap

The aggregate boundary term is:

```tex
C_{\text{pair}}(D)
=
\frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}} C_h(D).
```

The triviality penalty is explicit for declared hard pairs:

```tex
T_h(D)
=
\max\left(
0,
\frac{\mathrm{AUC}_h-\tau_{\text{easy}}}{1-\tau_{\text{easy}}}
\right),
\qquad
T(D)=\frac{1}{|\mathcal{H}_{\text{hard}}|}\sum_h T_h(D).
```

This is the direct explanation for the current red finding:

- `constant_acceleration vs maneuver` is declared hard
- observed pairwise AUC is `1.000`
- overlap is `0.000`
- therefore the pair is over-separated
- `T = 0.500`

This is why the corpus fails even though balance is perfect.

### 7.7 Class Validity

The common synthetic corpus is generator-defined, so the current adequacy audit
treats its labels as valid by construction:

```tex
V(D)=1.0
```

for this specific corpus family.

That is not the same thing as saying class validity is unimportant. It means
the detailed relabel logic lives one layer downstream in `class_validity.py`
for generated or objective-driven corpora, where labels are not guaranteed by
construction.

### 7.8 Degeneracy

The degeneracy penalty catches duplicate or structurally invalid corpora. The
implemented terms are:

```tex
G_{\text{dup}}(D)
=
\frac{
\#\{(i,j): i<j,\ d(\phi_i,\phi_j)<\epsilon\}
}{
\binom{N}{2}
},
```

```tex
G_{\text{invalid}}(D)
=
\frac{\#\{i : t_{i,k+1}\le t_{i,k}\ \text{for some }k\}}{N},
\qquad
G_{\text{physical}}(D)
=
\frac{\#\{i : \max_k |a_{ik}| > a_{\max}\}}{N}.
```

The aggregate penalty is:

```tex
G(D)=0.5\,G_{\text{dup}}+0.25\,G_{\text{invalid}}+0.25\,G_{\text{physical}}.
```

The current audit reports:

- `G_dup = 0.032`
- `G_invalid = 0.000`
- `G_physical = 0.029`
- `G = 0.023`

So degeneracy is present, but it is not the dominant reason for failure.

### 7.9 Implementation Mapping

- `corpus_adequacy_audit.py`
- `coverage_report.py`
- `generated_corpus_features.py`
- `corpus_classifier_scoring.py`

### 7.10 Current Methodological Use

The adequacy audit is now acting as a real scorecard and a real gate. It can
say not just `pass` or `fail`, but why:

- over-separated hard pairs
- class-linked irregular-sampling variables
- incomplete feature excitation
- duplicate or mildly nonphysical structure

The current common synthetic corpus therefore fails for principled reasons:

- `Q_corpus = 0.644`, below the yellow gate of `0.65`
- leakage is too high: `L = 0.613`
- hard-pair triviality is too high: `T = 0.500`
- the main red pair remains `constant_acceleration vs maneuver`

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
