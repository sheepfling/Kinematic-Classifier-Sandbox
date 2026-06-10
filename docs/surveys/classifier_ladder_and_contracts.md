# Classifier Ladder and Contracts

This note documents the repo's classifier ladder as a sequence of increasingly
structured evidence models. The point is not only to list which methods exist.
The point is to show:

- what random variables each method assumes
- what likelihood or score it computes
- how that score becomes a posterior
- where that rule lives in the code
- what concrete artifact demonstrates the rule on a real run

In the canonical repo story, this note is the `Classifier / Filter Ladder`
pillar. It sits between the selected corpus and the evaluation layer, and its
job is to answer one question consistently across methods: what is the next
`ell_t(.)` and why is that evidence justified?

## Repo Role

The ladder should be read as:

```tex
\text{Validated Corpus}
\rightarrow
\text{Evidence Provider}
\rightarrow
\ell_t(\cdot)
\rightarrow
\text{Posterior Update}
\rightarrow
\text{Evaluation / Promotion}.
```

The repo’s algorithm claim is deliberately narrow:

- each rung adds one new evidence capability
- each rung must be justified by a failure mode of the previous rung
- each rung should have at least one 1D witness problem that makes the upgrade
  legible

## 1. Problem Statement

The common classification problem in this repo is:

- observations arrive through time as `z_1, ..., z_t`
- the hidden class or motion mode is `c_t` or `s_t`
- the method must produce either:
  - a posterior over classes, `p_t(c)`, or
  - a posterior over modes, `p_t(s)`

Across the ladder, the downstream contract is intentionally stable:

```tex
\text{evidence provider}
\rightarrow
\text{log score by class or mode}
\rightarrow
\text{normalized posterior}
\rightarrow
\text{prediction row and metrics}
```

That is the architectural reason these methods can share evaluation and
artifact-generation code even though their internal state is very different.

## 2. Global Notation

The same symbols will be reused across methods:

- `z_t`: observation at time `t`
- `c`: class label in non-switching classifiers
- `s`: mode label in switching classifiers
- `p_t(c)`: posterior over classes after processing time `t`
- `p_t(s)`: posterior over modes after processing time `t`
- `L_t(c)`: class-conditioned likelihood term at time `t`
- `phi_t`: feature vector extracted from a trajectory prefix or window
- `ell_t(c)`: unnormalized log score before softmax normalization
- `x_{t|t}`: filtered latent state estimate in a state-space method
- `nu_t`: innovation residual in the Kalman-family methods

The recurrent posterior normalization step is always some form of:

```tex
p_t(c) = \frac{\exp(\ell_t(c))}{\sum_j \exp(\ell_t(j))}.
```

The important variation across methods is therefore not the final softmax. It is
how `ell_t(c)` is constructed.

## 3. Common Assumptions

The ladder reuses a small number of methodological assumptions:

- additive log evidence is numerically preferable to multiplying tiny
  probabilities directly
- posterior state from the previous step is the natural recursive summary for
  class memory
- higher ladder levels should only be justified by a failure mode that lower
  levels cannot explain
- shared evaluation should consume standardized outputs, not method-specific
  internal states

The methods differ in their stronger assumptions:

- `inference/pointwise_baseline.py`: observation at time `t` is sufficient evidence
- `inference/windowed_baseline.py`: a short feature window is a sufficient summary
- `state_estimate_evidence.py`: a filtered state and covariance are already
  provided
- `inference/sequential_bayes_accumulator.py`: likelihood streams can be accumulated with
  optional forgetting
- `inference/kalman_filter_bank.py`: each class or motion hypothesis induces a
  linear-Gaussian state-space model
- `inference/transition_matrix_accumulator.py`: mode persistence and switching can be
  represented with a finite transition matrix and emission model
- `inference/advanced_state_inference.py`: IMM mixes multiple linear-Gaussian mode
  models while preserving the same posterior/evidence/diagnostic contract and
  emits the current 1D proof artifacts for advanced switching inference

## 4. Ladder Overview

The current ladder is:

1. `inference/pointwise_baseline.py`
2. `inference/windowed_baseline.py`
3. `inference/sequential_bayes_accumulator.py`
4. `inference/kalman_filter_bank.py`
5. `inference/transition_matrix_accumulator.py`
6. `inference/advanced_state_inference.py`

The upgrade path is deliberate:

- `pointwise`: no temporal compression beyond the posterior itself
- `windowed`: compress recent history into engineered features
- `state_estimate`: score the provided filtered state against class templates
- `accumulator`: make recursive evidence accumulation explicit
- `kalman`: let a dynamics model predict the next observation and score the
  innovation
- `transition_matrix`: inject explicit switching structure before paying the
  complexity cost of full multi-model inference
- `IMM`: mix multiple linear-Gaussian mode models when switching structure and
  explicit state mixing are required

The reader-facing ladder is therefore:

| Rung | Algorithm | Adds | Failure addressed | 1D witness |
| --- | --- | --- | --- | --- |
| 0 | `pointwise` | local likelihood baseline | no audited local baseline | pointwise overlap |
| 1 | `windowed` | short local history | outlier and local-noise fragility | windowed outlier/extrema |
| 2 | `sequential_bayes` | recursive memory | pointwise history blindness | sequential history |
| 3 | `kalman_bank` | dynamics-conditioned innovations | endpoint ambiguity under irregular timing | Kalman endpoint match |
| 4 | `transition_matrix` | explicit mode switching | static-class assumption | transition switching |
| 5 | `IMM` proof | switching-aware state inference | demonstrated switching failures | advanced 1D switching witness |
| 6 | `particle_filter_bank` | sampled nonlinear / non-Gaussian evidence | linear-Gaussian assumptions fail under drag, outliers, or mean reversion | nonlinear drag and OU witnesses |
| 7 | `rbpf` | sampled latent mode path plus conditional Kalman state | mixed discrete/continuous latent structure | latent maneuver onset witness |

## 5. Pointwise Evidence Baseline

### 5.1 Problem

`inference/pointwise_baseline.py` asks the simplest possible question:

```tex
\text{How far can direct observation evidence go without explicit memory?}
```

### 5.2 Model

For each class `c`, the observation model is Gaussian with mean `mu_c` and
standard deviation `sigma_c`:

```tex
z_t \mid c \sim \mathcal{N}(\mu_c, \sigma_c^2).
```

### 5.3 Derivation

Bayes rule gives

```tex
p_t(c) \propto p_{t-1}(c)\,p(z_t \mid c).
```

Taking logs yields

```tex
\ell_t(c)
= \log p_{t-1}(c)
- \frac{1}{2}\left[
    \log(2\pi \sigma_c^2)
    + \frac{(z_t - \mu_c)^2}{\sigma_c^2}
  \right].
```

This matters because the method is already recursive even though it has no
explicit trajectory state. The memory is only in the posterior vector.

### 5.4 Implementation Mapping

- `GaussianPointwiseClassifier.update(...)`
- `_gaussian_logpdf(...)`
- `_normalize_log_scores(...)`

### 5.5 Failure Mode

This method fails whenever the class distinction is mostly temporal rather than
instantaneous. It cannot represent trend, persistence, or switching.

## 6. Windowed Evidence Baseline

### 6.1 Problem

`inference/windowed_baseline.py` addresses the first weakness of pointwise scoring:
classes may separate only after short-history statistics are computed.

### 6.2 Variables

The method constructs a feature vector

```tex
\phi_t = g(z_{1:t}) \quad \text{or} \quad \phi_t = g(z_{t-w+1:t}),
```

where `g` is the engineered feature extractor and `w` is the effective window.

Examples in the current code include:

- `running_min`
- `running_max`
- `running_range`
- `robust_min`
- `robust_max`
- `trimmed_range`
- `slope`
- `monotonicity`
- `sign_changes`

### 6.3 Assumptions

The key modeling assumption is conditional independence of selected feature
coordinates within a class:

```tex
p(\phi_t \mid c) \approx \prod_{f \in \mathcal{F}} p(\phi_{t,f} \mid c).
```

That is a naive-Bayes approximation over feature coordinates.

### 6.4 Derivation

With Gaussian feature marginals,

```tex
\ell_t(c)
= \log p_{t-1}(c)
+ \sum_{f \in \mathcal{F}}
  \log \mathcal{N}(\phi_{t,f}; \mu_{c,f}, \sigma_{c,f}^2).
```

This is the first rung where the evidence is no longer a raw observation but a
compressed summary of recent trajectory geometry.

### 6.5 Implementation Mapping

- `inference/windowed_baseline.py`
- `inference/irregular_window_comparison.py`
- `common_experiment_classifier_registry.py`

### 6.6 Failure Mode

The main risk is that the engineered feature family is either:

- under-expressive for the true class geometry, or
- spuriously expressive because the corpus makes the task too easy

That is why this rung must be read together with the feature-analysis and
corpus-adequacy documents.

## 7. Covariance-Aware State Estimate Evidence

Sometimes the input is not a raw observation stream but an already filtered
state estimate with covariance, for example:

```tex
(x_t, P_t) \quad \text{or} \quad (x_{t|t}, P_{t|t})
```

In that case the Kalman update has already been performed elsewhere, so the
right evidence atom is not a new innovation residual. The right rung is a
covariance-aware state-likelihood score.

The class-conditioned evidence can be written as:

```tex
\ell_t(c)
=
-\frac{1}{2}
\left[
    (x_t - \mu_c)^\top \Sigma_{t,c}^{-1}(x_t - \mu_c)
    + \log |\Sigma_{t,c}|
    + d \log(2\pi)
\right].
```

Here `Σ_{t,c}` is the supplied covariance or a class-conditioned covariance
model built from it. If the provided covariance is class-neutral, then
`Σ_{t,c}` can be taken as the supplied covariance plus any class-specific
process or measurement inflation used by the benchmark.

This rung is the right choice when:

- the tracker already gives a state estimate and covariance
- the task is to classify trajectories from filtered states rather than to
  re-run tracking
- the evidence should measure class fit to the filtered state, not the raw
  measurement innovation

In other words, the ladder step is covariance-aware state evidence, not full
Kalman filtering.

## 8. Sequential Bayes Accumulator

### 7.1 Problem

`inference/sequential_bayes_accumulator.py` turns the evidence-combination rule itself
into a first-class module.

### 7.2 Derivation

If the method is given direct likelihoods `L_t(c)`, the update is:

```tex
\ell_t(c) = \lambda \log p_{t-1}(c) + \log L_t(c),
```

followed by the standard normalization. The parameter `lambda` is the
`forgetting_factor`.

In direct Gaussian mode,

```tex
L_t(c) = \mathcal{N}(z_t; \mu_c, \sigma_c^2).
```

So the accumulator can be understood as the pure recursive shell that sits
between pointwise scoring and richer evidence providers.

### 7.3 Confidence Gate

Let

```tex
c_t^\star = \arg\max_c p_t(c).
```

The output label is

```tex
\hat{y}_t =
\begin{cases}
c_t^\star, & p_t(c_t^\star) \ge \tau \\
\texttt{unknown}, & p_t(c_t^\star) < \tau
\end{cases}
```

with `tau = confidence_threshold`.

This matters methodologically because the system is allowed to distinguish
ambiguity from error.

### 7.4 Implementation Mapping

- `SequentialBayesAccumulator.update_with_likelihoods(...)`
- `SequentialBayesAccumulator.update_with_gaussian_observation(...)`
- `inference/monte_carlo_benchmark.py`

### 7.5 Failure Mode

If the supplied evidence is poorly calibrated, recursive accumulation can make
the wrong answer more confident over time. This rung therefore sharpens both
the strengths and weaknesses of the upstream evidence provider.

## 9. Kalman Innovation Bank

### 8.1 Problem

`inference/kalman_filter_bank.py` handles the case where the class distinction is not
just about feature statistics but about consistency with a dynamics model.

### 8.2 State-Space Assumptions

For each model `m`, the code assumes a linear-Gaussian state-space model with
state `x_t^(m)`:

```tex
x_t^{(m)} = F_t^{(m)} x_{t-1}^{(m)} + w_t^{(m)}, \qquad
z_t = H_t^{(m)} x_t^{(m)} + v_t^{(m)},
```

with Gaussian process and measurement noise.

### 8.3 Derivation

The prediction step computes

```tex
\hat{x}_{t|t-1}^{(m)} = F_t^{(m)} x_{t-1|t-1}^{(m)},
\qquad
P_{t|t-1}^{(m)} = F_t^{(m)} P_{t-1|t-1}^{(m)} {F_t^{(m)}}^\top + Q_t^{(m)}.
```

The innovation residual and covariance are

```tex
\nu_t^{(m)} = z_t - H_t^{(m)} \hat{x}_{t|t-1}^{(m)},
\qquad
S_t^{(m)} = H_t^{(m)} P_{t|t-1}^{(m)} {H_t^{(m)}}^\top + R_t^{(m)}.
```

The class evidence is then the innovation likelihood:

```tex
\log L_t(m)
= -\frac{1}{2}\left[
    \log(2\pi S_t^{(m)})
    + \frac{(\nu_t^{(m)})^2}{S_t^{(m)}}
  \right].
```

So the posterior preference for a model is driven by how unsurprising the next
measurement is under that model's predicted state.

### 8.4 Implementation Mapping

- `_innovation_log_likelihood(...)`
- model-bank normalization logic in `inference/kalman_filter_bank.py`
- `inference/kalman_variant_comparison.py`
- `inference/kalman_observable_comparison.py`
- `inference/velocity_aided_kalman_comparison.py`

### 8.5 Failure Mode

This rung still assumes that the model family is expressive enough. If the true
behavior is switching, strongly nonlinear, or non-Gaussian, the innovation
likelihood can become systematically misleading even when numerically stable.

## 10. Transition-Aware Accumulation

### 9.1 Problem

`inference/transition_matrix_accumulator.py` addresses mode persistence and switching
before the repo commits to a full IMM-style design.

### 9.2 Variables

The method derives finite-difference kinematics from the measurement history:

- `hat v_t`: speed proxy
- `hat a_t`: acceleration proxy

Each mode `s` has Gaussian emission templates for:

- speed
- signed acceleration
- absolute acceleration

### 9.3 Derivation

The emission score is

```tex
\log E_t(s)
= \log \mathcal{N}(\hat{v}_t; \mu_s^{(v)}, {\sigma_s^{(v)}}^2)
+ \log \mathcal{N}(\hat{a}_t; \mu_s^{(a)}, {\sigma_s^{(a)}}^2)
+ \log \mathcal{N}(|\hat{a}_t|; \mu_s^{(|a|)}, {\sigma_s^{(|a|)}}^2).
```

If a transition matrix `T` is enabled, the propagated prior is

```tex
\bar{p}_t(s') = \sum_s p_{t-1}(s)\,T_{s,s'}.
```

Then the mode posterior update is

```tex
\log \tilde{p}_t(s) = \log \bar{p}_t(s) + \log E_t(s).
```

### 9.4 Implementation Mapping

- finite-difference feature derivation in `inference/transition_matrix_accumulator.py`
- transition propagation through the configured `T`
- normalization over modes after adding emission terms

### 9.5 Worked Example

The numeric artifact
[transition_matrix_numeric_walkthrough.md](artifacts/transition_matrix_accumulator_v1/transition_matrix_numeric_walkthrough.md)
shows a real switching trajectory around the first switch point, including:

- propagated prior by mode
- speed / acceleration / absolute-acceleration emission terms
- log numerators
- normalized posteriors

This is the current best compact example of the ladder's switching recursion.

### 9.6 Failure Mode

This method still uses a hand-specified transition structure and simple
emissions. It is a proof rung for “explicit switching pressure helps,” not a
final multiple-model solution.

## 10. Rung Sufficiency And Escalation Equations

The ladder is only useful if the repo can answer not just “what is the next
rung?” but “has the current rung earned the right to stay?” The
`rung_sufficiency` package makes that decision explicit.

For study `s` at current rung `r`, define the oracle gap

```tex
g_{\text{oracle}}(s,r)
=
\max\big(0, A_{\text{oracle}}(s,r) - A_{\text{current}}(s,r)\big),
```

the prior-fragility term

```tex
f_{\text{prior}}(s,r)
=
1 - \text{prior\_sensitivity\_score}(s,r),
```

and the measured next-rung gain

```tex
\Delta_{r \rightarrow r^{+}}(s)
=
A_{\text{current}}(s,r^{+}) - A_{\text{current}}(s,r),
```

where `r+` is the configured next rung from the capability matrix.

The current implementation treats a study as learnable when

```tex
\mathbf{1}_{\text{learnable}}(s,r)
=
\mathbf{1}\!\left[
    A_{\text{oracle}} \ge \tau_{\text{oracle}}
    \land
    \mathrm{overlap} \le \tau_{\text{overlap}}
    \land
    \mathrm{AUC}_{\text{pair}} \ge \tau_{\text{AUC}}
    \land
    m_{\text{post}} \ge \tau_{\text{margin}}
\right],
```

with the current default thresholds

```tex
\tau_{\text{oracle}} = 0.85,\qquad
\tau_{\text{overlap}} = 0.90,\qquad
\tau_{\text{AUC}} = 0.70,\qquad
\tau_{\text{margin}} = 0.05.
```

This is the mathematical version of the repo rule that feature and corpus
limits should be identified before algorithm blame is assigned.

The promotion decision is then a gated piecewise rule. Let `tau_prior = 0.12`
and `tau_Delta = 0.05`. Then

```tex
d(s,r)
=
\begin{cases}
    \texttt{revise\_corpus}, & \neg \text{can\_evaluate\_classifier}(s), \\
    \texttt{revise\_features}, & \text{feature gate fails}, \\
    \texttt{revise\_prior}, & f_{\text{prior}}(s,r) > \tau_{\text{prior}}, \\
    \texttt{feature\_limited}, & \text{learnability\_status}(s,r)=\texttt{feature\_limited}, \\
    \texttt{stay}, & r^{+} = \varnothing, \\
    \texttt{reject\_escalation}, & \neg \text{capability\_match}(r,r^{+}), \\
    \texttt{defer\_advanced}, & \Delta_{r \rightarrow r^{+}}(s)\ \text{is unavailable}, \\
    \texttt{promote}, & \Delta_{r \rightarrow r^{+}}(s) \ge \tau_{\Delta}, \\
    \texttt{stay}, & g_{\text{oracle}}(s,r) < \tau_{\text{gap}}, \\
    \texttt{defer\_advanced}, & \text{otherwise}.
\end{cases}
```

where `tau_gap = 0.08` in the default configuration. This is the explicit
ladder-sufficiency rule that stops the repo from escalating methods only
because they are available.

The switching witnesses are the cleanest examples because their measured gains
are hard-coded from dedicated benchmark outputs:

```tex
\Delta_{\text{kalman} \rightarrow \text{transition}}
=
A_{\text{post-switch}}^{\text{transition}}
-
A_{\text{post-switch}}^{\text{kalman}},
```

and

```tex
\Delta_{\text{transition} \rightarrow \text{IMM}}
=
A_{\text{post-switch}}^{\text{IMM}}
-
A_{\text{post-switch}}^{\text{transition}}.
```

That is why the current rung-sufficiency artifact can issue a real promotion
decision for transition-aware accumulation and a measured
`promote`/`defer_advanced` decision for IMM.

### 10.1 Implementation Mapping

- `rung_sufficiency/analysis.py`
- `rung_sufficiency/capability_matrix.py`
- `rung_sufficiency/contracts.py`
- `tests/test_rung_sufficiency.py`
- `tests/test_rung_promotion_decision.py`

## 11. Shared Contracts and Evaluation Surface

The ladder is only useful as a methodology if all methods can be compared
through one artifact schema.

### 10.1 Contract Statement

The practical repo contract is:

```tex
\text{method-specific state and evidence}
\rightarrow
\text{shared prediction rows}
\rightarrow
\text{shared metrics and artifact writers}.
```

### 10.2 Implementation Mapping

- `common_dataset_comparison.py`
- `technique_comparison.py`
- `common_1d_study_adapter.py`
- `common_experiment_harness.py`
- `common_experiment_classifier_registry.py`
- `shared_evaluation.py`
- `contracts.py`
- `generic_inference_contract.py`
- `generic_classification_evidence_proof.py`
- `generic_filtering_contract.py`
- `trajectory_backend_contract.py`
- `backend_adapter_proof.py`
- `inference/advanced_state_inference.py`

### 10.3 Why This Matters

The comparison layer does **not** require every method to share the same
internal representation. It only requires each method to emit compatible:

- identifiers
- predictions
- posterior values
- confidence values
- optional evidence or diagnostics

That is the core proof that the repo is becoming a methodology framework rather
than a set of unrelated scripts. The same contract now covers the IMM proof,
the shared particle-filter branch, the RBPF branch, and the OU-style
mean-reversion witness that sits inside the PF family.

## 12. What This Document Proves

This note is complete only if it supports the following claims:

- the classifier ladder is a sequence of distinct evidence constructions, not
  merely a list of files
- each ladder rung has a formal update rule
- each update rule has an explicit code location
- the transition-matrix rung has a real numeric worked example
- the common-harness layer consumes standardized outputs rather than special
  cases

The next methodological question after this document is therefore not “what
method exists?” but “which rung fails, and what stronger assumption would be
justified next?”.
