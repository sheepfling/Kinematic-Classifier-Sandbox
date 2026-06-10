



# Kinematic Classifier Methodology Compendium


This document combines the current survey notes into one reference file.

Use it when you want the full methodology stack in one place rather than reading the survey notes separately.

For a shorter narrative entry point, start with [artifacts/latex/kinematic_classifier_methodology.pdf](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/latex/kinematic_classifier_methodology.pdf).

This compendium is the long-form reference companion to that paper.
## Included Documents


1. [Posterior Update Math](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/posterior_update_math.md) with rendered companion [`artifacts/posterior_update_math.pdf`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/posterior_update_math.pdf).
2. [Methodology Evaluation Framework](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/methodology_evaluation_framework.md) with rendered companion [`artifacts/methodology_evaluation_framework.pdf`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/methodology_evaluation_framework.pdf).
3. [Classifier Ladder and Contracts](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/classifier_ladder_and_contracts.md) with rendered companion [`artifacts/classifier_ladder_and_contracts.pdf`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/classifier_ladder_and_contracts.pdf).
4. [Corpus Generation and Search](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/corpus_generation_and_search.md) with rendered companion [`artifacts/corpus_generation_and_search.pdf`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/corpus_generation_and_search.pdf).
5. [Dimensional Lift and Advanced Filter Gates](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/dimensional_lift_and_advanced_filter_gates.md) with rendered companion [`artifacts/dimensional_lift_and_advanced_filter_gates.pdf`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/dimensional_lift_and_advanced_filter_gates.pdf).
## Part 1. Posterior Update Math


Source: [posterior_update_math.md](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/posterior_update_math.md)

This note documents the posterior update math used by the two active benchmark
families in the sandbox:

- `witnesses/toy_1d/`: class-matched latent-state filter bank
- `inference/identity_1d.py`: direct speed-identity classifier over `bike`, `horse`, `car`

The goal is not to present an abstract Bayesian classifier. It is to show the
specific scoring structure the repo is actually using today.

It also serves as a methodology note. The repo is using 1D benchmarks to prove
an exploratory classification workflow:

- corpus generation and stress cases
- feature or filter based evidence extraction
- posterior accumulation through time
- confusion, entropy, adequacy, and separability analysis
- artifact generation that can later be reused in richer settings

So the relevant question is not only "what is the equation?" It is also "what
does this benchmark prove about the larger methodology?"

Primary implementation surfaces covered explicitly by this note:

- `witnesses/toy_1d/`
- `inference/identity_1d.py`
- `witnesses/toy_1d/posterior_explainer.py`
- `inference/identity_inference/posterior_explainer.py`
- `witnesses/toy_1d/bayesian_walkthroughs.py`
- posterior-oriented artifact writers in `artifacts.py`

## 0.1 Layered methodology view

The current architecture is best read as:

```tex
\text{corpus}
\rightarrow
\text{features or filters}
\rightarrow
\text{evidence}
\rightarrow
\text{posterior}
\rightarrow
\text{metrics and artifacts}.
```

That is why the same repo now contains:

- feature studies
- prior-sensitivity studies
- confusion and identifiability studies
- posterior walkthrough artifacts
- filtering comparisons

The 1D work matters because it exercises those layers cleanly before trying to
lift them into 3D.

## 1. Generic class-posterior update

For both benchmarks, the class posterior is updated recursively:

```tex
p(s_i \mid z_{1:t}) \propto p(s_i \mid z_{1:t-1}) \, L_i(z_t, z_{1:t-1})
```

with normalization

```tex
p(s_i \mid z_{1:t}) =
\frac{p(s_i \mid z_{1:t-1}) \, L_i(z_t, z_{1:t-1})}
{\sum_j p(s_j \mid z_{1:t-1}) \, L_j(z_t, z_{1:t-1})}.
```

In log space:

```tex
\log \tilde{w}_{i,t} = \log w_{i,t-1} + \log L_i(z_t, z_{1:t-1}),
\qquad
w_{i,t} = \frac{\exp(\log \tilde{w}_{i,t})}
{\sum_j \exp(\log \tilde{w}_{j,t})}.
```

That is the common structure across both files. The difference is what
`L_i(...)` contains.

## 1.0 Evidence-provider interpretation

At the meta level, each classifier family can be seen as an evidence provider:

```tex
\mathcal{E}(\text{history}) \rightarrow \{\log L_i\}_{i=1}^K.
```

What changes across methods is how the evidence is produced:

- pointwise methods score one observation or feature vector
- windowed methods score short temporal structure
- accumulators score cumulative feature evidence
- Kalman-style methods score innovations and residual structure

The posterior updater should not care which of those produced the
log-likelihoods, as long as the output has a common per-class form.

The implemented method families are easiest to read as a single table:

| method | evidence input `y_k` | evidence term `ℓ_k(c)` | role in the repo |
| --- | --- | --- | --- |
| pointwise | current observation or local feature | `log p(y_k \| c)` | weak lower-bound classifier |
| windowed | fixed-window feature vector `ϕ_k` | `log p(ϕ_k \| c)` | adds short temporal context |
| sequential Bayes | recursively emitted evidence stream | accumulated per-step `ℓ_k(c)` | composes evidence through time |
| Kalman bank | innovation `ν_{k,c}` | `log N(ν_{k,c}; 0, S_{k,c})` | scores class-conditioned dynamics mismatch |
| transition model | evidence plus class transition prior | `ℓ_k(c) + log \sum_j T_{jc} p_{k-1}(j)` | handles switching classes |

This table is the answer to "what is `y_k`?" in practice:

- pointwise uses the newest sample directly
- windowed uses a short feature summary
- sequential Bayes uses whatever per-step evidence the classifier emits
- Kalman bank uses model innovation residuals
- transition model uses a prior-predicted class distribution before the evidence update

## 1.1 Implementation mapping

The math in this note maps to a few concrete implementation surfaces:

- `witnesses.toy_1d.gaussian_interval_probability(...)`
  - symmetric Gaussian region mass for toy speed and acceleration envelopes
- `witnesses.toy_1d._innovation_log_likelihood(...)`
  - Gaussian innovation log density
- `witnesses.toy_1d.run_class_bank(...)`
  - toy recursive filter-bank classification loop
- `identity_1d.run_identity_benchmark(...)`
  - identity recursive direct-speed classification loop
- `witnesses/toy_1d/posterior_explainer.py`
  - toy success/failure/comparison/margin-trace posterior diagnostics
- `inference/identity_inference/posterior_explainer.py`
  - identity boundary-failure posterior diagnostics

So this is not just theory. It is intended to stay close to the actual code
paths that generate the benchmark and artifact outputs.

## 1.1 What Is Random In These Benchmarks

The cleanest way to read the repo is:

- `s_i` is the class label
- `z_t` is the measurement at time `t`
- `x_t` is a latent state only in the toy benchmark
- `p(s_i | z_{1:t})` is the posterior class probability after seeing data up to step `t`

The important split is:

- `toy_1d` is a latent-state model with class-conditioned filters
- `identity_1d` is a direct measurement classifier with no latent Kalman state

So the two 1D classifiers are not using the same probabilistic object.

## 1.2 Are We Chopping The Gaussian Probability Space?

Mostly no. The main measurement-fit terms are standard Gaussian log densities.

Where the code does integrate or "cut" probability space is in the soft
validity terms. Those terms ask questions like:

```tex
P(|v_t| \le c), \qquad P(|a_t| \le c), \qquad P(z_t \le c)
```

Those are not pointwise PDF values. They are integrated Gaussian mass over a
region:

- interval mass for `|v_t| <= c` or `|a_t| <= c`
- upper-tail or lower-tail mass for one-sided checks

So the repo currently uses both:

- Gaussian PDF or log-PDF terms for "how well does this exact measurement fit?"
- Gaussian CDF-derived terms for "how much class probability mass lies inside a valid region?"

## 2. Toy 1D benchmark

### 2.1 Latent state and measurement model

The toy benchmark uses a latent kinematic state

```tex
x_t = \begin{bmatrix} p_t \\ v_t \\ a_t \end{bmatrix}
```

with a scalar position measurement

```tex
z_t = H x_t + \nu_t,
\qquad
H = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix},
\qquad
\nu_t \sim \mathcal{N}(0, R).
```

Each class `s_i` maintains its own predicted and updated state moments
`(\mu_{i,t}, P_{i,t})`.

## 2.1 How The Toy 1D Classifier Is Set Up

The toy classifier is a class bank.

For each class:

1. keep a separate latent state mean and covariance
2. predict that state forward using class-specific dynamics
3. score the new position measurement under that class
4. add class-specific soft plausibility terms
5. update the class posterior weight

So the toy benchmark is not:

- one shared tracker followed by a classifier
- or a static feature classifier over a finished trajectory

It is a joint recursive filter-bank style classifier.

## 2.1a Toy 1D class semantics

The toy classes are behavioral, not identity-like:

- `brake`: deceleration and stopping pressure
- `coast`: low-drive, weakly structured motion
- `drift`: sustained reverse or backtracking motion
- `maneuver`: oscillatory or regime-switching motion
- `powered`: persistent positive drive and higher speed
- `unknown`: intentionally hard-to-anchor behavior

That is why the benchmark also tracks phases, transient summaries, terminal
summaries, and feature-vs-class confusion. These are not just end labels.

## 2.1b Toy implementation loop

`run_class_bank(...)` effectively does this for each step:

1. predict latent state moments for each class
2. compute the innovation log likelihood
3. compute soft speed and acceleration region terms
4. compute behavior, observed-kinematics, and within-class mode terms
5. add the previous log prior
6. normalize across classes
7. record posterior, entropy, detected phase, and feature probabilities

That last part is why the repo can generate rich posterior walkthroughs instead
of only final confusion matrices.

Primary recursive implementation surface:

- `run_class_bank(...)` in `witnesses/toy_1d/`

### 2.2 Innovation likelihood

For class `s_i`, after prediction:

```tex
\hat{x}_{i,t}^{-} = F_i \hat{x}_{i,t-1}^{+} + u_i,
\qquad
P_{i,t}^{-} = F_i P_{i,t-1}^{+} F_i^\top + Q_i.
```

The measurement innovation is

```tex
r_{i,t} = z_t - H \hat{x}_{i,t}^{-},
\qquad
S_{i,t} = H P_{i,t}^{-} H^\top + R.
```

The core measurement likelihood is Gaussian:

```tex
p(z_t \mid s_i, z_{1:t-1}) =
\mathcal{N}(r_{i,t}; 0, S_{i,t}).
```

In log form:

```tex
\log L^{\text{dyn}}_{i,t}
= -\frac{1}{2}
\left[
\log(2 \pi S_{i,t}) + \frac{r_{i,t}^2}{S_{i,t}}
\right].
```

This is the `dyn` term in the toy posterior artifacts.

This is the part that corresponds most directly to the textbook

```tex
p(z_t \mid s_i)
```

term. It is a Gaussian density on the innovation, not a thresholded region
probability.

### 2.3 Soft envelope terms

The toy model does not hard-threshold velocity or acceleration. It uses
interval probabilities under the class-conditioned posterior state:

```tex
P(|v_t| \le v_{\max,i} \mid s_i),
\qquad
P(|a_t| \le a_{\max,i} \mid s_i).
```

These are Gaussian CDF differences. For a scalar Gaussian
`y \sim \mathcal{N}(\mu, \sigma^2)`, the symmetric interval probability is

```tex
P(|y| \le c) = \Phi\!\left(\frac{c-\mu}{\sigma}\right)
-
\Phi\!\left(\frac{-c-\mu}{\sigma}\right).
```

The corresponding log terms are

```tex
\log L^{\text{speed}}_{i,t}
= \lambda_v \log(\epsilon + P(|v_t| \le v_{\max,i} \mid s_i)),
```

```tex
\log L^{\text{accel}}_{i,t}
= \lambda_a \log(\epsilon + P(|a_t| \le a_{\max,i} \mid s_i)).
```

These are the `speed` and `accel` terms in the toy artifacts.

### 2.3 What "Cutting Probability Space" Means In Toy 1D

This is the main place where the toy classifier "cuts" the Gaussian
probability space.

For example, if class `s_i` says velocity should stay inside `[-v_max, v_max]`,
the code does not ask only for the density at the current mean velocity. It
asks for the total posterior Gaussian mass that lies inside that interval.

So instead of using only:

```tex
\mathcal{N}(v_t; \mu_v, \sigma_v^2),
```

it also uses:

```tex
P(-v_{\max,i} \le v_t \le v_{\max,i} \mid s_i).
```

That is a soft geometric cut on the probability space. It says:

- classes are rewarded when much of their uncertainty lies in the valid region
- classes are penalized when much of their uncertainty lies outside it

The same logic applies to acceleration limits.

### 2.4 Behavior, observed-kinematics, and mode terms

The remaining toy likelihood is composite rather than a single closed-form
physical model.

Behavior terms compare latent posterior moments to class signatures:

```tex
\log L^{\text{behavior}}_{i,t}
= \log L^{\text{vel-center}}_{i,t}
+ \log L^{\text{accel-center}}_{i,t}
+ \log L^{\text{direction}}_{i,t}
+ \log L^{\text{oscillation}}_{i,t}.
```

Observed-kinematics terms use finite differences of the measurement history:

```tex
\hat{v}^{\text{obs}}_t = \frac{z_t - z_{t-1}}{\Delta t},
\qquad
\hat{a}^{\text{obs}}_t = \frac{z_t - 2 z_{t-1} + z_{t-2}}{\Delta t^2},
```

then score them against class signatures with Gaussian log densities:

```tex
\log L^{\text{obs}}_{i,t}
= \log L^{\text{obs-vel}}_{i,t}
+ \log L^{\text{obs-accel}}_{i,t}.
```

For `brake` and `maneuver`, there is also a small within-class mode mixture:

```tex
\log L^{\text{mode}}_{i,t}
\approx
\log\!\left(
\frac{1}{M_i}
\sum_{m=1}^{M_i} \exp(\ell_{i,t,m})
\right)
```

followed by a class-specific affine rescaling. This is the `mode_mix` term in
the toy benchmark.

### 2.5 Full toy class score

The implemented toy class score is:

```tex
\log L_{i,t}
= \log L^{\text{dyn}}_{i,t}
+ \log L^{\text{speed}}_{i,t}
+ \log L^{\text{accel}}_{i,t}
+ \log L^{\text{behavior}}_{i,t}
+ \log L^{\text{obs}}_{i,t}
+ \log L^{\text{mode}}_{i,t}
- \mathbf{1}\{s_i=\text{unknown}\}\,\gamma_{\text{unknown}}.
```

The posterior update is therefore

```tex
\log \tilde{w}_{i,t}
= \log w_{i,t-1} + \log L_{i,t}.
```

This is why the toy posterior walkthrough artifacts show per-class columns for:

- prior weight
- `dyn`
- `speed`
- `accel`
- behavior subterms
- observed-kinematics subterms
- `mode_mix`
- total
- posterior

## 2.5a How well the toy benchmark works inside the framework

The current toy benchmark is useful precisely because it is mixed:

- overall accuracy is `0.625`
- transient accuracy is `0.458`
- terminal accuracy is `0.542`
- `drift`, `powered`, and `unknown` are strong
- `brake` is weak
- `coast` is currently the clearest miss

Methodologically, that is still productive. The toy path now tells us whether
errors come from:

- poor feature excitation
- bad class geometry
- weak within-class mode design
- phase-label mismatch
- or posterior terms that become decisive for the wrong reason

The entropy trace is also informative: mean posterior entropy drops from about
`1.48` at step 1 to about `0.19` by step 20. So the posterior is becoming
decisive, even when some decisions are decisively wrong.

## 2.6 What The Toy Posterior Really Represents

The toy posterior after each step is:

```tex
p(s_i \mid z_{1:t})
```

not

```tex
p(x_t \mid z_{1:t})
```

The state posterior exists separately inside each class-conditioned filter.

So a more complete mental model is:

- inside each class: estimate `p(x_t | s_i, z_{1:t})`
- across classes: compare class scores and update `p(s_i | z_{1:t})`

That is why the benchmark can say:

- class `drift` currently has posterior `0.63`
- while the `drift` filter also has its own mean and covariance for `[p, v, a]`

## 3. Identity benchmark

### 3.1 Measurement model

The identity benchmark is simpler. The measurement at step `t` is a directly
observed scalar speed:

```tex
z_t = \text{observed speed in mph}.
```

There is no latent Kalman state here. Each class `s_i` is represented by a
speed-shape prior and a few history terms.

## 3.1 How The Identity 1D Classifier Is Set Up

The identity classifier is much simpler than toy.

At each step it takes one observed speed sample and asks:

- how likely is this speed under the `bike` speed model?
- how likely is it under the `horse` speed model?
- how likely is it under the `car` speed model?

Then it adjusts those instantaneous fits with:

- a soft upper-speed validity term
- a running-history term
- a short-window mode term
- a recent-dynamics term

So identity is not estimating a latent PVA state. It is recursively updating
class weights from direct speed evidence.

## 3.1a Identity 1D class semantics

The identity benchmark is simpler and closer to static regime identity:

- `bike`: lower-speed envelope with some surge behavior
- `horse`: middle-speed envelope, especially near horse limits
- `car`: higher-speed envelope with persistent push or overspeed relative to horse

This is useful because it isolates direct observation-space evidence before
latent-state filtering is introduced.

## 3.1b Identity implementation loop

`run_identity_benchmark(...)` is the lighter-weight analogue:

1. read the current observed speed
2. compute the direct speed-shape log density for each class
3. compute the one-sided speed-validity mass term
4. compute running-history, mode-shape, and recent-dynamics terms
5. add the previous log prior
6. normalize across classes
7. record posterior, entropy, and detected features

That shared structure is the important methodological point. Toy and identity
use different evidence models, but the recursive class-posterior story is the
same.

Primary recursive implementation surface:

- `run_identity_benchmark(...)` in `inference/identity_1d.py`

### 3.2 Base speed-shape likelihood

For class `s_i` with cruise mean `\mu_i` and class spread `\sigma_i`, the
instantaneous speed fit is

```tex
\log L^{\text{speed-shape}}_{i,t}
= \log \mathcal{N}(z_t; \mu_i, \sigma_i^2 + \sigma_{\text{obs}}^2).
```

This is the dominant `speed_shape` term shown in the identity artifacts.

This is again a Gaussian density term, not a chopped region probability.

### 3.3 Soft speed-validity term

Each class also has a maximum plausible speed. Instead of a hard gate, the code
uses the Gaussian probability that the current observed speed is below the
class limit plus a small class-specific margin:

```tex
P(z_t \le v_{\max,i} + \delta_i).
```

The log contribution is

```tex
\log L^{\text{valid}}_{i,t}
= 1.4 \, \log\!\big(P(z_t \le v_{\max,i} + \delta_i)\big).
```

This is the `speed_validity` term in the identity posterior artifacts.

### 3.3 What "Cutting Probability Space" Means In Identity 1D

This is the identity-side version of a soft cut.

The model uses a one-sided Gaussian mass term:

```tex
P(z_t \le v_{\max,i} + \delta_i)
```

instead of a hard statement like:

- valid if `z_t <= v_max`
- invalid otherwise

That matters because a noisy measurement slightly above the nominal limit does
not instantly collapse the class to zero. It only receives a softer penalty.

### 3.4 History, mode, and dynamics terms

The identity model also scores short-window behavior:

History-shape term:

```tex
\log L^{\text{history}}_{i,t}
= 0.45 \,
\log \mathcal{N}(\bar{z}_{1:t}; \mu_i, \sigma_i^2 + \sigma_{\text{hist},t}^2)
```

where `\bar{z}_{1:t}` is the running mean speed and `\sigma_{\text{hist},t}^2`
is inflated by observation noise and recent spread.

Mode-shape term:

```tex
\log L^{\text{mode}}_{i,t}
\approx
\alpha_i + \beta_i
\left[
\log\!\left(
\frac{1}{M_i}
\sum_{m=1}^{M_i} \exp(\ell^{\text{mode}}_{i,t,m})
\right)
\right].
```

This gives each class several short-window regimes instead of a single cruise
template.

Dynamics-shape term:

```tex
\log L^{\text{dyn-shape}}_{i,t}
= f_i(\text{mean delta}, \text{mean abs delta}, \text{flip rate}, \text{cadence-like score}),
```

where `f_i` is a weighted sum of Gaussian log densities and small log bonuses.

### 3.5 Full identity class score

The implemented identity score is:

```tex
\log L_{i,t}
=
\log L^{\text{speed-shape}}_{i,t}
\log L^{\text{valid}}_{i,t}
\log L^{\text{history}}_{i,t}
\log L^{\text{mode}}_{i,t}
\log L^{\text{dyn-shape}}_{i,t}.
```

Then the posterior update is the same recursive form:

```tex
\log \tilde{w}_{i,t} = \log w_{i,t-1} + \log L_{i,t}.
```

This is why the identity posterior artifacts show:

- `speed_shape`
- `speed_validity`
- `history_shape`
- `mode_shape`
- `dynamics_shape`
- total
- posterior

## 3.5a How well the identity benchmark works inside the framework

The current identity benchmark is stronger and simpler than toy:

- overall accuracy is `0.875`
- transient accuracy is `0.861`
- terminal accuracy is `0.889`
- `car` and `horse` are strong
- the main remaining boundary pressure is `bike` versus `horse`

This benchmark answers a different methodological question from toy. It asks
how far recursive class evidence can go using direct observation-space fits,
soft limits, and short-window shape terms, without a full latent-state filter.

## 3.6 What The Identity Posterior Really Represents

The identity posterior is directly:

```tex
p(s_i \mid z_{1:t})
```

with `z_t` equal to observed speed. There is no separate latent state posterior
inside the identity classifier.

So compared with toy:

- toy has class posterior plus per-class state posteriors
- identity has class posterior only

## 3.7 Calibration And Posterior Quality

Posterior quality is not the same thing as raw accuracy. In this repo it means:

- does the posterior put high probability on the correct class?
- does confidence mean what it says?
- does the posterior stay stable under small prior changes when the evidence is strong?
- does entropy drop for the right reason instead of just getting overconfident?

The standard calibration diagnostics are:

```tex
\mathrm{Brier}
=
\frac{1}{N}
\sum_{i=1}^N
\sum_{c \in \mathcal{C}}
\left(
p_i(c)-\mathbf{1}[y_i=c]
\right)^2
```

```tex
\mathrm{ECE}
=
\sum_{b=1}^B
\frac{|I_b|}{N}
\left|
\mathrm{acc}(I_b)-\mathrm{conf}(I_b)
\right|.
```

Posterior entropy is the complementary uncertainty view:

```tex
H_t = - \sum_{c \in \mathcal{C}} p_t(c)\log p_t(c).
```

Low entropy means the model is decisive. That is only good when the decisive
label is also correct. If entropy falls while the model is wrong, the posterior
is confidently wrong rather than calibrated.

The current calibration bins for the accumulator make this concrete:

- `0.9-1.0` confidence bin: 175 cases, accuracy `0.96`, mean confidence `0.9992854302781583`, gap `0.03928543027815834`
- `0.4-0.5` confidence bin: 7 cases, accuracy `0.0`, mean confidence `0.5`, gap `0.5`

That means the accumulator is usually strong and highly confident, but the
small ambiguous regime is not yet well calibrated. This is a useful warning:
the method is reliable on most runs, but uncertainty is not yet fully
informative when it appears.

The shared-corpus comparison tells the same story at the method level:

| method | overall accuracy | prior flip fraction | interpretation |
| --- | ---: | ---: | --- |
| pointwise | 0.875 | 0.479 | useful lower bound, but brittle to prior shifts |
| windowed_raw | 0.750 | 0.021 | stable under priors, but not separative enough |
| windowed_robust | 0.750 | 0.000 | very stable, but no accuracy gain over raw windowing |
| accumulator | 0.958 | 0.208 | best current overall method |
| kalman_bank | 0.740 | 0.208 | model-based, but weaker than the accumulator on this corpus |
| kalman_bank_velocity_aided | 0.844 | 0.125 | improved by an actual velocity stream |

The practical reading is:

- the accumulator is the strongest end-to-end evidence combiner we have here
- pointwise is a good sanity check, not the finish line
- windowed robustness buys stability, but it does not automatically buy better separation
- the velocity-aided Kalman bank is meaningful only when extra sensing is actually available

Prior sensitivity is the stability counterpart to calibration:

- pointwise flips often under prior perturbation
- robust windowed methods rarely flip
- the accumulator sits in between: strong enough to be useful, but still sensitive to ambiguous cases
- the Kalman bank is stable enough to be meaningful, but not yet the best on this corpus

These summaries come from the generated artifact bundles in:

- `artifacts/common_dataset_comparison_v1/common_dataset_comparison_report.md`
- `artifacts/prior_sensitivity_v1/prior_sensitivity_report.md`
- `artifacts/monte_carlo_accumulator/calibration_bins.csv`

So posterior quality in this repo is a bundle of metrics, not a single number:
accuracy, calibration, entropy, and prior robustness all matter.

## 4. PDF Terms Versus CDF Terms

The repo currently mixes two different probabilistic objects.

### 4.1 Density terms

These answer:

```tex
\text{How well does this exact observed value fit the class model?}
```

Examples:

- toy innovation likelihood
- toy latent center terms
- identity `speed_shape`
- identity `history_shape`

These use Gaussian PDF or log-PDF forms.

### 4.2 Region-probability terms

These answer:

```tex
\text{How much class probability mass lies in a valid region?}
```

Examples:

- toy `speed`
- toy `accel`
- identity `speed_validity`

These use Gaussian CDF or tail-probability forms.

That distinction is the clean answer to the user's question about "cutting" the
probability space:

- PDF terms do not cut the space; they score a point
- CDF terms do cut the space; they integrate over a region

## 5. Feature probabilities are not the class posterior

Both benchmark families also compute feature probabilities, but those are
derived diagnostic quantities, not the class posterior itself.

Examples:

- toy: `reverse_motion`, `hard_brake`, `oscillatory`, `near_envelope`
- identity: `bike_envelope`, `horse_envelope`, `persistent_push`, `surging_trace`

Conceptually:

```tex
p(f_k \mid z_{1:t})
```

is being estimated from the current state estimate or recent observed history.
Those feature probabilities are then thresholded or aggregated over time to
produce:

- detected feature sets
- feature confusion counts
- feature-vs-class matrices

They help explain why a class won or lost, but they are not themselves the
normalized class posterior.

## 5.1 Feature families and transfer principles

The repo is moving toward a feature taxonomy where features are understood by
role, not just by name:

- instantaneous features
- windowed or shape features
- cumulative-style features
- derivative features
- model-residual features

This matters for transfer. A 1D feature is not expected to survive to 3D by
naive copy-paste. It survives when its structural role survives:

- scalar speed features become magnitude or projection features
- scalar acceleration features become norms or frame-aware components
- scalar innovation features become Mahalanobis residual summaries

That is the beginning of a generic feature methodology rather than a bag of 1D
tricks.

## 5.2 From features to artifacts

The repo is increasingly treating features as study objects, not only classifier
inputs:

```tex
\text{feature extractor}
\rightarrow
\text{feature table}
\rightarrow
\text{evidence behavior}
\rightarrow
\text{confusion / overlap / PCA / adequacy artifacts}.
```

That is why feature information appears in several places:

- feature-vs-class confusion heatmaps
- feature precision / recall / lift summaries
- feature-set inspection bundles
- PCA scatter and loading plots
- corpus adequacy and excitation reports

This separation is important for future 3D work. The feature layer should be
swappable without rewriting the rest of the study machinery.

## 6. Why the confusion matrices matter

The confusion artifacts are downstream summaries of these posterior updates.

Class confusion matrix:

```tex
C_{a,b} = \#\{\text{runs with true class } a \text{ and predicted class } b\}.
```

Feature-vs-class matrices:

```tex
F^{\text{true}}_{k,b}
= \#\{\text{runs where true feature } f_k \text{ is present and class } b \text{ is predicted}\},
```

```tex
F^{\text{det}}_{k,b}
= \#\{\text{runs where detected feature } f_k \text{ is present and class } b \text{ is predicted}\}.
```

Posterior entropy by step:

```tex
H_t = - \sum_i p(s_i \mid z_{1:t}) \log p(s_i \mid z_{1:t}).
```

That entropy trace is useful because it shows whether the classifier is:

- becoming confidently correct
- becoming confidently wrong
- or staying ambiguous over time

## 7. Practical interpretation

The clean mental model for the repo is:

1. Each class proposes a probabilistic explanation for the new measurement.
2. That explanation is not just one Gaussian. It is a composite score.
3. Prior class mass is multiplied by that composite likelihood.
4. The result is normalized into the next posterior over classes.
5. Features, confusion matrices, and entropy are diagnostics of that update.

So the code is absolutely following the Bayesian pattern

```tex
p(s_i \mid z) \propto p(z \mid s_i) p(s_i),
```

but in both benchmark families, `p(z \mid s_i)` is implemented as a composite,
engineered likelihood rather than a single simple closed-form density.

## 8. What the current 1D work proves

The present 1D work supports a few concrete claims:

1. The repo can express both direct-evidence classifiers and latent-state
   filter-bank classifiers through the same recursive posterior story.
2. The artifact layer is rich enough to diagnose feature failure, phase
   failure, prior sensitivity, and confusion structure, not just top-line
   accuracy.
3. The toy and identity paths are complementary:
   - `toy_1d` stresses latent-state, mode-mixture, and phase reasoning
   - `identity_1d` stresses direct evidence and boundary calibration
4. The repo is already more than a 1D demo, but it is still using 1D as an
   exploratory proving ground rather than claiming a finished 3D-ready system.

## 8.1 Posterior walkthrough generation

The posterior walkthrough artifacts are built primarily through:

- `witnesses/toy_1d/posterior_explainer.py`
- `inference/identity_inference/posterior_explainer.py`
- `witnesses/toy_1d/bayesian_walkthroughs.py`

Those modules bridge recursive posterior math, implementation-level evidence
terms, and team-readable success and failure artifacts.

## 9. How to keep this documentation useful

The most useful future extensions of this guide should keep three things linked:

1. the mathematical object
2. the implementation surface that computes it
3. the artifact family that exposes its behavior

That pattern is what keeps the repo understandable as it grows from:

- 1D feature studies
- 1D posterior studies
- filtering comparisons

toward more generic methodology and later 3D-specific adapters.
## Part 2. Methodology Evaluation Framework


Source: [methodology_evaluation_framework.md](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/methodology_evaluation_framework.md)

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
## Part 3. Classifier Ladder and Contracts


Source: [classifier_ladder_and_contracts.md](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/classifier_ladder_and_contracts.md)

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
## Part 4. Corpus Generation and Search


Source: [corpus_generation_and_search.md](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/corpus_generation_and_search.md)

This note is the corpus-side pillar of the methodology stack. It is not just a
description of how trajectories are synthesized. It is meant to answer:

- what variables define a corpus candidate
- what objective function is being optimized
- how corpus quality is measured before classifiers are evaluated
- how adequacy pressure is turned into a scalar score and Pareto surface
- how corpus candidates become promoted studies
- which artifacts demonstrate those claims numerically

In the canonical repo story, this note is the `Corpus Explorer` pillar. It is
the part of the repo that turns an objective and backend into a selected corpus
that the Study Candidate Evaluator can trust.

## Scope and Relation to Other Documents

This document owns the corpus lifecycle:

- corpus objectives and candidate generation
- corpus adequacy and leakage evaluation
- archive-based exploration and selection
- CorpusGym-style execution and reward surfaces
- backend-aware planning for corpus search

It is intentionally narrower than the other two core documents:

- the methodology evaluation framework explains how to judge studies
- the classifier ladder explains how evidence providers generate posteriors
- this document explains how the corpus is produced, measured, and selected
  before those classifiers are asked to interpret it

## Corpus Explorer Contract

The generic corpus-side contract is:

```tex
(o, b, q, G_b) \mapsto D^\star,
```

where:

- `o` is the corpus objective
- `b` is the backend family
- `q` is the candidate proposal distribution
- `G_b` is the backend-specific generator
- `D^*` is the selected corpus after adequacy, leakage, and validity audits

In study terms, the explorer is upstream of the study candidate

```tex
s = (D^\star, f, C, m, \pi, b).
```

That is why this document is not merely about synthetic generation. It is about
corpus governance: which candidate corpora exist, which are rejected, which are
selected, and which study claims they can support.

## Corpus Evaluation Criteria

Before any study is promoted, the corpus itself must be judged on its own
merits. The corpus-side evaluation question is not whether the classifier is
already good; it is whether the generated corpus is broad, valid, auditable,
and hard in the intended way.

The most important corpus-level criteria are:

- class balance and class-pair balance
- boundary coverage and ambiguity pressure
- feature excitation over the active feature set
- difficulty diversity across tiers and regimes
- leakage control from duration, noise, sampling, or environment
- degeneracy control so the corpus does not collapse to trivial repeats
- provenance completeness so the selected corpus is reproducible

These checks happen before classifier conclusions are trusted. A strong
classifier result on a leaky or trivial corpus is not a strong study.

A useful corpus-evaluation summary vector is:

```tex
\mathbf{m}_k
=
\big[
    B_k,\,
    C_k,\,
    F_k,\,
    D_k,\,
    1-L_k,\,
    1-T_k,\,
    1-G_k,\,
    P_k
\big]
```

where `P_k` is provenance completeness. The corpus score in
`corpus_autodevelopment.py` is one concrete scalarization of that vector,
while the archive and selected-corpus artifacts preserve the non-scalarized
tradeoffs.

## 1. Problem Statement

The corpus layer exists to solve a methodological problem, not only a data
generation problem:

```tex
\text{How do we generate and select datasets that are informative enough to test
classification methods without making the task trivial or biased?}
```

That means corpus generation must be tied to explicit objectives rather than
only to “more synthetic trajectories.”

## 2. Global Notation

The main objects are:

- `tau`: a generated trajectory
- `D_k`: corpus candidate `k`
- `D^*`: selected corpus after audit and selection
- `theta_class`: class-specific motion parameters
- `theta_tier`: difficulty-tier controls
- `theta_noise`: corruption parameters
- `theta_sampling`: timing and sample-count parameters
- `S_k`: scalar score for corpus candidate `k`
- `o_k`: Pareto objective vector for candidate `k`

The front-door corpus story is:

```tex
\theta_k \sim q(\theta \mid o, b), \qquad
\tau_i \sim G_b(\theta_k, \xi_i), \qquad
D_k = \{\tau_i\}_{i=1}^{N_k}, \qquad
D^\star = \operatorname*{select}_k(D_k \mid S_k, o_k, \text{adequacy gates}).
```

The selected corpus is therefore not merely the most recently generated one. It
is the candidate that survives the declared score, Pareto, and gate logic.

## 3. Trajectory Parameterization and Witness Problems

### 3.1 Problem

`trajectory_generator.py` and `corpus_objectives.py` define the base witness
problems used by the rest of the repo.

### 3.2 Assumptions

The core assumptions are:

- class semantics can be represented by parameterized families of trajectories
- difficulty tiers can be represented by controlled changes in noise,
  irregularity, outliers, and step counts
- the induced synthetic geometry is meaningful enough to support feature,
  posterior, and adequacy studies

### 3.3 Implementation Mapping

- `trajectory_generator.py`
- `corpus_objectives.py`
- objective YAML in
  `experiments/corpus_objectives/common_1d_corpus_objectives.yaml`

### 3.4 Why This Matters

Every later artifact inherits the geometry defined here. If the synthetic class
definitions are weak or biased, no downstream classifier result is trustworthy.

## 4. Corpus-Shaping Layers

The repo has several modules that perturb or search the corpus distribution:

- `adaptive_stress_corpus.py`
- `environment_aware_corpus.py`
- `quality_diversity_corpus.py`
- `objective_driven_qd_archive.py`

At a high level, they move from one corpus distribution to another:

```tex
\mathcal{D}
\rightarrow
\mathcal{D}'(\text{noise}, \text{irregularity}, \text{outliers}, \text{stress}, \text{diversity}).
```

These are not separate data silos. They are search directions over corpus
properties that may reveal different classifier or feature failures.

## 5. Corpus Autodevelopment

### 5.1 Problem

`corpus_autodevelopment.py` asks:

```tex
\text{Can the repo score and select between multiple candidate corpora using declared adequacy goals?}
```

### 5.2 Score Construction

For candidate `k`, the implemented scalar score is:

```tex
S_k
= B_k + C_k + F_k + D_k - L_k - T_k - G_k,
```

where:

- `B_k`: class-balance score
- `C_k`: class-pair boundary coverage score
- `F_k`: feature-excitation score
- `D_k`: difficulty-diversity score
- `L_k`: leakage penalty
- `T_k`: triviality penalty
- `G_k`: degeneracy penalty

This is not merely conceptual. These terms are computed explicitly by:

- `_balance_score(...)`
- `_boundary_coverage_score(...)`
- `_feature_excitation_score(...)`
- `_difficulty_diversity_score(...)`
- `_leakage_penalty(...)`
- `_triviality_penalty(...)`
- `_degeneracy_penalty(...)`

### 5.3 Pareto Surface

The same module also defines a vector objective:

```tex
\mathbf{o}_k =
\big[
  B_k,\,
  C_k,\,
  F_k,\,
  D_k,\,
  -L_k,\,
  -T_k,\,
  -G_k
\big].
```

A candidate is dominated when another candidate is no worse in every coordinate
and strictly better in at least one. This is the actual meaning of the Pareto
front in the code.

### 5.4 Assumptions

The score assumes:

- the positive terms should be maximized
- the penalties should be minimized
- a single scalar score is useful for selection
- but the Pareto front should still be preserved so non-dominated tradeoffs are
  not erased

### 5.5 Worked Example

The numeric artifact
[corpus_autodevelopment_numeric_walkthrough.md](artifacts/corpus_autodevelopment_v1/corpus_autodevelopment_numeric_walkthrough.md)
is the current concrete proof for this section. It shows one real selected
candidate and:

- substitutes the actual score-term values into `S_k`
- expands the difficulty-diversity subscore against the configured target
  fractions
- shows leakage-threshold rows
- compares the selected candidate against the highest-scoring rejected one
- explains why selection and Pareto non-dominance are not the same claim

That artifact is the proper bridge from symbolic objective to implementation.

## 6. Corpus Search and Baseline Ranking

The broader search layer is implemented through:

- `corpus_search_baseline.py`
- `corpus_synthesis_comparison.py`
- `generic_corpus_exploration.py`
- `selected_generated_corpus.py`

The methodological statement here is:

```tex
\text{corpus search}
\neq
\text{generate more random trajectories}.
```

Instead, the repo is moving toward objective-driven exploration over corpus
properties that affect identifiability, calibration, leakage, and robustness.

### 6.1 Generic Corpus Explorer

`generic_corpus_exploration.py` is the clearest implementation of the repo’s
Corpus Explorer idea. It starts from a heterogeneous candidate pool of
backend-specific trajectory specifications and scores each executed run by a
normalized utility rather than by one raw classification metric.

For one executed run, the implemented explorer utility is:

```tex
U_{\text{explore}}
= 0.22 \cdot \text{validity}
+ 0.18 \cdot \text{coverage novelty}
+ 0.18 \cdot \text{boundary score}
+ 0.18 \cdot \text{classifier stress}
+ 0.12 \cdot \text{environment score}
+ 0.12 \cdot \text{provenance completeness}.
```

This score is intentionally mixed. It rewards:

- validity of the executed run
- novelty of the candidate’s coverage cell
- pressure on known class boundaries
- stress placed on current classifier families
- usefulness of environment-regime structure
- preservation of provenance metadata for later audit

The utility is easier to interpret if written as a short symbol map:

```tex
U_{\text{explore}} = 0.22 V + 0.18 N + 0.18 B + 0.18 S + 0.12 E + 0.12 P.
```

That is the form used by the numeric walkthrough artifact, which substitutes
the concrete values for `V`, `N`, `B`, `S`, `E`, and `P` before comparing the
selected row with a random baseline.

So the explorer is not only asking “which trajectory is hardest?” It is asking
“which executed trajectories make the corpus more useful as a study object?”

### 6.2 Explorer Archive and Selection Logic

The explorer also constructs archive cells over:

- backend
- scenario family
- target class
- difficulty tier

For each cell, the elite is the run with maximal `total_utility`. The selected
corpus is then compared against a same-size random baseline by coverage:

```tex
\Delta_{\text{coverage}}
=
\#\{\text{selected archive cells}\}
-
\#\{\text{random-baseline archive cells}\}.
```

If `h(tau)` is the archive-cell map for a trajectory `tau`, then the selected
elite is

```tex
A[h(\tau)]
\leftarrow
\arg\max_{\tau' : h(\tau') = h(\tau)} U_{\text{explore}}(\tau').
```

That gives the explorer a clear audit question: does the selected corpus cover
more useful behavioral cells than a naïve random sample of equal size?

### 6.3 Worked Example

The numeric artifact
[generic_corpus_explorer_numeric_walkthrough.md](artifacts/generic_corpus_exploration/generic_corpus_explorer_numeric_walkthrough.md)
now expands one selected corpus row into:

- its `U_explore` utility decomposition
- its archive-cell role
- its elite interpretation
- its contribution to the selected-versus-random coverage comparison

That artifact is the Explorer-side proof that corpus selection is not only a
visual dashboard. It is a numeric utility and coverage argument on one concrete
selected row.

## 7. Corpus Gym

### 7.1 Problem

`corpus_gym.py` reframes corpus generation as a targeted environment rather than
just a batch sampler. The question is:

```tex
\text{Can we specify desired failure pressure or feature geometry and reward matching trajectories?}
```

### 7.2 Variables

The main objects are:

- `target`: a desired class, class pair, feature cell, failure mode, or prior-sensitive regime
- `action`: a parameterized perturbation of the base tier
- `reward`: a structured utility decomposition
- `episode`: `(target, action, trajectory, diagnostics, reward)`

### 7.3 Reward Construction

The reward in `CorpusGymReward` is multi-term rather than monolithic:

- `class_validity`
- `feature_excitation`
- `coverage_gain`
- `boundary_closeness`
- `classifier_stress`
- `prior_sensitivity`
- `leakage_penalty`
- `physical_invalidity_penalty`

The actual implemented utility is:

```tex
U_{\text{gym}}
= 0.22 \cdot V
+ 0.14 \cdot E
+ 0.14 \cdot G
+ 0.14 \cdot B
+ 0.14 \cdot S
+ 0.12 \cdot P
- 0.10 \cdot L
- 0.14 \cdot I,
```

where:

- `V`: class validity
- `E`: feature excitation match
- `G`: coverage gain
- `B`: boundary closeness
- `S`: classifier-stress pressure
- `P`: prior-sensitivity pressure
- `L`: leakage penalty
- `I`: physical invalidity penalty

The current code computes the component terms explicitly through:

- `_class_validity_score(...)`
- `_feature_excitation_score(...)`
- `_coverage_gain_score(...)`
- `_boundary_closeness_score(...)`
- `_classifier_stress_score(...)`
- `_prior_sensitivity_score(...)`
- `_leakage_penalty(...)`
- `_physical_invalidity_penalty(...)`

So the Gym is already a concrete objective function, not just an idea for one.

### 7.4 Coverage, Leakage, and Invalidity Terms

Three subterms are especially important because they prevent the Gym from
rewarding obviously biased or pathological cases.

The coverage-gain term is:

```tex
G
= 0.40 \cdot \text{tier match}
+ 0.30 \cdot \text{class match}
+ 0.30 \cdot \text{novelty}(a),
```

where the novelty term is the average of capped measurement, irregularity,
outlier, and step scales induced by the action.

The leakage penalty is:

```tex
L
= 0.30 \cdot \text{duration risk}
+ 0.30 \cdot \text{sample-count risk}
+ 0.40 \cdot \text{noise risk}.
```

This is how the Gym avoids rewarding trajectories merely because duration,
sample count, or noise level become class-identifying shortcuts.

The physical-invalidity penalty guards against:

- non-increasing time grids
- implausibly large absolute accelerations

So the Gym can search aggressively without silently drifting into invalid
trajectory geometry.

### 7.5 Environment Contract

`CorpusGymEnvironment` makes the corpus-search loop explicit. It provides:

- `reset(target)`
- `simulate(action)`
- `step(action)`
- `trajectory()`
- `score(trajectory)`
- `render_diagnostics(trajectory)`

Methodologically, the Gym can therefore be written as:

```tex
\text{target}
\xrightarrow{\text{reset}}
\text{state}
\xrightarrow{\text{action}}
\text{trajectory, reward, diagnostics}.
```

That matters because it makes targeted corpus search a reusable interface rather
than one hard-coded artifact generator.

### 7.6 Objective-to-Gym Execution Bridge

`objective_corpus_gym_runner.py` ties the declarative corpus-objective layer to
the Gym layer. It defines the maps:

```tex
\Psi_{\text{target}}(\text{objective}) \rightarrow \text{CorpusGymTarget},
\qquad
\Psi_{\text{action}}(\text{candidate}) \rightarrow \text{CorpusGymAction}.
```

Then the actual execution chain becomes:

```tex
\text{objective}
\rightarrow
\text{candidate}
\rightarrow
\text{target, action}
\rightarrow
\text{episode}
\rightarrow
\text{validated trajectory run}.
```

This is the critical bridge from declarative study design to executed explorer
or gym records. Without it, the Gym would be an isolated search environment.

### 7.7 Why This Matters

This turns “find me a hard example” into a measurable optimization target. The
corpus gym is therefore the bridge from declarative corpus goals to targeted
trajectory proposals.

### 7.8 Worked Example

The numeric artifact
[corpus_gym_numeric_walkthrough.md](artifacts/corpus_gym/corpus_gym_numeric_walkthrough.md)
now works through one real Gym episode. It shows:

- the selected target
- the actual action scales
- the resulting trajectory diagnostics
- the implemented reward equation
- the numeric substitution for each reward component

That artifact is the Gym-side proof that the search reward is not just
qualitative prose. It is a concrete score decomposition on one executed
trajectory.

## 8. Objective-Driven Quality-Diversity Archive

### 8.1 Problem

`objective_driven_qd_archive.py` asks a different question from the scalar
autodevelopment score:

```tex
\text{How do we preserve diverse elites across multiple difficulty and evidence cells instead of selecting only one best corpus?}
```

### 8.2 Archive Utility

For one successful archive cell, the current elite utility is assembled from:

```tex
U_{\text{archive}}
= 0.30 \cdot \text{validity}
+ 0.25 \cdot \text{acceleration-range pressure}
+ 0.25 \cdot \text{classifier stress}
+ 0.20 \cdot (1 - \text{mean margin}).
```

The cell definition itself also depends on discretized buckets over:

- assigned class
- difficulty tier
- backend
- duration
- acceleration range
- entropy
- prior-flip threshold

So the archive is not only ranking trajectories. It is preserving structured
coverage over behaviorally distinct cells.

### 8.3 Implementation Mapping

- `objective_driven_qd_archive.py`
- `generated_corpus_features.py`
- `corpus_classifier_scoring.py`

### 8.4 Methodological Use

This layer matters because it separates:

- “best current elite in a cell”
- “which cells have been covered at all”
- “which mutation lineages produce useful diversity”

That is a stronger corpus-search story than one scalar winner alone.

The implementation also makes the archive utility operational:

```tex
U_{\text{archive}}
=
0.30 \cdot \text{validity score}
+ 0.25 \cdot \min\!\left(\frac{\text{accel range}}{0.40}, 1\right)
+ 0.25 \cdot \text{max classifier stress}
+ 0.20 \cdot (1 - \text{mean posterior margin}).
```

So the archive preserves high-validity, high-stress, low-margin witnesses in
distinct cells rather than preserving diversity in the abstract.

Successful and failed archive cells are tracked separately. If `tau_t` is the
trajectory processed at iteration `t`, then

```tex
A_t^{\text{succ}}[h(\tau_t)]
\leftarrow
\arg\max_{\tau' : h(\tau') = h(\tau_t)}
U_{\text{archive}}(\tau')
```

only when the run succeeds and the label status is `valid_target_class`;
otherwise the failed-cell counter is updated in `A_t^{fail}`. The emitted
coverage curves are

```tex
C_t^{\text{succ}}
=
\frac{|A_t^{\text{succ}}|}{|A_T^{\text{succ}}|},
\qquad
C_t^{\text{fail}}
=
\frac{|A_t^{\text{fail}}|}{|A_T^{\text{fail}}|}.
```

That is why invalid or failed runs do not inflate successful coverage.

### 8.5 Quality-Diversity Corpus Layer

`quality_diversity_corpus.py` implements a lighter-weight archive on top of
CorpusGym episodes. Its cell key is

```tex
h_{\text{qd}}(\tau)
=
\big(
    c(\tau),
    \mathrm{tier}(\tau),
    b_{\mathrm{dur}}(\tau),
    b_{\mathrm{acc}}(\tau),
    b_{\mathrm{turn}}(\tau)
\big),
```

where the last three coordinates are duration, acceleration-range, and
direction-change buckets. The elite replacement rule is the same `argmax` rule
as above, but the utility is the episode reward `U_gym` rather than
`U_archive`. The current coverage fraction is

```tex
\mathrm{coverage\_fraction}_t
=
\frac{\#\{\text{filled QD cells at iteration } t\}}{81},
```

because the current 1D archive discretizes `3 x 3 x 3 x 3` regime
combinations across the non-class axes.

## 8A. Corpus Hyperparameter Policy And Tuning

The corpus-search layer now has an explicit hyperparameter surface in
`corpus_policy.py` and `corpus_policy_sweep.py`. A policy is

```tex
p
=
\big(
    w^{+},\,
    w^{-},\,
    w^{\text{explore}},\,
    w^{\text{gym}},\,
    w^{\text{archive}},\,
    n,\,
    g
\big),
```

where `w+` are positive corpus weights, `w-` penalty weights, `w^explore`
generic-explorer weights, `w^gym` CorpusGym weights, `w^archive` archive
weights, `n` sampler budgets, and `g` adequacy gates.

Whenever a weight group is normalized, the implementation applies

```tex
\bar{w}_r
=
\frac{w_r}{\sum_u w_u},
```

so the scalar objectives remain comparable under reweighting. This is the role
of `normalize_corpus_policy_spec()`.

For policy `p`, the generic-explorer utility becomes

```tex
U_{\text{explore}}^{(p)}
=
\sum_{r \in \mathcal{R}_{\text{explore}}}
\bar{w}^{(p)}_r u_r,
```

the archive utility becomes

```tex
U_{\text{archive}}^{(p)}
=
\sum_{r \in \mathcal{R}_{\text{archive}}}
\bar{w}^{(p)}_r a_r,
```

and the sampler mixture becomes

```tex
\pi_s^{(p)}
=
\frac{n_s^{(p)}}{\sum_{s'} n_{s'}^{(p)}}.
```

So a policy changes both how candidates are scored and how search effort is
allocated.

The current tuning sweep evaluates a policy by a downstream adequacy proxy:

```tex
A_{\text{policy}}(p)
=
0.25 \cdot \text{validity}
+ 0.20 \cdot \text{boundary coverage}
+ 0.20 \cdot \min\!\left(\frac{\text{feature excitation}}{1.5}, 1\right)
+ 0.15 \cdot \text{classifier stress}
+ 0.20 \cdot \text{provenance completeness}
- 0.20 \cdot \text{leakage},
```

followed by the bounded policy score

```tex
J_{\text{policy}}(p)
=
\operatorname{clip}
\big(
    A_{\text{policy}}(p) + 0.10 \cdot \text{classifier stress},
    0,
    1
\big).
```

This is the quantity emitted as `policy_score` in the sweep results.

There is now a numeric walkthrough artifact for one real recommended policy
row:

- [corpus_policy_numeric_walkthrough.md](artifacts/corpus_hyperparameter_tuning_v1/corpus_policy_numeric_walkthrough.md)

That artifact expands the adequacy proxy, stress bonus, selected-set Jaccard,
rank stability, and dev-vs-holdout comparison numerically for the recommended
policy.

The sweep also checks whether a better score is only a re-ranking accident. For
selected sets `S(p_a)` and `S(p_b)`, the stability metric is

```tex
J_{\text{set}}(p_a, p_b)
=
\frac{|S(p_a) \cap S(p_b)|}{|S(p_a) \cup S(p_b)|}.
```

Rank stability is then reported through Spearman and Kendall correlations of
the ranked candidate lists. The policy question is therefore not only “which
weights maximize one scalar?” but also “which weights preserve a stable,
scientifically sensible selected set?”

## 9. Study Candidate Generation

### 9.1 Problem

Once corpus candidates exist, the next question is not “which dataset is best
in the abstract?” but “which study should the team run next?”

The study-candidate layer is implemented across:

- `candidate_generation.py`
- `capability_aware_search.py`
- `study_candidate_generation.py`
- `study_candidate_protocol.py`

### 9.2 Variables

For a candidate study built from:

- class pair `p`
- classifier `m`
- feature set `f`
- prior regime `r`

the current study-selection logic defines a static screening score
`Q_static(p,m,f,r)` before any Monte Carlo acceptance score is considered:

```tex
Q_{\text{static}}(p,m,f,r)
= 0.18 \cdot \text{feature-class compatibility}
+ 0.18 \cdot \text{expected separability}
+ 0.14 \cdot \text{classifier fit}
+ 0.14 \cdot \text{corpus coverage}
+ 0.12 \cdot \text{dimensional transfer}
+ 0.12 \cdot \text{implementation readiness}
+ 0.12 \cdot (1 - \text{dependency risk})
- 0.10 \cdot \text{cumulative-history risk}
- 0.10 \cdot \text{prior-sensitivity risk}.
```

In the current code, that high-level score is instantiated through named
subterms such as:

- `feature_class_compatibility_score`
- `expected_separability_score`
- `classifier_assumption_fit`
- `corpus_coverage_score`
- `dimensional_transfer_score`
- `implementation_readiness_score`
- `feature_dependency_risk`
- `cumulative_double_counting_risk`
- `prior_sensitivity_risk`

The important point is that these are not placeholders. They are the actual
named columns written into the `static_candidate_scores.csv` artifact by
`study_candidate_generation.py`.

### 9.3 Assumptions

The current static score assumes:

- feature/class compatibility and oracle separability are the strongest early
  evidence for whether a study is worth running
- corpus coverage and classifier-family fit should matter, but not dominate
- 3D transferability and implementation readiness should influence prioritization
  before the repo is fully 3D-ready
- dependency growth, cumulative-history reuse, and prior fragility are real
  methodological risks and should subtract from the screening score

It also assumes a two-stage process:

```tex
\text{static screening} \rightarrow \text{Monte Carlo confirmation} \rightarrow \text{promotion decision}.
```

That is a stronger claim than “rank everything once.” It means the repo is
already distinguishing between proposal quality before execution and evidence
quality after execution.

### 9.4 Monte Carlo Confirmation Layer

After static screening, the module uses cross-method metrics from
`analyze_common_experiment(...)` to compute a second score:

```tex
Q_{\text{mc}}
= 0.60 \cdot \text{accuracy}
+ 0.25 \cdot (1 - \text{prior flip fraction})
+ 0.15 \cdot (1 - \max(0, \text{oracle gap})).
```

This is the acceptance surface that the current code actually uses to decide
whether a study is promoted, revised, or rejected once benchmark evidence
exists.

The current promote gate is approximately:

```tex
\text{compatible}
\land
Q_{\text{static}} \ge 0.45
\land
Q_{\text{mc}} \ge 0.90
\land
\text{accuracy} \ge 0.83
\land
\text{prior flip fraction} \le 0.12.
```

If the feature set is compatible and the accuracy is merely decent, the
decision is usually `revise`; otherwise the decision falls to `reject` or
`defer`. This is the point where the repo stops being a proposal generator and
starts acting like an evidence-gated study selector.

### 9.5 Implementation Mapping

- `candidate_generation.py`
  - sampler families: random, grid, LHS, boundary mutation, archive mutation,
    stress mutation
- `capability_aware_search.py`
  - backend-aware search-method planning and runtime-budget logic
- `study_candidate_generation.py`
  - static score assembly, Monte Carlo lookup, and decision bucketing
- `study_candidate_protocol.py`
  - schema and validation-ladder contract for promoted studies

The lower-level sampler lives in `candidate_generation.py`; the promotion logic
lives in the higher-level study-candidate modules.

The current generated tables for this layer are not generic placeholders. They
include:

- `generated_study_candidates.json`
- `static_candidate_scores.csv`
- `monte_carlo_candidate_scores.csv`
- `promoted_candidates.csv`
- `rejected_candidates.csv`

That means the score decomposition and the decision vocabulary can already be
audited without reading source code.

### 9.6 Why This Matters

This is the point where corpus logic, feature logic, classifier logic, and
readiness logic start to interact. That is why this layer belongs in the
methodology docs, not only in artifact indexes.

## 10. Candidate Generation as a Sampler Family

`candidate_generation.py` does not emit one candidate per objective. For an
objective `o` and backend `b`, it effectively builds a candidate population

```tex
\mathcal{C}(o,b)
=
\mathcal{C}_{\text{random}}
\cup
\mathcal{C}_{\text{grid}}
\cup
\mathcal{C}_{\text{lhs}}
\cup
\mathcal{C}_{\text{boundary mutation}}
\cup
\mathcal{C}_{\text{archive mutation}}
\cup
\mathcal{C}_{\text{stress mutation}}.
```

That means candidate generation is already a search policy:

- `random` provides broad stochastic perturbation
- `grid` provides deterministic baseline coverage
- `lhs` provides space-filling parameter coverage
- `boundary_mutation` pushes toward harder near-boundary proposals
- `archive_mutation` performs local search around previously promising cells
- `stress_mutation` deliberately increases corruption or compression pressure

In probabilistic language, the module is not using one proposal distribution. It
is using a mixture over search heuristics:

```tex
q(c \mid o,b)
= \sum_s \pi_s \, q_s(c \mid o,b),
```

where `s` ranges over sampler families and the mixture weights are implemented
implicitly through per-family candidate budgets rather than estimated online.

This matters because the search surface is already hybrid:

- broad coverage samplers explore
- mutation samplers exploit
- stress samplers deliberately push toward failure regimes

## 11. Capability-Aware Search Planning

`capability_aware_search.py` determines which search methods should be used for
which backend family. The planner conditions on backend capability attributes
such as:

- runtime class
- dimensionality
- environment support
- sequential-control support
- stochastic versus deterministic execution

At the methodology level, that planner is a map

```tex
\Pi(\text{backend capabilities})
\rightarrow
\{\text{recommended search methods}, \text{budget class}, \text{planner rationale}\}.
```

This is important because the repo is no longer assuming that all backends
should be searched the same way.

The implemented rule set is not abstract. It contains explicit branches for:

- runtime class: `cheap`, `medium`, `expensive`
- environment support
- sequential-control support
- stochastic versus deterministic execution

Operationally, the current rule families behave like

```tex
M_{\text{rt}}(\kappa)
=
\begin{cases}
    \{\text{random},\text{lhs},\text{sobol},\text{qd}\}, & \kappa_{\text{runtime}}=\text{cheap}, \\
    \{\text{lhs},\text{sobol},\text{qd}\}, & \kappa_{\text{runtime}}=\text{medium}, \\
    \{\text{small DOE},\text{surrogate},\text{active learning}\}, & \kappa_{\text{runtime}}=\text{expensive},
\end{cases}
```

and

```tex
M_{\text{ctl}}(\kappa)
=
\begin{cases}
    \{\text{adaptive stress},\text{cross entropy}\}, & \kappa_{\text{seq}}=1, \\
    \varnothing, & \kappa_{\text{seq}}=0.
\end{cases}
```

with analogous environment- and stochasticity-dependent additions. So search
planning is already encoded as a capability map, not a loose recommendation
paragraph.

So the actual planner is closer to:

```tex
\Pi(\kappa)
=
\big(
M_{\text{runtime}}(\kappa),
M_{\text{environment}}(\kappa),
M_{\text{control}}(\kappa),
M_{\text{stochastic}}(\kappa)
\big),
```

where `kappa` is the backend capability vector and the resulting method set is
deduplicated into one backend plan row. Cheap stochastic backends receive broad
search budgets; expensive deterministic backends are pushed toward smaller DOE,
surrogate assistance, and cache-priority execution.

## 12. Study Candidate Protocol and Validation Ladder

`study_candidate_protocol.py` defines the contract that the generated-candidate
layer must satisfy. It specifies:

- the `StudyCandidate` schema
- the `ValidationLadder` schema
- the required terminal decision vocabulary: `promote`, `revise`, `reject`, `defer`

The promotion story is therefore not just:

```tex
\text{good score} \Rightarrow \text{promote}.
```

It is:

```tex
\text{candidate specification}
\rightarrow
\text{validation ladder evidence}
\rightarrow
\text{terminal decision in a constrained vocabulary}.
```

That is what makes the study layer auditable rather than ad hoc.

The protocol is also stronger than a single schema file. It defines two linked
objects:

- `StudyCandidate`
- `ValidationLadder`

The validation ladder itself has ordered levels, including:

- static compatibility
- corpus adequacy
- feature separability
- oracle separability
- classifier performance
- posterior and calibration quality
- prior sensitivity
- stress and adversarial robustness
- dimensional transfer assessment
- promotion decision

So the real promotion contract is:

```tex
\text{proposal}
\rightarrow
\{\ell_1, \ell_2, \dots, \ell_{10}\}
\rightarrow
d,
\qquad
d \in \{\text{promote}, \text{revise}, \text{reject}, \text{defer}\}.
```

This matters because it prevents a high static score from bypassing evidence,
and it prevents a visually interesting candidate from being promoted without an
explicit decision trail.

## 13. Generated Class/Feature Exploration

### 13.1 Generated Corpus Features

`generated_corpus_features.py` routes objective-driven generated trajectories
back through the real feature pipeline. This is important because generated
candidates are not only evaluated by proxy score columns. They are turned into
real `TrajectoryArtifact` rows, relabeled through class-validity logic, grouped
into tier datasets, and fed to `analyze_feature_datasets(...)`.

Methodologically, the feature pipeline therefore becomes:

```tex
\text{generated candidate}
\rightarrow
\text{executed trajectory}
\rightarrow
\text{validity-adjusted label}
\rightarrow
\text{feature row}
\rightarrow
\text{excitation and separability analysis}.
```

### 13.2 Corpus-Conditioned Classifier Scoring

`corpus_classifier_scoring.py` then asks how the current classifier ladder
behaves on those generated and relabeled trajectories. It rebuilds pointwise,
accumulator, windowed, and Kalman-family scoring surfaces and tracks:

- posterior entropy
- top-two posterior margin
- confident errors
- time to confidence
- measured classifier stress
- prior-flip sensitivity

The classifier-stress proxy can be read as:

```tex
\text{stress}
\approx
0.5 \cdot (1 - \text{margin})
+ 0.5 \cdot \text{entropy}
+ 0.35 \cdot \mathbf{1}\{\text{final prediction wrong}\}.
```

That makes this layer the bridge from corpus search to actual classifier
pressure. A generated corpus candidate is valuable only if it changes the
downstream evidence and decision landscape in an interpretable way.

## 14. End-to-End Corpus Methodology Flow

The current intended flow is:

```tex
\text{trajectory generator}
\rightarrow
\text{corpus candidate}
\rightarrow
\text{adequacy-scored corpus}
\rightarrow
\text{backend-aware candidate population}
\rightarrow
\text{study candidate}
\rightarrow
\text{validation ladder}
\rightarrow
\text{promotion / revise / reject / defer}.
```

This is the most important interpretation point in the file: synthetic witness
problems are not the final product. They are inputs to a reusable study-design
loop.

## 15. Failure Modes

The corpus-side methodology can still fail in several ways:

- the synthetic class families may be too stylized
- the difficulty tiers may not match the claimed hard boundaries
- the scalar score may hide a meaningful Pareto tradeoff
- study promotion may overvalue convenience and undervalue scientific pressure

That is why the worked example, the adequacy audit, and the search artifacts
must be read together.

## 16. What This Document Proves

This note is complete only if it supports the following claims:

- corpus generation is tied to explicit variables and objectives
- corpus autodevelopment has a formal score and Pareto definition
- the score terms have code-level implementations
- at least one real candidate score has been decomposed numerically
- study promotion is treated as a methodology problem, not only as a report
  listing
## Part 5. Dimensional Lift and Advanced Filter Gates


Source: [dimensional_lift_and_advanced_filter_gates.md](/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/dimensional_lift_and_advanced_filter_gates.md)

This note documents two related proof obligations in the repo:

- can the current methodology survive a lift from scalar 1D to vector-valued
  trajectories?
- when, exactly, would IMM, particle filtering, or RBPF become justified?

The goal is to turn those questions into explicit contracts and evidence gates,
not informal roadmap phrases.

## 1. Problem Statement

The central questions are:

```tex
\text{Is the current framework dimension-agnostic enough to survive a 3D lift?}
```

and

```tex
\text{What measured failure evidence would justify more advanced filter backends?}
```

Both questions are architectural. Neither should be answered only by intuition.

## 2. Global Notation

The main objects in this document are:

- `a_i`: one scalar-assumption audit row
- `z_t in R^d`: vector observation at time `t`
- `x_t = (u_t, v_t)`: split latent state for RBPF reasoning
- `Delta`: performance gain between two ladder rungs

## 3. Dimensional Lift Audit

### 3.1 Problem

`dimensional_lift_audit.py` and `pca_dimensionality_audit.py` ask whether the
current code assumes scalar 1D structure in ways that would block 3D reuse.

### 3.2 Status Classes

Each module is classified as one of:

- `dimension-agnostic`
- `adapter-compatible`
- `rewrite-required`

Formally, the audit is assigning

```tex
\text{module status}
\in
\{\text{agnostic}, \text{adapter-compatible}, \text{rewrite-required}\}.
```

### 3.3 Scalar-Assumption Inventory

Each recorded assumption row is effectively:

```tex
a_i =
(
  \text{module},
  \text{assumption id},
  \text{severity},
  \text{blocking-for-3D},
  \text{current assumption},
  \text{3D requirement}
).
```

The point is not merely to say “this file is 1D.” The point is to say **why**
it is 1D and what adapter or rewrite would remove that limitation.

### 3.4 Contract Smoke Test

The fake vector corpus in `dimensional_lift_audit.py` is a schema test, not a
full 3D physics benchmark. It constructs vector-compatible features such as:

```tex
\text{path length} = \sum_{t=2}^{T} \lVert z_t - z_{t-1} \rVert_2
```

and

```tex
\text{displacement norm} = \lVert z_T - z_1 \rVert_2.
```

The methodological claim is modest but important: the shared artifact surfaces
already tolerate vector-valued trajectories if the feature layer and adapters do
their job.

### 3.5 Implementation Mapping

- `dimensional_lift_audit.py`
- `pca_dimensionality_audit.py`

## 4. Advanced Filter Decision Gates

### 4.1 Problem

The advanced-filter decision layer asks whether the current ladder has failed in
a way that actually justifies more complex filtering.

Primary surfaces:

- `advanced_filter_decision.py`
- `rl_backend_decision.py`
- advanced-gate portions of `generic_filtering_contract.py`

### 4.2 Methodological Principle

The repo is currently enforcing:

- use the simplest method that explains the failure case
- do not add IMM, PF, or RBPF before the existing ladder demonstrably fails

That principle is only credible if the failure evidence is measured.

## 5. IMM As An Actual Algorithm Family

The repo now has an explicit IMM proof surface in
`inference/advanced_state_inference.py`. That means IMM should no longer be documented
only as a gate label. It should be documented as a switching linear-Gaussian
algorithm whose justification is still evidence-gated.

For mode set `M = {1, ..., M}`, let `mu_{k-1}^{(i)}` be the mode posterior
after step `k-1`, and let `Pi_{ij}` be the transition probability from mode `i`
to mode `j`. The predicted mode prior is:

```tex
\bar{\mu}_{k}^{(j)}
=
\sum_{i \in \mathcal{M}}
\Pi_{ij}\mu_{k-1}^{(i)}.
```

The mode-mixing weights are then:

```tex
\omega_{ij}^{(k)}
=
\frac{\Pi_{ij}\mu_{k-1}^{(i)}}{\bar{\mu}_{k}^{(j)}}.
```

These are the exact quantities emitted by the current implementation as the
predicted mode prior and `mode_mixing` rows.

If each mode `j` carries state estimate `x_{k-1|k-1}^{(j)}` and covariance
`P_{k-1|k-1}^{(j)}`, the mixed initial condition for target mode `j` is:

```tex
\bar{x}_{k-1|k-1}^{(j)}
=
\sum_{i \in \mathcal{M}}
\omega_{ij}^{(k)} x_{k-1|k-1}^{(i)},
```

```tex
\bar{P}_{k-1|k-1}^{(j)}
=
\sum_{i \in \mathcal{M}}
\omega_{ij}^{(k)}
\Big(
P_{k-1|k-1}^{(i)}
+
\big(x_{k-1|k-1}^{(i)}-\bar{x}_{k-1|k-1}^{(j)}\big)
\big(x_{k-1|k-1}^{(i)}-\bar{x}_{k-1|k-1}^{(j)}\big)^\top
\Big).
```

This is the mathematical distinction between IMM and the simpler
transition-matrix accumulator: IMM mixes full latent-state distributions, not
only class probabilities.

Each mode then executes its own predict/update step:

```tex
\hat{x}_{k|k-1}^{(j)}
=
F_k^{(j)}\bar{x}_{k-1|k-1}^{(j)} + b_k^{(j)},
```

```tex
\hat{P}_{k|k-1}^{(j)}
=
F_k^{(j)}\bar{P}_{k-1|k-1}^{(j)}F_k^{(j)\top} + Q_k^{(j)},
```

```tex
\nu_k^{(j)}
=
z_k - H_k^{(j)}\hat{x}_{k|k-1}^{(j)},
\qquad
S_k^{(j)}
=
H_k^{(j)}\hat{P}_{k|k-1}^{(j)}H_k^{(j)\top} + R_k^{(j)}.
```

The mode likelihood is:

```tex
\Lambda_k^{(j)}
=
\mathcal{N}\!\big(\nu_k^{(j)}; 0, S_k^{(j)}\big),
```

or in log form:

```tex
\ell_k^{(j)}
=
\log \bar{\mu}_k^{(j)} + \log \Lambda_k^{(j)}.
```

The updated mode posterior is therefore:

```tex
\mu_k^{(j)}
=
\frac{\Lambda_k^{(j)}\bar{\mu}_k^{(j)}}{\sum_{m \in \mathcal{M}} \Lambda_k^{(m)}\bar{\mu}_k^{(m)}}.
```

The final combined state estimate is:

```tex
x_{k|k}
=
\sum_{j \in \mathcal{M}} \mu_k^{(j)} x_{k|k}^{(j)},
```

```tex
P_{k|k}
=
\sum_{j \in \mathcal{M}} \mu_k^{(j)}
\Big(
P_{k|k}^{(j)}
+
\big(x_{k|k}^{(j)}-x_{k|k}\big)\big(x_{k|k}^{(j)}-x_{k|k}\big)^\top
\Big).
```

The current implementation also aggregates mode posteriors into class
posteriors by log-sum-exp over modes belonging to the same class, so the IMM
proof still fits the shared posterior/evidence contract used elsewhere in the
repo.

## PF As A Nonlinear / Non-Gaussian Algorithm Family

Particle filtering is the next rung only when the evidence shows that a
Gaussian mode model is the wrong approximation. In that case the state is
represented by a weighted sample set

```tex
\{x_k^{(i)}, w_k^{(i)}\}_{i=1}^{N_p},
\qquad
\sum_{i=1}^{N_p} w_k^{(i)} = 1.
```

For a bootstrap particle filter, the proposal is the state dynamics itself:

```tex
x_k^{(i)} \sim p(x_k \mid x_{k-1}^{(i)}).
```

The general sequential-importance proposal is:

```tex
x_k^{(i)} \sim q(x_k \mid x_{0:k-1}^{(i)}, z_{1:k}),
```

which yields the unnormalized importance weight:

```tex
\tilde{w}_k^{(i)}
=
w_{k-1}^{(i)}
\frac{
    p(z_k \mid x_k^{(i)})\,p(x_k^{(i)} \mid x_{k-1}^{(i)})
}{
    q(x_k^{(i)} \mid x_{0:k-1}^{(i)}, z_{1:k})
}.
```

This is the core PF identity missing from the simpler ladder: particles are not
weighted only by likelihood, but by likelihood corrected by the proposal-to-
prior ratio.

For the bootstrap choice above, the proposal ratio cancels, so the update
reduces to

```tex
\tilde{w}_k^{(i)}
=
w_{k-1}^{(i)}\,p(z_k \mid x_k^{(i)}),
```

with normalization

```tex
w_k^{(i)}
=
\frac{\tilde{w}_k^{(i)}}{\sum_{j=1}^{N_p}\tilde{w}_k^{(j)}}.
```

The state estimate is the weighted mean

```tex
\hat{x}_k
=
\sum_{i=1}^{N_p} w_k^{(i)} x_k^{(i)},
```

and the particle-set effective sample size is

```tex
N_{\text{eff}}
=
\frac{1}{\sum_{i=1}^{N_p} (w_k^{(i)})^2}.
```

The code gates resampling when `N_eff` falls below a threshold fraction of
`N_p`, because particle degeneracy is the failure mode that makes PF
numerically unusable even when the model is conceptually right.

If `N_eff < tau_resample N_p`, the intended resampling step is:

```tex
\{x_k^{(i)}, w_k^{(i)}\}_{i=1}^{N_p}
\mapsto
\{\tilde{x}_k^{(i)}, 1/N_p\}_{i=1}^{N_p},
```

where `tilde{x}_k^{(i)}` are resampled particles drawn with probabilities
proportional to `w_k^{(i)}`. Resampling is not part of the Bayesian model
itself; it is the numerical stabilization step that prevents one particle from
carrying almost all posterior mass.

To fit the repo's shared evidence contract, class evidence must still be
extracted from the particle cloud. The intended class-conditioned predictive
evidence is:

```tex
\hat{p}(z_k \mid c)
\approx
\sum_{i=1}^{N_p}
w_{k-1}^{(i,c)} p(z_k \mid x_k^{(i,c)}, c),
```

where the superscript `(i,c)` denotes particles propagated under class or mode
hypothesis `c`. The posterior update then reuses the same shared form:

```tex
p_k(c)
\propto
\hat{p}(z_k \mid c)\,p_{k-1}(c).
```

That is the key connection to the rest of the repo: PF is allowed only if it
can still emit evidence rows that the shared posterior and evaluation layers
know how to consume.

PF becomes evidence-gated when the current ladder cannot explain a witness
with strongly nonlinear dynamics, multimodal posteriors, non-Gaussian
observation noise, or censoring/clipping/saturation that breaks innovation
Gaussianity.

## PF Gate-To-Algorithm Example

The current gate artifact already provides the compact numeric reason PF is
still deferred. The current evidence says:

```tex
\text{mean normalized gap} = 0.438\sigma,
\qquad
\text{final normalized gap} = 1.125\sigma,
```

```tex
\Delta_{\text{short\_noisy}}^{\text{vel-aided}} = 0.188,
\qquad
A_{\text{outlier}}^{\text{best Kalman}} = 0.812.
```

These numbers should be read against the PF justification logic:

- a direct-velocity gain of `0.188` says the hardest short-horizon case is
  still materially sensing-limited
- best-Kalman outlier accuracy of `0.812` says robust linear-Gaussian variants
  still recover substantial performance without particles
- no dedicated nonlinear or non-Gaussian witness yet exists, so the required
  failure mode for PF is still missing

So the gate does not currently say “PF would fail.” It says the repo has not
yet earned the right to claim PF is the simplest necessary next rung.

## PF Numeric Walkthrough

Take a bootstrap PF with three particles at step `k`. Assume the prior weights
are uniform:

```tex
w_{k-1}^{(1)} = w_{k-1}^{(2)} = w_{k-1}^{(3)} = \frac{1}{3},
```

and the observation likelihoods at the new measurement are:

```tex
p(z_k \mid x_k^{(1)}) = 0.50,\qquad
p(z_k \mid x_k^{(2)}) = 0.30,\qquad
p(z_k \mid x_k^{(3)}) = 0.20.
```

The unnormalized weights are therefore:

```tex
\tilde{w}_k^{(1)} = 0.1667,\quad
\tilde{w}_k^{(2)} = 0.1000,\quad
\tilde{w}_k^{(3)} = 0.0667,
```

with total mass `0.3334`. After normalization:

```tex
w_k^{(1)} = 0.50,\qquad
w_k^{(2)} = 0.30,\qquad
w_k^{(3)} = 0.20.
```

If the particle states are:

```tex
x_k^{(1)} = 0.90,\qquad
x_k^{(2)} = 1.10,\qquad
x_k^{(3)} = 1.50,
```

then the weighted state estimate is:

```tex
\hat{x}_k
=
0.50(0.90) + 0.30(1.10) + 0.20(1.50)
=
1.08.
```

The effective sample size is:

```tex
N_{\text{eff}}
=
\frac{1}{0.50^2 + 0.30^2 + 0.20^2}
\approx 2.63.
```

Since `2.63` is above the usual resampling threshold of `0.5 N_p = 1.5`, this
update would not resample. The point of the walkthrough is not that these
numbers are special; it is that the PF rung now has a fully numeric posterior
update, state estimate, and degeneracy diagnostic in the same style as IMM.

## PF Process Summary

The intended PF recursion in repo terms is:

- propagate particles with either the dynamics prior or a proposal `q(.)`
- evaluate observation likelihoods at the new measurement
- update and normalize importance weights
- aggregate particle evidence into class-conditioned predictive evidence
  `\hat p(z_k | c)`
- update class posteriors with the shared posterior contract
- compute `N_eff` and resample only if the cloud has become too degenerate
- emit the standard evidence/posterior/diagnostic rows used everywhere else in
  the repo

That is the operational bridge from particle mechanics to the repository's
shared evaluation layer.

## RBPF As A Hybrid Sampled / Analytic Algorithm Family

Rao-Blackwellized particle filtering is only justified when the latent state
splits into a sampled hard part and a conditionally tractable continuous part.
Write

```tex
x_k = (r_k, s_k),
```

where `r_k` is the sampled latent variable and `s_k` is the continuous
substate that can still be filtered analytically. The particle update samples
the hard part,

```tex
r_k^{(i)} \sim p(r_k \mid r_{k-1}^{(i)}),
```

and then runs a conditional Kalman-style update for the tractable substate,

```tex
\hat{s}_{k|k-1}^{(i)}
=
F_k^{(r_i)} \hat{s}_{k-1|k-1}^{(i)} + b_k^{(r_i)},
```

```tex
\hat{P}_{k|k-1}^{(i)}
=
F_k^{(r_i)} \hat{P}_{k-1|k-1}^{(i)} F_k^{(r_i)\top} + Q_k^{(r_i)}.
```

The particle weight is then driven by the conditional innovation likelihood,

```tex
\tilde{w}_k^{(i)}
=
w_{k-1}^{(i)}
\mathcal{N}\!\big(\nu_k^{(i)}; 0, S_k^{(i)}\big),
```

followed by the same normalization and optional resampling used by PF.

In the current 1D implementation, this split is no longer hypothetical. The
sampled variable is the discrete mode index

```tex
r_k^{(i)} \in \{\text{coast},\text{accelerate},\text{brake},\text{maneuver}\},
```

while the conditional continuous substate is the 1D PVA block

```tex
s_k^{(i)} = \big(p_k^{(i)}, v_k^{(i)}, a_k^{(i)}\big)^\top.
```

The mode transition matrix in `rbpf_models_1d.py` samples the discrete
hypothesis, and `kalman_predict_update()` in `rbpf.py` performs the conditional
analytic update for each particle.

RBPF is the right rung only when the repo can point to a genuine split of this
form. If the state is not naturally decomposable into `(r_k, s_k)`, then RBPF
is just a more complicated PF claim, not a justified hybrid method.

## RBPF Numeric Walkthrough

Take a two-particle RBPF for an unknown maneuver-onset witness. Suppose the
sampled latent variable is the onset index and the conditional continuous state
is the usual `(p, v, a)` block. Let the two sampled onset hypotheses have prior
weights:

```tex
w_{k-1}^{(1)} = 0.60,\qquad
w_{k-1}^{(2)} = 0.40.
```

After the conditional Kalman updates, suppose the innovation likelihoods are:

```tex
\mathcal{N}(\nu_k^{(1)};0,S_k^{(1)}) = 0.42,\qquad
\mathcal{N}(\nu_k^{(2)};0,S_k^{(2)}) = 0.14.
```

The unnormalized particle weights become:

```tex
\tilde{w}_k^{(1)} = 0.252,\qquad
\tilde{w}_k^{(2)} = 0.056,
```

so after normalization:

```tex
w_k^{(1)} \approx 0.818,\qquad
w_k^{(2)} \approx 0.182.
```

If the conditional continuous-state means are:

```tex
\hat{s}_{k|k}^{(1)} = (10.0,\ 1.0,\ 0.0),\qquad
\hat{s}_{k|k}^{(2)} = (9.2,\ 0.6,\ -0.2),
```

then the RBPF combined state estimate is:

```tex
\hat{s}_{k|k}
=
0.818(10.0,\ 1.0,\ 0.0)
+
0.182(9.2,\ 0.6,\ -0.2)
\approx
(9.855,\ 0.927,\ -0.036).
```

The effective sample size is:

```tex
N_{\text{eff}}
=
\frac{1}{0.818^2 + 0.182^2}
\approx 1.42.
```

That is the numeric point of RBPF: the hard latent onset hypothesis is carried
by the particles, while the continuous kinematics remain analytically filtered
inside each particle. If a future witness does not really split that way, RBPF
is not the right rung.

## RBPF Process Summary

The intended RBPF recursion is:

- sample the hard latent variable `r_k^(i)` for each particle
- condition on `r_k^(i)` and run an analytic predict/update step for the
  tractable substate `s_k^(i)`
- use the conditional innovation likelihood to update particle weights
- normalize and optionally resample exactly as in PF
- combine the conditional state estimates into a weighted global state summary
- aggregate particle-level evidence into class or mode evidence for the shared
  posterior surface

So RBPF is not a different outer loop from PF. It is PF with an internal
analytic subfilter that reduces variance when a real sampled/analytic split is
available.

## Shared Output Contract For Advanced Filters

IMM, PF, and RBPF are only useful in the repo if they emit the same public
surface:

```tex
\text{trajectory id}, \text{time}, \text{filter id}, \text{posterior rows},
\text{state summary}, \text{evidence summary}, \text{diagnostics}.
```

That is why the advanced-state-inference implementation writes posterior
histories, state-estimate histories, likelihood histories, and diagnostics
history rows. The filter family is not the point; the shared evaluation
contract is the point.

## Code Mapping For PF And RBPF

The current repo now has concrete PF and RBPF implementations, even though they
remain decision-gated rather than promoted defaults. The main code surfaces
are:

- `advanced_filters/particle_filter.py`: bootstrap PF state, weight update,
  ESS computation, and resampling
- `advanced_filters/particle_filter_bank.py`: class-conditioned PF evidence
  extraction and posterior normalization across labels
- `advanced_filters/rbpf.py`: sampled discrete latent update plus conditional
  Kalman subfilter
- `advanced_filters/rbpf_models_1d.py`: current repo-specific RBPF mode split
  with 1D PVA conditional state blocks
- `advanced_filters/resampling.py`: shared log-weight normalization, ESS, and
  systematic resampling
- `advanced_filter_decision.py`: current go/no-go logic and numeric gate
  evidence for IMM/PF/RBPF
- `generic_filtering_contract.py`: required shared output contract for any
  advanced-filter backend
- `tests/test_particle_filter.py` and `tests/test_rbpf.py`: normalization,
  ESS, resampling, and posterior sanity checks

This means the repo can now state not only what PF and RBPF must compute, but
also where the implemented weight updates, posterior rows, and diagnostics are
already tested. Methodological gating still applies, but the algorithms are no
longer only hypothetical.

## 6. Why IMM Is Stronger Than The Transition-Matrix Rung

The transition-matrix accumulator already updates

```tex
p_k(c)
\propto
p(y_k \mid c)\sum_j T_{jc}p_{k-1}(j),
```

so it handles switching at the label level. IMM is stronger because it also
mixes mode-conditioned state estimates and covariances before the next Kalman
update. In other words:

- transition-matrix accumulation mixes class probability mass
- IMM mixes state distributions and then scores innovations per mode

That is the real upgrade claim. IMM is not just “transition matrix plus more
math.” It is the first rung where switching structure changes the latent-state
estimate itself.

## 7. IMM Gate

### 7.1 Variables

The current switching evidence compares the transition-matrix accumulator
against:

- the static accumulator
- the switching Kalman bank
- the IMM proof surface

### 7.2 Derivation

The key gains are:

```tex
\Delta_{\text{post-switch}}^{\text{TM-static}}
= A_{\text{post-switch}}^{\text{transition matrix}}
- A_{\text{post-switch}}^{\text{static accumulator}},
```

```tex
\Delta_{\text{post-switch}}^{\text{TM-kalman}}
= A_{\text{post-switch}}^{\text{transition matrix}}
- A_{\text{post-switch}}^{\text{switching Kalman bank}}.
```

```tex
\Delta_{\text{post-switch}}^{\text{IMM-TM}}
= A_{\text{post-switch}}^{\text{IMM}}
- A_{\text{post-switch}}^{\text{transition matrix}}.
```

The current interpretation is:

- if `Delta_post-switch^(TM-static) > 0`, switching structure matters at least
  at the label level
- if `Delta_post-switch^(IMM-TM) > 0` materially and consistently, then full
  state mixing may be justified rather than only transition accumulation
- if the switching Kalman bank still loses to the transition rung, then the
  simpler switching ladder is not exhausted yet

That is why IMM remains evidence-gated even though the repo now has an IMM
proof surface. Existence of an implementation is not the same thing as a claim
that the implementation is necessary.

### 7.3 Implementation Mapping

- `analyze_advanced_filter_decision()`
- `run_transition_benchmark(...)`
- `run_imm_filter(...)`
- `inference/advanced_state_inference.py`

## 8. Particle-Filter Gate

### 8.1 Problem

Particle filtering should not be justified only because it is more general. It
should be justified because the problem has become nonlinear, non-Gaussian, or
multimodal in a way that the current ladder cannot absorb.

### 8.2 Decision Logic

The intended gate is:

```tex
\text{PF justified}
\Longrightarrow
\text{nonlinear benchmark exists}
\land
\text{robust Kalman still fails}
\land
\text{failure is not primarily sensing-limited}.
```

This matters because the current hard case is still largely evidence-limited.
If direct velocity sensing resolves a large fraction of the failure, then the
bottleneck is not yet “we need a particle filter.”

### 8.3 Implementation Mapping

- shared PF/RBPF registry adapters in
  `advanced_filters/shared_classifier_methods.py`
- short-horizon identifiability metrics from
  `short_horizon_identifiability.py`
- velocity-aided comparison from `inference/velocity_aided_kalman_comparison.py`
- robust Kalman comparison from `inference/kalman_variant_comparison.py`
- gate assembly in `advanced_filter_decision.py`
- mean-reverting witness in `advanced_filters/ou_witness.py`

## 9. RBPF Gate

### 9.1 Structural Requirement

RBPF has a stricter justification condition. The state must decompose as:

```tex
x_t = (u_t, v_t),
```

where:

- `u_t` is a sampled discrete or nonlinear latent structure
- `v_t` is a conditionally linear-Gaussian substate that remains analytically
  filterable

### 9.2 Interpretation

RBPF is therefore justified only when:

- part of the problem truly needs particles
- another part of the problem is still tractable enough to Rao-Blackwellize

Without that split, “RBPF” is only a fancy name for unnecessary complexity.

## 10. Worked Example

The numeric artifact
[advanced_filter_decision_numeric_walkthrough.md](artifacts/advanced_filter_decision_v1/advanced_filter_decision_numeric_walkthrough.md)
is the current concrete proof for this document. It works through one real
decision pass and shows:

- the actual transition gains
- the nominal-noise identifiability gaps
- the velocity-aided `short_noisy` gain
- the best current Kalman outlier accuracy
- the status of each IMM and PF evidence row
- why those values lead to `defer IMM` and `defer PF`

This is the proper bridge from gate logic to code.

## 11. Failure Modes

This methodology layer can still fail if:

- the dimensional audit misses a hidden scalar assumption
- the advanced-filter decision study lacks the right witness benchmark
- a sensing-limited failure is mistaken for an inference-limited failure
- a nonlinear or switching witness case is added but not compared fairly against
  the simpler ladder

That is why this document is paired with concrete gate artifacts rather than
only prose recommendations.

## 12. What This Document Proves

This note is complete only if it supports the following claims:

- dimensional lift is being audited through explicit contracts and assumption
  rows
- 3D readiness is defined by adapters and interfaces, not slogans
- IMM, PF, and RBPF are decision-gated by measured evidence even though PF and RBPF now have shared-surface implementations
- the current go/no-go outcome is numerically justified by a real walkthrough