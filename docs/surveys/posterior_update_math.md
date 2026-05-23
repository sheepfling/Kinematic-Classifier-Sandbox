# Posterior Update Math for Toy and Identity Benchmarks

This note documents the posterior update math used by the two active benchmark
families in the sandbox:

- `toy_1d.py`: class-matched latent-state filter bank
- `identity_1d.py`: direct speed-identity classifier over `bike`, `horse`, `car`

The goal is not to present an abstract Bayesian classifier. It is to show the
specific scoring structure the repo is actually using today.

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
