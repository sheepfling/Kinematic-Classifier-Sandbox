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

## 3. Identity benchmark

### 3.1 Measurement model

The identity benchmark is simpler. The measurement at step `t` is a directly
observed scalar speed:

```tex
z_t = \text{observed speed in mph}.
```

There is no latent Kalman state here. Each class `s_i` is represented by a
speed-shape prior and a few history terms.

### 3.2 Base speed-shape likelihood

For class `s_i` with cruise mean `\mu_i` and class spread `\sigma_i`, the
instantaneous speed fit is

```tex
\log L^{\text{speed-shape}}_{i,t}
= \log \mathcal{N}(z_t; \mu_i, \sigma_i^2 + \sigma_{\text{obs}}^2).
```

This is the dominant `speed_shape` term shown in the identity artifacts.

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

## 4. Feature probabilities are not the class posterior

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

## 5. Why the confusion matrices matter

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

## 6. Practical interpretation

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
