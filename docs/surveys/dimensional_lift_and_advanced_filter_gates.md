# Dimensional Lift and Advanced Filter Gates

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
