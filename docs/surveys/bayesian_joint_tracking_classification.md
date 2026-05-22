# Bayesian Joint Tracking and Classification Baseline

This note captures the recommended baseline for the sandbox: Bayesian joint
tracking and classification with class-conditioned physical constraints rather
than a plain feature classifier over derived kinematic statistics.

## Core formulation

Use a latent target state such as:

- `x_k = [r_k, v_k, a_k]`
- or `x_k = [r_k, v_k]` with acceleration inferred by the model

Alongside:

- class `c` in `{known classes} U {unknown}`
- within-class dynamic mode `m_k`
- optional physical parameters `theta_k = [log(beta), L/D, turn params, thrust/drag params, ...]`

Class is typically static or slow-changing. Mode is allowed to switch at the
time scale of target behavior.

## Recommended architecture

The baseline architecture is:

1. Class-matched filter bank across feasible object classes.
2. IMM or related multiple-model layer inside each class.
3. Constraint likelihoods computed from the state covariance.
4. Optional parameter-evidence terms for `beta`, `L/D`, turn, and drag proxies.
5. Explicit unknown-class handling.

This keeps the tracking and classification pieces coupled in the likelihood
rather than bolting a classifier onto already-decoded trajectories.

## Posterior update

Use class weights updated in log space:

`w_{c,k} proportional w_{c,k|k-1} * L_{c,k}`

If class is static, preserve the prior class weight. If recovery from early
mistakes matters, use a mostly diagonal class-transition matrix with small mass
for the unknown class.

## Likelihood composition

The baseline class likelihood should combine dynamics, soft validity, and
optional parameter evidence:

`log L_{c,k} = log L_dyn + lambda_env * log(epsilon + P_valid,c) + log L_theta`

Useful concrete terms include:

- innovation or residual likelihood from the class-matched filter
- speed envelope probability
- altitude envelope probability
- horizontal and vertical acceleration envelope probabilities
- optional `beta` likelihood
- optional `L/D` likelihood

The repo should prefer covariance-aware probabilities instead of hard pass/fail
gates. Scalar envelope terms can start with Gaussian CDFs, while nonlinear
constraints can use linearization, sigma points, or particle approximations.

## Unknown-class handling

Do not normalize only over known classes. The sandbox should maintain explicit
unknown-class mass with a broad fallback dynamic model so unsupported objects do
not get forced into the nearest known bucket.

## Practical cautions

- If upstream PVA states already come from a filter, successive errors are
  correlated and absolute-state likelihoods can double-count evidence.
- Some physical parameters are only informative when dynamic pressure is high.
- IMM performance depends heavily on the component model set and transition
  design.

## Sandbox implications

The first implementation pass should target:

- class-bank and mode-bank interfaces
- covariance-aware envelope probabilities
- log-space class-weight updates
- optional augmented-state parameter evidence hooks
- synthetic scenarios for ballistic, glide, powered, hover, and high-turn cases
