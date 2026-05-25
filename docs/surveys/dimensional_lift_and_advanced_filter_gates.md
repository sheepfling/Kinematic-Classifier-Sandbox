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

## 5. IMM Gate

### 5.1 Variables

The current switching evidence compares the transition-matrix accumulator
against:

- the static accumulator
- the switching Kalman bank

### 5.2 Derivation

The key gains are:

```tex
\Delta_{\text{post-switch}}^{\text{TM-static}}
= A_{\text{post-switch}}^{\text{transition matrix}}
- A_{\text{post-switch}}^{\text{static accumulator}},
```

```tex
\Delta_{\text{overall}}^{\text{TM-static}}
= A_{\text{overall}}^{\text{transition matrix}}
- A_{\text{overall}}^{\text{static accumulator}},
```

```tex
\Delta_{\text{post-switch}}^{\text{TM-kalman}}
= A_{\text{post-switch}}^{\text{transition matrix}}
- A_{\text{post-switch}}^{\text{switching Kalman bank}}.
```

As long as the transition-matrix accumulator still buys measurable post-switch
gain, and the switching Kalman bank has not displaced it, the repo does not yet
have evidence that IMM complexity is required.

### 5.3 Implementation Mapping

- `analyze_advanced_filter_decision()`
- `run_transition_benchmark(...)`

## 6. Particle-Filter Gate

### 6.1 Problem

Particle filtering should not be justified only because it is more general. It
should be justified because the problem has become nonlinear, non-Gaussian, or
multimodal in a way that the current ladder cannot absorb.

### 6.2 Decision Logic

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

### 6.3 Implementation Mapping

- short-horizon identifiability metrics from
  `short_horizon_identifiability.py`
- velocity-aided comparison from `velocity_aided_kalman_comparison.py`
- robust Kalman comparison from `kalman_variant_comparison.py`
- gate assembly in `advanced_filter_decision.py`

## 7. RBPF Gate

### 7.1 Structural Requirement

RBPF has a stricter justification condition. The state must decompose as:

```tex
x_t = (u_t, v_t),
```

where:

- `u_t` is a sampled discrete or nonlinear latent structure
- `v_t` is a conditionally linear-Gaussian substate that remains analytically
  filterable

### 7.2 Interpretation

RBPF is therefore justified only when:

- part of the problem truly needs particles
- another part of the problem is still tractable enough to Rao-Blackwellize

Without that split, “RBPF” is only a fancy name for unnecessary complexity.

## 8. Worked Example

The numeric artifact
[advanced_filter_decision_numeric_walkthrough.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/advanced_filter_decision_v1/advanced_filter_decision_numeric_walkthrough.md)
is the current concrete proof for this document. It works through one real
decision pass and shows:

- the actual transition gains
- the nominal-noise identifiability gaps
- the velocity-aided `short_noisy` gain
- the best current Kalman outlier accuracy
- the status of each IMM and PF evidence row
- why those values lead to `defer IMM` and `defer PF`

This is the proper bridge from gate logic to code.

## 9. Failure Modes

This methodology layer can still fail if:

- the dimensional audit misses a hidden scalar assumption
- the advanced-filter decision study lacks the right witness benchmark
- a sensing-limited failure is mistaken for an inference-limited failure
- a nonlinear or switching witness case is added but not compared fairly against
  the simpler ladder

That is why this document is paired with concrete gate artifacts rather than
only prose recommendations.

## 10. What This Document Proves

This note is complete only if it supports the following claims:

- dimensional lift is being audited through explicit contracts and assumption
  rows
- 3D readiness is defined by adapters and interfaces, not slogans
- IMM, PF, and RBPF are decision-gated by measured evidence
- the current go/no-go outcome is numerically justified by a real walkthrough
