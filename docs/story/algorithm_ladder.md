# Algorithm Ladder

Epic 2 is the classifier/filter evidence story.

The classifier/filter ladder is an evidence ladder. Each rung consumes the same
tracklet surface and emits comparable posterior histories. Complexity is added
only when a diagnosed failure mode requires a richer evidence model.

The ladder is not a ranking. It is a sequence of evidence capabilities.

Epic 2 proves both discipline and ambition: simple rungs establish the shared
posterior-evidence contract, while advanced IMM/PF/RBPF witnesses demonstrate
that the same architecture can escalate toward the nonlinear, switching, and
latent-state problems expected in a 3D tracking lift.

Epic 1 screened the proposed feature/class/prior setup. Now the study is
meaningful enough to run, so the next question is not which method is fanciest.
The question is: how do we build an evidence ladder that starts simple, then
deliberately excites advanced algorithms with named failure regimes that
anticipate the 3D lift?

## Core Question

Return to one question across every rung:

> What evidence is available up to time `t`, and how should it update belief
> over classes?

That is what lets pointwise classifiers, sequential Bayes, Kalman residuals,
transition logic, IMM, PF, and RBPF live under one evaluation vocabulary.

## Shared Contract

Every method must emit the same evidence and posterior contract:

```text
observations up to time t
        -> method-specific evidence provider
        -> class evidence / likelihood-like scores
        -> posterior updater under explicit prior
        -> posterior history over classes
        -> evaluation and promotion decision
```

This shared contract supports two separate decisions:

- evaluate all applicable methods on the same study surface
- promote only the simplest method whose added complexity earns its keep

The shared evaluation surface is therefore not just final accuracy. It includes:

- posterior behavior
- switch timing
- calibration
- confusion localization
- prior sensitivity
- oracle gap
- rung sufficiency
- failure-mode diagnosis

## Evaluation Design

Epic 2 uses three layers because they answer different questions.

### Layer A: Baseline Ladder

Purpose: prove the shared contract.

Run:

- pointwise
- windowed
- sequential Bayes
- Kalman bank
- transition matrix

These rungs prove that all methods can emit comparable evidence and posterior
histories.

Primary outputs:

- `full_ladder_metrics.csv`
- `posterior_history_by_method.csv`
- `runtime_by_method.csv`
- `rung_comparison_table.md`
- `07b_full_ladder_comparison_dashboard.png`

Interpretation: this layer compares behavior and establishes sanity. It does
not automatically justify the most complex method.

### Layer B: Advanced Algorithm Showcase

Purpose: prove the architecture can support serious inference.

Showcase methods:

- IMM
- PF
- RBPF

Primary outputs:

- `advanced_filter_shine_report.md`
- `imm_mode_switching_state_mixing.md`
- `pf_nonlinear_nongaussian_posterior.md`
- `rbpf_latent_event_timing.md`
- `10e_advanced_filter_sweet_spot_matrix.png`
- `10f_simple_to_advanced_witness_bridge.png`

Interpretation: these algorithms are not decorative. Each has a deliberately
constructed witness where its assumptions are relevant.

### Layer C: 3D Lift Bridge

Purpose: explain why advanced algorithms matter even if simple 1D examples
often look sufficient.

Primary outputs:

- `method_win_by_regime.csv`
- `10f_method_win_by_regime_map.png`
- dimensional-lift report links

Interpretation: 1D is where we make the contract readable. 3D is where the
advanced algorithms become operationally important because vector PVA,
nonlinear geometry, maneuvering targets, mode uncertainty, sensor noise,
occlusion, and multimodal hypotheses become normal rather than exceptional.

## Story Arc

Epic 2 should read in seven beats:

1. Static audit says the study is meaningful.
2. Now we ask how evidence should accumulate over time.
3. Every method must emit the same evidence/posterior contract.
4. The ladder adds one capability at a time.
5. Witness problems isolate why each capability matters.
6. Rung sufficiency keeps complexity honest.
7. Advanced witnesses prove the escalation path toward 3D state inference.

## Capability Ladder

| Rung | Method | New capability | Best for | Main failure mode | Promotion evidence |
| --- | --- | --- | --- | --- | --- |
| 0 | Pointwise | local evidence | separable instantaneous features | temporal ambiguity and prior-sensitive local overlap | `pointwise_overlap` shows local evidence is meaningful and prior behavior is legible |
| 1 | Windowed | short-horizon shape | extrema, outliers, and local patterns | long-horizon ambiguity and lack of explicit accumulation | `windowed_outlier_extrema` improves hard local witnesses |
| 2 | Sequential Bayes | history accumulation | repeated weak evidence | switching and correlated or nonstationary evidence | `sequential_history` stabilizes posterior belief over time |
| 3 | Kalman bank | dynamic residual evidence | model-matched motion regimes | one model cannot explain switching or latent events | `kalman_endpoint_match` shows residual evidence separates matched-endpoint tracks |
| 4 | Transition matrix | switching logic | class or mode evolution with explicit transition constraints | label-level logic cannot mix state uncertainty | `transition_switching` improves post-switch behavior before IMM |
| 5 | IMM | mode-mixed dynamic state inference | switching dynamics where state mixing matters | strongly nonlinear or non-Gaussian posterior structure | `imm_switching_v1` shows state mixing improves switching-state evidence |
| 6 | PF | sampled nonlinear or non-Gaussian posterior evidence | nonlinear, non-Gaussian, or multimodal posterior regimes | particle degeneracy, compute cost, or missing latent structure exploitation | `pf_abs_range_multimodal_oracle_v1` and OU-style witnesses beat cheaper Gaussian summaries for the right reason |
| 7 | RBPF | sampled discrete path plus conditional continuous state inference | latent event timing or mode paths with conditional linear-Gaussian state | model mismatch or unnecessary complexity if the split adds nothing | `latent_maneuver_onset_1d` and the PF-vs-RBPF frontier justify the sampled/marginalized split |

The ladder rule is simple: each rung must be justified by a failure mode that
the previous rung cannot explain or solve.

## Status Vocabulary

Separate these meanings explicitly:

| Status | Meaning |
| --- | --- |
| `evaluated` | The method ran on the study surface and emitted comparable evidence and posterior rows. |
| `applicable` | The study excites the assumptions the method is designed for. |
| `competitive` | The method performs reasonably on the declared metrics. |
| `simplest_sufficient` | No more complex rung materially improves the study decision. |
| `witness_supported` | The method wins a named failure-mode witness it was designed to address. |
| `promoted` | The method is selected for a specific study or witness decision. |
| `deferred` | The method ran but did not earn complexity justification. |
| `not_applicable` | The study surface does not exercise the method's assumptions fairly. |

The point is to allow "evaluated" without forcing "promoted."

## Rung Interpretation

Pointwise is the sanity check: if classes are separable locally, do not
overcomplicate the method.

Windowed features help when a class is visible in local shape rather than a
single observation.

Sequential Bayes shows the difference between a classifier that reacts and a
classifier that accumulates belief.

The Kalman bank turns tracking residuals into class evidence.

The transition matrix is the first explicit switching logic; it often deserves
to beat fancier filters unless state mixing is truly needed.

IMM earns its complexity only when mode mixing improves switching-state
evidence beyond transition logic and Kalman-bank baselines.

PF is justified only when nonlinear or non-Gaussian evidence breaks the simpler
filters for the right reason.

RBPF earns its complexity when latent event structure plus conditional Kalman
filtering beats both IMM and plain PF.

## Witness Cards

Each rung should carry a small witness card:

- Witness: named failure case.
- Failure mode: what simpler evidence construction misses.
- Simpler rung expected to fail: the current lower bound.
- Rung expected to help: the capability being justified.
- Observed result: what the posterior history or diagnostics changed.
- Decision: promote, defer, revise, or keep as witness-specific support.
- Limitation: what the witness still does not prove.

The important witness interpretations are:

- `transition_switching`: transition logic is justified before IMM.
- `imm_switching_v1`: witness support for switching-state evidence, not a
  blanket promotion.
- `pf_abs_range_multimodal_oracle_v1`: PF is justified when Gaussian summaries
  collapse a real multimodal posterior.
- `latent_maneuver_onset_1d`: RBPF is justified when latent event structure and
  conditional Kalman updates beat broader sampling.

## Claim Boundary

The strongest claim is not that advanced filters are best.

The strongest claim is that the repo can compare classifier and filter families
through a shared posterior/evidence contract, decide when added complexity is
justified, and show positive escalation witnesses for the algorithms that will
matter more in 3D.

The mature position is:

> We evaluate the full ladder to understand capability, failure modes, and
> tradeoffs. The decision layer still asks whether the added complexity earned
> its keep.

The safe current advanced-filter claim is:

> Advanced filters are gated, but the repo includes positive showcase witnesses
> for each one so the architecture proves it can scale beyond simple 1D
> separability. IMM has witness support for switching-state evidence. PF/GSF
> and RBPF have run-backed witness routes for nonlinear/non-Gaussian posterior
> structure and latent-event inference. Those witnesses prove integration,
> traceability, and promotion machinery; broader usefulness still depends on a
> study that excites the matching assumptions.

That caveat is intentional. It prevents the ladder from collapsing into an
accuracy leaderboard or an "advanced filters win" story.

## Decision Rule

The main-study selection rule is:

> pick the simplest method that satisfies the evidence, posterior, calibration,
> prior-sensitivity, and failure-mode gates.

Promotion therefore requires all of the following:

- material improvement on a predeclared metric
- calibration and posterior sanity still pass
- improvement occurs in the relevant hard pair or failure regime
- the improvement is not better explained by corpus, feature, or prior defects
- the added complexity is worth the runtime and implementation cost

PF and RBPF should be promoted only for named witnesses or studies where the
posterior geometry or latent structure actually requires them.

## Advanced Showcase Witnesses

| Algorithm | Witness | What it proves |
| --- | --- | --- |
| IMM | `mode_switching_state_mixing` | Mode probabilities and mixed dynamic state improve switching-state evidence. |
| PF | `nonlinear_nongaussian_posterior` | Sampled posterior representation helps when Gaussian residual models are misleading. |
| RBPF | `latent_event_timing` | Sampling discrete mode/event structure while marginalizing continuous state can beat plain PF/IMM. |
