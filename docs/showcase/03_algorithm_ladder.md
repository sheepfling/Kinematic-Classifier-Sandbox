# Epic 2: Classifier / Filter Evidence Ladder

Subtitle: Build an evidence ladder that starts simple, then deliberately
excites advanced algorithms with named failure regimes that anticipate the 3D
lift.

Epic 1 screened the proposed feature/class/prior setup. Now we assume the study
is meaningful enough to run. The next question is not "which classifier is
fanciest?" The question is how to build an evidence ladder that starts simple,
then deliberately excites advanced algorithms with named failure regimes that
anticipate the 3D lift.

The classifier/filter ladder is an evidence ladder. Each rung consumes the same
tracklet surface and emits comparable posterior histories. Simple 1D witnesses
prove the shared contract; advanced witnesses prove that the architecture can
escalate into richer state inference when the problem actually demands it.

We evaluate the full ladder, promote selectively, and keep positive showcase
witnesses for IMM, PF, and RBPF as the route to 3D.

## Main Message

Classification is posterior evidence over a tracklet, not just a final label.

The deck should emphasize:

- every method emits a posterior history
- all applicable methods are evaluated on the same study surface
- posterior behavior matters more than final labels alone
- methods are compared through a shared contract
- advanced filters are not magic
- advanced filters are gated but must have positive showcase witnesses
- escalation is justified by diagnosed failure modes and 3D lift relevance

## Central Question

Use one repeated question throughout the section:

> What evidence is available up to time `t`, and how should it update belief
> over classes?

## Shared Contract

Every rung should be presented through the same contract:

```text
Tracklet observations
  -> EvidenceProvider(method)
  -> evidence_by_class[t, c]
  -> PosteriorUpdater(prior)
  -> posterior_history[t, c]
  -> Evaluation
```

That is why the ladder is disciplined rather than leaderboard-driven.

The decision rule is separate from the evaluation rule:

- evaluation rule: run every applicable rung
- promotion rule: select the simplest sufficient rung

## Ladder

The ladder is:

```text
pointwise
  -> windowed
  -> sequential Bayes
  -> Kalman bank
  -> transition matrix
  -> IMM
  -> particle filter
  -> RBPF
```

The ladder is not a ranking. It is a sequence of evidence capabilities.

## Three Layers

Layer A: Baseline ladder.
Purpose: prove the shared contract across pointwise, windowed, sequential
Bayes, Kalman bank, and transition matrix.

Layer B: Advanced algorithm showcase.
Purpose: prove the architecture supports IMM, PF, and RBPF on named witnesses
that excite their assumptions.

Layer C: 3D lift bridge.
Purpose: explain why nonlinear geometry, vector PVA, sensor noise, occlusion,
mode uncertainty, and latent maneuver structure make these algorithms
operationally important beyond readable 1D examples.

## Seven Beats

1. Static audit says the study is meaningful.
2. Now we ask how evidence should accumulate over time.
3. Every method must emit the same evidence/posterior contract.
4. The ladder adds one capability at a time.
5. Witness problems isolate why each capability matters.
6. Rung sufficiency keeps complexity honest.
7. Advanced witnesses prove the escalation path toward 3D state inference.

## Main-Deck Slides

Slide 1: Every rung emits comparable evidence.
Visual: `06b_evidence_contract_spine.png`

Slide 2: Simple rungs establish sanity and sufficiency.
Visual: `07_rung_sufficiency_map.png`

Slide 3: Complexity is not automatic; it must earn promotion.
Visual: `07_rung_sufficiency_map.png`

Slide 4: Advanced algorithms need harder witnesses.
Visual: `10f_simple_to_advanced_witness_bridge.png`

Slide 5: Advanced filters shine in different failure regimes.
Visual: `10e_advanced_filter_sweet_spot_matrix.png`

Slide 6: 3D lift explains why advanced inference matters.
Visual: `10f_method_win_by_regime_map.png`

## Capability Table

| Rung | New capability | Failure mode it addresses |
| --- | --- | --- |
| Pointwise | local evidence | local overlap and weak single-step evidence |
| Windowed | short-horizon shape | noisy local observations |
| Sequential Bayes | evidence accumulation | repeated weak evidence |
| Kalman bank | dynamic residual evidence | matched local features with different dynamics |
| Transition matrix | switching logic | class or mode changes over time |
| IMM | mode-mixed dynamic state evidence | switching dynamics where state mixing matters |
| PF | nonlinear or non-Gaussian posterior evidence | Gaussian summaries are misleading |
| RBPF | latent discrete structure plus conditional continuous state | mixed discrete/continuous latent structure |

## Evaluation Layers

Layer 1: Common benchmark surface.
Question: how do all applicable methods compare on the same admissible study?

Layer 2: Failure-mode witness surface.
Question: when a simpler rung fails, which richer rung fixes that failure?

Layer 3: Shine-regime stress surface.
Question: where does each advanced filter actually shine?

These three layers should yield:

- `classifier_ladder_report.md`
- `full_ladder_metrics.csv`
- `method_status_table.csv`
- `posterior_history_by_method.csv`
- `calibration_by_method.csv`
- `confusion_by_method.csv`
- `runtime_by_method.csv`
- `rung_sufficiency_report.md`
- `advanced_filter_shine_report.md`
- `method_win_by_regime.csv`
- `classifier_ladder_decision_card.md`

Additional hero charts:

- `07b_full_ladder_comparison_dashboard.png`
- `10e_advanced_filter_sweet_spot_matrix.png`
- `10f_simple_to_advanced_witness_bridge.png`
- `10f_method_win_by_regime_map.png`
- optionally `07c_complexity_benefit_pareto.png`

Advanced witness cards:

- `imm_mode_switching_state_mixing.md`
- `pf_nonlinear_nongaussian_posterior.md`
- `rbpf_latent_event_timing.md`

## Status Vocabulary

Use statuses that separate running a method from promoting it:

- `evaluated`
- `applicable`
- `competitive`
- `simplest_sufficient`
- `witness_supported`
- `promoted`
- `deferred`
- `not_applicable`

## Claim Boundary

Do not say:

- advanced filters are the final goal
- the ladder is an accuracy leaderboard
- IMM, PF, or RBPF are globally better

Say instead:

- the repo compares methods through a shared evidence contract
- all applicable methods can be evaluated without being promoted
- posterior histories are the main behavioral object
- complexity is added only when a failure mode requires it
- IMM, PF, and RBPF each target different failure regimes
- advanced filters are gated, but the repo includes positive showcase witnesses
  that prove the escalation path toward 3D
- PF and RBPF stay as required showcase/candidate diagnostic claims until named
  run-backed witnesses promote them for targeted applicability
