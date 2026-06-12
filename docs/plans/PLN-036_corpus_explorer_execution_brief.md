# PLN-036 Corpus Explorer Execution Brief

Title: Corpus Explorer execution brief and advanced-algorithm route plan
Plan ID: PLN-036
Status: active
Owner: @rick
Priority: P1
Last Updated: 2026-06-11

## Short Goal Blurb

Implement V5C Corpus Explorer MVP as a corpus decision system, not a data-generator demo. The agent must turn static warnings and classifier-ladder failures into audited search objectives, discover valid hard cases, reject invalid or leaky cases, and route every selected case to a concrete ladder or study action.

## Objective

Put the Epic 3 execution plan on disk in a form that is tight enough for agent handoff and strict enough to keep the work aligned through completion.

The plan must preserve two architectural claims:

1. Epic 3 is about valid hard-case discovery, not synthetic-data novelty.
2. Advanced algorithms are showcased as study architecture: the repo can excite, diagnose, and route them before the 3D lift fully operationalizes them.

## Section Thesis

Corpus Explorer turns static warnings and classifier failures into targeted
search objectives, then discovers valid, non-leaky, diagnostically useful
tracklets that stress the evidence ladder.

## Architectural Framing

The three-epic flow is:

1. Epic 1 identifies static risks.
2. Epic 2 identifies evidence and inference failure regimes.
3. Epic 3 turns those risks into corpus-search objectives, audits generated candidates, and routes valid discoveries back into feature, prior, corpus, or classifier/filter decisions.

The output is not "generated data." The output is a selected corpus manifest,
hard-case cards, backend comparisons, and decision cards that change what the
study does next.

## Scope

### In Scope

- V5C Corpus Explorer packet under `artifacts/packets/corpus_explorer_mvp/`
- Advanced algorithm routing proof for `IMM`, `PF/GSF`, `RBPF`, and
  stochastic-dynamics witness routes
- Backend comparison for random baselines, `QD` archive, `CEM`, and `PPO`
- Validity, leakage, coverage, and diagnostic-yield gating
- Hard-case cards for both selected and rejected candidates

### Out of Scope

- Claiming `PPO` or `CEM` is generally promoted without matched baseline
  comparison, ablation, seed stability, and downstream yield
- Claiming advanced filters are globally superior on the main toy study
- Treating invalid but difficult cases as evidence
- Treating the 1D witnesses as the final 3D proof

## Execution Rules

- Hard is not enough. Selected cases must be valid, non-leaky, and online-safe.
- Every selected hard case must route to an action:
  `revise_features`, `revise_priors`, `revise_corpus`,
  `evaluate_specific_rung`, or `create_advanced_filter_witness`.
- Every rejected hard case must include a rejection reason.
- `CEM` and `PPO` are search backends, not promoted novelty methods, until the
  baseline and stability gates are cleared.
- Advanced algorithms can pass a shine witness without being generally best.
- The packet must terminate in a `corpus_explorer_decision_card`.

## Deliverables

- `artifacts/packets/corpus_explorer_mvp/README.md`
- `artifacts/packets/corpus_explorer_mvp/corpus_explorer_decision_card.md`
- `artifacts/packets/corpus_explorer_mvp/corpus_objective.yaml`
- `artifacts/packets/corpus_explorer_mvp/selected_corpus_manifest.csv`
- `artifacts/packets/corpus_explorer_mvp/corpus_candidate_frontier.csv`
- `artifacts/packets/corpus_explorer_mvp/corpus_adequacy_report.md`
- `artifacts/packets/corpus_explorer_mvp/leakage_adequacy_audit.csv`
- `artifacts/packets/corpus_explorer_mvp/search_backend_comparison.csv`
- `artifacts/packets/corpus_explorer_mvp/downstream_diagnostic_yield.csv`
- `artifacts/packets/corpus_explorer_mvp/novelty_to_filter_escalation_report.md`
- `artifacts/packets/corpus_explorer_mvp/advanced_algorithm_route_matrix.csv`
- `artifacts/packets/corpus_explorer_mvp/advanced_algorithm_route_proof.md`
- `artifacts/packets/corpus_explorer_mvp/hard_case_cards/*.md`

## Hero Charts

- `03_corpus_candidate_frontier.png`
- `18_leakage_adequacy_audit.png`
- `21_search_backend_comparison_frontier.png`
- `26_downstream_diagnostic_yield.png`
- `27_novelty_to_filter_escalation_bridge.png`

## Algorithms: Proved Out vs Planned

### Proved Out or Witness-Supported Now

| Algorithm | Current role | Proof standard |
| --- | --- | --- |
| `IMM` | switching-state witness route | trace-validated and witness-supported on switching failures |
| `PF/GSF` | nonlinear or non-Gaussian posterior witness route | trace-validated and justified for multimodal posterior cases |
| `RBPF` | latent-event witness route | trace-validated and witness-supported on latent discrete structure |
| `OU/PF` stochastic witness | slow-velocity and stochastic-dynamics bridge | witness-supported route for state-process ambiguity |
| `TS2Vec` proxy route | representation-learning frontier | proxy witness only; not a broad finished claim |

### Planned or Gated

| Algorithm | Planned role | Promotion gate |
| --- | --- | --- |
| `UKF/EKF` | nonlinear Gaussian intermediate rung | run-backed witness beyond current linear-Gaussian cases |
| `Student-t / robust Kalman` | heavy-tail robustness witness | matched outlier witness plus baseline comparison |
| `BOCPD` / changepoint methods | event-boundary witness | explicit event-timing route and decision yield |
| learned sequence or embedding generators | richer open-loop frontier | comparable artifacts and claim-bounded witness routes |
| 3D advanced filters | full operational lift | geometry, ambiguity, and latent-state witnesses in 3D |

## Exploration Types: Proved Out vs Planned

### Proved Out or Packet-Backed Now

| Exploration type | Current status | What is actually proved |
| --- | --- | --- |
| random sampling baseline | baseline | establishes search floor and comparison point |
| scripted profile families | baseline family | hand-authored boundary families remain useful controls |
| design-of-experiments schedules | baseline family | parameter sweeps and schedule banks provide open-loop controls |
| `QD` / archive search | useful | valid diversity and medium diagnostic yield are packet-backed |
| `CEM` | experimental or run-backed | interpretable parameter search can find useful hard cases |
| `PPO` | experimental witness | sequential boundary shaping is exercised but not fully promoted |
| adequacy and leakage gating | proved | invalid hard cases are filtered before ladder influence |
| novelty-to-filter escalation | proved | selected cases route into ladder or study actions |

### Planned or Gated

| Exploration type | Planned role | Gate |
| --- | --- | --- |
| Latin hypercube / Sobol coverage designs | stronger coverage-first baseline | explicit coverage packet and comparable search budgets |
| MAP-Elites style illumination map | broader behavior-space archive | stable descriptor scheme and archive-yield evidence |
| active-learning acquisition | disagreement-driven generation | committee contract and density-aware acquisition packet |
| falsification / adaptive stress testing | failure-region search | robustness objective and validity-constrained counterexamples |
| off-policy RL backends such as `SAC` or `TD3` | richer sequential control frontier | stricter baseline, ablation, and runtime discipline |
| 3D geometry-aware search | operational hard-case discovery | sensor geometry, occlusion, and latent maneuver witnesses |

## Workstreams

1. Objective intake
   Convert Epic 1 warnings and Epic 2 failures into explicit corpus objectives.
2. Candidate generation and search
   Run baseline, archive, and advanced backends under a shared utility surface.
3. Adequacy and leakage audit
   Reject candidates that are invalid, leaky, unreachable, or artifact-driven.
4. Selection and routing
   Promote only candidates with diagnostic value and an explicit downstream action.
5. Decision closure
   End in a packet and decision card, not a pile of generated traces.

## Definition Of Done

- The V5C packet exists and all listed files are present.
- Every selected candidate has `validity_status`, `leakage_status`, and
  `routed_action`.
- Every rejected candidate has a `rejection_reason`.
- `CEM` and `PPO` are not presented as promoted methods without baseline and
  diagnostic-yield support.
- Downstream-yield claims cite a concrete ladder route.
- No invalid or leaky candidate is selected.
- The packet ends in `corpus_explorer_decision_card.md`.

## Agent Handoff Blurb

Use Epic 3 as a corpus decision system. Start from static warnings and
classifier-ladder failures, convert them into explicit corpus objectives, run
baseline and advanced search backends under matched claim discipline, reject
invalid hard cases, and route every selected case to a concrete ladder or
study action. Finish only when the packet, hard-case cards, backend comparison,
and decision card all agree on what was selected, what was rejected, and what
changed downstream.
