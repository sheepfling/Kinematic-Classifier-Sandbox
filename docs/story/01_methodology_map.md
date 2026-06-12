# Methodology Map

The methodology is a compositional flow from study intent to decision.

```text
Study Candidate Intent
  -> Static Feature/Class/Prior Audit
  -> Corpus Objective
  -> Backend / Generator / CorpusGym / Search / QD
  -> Backend Registry / Capability Gating / Benchmark Lane Selection
  -> Validated Corpus
  -> Feature Set + Class Validity
  -> Evidence Provider
  -> Posterior Updater + Prior
  -> State Tracker / Switching Filter / Particle Witness
  -> Metrics / Separability / Confusion / Calibration
  -> Validation Ladder
  -> Promote / Revise / Reject / Defer
```

## Central Study Unit

```text
s = (D, f, C, m, pi, b)
```

The study candidate is the unit that lets the repo compare feature sets, class sets, classifier/filter families, prior regimes, corpora, and optional backends without rewriting the evaluation stack.

## Layer Responsibilities

| Layer | Responsibility | Representative artifacts |
| --- | --- | --- |
| Static feature/class/prior audit | Decide whether the proposed feature set, class set, and prior regime are identifiable, non-pathological, non-leaky, covered enough, and decisionable before corpus search or classifier escalation | `artifacts/static_feature_class_prior_audit_v1/static_decision_card.md`; `artifacts/static_feature_class_prior_audit_v1/02b_static_audit_decision_card.png`; `artifacts/static_feature_class_prior_audit_v1/02g_prior_pathology_surface.png` |
| Corpus objective | State the intended coverage, stress, backend, and validity target | `artifacts/corpus_objectives/objective_validation_report.md` |
| Corpus explorer | Generate, search, score, and select trajectories | `artifacts/generic_corpus_exploration/candidate_scores.csv` |
| Exploration backend registry | Track implemented, planned, sequential-control, and diversity-native generator backends | `artifacts/trajectory_exploration_backend_registry_v1/backend_registry.csv` |
| Corpus adequacy | Audit balance, leakage, coverage, degeneracy, and validity | `artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv` |
| Feature analysis | Check excitation, overlap, AUC, PCA, and class confusability as supporting diagnostics | `artifacts/feature_analysis_v1/feature_separation_scores.csv` |
| Evidence provider | Convert observations, features, or residuals into comparable evidence | `artifacts/classification_evidence_proof/evidence_provider_manifest.json` |
| Posterior updater | Apply priors and accumulate evidence into posterior histories | `artifacts/generic_inference_contract/posterior_history_schema.json` |
| State tracking | Maintain latent state, switching state, or particle-supported state summaries when the ladder needs more than feature-only evidence | `artifacts/advanced_state_inference_v1/posterior_history.csv` |
| Evaluation | Inspect separability, calibration, confusion, oracle gap, and prior sensitivity | `artifacts/monte_carlo_accumulator/calibration_bins.csv` |
| Promotion | Assign promote, revise, reject, or defer | `artifacts/validation_ladder/validation_ladder_decisions.csv` |

The algorithm side now has a parallel split:

- the proof ladder stays narrow and witness-gated
- the broader algorithm map tracks benchmark classifiers, neural sequence
  baselines, learned filters, uncertainty wrappers, and future tracking lanes
  without pretending they are all promoted

## Claim Discipline

Every claim should resolve to:

- one or more docs that state the rule
- one or more artifacts that show evidence
- tests that protect the contract
- a current limitation
- next work

The canonical index is [claim_evidence_matrix.md](claim_evidence_matrix.md).
