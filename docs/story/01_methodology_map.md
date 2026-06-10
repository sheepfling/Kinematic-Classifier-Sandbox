# Methodology Map

The methodology is a compositional flow from study intent to decision.

```text
Corpus Objective
  -> Backend / Generator / CorpusGym / Search / QD
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
| Corpus objective | State the intended coverage, stress, backend, and validity target | `artifacts/corpus_objectives/objective_validation_report.md` |
| Corpus explorer | Generate, search, score, and select trajectories | `artifacts/generic_corpus_exploration/candidate_scores.csv` |
| Corpus adequacy | Audit balance, leakage, coverage, degeneracy, and validity | `artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv` |
| Feature analysis | Check excitation, overlap, AUC, PCA, and class confusability | `artifacts/feature_analysis_v1/feature_separation_scores.csv` |
| Evidence provider | Convert observations, features, or residuals into comparable evidence | `artifacts/classification_evidence_proof/evidence_provider_manifest.json` |
| Posterior updater | Apply priors and accumulate evidence into posterior histories | `artifacts/generic_inference_contract/posterior_history_schema.json` |
| State tracking | Maintain latent state, switching state, or particle-supported state summaries when the ladder needs more than feature-only evidence | `artifacts/advanced_state_inference_v1/posterior_history.csv` |
| Evaluation | Inspect separability, calibration, confusion, oracle gap, and prior sensitivity | `artifacts/monte_carlo_accumulator/calibration_bins.csv` |
| Promotion | Assign promote, revise, reject, or defer | `artifacts/validation_ladder/validation_ladder_decisions.csv` |

## Claim Discipline

Every claim should resolve to:

- one or more docs that state the rule
- one or more artifacts that show evidence
- tests that protect the contract
- a current limitation
- next work

The canonical index is [claim_evidence_matrix.md](claim_evidence_matrix.md).
