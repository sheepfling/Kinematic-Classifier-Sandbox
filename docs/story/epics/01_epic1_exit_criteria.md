# Epic 1 Exit Criteria

Epic 1 is the point where the project stops being a presentation generator and
becomes a usable methodology workbench with a presentation export profile.

## Name

Epic 1: Evidence-Tiered Kinematic Classification Workbench MVP

## Mission

Build a single-repo workflow that can declare a kinematic-classification study,
run the evidence/posterior workflow, emit diagnostics and decision cards, and
export a public-safe presentation packet from the same artifact spine.

## Completion Gates

### Gate A: Presentation Packet Readiness

The public packet must include the main deck, appendix, decision card, speaker
script, hero chart manifest, lane proof matrix, contact sheets, and a packet
readme. Every chart needs an evidence tier, source artifact, and claim boundary.

The packet must not leak local paths. It must not label PF/RBPF, CEM, or PPO as
generally promoted without the evidence required by `docs/story/claim_registry.yaml`.

### Gate B: Workbench Usability

A user must be able to run the workbench without touching presentation code:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-study \
  experiments/common_1d_classifier_study/common_experiment_config.yaml

PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-study \
  experiments/common_1d_classifier_study/common_experiment_config.yaml \
  --output-dir artifacts/runs/interview_demo

PYTHONPATH=src python3 -m kinematic_classifier_sandbox analyze-run \
  --run-dir artifacts/runs/interview_demo

PYTHONPATH=src python3 -m kinematic_classifier_sandbox export-packet \
  --profile workbench \
  --run-dir artifacts/runs/interview_demo \
  --output-dir artifacts/workbench_reports/interview_demo
```

The standard run directory must include:

- `study_spec.yaml`
- `study_run_manifest.json`
- `corpus_manifest.json`
- `selected_corpus_manifest.json`
- `evidence_contract.json`
- `posterior_history.csv`
- `metrics_by_method.csv`
- `rung_sufficiency.csv`
- `prior_sensitivity.csv`
- `calibration_metrics.csv`
- `oracle_gap.csv`
- `confusion_localization.csv`
- `leakage_adequacy_audit.csv`
- `backend_capability_matrix.csv`
- `decision_card.md`
- `decision_card.json`
- `workbench_report.md`

### Gate C: Claim Governance

The project must keep claims aligned with evidence. The authorities are:

- `docs/story/claim_registry.yaml`
- `docs/story/visualization_registry.yaml`
- the generated `decision_card.md`
- packet validators

Candidate and experimental evidence can appear in the packet, but inclusion in a
slide does not promote the method. CEM/PPO remain search-backend candidates
unless baseline comparison and downstream diagnostic yield support a stronger
claim. PF/RBPF remain witness-specific unless a named run-backed witness clears
the relevant failure-mode gate.

## Scope Boundary

Epic 1 proves the workbench workflow and public presentation export. It does not
prove a full 3D operational tracker, real sensor integration, general PF/RBPF
superiority, general CEM/PPO superiority, or deployment-ready classification.

## Closing Sentence

Epic 1 delivers a real kinematic-classification workbench with a public
presentation export: one study contract, one evidence/posterior pipeline, one
decision-card authority, and strict evidence tiers for every claim.
