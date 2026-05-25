# PLN-026 Rung Sufficiency And Escalation Evaluator

Title: Rung Sufficiency And Escalation Evaluator
Plan ID: PLN-026
Status: in_progress
Owner: @rick
Priority: P1
Last Updated: 2026-05-25

Objective:
Implement a formal rung sufficiency evaluator that decides whether the current classifier rung is sufficient for a validated corpus + feature set + class set + prior study, or whether escalation to a stronger rung is justified. The evaluator must distinguish corpus limitations, feature limitations, prior limitations, calibration limitations, switching limitations, nonlinear limitations, non-Gaussian limitations, and latent-event-timing limitations.

Scope:
- Add a dedicated rung-sufficiency package under `src/kinematic_classifier_sandbox/rung_sufficiency/`.
- Reuse the existing corpus adequacy, validation ladder, common-experiment, and advanced-filter evidence surfaces instead of inventing a parallel evaluation stack.
- Define a capability matrix for the current and future rungs: pointwise, windowed, sequential Bayes, Kalman bank, transition matrix, IMM, PF, and RBPF.
- Compute corpus preconditions, oracle gap, posterior quality, failure-mode diagnosis, and promotion decision summaries.
- Write artifact bundles, reports, and plots that make the evaluator inspectable on a per-study basis.

Out of Scope:
- Implementing PF or RBPF as promoted rungs before the evaluator shows a failure mode that justifies them.
- Reworking the existing corpus adequacy audit or validation ladder beyond the minimum integration needed for sufficiency scoring.
- Adding new data-generating backends or 3D trajectory engines as part of this plan.
- Replacing the existing validation ladder; this plan adds a governance layer above it.

Implementation Steps:
1. Define the rung capability matrix and shared evaluator contracts.
   - Add a canonical capability mapping for pointwise, windowed, sequential Bayes, Kalman bank, transition matrix, IMM, PF, and RBPF.
   - Define result dataclasses for corpus preconditions, oracle gap, posterior quality, failure diagnosis, and promotion decisions.
2. Reuse the existing evidence stack.
   - Pull corpus adequacy from the corpus audit.
   - Pull current-rung performance, posterior quality, prior sensitivity, and robustness from the validation ladder.
   - Pull oracle and learnability evidence from the common experiment and feature identifiability rows.
   - Pull switching evidence from the transition benchmark and the IMM lift artifact where applicable.
3. Implement failure-mode diagnosis and escalation rules.
   - Distinguish corpus-limited, feature-limited, prior-limited, calibration-limited, switching-limited, nonlinear-limited, non-Gaussian-limited, and latent-event-timing-limited cases.
   - Escalate only when the corpus gates pass, the current rung has a diagnosable failure, the next rung capability matches that failure, and the measured improvement clears thresholds.
4. Write artifacts and plots.
   - Emit CSV summaries for preconditions, oracle gap, posterior quality, failure modes, and promotion decisions.
   - Emit a markdown report and supporting plots that show score vs oracle, oracle gap by pair, failure mode heatmap, promotion matrix, and posterior quality by rung.
5. Add tests and wire exports.
   - Add unit tests for capability ordering, corpus gate blocking, oracle-gap interpretation, failure diagnosis, and promotion decisions.
   - Export the new artifacts through the standard artifact writer path and package namespace.

Validation:
- Corpus failures block escalation before algorithm blame is assigned.
- Oracle/feature failures are distinguished from model failures.
- Posterior quality metrics are reported per rung and study.
- Switching failures can justify transition/IMM escalation only when measured improvement is present.
- Capability mismatches reject escalation.
- The evaluator emits reproducible CSV, markdown, and plot artifacts.

Artifacts / Config:
- `src/kinematic_classifier_sandbox/rung_sufficiency/`
- `tests/test_rung_sufficiency.py`
- `tests/test_rung_failure_diagnosis.py`
- `tests/test_rung_promotion_decision.py`
- `artifacts/rung_sufficiency/`
- `experiments/rung_sufficiency/rung_sufficiency_config.yaml`
- updated `src/kinematic_classifier_sandbox/__init__.py`
- updated `scripts/export_artifacts.py`

Dependencies:
- `PLN-021` objective-driven corpus explorer v1.
- `PLN-022` corpus explorer hardening.
- `PLN-023` math document hardening and equation traceability.
- `PLN-024` repo story coherence and canonical navigation.
- `PLN-025` advanced state inference and IMM lift.
- `validation_ladder.py`
- `corpus_adequacy_audit.py`
- `common_experiment_harness.py`
- `transition_matrix_accumulator.py`
- `advanced_state_inference.py`

Milestones:
- `M62`: Rung capability matrix.
  - Deliverables:
    - `artifacts/rung_sufficiency/rung_capability_matrix.csv`
  - Exit criterion:
    - The evaluator has a canonical rung-to-capability map and next-rung ordering.
- `M63`: Corpus and feature precondition gates.
  - Deliverables:
    - `artifacts/rung_sufficiency/corpus_precondition_report.csv`
  - Exit criterion:
    - Corpus failures are identified before algorithm escalation is considered.
- `M64`: Oracle gap and practical limit estimator.
  - Deliverables:
    - `artifacts/rung_sufficiency/oracle_gap_report.csv`
  - Exit criterion:
    - The evaluator can say whether the current rung is close to the practical feature/oracle limit.
- `M65`: Posterior quality gate.
  - Deliverables:
    - `artifacts/rung_sufficiency/posterior_quality_by_rung.csv`
  - Exit criterion:
    - The evaluator reports accuracy, calibration, confidence, and prior sensitivity by rung.
- `M66`: Failure mode diagnosis.
  - Deliverables:
    - `artifacts/rung_sufficiency/failure_mode_diagnosis.csv`
  - Exit criterion:
    - The evaluator distinguishes corpus, feature, prior, calibration, switching, nonlinear, and non-Gaussian failure modes.
- `M67`: Promotion decision engine.
  - Deliverables:
    - `artifacts/rung_sufficiency/rung_promotion_matrix.csv`
  - Exit criterion:
    - The evaluator emits a promote / revise / reject / defer decision that is tied to evidence, not model novelty.
- `M68`: Report and plots.
  - Deliverables:
    - `artifacts/rung_sufficiency/rung_sufficiency_report.md`
    - `artifacts/rung_sufficiency/plots/`
  - Exit criterion:
    - The evaluator can be inspected visually and narratively without opening the implementation.

Success Criteria:
- A study can be classified as sufficient or insufficient for specific reasons, not just by raw accuracy.
- The evaluator separates “fix the corpus” from “fix the model”.
- IMM, PF, and RBPF are only justified when the diagnosed failure matches their added capability.
- The resulting artifacts are compatible with the repo’s existing promotion and evidence stack.
