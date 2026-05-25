# PLN-025 Advanced State Inference And IMM Lift

Title: Advanced State Inference And IMM Lift
Plan ID: PLN-025
Status: done
Owner: @rick
Priority: P1
Last Updated: 2026-05-25

Objective:
Implement an abstract advanced-filter backend contract and a 1D IMM proof so the repo can evaluate switching trajectories with a state/evidence/diagnostic surface that plugs into the existing classifier ladder. The implementation should be future-ready for a 3D PVA lift, but it should not claim PF/RBPF capability until there is failure evidence that justifies those rungs.

Scope:
- Define a shared advanced-filter contract that emits state summaries, evidence summaries, posterior-compatible likelihood rows, and diagnostics.
- Implement a 1D IMM witness backend over the current switching trajectories.
- Compare IMM behavior against the existing transition-matrix accumulator and Kalman-bank style baselines where the same witness supports that comparison.
- Keep the state-space abstraction compatible with a future 3D PVA lift by using a generic state/evidence contract rather than a 1D-only shape.
- Add artifact bundles, tests, and documentation that explain how to read the outputs.

Out of Scope:
- Particle filtering, Rao-Blackwellized particle filtering, or other sampling-based nonlinear inference unless later failure evidence justifies them.
- External 3D engine integration.
- Reworking the existing transition-matrix accumulator or Kalman bank implementations unless a concrete contract mismatch is discovered.
- Claiming the IMM backend is the final advanced-filter answer for all trajectory families.

Implementation Steps:
1. Add the advanced-filter contract layer.
   - Define a shared filter backend contract for:
     - initialize
     - predict
     - update
     - state summary
     - evidence summary
     - diagnostics
     - history
   - Define output schemas for state rows, evidence rows, posterior rows, and diagnostics rows.
   - Make the schema generic enough to support future 3D PVA state vectors.
2. Implement a 1D IMM backend.
   - Use the existing switching witness generator as the proof scenario.
   - Run one linear-Gaussian mode filter per mode.
   - Add mode mixing, mode posteriors, innovation likelihoods, and combined state estimates.
   - Emit the shared contract rows plus IMM-specific diagnostics such as mixing probabilities and switch-delay summaries.
3. Add the 1D evaluation artifact bundle.
   - Produce CSVs, a config file, a report, and plots for the IMM run.
   - Include a comparison table against the current transition-matrix rung.
4. Document the 3D lift path.
   - Explain how the same contract lifts from scalar 1D witness trajectories to 3D PVA state vectors.
   - State what would change for 3D and what stays invariant.

Validation:
- The advanced-filter contract validates as a reusable schema.
- IMM mode probabilities and mixing probabilities normalize correctly.
- IMM emits posterior-compatible evidence rows and shared diagnostics rows.
- The IMM run completes on the existing 1D switching witnesses.
- The report shows where IMM is stronger or weaker than the current transition-matrix rung.
- The contract can describe a future 3D PVA backend without changing the downstream evaluation surface.

Artifacts / Config:
- `src/kinematic_classifier_sandbox/advanced_state_inference.py`
- `tests/test_advanced_state_inference.py`
- `artifacts/advanced_state_inference_v1/`
- `artifacts/advanced_state_inference_contract/`
- `docs/plans/PLN-025_advanced_state_inference_imm_1d_to_3d.md`
- updated `src/kinematic_classifier_sandbox/__init__.py`
- updated `scripts/export_artifacts.py`

Dependencies:
- `PLN-013` generic filtering contract.
- `PLN-016` team-facing methodology showcase.
- `PLN-023` math document hardening and equation traceability.
- `transition_matrix_accumulator.py` for the switching witness baseline.
- `kalman_filter_bank.py` for the model-based 1D reference surface.
- `trajectory_generator.py` for the switching witness trajectories.

Milestones:
- `M55`: Advanced filter contract.
  - Deliverables:
    - `artifacts/advanced_state_inference_contract/filter_backend_contract.json`
    - `artifacts/advanced_state_inference_contract/advanced_filter_output_schema.json`
    - `artifacts/advanced_state_inference_contract/diagnostics_schema.json`
    - `artifacts/advanced_state_inference_contract/contract_report.md`
  - Exit criterion:
    - The repo has a shared schema for advanced-filter state, evidence, and diagnostics rows.
- `M56`: IMM v1 for 1D switching witnesses.
  - Deliverables:
    - `artifacts/advanced_state_inference_v1/imm_config.yaml`
    - `artifacts/advanced_state_inference_v1/mode_probability_history.csv`
    - `artifacts/advanced_state_inference_v1/mixing_probability_history.csv`
    - `artifacts/advanced_state_inference_v1/mode_likelihood_history.csv`
    - `artifacts/advanced_state_inference_v1/state_estimate_history.csv`
    - `artifacts/advanced_state_inference_v1/posterior_history.csv`
    - `artifacts/advanced_state_inference_v1/imm_report.md`
  - Exit criterion:
    - A 1D switching witness can be evaluated end to end with an IMM backend that emits the shared contract rows.
- `M57`: 3D PVA lift notes.
  - Deliverables:
    - `docs/story/advanced_state_inference_1d_to_3d.md`
    - updated advanced-filter math references where needed
  - Exit criterion:
    - The contract and report explain how the same backend extends to 3D PVA without changing the downstream evaluation contract.

Success Criteria:
- The advanced filter backend is abstract enough to support future PF/RBPF or 3D variants without changing the evaluation surface.
- The 1D IMM proof is tied to a real switching witness and produces inspectable evidence rows.
- The repo can explain when IMM is justified, what it measures, and what the 3D lift path is.
