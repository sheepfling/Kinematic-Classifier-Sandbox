# PLN-006 M9 Generator Completion

Title: Finish Remaining M9 Generator and Scenario Coverage Gaps
Plan ID: PLN-006
Status: done
Owner: @codex
Priority: P1
Objective: Complete the remaining M9 scope so the synthetic trajectory generator stack satisfies the roadmap definition beyond the current five-tier foundation.
Scope:
- Close the gap between the current generator foundation and the roadmap’s full M9 scenario coverage.
- Add explicit scenario families for short-horizon stressors, sweep-style perturbations, and switching-mode tracks.
- Add tests and reports that make these scenario families auditable.
Out of Scope:
- M10 common experiment harness work.
- Transition-matrix accumulation logic from M16.
- Advanced-filter decision gating from M17.
Implementation Steps:
1. Audit the current generator outputs against the M9 roadmap bullets.
2. Add explicit short-horizon and sweep-style scenario coverage where only implicit tier coverage exists today.
3. Add switching scenarios such as `stationary_then_moving`, `constant_velocity_then_braking`, and `constant_velocity_then_maneuver`.
4. Add generator-side coverage artifacts or manifests that show these scenarios are present and reproducible.
5. Add validation tests for the new scenario families and their metadata.
Validation:
- Unit tests for new generator scenarios and manifests.
- Full-suite regression run with `python3 -m pytest -q`.
- Manual artifact check confirming the new scenario families appear in generator outputs.
Artifacts / Config:
- `src/kinematic_classifier_sandbox/trajectory_generator.py`
- `tests/test_trajectory_generator.py`
- `artifacts/trajectory_generator_v1/`
- `docs/plans/PLN-002_kinematic_classification_roadmap.md`
Dependencies:
- Existing class-generating models and dataset tiers.
- Current adequacy and coverage reporting so new scenarios can be inspected after generation.
Last Updated: 2026-05-23

Completion Notes:
- Added supplemental generator scenario libraries for `short_horizon_v1`, `perturbation_sweeps_v1`, and `switching_scenarios_v1`.
- Extended `trajectory_generator_v1` artifacts to emit manifests plus generated-trajectory and true-state CSVs for those scenario families.
- Added generator tests covering the new scenario families and verified the full suite passes.
