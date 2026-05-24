# PLN-009 M11 Feature-Set And Class-Pair Comparison

Title: M11 Feature-Set And Class-Pair Comparison
Plan ID: PLN-009
Status: done
Owner: @codex
Priority: P1
Objective: Extend the common experiment harness so feature-set studies and class-pair studies are emitted as first-class executable artifacts, and so `M12` through `M17` can treat feature bundles and hard boundaries as explicit methodology-level study dimensions rather than hidden byproducts of the broader `M10` run surface.
Scope:
- Add a manifest-driven feature-set comparison study to the common harness.
- Add explicit class-pair duration and scenario/noise comparison artifacts.
- Extend the common experiment report and artifact bundle to surface these comparisons.
- Keep the work within the current `common_1d_classifier_study` while making the output contract reusable.
- Produce outputs that the generic inference contract, feature taxonomy, and corpus adequacy plans can consume directly.
Out of Scope:
- Adding a second study domain.
- Building a real 3D study.
- Replacing all current classifier families with fully generic manifest-instantiated executors.
- Implementing PF, RBPF, or IMM-specific studies.
Implementation Steps:
1. Add feature-set comparison rows that evaluate the declared feature bundles on the executable pair corpus.
2. Add class-pair duration slices from posterior-history prefixes.
3. Add class-pair scenario/noise slices from the executable pair predictions.
4. Write the new CSV artifacts and extend the markdown report.
5. Add regression tests for the new `M11` surfaces.
6. Ensure outputs are shaped so later plans can reuse them.
   - Preserve stable feature-set ids, class-pair ids, scenario-family ids, and classifier ids.
   - Keep the pairwise output family suitable for later dimensional-lift and evidence-provider audits.
Validation:
- `python3 -m pytest tests/test_common_experiment_harness.py -q`
- `python3 -m pytest -q`
- Smoke run via `python3 scripts/run_study.py ...`
Artifacts / Config:
- `src/kinematic_classifier_sandbox/common_experiment_harness.py`
- `tests/test_common_experiment_harness.py`
- `experiments/common_1d_classifier_study/common_experiment_config.yaml`
- `docs/plans/PLN-002_kinematic_classification_roadmap.md`
Dependencies:
- Existing `M10` common experiment harness
- `feature_sets.json`
- `class_pair_manifest.json`
- executable pair generator and posterior-history surface
- follow-on generic methodology proof milestones in `PLN-010` through `PLN-015`
Last Updated: 2026-05-24
