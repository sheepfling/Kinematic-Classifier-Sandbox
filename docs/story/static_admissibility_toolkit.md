# Product 1 Toolkit: Static Admissibility

Static Admissibility is the quick-turn front door for a new study. Give it a
class set, feature set, labeled samples, provenance metadata, and a prior
regime. It runs the static checks, writes a decision packet, and routes the
study toward revision, Corpus Explorer, or classifier evaluation.

The product name is **Static Admissibility**. “Static analysis” is a useful
implementation shorthand, but the product decision is narrower: is this
feature/class/prior study candidate meaningful and clean enough to pursue?

## The quick-turn path

The normal path is three commands. The prior is not a separate manual step;
the declared `priors` in `static_audit_bundle.yaml` are consumed by the same
audit that checks the features and classes.

```bash
# 1. Create a starter bundle.
PYTHONPATH=src python3 -m kinematic_classifier_sandbox init-static-audit-bundle \
  --output-dir work/static_admissibility/my_study

# 2. Edit the YAML and three CSV files, then run every static check.
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit \
  --bundle work/static_admissibility/my_study/static_audit_bundle.yaml \
  --output-dir artifacts/packets/my_study

# 3. Validate the generated packet.
PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet \
  artifacts/packets/my_study
```

The audit packet is self-contained: it copies the bundle inputs, writes the
machine-readable tables, renders the diagnostic figures, and records the
decision and next action.

## Input contract

| Input | Required content | Why it matters |
| --- | --- | --- |
| `static_audit_bundle.yaml` | `study_id`, `priors`, input file paths, and feature names | Declares the study and prior regime |
| `class_schema.csv` | `class_name`, optional notes | Defines the intended class surface |
| `feature_schema.csv` | `feature_name`, provenance tags, `online_available`, `label_rule_overlap` | Defines whether a feature is admissible evidence |
| `samples.csv` | `true_class`, numeric feature columns, optional `sample_id` | Supplies the labeled feature surface to inspect |
| optional `class_feature_signature.csv` | `class_name`, `feature_name`, optional expected mean/std and source | Records a prior expected signature for a declared class that has no current labeled samples |

Start from the [static audit bundle template](../../templates/static_audit_bundle.yaml)
and the companion [CSV templates](../../templates/static_audit_samples.csv), including the optional
[future-class signature template](../../templates/static_audit_class_feature_signature.csv).
The complete input rules and path behavior are in the [Static Audit Bundle
User Guide](../workflows/static_audit_bundle_user_guide.md).

## What the automatic audit answers

| Question | Automatic output | What to inspect | Current claim boundary |
| --- | --- | --- | --- |
| Which classes are most confusable? | `class_confusability_matrix.csv`, `static_audit_report.md`, `02c_class_pair_confusability_matrix.png` | Pairwise AUC, overlap coefficient, Mahalanobis distance, Jensen–Shannon divergence, and `hard`/`medium`/`pass` status | Pairwise, sample-backed confusability; not a final multiclass performance guarantee |
| Which class pair is the bottleneck? | `static_audit_report.md`, `decision_card.md` | The hardest pair and the `revise_class_set` or `revise_feature_set` route | The result is conditional on the declared samples and features |
| Do classes share exact or near-identical feature vectors? | `class_pair_diagnostics.csv`, `class_observability.csv` | Exact shared-vector count/rate, normalized signature distance, expected-signature collision status, and selection status | An exact collision bounds the collided observations; it does not prove the entire classes are globally identical |
| Which declared future classes are not currently selectable? | `class_observability.csv`, `class_feature_signature.csv` | `unobserved_class`, `cannot_validate_from_current_surface` or `unverified_expected_collision`, expected-signature coverage, and source | “Cannot validate” is the honest static result; no sample-backed tool can prove a class can never occur |
| Which features carry class information? | `feature_relevance_table.csv`, `02d_feature_relevance_rank.png` | Mutual information with class, maximum pairwise AUC, effect size, worst-pair overlap, missing rate, and recommended status | Relevance is not causal importance |
| Are features redundant or dependent? | `feature_redundancy_matrix.csv`, `02e_feature_redundancy_graph.png` | Spearman correlation, feature mutual information, and `high_redundancy` status | This is a dependence/redundancy screen, not a formal proof of statistical independence |
| Do weak features help jointly? | `feature_synergy_candidates.csv`, `02f_feature_synergy_map.png` | Joint mutual information, best single-feature mutual information, and pair gain | Synergy remains candidate-level until ablation confirms incremental value |
| Can the prior overwhelm the evidence? | `prior_regime.csv`, `prior_pathology_report.csv`, `prior_flip_thresholds.csv`, `02g`/`02h` figures | Prior odds, observed likelihood-ratio range, flip threshold, posterior collapse rate, and pathology flag | The current likelihood check is a Gaussian proxy and an admissibility warning, not calibration proof |
| Are some classes almost never selected under the declared prior? | `prior_selection_balance.csv`, `static_resolution_plan.csv` | Prior-weighted Gaussian proxy selection rate, own-class selection rate, selection-to-prior ratio, and `PRIOR_SELECTION_SKEW` recommendations | This is a static proxy for selection imbalance, not a deployed classifier confusion matrix |
| Is the sample surface sufficiently covered? | `static_coverage_feasibility.csv`, `02i_static_coverage_feasibility.png` | Per-class/per-feature sample counts, occupied bins, empty-bin rate, and low-count status | Coverage of the supplied samples is not proof of operational corpus coverage |
| Is any evidence invalid or leaky? | `static_leakage_provenance_audit.csv`, `02j_static_leakage_provenance_audit.png` | Online availability, label-rule overlap, future dependency, metadata leakage, and blocker status | Leakage blockers cannot promote |
| What should happen next? | `decision_card.md`, `static_audit_decision_card.md`, `02k_static_audit_to_action_router.png` | Decision status, blockers, warnings, and next-work actions | Routing is a study decision, not a classifier result |
| How should a detected issue be resolved? | `static_resolution_plan.csv` | Programmatic issue code, severity, affected scope, recommended action, verification step, and route | Recommendations are bounded follow-up guidance; they do not automatically change the study |

## Additional static-analysis categories

The repository also has a richer multi-domain static packet that demonstrates
the next level of admissibility analysis. These categories are real prototypes,
but they are not all part of the normal generic four-file bundle command yet.

| Category | Current support | What it answers |
| --- | --- | --- |
| Class-feature excitation and observability | Generic bundle: `class_feature_signature.csv`, `class_observability.csv` | Does each declared class have a usable feature signature, or is it unobserved/bounded by an exact or near collision? |
| Unsupported/future-set classes | Generic bundle with `allow_unobserved_classes: true` and optional `class_feature_signature.csv` | Which declared future classes have no current labeled evidence, and which expected feature signatures remain unverified? |
| Exact and semantic feature aliases | Generic bundle: `feature_alias_candidates.csv`; multi-domain packet adds richer schema heuristics | Are two features duplicates, affine aliases, semantic near-duplicates, or threshold aliases? |
| Threshold subsumption | Generic bundle supports declared threshold metadata; multi-domain packet adds boundary-slice reporting | Do two threshold features differ only by a gap smaller than measurement resolution or boundary-slice evidence? |
| Functional equivalence | Multi-domain teaching packet | Do two features behave nearly the same over the declared sample surface even when their names differ? |
| Decision redundancy | Multi-domain teaching packet | Does a feature add class evidence after a neighboring feature is already present? |
| Estimator/sample-size reliability | Multi-domain teaching packet | Are metric uncertainty, sample counts, and proxy bounds visible before trusting a static score? |

The multi-domain packet is generated with:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox \
  run-static-audit-multi-domain-3d \
  --output-dir artifacts/validation_packets/01_static_admissibility_multi_domain_3d
```

It is a normalized feature-bundle audit, not a raw 3D tracker. Its key source
tables include `multi_domain_3d_observability_gaps.csv`,
`feature_alias_candidates.csv`, `feature_threshold_subsumption.csv`,
`feature_functional_equivalence.csv`, and `feature_decision_redundancy.csv`.

## Important gaps before calling the generic tool complete

The standard quick-turn bundle still needs the following for a complete
admissibility claim:

1. A richer functional-equivalence and boundary-slice report for threshold
   features.
2. Ablation-backed conditional utility so decision redundancy and synergy are
   confirmed rather than treated as proxies.
3. Regime/invariance checks that ask whether a class remains observable across
   sensor, duration, noise, or operating-condition slices.

Those are the main additions needed to make the generic tool a genuine
“future class set + feature set + priors → static admissibility answer” tool.

## Decision routes

The packet’s decision field is the fast answer for a new study:

| Route | Meaning | Typical action |
| --- | --- | --- |
| `promote_to_corpus_explorer` | The declared study is admissible enough to search for useful cases | Build or select a governed corpus |
| `promote_with_warnings` | The study can move forward with explicit limitations | Carry warnings into corpus/classifier gates |
| `revise_class_set` | One or more class pairs are too overlapping or not decisionable | Tighten class definitions or split/merge classes |
| `revise_feature_set` | The proposed features are weak, unsupported, or unavailable online | Add or replace features and rerun |
| `revise_prior` | Prior odds dominate the available evidence | Revise the prior regime or require a prior sweep |
| `reject` | Leakage or another hard blocker invalidates the study | Remove the blocker before rerunning |

## Tool links

For a lead-facing walkthrough, start with the [Product 1 Pitch Brief](static_admissibility_pitch_brief.md).

### Run the product

- [CLI entrypoint and commands](../../src/kinematic_classifier_sandbox/__main__.py)
- [Static audit bundle loader](../../src/kinematic_classifier_sandbox/static_admissibility/study_bundle.py)
- [Future-class signature template](../../templates/static_audit_class_feature_signature.csv)
- [Static audit runner and packet writer](../../src/kinematic_classifier_sandbox/static_admissibility/io.py)
- [Static audit analysis engine](../../src/kinematic_classifier_sandbox/analysis/static_feature_class_prior_audit.py)
- [Packet validator](../../src/kinematic_classifier_sandbox/static_admissibility/validation.py)
- [Exemplar suite builder](../../src/kinematic_classifier_sandbox/static_admissibility/exemplar_suite.py)
- Generated suite report: `executive_brief.md`

### Use known-good examples

- [Repeatable positive bundle](../../experiments/static_admissibility/repeatable_lane_demo/repeatable_lane_demo.yaml)
- [Class-overlap boundary bundle](../../experiments/static_admissibility/class_overlap_boundary_family/class_overlap_boundary_family.yaml)
- [Prior-domination bundle](../../experiments/static_admissibility/prior_domination_family/prior_domination_family.yaml)
- [Redundancy/synergy bundle](../../experiments/static_admissibility/redundancy_synergy_family/redundancy_synergy_family.yaml)
- [Thin-coverage bundle](../../experiments/static_admissibility/coverage_thin_cells_family/coverage_thin_cells_family.yaml)
- [Leakage-blocker bundle](../../experiments/static_admissibility/leakage_blocker_family/leakage_blocker_family.yaml)
- [Full Epic 1 exemplar suite](../../experiments/static_admissibility/epic1_exemplar_suite.yaml)

Run the full teaching suite with:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-suite \
  experiments/static_admissibility/epic1_exemplar_suite.yaml \
  --output-dir artifacts/validation_packets/01_static_admissibility
```

### Protect the product

- [Static packet tests](../../tests/static_admissibility/test_static_admissibility_packet.py)
- [Static analysis tests](../../tests/analysis/test_static_feature_class_prior_audit.py)
- [Alias and redundancy tests](../../tests/static_admissibility/test_feature_alias_redundancy.py)
- [Exemplar-suite tests](../../tests/static_admissibility/test_static_admissibility_exemplar_suite.py)
- [Claim/evidence matrix](claim_evidence_matrix.md)
- [Static Admissibility product specification](epics/01_static_admissibility.md)

Run the focused test lane with:

```bash
PYTHONPATH=src python3 scripts/test.py --lane static
```

## What “done” means for a new study

A new feature/class/prior set has completed the Static Admissibility turn when:

1. The bundle is portable and its provenance fields are filled in.
2. The audit packet validates without missing or inconsistent artifacts.
3. The class confusability output identifies the hardest pairs.
4. Exact/near collision and class observability outputs have been reviewed,
   including any declared future classes.
5. Feature relevance, alias, and dependence/redundancy outputs have been reviewed.
6. Prior pathology and flip thresholds have been reviewed for the intended
   prior regime.
7. Leakage blockers and online-availability warnings have been resolved or
   explicitly accepted.
8. The decision route is recorded and its next action is handed to Corpus
   Explorer or the classifier ladder.

For the current repository-level product map, see [Three Products](three_products.md).
