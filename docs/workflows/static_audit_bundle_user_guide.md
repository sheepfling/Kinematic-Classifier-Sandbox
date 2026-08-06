# Static Audit Bundle Guide

The static admissibility lane now supports a file-backed study bundle so new feature/class/prior sets can be ingested without writing Python.

For the complete Product 1 command and diagnostic map, see the [Static Admissibility Toolkit](../story/static_admissibility_toolkit.md).

## Fast Start

Create a starter bundle from the built-in templates:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox init-static-audit-bundle \
  --output-dir my_static_audit_bundle
```

## Canonical Bundle Format

Place these files together in one directory:

- `static_audit_bundle.yaml`
- `class_schema.csv`
- `feature_schema.csv`
- `samples.csv`
- optional `class_feature_signature.csv` for declared future classes

The YAML file points at the CSV files and declares priors. Relative paths are resolved from the YAML location.

## Template

Start from:

- `templates/static_audit_bundle.yaml`
- `templates/static_audit_class_schema.csv`
- `templates/static_audit_feature_schema.csv`
- `templates/static_audit_samples.csv`
- `templates/static_audit_class_feature_signature.csv`

## Example

The canonical repeatable demo lives at:

- `experiments/static_admissibility/repeatable_lane_demo/repeatable_lane_demo.yaml`

The Epic 1 exemplar family suite lives at:

- `experiments/static_admissibility/epic1_exemplar_suite.yaml`
- `docs/story/epics/01_static_admissibility_exemplars.md`

Run it with:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit \
  --bundle experiments/static_admissibility/repeatable_lane_demo/repeatable_lane_demo.yaml \
  --output-dir artifacts/packets/repeatable_lane_demo
```

Validate the packet with:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet \
  artifacts/packets/repeatable_lane_demo
```

## CSV Rules

- `class_schema.csv` must contain `class_name`.
- `feature_schema.csv` must contain `feature_name`.
- `samples.csv` must contain `true_class` plus one numeric column per feature.
- `sample_id` is optional but recommended.
- `label_rule_overlap` and `online_available` accept `true/false`, `yes/no`, or `1/0`.
- `provenance_tags` may be comma-separated or pipe-separated.
- `dimension` in the YAML is descriptive metadata; the audit accepts arbitrary
  feature-vector dimension rather than assuming 1D, 2D, or 3D.
- Optional feature metadata includes `semantic_group`, `units`, `aggregation`,
  `dimension`, `measurement_resolution`, `threshold_operator`, and
  `threshold_value`. These enable generic alias and threshold checks.
- To keep a future class in the declared set before samples exist, set
  `input_bundle.allow_unobserved_classes: true`. You may then point
  `input_bundle.class_feature_signature` at a CSV with `class_name`,
  `feature_name`, optional `expected_mean`/`expected_std`, and `source`.
- An unobserved class is reported as `unobserved_class` and blocks promotion;
  an expected signature is recorded as unverified evidence, not treated as a
  substitute for labeled samples.

## Path Resolution

- Relative bundle paths resolve from the YAML file location.
- The packet copies the YAML and CSV inputs into the output so the run preserves provenance.
- The bundle format is designed to stay portable; do not encode local absolute paths in the YAML.

## Interpreting Routes

- `promote_to_corpus_explorer`: the study is worth pushing forward
- `promote_with_warnings`: the study is usable but has scoped cautions
- `revise_feature_set`: the declared evidence is insufficient or unsupported
- `revise_class_set`: the class surface is overlapping or not decisionable
- `revise_prior`: the prior regime dominates achievable evidence
- `reject`: leakage or a hard blocker invalidates the study

## Packet Provenance

When the lane runs from a study bundle, the output packet copies the source inputs as:

- `study_bundle_source.yaml`
- `study_bundle_samples.csv`
- `study_bundle_feature_schema.csv`
- `study_bundle_class_schema.csv`
- `study_bundle_class_feature_signature.csv` when supplied

The generated packet also includes generic collision and alias tables:

- `class_pair_diagnostics.csv`
- `class_feature_signature.csv`
- `class_observability.csv`
- `feature_alias_candidates.csv`
