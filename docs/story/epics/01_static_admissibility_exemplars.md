# Epic 1 Static Admissibility Exemplars

Epic 1 should be presented as a reusable study-screening tool, not only a single packet. The exemplar suite under `experiments/static_admissibility/` shows the recurring feature/class/prior family patterns that the lane is meant to catch.

## Exemplar Matrix

| exemplar family | config | salient Epic 1 deliverables | input matrix signature | expected route |
| --- | --- | --- | --- | --- |
| promote separable family | `experiments/static_admissibility/repeatable_lane_demo/repeatable_lane_demo.yaml` | class separability, feature relevance, decisionability | balanced priors and well-separated feature rows with online-safe provenance | `promote_to_corpus_explorer` |
| class overlap boundary family | `experiments/static_admissibility/class_overlap_boundary_family/class_overlap_boundary_family.yaml` | confusability matrix, boundary diagnosis, decisionability | stationary and slow-velocity rows occupy nearly the same feature region | `revise_class_set` |
| prior domination family | `experiments/static_admissibility/prior_domination_family/prior_domination_family.yaml` | prior pathology surface, flip thresholds, decisionability | weak evidence range plus extreme rare-class prior skew | `revise_prior` |
| redundancy and synergy family | `experiments/static_admissibility/redundancy_synergy_family/redundancy_synergy_family.yaml` | feature redundancy graph, feature synergy map, feature relevance | one duplicated speed column and XOR-like companion features | `promote_to_corpus_explorer` with redundancy/synergy warnings |
| coverage thin-cells family | `experiments/static_admissibility/coverage_thin_cells_family/coverage_thin_cells_family.yaml` | coverage feasibility, decisionability | separable class rows but only two samples per class-feature cell | `promote_to_corpus_explorer` with coverage warning |
| leakage blocker family | `experiments/static_admissibility/leakage_blocker_family/leakage_blocker_family.yaml` | leakage provenance audit, hard gate, decisionability | one feature directly overlaps the label rule | `reject` |

## Multi-Domain 3D Teaching Exemplar

The optional MD3D packet is the strongest Epic 1 teaching artifact:

- packet: `artifacts/validation_packets/01_static_admissibility_multi_domain_3d/`
- command: `PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-multi-domain-3d --output-dir artifacts/validation_packets/01_static_admissibility_multi_domain_3d`

What it proves:

- many features do not automatically make a study admissible
- prior regimes can hide rare classes before classifier work
- unsupported maritime and space-like classes can be screened out statically
- blocked identity, catalog, generator, or future-window features can invalidate the study
- redundancy includes aliases, affine transforms, threshold subsumption, and decision redundancy

This packet is intentionally a normalized feature bundle, not a full 3D tracking implementation.

## Alias And Threshold Example

Epic 1 now includes a concrete near-threshold alias case:

- `min_altitude_ge_300m`
- `min_altitude_ge_301m`

The lane does not treat that pair as a simple correlation problem. It checks:

- same semantic quantity and aggregation
- nearby thresholds
- logical implication
- boundary-slice occupancy and class mix
- threshold gap relative to declared uncertainty
- incremental decision value

If the threshold gap is below declared uncertainty, retention remains candidate-level until ablation or observability follow-up confirms that the distinction is meaningful.

## What The Inputs Mean

- `samples.csv` is the concrete class-labeled feature matrix being screened.
- `feature_schema.csv` says whether a feature is online-safe, provenance-clean, or label-overlapping.
- `class_schema.csv` defines the intended class surface explicitly enough to expose overlap.
- `priors` in the YAML determine whether evidence can realistically flip the posterior.

## Epic 1 Takeaway

The big takeaway is not only that Epic 1 produces charts. It gives the user a repeatable bundle path for asking:

`What sort of study candidate is this feature/class/prior setup?`

The exemplar suite makes that concrete:

- admissible and ready for corpus search
- semantically overlapping and in need of class revision
- prior-dominated and in need of prior revision
- redundant or interaction-heavy and in need of feature follow-up
- thinly covered and in need of more witness rows
- blocked outright because the features leak the answer
