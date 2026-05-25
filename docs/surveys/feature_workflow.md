# Feature Workflow

This note describes the current feature-analysis workflow in the sandbox, with emphasis on:

- where features are defined
- how feature sets are selected
- how to rerun feature analysis and PCA
- how to add a new feature with minimal code changes

## Current Model

The generic feature-analysis path is centered on:

- [feature_analysis.py](src/kinematic_classifier_sandbox/feature_analysis.py)
- [pca_analysis.py](src/kinematic_classifier_sandbox/pca_analysis.py)
- [feature_sets.json](experiments/common_1d_classifier_study/feature_sets.json)
- [class_pair_manifest.json](experiments/common_1d_classifier_study/class_pair_manifest.json)

The important abstraction is the feature registry in `feature_analysis.py`:

- `BaseFeatureComputationContext`
- `OneDimensionalFeatureComputationContext`
- `FeatureSpec`
- `FEATURE_REGISTRY`

Each feature is now declared once with:

- `name`
- `group`
- `description`
- `default_excitation_thresholds`
- `extractor`

`FeatureRow` now stores:

- stable metadata
- a dynamic `feature_values` map

That means the analysis pipeline no longer depends on a fixed hard-coded numeric schema for every feature.

The context layer is now split deliberately:

- `BaseFeatureComputationContext`
  - generic trajectory metadata and observed time-series structure
  - reusable for future non-1D feature families
- `OneDimensionalFeatureComputationContext`
  - current 1D-derived signals such as sign changes, monotonicity, and fit residuals
  - the current `FEATURE_REGISTRY` is intentionally scoped to this 1D context

`FeatureComputationContext` remains as a backward-compatible alias to the 1D context for the current codebase. The practical meaning is: the feature pipeline is generic, but the shipped engineered features are still explicitly 1D.

## Default Rerun Path

From repo root:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/export_artifacts.py
```

That regenerates the default artifact set, including:

- `artifacts/abstract_inspection_v1/`
- `artifacts/feature_analysis_v1/`
- `artifacts/pca_analysis_v1/`
- `artifacts/corpus_adequacy_audit_v1/`

If you only want the abstract feature-space inspection outputs, use:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/run_abstract_inspection.py
```

That regenerates the focused bundle without rerunning the entire artifact tree.

If you want the short terminal recommendation view after that, use:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/inspect_abstract_recommendations.py
```

That prints:

- the currently recommended feature set
- its adequacy/separability summary
- the currently hardest class boundary

## Programmatic Use

Use the package entrypoints directly when you want a specific feature set or custom subset.

### Named Feature Set

```python
from kinematic_classifier_sandbox import analyze_feature_datasets, analyze_feature_pca

feature_result = analyze_feature_datasets(feature_set="shape_window")
pca_result = analyze_feature_pca(feature_set="shape_window", n_components=2)
```

### Explicit Custom Subset

```python
from kinematic_classifier_sandbox import analyze_feature_datasets

feature_result = analyze_feature_datasets(
    feature_names=("position_range", "speed_range", "linear_fit_residual"),
)
```

### Discover Available Sets

```python
from kinematic_classifier_sandbox import load_feature_set_manifest

manifest = load_feature_set_manifest()
print(sorted(manifest))
```

### Inspect the Registry

```python
from kinematic_classifier_sandbox import load_feature_registry

registry = load_feature_registry()
spec = registry["speed_range"]
print(spec.group)
print(spec.description)
print(spec.default_excitation_thresholds)
```

### Generate The Full Abstract Inspection Bundle

```python
from kinematic_classifier_sandbox import write_abstract_inspection_artifacts

artifacts = write_abstract_inspection_artifacts("artifacts")
print(artifacts.index_path)
```

That bundle writes:

- the full-set feature analysis and PCA runs
- the named feature-set runs declared in `feature_sets.json`
- corpus adequacy artifacts
- coverage report artifacts
- one index markdown file that links the inspection outputs together
- one machine-readable summary JSON for downstream scripts

## How To Add A New Feature

The current lowest-friction path is:

1. Add the feature to `FEATURE_REGISTRY` in [feature_analysis.py](src/kinematic_classifier_sandbox/feature_analysis.py).
2. Provide:
   - `name`
   - `group`
   - `description`
   - `default_excitation_thresholds`
   - `extractor=lambda context: ...`
3. If the extractor depends only on generic trajectory structure, add that ingredient to `BaseFeatureComputationContext`.
4. If the extractor depends on 1D-specific derived signals, add that ingredient to `OneDimensionalFeatureComputationContext`.
5. Keep the extractor itself in the registry, not in downstream analysis code.
6. Add the feature to one or more entries in [feature_sets.json](experiments/common_1d_classifier_study/feature_sets.json).
7. Rerun:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/export_artifacts.py
```

8. Check:
   - `feature_analysis_report.md`
   - `feature_ranking_summary.png`
   - `pairwise_overlap_matrix.csv`
   - `pairwise_auc_matrix.csv`
   - `pca_report.md`
   - `corpus_adequacy_report.md`

## How To Add A New Feature Set

Edit [feature_sets.json](experiments/common_1d_classifier_study/feature_sets.json).

For a direct set:

```json
"timing_only": {
  "description": "Timing and cadence related features only.",
  "features": ["duration", "mean_dt", "std_dt", "max_dt", "sampling_irregularity"],
  "history_behavior": "mixed"
}
```

For a composed set:

```json
"timing_plus_shape": {
  "description": "Timing and shape feature union.",
  "includes": ["timing_only", "shape_window"],
  "history_behavior": "mixed"
}
```

The resolver expands `includes` recursively and removes duplicates while preserving order.

## What This Means For 3D

The current feature-analysis framework should carry forward to 3D in three layers:

1. Keep the registry, feature-set manifest, PCA, overlap analysis, and corpus adequacy flow.
2. Reuse `BaseFeatureComputationContext` for generic metadata and observed time-series inputs.
3. Add a new 3D-specific derived context and a parallel 3D feature registry instead of overloading the current 1D registry.

That is the intended extension point. A 3D migration should mostly be:

- new derived signals
- new registry entries
- new feature sets

rather than a rewrite of the analysis and artifact pipeline.

## What The Outputs Mean

### Feature Analysis

`feature_analysis_v1/` or `feature_analysis_<feature_set>_v1/` contains:

- `feature_matrix.csv`
  - trajectory metadata plus active feature values
- `feature_excitation_matrix.csv`
  - per-trajectory feature values and excitation labels
- `identifiability_matrix.csv`
  - pairwise class distances and overlap metrics
- `pairwise_overlap_matrix.csv`
  - direct overlap summary by class pair
- `pairwise_auc_matrix.csv`
  - direct separability summary by class pair
- `feature_ranking_summary.png`
  - top features ranked by average pairwise AUC
- `class_confusability_heatmap.png`
  - class-pair confusion pressure view

### PCA

`pca_analysis_v1/` or `pca_analysis_<feature_set>_v1/` contains:

- `pca_coordinates.csv`
- `pca_loadings.csv`
- `pca_explained_variance.csv`
- `pc1_pc2_by_class.png`
- `loadings_heatmap.png`

Use PCA as a diagnostic:

- if classes separate well in low dimensions, the feature set may already be sufficient
- if major overlap remains in PC space, the problem may be feature identifiability rather than classifier weakness

### Corpus Adequacy

`corpus_adequacy_audit_v1/` checks whether the synthetic corpus is actually stressing the feature sets and class pairs.

Use it to answer:

- are some feature sets weakly excited?
- are some class pairs underrepresented or under-stressed?
- is class balance too skewed?
- do covariates differ by class in suspicious ways?

### Abstract Inspection Bundle

`abstract_inspection_v1/` is the top-level inspection landing zone. It now contains:

- `abstract_inspection_index.md`
  - human-readable navigation and summary tables
- `abstract_inspection_summary.json`
  - machine-readable summary of feature-set rankings, hardest class pairs, corpus adequacy, and coverage
- `feature_set_inspection_summary.csv`
  - feature-set comparison table
- `feature_set_inspection_summary.png`
  - feature-set comparison chart
- `hardest_class_pairs.csv`
  - hardest pairwise class boundaries from the baseline feature space
- `hardest_class_pairs.png`
  - visual chart of those hardest class boundaries

Use this bundle when you want one place to answer:

- which feature bundles are strongest?
- which feature bundles are weakly supported by the corpus?
- which class boundaries remain hard even with the full baseline feature set?
- what should a downstream script focus on next without scraping markdown?

The companion terminal script `scripts/inspect_abstract_recommendations.py` is the fastest path when you only want the current recommendation instead of the full artifact review.

## Recommended Feature-Development Loop

1. Add or revise a feature in the registry.
2. Put it in one or more named feature sets.
3. Run feature analysis for that set.
4. Run PCA for that set.
5. Check corpus adequacy.
6. Only then decide whether classifier changes are warranted.

That sequence helps separate:

- feature engineering problems
- corpus/excitation problems
- actual classifier-model problems
