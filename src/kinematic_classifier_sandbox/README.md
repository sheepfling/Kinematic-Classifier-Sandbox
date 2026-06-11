# Package Map

The package is organized around the methodology layers, not around one single
algorithm.

Read next:

- [Repo story](../../docs/story/00_repo_story.md)
- [Package boundaries](../../docs/story/package_boundaries.md)
- [Scripts layout](../../scripts/README.md)

## Utility lanes

Shared helpers are collected into a small number of neutral utility lanes:

- `utils/math.py`
  - scalar math, normalization, vector/matrix helpers, clustering/PCA primitives,
    and Kalman-neutral linear algebra
- `utils/io.py`
  - CSV/JSON/text readers and writers, fieldname union helpers, and small tabular helpers
- `utils/plotting.py`
  - typed matplotlib access plus reusable plotting primitives such as heatmaps and figure writers
- `utils/types.py`
  - reusable typing aliases such as array and callable contracts
- `utils/categorical.py`
  - shared categorical helpers such as status scoring and threshold bucketization

Promotion rule:

- move code into `utils/` only when it is domain-neutral and reused, or clearly reusable, across multiple subsystems
- keep report-specific figure composition, domain scoring, and study-specific thresholds near the owning subsystem

The `common_experiment/` area is the current reference pattern for deeper
splitting:

- `contracts.py`: dataclass result and artifact shapes
- `protocols.py`: callable interfaces and ABC-like protocols
- `config_models.py`: Pydantic-backed config validation
- `config.py`: config loading and adapter resolution
- `adapters.py`: executable dataset/pair adapters
- `scoring.py`: family evidence scoring
- `runner.py`: execution/analysis
- `reporting.py`: markdown report rendering
- `artifact_io.py`: artifact-export entrypoints

## Main subpackages

- `inference/`
  - pointwise, windowed, accumulator, Kalman-bank, transition-matrix, and 1D witness classifiers
  - also includes posterior explainers, prior-sensitivity, and benchmark runners
- `corpus/`
  - corpus generation, adequacy, policy, coverage, search, and exploration logic
  - `corpus/exploration/` contains the study-candidate and corpus-search machinery
- `analysis/`
  - feature analysis, PCA, identifiability, dimensional audit, and inspection bundles
- `validation/`
  - cross-method evaluation, class validity, and decision/report helpers
- `advanced_filters/`
  - IMM, particle-filter, and RBPF contracts, runners, and witness implementations
- `methodology/`
  - generic inference contracts, feature taxonomy proofs, evidence-provider proofs,
    and filtering-contract artifacts
- `showcase/`
  - team-facing showcase packet contracts, runner/export logic, and packet validation

## Root Modules

The package root is intentionally small. It contains only process front doors,
shared contracts, and cross-cutting orchestration that do not belong to a more
specific domain package:

- `__init__.py`: public exports
- `__main__.py`: package CLI
- `api_core.py`: curated core entry points
- `artifacts.py`: shared artifact writers
- `common_experiment_harness.py`: compatibility wrapper over the grouped `common_experiment/` package
- `common_experiment_classifier_registry.py`: classifier registry for the common harness
- `common_1d_study_adapter.py`: common 1D study adapter
- `contracts.py`: shared schemas and contracts
- `milestones.py`: milestone orchestration
- `showcase_builder.py`: team-facing showcase packet builder
- `repo_story.py`: canonical repo-story generator

Root-level compatibility wrappers are not allowed. If a function lives in
`analysis/`, `corpus/`, `inference/`, `validation/`, `methodology/`, or
`registry/`, import it from that owning module rather than adding a package-root
forwarder.

Examples:

- Use `kinematic_classifier_sandbox.analysis.feature_analysis`.
- Use `kinematic_classifier_sandbox.analysis.feature_analysis_artifact_io`.
- Use `kinematic_classifier_sandbox.corpus.coverage_artifact_io`.
- Use `kinematic_classifier_sandbox.registry.formal_math_registry`.

Do not add new files like `src/kinematic_classifier_sandbox/feature_analysis.py`
or populate package `__init__.py` files with broad convenience reexports.

## Rule of thumb

- Put new algorithm implementations under the most specific subpackage that fits.
- Keep package-root modules for shared contracts, orchestration, CLI entrypoints,
  or genuinely cross-cutting builders.
- Prefer extending existing grouped areas before adding more flat top-level
  modules in this package.
- Keep package imports boring: no wildcard facades, no dynamic `__all__`, no
  module-level `__getattr__`, and no import-time runtime setup.
- Run `PYTHONPATH=src python3 scripts/audit/audit_import_simplicity.py --strict`
  after changing imports or package layout.

## Exported surface coverage

The canonical exported artifact surface is the set of writers/builders called by
`scripts/export_artifacts.py`.

Coverage policy:

- every exported surface should declare a durable report artifact
- every exported surface should declare at least one machine-consumable class:
  tabular or summary
- visualization is required unless the exported-surface inventory records an
  explicit exemption reason
- the coverage audit lives under `registry/exported_surface_coverage.py`

Rerun commands:

- full static audit:
  `python3 scripts/render/render_exported_surface_coverage.py --output-dir artifacts`
- materialized subset audit:
  `python3 scripts/render/render_exported_surface_coverage.py --output-dir artifacts --materialize --surface-id feature_analysis --surface-id functional_surface_catalog`
- corpus-evaluation gap matrix:
  `python3 scripts/render/render_corpus_evaluation_gap_matrix.py --output-dir artifacts`
- materialized corpus-evaluation subset:
  `python3 scripts/render/render_corpus_evaluation_gap_matrix.py --output-dir artifacts --materialize --capability-id corpus_adequacy_scoring --capability-id selected_corpus_closed_loop_rerun`
