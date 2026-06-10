# PLN-033 Repo Shape and Cruft Reduction

Status: in_progress
Owner: @codex
Priority: P1
Last Updated: 2026-06-10

## Objective

Reshape the repository around the declared methodology architecture, remove duplicate legacy surfaces, prune generated cruft from source-controlled areas, and add a repo-shape guard so new drift is caught automatically.

## Scope

- Package root cleanup and canonical subpackage enforcement.
- Duplicate script removal.
- Generated/cache cruft removal from source, docs, tests, scripts, experiments, and templates.
- Repo-shape inventory and strict audit guard.
- Canonical documentation navigation and duplicate-doc pruning.

## Out of Scope

- Changing classifier, corpus, or filter algorithms.
- Rewriting study semantics or artifact schemas unless needed for path migration.
- Removing public `__init__.py` exports that are intentionally supported front-door API names.

## Implementation Steps

1. Add repo-shape inventory and audit tooling.
2. Remove ignored local build/cache byproducts.
3. Collapse duplicate root render scripts into `scripts/render/`.
4. Migrate or delete safe root compatibility wrappers once imports point to grouped modules.
5. Prune duplicate docs after canonical replacements are linked.
6. Split oversized modules only by moving rendering, contracts, and artifact I/O out of algorithm/core files.
7. Wire the audit into the normal check path.

## Validation

- Repo-shape audit passes.
- Root duplicate render scripts are gone.
- Generated/cache cruft is absent from source/doc/test/script trees.
- Focused migration tests pass after each module batch.
- Full regression passes with `python3 -m unittest discover tests`.

## Artifacts / Config

- `artifacts/repo_shape_audit_v1/repo_shape_audit_report.md`
- `artifacts/repo_shape_audit_v1/repo_shape_audit_summary.json`
- `artifacts/repo_shape_audit_v1/root_module_inventory.csv`
- `artifacts/repo_shape_audit_v1/duplicate_module_inventory.csv`
- `artifacts/repo_shape_audit_v1/duplicate_script_inventory.csv`
- `artifacts/repo_shape_audit_v1/generated_cruft_inventory.csv`
- `artifacts/repo_shape_audit_v1/oversized_module_inventory.csv`

## Dependencies

- Current package map in `src/kinematic_classifier_sandbox/README.md`.
- Existing grouped packages under `analysis/`, `corpus/`, `inference/`, `validation/`, `registry/`, `methodology/`, `witnesses/`, `advanced_filters/`, and `common_experiment/`.

