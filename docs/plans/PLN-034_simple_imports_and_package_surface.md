# PLN-034 Simple Imports and Package Surface

## Summary

Remove clever package behavior from the repository. The package should be easy to import, easy to trace, and boring to run: no path sniffing scattered through source files, no package import-time environment mutation, no wildcard compatibility facades, no dynamic public API tricks, and no circular-import workarounds hidden behind local imports.

This plan follows PLN-033. PLN-033 made the repo shape visible; PLN-034 makes the import and runtime surface simple.

## Target State

- Importing `kinematic_classifier_sandbox` has no runtime side effects.
- Runtime/cache path setup happens only in explicit CLI/script entrypoints.
- Repository and artifact paths are resolved through one explicit path utility.
- Internal source imports canonical subpackages, not package-root compatibility wrappers.
- Package-root compatibility modules are deleted or replaced with explicit, temporary named imports.
- No wildcard imports are used as public API facades.
- No script repeats `sys.path` or `PYTHONPATH` bootstrapping logic.
- Any remaining import indirection is documented, narrow, and enforced by an audit.

## Non-Goals

- Do not change classifier, corpus, witness, or validation behavior.
- Do not rename public concepts.
- Do not make broad algorithm refactors while import cleanup is underway.

## Rules

1. Package import must be passive.
2. Source modules must not mutate `sys.path` or `PYTHONPATH`.
3. Source modules must not discover the repo by hard-coded `Path(__file__).resolve().parents[...]` depth except inside the approved path utility.
4. Scripts may bootstrap only through one shared helper.
5. Wildcard imports are forbidden in package code.
6. Internal code must import canonical domain modules directly.
7. `__all__` may be an explicit static list, but must not be built dynamically or used to hide wildcard facades.
8. Function-local imports are allowed only for optional heavy dependencies or to avoid mandatory optional packages, with a short comment.

## Milestones

### M69: Import Simplicity Audit

Deliverables:

- `src/kinematic_classifier_sandbox/meta/import_simplicity_audit.py`
- `scripts/audit/audit_import_simplicity.py`
- `artifacts/import_simplicity_audit_v1/`

The audit inventories:

- wildcard imports
- path bootstrapping and `sys.path` mutation
- `PYTHONPATH` mutation
- hard-coded repo-root discovery with `parents[...]`
- package import-time runtime setup
- module-level `__getattr__`
- dynamic `__all__`
- internal imports from legacy root wrappers

### M70: Remove Package Import Side Effects

Remove runtime environment setup from `src/kinematic_classifier_sandbox/__init__.py`.

Runtime setup should be called by:

- `scripts/check.py`
- broad root scripts
- CLI entrypoints that write artifacts or render plots

### M71: Collapse Wildcard Facades

Replace wildcard root wrappers with explicit named imports or delete them after callers move to canonical subpackages.

Priority order:

1. package-root wrappers
2. `inference/*` wrappers over witness benchmark modules
3. witness benchmark package `__init__` wildcard exports

### M72: Replace Source Path Sniffing

Move all source repo-root references to `kinematic_classifier_sandbox.utils.runtime.repo_root`.

Allowed direct repo-root derivation:

- `src/kinematic_classifier_sandbox/utils/runtime.py`

### M73: Consolidate Script Bootstrapping

Create one script bootstrap helper and remove repeated `sys.path` / `PYTHONPATH` snippets from scripts.

### M74: Strict Guard

Wire the import-simplicity audit into `scripts/check.py` in strict mode once the violation count is zero.

## Validation

- `python3 scripts/audit/audit_import_simplicity.py --write-artifacts`
- `python3 scripts/audit/audit_import_simplicity.py --strict`
- `python3 scripts/audit/audit_repo_shape.py`
- `python3 -m unittest discover tests`
- `python3 scripts/check.py`

## Completion Criteria

- Import simplicity audit passes in strict mode.
- Repo shape audit still passes.
- Full tests pass.
- Package import has no environment side effects.
- No package code uses wildcard imports.
- No package code mutates import paths.
- Scripts share one bootstrap helper.
