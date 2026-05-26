# PLN-031 Tuple Return Type Normalization

Status: proposed
Owner: @rick
Priority: P2
Last Updated: 2026-05-26

## Metadata

- Plan ID: PLN-031
- Title: Tuple Return Type Normalization
- Objective: Replace large positional tuple return signatures with explicit named return types so call sites can read and maintain them without memorizing tuple order.

## Scope

- Replace “mystery tuple” returns that carry multiple related values in positional order.
- Prefer `NamedTuple` or small dataclasses for return values like:
  - `tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]`
  - `tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float, float]`
  - `tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]`
- Start with the highest-value surfaces:
  - `trajectory_generator.py`
  - `common_experiment/adapters.py`
  - `common_experiment/pair_evaluation.py`
  - `corpus/exploration/backend_adapter_proof.py`
  - `corpus/adaptive_stress_utils.py`
  - `analysis/dimensional_lift_audit.py`
  - `witnesses/toy_1d/bayesian_walkthroughs.py`

## Out of Scope

- Matrix-like tuple shapes used as mathematical primitives, unless they are exposed as opaque API results.
- Broad algorithm redesign or numerical changes.
- Serialization format changes unless they are required to preserve compatibility with the new named return types.
- Downstream cleanup of every tuple-like internal helper in one pass.

## Implementation Steps

1. Inventory the remaining tuple-return smells and group them by shape and caller impact.
2. Introduce shared named result types for repeated patterns:
   - trajectory series triples
   - paired row-group returns
   - prediction series records
3. Replace the most confusing public or cross-module return signatures first.
4. Update callers to use named fields instead of positional unpacking.
5. Keep compatibility shims only where the caller surface is large enough that a staged migration is safer.
6. Remove the compatibility layer once the caller set is updated.

## Validation

- Confirm each replaced signature has a named return type with explicit field names.
- Confirm the top call sites no longer rely on positional unpacking for multi-value results.
- Prefer targeted type-checking and focused import/call-site review over broad test runs unless a migration affects behavior.

## Artifacts / Config

- `docs/plans/PLN-031_tuple_return_type_normalization.md`
- Potential shared type modules for named tuple/dataclass return records

## Dependencies

- `PLN-030_row_schema_dataclass_inventory_and_conversion.md`
- Existing typed record modules already introduced in `corpus/` and `analysis/`

