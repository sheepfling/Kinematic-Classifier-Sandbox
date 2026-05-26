# PLN-030 Row Schema Dataclass Inventory And Conversion

Status: proposed
Owner: @codex
Priority: P1
Last Updated: 2026-05-25

## Title

Row schema dataclass inventory and conversion

## Plan ID

PLN-030

## Objective

Replace brittle row-shaped `dict` payloads with explicit dataclasses at the highest-value analysis boundaries, starting with calibration and decision metrics, so downstream code consumes typed records instead of ad hoc mappings.

## Scope

- Build a repository-wide inventory of row schemas grouped by inferred dataclass name.
- Convert the most valuable row pipelines to dataclasses, starting with calibration and posterior-quality metrics in `rung_sufficiency`.
- Keep dictionary serialization only at file and CSV boundaries.
- Preserve existing outputs and artifact formats.

## Out of Scope

- Rewriting every `dict[str, object]` in the repo.
- Changing JSON or CSV schemas in a breaking way.
- Introducing a new ORM or record framework.

## Implementation Steps

1. Inventory the row-shaped payload families across `src/` and `scripts/`.
2. Group each family under an inferred dataclass name.
3. Convert the top-priority metric rows in `rung_sufficiency/analysis.py`.
4. Move the same pattern into the other high-volume analysis and experiment runners.
5. Keep serialization helpers at the edge and type the internal pipeline.

## Validation

- Ensure the typed rows still serialize to the same CSV columns.
- Confirm the repo check scripts still pass after each conversion batch.

## Artifacts / Config

- `docs/row_schema_inventory.md`
- `src/kinematic_classifier_sandbox/rung_sufficiency/analysis.py`

## Dependencies

- Existing row producers in `common_experiment`, `advanced_filters`, `inference`, `analysis`, `validation`, and `corpus`.

## Last Updated

2026-05-25
