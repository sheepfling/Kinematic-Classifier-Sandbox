# SEA-SUB PR merge handoff

## Stack order

This tranche is intentionally stacked:

1. PR #1 — real-world corpus foundation and LAND reference implementation.
2. PR #2 — Product 4 declaration and episode/state-view direction.
3. PR #7 — this SEA-SUB source-evidence tranche.

Merge or otherwise establish each base before the dependent PR. PR #7 should remain scoped to its
current base until PR #2 is integrated; after retargeting, verify that its diff still contains only
SEA-SUB research evidence, fixtures, validation, and the focused workflow.

## Mergeable tranche claim

The tranche is ready to merge as research and contract evidence when its focused checks pass. It
establishes:

- a selected anchor and independent-validation portfolio;
- a retained, hashed, schema-inspected, and mapped 99-row IOOS/UAF anchor profile;
- two restricted source-grounded contract fixtures;
- explicit measured, calculated, dead-reckoned, contextual, and missing-state distinctions;
- deployment-safe grouping and classifier-feature exclusions;
- a common-front clarification for measured pressure versus derived depth.

## Current follow-on gates

These items are follow-on work, not hidden merge blockers for the research tranche:

- implement the production `P4-010` adapter and prepared pilot only after a defensible target cohort exists;
- retain and validate one Sentry PPL artifact;
- run classifier studies or make performance claims.

G2 is now complete through the canonical channel-aware common front. Do not rewrite the PR
description or registry to claim independent validation, study readiness, or classifier performance.

## Recommended merge handling

- Confirm PR #7 is mergeable after its base is integrated or retargeted.
- Confirm focused pytest, Ruff, Ruff-format, and Pyright checks are green.
- Confirm no unresolved review threads remain.
- Prefer squash merge so the research tranche lands as one coherent Product 4 increment.
- Preserve `SCR-SEA-SUB-001` for COMMON-FRONT adjudication after fixture convergence.
