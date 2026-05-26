# Package Boundaries

The codebase now has a deliberate canonical core entry point in `src/kinematic_classifier_sandbox/api_core.py`.

## Canonical Front Door

Use [``kinematic_classifier_sandbox.api_core``](../../src/kinematic_classifier_sandbox/api_core.py) for the stable, reader-oriented entry points.

It groups the main methodology surfaces:

- story and catalog generation
- corpus generation, exploration, adequacy, and policy tuning
- feature and coverage evaluation
- classifier ladder and validation ladder entry points
- witness-suite and methodology-document writers

## What This Means

- New readers should start from the story docs and then use `api_core.py` for code entry points.
- Existing imports should target `api_core.py` or the owning module directly.
- The package no longer exposes a compatibility facade.
