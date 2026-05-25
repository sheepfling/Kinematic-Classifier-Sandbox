# Package Boundaries

The codebase now has a deliberate distinction between a canonical front door and a compatibility surface.

## Canonical Front Door

Use [``kinematic_classifier_sandbox.api``](../../src/kinematic_classifier_sandbox/api.py) for the stable, reader-oriented entry points.

It groups the main methodology surfaces:

- story and catalog generation
- corpus generation, exploration, adequacy, and policy tuning
- feature and coverage evaluation
- classifier ladder and validation ladder entry points
- witness-suite and methodology-document writers

## Compatibility Surface

The package root, ``kinematic_classifier_sandbox.__init__``, remains a broader compatibility layer.

That keeps the current tests and legacy call sites working while the repo converges on a smaller canonical import surface.

## What This Means

- New readers should start from the story docs and then use `api.py` for code entry points.
- Existing imports from the package root remain valid.
- Future cleanup can move more call sites to the curated facade without breaking the current suite.
