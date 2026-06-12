# Package Utility Lane

## Purpose

The package utility lane keeps the workbench installable, runnable, validated, and reusable.

The expected public commands are:

```bash
python -m kinematic_classifier_sandbox run-static-audit ...
python -m kinematic_classifier_sandbox run-study ...
python -m kinematic_classifier_sandbox export-packet ...
python -m kinematic_classifier_sandbox validate-packet ...
```

## Outputs

- CLI commands.
- API helpers.
- Config schemas.
- Packet validators.
- Regression tests.

## Claim Boundary

Utility code should keep the pipeline reproducible without package-root path hacks, broad reexports, or duplicate implementation files.

## Next Work

- Add a shared `StudySpec`.
- Add a shared `DecisionCard`.
- Keep package import surfaces simple and explicit.

