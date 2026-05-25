# Scripts Layout

The `scripts/` directory is split by purpose.

Keep high-level user entrypoints at the root of `scripts/`:

- `all.py`
- `export_artifacts.py`
- `export_team_packet.py`
- `run_study.py`
- `run_milestone.py`
- `run_abstract_inspection.py`

Group lower-level helpers by function:

- `scripts/audit/`: corpus, dimensional, PCA, and artifact validation audits
- `scripts/build/`: PDF and methodology document build pipelines
- `scripts/render/`: artifact renderers and report generators
- `scripts/run/`: benchmark and advanced-filter witness runners
- `scripts/workflows/`: multi-step study workflows and orchestration helpers

Rule of thumb:

- If a script is a canonical entrypoint a user is likely to run directly, keep it at `scripts/`.
- If a script is a specialized helper for one subsystem, place it in the matching subdirectory.
