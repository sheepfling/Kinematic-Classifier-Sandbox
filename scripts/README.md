# Scripts Layout

The `scripts/` directory is split by purpose.

Keep high-level user entrypoints at the root of `scripts/`:

- `all.py`
- `check.py`
- `dev.py`
- `export_artifacts.py`
- `export_team_packet.py`
- `run_study.py`
- `run_milestone.py`
- `run_abstract_inspection.py`

Group lower-level helpers by function:

- `scripts/audit/`: repo-shape, corpus, dimensional, PCA, and artifact validation audits
- `scripts/build/`: PDF and methodology document build pipelines
- `scripts/render/`: artifact renderers and report generators, including
  `render_exported_surface_coverage.py` for the canonical `export_artifacts.py`
  surface-coverage audit and `render_corpus_evaluation_gap_matrix.py` for the
  canonical corpus-evaluation capability/coherence audit. Use
  `render_methodology_latex.py` for the narrow methodology LaTeX rebuild and
  `render_methodology_section_symbol_audit.py` for the section-symbol audit.
- `scripts/run/`: benchmark and advanced-filter witness runners
- `scripts/workflows/`: multi-step study workflows and orchestration helpers

Rule of thumb:

- If a script is a canonical entrypoint a user is likely to run directly, keep it at `scripts/`.
- If a script is a specialized helper for one subsystem, place it in the matching subdirectory.

Common LaTeX rerun commands:

- Narrow methodology packet: `python3 scripts/render/render_methodology_latex.py`
- Narrow methodology packet without PDF: `python3 scripts/render/render_methodology_latex.py --fast`
- Section-symbol audit: `python3 scripts/render/render_methodology_section_symbol_audit.py`
- Full artifact bundle: `python3 scripts/export_artifacts.py`

Common audit and rerun commands:

- Repo checks: `python3 scripts/check.py`
- Repo-shape audit: `python3 scripts/audit/audit_repo_shape.py`
- Human-operability audit: `python3 scripts/audit/audit_human_operability.py --write-artifacts`
- Artifact/showcase validation: `python3 scripts/audit/validate_artifacts.py`
- Corpus audit: `python3 scripts/audit/audit_corpus.py`
- Dimensional audit: `python3 scripts/audit/audit_dimensions.py`
