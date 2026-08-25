# Scripts Layout

The `scripts/` directory is split by purpose.

Keep high-level user entrypoints at the root of `scripts/`:

- `all.py`
- `check.py`
- `dev.py`
- `test.py`
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
  Use `render_classifier_capability_ladder_mvp.py` for the Epic 2 capability
  packet, `render_advanced_algorithm_showcase.py` for the advanced-algorithm
  shine-witness packet, and `render_corpus_explorer_mvp.py` for the V5C Corpus
  Explorer packet.
- `scripts/run/`: benchmark and advanced-filter witness runners
- `scripts/workflows/`: multi-step study workflows and orchestration helpers

Rule of thumb:

- If a script is a canonical entrypoint a user is likely to run directly, keep it at `scripts/`.
- If a script is a specialized helper for one subsystem, place it in the matching subdirectory.

Product 4 lane tests can run in isolated worker processes:

- `PYTHONPATH=src python3 scripts/run/run_product4_tests.py --workers 4`

The runner uses the explicit `product4_*` pytest markers and keeps common-contract, cross-domain,
and six lane groups separate. It is safe for the current real-world tests because they use
repository fixtures and per-test temporary directories; add a dedicated output root before
parallelizing any test that writes shared generated artifacts.

The complete product/tier/parallel-safety map is [the Product Test Matrix](../docs/testing/TEST_MATRIX.md).
Use the direct markers for Product 1–3 and shared analysis suites; keep Product 2, Product 3,
shared analysis, and full-repository gates sequential until their artifact namespaces are isolated.
- Scripts do not mutate `sys.path`. Run them after `python3 -m pip install -e '.[dev]'`
  or with an explicit `PYTHONPATH=src` environment.

Import and package-surface rules:

- Internal code imports concrete owner modules, not package-root compatibility wrappers.
- Do not add broad `__init__.py` reexports, wildcard imports, dynamic `__all__`, or
  module-level `__getattr__` public-surface tricks.
- Do not add script-local `sys.path.insert(...)` bootstraps.
- `python3 scripts/check.py` runs `scripts/audit/audit_import_simplicity.py --strict`;
  use `PYTHONPATH=src python3 scripts/audit/audit_import_simplicity.py --strict`
  for a narrow pre-commit check.

Common LaTeX rerun commands:

- Narrow methodology packet: `python3 scripts/render/render_methodology_latex.py`
- Narrow methodology packet without PDF: `python3 scripts/render/render_methodology_latex.py --fast`
- Section-symbol audit: `python3 scripts/render/render_methodology_section_symbol_audit.py`
- Front-door artifact bundle: `python3 scripts/export_artifacts.py --scope front-door`
- Front-door artifact bundle without PDF-heavy steps: `python3 scripts/export_artifacts.py --scope front-door --fast`
- Full artifact bundle: `python3 scripts/export_artifacts.py`
- Epic 2 capability packet: `PYTHONPATH=src python3 scripts/render/render_classifier_capability_ladder_mvp.py`
- Advanced Algorithm Showcase packet: `PYTHONPATH=src python3 scripts/render/render_advanced_algorithm_showcase.py`
- V5C Corpus Explorer packet: `PYTHONPATH=src python3 scripts/render/render_corpus_explorer_mvp.py`
- V5C Corpus Explorer packet validator: `PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet artifacts/packets/corpus_explorer_mvp --profile corpus_explorer_mvp`

Common audit and rerun commands:

- Repo checks: `python3 scripts/check.py`
- Fast package-shape tests: `PYTHONPATH=src python3 scripts/test.py`
- Shape/import/front-door tests only: `PYTHONPATH=src python3 scripts/test.py --lane shape`
- Corpus policy tests only: `PYTHONPATH=src python3 scripts/test.py --lane corpus-policy`
- Advanced-filter tests only: `PYTHONPATH=src python3 scripts/test.py --lane filters`
- Broad light pytest run without PDF/doc-build tests: `PYTHONPATH=src python3 scripts/test.py --lane all-light`
- Heavy methodology documentation tests: `PYTHONPATH=src python3 scripts/test.py --lane docs-heavy`
- Repo-shape audit: `python3 scripts/audit/audit_repo_shape.py`
- Human-operability audit: `python3 scripts/audit/audit_human_operability.py --write-artifacts`
- Analysis-cache summary: `python3 scripts/audit/manage_analysis_cache.py summary`
- Package-CLI analysis-cache summary: `python3 -m kinematic_classifier_sandbox analysis-cache summary`
- Clear one analysis-cache namespace: `python3 scripts/audit/manage_analysis_cache.py clear --namespace feature_analysis --yes`
- Package-CLI clear one namespace: `python3 -m kinematic_classifier_sandbox analysis-cache clear --namespace feature_analysis --yes`
- Clear all analysis caches: `python3 scripts/audit/manage_analysis_cache.py clear --yes`
- Artifact/showcase validation: `python3 scripts/audit/validate_artifacts.py`
- Corpus audit: `python3 scripts/audit/audit_corpus.py`
- Product 4 corpus gates: `PYTHONPATH=src python3 scripts/audit/evaluate_product4_gates.py`
- Build Product 4 snapshot manifest: `PYTHONPATH=src python3 scripts/run/build_product4_snapshot.py --snapshot-root /external/product4-snapshot --snapshot-id product4-snapshot-v0.1`
- Build SPACE-NEAR validation tranche: `PYTHONPATH=src python3 scripts/run/build_space_near_validation_snapshot.py --snapshot-root /external/product4-space-near --snapshot-id product4-space-near-validation-v0.1`
- Build six-lane validation tranche: `PYTHONPATH=src python3 scripts/run/build_product4_validation_snapshot.py --snapshot-root /external/product4-six-lane --snapshot-id product4-six-lane-validation-v0.1 --cmre-tracklets /external/tracklets.csv --cmre-nomenclature /external/nomen.csv --cmre-identity-key /secure/cmre-identity.key`
- Dimensional audit: `python3 scripts/audit/audit_dimensions.py`
