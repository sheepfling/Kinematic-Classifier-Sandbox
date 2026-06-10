## Repo Location Policy

- The git repository lives in the current repo checkout.
- The local working entry under `../active/kinematic-classifier-sandbox`
  is a symlink to this repo.
- Keep large generated state outside the repo tree in the local cache tree
  under `../active/CACHE/kinematic-classifier-sandbox`.
- Root-level `.venv`, `.cache`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `cache`, and `archive` are symlinks into that local cache tree.
- Root-level `artifacts` stays in the repo tree so generated survey outputs sync, but `artifacts/` is ignored by git.
- Do not place build artifacts, caches, virtual environments, or large generated binary dumps directly in the CloudDocs repo tree.
- Keep `origin` local-only if a remote is added later. Do not add GitHub or any other hosted remote unless the user explicitly asks for it by name.
- When running Python directly, prefer setting `PYTHONPYCACHEPREFIX` to the
  local cache tree's `.pycache` directory so `__pycache__` stays out of the repo.

## Import and Package Surface Policy

- Internal package code must import the owning module directly. Do not add
  package-root compatibility wrappers or broad package `__init__` reexports.
- Do not mutate `sys.path` or `PYTHONPATH` from package code. Scripts should
  rely on an editable install or an explicit caller-provided `PYTHONPATH=src`.
- Keep package `__init__.py` files passive. They should not configure runtime
  paths, caches, plotting, or other process state at import time.
- Do not use wildcard imports, dynamic `__all__`, module-level `__getattr__`,
  or other clever public-surface tricks.
- Run `PYTHONPATH=src python3 scripts/audit/audit_import_simplicity.py --strict`
  before committing import or package-layout changes. `scripts/check.py` also
  runs this audit in strict mode.
