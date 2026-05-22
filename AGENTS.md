## Repo Location Policy

- The git repository lives in CloudDocs at:
  `/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox`
- The local working entry in:
  `/Users/rick/LocalStorage/GIT_LOCAL/active/kinematic-classifier-sandbox`
  is a symlink to that repo.
- Keep large generated state outside CloudDocs in:
  `/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox`
- Root-level `.venv`, `.cache`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `cache`, and `archive` are symlinks into that local cache tree.
- Root-level `artifacts` stays in the repo tree so generated survey outputs sync, but `artifacts/` is ignored by git.
- Do not place build artifacts, caches, virtual environments, or large generated binary dumps directly in the CloudDocs repo tree.
- Keep `origin` local-only if a remote is added later. Do not add GitHub or any other hosted remote unless the user explicitly asks for it by name.
- When running Python directly, prefer setting `PYTHONPYCACHEPREFIX` to:
  `/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache`
  so `__pycache__` stays off CloudDocs.
