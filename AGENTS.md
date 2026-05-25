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
