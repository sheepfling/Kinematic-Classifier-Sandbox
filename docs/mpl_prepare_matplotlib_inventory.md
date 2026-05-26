# Matplotlib Prepare Inventory

Generated: 2026-05-25

Single source of truth:

`src/kinematic_classifier_sandbox/runtime_paths.py`

Defines both:

- `configure_matplotlib_environment()` to set `MPLCONFIGDIR`.
- `prepare_matplotlib()` to apply the environment, force `Agg`, and return `matplotlib.pyplot`.

Each caller module imports `_prepare_matplotlib()` directly from `runtime_paths`, so cache setup is delegated to one shared implementation.

## `_prepare_matplotlib` call sites

`src/kinematic_classifier_sandbox` has 39 `_prepare_matplotlib` call sites, each implemented as:

```py
from .runtime_paths import _prepare_matplotlib
```
