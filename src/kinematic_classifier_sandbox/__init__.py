"""Package initialization for the methodology workbench.

This package root intentionally exposes only the small set of canonical
build entrypoints used by the repository's shell scripts.
"""

from __future__ import annotations

from .runtime_paths import configure_runtime_environment
from .methodology.compendium import write_methodology_compendium_artifacts
from .methodology.latex import write_methodology_latex_artifacts

configure_runtime_environment()
