"""Package initialization for the methodology workbench.

This package root intentionally exposes only the small set of canonical
build entrypoints used by repository scripts, while keeping import-time side
effects and dependency loading minimal.
"""

from __future__ import annotations

from .runtime_paths import configure_runtime_environment

configure_runtime_environment()

__all__ = [
    "write_methodology_compendium_artifacts",
    "write_methodology_latex_artifacts",
    "write_methodology_section_symbol_audit_artifacts",
]


def write_methodology_compendium_artifacts(*args, **kwargs):
    from .methodology.compendium import write_methodology_compendium_artifacts as _impl

    return _impl(*args, **kwargs)


def write_methodology_latex_artifacts(*args, **kwargs):
    from .methodology.latex import write_methodology_latex_artifacts as _impl

    return _impl(*args, **kwargs)


def write_methodology_section_symbol_audit_artifacts(*args, **kwargs):
    from .methodology.latex import write_methodology_section_symbol_audit_artifacts as _impl

    return _impl(*args, **kwargs)
