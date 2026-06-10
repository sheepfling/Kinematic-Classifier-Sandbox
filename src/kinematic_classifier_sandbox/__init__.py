"""Package initialization for the methodology workbench.

The package root intentionally exposes only a small supported API surface.
Importing the package must not configure caches, plotting, paths, or other
runtime state; scripts and CLI entrypoints own that setup explicitly.
"""

from __future__ import annotations

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
