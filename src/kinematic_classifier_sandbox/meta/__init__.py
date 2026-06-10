from __future__ import annotations

__all__ = [
    "analyze_methodology_compendium",
    "analyze_methodology_latex",
    "write_methodology_section_symbol_audit_artifacts",
    "write_methodology_compendium_artifacts",
    "write_methodology_latex_artifacts",
]


def analyze_methodology_compendium(*args, **kwargs):
    from ..methodology_compendium import analyze_methodology_compendium as _impl

    return _impl(*args, **kwargs)


def analyze_methodology_latex(*args, **kwargs):
    from ..methodology.latex import analyze_methodology_latex as _impl

    return _impl(*args, **kwargs)


def write_methodology_section_symbol_audit_artifacts(*args, **kwargs):
    from ..methodology.latex import write_methodology_section_symbol_audit_artifacts as _impl

    return _impl(*args, **kwargs)


def write_methodology_compendium_artifacts(*args, **kwargs):
    from ..methodology_compendium import write_methodology_compendium_artifacts as _impl

    return _impl(*args, **kwargs)


def write_methodology_latex_artifacts(*args, **kwargs):
    from ..methodology.latex import write_methodology_latex_artifacts as _impl

    return _impl(*args, **kwargs)
