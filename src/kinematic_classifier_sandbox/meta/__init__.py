from __future__ import annotations

__all__ = [
    "analyze_methodology_compendium",
    "analyze_methodology_latex",
    "analyze_human_operability_audit",
    "analyze_import_simplicity",
    "write_methodology_section_symbol_audit_artifacts",
    "write_methodology_compendium_artifacts",
    "write_methodology_latex_artifacts",
    "analyze_repo_shape",
    "write_human_operability_audit_artifacts",
    "write_import_simplicity_audit_artifacts",
    "write_repo_shape_audit_artifacts",
]


def analyze_methodology_compendium(*args, **kwargs):
    from ..methodology_compendium import analyze_methodology_compendium as _impl

    return _impl(*args, **kwargs)


def analyze_methodology_latex(*args, **kwargs):
    from ..methodology.latex import analyze_methodology_latex as _impl

    return _impl(*args, **kwargs)


def analyze_human_operability_audit(*args, **kwargs):
    from .human_operability_audit import analyze_human_operability_audit as _impl

    return _impl(*args, **kwargs)


def analyze_import_simplicity(*args, **kwargs):
    from .import_simplicity_audit import analyze_import_simplicity as _impl

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


def analyze_repo_shape(*args, **kwargs):
    from .repo_shape_audit import analyze_repo_shape as _impl

    return _impl(*args, **kwargs)


def write_repo_shape_audit_artifacts(*args, **kwargs):
    from .repo_shape_audit import write_repo_shape_audit_artifacts as _impl

    return _impl(*args, **kwargs)


def write_human_operability_audit_artifacts(*args, **kwargs):
    from .human_operability_audit import write_human_operability_audit_artifacts as _impl

    return _impl(*args, **kwargs)


def write_import_simplicity_audit_artifacts(*args, **kwargs):
    from .import_simplicity_audit import write_import_simplicity_audit_artifacts as _impl

    return _impl(*args, **kwargs)
