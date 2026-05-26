from __future__ import annotations

from .catalog import METHOD_CATALOG, MethodEntry, method_families
from .formal_math_registry import (
    ARTIFACT_DIR as FORMAL_MATH_REGISTRY_ARTIFACT_DIR,
)
from .formal_math_registry import (
    EQUATION_REGISTRY_PATH,
    FormalMathEquationRow,
    FormalMathFunctionRow,
    FormalMathRegistryArtifacts,
    FormalMathRegistryResult,
    analyze_formal_math_registry,
    load_equation_registry,
    render_formal_math_registry_report,
    write_formal_math_registry_artifacts,
)
from .functional_surface_catalog import (
    FUNCTIONAL_SURFACE_REGISTRY,
    FunctionalSurfaceCatalogArtifacts,
    FunctionalSurfaceCatalogResult,
    FunctionalSurfaceRow,
    FunctionalSurfaceSpec,
    analyze_functional_surface_catalog,
    render_functional_surface_catalog_report,
    write_functional_surface_catalog_artifacts,
)
from .strict_equation_audit import (
    StrictEquationAuditArtifacts,
    StrictEquationAuditResult,
    StrictEquationAuditRow,
    analyze_strict_equation_audit,
    render_strict_equation_audit_report,
    write_strict_equation_audit_artifacts,
)

__all__ = [
    "METHOD_CATALOG",
    "MethodEntry",
    "method_families",
    "FORMAL_MATH_REGISTRY_ARTIFACT_DIR",
    "EQUATION_REGISTRY_PATH",
    "FormalMathEquationRow",
    "FormalMathFunctionRow",
    "FormalMathRegistryArtifacts",
    "FormalMathRegistryResult",
    "analyze_formal_math_registry",
    "load_equation_registry",
    "render_formal_math_registry_report",
    "write_formal_math_registry_artifacts",
    "FUNCTIONAL_SURFACE_REGISTRY",
    "FunctionalSurfaceCatalogArtifacts",
    "FunctionalSurfaceCatalogResult",
    "FunctionalSurfaceRow",
    "FunctionalSurfaceSpec",
    "analyze_functional_surface_catalog",
    "render_functional_surface_catalog_report",
    "write_functional_surface_catalog_artifacts",
    "StrictEquationAuditArtifacts",
    "StrictEquationAuditResult",
    "StrictEquationAuditRow",
    "analyze_strict_equation_audit",
    "render_strict_equation_audit_report",
    "write_strict_equation_audit_artifacts",
]
