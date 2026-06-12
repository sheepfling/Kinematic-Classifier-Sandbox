from __future__ import annotations

from .catalog import METHOD_CATALOG, MethodEntry, method_families

__all__ = [
    "METHOD_CATALOG",
    "MethodEntry",
    "method_families",
    "analyze_algorithm_coverage_matrix",
    "write_algorithm_coverage_matrix_artifacts",
    "analyze_classifier_family_scorecard",
    "write_classifier_family_scorecard_artifacts",
    "analyze_corpus_evaluation_gap_matrix",
    "render_corpus_evaluation_gap_matrix_report",
    "write_corpus_evaluation_gap_matrix_artifacts",
    "analyze_exported_surface_coverage",
    "render_exported_surface_coverage_report",
    "write_exported_surface_coverage_artifacts",
    "analyze_formal_math_registry",
    "load_equation_registry",
    "render_formal_math_registry_report",
    "write_formal_math_registry_artifacts",
    "analyze_functional_surface_catalog",
    "render_functional_surface_catalog_report",
    "write_functional_surface_catalog_artifacts",
    "analyze_embedding_baseline_frontier",
    "write_embedding_baseline_frontier_artifacts",
    "analyze_neural_sequence_robustness_frontier",
    "write_neural_sequence_robustness_frontier_artifacts",
    "analyze_physics_family_promotion_audit",
    "write_physics_family_promotion_audit_artifacts",
    "analyze_drcif_interval_promotion_audit",
    "write_drcif_interval_promotion_audit_artifacts",
    "analyze_gsf_multimodal_promotion_audit",
    "write_gsf_multimodal_promotion_audit_artifacts",
    "analyze_ukf_nonlinear_promotion_audit",
    "write_ukf_nonlinear_promotion_audit_artifacts",
    "analyze_imm_switching_promotion_audit",
    "write_imm_switching_promotion_audit_artifacts",
    "analyze_ts2vec_backend_parity",
    "write_ts2vec_backend_parity_artifacts",
    "analyze_method_validation_os",
    "write_method_validation_os_artifacts",
    "analyze_strict_equation_audit",
    "render_strict_equation_audit_report",
    "write_strict_equation_audit_artifacts",
]


def analyze_algorithm_coverage_matrix(*args, **kwargs):
    from .algorithm_coverage_matrix import analyze_algorithm_coverage_matrix as _impl

    return _impl(*args, **kwargs)


def analyze_classifier_family_scorecard(*args, **kwargs):
    from ..analysis.classifier_family_scorecard import (
        analyze_classifier_family_scorecard as _impl,
    )

    return _impl(*args, **kwargs)


def write_classifier_family_scorecard_artifacts(*args, **kwargs):
    from ..analysis.classifier_family_scorecard import (
        write_classifier_family_scorecard_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_formal_math_registry(*args, **kwargs):
    from .formal_math_registry import analyze_formal_math_registry as _impl

    return _impl(*args, **kwargs)


def analyze_corpus_evaluation_gap_matrix(*args, **kwargs):
    from .corpus_evaluation_gap_matrix import analyze_corpus_evaluation_gap_matrix as _impl

    return _impl(*args, **kwargs)


def render_corpus_evaluation_gap_matrix_report(*args, **kwargs):
    from .corpus_evaluation_gap_matrix import (
        render_corpus_evaluation_gap_matrix_report as _impl,
    )

    return _impl(*args, **kwargs)


def write_corpus_evaluation_gap_matrix_artifacts(*args, **kwargs):
    from .corpus_evaluation_gap_matrix import (
        write_corpus_evaluation_gap_matrix_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_exported_surface_coverage(*args, **kwargs):
    from .exported_surface_coverage import analyze_exported_surface_coverage as _impl

    return _impl(*args, **kwargs)


def render_exported_surface_coverage_report(*args, **kwargs):
    from .exported_surface_coverage import (
        render_exported_surface_coverage_report as _impl,
    )

    return _impl(*args, **kwargs)


def write_exported_surface_coverage_artifacts(*args, **kwargs):
    from .exported_surface_coverage import (
        write_exported_surface_coverage_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def load_equation_registry(*args, **kwargs):
    from .formal_math_registry import load_equation_registry as _impl

    return _impl(*args, **kwargs)


def render_formal_math_registry_report(*args, **kwargs):
    from .formal_math_registry import render_formal_math_registry_report as _impl

    return _impl(*args, **kwargs)


def write_formal_math_registry_artifacts(*args, **kwargs):
    from .formal_math_registry import write_formal_math_registry_artifacts as _impl

    return _impl(*args, **kwargs)


def analyze_functional_surface_catalog(*args, **kwargs):
    from .functional_surface_catalog import analyze_functional_surface_catalog as _impl

    return _impl(*args, **kwargs)


def render_functional_surface_catalog_report(*args, **kwargs):
    from .functional_surface_catalog import (
        render_functional_surface_catalog_report as _impl,
    )

    return _impl(*args, **kwargs)


def write_functional_surface_catalog_artifacts(*args, **kwargs):
    from .functional_surface_catalog import (
        write_functional_surface_catalog_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_embedding_baseline_frontier(*args, **kwargs):
    from ..analysis.embedding_baseline_frontier import (
        analyze_embedding_baseline_frontier as _impl,
    )

    return _impl(*args, **kwargs)


def write_embedding_baseline_frontier_artifacts(*args, **kwargs):
    from ..analysis.embedding_baseline_frontier import (
        write_embedding_baseline_frontier_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_neural_sequence_robustness_frontier(*args, **kwargs):
    from ..analysis.neural_sequence_robustness import (
        analyze_neural_sequence_robustness_frontier as _impl,
    )

    return _impl(*args, **kwargs)


def write_neural_sequence_robustness_frontier_artifacts(*args, **kwargs):
    from ..analysis.neural_sequence_robustness import (
        write_neural_sequence_robustness_frontier_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_physics_family_promotion_audit(*args, **kwargs):
    from ..analysis.physics_family_promotion_audit import (
        analyze_physics_family_promotion_audit as _impl,
    )

    return _impl(*args, **kwargs)


def write_physics_family_promotion_audit_artifacts(*args, **kwargs):
    from ..analysis.physics_family_promotion_audit import (
        write_physics_family_promotion_audit_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_drcif_interval_promotion_audit(*args, **kwargs):
    from ..analysis.drcif_interval_promotion_audit import (
        analyze_drcif_interval_promotion_audit as _impl,
    )

    return _impl(*args, **kwargs)


def write_drcif_interval_promotion_audit_artifacts(*args, **kwargs):
    from ..analysis.drcif_interval_promotion_audit import (
        write_drcif_interval_promotion_audit_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_gsf_multimodal_promotion_audit(*args, **kwargs):
    from ..analysis.gsf_multimodal_promotion_audit import (
        analyze_gsf_multimodal_promotion_audit as _impl,
    )

    return _impl(*args, **kwargs)


def write_gsf_multimodal_promotion_audit_artifacts(*args, **kwargs):
    from ..analysis.gsf_multimodal_promotion_audit import (
        write_gsf_multimodal_promotion_audit_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_ukf_nonlinear_promotion_audit(*args, **kwargs):
    from ..analysis.ukf_nonlinear_promotion_audit import (
        analyze_ukf_nonlinear_promotion_audit as _impl,
    )

    return _impl(*args, **kwargs)


def write_ukf_nonlinear_promotion_audit_artifacts(*args, **kwargs):
    from ..analysis.ukf_nonlinear_promotion_audit import (
        write_ukf_nonlinear_promotion_audit_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_imm_switching_promotion_audit(*args, **kwargs):
    from ..analysis.imm_switching_promotion_audit import (
        analyze_imm_switching_promotion_audit as _impl,
    )

    return _impl(*args, **kwargs)


def write_imm_switching_promotion_audit_artifacts(*args, **kwargs):
    from ..analysis.imm_switching_promotion_audit import (
        write_imm_switching_promotion_audit_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_ts2vec_backend_parity(*args, **kwargs):
    from ..analysis.ts2vec_backend_parity import analyze_ts2vec_backend_parity as _impl

    return _impl(*args, **kwargs)


def write_ts2vec_backend_parity_artifacts(*args, **kwargs):
    from ..analysis.ts2vec_backend_parity import (
        write_ts2vec_backend_parity_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def write_algorithm_coverage_matrix_artifacts(*args, **kwargs):
    from .algorithm_coverage_matrix import (
        write_algorithm_coverage_matrix_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_method_validation_os(*args, **kwargs):
    from .method_validation_os import analyze_method_validation_os as _impl

    return _impl(*args, **kwargs)


def write_method_validation_os_artifacts(*args, **kwargs):
    from .method_validation_os import (
        write_method_validation_os_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def analyze_strict_equation_audit(*args, **kwargs):
    from .strict_equation_audit import analyze_strict_equation_audit as _impl

    return _impl(*args, **kwargs)


def render_strict_equation_audit_report(*args, **kwargs):
    from .strict_equation_audit import render_strict_equation_audit_report as _impl

    return _impl(*args, **kwargs)


def write_strict_equation_audit_artifacts(*args, **kwargs):
    from .strict_equation_audit import write_strict_equation_audit_artifacts as _impl

    return _impl(*args, **kwargs)
