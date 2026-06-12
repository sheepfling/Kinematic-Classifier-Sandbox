from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import check_environment

TEST_LANES: dict[str, tuple[str, ...]] = {
    "shape": (
        "tests/api/test_api_core.py",
        "tests/api/test_no_root_compat_surfaces.py",
        "tests/api/test_root_compat_surfaces.py",
        "tests/meta/test_import_guardrail_docs.py",
        "tests/meta/test_import_simplicity_audit.py",
        "tests/meta/test_human_operability_audit.py",
        "tests/meta/test_repo_shape_audit.py",
        "tests/registry/test_function_shape_audit.py",
        "tests/showcase/test_showcase_builder.py",
        "tests/story/test_repo_story.py",
    ),
    "correctness-smoke": (
        "tests/static_admissibility/test_static_admissibility_packet.py",
        "tests/analysis/test_static_feature_class_prior_audit.py",
        "tests/corpus/test_corpus_policy.py",
        "tests/validation/test_class_validity.py",
        "tests/advanced_filters/test_advanced_filter_contract.py",
        "tests/meta/test_repo_shape_audit.py",
        "tests/validation/test_correctness.py",
    ),
    "correctness-full": (
        "tests/static_admissibility/test_static_admissibility_packet.py",
        "tests/analysis/test_static_feature_class_prior_audit.py",
        "tests/corpus/test_corpus_policy.py",
        "tests/validation/test_class_validity.py",
        "tests/validation/test_validation_ladder.py",
        "tests/advanced_filters/test_advanced_filter_contract.py",
        "tests/advanced_filters/test_oracle_1d.py",
        "tests/advanced_filters/test_oracle_pf_1d.py",
        "tests/advanced_filters/test_imm_filter.py",
        "tests/advanced_filters/test_particle_filter.py",
        "tests/advanced_filters/test_rbpf.py",
        "tests/corpus/test_corpus_policy_sweep.py",
        "tests/corpus/exploration/test_generic_corpus_exploration.py",
    ),
    "correctness-presentation": (
        "tests/showcase/test_showcase_builder.py",
        "tests/meta/test_human_operability_audit.py",
        "tests/story/test_repo_story.py",
        "tests/static_admissibility/test_static_admissibility_packet.py",
        "tests/meta/test_repo_shape_audit.py",
    ),
    "static": (
        "tests/static_admissibility/test_static_admissibility_packet.py",
        "tests/analysis/test_static_feature_class_prior_audit.py",
        "tests/analysis/test_inspection_bundle.py",
    ),
    "study": (
        "tests/study_candidate_generation/test_study_candidate_generation.py",
        "tests/study_candidate_generation/test_study_candidate_protocol.py",
        "tests/common_experiment/test_common_experiment_harness.py",
        "tests/validation/test_validation_ladder.py",
    ),
    "filters": (
        "tests/advanced_filters",
        "tests/tracing/test_filter_trace_schema.py",
        "tests/tracing/test_filter_trace_validation_packet.py",
    ),
    "corpus-policy": (
        "tests/corpus/test_corpus_policy.py",
        "tests/corpus/test_corpus_policy_sweep.py",
        "tests/corpus/exploration/test_generic_corpus_exploration_weight_sweep.py",
        "tests/corpus/exploration/test_generic_corpus_exploration.py",
        "tests/corpus/test_corpus_adequacy_audit.py",
        "tests/corpus/test_coverage_report.py",
    ),
    "inference": (
        "tests/inference/test_pointwise_baseline.py",
        "tests/inference/test_windowed_baseline.py",
        "tests/inference/test_sequential_bayes_accumulator.py",
        "tests/inference/test_kalman_filter_bank.py",
        "tests/inference/test_transition_matrix_accumulator.py",
        "tests/inference/test_prior_sensitivity_analysis.py",
    ),
    "methodology-fast": (
        "tests/methodology/test_cached_analysis.py",
        "tests/methodology/test_generic_classification_evidence_proof.py",
        "tests/methodology/test_generic_feature_taxonomy.py",
        "tests/methodology/test_generic_filtering_contract.py",
        "tests/methodology/test_generic_inference_contract.py",
        "tests/methodology/test_methodology_compendium.py",
    ),
    "docs-heavy": (
        "tests/meta/test_methodology_doc_coverage.py",
        "tests/methodology/test_methodology_latex.py",
    ),
}

FAST_LANES = ("shape", "static", "study", "methodology-fast")


def _existing_targets(root: Path, targets: Sequence[str]) -> list[str]:
    missing = [target for target in targets if not (root / target).exists()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"unknown test target(s): {joined}")
    return list(targets)


def _resolve_lanes(lane: str) -> tuple[str, ...]:
    if lane == "fast":
        targets: list[str] = []
        for lane_name in FAST_LANES:
            targets.extend(TEST_LANES[lane_name])
        return tuple(dict.fromkeys(targets))
    if lane == "all-light":
        return (
            "tests",
            "--ignore=tests/meta/test_methodology_doc_coverage.py",
            "--ignore=tests/methodology/test_methodology_latex.py",
        )
    if lane == "all":
        return ("tests",)
    try:
        return TEST_LANES[lane]
    except KeyError as exc:
        raise SystemExit(f"unknown lane: {lane}") from exc


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    lanes = ("fast", "all-light", "all", *TEST_LANES.keys())
    parser = argparse.ArgumentParser(description="Run lane-scoped pytest targets.")
    parser.add_argument(
        "--lane",
        choices=lanes,
        default="fast",
        help="test lane to run; defaults to fast package-shape and methodology smoke tests",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra args passed to pytest after the selected lane targets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root, env = check_environment()
    targets = list(_resolve_lanes(args.lane))
    path_targets = [target for target in targets if not target.startswith("--")]
    _existing_targets(root, path_targets)
    extra_args = list(args.pytest_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return run(
        [sys.executable, "-m", "pytest", *targets, *extra_args],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
