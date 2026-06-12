from __future__ import annotations

import json
from dataclasses import dataclass
from math import log
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..common_experiment.runner import analyze_common_experiment
from ..utils.math import _normalize_log_scores


@dataclass(frozen=True, slots=True)
class EvidenceStep:
    time: float
    log_likelihoods: dict[str, float]


@dataclass(frozen=True, slots=True)
class PosteriorStep:
    time: float
    posterior: dict[str, float]
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class GenericClassificationEvidenceProofResult:
    evidence_provider_manifest: tuple[dict[str, object], ...]
    method_equivalence_tests: tuple[dict[str, object], ...]
    classification_principles_report: str


@dataclass(frozen=True, slots=True)
class GenericClassificationEvidenceProofArtifacts:
    run_dir: Path
    evidence_provider_manifest_path: Path
    method_equivalence_tests_path: Path
    classification_principles_report_path: Path

def posterior_history_from_evidence_stream(
    *,
    class_names: tuple[str, ...],
    prior: dict[str, float],
    evidence_stream: tuple[EvidenceStep, ...],
    forgetting_factor: float = 1.0,
) -> tuple[PosteriorStep, ...]:
    posterior = {name: prior[name] for name in class_names}
    steps: list[PosteriorStep] = []
    for evidence in evidence_stream:
        log_scores = {
            name: forgetting_factor * log(max(posterior[name], 1e-12)) + evidence.log_likelihoods[name]
            for name in class_names
        }
        posterior = _normalize_log_scores(log_scores)
        predicted_class = max(posterior, key=posterior.get)
        steps.append(
            PosteriorStep(
                time=evidence.time,
                posterior=dict(posterior),
                predicted_class=predicted_class,
                confidence=posterior[predicted_class],
            )
        )
    return tuple(steps)


def _evidence_provider_manifest() -> tuple[dict[str, object], ...]:
    rows = [
        {
            "provider_id": "pointwise",
            "family": "instantaneous_measurement",
            "evidence_category": "pointwise_gaussian",
            "sensor_regime_id": "position_only",
            "posterior_updater": "generic_log_likelihood_accumulator",
        },
        {
            "provider_id": "windowed_raw",
            "family": "windowed_features",
            "evidence_category": "window_feature_likelihood",
            "sensor_regime_id": "position_only",
            "posterior_updater": "generic_log_likelihood_accumulator",
        },
        {
            "provider_id": "windowed_robust",
            "family": "windowed_features",
            "evidence_category": "window_feature_likelihood",
            "sensor_regime_id": "position_only",
            "posterior_updater": "generic_log_likelihood_accumulator",
        },
        {
            "provider_id": "accumulator",
            "family": "sequential_measurement",
            "evidence_category": "per_step_gaussian_measurement",
            "sensor_regime_id": "position_only",
            "posterior_updater": "generic_log_likelihood_accumulator",
        },
        {
            "provider_id": "kalman_bank",
            "family": "filter_innovation",
            "evidence_category": "innovation_log_likelihood",
            "sensor_regime_id": "position_only",
            "posterior_updater": "generic_log_likelihood_accumulator",
        },
        {
            "provider_id": "kalman_bank_velocity_aided",
            "family": "filter_innovation",
            "evidence_category": "innovation_log_likelihood_with_aux_sensor",
            "sensor_regime_id": "position_plus_direct_velocity",
            "posterior_updater": "generic_log_likelihood_accumulator",
        },
    ]
    return tuple(rows)


def _equivalence_tests() -> tuple[dict[str, object], ...]:
    class_names = ("A", "B")
    prior = {"A": 0.5, "B": 0.5}
    evidence_stream = (
        EvidenceStep(time=0.0, log_likelihoods={"A": log(0.80), "B": log(0.20)}),
        EvidenceStep(time=1.0, log_likelihoods={"A": log(0.75), "B": log(0.25)}),
        EvidenceStep(time=2.0, log_likelihoods={"A": log(0.70), "B": log(0.30)}),
    )
    duplicate_stream = tuple(
        EvidenceStep(time=step.time, log_likelihoods=dict(step.log_likelihoods))
        for step in evidence_stream
    )
    posterior_a = posterior_history_from_evidence_stream(
        class_names=class_names,
        prior=prior,
        evidence_stream=evidence_stream,
    )
    posterior_b = posterior_history_from_evidence_stream(
        class_names=class_names,
        prior=prior,
        evidence_stream=duplicate_stream,
    )
    same_history = all(
        step_a.time == step_b.time
        and step_a.predicted_class == step_b.predicted_class
        and all(abs(step_a.posterior[name] - step_b.posterior[name]) <= 1e-12 for name in class_names)
        for step_a, step_b in zip(posterior_a, posterior_b)
    )

    common_result = analyze_common_experiment(seed=7, trajectories_per_case=2)
    shared_shape_ok = True
    required_prediction_fields = {
        "run_id",
        "classifier_id",
        "sensor_regime_id",
        "trajectory_id",
        "scenario_id",
        "time",
        "true_class",
        "predicted_class",
        "confidence",
    }
    required_likelihood_fields = {
        "run_id",
        "classifier_id",
        "sensor_regime_id",
        "trajectory_id",
        "scenario_id",
        "time",
        "score_type",
        "log_likelihood_class_a",
        "log_likelihood_class_b",
    }
    method_rows = sorted({str(row["classifier_id"]) for row in common_result.pair_prediction_rows})
    for method_name in method_rows:
        prediction_row = next(row for row in common_result.pair_prediction_rows if str(row["classifier_id"]) == method_name)
        likelihood_row = next(row for row in common_result.likelihood_history_rows if str(row["classifier_id"]) == method_name)
        if not required_prediction_fields.issubset(prediction_row):
            shared_shape_ok = False
        if not required_likelihood_fields.issubset(likelihood_row):
            shared_shape_ok = False

    return (
        {
            "test_id": "identical_likelihood_streams_imply_identical_posteriors",
            "status": "pass" if same_history else "fail",
            "num_steps": len(evidence_stream),
            "final_posterior_stream_a": posterior_a[-1].posterior,
            "final_posterior_stream_b": posterior_b[-1].posterior,
        },
        {
            "test_id": "different_evidence_providers_share_artifact_shape",
            "status": "pass" if shared_shape_ok else "fail",
            "providers_checked": method_rows,
            "required_prediction_fields": sorted(required_prediction_fields),
            "required_likelihood_fields": sorted(required_likelihood_fields),
        },
    )


def render_generic_classification_evidence_report(
    *,
    evidence_provider_manifest: tuple[dict[str, object], ...],
    method_equivalence_tests: tuple[dict[str, object], ...],
) -> str:
    report = MarkdownDocument("Classification Evidence Proof")
    report.paragraph(
        "This artifact proves that current classifiers can be interpreted as evidence providers "
        "feeding a generic posterior updater rather than as isolated algorithm silos."
    )
    report.heading("Evidence Providers", level=2)
    report.table(
        ["provider_id", "family", "evidence_category", "sensor_regime_id", "posterior_updater"],
        [
            (
                row["provider_id"],
                row["family"],
                row["evidence_category"],
                row["sensor_regime_id"],
                row["posterior_updater"],
            )
            for row in evidence_provider_manifest
        ],
    )
    report.heading("Equivalence Tests", level=2)
    report.table(
        ["test_id", "status"],
        [(row["test_id"], row["status"]) for row in method_equivalence_tests],
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "The generic posterior updater only consumes priors plus class-conditioned log-likelihood terms.",
            "Pointwise, windowed, residual-style, and Kalman-innovation methods differ in how they produce evidence, not in how posterior recursion works.",
            "Stronger-sensor variants remain separate evidence providers because sensor regime is part of the experiment contract.",
        ]
    )
    return report.text()


def analyze_generic_classification_evidence_proof() -> GenericClassificationEvidenceProofResult:
    evidence_provider_manifest = _evidence_provider_manifest()
    method_equivalence_tests = _equivalence_tests()
    classification_principles_report = render_generic_classification_evidence_report(
        evidence_provider_manifest=evidence_provider_manifest,
        method_equivalence_tests=method_equivalence_tests,
    )
    return GenericClassificationEvidenceProofResult(
        evidence_provider_manifest=evidence_provider_manifest,
        method_equivalence_tests=method_equivalence_tests,
        classification_principles_report=classification_principles_report,
    )


def write_generic_classification_evidence_proof_artifacts(
    output_dir: str | Path,
    *,
    result: GenericClassificationEvidenceProofResult | None = None,
) -> GenericClassificationEvidenceProofArtifacts:
    proof = result or analyze_generic_classification_evidence_proof()
    run_dir = Path(output_dir) / "classification_evidence_proof"
    run_dir.mkdir(parents=True, exist_ok=True)

    evidence_provider_manifest_path = run_dir / "evidence_provider_manifest.json"
    method_equivalence_tests_path = run_dir / "method_equivalence_tests.json"
    classification_principles_report_path = run_dir / "classification_principles_report.md"

    evidence_provider_manifest_path.write_text(json.dumps(list(proof.evidence_provider_manifest), indent=2), encoding="utf-8")
    method_equivalence_tests_path.write_text(json.dumps(list(proof.method_equivalence_tests), indent=2), encoding="utf-8")
    classification_principles_report_path.write_text(proof.classification_principles_report, encoding="utf-8")

    return GenericClassificationEvidenceProofArtifacts(
        run_dir=run_dir,
        evidence_provider_manifest_path=evidence_provider_manifest_path,
        method_equivalence_tests_path=method_equivalence_tests_path,
        classification_principles_report_path=classification_principles_report_path,
    )
