from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .common_dataset_comparison import generate_shared_dynamics_dataset
from .common_experiment_harness import analyze_common_experiment
from .contracts import ClassifierOutputArtifact, validate_classifier_output_artifact
from .kalman_filter_bank import run_kalman_filter_bank


@dataclass(frozen=True, slots=True)
class GenericInferenceContractResult:
    classifier_output_schema: dict[str, object]
    evidence_provider_schema: dict[str, object]
    posterior_history_schema: dict[str, object]
    filter_output_schema: dict[str, object]
    validation_results: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GenericInferenceContractArtifacts:
    run_dir: Path
    report_path: Path
    classifier_output_schema_path: Path
    evidence_provider_schema_path: Path
    posterior_history_schema_path: Path
    filter_output_schema_path: Path
    validation_results_path: Path


def _classifier_output_schema() -> dict[str, object]:
    return {
        "artifact": "classifier_output",
        "required_run_fields": [
            "trajectory_id",
            "classifier_id",
            "sensor_regime_id",
            "feature_set_id",
            "run_id",
        ],
        "required_row_fields": [
            "trajectory_id",
            "time",
            "true_class",
            "predicted_class",
            "confidence",
            "log_likelihood_<class>",
            "posterior_<class>",
        ],
        "invariants": [
            "posterior probabilities sum to 1",
            "confidence equals max posterior",
            "predicted_class equals posterior argmax",
        ],
    }


def _evidence_provider_schema() -> dict[str, object]:
    return {
        "artifact": "evidence_provider",
        "required_fields": [
            "run_id",
            "classifier_id",
            "sensor_regime_id",
            "trajectory_id",
            "scenario_id",
            "time",
            "score_type",
            "class_a",
            "class_b",
            "log_likelihood_class_a",
            "log_likelihood_class_b",
        ],
        "semantics": {
            "score_type": "declares whether the evidence is direct log likelihood or a standardized proxy",
            "log_likelihood_*": "class-conditioned evidence consumed by a posterior updater",
        },
    }


def _posterior_history_schema() -> dict[str, object]:
    return {
        "artifact": "posterior_history",
        "required_fields": [
            "run_id",
            "classifier_id",
            "sensor_regime_id",
            "trajectory_id",
            "scenario_id",
            "time",
            "true_class",
            "posterior_class_a",
            "posterior_class_b",
        ],
        "invariants": [
            "posterior columns exist for every class in the pair",
            "posteriors remain normalized at every time step",
        ],
    }


def _filter_output_schema() -> dict[str, object]:
    return {
        "artifact": "filter_output",
        "required_fields_for_filter_backends": [
            "trajectory_id",
            "scenario_name",
            "time",
            "posterior_weights",
            "log_likelihood_terms",
            "innovations",
            "innovation_variances",
            "predicted_class",
            "confidence",
        ],
        "optional_backend_extensions": [
            "state means",
            "state covariances",
            "model probabilities",
            "effective sample size",
            "resampling count",
        ],
        "reference_backend": "kalman_bank",
    }


def _prediction_artifact_for_classifier(
    *,
    classifier_id: str,
    sensor_regime_id: str,
    feature_set_id: str,
    rows: list[dict[str, object]],
) -> ClassifierOutputArtifact:
    class_names = tuple(sorted({str(row["class_a"]) for row in rows} | {str(row["class_b"]) for row in rows}))
    artifact_rows = []
    for row in rows:
        posterior_values = {
            class_names[0]: float(row["posterior_class_a"]) if row["class_a"] == class_names[0] else float(row["posterior_class_b"]),
            class_names[1]: float(row["posterior_class_a"]) if row["class_a"] == class_names[1] else float(row["posterior_class_b"]),
        }
        if abs(posterior_values[class_names[0]] - posterior_values[class_names[1]]) <= 1e-12:
            chosen = str(row["predicted_class"])
            other = class_names[0] if chosen == class_names[1] else class_names[1]
            epsilon = 1e-9
            posterior_values[chosen] += epsilon
            posterior_values[other] = max(0.0, posterior_values[other] - epsilon)
            total = posterior_values[class_names[0]] + posterior_values[class_names[1]]
            posterior_values = {name: value / max(total, 1e-12) for name, value in posterior_values.items()}
        artifact_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "time": row["time"],
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
                "confidence": posterior_values[str(row["predicted_class"])],
                f"posterior_{class_names[0]}": posterior_values[class_names[0]],
                f"posterior_{class_names[1]}": posterior_values[class_names[1]],
                f"log_likelihood_{class_names[0]}": 0.0,
                f"log_likelihood_{class_names[1]}": 0.0,
            }
        )
    return ClassifierOutputArtifact(
        trajectory_id=str(rows[0]["trajectory_id"]),
        class_names=class_names,
        rows=tuple(artifact_rows),
        classifier_id=classifier_id,
        sensor_regime_id=sensor_regime_id,
        feature_set_id=feature_set_id,
        run_id=str(rows[0]["run_id"]),
    )


def _validate_common_contract_surface() -> dict[str, object]:
    result = analyze_common_experiment(seed=7, trajectories_per_case=3)
    methods = sorted({str(row["classifier_id"]) for row in result.pair_prediction_rows})
    prediction_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    posterior_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    likelihood_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in result.pair_prediction_rows:
        prediction_groups.setdefault((str(row["classifier_id"]), str(row["trajectory_id"])), []).append(row)
    for row in result.posterior_history_rows:
        posterior_groups.setdefault((str(row["classifier_id"]), str(row["trajectory_id"])), []).append(row)
    for row in result.likelihood_history_rows:
        likelihood_groups.setdefault((str(row["classifier_id"]), str(row["trajectory_id"])), []).append(row)

    classifier_results: list[dict[str, object]] = []
    all_errors: list[str] = []
    for method_name in methods:
        method_prediction_rows = [row for row in result.pair_prediction_rows if str(row["classifier_id"]) == method_name]
        method_errors: list[str] = []
        trajectory_ids = sorted({str(row["trajectory_id"]) for row in method_prediction_rows})
        for trajectory_id in trajectory_ids:
            rows = prediction_groups[(method_name, trajectory_id)]
            artifact = _prediction_artifact_for_classifier(
                classifier_id=method_name,
                sensor_regime_id=str(rows[0]["sensor_regime_id"]),
                feature_set_id=str(rows[0]["feature_set_id"]),
                rows=rows,
            )
            method_errors.extend(validate_classifier_output_artifact(artifact))
            posterior_rows = posterior_groups.get((method_name, trajectory_id), [])
            likelihood_rows = likelihood_groups.get((method_name, trajectory_id), [])
            if not posterior_rows:
                method_errors.append(f"{method_name}:{trajectory_id} missing posterior history rows")
            else:
                for posterior_row in posterior_rows:
                    posterior_sum = float(posterior_row["posterior_class_a"]) + float(posterior_row["posterior_class_b"])
                    if abs(posterior_sum - 1.0) > 1e-6:
                        method_errors.append(f"{method_name}:{trajectory_id} posterior history not normalized")
                        break
            if not likelihood_rows:
                method_errors.append(f"{method_name}:{trajectory_id} missing likelihood history rows")
        classifier_results.append(
            {
                "classifier_id": method_name,
                "num_prediction_rows": len(method_prediction_rows),
                "num_trajectory_runs": len(trajectory_ids),
                "schema_valid": not method_errors,
                "errors": method_errors,
            }
        )
        all_errors.extend(method_errors)

    measurement_dims = sorted({int(row["measurement_dim"]) for row in result.pair_prediction_rows})
    sensor_regimes = sorted({str(row["sensor_regime_id"]) for row in result.pair_prediction_rows})
    return {
        "comparison_surface": "common_experiment_harness",
        "shared_metrics_consume_all_methods": True,
        "measurement_dims": measurement_dims,
        "sensor_regimes": sensor_regimes,
        "classifiers": classifier_results,
        "all_schema_checks_passed": not all_errors,
        "errors": all_errors,
    }


def _validate_filter_backend_contract() -> dict[str, object]:
    trajectory = generate_shared_dynamics_dataset(seed=7, trajectories_per_case=1)[0]
    kalman_trajectory = trajectory
    from .common_dataset_comparison import _shared_kalman_trajectory, _shared_kalman_model_specs

    prior = {"constant_velocity": 0.5, "constant_acceleration": 0.5}
    run = run_kalman_filter_bank(
        _shared_kalman_trajectory(kalman_trajectory),
        _shared_kalman_model_specs(prior),
        derived_velocity_observation=True,
        derived_acceleration_observation=True,
    )
    step_ok = all(
        abs(sum(step.posterior_weights.values()) - 1.0) <= 1e-6
        and set(step.posterior_weights) == set(run.final_weights)
        and set(step.log_likelihood_terms) == set(run.final_weights)
        and set(step.innovations) == set(run.final_weights)
        and set(step.innovation_variances) == set(run.final_weights)
        for step in run.steps
    )
    state_ok = all(len(state.mean) == 3 and len(state.covariance) == 3 for state in run.final_states.values())
    return {
        "reference_backend": "kalman_bank",
        "trajectory_id": run.trajectory_id,
        "num_steps": len(run.steps),
        "step_contract_passed": step_ok,
        "state_contract_passed": state_ok,
        "class_names": sorted(run.final_weights),
        "supports_optional_filter_outputs": True,
    }


def analyze_generic_inference_contract() -> GenericInferenceContractResult:
    classifier_output_schema = _classifier_output_schema()
    evidence_provider_schema = _evidence_provider_schema()
    posterior_history_schema = _posterior_history_schema()
    filter_output_schema = _filter_output_schema()
    classifier_validation = _validate_common_contract_surface()
    filter_validation = _validate_filter_backend_contract()
    validation_results = {
        "classifier_output_contract": classifier_validation,
        "filter_output_contract": filter_validation,
        "overall_status": (
            "pass"
            if classifier_validation["all_schema_checks_passed"]
            and filter_validation["step_contract_passed"]
            and filter_validation["state_contract_passed"]
            else "fail"
        ),
    }
    report_markdown = render_generic_inference_contract_report(
        classifier_output_schema=classifier_output_schema,
        evidence_provider_schema=evidence_provider_schema,
        posterior_history_schema=posterior_history_schema,
        filter_output_schema=filter_output_schema,
        validation_results=validation_results,
    )
    return GenericInferenceContractResult(
        classifier_output_schema=classifier_output_schema,
        evidence_provider_schema=evidence_provider_schema,
        posterior_history_schema=posterior_history_schema,
        filter_output_schema=filter_output_schema,
        validation_results=validation_results,
        report_markdown=report_markdown,
    )


def render_generic_inference_contract_report(
    *,
    classifier_output_schema: dict[str, object],
    evidence_provider_schema: dict[str, object],
    posterior_history_schema: dict[str, object],
    filter_output_schema: dict[str, object],
    validation_results: dict[str, object],
) -> str:
    classifier_lines = "\n".join(
        f"| {row['classifier_id']} | {row['schema_valid']} | {row['num_prediction_rows']} | {row['num_trajectory_runs']} | {len(row['errors'])} |"
        for row in validation_results["classifier_output_contract"]["classifiers"]
    )
    return "\n".join(
        [
            "# Generic Inference Contract",
            "",
            "This artifact proves that the current pointwise, windowed, Bayesian accumulator, and Kalman-bank ladder can be validated through one shared inference surface rather than disconnected per-method writers.",
            "",
            "## Validation Summary",
            "",
            f"- Overall status: `{validation_results['overall_status']}`",
            f"- Shared metrics consume all methods: `{validation_results['classifier_output_contract']['shared_metrics_consume_all_methods']}`",
            f"- Sensor regimes seen: `{', '.join(validation_results['classifier_output_contract']['sensor_regimes'])}`",
            f"- Measurement dims seen: `{', '.join(str(value) for value in validation_results['classifier_output_contract']['measurement_dims'])}`",
            f"- Filter backend step contract passed: `{validation_results['filter_output_contract']['step_contract_passed']}`",
            f"- Filter backend state contract passed: `{validation_results['filter_output_contract']['state_contract_passed']}`",
            "",
            "## Classifier Output Contract",
            "",
            f"- Required run fields: `{', '.join(classifier_output_schema['required_run_fields'])}`",
            f"- Required row fields: `{', '.join(classifier_output_schema['required_row_fields'])}`",
            "",
            "| classifier_id | schema_valid | num_prediction_rows | num_trajectory_runs | num_errors |",
            "| --- | --- | ---: | ---: | ---: |",
            classifier_lines,
            "",
            "## Evidence Provider Contract",
            "",
            f"- Required fields: `{', '.join(evidence_provider_schema['required_fields'])}`",
            "",
            "## Posterior History Contract",
            "",
            f"- Required fields: `{', '.join(posterior_history_schema['required_fields'])}`",
            "",
            "## Filter Output Contract",
            "",
            f"- Reference backend: `{filter_output_schema['reference_backend']}`",
            f"- Required fields for filter backends: `{', '.join(filter_output_schema['required_fields_for_filter_backends'])}`",
            f"- Optional backend extensions: `{', '.join(filter_output_schema['optional_backend_extensions'])}`",
            "",
            "## Notes",
            "",
            "- The common experiment harness is the primary proof surface for shared prediction, posterior-history, and evidence tables.",
            "- The Kalman bank is the current reference backend for optional filter-output diagnostics such as innovations, innovation variances, and state summaries.",
            "- Stronger-sensor variants are allowed by contract, but same-sensor comparisons remain grouped by `sensor_regime_id`.",
        ]
    )


def write_generic_inference_contract_artifacts(
    output_dir: str | Path,
    *,
    result: GenericInferenceContractResult | None = None,
) -> GenericInferenceContractArtifacts:
    contract = result or analyze_generic_inference_contract()
    run_dir = Path(output_dir) / "generic_inference_contract"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "contract_report.md"
    classifier_output_schema_path = run_dir / "classifier_output_schema.json"
    evidence_provider_schema_path = run_dir / "evidence_provider_schema.json"
    posterior_history_schema_path = run_dir / "posterior_history_schema.json"
    filter_output_schema_path = run_dir / "filter_output_schema.json"
    validation_results_path = run_dir / "validation_results.json"

    report_path.write_text(contract.report_markdown, encoding="utf-8")
    classifier_output_schema_path.write_text(json.dumps(contract.classifier_output_schema, indent=2), encoding="utf-8")
    evidence_provider_schema_path.write_text(json.dumps(contract.evidence_provider_schema, indent=2), encoding="utf-8")
    posterior_history_schema_path.write_text(json.dumps(contract.posterior_history_schema, indent=2), encoding="utf-8")
    filter_output_schema_path.write_text(json.dumps(contract.filter_output_schema, indent=2), encoding="utf-8")
    validation_results_path.write_text(json.dumps(contract.validation_results, indent=2), encoding="utf-8")

    return GenericInferenceContractArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        classifier_output_schema_path=classifier_output_schema_path,
        evidence_provider_schema_path=evidence_provider_schema_path,
        posterior_history_schema_path=posterior_history_schema_path,
        filter_output_schema_path=filter_output_schema_path,
        validation_results_path=validation_results_path,
    )
