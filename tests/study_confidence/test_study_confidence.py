from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.study_confidence import (
    analyze_study_confidence,
    write_study_confidence_artifacts,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    text_lines = [",".join(header)]
    for row in rows:
        text_lines.append(",".join(str(value) for value in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


class StudyConfidenceTests(unittest.TestCase):
    def test_confidence_is_capped_by_failed_corpus_gate(self) -> None:
        study = {"study_id": "synthetic", "title": "Synthetic"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "synthetic"
            audit_dir = root / "03_corpus_audit"
            ladder_dir = root / "04_ladder_evaluation"
            audit_dir.mkdir(parents=True, exist_ok=True)
            ladder_dir.mkdir(parents=True, exist_ok=True)

            (audit_dir / "corpus_decision_gate.json").write_text(
                json.dumps(
                    {
                        "overall_status": "fail",
                        "overall_pass": False,
                        "recommendation_count": 2,
                        "class_validity": {
                            "total_rows": 10,
                            "valid_target_class_fraction": 0.9,
                            "ambiguous_fraction": 0.0,
                            "relabel_candidate_fraction": 0.0,
                            "invalid_fraction": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (audit_dir / "corpus_adequacy_summary.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "overall_status": "fail",
                            "overall_pass": False,
                            "q_corpus": 0.88,
                            "class_validity_score": 0.91,
                            "feature_status": "pass",
                            "class_pair_status": "pass",
                            "class_balance_status": "pass",
                            "leakage_penalty": 0.05,
                            "degeneracy_penalty": 0.05,
                            "triviality_penalty": 0.05,
                        }
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(audit_dir / "leakage_audit.csv", ["feature_name", "status"], [["speed", "green"]])
            _write_csv(
                ladder_dir / "method_metrics.csv",
                ["classifier_id", "rung_id", "feature_set_id", "class_pair_id", "num_predictions", "overall_accuracy", "mean_confidence"],
                [["kalman_bank", "kalman_bank", "model_residuals", "stationary_vs_constant_velocity", 20, 0.95, 0.96]],
            )
            _write_csv(
                ladder_dir / "sufficiency_matrix.csv",
                [
                    "study_id",
                    "class_pair_id",
                    "feature_set_id",
                    "classifier_id",
                    "current_rung_id",
                    "candidate_next_rung_id",
                    "current_accuracy",
                    "oracle_accuracy",
                    "oracle_gap",
                    "measured_next_accuracy",
                    "measured_improvement",
                    "runtime_cost_ratio",
                    "decision",
                    "rationale",
                ],
                [[
                    "synthetic",
                    "stationary_vs_constant_velocity",
                    "model_residuals",
                    "kalman_bank",
                    "kalman_bank",
                    "transition_matrix",
                    0.95,
                    0.97,
                    0.02,
                    0.96,
                    0.01,
                    1.1,
                    "stay",
                    "close to oracle",
                ]],
            )
            _write_csv(
                ladder_dir / "insufficiency_matrix.csv",
                ["study_id", "class_pair_id", "feature_set_id", "classifier_id", "current_rung_id", "candidate_next_rung_id", "failure_mode", "failure_rationale"],
                [["synthetic", "stationary_vs_constant_velocity", "model_residuals", "kalman_bank", "kalman_bank", "transition_matrix", "corpus_limited", "corpus gate failed"]],
            )
            _write_csv(
                ladder_dir / "prior_sensitivity_by_method.csv",
                ["classifier_id", "class_pair_id", "prior_id", "accuracy"],
                [["kalman_bank", "stationary_vs_constant_velocity", "uniform", 0.95], ["kalman_bank", "stationary_vs_constant_velocity", "mild_bias", 0.94]],
            )

            result = analyze_study_confidence(root, study)
            self.assertEqual(len(result.classifier_rows), 1)
            self.assertLess(float(result.classifier_rows[0]["final_confidence"]), 0.35)
            self.assertEqual(result.classifier_rows[0]["confidence_band"], "blocked")

    def test_artifact_writer_emits_expected_files(self) -> None:
        study = {"study_id": "synthetic", "title": "Synthetic"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "synthetic"
            audit_dir = root / "03_corpus_audit"
            ladder_dir = root / "04_ladder_evaluation"
            audit_dir.mkdir(parents=True, exist_ok=True)
            ladder_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / "corpus_decision_gate.json").write_text(
                json.dumps(
                    {
                        "overall_status": "pass",
                        "overall_pass": True,
                        "recommendation_count": 0,
                        "class_validity": {
                            "total_rows": 10,
                            "valid_target_class_fraction": 1.0,
                            "ambiguous_fraction": 0.0,
                            "relabel_candidate_fraction": 0.0,
                            "invalid_fraction": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (audit_dir / "corpus_adequacy_summary.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "overall_status": "pass",
                            "overall_pass": True,
                            "q_corpus": 0.92,
                            "class_validity_score": 0.94,
                            "feature_status": "pass",
                            "class_pair_status": "pass",
                            "class_balance_status": "pass",
                            "leakage_penalty": 0.02,
                            "degeneracy_penalty": 0.03,
                            "triviality_penalty": 0.02,
                        }
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(audit_dir / "leakage_audit.csv", ["feature_name", "status"], [["speed", "green"]])
            _write_csv(
                ladder_dir / "method_metrics.csv",
                ["classifier_id", "rung_id", "feature_set_id", "class_pair_id", "num_predictions", "overall_accuracy", "mean_confidence"],
                [["bayes_accumulator", "sequential_bayes", "shape_window", "stationary_vs_constant_velocity", 20, 0.84, 0.82]],
            )
            _write_csv(
                ladder_dir / "sufficiency_matrix.csv",
                [
                    "study_id",
                    "class_pair_id",
                    "feature_set_id",
                    "classifier_id",
                    "current_rung_id",
                    "candidate_next_rung_id",
                    "current_accuracy",
                    "oracle_accuracy",
                    "oracle_gap",
                    "measured_next_accuracy",
                    "measured_improvement",
                    "runtime_cost_ratio",
                    "decision",
                    "rationale",
                ],
                [[
                    "synthetic",
                    "stationary_vs_constant_velocity",
                    "shape_window",
                    "bayes_accumulator",
                    "sequential_bayes",
                    "kalman_bank",
                    0.84,
                    0.88,
                    0.04,
                    0.84,
                    0.00,
                    1.0,
                    "stay",
                    "adequate",
                ]],
            )
            _write_csv(
                ladder_dir / "insufficiency_matrix.csv",
                ["study_id", "class_pair_id", "feature_set_id", "classifier_id", "current_rung_id", "candidate_next_rung_id", "failure_mode", "failure_rationale"],
                [["synthetic", "stationary_vs_constant_velocity", "shape_window", "bayes_accumulator", "sequential_bayes", "kalman_bank", "model_limited", "none"]],
            )
            _write_csv(
                ladder_dir / "prior_sensitivity_by_method.csv",
                ["classifier_id", "class_pair_id", "prior_id", "accuracy"],
                [["bayes_accumulator", "stationary_vs_constant_velocity", "uniform", 0.84], ["bayes_accumulator", "stationary_vs_constant_velocity", "mild_bias", 0.80]],
            )

            artifacts = write_study_confidence_artifacts(root / "04b_confidence", workflow_root=root, study=study)
            self.assertTrue(artifacts.components_path.exists())
            self.assertTrue(artifacts.classifier_scores_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.dashboard_path.exists())


if __name__ == "__main__":
    unittest.main()
