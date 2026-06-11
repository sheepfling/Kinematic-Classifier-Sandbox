from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit import (
    StaticAuditFeatureSchemaEntry,
    StaticAuditSample,
    analyze_static_feature_class_prior_audit,
    render_static_feature_class_prior_audit_report,
)
from kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit_artifact_io import (
    write_static_feature_class_prior_audit_artifacts,
)


def _sample(class_name: str, index: int, speed: float, accel: float, signs: float) -> StaticAuditSample:
    return StaticAuditSample(
        true_class=class_name,
        sample_id=f"{class_name}_{index}",
        feature_values={
            "speed_mean": speed,
            "speed_final": speed + 0.01,
            "acceleration_variance": accel,
            "sign_changes": signs,
        },
    )


class StaticFeatureClassPriorAuditTests(unittest.TestCase):
    def test_static_audit_promotes_separable_feature_class_prior_candidate(self) -> None:
        samples = [
            _sample("stationary", 0, 0.00, 0.00, 0.0),
            _sample("stationary", 1, 0.02, 0.01, 0.0),
            _sample("stationary", 2, 0.03, 0.00, 0.0),
            _sample("stationary", 3, 0.01, 0.01, 0.0),
            _sample("constant_velocity", 0, 1.00, 0.02, 0.0),
            _sample("constant_velocity", 1, 1.10, 0.03, 0.0),
            _sample("constant_velocity", 2, 0.95, 0.02, 0.0),
            _sample("constant_velocity", 3, 1.05, 0.02, 0.0),
            _sample("maneuver", 0, 1.00, 1.10, 2.0),
            _sample("maneuver", 1, 1.10, 1.20, 3.0),
            _sample("maneuver", 2, 0.90, 1.00, 2.0),
            _sample("maneuver", 3, 1.05, 1.30, 3.0),
        ]

        result = analyze_static_feature_class_prior_audit(
            samples,
            priors={"stationary": 0.34, "constant_velocity": 0.33, "maneuver": 0.33},
            feature_schema=(
                StaticAuditFeatureSchemaEntry("speed_mean", provenance_tags=("online", "finite_difference")),
                StaticAuditFeatureSchemaEntry("speed_final", provenance_tags=("online", "finite_difference")),
                StaticAuditFeatureSchemaEntry("acceleration_variance", provenance_tags=("online", "residual")),
                StaticAuditFeatureSchemaEntry("sign_changes", provenance_tags=("online", "shape")),
            ),
            study_name="controlled_kinematic_static_gate",
        )
        report = render_static_feature_class_prior_audit_report(result)

        self.assertIn("Static Feature/Class/Prior Audit", report)
        self.assertEqual(result.static_decision["status"], "promote_to_corpus_explorer")
        self.assertEqual(result.static_decision["adequacy_label"], "sufficient_for_corpus_search")
        self.assertEqual(len(result.class_pair_rows), 3)
        self.assertTrue(all(row["pairwise_auc"] >= 0.9 for row in result.class_pair_rows))
        self.assertTrue(any(row["feature"] == "acceleration_variance" for row in result.feature_relevance_rows))
        self.assertTrue(any(row["status"] == "high_redundancy" for row in result.feature_redundancy_rows))
        self.assertTrue(all(row["status"] == "pass" for row in result.leakage_rows))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_static_feature_class_prior_audit_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "static_feature_class_prior_audit_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertTrue(artifacts.decision_card_png_path.exists())
            self.assertTrue(artifacts.class_confusability_matrix_path.exists())
            self.assertTrue(artifacts.feature_relevance_table_path.exists())
            self.assertTrue(artifacts.feature_redundancy_matrix_path.exists())
            self.assertTrue(artifacts.feature_synergy_candidates_path.exists())
            self.assertTrue(artifacts.prior_pathology_report_path.exists())
            self.assertTrue(artifacts.coverage_static_report_path.exists())
            self.assertTrue(artifacts.leakage_static_report_path.exists())
            decision_text = artifacts.decision_card_path.read_text(encoding="utf-8")
            self.assertIn("decisionability", decision_text)

    def test_static_audit_exposes_prior_domination_in_pair_table(self) -> None:
        samples = [
            StaticAuditSample("common", {"weak_feature": 0.00}),
            StaticAuditSample("common", {"weak_feature": 0.02}),
            StaticAuditSample("common", {"weak_feature": -0.02}),
            StaticAuditSample("rare", {"weak_feature": 0.05}),
            StaticAuditSample("rare", {"weak_feature": 0.06}),
            StaticAuditSample("rare", {"weak_feature": 0.04}),
        ]

        result = analyze_static_feature_class_prior_audit(
            samples,
            priors={"common": 0.995, "rare": 0.005},
            study_name="rare_class_prior_gate",
        )

        self.assertTrue(
            any(row["pathology_flag"] == "prior_domination" for row in result.prior_pathology_rows)
        )
        self.assertIn(result.static_decision["status"], {"revise_class_set", "revise_prior"})

    def test_static_audit_rejects_label_rule_leakage(self) -> None:
        samples = [
            StaticAuditSample("a", {"label_code": 0.0, "useful": 0.0}),
            StaticAuditSample("a", {"label_code": 0.0, "useful": 0.1}),
            StaticAuditSample("b", {"label_code": 1.0, "useful": 1.0}),
            StaticAuditSample("b", {"label_code": 1.0, "useful": 1.1}),
        ]

        result = analyze_static_feature_class_prior_audit(
            samples,
            feature_schema=(
                StaticAuditFeatureSchemaEntry("label_code", label_rule_overlap=True),
                StaticAuditFeatureSchemaEntry("useful"),
            ),
        )

        self.assertEqual(result.static_decision["status"], "reject")
        self.assertEqual(result.static_decision["adequacy_label"], "insufficient_due_to_leakage_risk")
        self.assertTrue(any(row["status"] == "blocker" for row in result.leakage_rows))


if __name__ == "__main__":
    unittest.main()
