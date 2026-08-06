from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.static_admissibility.audit import run_static_admissibility_audit
from kinematic_classifier_sandbox.static_admissibility.validation import (
    validate_static_admissibility_packet,
)


class StaticAdmissibilityPacketTests(unittest.TestCase):
    def test_static_admissibility_packet_contains_required_mvp_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet = run_static_admissibility_audit(
                "experiments/static_admissibility/common_1d_static_audit.yaml",
                Path(temp_dir) / "static_admissibility_mvp",
            )

            self.assertTrue(packet.readme_path.exists())
            self.assertTrue(packet.decision_card_path.exists())
            self.assertTrue(packet.static_audit_report_path.exists())
            self.assertTrue(packet.static_audit_decision_card_path.exists())
            self.assertTrue(packet.figure_manifest_path.exists())
            self.assertTrue(packet.lane_proof_matrix_path.exists())
            self.assertTrue(packet.contact_sheet_path.exists())
            self.assertTrue((packet.packet_dir / "02b_static_audit_decision_card.png").exists())
            self.assertTrue((packet.packet_dir / "02c_class_pair_confusability_matrix.png").exists())
            self.assertTrue((packet.packet_dir / "02e_feature_redundancy_graph.png").exists())
            self.assertTrue((packet.packet_dir / "02g_prior_pathology_surface.png").exists())
            self.assertTrue((packet.packet_dir / "class_pair_diagnostics.csv").exists())
            self.assertTrue((packet.packet_dir / "class_feature_signature.csv").exists())
            self.assertTrue((packet.packet_dir / "class_observability.csv").exists())
            self.assertTrue((packet.packet_dir / "feature_alias_candidates.csv").exists())
            self.assertTrue((packet.packet_dir / "prior_selection_balance.csv").exists())
            self.assertTrue((packet.packet_dir / "static_resolution_plan.csv").exists())
            self.assertEqual(validate_static_admissibility_packet(packet.packet_dir), [])

    def test_static_validator_rejects_zero_mass_declared_prior(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet = run_static_admissibility_audit(
                "experiments/static_admissibility/common_1d_static_audit.yaml",
                Path(temp_dir) / "static_admissibility_mvp",
            )
            prior_path = packet.packet_dir / "prior_regime.csv"
            lines = prior_path.read_text(encoding="utf-8").splitlines()
            first_data = lines[1].split(",")
            first_data[1] = "0.0"
            lines[1] = ",".join(first_data)
            prior_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            issues = validate_static_admissibility_packet(packet.packet_dir)
            self.assertTrue(any("zero mass" in issue for issue in issues))

    def test_static_admissibility_packet_accepts_file_backed_study_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet = run_static_admissibility_audit(
                "experiments/static_admissibility/repeatable_lane_demo/repeatable_lane_demo.yaml",
                Path(temp_dir) / "repeatable_lane_demo",
            )

            self.assertEqual(validate_static_admissibility_packet(packet.packet_dir), [])
            decision_text = packet.decision_card_path.read_text(encoding="utf-8")
            readme_text = packet.readme_path.read_text(encoding="utf-8")
            self.assertIn("repeatable_lane_demo", decision_text)
            self.assertIn("study_bundle_samples.csv", readme_text)
            self.assertTrue((packet.packet_dir / "study_bundle_source.yaml").exists())
            self.assertTrue((packet.packet_dir / "study_bundle_samples.csv").exists())
            self.assertTrue((packet.packet_dir / "study_bundle_feature_schema.csv").exists())
            self.assertTrue((packet.packet_dir / "study_bundle_class_schema.csv").exists())

    def test_static_admissibility_packet_accepts_declared_unobserved_future_class(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir) / "future_bundle"
            bundle_dir.mkdir()
            (bundle_dir / "study.yaml").write_text(
                """static_admissibility:
  study_id: future_class_bundle
  priors:
    class_a: 0.4
    class_b: 0.4
    future_class: 0.2
  input_bundle:
    dimension: 2d
    allow_unobserved_classes: true
    sample_table: samples.csv
    feature_schema: feature_schema.csv
    class_schema: class_schema.csv
    class_feature_signature: class_feature_signature.csv
    feature_names:
      - position_x
      - position_y
""",
                encoding="utf-8",
            )
            (bundle_dir / "class_schema.csv").write_text(
                "class_name,notes\nclass_a,observed\nclass_b,observed\nfuture_class,planned\n",
                encoding="utf-8",
            )
            (bundle_dir / "feature_schema.csv").write_text(
                "feature_name,provenance_tags,online_available,label_rule_overlap\n"
                "position_x,online,true,false\n"
                "position_y,online,true,false\n",
                encoding="utf-8",
            )
            (bundle_dir / "samples.csv").write_text(
                "sample_id,true_class,position_x,position_y\n"
                "a0,class_a,0.0,0.0\n"
                "a1,class_a,0.1,0.0\n"
                "a2,class_a,-0.1,0.0\n"
                "b0,class_b,1.0,1.0\n"
                "b1,class_b,1.1,1.0\n"
                "b2,class_b,0.9,1.0\n",
                encoding="utf-8",
            )
            (bundle_dir / "class_feature_signature.csv").write_text(
                "class_name,feature_name,expected_mean,expected_std,source\n"
                "future_class,position_x,2.0,0.2,design_prior\n"
                "future_class,position_y,2.0,0.2,design_prior\n",
                encoding="utf-8",
            )

            packet = run_static_admissibility_audit(
                bundle_dir / "study.yaml",
                Path(temp_dir) / "future_packet",
            )

            self.assertEqual(validate_static_admissibility_packet(packet.packet_dir), [])
            self.assertIn("unobserved_class", (packet.packet_dir / "class_observability.csv").read_text(encoding="utf-8"))
            self.assertTrue((packet.packet_dir / "study_bundle_class_feature_signature.csv").exists())


if __name__ == "__main__":
    unittest.main()
