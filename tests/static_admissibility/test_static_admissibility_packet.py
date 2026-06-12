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


if __name__ == "__main__":
    unittest.main()
