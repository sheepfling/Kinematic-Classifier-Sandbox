from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.static_admissibility.exemplar_suite import (
    write_static_admissibility_exemplar_suite_packet,
)
from kinematic_classifier_sandbox.static_admissibility.validation import (
    validate_static_admissibility_packet,
)
from kinematic_classifier_sandbox.utils.runtime import repo_root


class StaticAdmissibilityExemplarSuitePacketTests(unittest.TestCase):
    def test_exemplar_suite_packet_builds_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet = write_static_admissibility_exemplar_suite_packet(
                Path(temp_dir) / "01_static_admissibility",
            )
            self.assertTrue(packet.readme_path.exists())
            self.assertTrue(packet.quickstart_path.exists())
            self.assertTrue(packet.packet_manifest_path.exists())
            self.assertTrue(packet.decision_card_path.exists())
            self.assertTrue(packet.validation_report_path.exists())
            self.assertTrue(packet.claim_boundary_path.exists())
            self.assertTrue(packet.hero_chart_manifest_path.exists())
            self.assertTrue(packet.lane_proof_matrix_path.exists())
            self.assertTrue(packet.automated_brief_path.exists())
            self.assertTrue(packet.executive_brief_path.exists())
            executive_brief = packet.executive_brief_path.read_text(encoding="utf-8")
            self.assertIn("What it screens upfront", executive_brief)
            self.assertIn("PRIOR_SELECTION_SKEW", executive_brief)
            self.assertIn("future_constant_velocity", executive_brief)
            self.assertTrue(packet.source_manifest_path.exists())
            self.assertTrue(packet.route_matrix_path.exists())
            self.assertTrue(packet.fingerprint_scores_path.exists())
            self.assertTrue(packet.card_manifest_path.exists())
            self.assertTrue((packet.packet_dir / "figures" / "02a_static_bundle_ingestion_spine.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "02a_static_exemplar_suite_routing_matrix.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "02b_static_audit_decision_card.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "02m_static_exemplar_fingerprint_strip.png").exists())
            self.assertEqual(
                validate_static_admissibility_packet(packet.packet_dir, repo_root=repo_root()),
                [],
            )


if __name__ == "__main__":
    unittest.main()
