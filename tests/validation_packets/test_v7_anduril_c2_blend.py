from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.validation_packets import (
    validate_v7_anduril_c2_blend_packet,
    write_v7_anduril_c2_blend_packet,
)


class V7AndurilC2BlendPacketTests(unittest.TestCase):
    def test_v7_packet_builds_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet = write_v7_anduril_c2_blend_packet(Path(temp_dir) / "v7_anduril_c2_blend")
            self.assertTrue(packet.manifest_path.exists())
            self.assertTrue(packet.decision_card_path.exists())
            self.assertTrue(packet.claim_boundary_path.exists())
            self.assertTrue(packet.epic_summary_path.exists())
            self.assertTrue(packet.hero_chart_manifest_path.exists())
            self.assertTrue(packet.lane_proof_matrix_path.exists())
            self.assertTrue(packet.validation_report_path.exists())
            self.assertTrue(packet.main_deck_path.exists())
            self.assertTrue(packet.appendix_deck_path.exists())
            self.assertTrue(packet.whitepaper_main_path.exists())
            self.assertTrue((packet.packet_dir / "epic_packets" / "01_static_admissibility_gate" / "static_bundle_schema.json").exists())
            self.assertTrue((packet.packet_dir / "epic_packets" / "02_evidence_construction_ladder" / "static_ceiling_capture.csv").exists())
            self.assertTrue((packet.packet_dir / "epic_packets" / "03_corpus_explorer_design_engine" / "hard_case_route_ledger.csv").exists())
            self.assertEqual(validate_v7_anduril_c2_blend_packet(packet.packet_dir), [])

    def test_integrated_decision_card_preserves_claim_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet = write_v7_anduril_c2_blend_packet(Path(temp_dir) / "v7_anduril_c2_blend")
            text = packet.decision_card_path.read_text(encoding="utf-8")
            self.assertIn("presentable_methodology_workbench", text)
            self.assertIn("general PF/RBPF promotion without run-backed shine witnesses", text)
            self.assertIn(
                "general CEM/PPO superiority without baseline, ablation, seed stability, and downstream-yield evidence",
                text,
            )


if __name__ == "__main__":
    unittest.main()
