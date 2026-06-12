from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.static_admissibility.multi_domain_3d import (
    write_multidomain_3d_static_admissibility_packet,
)
from kinematic_classifier_sandbox.static_admissibility.validation import (
    validate_static_admissibility_packet,
)
from kinematic_classifier_sandbox.utils.runtime import repo_root


class MultiDomain3dStaticAdmissibilityPacketTests(unittest.TestCase):
    def test_multidomain_packet_builds_with_expected_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet = write_multidomain_3d_static_admissibility_packet(
                Path(temp_dir) / "01_static_admissibility_multi_domain_3d",
            )

            self.assertTrue(packet.readme_path.exists())
            self.assertTrue(packet.quickstart_path.exists())
            self.assertTrue(packet.decision_card_path.exists())
            self.assertTrue(packet.validation_report_path.exists())
            self.assertTrue(packet.claim_boundary_path.exists())
            self.assertTrue(packet.packet_manifest_path.exists())
            self.assertTrue(packet.hero_chart_manifest_path.exists())
            self.assertTrue(packet.lane_proof_matrix_path.exists())
            self.assertTrue(packet.automated_brief_path.exists())
            self.assertTrue(packet.latex_path.exists())
            self.assertTrue(packet.estimator_reliability_report_path.exists())
            self.assertTrue((packet.packet_dir / "feature_alias_and_redundancy_report.md").exists())
            self.assertTrue((packet.packet_dir / "figures" / "MD3D_05_class_feature_excitation_matrix.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "MD3D_07_prior_pathology_surface.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "MD3D_10_unobservable_and_leakage_audit.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "MD3D_13_estimator_reliability_dashboard.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "MD3D_15_prior_evidence_budget.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "MD3D_19_threshold_subsumption_map.png").exists())
            self.assertTrue((packet.packet_dir / "figures" / "MD3D_21_decision_redundancy_matrix.png").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "static_metric_uncertainty.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "pairwise_error_bound_proxy.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "prior_evidence_budget.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "sample_size_adequacy_report.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "metric_assumption_registry.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "feature_alias_candidates.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "feature_threshold_subsumption.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "feature_functional_equivalence.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "feature_decision_redundancy.csv").exists())
            self.assertTrue((packet.packet_dir / "source_artifacts" / "feature_redundancy_clusters.csv").exists())

            route_rows = list(
                csv.DictReader(
                    (packet.packet_dir / "source_artifacts" / "multidomain_bundle_route_matrix.csv").open(
                        newline="",
                        encoding="utf-8",
                    )
                )
            )
            route_map = {row["bundle_id"]: row["actual_route"] for row in route_rows}
            self.assertEqual(route_map["clean_multidomain_3d_bundle"], "promote_to_corpus_explorer")
            self.assertEqual(route_map["prior_pathology_multidomain_3d_bundle"], "revise_prior")
            self.assertEqual(route_map["redundancy_synergy_multidomain_3d_bundle"], "promote_to_corpus_explorer")
            self.assertEqual(route_map["unobservable_navy_space_bundle"], "revise_class_set")
            self.assertEqual(route_map["leakage_blocker_multidomain_3d_bundle"], "reject")
            self.assertIn("decision_confidence", packet.decision_card_path.read_text(encoding="utf-8"))

            self.assertEqual(
                validate_static_admissibility_packet(packet.packet_dir, repo_root=repo_root()),
                [],
            )


if __name__ == "__main__":
    unittest.main()
