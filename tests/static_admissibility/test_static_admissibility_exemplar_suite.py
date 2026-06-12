from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from kinematic_classifier_sandbox.static_admissibility.audit import run_static_admissibility_audit
from kinematic_classifier_sandbox.static_admissibility.validation import (
    validate_static_admissibility_packet,
)
from kinematic_classifier_sandbox.utils.runtime import repo_root


class StaticAdmissibilityExemplarSuiteTests(unittest.TestCase):
    def test_epic1_exemplar_suite_routes_each_bundle_as_expected(self) -> None:
        root = repo_root()
        manifest_path = root / "experiments" / "static_admissibility" / "epic1_exemplar_suite.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        exemplars = manifest.get("epic1_static_admissibility_exemplars", ())

        with tempfile.TemporaryDirectory() as temp_dir:
            packet_root = Path(temp_dir)
            for exemplar in exemplars:
                config_path = root / "experiments" / "static_admissibility" / str(exemplar["config"])
                packet = run_static_admissibility_audit(
                    config_path,
                    packet_root / str(exemplar["exemplar_id"]),
                )
                self.assertEqual(validate_static_admissibility_packet(packet.packet_dir), [])
                decision_text = packet.decision_card_path.read_text(encoding="utf-8")
                self.assertIn(str(exemplar["expected_status"]), decision_text)
                self.assertIn(str(exemplar["expected_adequacy_label"]), decision_text)


if __name__ == "__main__":
    unittest.main()
