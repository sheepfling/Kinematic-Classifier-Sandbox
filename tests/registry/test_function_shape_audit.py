from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.function_shape_audit import (
    analyze_function_shape_audit,
    write_function_shape_audit_artifacts,
)


class FunctionShapeAuditTests(unittest.TestCase):
    def test_audit_classifies_known_generic_and_scenario_specific_callables(self) -> None:
        result = analyze_function_shape_audit()
        rows = {(row.module_path, row.qualified_name): row for row in result.function_rows}

        generic_row = rows[("src/kinematic_classifier_sandbox/utils/math.py", "_logsumexp")]
        self.assertEqual(generic_row.specificity, "generic")
        self.assertIn(generic_row.role, {"utility", "helper"})

        scenario_row = rows[
            (
                "src/kinematic_classifier_sandbox/witnesses/benchmarks/kalman_filter_bank.py",
                "run_kalman_filter_bank",
            )
        ]
        self.assertEqual(scenario_row.specificity, "scenario_specific")
        self.assertEqual(scenario_row.role, "runner")

        method_row = rows[
            (
                "src/kinematic_classifier_sandbox/corpus/gym.py",
                "CorpusGymEnvironment.score",
            )
        ]
        self.assertEqual(method_row.symbol_kind, "method")
        self.assertEqual(method_row.heuristic_specificity, "scenario_specific")
        self.assertEqual(method_row.specificity, "study_specific")
        self.assertEqual(
            method_row.override_source,
            "function:src/kinematic_classifier_sandbox/corpus/gym.py::CorpusGymEnvironment.score",
        )

        overridden_module_row = rows[
            (
                "src/kinematic_classifier_sandbox/markdown_builder.py",
                "build_mermaid_flow",
            )
        ]
        self.assertEqual(overridden_module_row.heuristic_specificity, "study_specific")
        self.assertEqual(overridden_module_row.specificity, "generic")
        self.assertEqual(
            overridden_module_row.override_source,
            "module:src/kinematic_classifier_sandbox/markdown_builder.py",
        )

    def test_writer_emits_audit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_function_shape_audit_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "function_shape_audit_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.function_rows_path.exists())
            self.assertTrue(artifacts.file_rows_path.exists())

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertGreater(summary["file_count"], 10)
            self.assertGreater(summary["callable_count"], 100)
            self.assertIn("inference", summary["family_counts"])
            self.assertGreaterEqual(summary["override_count"], 1)


if __name__ == "__main__":
    unittest.main()
