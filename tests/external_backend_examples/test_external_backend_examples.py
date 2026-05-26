from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.exploration.external_backend_examples import (
    analyze_external_backend_examples,
)
from kinematic_classifier_sandbox.corpus.exploration.external_backend_examples_rendering import (
    write_external_backend_examples_artifacts,
)


class ExternalBackendExampleTests(unittest.TestCase):
    def test_analysis_exposes_taos_and_tgx_like_examples(self) -> None:
        result = analyze_external_backend_examples()
        example_ids = {row.example_id for row in result.example_rows}
        self.assertIn("taos_like_1d_environment_adapter", example_ids)
        self.assertIn("tgx_like_1d_file_adapter", example_ids)
        self.assertIn("external_1d_boundary_reference", example_ids)
        self.assertIn("prepare(candidate) -> input_bundle", result.report_markdown)
        self.assertIn("TAOS-like example", result.report_markdown)
        self.assertIn("TGx-like example", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_external_backend_examples_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "external_backend_examples")
            self.assertTrue(artifacts.example_index_path.exists())
            self.assertTrue(artifacts.report_path.exists())

            index_text = artifacts.example_index_path.read_text(encoding="utf-8")
            self.assertIn("taos_like_1d_environment_adapter", index_text)
            self.assertIn("tgx_like_1d_file_adapter", index_text)


if __name__ == "__main__":
    unittest.main()
