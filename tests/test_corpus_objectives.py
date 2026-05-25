from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_corpus_objectives,
    default_corpus_objectives,
    load_corpus_objectives_from_yaml,
    validate_corpus_objective,
    write_corpus_objective_artifacts,
)


class CorpusObjectivesTests(unittest.TestCase):
    def test_default_objectives_are_valid(self) -> None:
        objectives = default_corpus_objectives()
        self.assertGreaterEqual(len(objectives), 3)
        for objective in objectives:
            self.assertEqual(validate_corpus_objective(objective), [])

    def test_yaml_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_objective_artifacts(temp_dir)
            loaded = load_corpus_objectives_from_yaml(artifacts.example_objectives_path)
            self.assertEqual(len(loaded), len(default_corpus_objectives()))

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_objective_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_objectives")
            self.assertTrue(artifacts.schema_path.exists())
            self.assertTrue(artifacts.example_objectives_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.relationship_png_path.exists())
            self.assertTrue(artifacts.coverage_png_path.exists())

            payload = json.loads(artifacts.schema_path.read_text(encoding="utf-8"))
            self.assertIn("properties", payload)


if __name__ == "__main__":
    unittest.main()
