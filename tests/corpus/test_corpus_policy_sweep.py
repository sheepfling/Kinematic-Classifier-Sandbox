from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from kinematic_classifier_sandbox.corpus.policy_sweep import write_corpus_policy_tuning_artifacts
from kinematic_classifier_sandbox.utils.io import _read_csv


class CorpusPolicySweepTests(unittest.TestCase):
    def test_tuning_artifacts_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_policy_tuning_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_hyperparameter_tuning_v1")
            required = (
                "weight_spec_schema.json",
                "default_weight_spec.yaml",
                "sweep_design.csv",
                "sweep_results.csv",
                "ablation_results.csv",
                "local_perturbation_results.csv",
                "selected_set_jaccard.csv",
                "rank_stability.csv",
                "sampler_budget_sweep.csv",
                "gate_threshold_sweep.csv",
                "dev_holdout_results.csv",
                "pareto_front.csv",
                "recommended_policy.yaml",
                "corpus_hyperparameter_tuning_report.md",
                "corpus_policy_numeric_walkthrough.md",
                "weight_sensitivity_tornado.png",
                "selected_set_jaccard_heatmap.png",
                "rank_correlation_heatmap.png",
                "ablation_tradeoff_bars.png",
                "pareto_tradeoff_scatter.png",
                "sampler_budget_efficiency.png",
                "gate_sensitivity_curves.png",
                "dev_vs_holdout_policy_scores.png",
            )
            for name in required:
                path = artifacts.run_dir / name
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_sweep_outputs_nonempty_policy_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_policy_tuning_artifacts(temp_dir)
            rows = _read_csv(artifacts.sweep_results_path)
            self.assertGreaterEqual(len(rows), 2)
            self.assertTrue(all(row["selected_set"] for row in rows))
            self.assertIn("adequacy_score", rows[0])

    def test_ablation_and_stability_outputs_have_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_policy_tuning_artifacts(temp_dir)
            ablations = _read_csv(artifacts.ablation_results_path)
            self.assertEqual(len(ablations), 8)
            self.assertTrue(any(row["removed_term"] == "classifier_stress" for row in ablations))
            jaccard = _read_csv(artifacts.stability_path)
            self.assertTrue(jaccard)
            self.assertIn("selected_set_jaccard", jaccard[0])

    def test_sampler_gate_and_holdout_results_are_measured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_policy_tuning_artifacts(temp_dir)
            sampler_rows = _read_csv(artifacts.run_dir / "sampler_budget_sweep.csv")
            gate_rows = _read_csv(artifacts.run_dir / "gate_threshold_sweep.csv")
            holdout_rows = _read_csv(artifacts.run_dir / "dev_holdout_results.csv")
            self.assertTrue(any(row["sampler_family"] == "stress_mutation" for row in sampler_rows))
            self.assertGreater(len({row["accepted_count"] for row in gate_rows}), 1)
            self.assertTrue(all(row["dev_score"] and row["holdout_score"] for row in holdout_rows))

    def test_recommended_policy_references_evaluated_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_policy_tuning_artifacts(temp_dir)
            sweep_rows = _read_csv(artifacts.sweep_results_path)
            evaluated = {row["policy_id"] for row in sweep_rows}
            recommended = yaml.safe_load(artifacts.recommended_policy_path.read_text(encoding="utf-8"))
            self.assertIn(recommended["recommendation"]["recommended_policy_id"], evaluated)

    def test_numeric_walkthrough_contains_real_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_policy_tuning_artifacts(temp_dir)
            text = artifacts.numeric_walkthrough_path.read_text(encoding="utf-8")
            self.assertIn("Adequacy Proxy Substitution", text)
            self.assertIn("J_{\\text{policy}}", text)
            self.assertIn("Selected-set Jaccard vs default", text)


if __name__ == "__main__":
    unittest.main()
