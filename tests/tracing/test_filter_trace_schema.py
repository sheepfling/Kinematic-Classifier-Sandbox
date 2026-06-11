from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.render.intermediate_plots import (
    render_likelihood_strip,
    render_posterior_timeline,
    render_prior_likelihood_posterior_waterfall,
)
from kinematic_classifier_sandbox.render.step_cards import write_step_card
from kinematic_classifier_sandbox.tracing.filter_trace import FilterStepTrace, write_filter_step_trace_csv
from kinematic_classifier_sandbox.tracing.trace_schema import filter_step_trace_schema
from kinematic_classifier_sandbox.tracing.trace_validation import validate_filter_step_trace_set


def _example_traces() -> tuple[FilterStepTrace, ...]:
    rows: list[FilterStepTrace] = []
    for time_index, time in enumerate((0.0, 1.0)):
        for label, posterior in (("slow", 0.25 + 0.25 * time_index), ("fast", 0.75 - 0.25 * time_index)):
            rows.append(
                FilterStepTrace(
                    run_id="example_run",
                    study_id="example_study",
                    trajectory_id="traj_001",
                    method_id="example_filter",
                    rung="test",
                    time_index=time_index,
                    time=time,
                    dt=1.0,
                    class_or_model=label,
                    true_class="fast",
                    true_mode="fast",
                    prior_probability=0.5,
                    predicted_probability=0.5,
                    log_transition_probability=None,
                    measurement=(time,),
                    predicted_measurement=(time + 0.1,),
                    innovation=(-0.1,),
                    innovation_covariance_diag=(0.04,),
                    normalized_innovation_squared=0.25,
                    log_likelihood=-0.7 if label == "fast" else -1.2,
                    incremental_log_evidence=-0.7 if label == "fast" else -1.2,
                    posterior_probability=posterior,
                    posterior_entropy=0.69,
                    predicted_state_mean=(time, 1.0),
                    predicted_state_covariance_diag=(0.2, 0.3),
                    updated_state_mean=(time + 0.05, 1.0),
                    updated_state_covariance_diag=(0.1, 0.2),
                    effective_sample_size=None,
                    is_resampled=None,
                )
            )
    return tuple(rows)


class FilterTraceSchemaTests(unittest.TestCase):
    def test_schema_names_required_fields(self) -> None:
        schema = filter_step_trace_schema()
        self.assertEqual(schema["artifact"], "filter_step_trace")
        self.assertIn("posterior_probability", schema["fieldnames"])
        self.assertIn("updated_state_mean", schema["fields"])

    def test_trace_validation_and_csv_roundtrip_surface(self) -> None:
        traces = _example_traces()
        validate_filter_step_trace_set(traces)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_filter_step_trace_csv(Path(temp_dir) / "filter_step_trace.csv", traces)
            text = path.read_text(encoding="utf-8")
            self.assertIn("posterior_probability", text)
            self.assertIn("updated_state_mean", text)

    def test_intermediate_renderers_write_artifacts(self) -> None:
        traces = _example_traces()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            posterior = render_posterior_timeline(root / "posterior.png", traces)
            strip = render_likelihood_strip(root / "strip.png", traces)
            waterfall = render_prior_likelihood_posterior_waterfall(root / "waterfall.png", traces, time_index=1)
            card = write_step_card(root / "step_card.md", tuple(trace for trace in traces if trace.time_index == 1))
            self.assertTrue(posterior.exists())
            self.assertTrue(strip.exists())
            self.assertTrue(waterfall.exists())
            self.assertTrue(card.exists())
            self.assertIn("Prior Before Measurement", card.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
