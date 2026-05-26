from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "SwitchingModeSpec": (".contracts", "SwitchingModeSpec"),
    "SwitchingScenario": (".contracts", "SwitchingScenario"),
    "TransitionPosteriorStep": (".contracts", "TransitionPosteriorStep"),
    "TransitionRun": (".contracts", "TransitionRun"),
    "TransitionBenchmarkSummary": (".contracts", "TransitionBenchmarkSummary"),
    "TransitionBenchmarkResult": (".contracts", "TransitionBenchmarkResult"),
    "TransitionBenchmarkArtifacts": (".contracts", "TransitionBenchmarkArtifacts"),
    "default_switching_mode_specs": (".runner", "default_switching_mode_specs"),
    "default_transition_matrix": (".runner", "default_transition_matrix"),
    "generate_transition_switching_scenarios": (".runner", "generate_transition_switching_scenarios"),
    "run_transition_benchmark": (".runner", "run_transition_benchmark"),
    "_run_mode_accumulator": (".runner", "_run_mode_accumulator"),
    "render_transition_benchmark_report": (".reporting", "render_transition_benchmark_report"),
    "render_transition_numeric_walkthrough_markdown": (".reporting", "render_transition_numeric_walkthrough_markdown"),
    "write_transition_benchmark_artifacts": (".artifact_io", "write_transition_benchmark_artifacts"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
