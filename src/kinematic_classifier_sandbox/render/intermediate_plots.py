from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..tracing.filter_trace import FilterStepTrace
from ..utils.plotting import plt


def render_posterior_timeline(path: str | Path, traces: tuple[FilterStepTrace, ...] | list[FilterStepTrace]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    true_modes: dict[float, str] = {}
    for trace in traces:
        if trace.posterior_probability is None:
            continue
        by_model[trace.class_or_model].append((trace.time, trace.posterior_probability))
        if trace.true_mode:
            true_modes[trace.time] = trace.true_mode
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for model, values in sorted(by_model.items()):
        ordered = sorted(values)
        ax.plot([value[0] for value in ordered], [value[1] for value in ordered], label=model)
    if true_modes:
        last_mode = None
        for time_value in sorted(true_modes):
            mode = true_modes[time_value]
            if mode != last_mode:
                ax.axvline(time_value, color="#6b7280", alpha=0.25, linewidth=1.0)
            last_mode = mode
    ax.set_title("Posterior Timeline With Regime Markers", loc="left")
    ax.set_xlabel("time")
    ax.set_ylabel("posterior")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_likelihood_strip(path: str | Path, traces: tuple[FilterStepTrace, ...] | list[FilterStepTrace]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({trace.class_or_model for trace in traces})
    times = sorted({trace.time for trace in traces})
    model_index = {model: index for index, model in enumerate(models)}
    time_index = {time_value: index for index, time_value in enumerate(times)}
    values = [[0.0 for _ in times] for _ in models]
    for trace in traces:
        if trace.log_likelihood is None:
            continue
        values[model_index[trace.class_or_model]][time_index[trace.time]] = trace.log_likelihood
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(models))), dpi=150)
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title("Innovation / Evidence Strip", loc="left")
    ax.set_xlabel("time index")
    ax.set_ylabel("class/model")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    if times:
        tick_step = max(1, len(times) // 6)
        tick_positions = list(range(0, len(times), tick_step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{times[index]:.2g}" for index in tick_positions], fontsize=7)
    fig.colorbar(image, ax=ax, label="log likelihood")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_prior_likelihood_posterior_waterfall(path: str | Path, traces: tuple[FilterStepTrace, ...] | list[FilterStepTrace], *, time_index: int | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_time_index = max(trace.time_index for trace in traces) if time_index is None else time_index
    rows = [trace for trace in traces if trace.time_index == selected_time_index]
    labels = [trace.class_or_model for trace in rows]
    priors = [0.0 if trace.predicted_probability is None else trace.predicted_probability for trace in rows]
    likelihoods = [0.0 if trace.log_likelihood is None else trace.log_likelihood for trace in rows]
    posteriors = [0.0 if trace.posterior_probability is None else trace.posterior_probability for trace in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10, max(3, 0.35 * len(labels))), dpi=150, sharey=True)
    axes[0].barh(labels, priors, color="#2563eb")
    axes[0].set_title("predicted prior", loc="left")
    axes[1].barh(labels, likelihoods, color="#0f766e")
    axes[1].set_title("log likelihood", loc="left")
    axes[2].barh(labels, posteriors, color="#7c3aed")
    axes[2].set_title("posterior", loc="left")
    axes[0].set_xlim(0.0, 1.0)
    axes[2].set_xlim(0.0, 1.0)
    fig.suptitle(f"Prior -> Likelihood -> Posterior, t={selected_time_index}", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
