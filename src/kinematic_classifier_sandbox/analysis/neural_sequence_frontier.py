from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _kalman_predict,
    _rocket_proxy_predict,
    _windowed_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import (
    SharedDynamicsTrajectory,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - current test env includes torch
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


CLASS_NAMES = ("constant_velocity", "constant_acceleration")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: label for label, index in CLASS_TO_INDEX.items()}
RESAMPLED_LENGTH = 24
TRAINING_EPOCHS = 24
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1.0e-4
CALIBRATION_STEPS = 80


@dataclass(frozen=True, slots=True)
class NeuralFrontierPredictionRow:
    trajectory_id: str
    scenario_name: str
    true_class: str
    split: str
    method_name: str
    predicted_class: str
    confidence: float
    negative_log_likelihood: float
    ece_bin_confidence: float


@dataclass(frozen=True, slots=True)
class NeuralFrontierMetricRow:
    method_name: str
    overall_accuracy: float
    test_accuracy: float
    short_noisy_accuracy: float
    endpoint_match_accuracy: float
    test_nll: float
    test_ece: float
    calibration_delta_nll: float
    applicability_status: str
    claim_level: str


@dataclass(frozen=True, slots=True)
class TrainingCurveRow:
    method_name: str
    epoch: int
    train_loss: float
    calibration_loss: float


@dataclass(frozen=True, slots=True)
class NeuralSequenceFrontierResult:
    prediction_rows: tuple[NeuralFrontierPredictionRow, ...]
    metric_rows: tuple[NeuralFrontierMetricRow, ...]
    training_curve_rows: tuple[TrainingCurveRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class NeuralSequenceFrontierArtifacts:
    run_dir: Path
    prediction_summary_path: Path
    metric_summary_path: Path
    training_curve_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _SplitBundle:
    train: tuple[SharedDynamicsTrajectory, ...]
    calibration: tuple[SharedDynamicsTrajectory, ...]
    test: tuple[SharedDynamicsTrajectory, ...]


@dataclass(frozen=True, slots=True)
class _TorchBundle:
    inputs: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True, slots=True)
class _TrainedSequenceModel:
    method_name: str
    model: nn.Module
    temperature: float
    training_curve_rows: tuple[TrainingCurveRow, ...]
    calibration_nll_before: float
    calibration_nll_after: float


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("torch is required for neural_sequence_frontier") from _TORCH_IMPORT_ERROR


def _set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def _split_dataset(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
) -> _SplitBundle:
    train_rows: list[SharedDynamicsTrajectory] = []
    calibration_rows: list[SharedDynamicsTrajectory] = []
    test_rows: list[SharedDynamicsTrajectory] = []
    for trajectory in trajectories:
        example_index = int(trajectory.trajectory_id.rsplit("_", 1)[-1])
        if example_index < 3:
            train_rows.append(trajectory)
        elif example_index == 3:
            calibration_rows.append(trajectory)
        else:
            test_rows.append(trajectory)
    return _SplitBundle(tuple(train_rows), tuple(calibration_rows), tuple(test_rows))


def _resample_measurements(trajectory: SharedDynamicsTrajectory, *, output_length: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    times = numpy.asarray(trajectory.times, dtype=float)
    measurements = numpy.asarray(trajectory.measurements, dtype=float)
    if len(measurements) == 1:
        repeated = numpy.repeat(measurements, output_length)
        return repeated, numpy.linspace(0.0, 1.0, output_length, dtype=float)
    shifted_times = times - times[0]
    duration = max(float(shifted_times[-1]), 1.0e-6)
    normalized_times = shifted_times / duration
    grid = numpy.linspace(0.0, 1.0, output_length, dtype=float)
    resampled = numpy.interp(grid, normalized_times, measurements)
    return resampled, grid


def _trajectory_to_channels(trajectory: SharedDynamicsTrajectory) -> numpy.ndarray:
    values, grid = _resample_measurements(trajectory, output_length=RESAMPLED_LENGTH)
    centered = values - float(numpy.mean(values))
    scale = float(numpy.std(centered))
    if scale < 1.0e-6:
        scale = 1.0
    normalized = centered / scale
    slope = numpy.gradient(normalized)
    return numpy.asarray([normalized, slope, grid], dtype=numpy.float32)


def _build_torch_bundle(trajectories: tuple[SharedDynamicsTrajectory, ...]) -> _TorchBundle:
    channels = numpy.stack([_trajectory_to_channels(trajectory) for trajectory in trajectories], axis=0)
    lengths = numpy.full(len(trajectories), RESAMPLED_LENGTH, dtype=numpy.int64)
    labels = numpy.asarray([CLASS_TO_INDEX[trajectory.true_class] for trajectory in trajectories], dtype=numpy.int64)
    return _TorchBundle(
        inputs=torch.tensor(channels, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        lengths=torch.tensor(lengths, dtype=torch.long),
    )


class _ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class _TinyTCN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _ResidualTCNBlock(hidden_channels, dilation=1),
            _ResidualTCNBlock(hidden_channels, dilation=2),
            _ResidualTCNBlock(hidden_channels, dilation=4),
        )
        self.head = nn.Linear(hidden_channels, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(inputs)
        hidden = self.blocks(hidden)
        pooled = hidden.mean(dim=-1)
        return self.head(pooled)


class _InceptionBranch(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.branch(inputs)


class _TinyInceptionTime(nn.Module):
    def __init__(self, input_channels: int, branch_channels: int, classes: int) -> None:
        super().__init__()
        self.branch3 = _InceptionBranch(input_channels, branch_channels, kernel_size=3)
        self.branch5 = _InceptionBranch(input_channels, branch_channels, kernel_size=5)
        self.branch7 = _InceptionBranch(input_channels, branch_channels, kernel_size=7)
        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(branch_channels * 4, branch_channels * 2, kernel_size=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(branch_channels * 2, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat(
            (
                self.branch3(inputs),
                self.branch5(inputs),
                self.branch7(inputs),
                self.pool_branch(inputs),
            ),
            dim=1,
        )
        mixed = self.mix(hidden)
        pooled = mixed.mean(dim=-1)
        return self.head(pooled)


def _make_model(method_name: str) -> nn.Module:
    if method_name == "tcn":
        return _TinyTCN(input_channels=3, hidden_channels=24, classes=len(CLASS_NAMES))
    if method_name == "inceptiontime":
        return _TinyInceptionTime(input_channels=3, branch_channels=12, classes=len(CLASS_NAMES))
    raise ValueError(f"Unsupported method_name: {method_name}")


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    before = float(_cross_entropy(logits, labels).item())
    for _ in range(CALIBRATION_STEPS):
        optimizer.zero_grad()
        temperature = torch.exp(parameter) + 1.0e-6
        loss = _cross_entropy(logits / temperature, labels)
        loss.backward()
        optimizer.step()
    temperature = float(torch.exp(parameter).item())
    after = float(_cross_entropy(logits / temperature, labels).item())
    return temperature, before, after


def _train_sequence_model(
    method_name: str,
    split: _SplitBundle,
    *,
    seed: int,
) -> _TrainedSequenceModel:
    _set_seed(seed)
    model = _make_model(method_name)
    train_bundle = _build_torch_bundle(split.train)
    calibration_bundle = _build_torch_bundle(split.calibration)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curve_rows: list[TrainingCurveRow] = []
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(train_bundle.inputs)
        train_loss = _cross_entropy(logits, train_bundle.labels)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            calibration_logits = model(calibration_bundle.inputs)
            calibration_loss = _cross_entropy(calibration_logits, calibration_bundle.labels)
        curve_rows.append(
            TrainingCurveRow(
                method_name=method_name,
                epoch=epoch + 1,
                train_loss=float(train_loss.item()),
                calibration_loss=float(calibration_loss.item()),
            )
        )
    model.eval()
    with torch.no_grad():
        calibration_logits = model(calibration_bundle.inputs)
    temperature, before, after = _fit_temperature(calibration_logits, calibration_bundle.labels)
    return _TrainedSequenceModel(
        method_name=method_name,
        model=model,
        temperature=temperature,
        training_curve_rows=tuple(curve_rows),
        calibration_nll_before=before,
        calibration_nll_after=after,
    )


def _probabilities_from_logits(logits: torch.Tensor, *, temperature: float) -> numpy.ndarray:
    probabilities = torch.softmax(logits / max(temperature, 1.0e-6), dim=1)
    return probabilities.detach().cpu().numpy()


def _evaluate_ece(probabilities: numpy.ndarray, labels: numpy.ndarray, *, bins: int = 10) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    total = max(len(labels), 1)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            index
            for index, confidence in enumerate(confidences.tolist())
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == upper)
        ]
        if not selected:
            continue
        mean_confidence = float(numpy.mean(confidences[selected]))
        mean_accuracy = float(numpy.mean((predictions[selected] == labels[selected]).astype(float)))
        ece += abs(mean_confidence - mean_accuracy) * (len(selected) / total)
    return float(ece)


def _prediction_rows_for_model(
    trained: _TrainedSequenceModel,
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    split_name: str,
) -> list[NeuralFrontierPredictionRow]:
    bundle = _build_torch_bundle(trajectories)
    with torch.no_grad():
        logits = trained.model(bundle.inputs)
    probabilities = _probabilities_from_logits(logits, temperature=trained.temperature)
    rows: list[NeuralFrontierPredictionRow] = []
    for trajectory, label_index, probability_row in zip(trajectories, bundle.labels.numpy(), probabilities, strict=True):
        predicted_index = int(numpy.argmax(probability_row))
        predicted_probability = float(probability_row[predicted_index])
        rows.append(
            NeuralFrontierPredictionRow(
                trajectory_id=trajectory.trajectory_id,
                scenario_name=trajectory.scenario_name,
                true_class=trajectory.true_class,
                split=split_name,
                method_name=trained.method_name,
                predicted_class=INDEX_TO_CLASS[predicted_index],
                confidence=predicted_probability,
                negative_log_likelihood=float(-math.log(max(float(probability_row[label_index]), 1.0e-12))),
                ece_bin_confidence=predicted_probability,
            )
        )
    return rows


def _proxy_prediction_rows(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    split_name: str,
) -> list[NeuralFrontierPredictionRow]:
    rows: list[NeuralFrontierPredictionRow] = []
    for trajectory in trajectories:
        for method_name, run in (
            ("windowed_robust", _windowed_predict(trajectory, robust=True)),
            ("rocket_proxy", _rocket_proxy_predict(trajectory)),
            ("kalman_bank", _kalman_predict(trajectory)),
        ):
            predicted_probability = float(run.final_confidence)
            truth_probability = float(run.final_weights.get(trajectory.true_class, 1.0e-12))
            rows.append(
                NeuralFrontierPredictionRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    true_class=trajectory.true_class,
                    split=split_name,
                    method_name=method_name,
                    predicted_class=run.final_predicted_class,
                    confidence=predicted_probability,
                    negative_log_likelihood=float(-math.log(max(truth_probability, 1.0e-12))),
                    ece_bin_confidence=predicted_probability,
                )
            )
    return rows


def _accuracy(rows: list[NeuralFrontierPredictionRow], *, scenario_name: str | None = None) -> float:
    selected = rows if scenario_name is None else [row for row in rows if row.scenario_name == scenario_name]
    return sum(1.0 if row.predicted_class == row.true_class else 0.0 for row in selected) / max(len(selected), 1)


def _mean_nll(rows: list[NeuralFrontierPredictionRow], *, scenario_name: str | None = None) -> float:
    selected = rows if scenario_name is None else [row for row in rows if row.scenario_name == scenario_name]
    return sum(row.negative_log_likelihood for row in selected) / max(len(selected), 1)


def analyze_neural_sequence_vs_physics_frontier(
    *,
    seed: int = 907,
    trajectories_per_case: int = 8,
) -> NeuralSequenceFrontierResult:
    _require_torch()
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    split = _split_dataset(trajectories)
    trained_tcn = _train_sequence_model("tcn", split, seed=seed + 11)
    trained_inception = _train_sequence_model("inceptiontime", split, seed=seed + 29)

    prediction_rows: list[NeuralFrontierPredictionRow] = []
    training_curve_rows = list(trained_tcn.training_curve_rows) + list(trained_inception.training_curve_rows)
    for split_name, group in (("train", split.train), ("calibration", split.calibration), ("test", split.test)):
        prediction_rows.extend(_proxy_prediction_rows(group, split_name))
        prediction_rows.extend(_prediction_rows_for_model(trained_tcn, group, split_name))
        prediction_rows.extend(_prediction_rows_for_model(trained_inception, group, split_name))

    metric_rows: list[NeuralFrontierMetricRow] = []
    calibration_delta_lookup = {
        "tcn": trained_tcn.calibration_nll_before - trained_tcn.calibration_nll_after,
        "inceptiontime": trained_inception.calibration_nll_before - trained_inception.calibration_nll_after,
        "windowed_robust": 0.0,
        "rocket_proxy": 0.0,
        "kalman_bank": 0.0,
    }
    claim_level_lookup = {
        "windowed_robust": "baseline",
        "rocket_proxy": "implemented_proxy",
        "kalman_bank": "baseline",
        "tcn": "trained_local",
        "inceptiontime": "trained_local",
    }
    for method_name in ("windowed_robust", "rocket_proxy", "kalman_bank", "tcn", "inceptiontime"):
        method_rows = [row for row in prediction_rows if row.method_name == method_name]
        test_rows = [row for row in method_rows if row.split == "test"]
        test_probabilities: list[list[float]] = []
        test_labels: list[int] = []
        for row in test_rows:
            if row.predicted_class == CLASS_NAMES[0]:
                probability_row = [row.confidence, 1.0 - row.confidence]
            else:
                probability_row = [1.0 - row.confidence, row.confidence]
            test_probabilities.append(probability_row)
            test_labels.append(CLASS_TO_INDEX[row.true_class])
        metric_rows.append(
            NeuralFrontierMetricRow(
                method_name=method_name,
                overall_accuracy=_accuracy(method_rows),
                test_accuracy=_accuracy(test_rows),
                short_noisy_accuracy=_accuracy(test_rows, scenario_name="short_noisy"),
                endpoint_match_accuracy=_accuracy(test_rows, scenario_name="endpoint_match"),
                test_nll=_mean_nll(test_rows),
                test_ece=_evaluate_ece(numpy.asarray(test_probabilities, dtype=float), numpy.asarray(test_labels, dtype=int)),
                calibration_delta_nll=float(calibration_delta_lookup[method_name]),
                applicability_status="supported",
                claim_level=claim_level_lookup[method_name],
            )
        )

    row_map = {row.method_name: row for row in metric_rows}
    best_neural_accuracy = max(row_map["tcn"].test_accuracy, row_map["inceptiontime"].test_accuracy)
    best_proxy_accuracy = max(row_map["rocket_proxy"].test_accuracy, row_map["windowed_robust"].test_accuracy)
    best_neural_nll = min(row_map["tcn"].test_nll, row_map["inceptiontime"].test_nll)
    metrics: dict[str, float | int | str] = {
        "study_id": "neural_sequence_vs_physics_frontier_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "train_count": len(split.train),
        "calibration_count": len(split.calibration),
        "test_count": len(split.test),
        "tcn_test_accuracy": row_map["tcn"].test_accuracy,
        "inception_test_accuracy": row_map["inceptiontime"].test_accuracy,
        "rocket_test_accuracy": row_map["rocket_proxy"].test_accuracy,
        "kalman_test_accuracy": row_map["kalman_bank"].test_accuracy,
        "tcn_test_ece": row_map["tcn"].test_ece,
        "inception_test_ece": row_map["inceptiontime"].test_ece,
        "tcn_calibration_delta_nll": row_map["tcn"].calibration_delta_nll,
        "inception_calibration_delta_nll": row_map["inceptiontime"].calibration_delta_nll,
        "promotion_decision": (
            "promote_trained_neural_sequence_frontier"
            if best_neural_accuracy >= best_proxy_accuracy and best_neural_nll <= row_map["rocket_proxy"].test_nll
            else "hold_trained_neural_sequence_frontier"
        ),
    }
    return NeuralSequenceFrontierResult(
        prediction_rows=tuple(prediction_rows),
        metric_rows=tuple(metric_rows),
        training_curve_rows=tuple(training_curve_rows),
        metrics=metrics,
    )


def _render_frontier_bars(result: NeuralSequenceFrontierResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    labels = [row.method_name for row in result.metric_rows]
    values = [row.test_accuracy for row in result.metric_rows]
    colors = ["#9ca3af", "#16a34a", "#2563eb", "#7c3aed", "#dc2626"]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Neural Sequence Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_scenario_panel(result: NeuralSequenceFrontierResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    method_names = [row.method_name for row in result.metric_rows]
    x = list(range(len(method_names)))
    width = 0.35
    endpoint = [row.endpoint_match_accuracy for row in result.metric_rows]
    noisy = [row.short_noisy_accuracy for row in result.metric_rows]
    ax.bar([value - width / 2 for value in x], endpoint, width=width, label="endpoint_match", color="#2563eb")
    ax.bar([value + width / 2 for value in x], noisy, width=width, label="short_noisy", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Scenario Slice: Neural Sequence vs Physics", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_training_curves(result: NeuralSequenceFrontierResult):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for method_name, color in (("tcn", "#7c3aed"), ("inceptiontime", "#dc2626")):
        rows = [row for row in result.training_curve_rows if row.method_name == method_name]
        ax.plot([row.epoch for row in rows], [row.train_loss for row in rows], color=color, linewidth=2.0, label=f"{method_name}_train")
        ax.plot([row.epoch for row in rows], [row.calibration_loss for row in rows], color=color, linestyle="--", linewidth=1.6, label=f"{method_name}_cal")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross entropy")
    ax.set_title("Neural Sequence Training Curves", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def write_neural_sequence_vs_physics_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: NeuralSequenceFrontierResult | None = None,
    seed: int = 907,
    trajectories_per_case: int = 8,
) -> NeuralSequenceFrontierArtifacts:
    payload = result or analyze_neural_sequence_vs_physics_frontier(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    run_dir = Path(output_dir) / "neural_sequence_vs_physics_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary_path = run_dir / "prediction_summary.csv"
    metric_summary_path = run_dir / "metric_summary.csv"
    training_curve_path = run_dir / "training_curve.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "neural_sequence_vs_physics_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frontier_plot_path = plots_dir / "frontier_test_accuracy.png"
    scenario_plot_path = plots_dir / "scenario_slice_accuracy.png"
    training_plot_path = plots_dir / "training_curves.png"

    write_csv(
        prediction_summary_path,
        [asdict(row) for row in payload.prediction_rows],
        list(NeuralFrontierPredictionRow.__dataclass_fields__.keys()),
    )
    write_csv(
        metric_summary_path,
        [asdict(row) for row in payload.metric_rows],
        list(NeuralFrontierMetricRow.__dataclass_fields__.keys()),
    )
    write_csv(
        training_curve_path,
        [asdict(row) for row in payload.training_curve_rows],
        list(TrainingCurveRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.metric_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Neural Sequence vs Physics Frontier",
        "",
        "- Study: `neural_sequence_vs_physics_frontier_v1`",
        "- Sequence methods: `tcn`, `inceptiontime`",
        "- Baselines: `windowed_robust`, `rocket_proxy`, `kalman_bank`",
        "- Training stack: local `torch` models with held-out temperature scaling",
        "",
        "## Claim Boundary",
        "",
        "This packet now reflects real local neural training rather than handcrafted sequence proxies.",
        "It is still not an external-library benchmark claim, and robustness sweeps beyond the first witness remain open.",
        "",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous methods: `gradient_boosted_features`, `minirocket_family`, `kalman_bank`",
        "- Candidate methods: `tcn`, `inceptiontime`",
        "- Implementation state: real local training with held-out calibration",
        "- Decision: `promote to witness-backed neural frontier only if trained models beat proxy baselines on test accuracy and NLL`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    frontier_plot_path.write_bytes(_figure_to_png(_render_frontier_bars(payload)))
    scenario_plot_path.write_bytes(_figure_to_png(_render_scenario_panel(payload)))
    training_plot_path.write_bytes(_figure_to_png(_render_training_curves(payload)))

    return NeuralSequenceFrontierArtifacts(
        run_dir=run_dir,
        prediction_summary_path=prediction_summary_path,
        metric_summary_path=metric_summary_path,
        training_curve_path=training_curve_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(frontier_plot_path, scenario_plot_path, training_plot_path),
    )


__all__ = [
    "NeuralFrontierMetricRow",
    "NeuralFrontierPredictionRow",
    "NeuralSequenceFrontierArtifacts",
    "NeuralSequenceFrontierResult",
    "TrainingCurveRow",
    "analyze_neural_sequence_vs_physics_frontier",
    "write_neural_sequence_vs_physics_frontier_artifacts",
]
