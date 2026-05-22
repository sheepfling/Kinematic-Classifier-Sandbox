from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MethodEntry:
    name: str
    family: str
    style: str
    strengths: tuple[str, ...]
    limits: tuple[str, ...]
    typical_inputs: tuple[str, ...]
    typical_use_cases: tuple[str, ...]


METHOD_CATALOG: tuple[MethodEntry, ...] = (
    MethodEntry(
        name="Feature-engineered window classifier",
        family="traditional",
        style="windowed_features",
        strengths=(
            "strong baseline on modest datasets",
            "high interpretability",
            "easy error analysis",
        ),
        limits=(
            "depends on feature design",
            "weaker long-range temporal modeling",
        ),
        typical_inputs=("position", "velocity", "acceleration", "IMU"),
        typical_use_cases=("activity recognition", "maneuver detection", "trajectory archetypes"),
    ),
    MethodEntry(
        name="Hidden Markov model",
        family="traditional",
        style="probabilistic_sequence",
        strengths=(
            "explicit temporal state model",
            "works with small labeled datasets",
        ),
        limits=(
            "simplifying state assumptions",
            "limited representational capacity",
        ),
        typical_inputs=("windowed kinematic features", "discrete motion states"),
        typical_use_cases=("gesture phases", "maneuver stage decoding"),
    ),
    MethodEntry(
        name="Kalman residual maneuver classifier",
        family="model_based",
        style="state_space",
        strengths=(
            "close to physics and tracking practice",
            "fast online inference",
        ),
        limits=(
            "performance depends on motion-model fit",
            "narrow class vocabulary",
        ),
        typical_inputs=("track states", "residuals", "turn rate"),
        typical_use_cases=("target maneuver classification", "vehicle dynamics mode tagging"),
    ),
    MethodEntry(
        name="IMM or MMAE multiple-model classifier",
        family="model_based",
        style="multiple_model",
        strengths=(
            "joint state estimation and mode inference",
            "good classical benchmark for maneuvering targets",
        ),
        limits=(
            "requires careful model bank design",
            "less suitable for broad semantic labels",
        ),
        typical_inputs=("track states", "motion-model likelihoods"),
        typical_use_cases=("CV/CA/CTRV/CTRA mode selection", "manoeuvrability assessment"),
    ),
    MethodEntry(
        name="Bayesian joint tracking and classification filter bank",
        family="model_based",
        style="class_matched_multiple_model",
        strengths=(
            "jointly reasons over state, class, and mode uncertainty",
            "supports soft physical constraints and unknown-class handling",
            "matches online tracking workflows better than a detached feature classifier",
        ),
        limits=(
            "requires careful class and mode bank design",
            "likelihood calibration and observability can be regime dependent",
        ),
        typical_inputs=(
            "PVA track states with covariance",
            "class-conditioned motion models",
            "constraint likelihoods",
            "optional beta and L/D proxies",
        ),
        typical_use_cases=(
            "air target classification from kinematic tracks",
            "ballistic versus glide versus powered discrimination",
            "online open-set track classification",
        ),
    ),
    MethodEntry(
        name="1D CNN or TCN classifier",
        family="deep_learning",
        style="raw_window_encoder",
        strengths=(
            "strong raw-signal baseline",
            "efficient on fixed windows",
        ),
        limits=(
            "fixed receptive field unless carefully designed",
            "less transparent than feature baselines",
        ),
        typical_inputs=("multivariate time series", "normalized kinematic windows"),
        typical_use_cases=("IMU activity recognition", "short-horizon trajectory classification"),
    ),
    MethodEntry(
        name="CNN-LSTM or GRU hybrid",
        family="deep_learning",
        style="hybrid_sequence",
        strengths=(
            "captures local patterns and sequence order",
            "useful on moderate datasets",
        ),
        limits=(
            "heavier training than simpler baselines",
            "less parallel than pure convolution",
        ),
        typical_inputs=("multivariate time series", "segmented tracks"),
        typical_use_cases=("activity recognition", "behavior phase recognition"),
    ),
    MethodEntry(
        name="Transformer time-series classifier",
        family="advanced",
        style="attention_sequence",
        strengths=(
            "handles longer context",
            "flexible for multimodal fusion",
        ),
        limits=(
            "data hungry",
            "heavier than classical and CNN baselines",
        ),
        typical_inputs=("multichannel time series", "sensor fusion streams"),
        typical_use_cases=("long-context HAR", "trajectory context classification"),
    ),
    MethodEntry(
        name="Self-supervised motion representation learner",
        family="advanced",
        style="pretrain_then_finetune",
        strengths=(
            "useful when labels are scarce",
            "supports multiple downstream tasks",
        ),
        limits=(
            "more pipeline complexity",
            "harder evaluation discipline",
        ),
        typical_inputs=("unlabeled motion windows", "large sensor corpora"),
        typical_use_cases=("foundation embeddings", "transfer to classification"),
    ),
)


def method_families() -> tuple[str, ...]:
    return tuple(sorted({entry.family for entry in METHOD_CATALOG}))
