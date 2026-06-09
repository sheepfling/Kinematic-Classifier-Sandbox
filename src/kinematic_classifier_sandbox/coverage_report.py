from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

from .corpus_adequacy_audit import (
    CLASS_PAIR_MANIFEST_PATH,
    CorpusAdequacyArtifacts,
    CorpusAdequacyResult,
    CorpusAdequacyThresholds,
    analyze_corpus_adequacy,
    render_corpus_adequacy_report,
)
from .feature_analysis import FEATURE_SET_MANIFEST_PATH, load_feature_registry, load_feature_set_manifest


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}[status]


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "green"
    return max(statuses, key=_status_rank)


def _summary_status_to_color(status: str) -> str:
    return {"pass": "green", "warn": "yellow", "fail": "red"}[status]


CLASSIFIER_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "common_1d_classifier_study"
    / "classifier_manifest.json"
)


@dataclass(frozen=True, slots=True)
class CoverageReportSummary:
    overall_status: str
    corpus_status: str
    classifier_support_status: str
    feature_set_count: int
    classifier_count: int
    green_classifier_count: int
    yellow_classifier_count: int
    red_classifier_count: int


@dataclass(frozen=True, slots=True)
class CoverageReportResult:
    corpus_adequacy: CorpusAdequacyResult
    feature_set_summary_rows: tuple[dict[str, object], ...]
    feature_group_rows: tuple[dict[str, object], ...]
    classifier_support_rows: tuple[dict[str, object], ...]
    summary: CoverageReportSummary


@dataclass(frozen=True, slots=True)
class CoverageReportArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    feature_set_summary_path: Path
    feature_group_summary_path: Path
    classifier_support_path: Path


def load_classifier_manifest(manifest_path: str | Path | None = None) -> tuple[dict[str, object], ...]:
    path = Path(manifest_path) if manifest_path is not None else CLASSIFIER_MANIFEST_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(payload.get("classifiers", ()))


def _feature_set_summary_rows(corpus_adequacy: CorpusAdequacyResult) -> list[dict[str, object]]:
    manifest = load_feature_set_manifest(FEATURE_SET_MANIFEST_PATH)
    rows: list[dict[str, object]] = []
    for feature_set_name, entry in manifest.items():
        selected = [row for row in corpus_adequacy.feature_set_rows if row["feature_set"] == feature_set_name]
        if not selected:
            continue
        status = _worst_status([str(row["status"]) for row in selected])
        rows.append(
            {
                "feature_set": feature_set_name,
                "feature_count": len(selected),
                "history_behavior": str(entry.get("history_behavior", "unknown")),
                "description": str(entry.get("description", "")),
                "status": status,
                "green_features": sum(1 for row in selected if row["status"] == "green"),
                "yellow_features": sum(1 for row in selected if row["status"] == "yellow"),
                "red_features": sum(1 for row in selected if row["status"] == "red"),
                "avg_moderate_or_strong_fraction": sum(float(row["moderate_or_strong_fraction"]) for row in selected)
                / max(len(selected), 1),
            }
        )
    return rows


def _feature_group_rows(corpus_adequacy: CorpusAdequacyResult) -> list[dict[str, object]]:
    registry = load_feature_registry()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in corpus_adequacy.feature_set_rows:
        group_name = registry[str(row["feature"])].group
        grouped.setdefault(group_name, []).append(row)
    rows: list[dict[str, object]] = []
    for group_name, group_rows in sorted(grouped.items()):
        status = _worst_status([str(row["status"]) for row in group_rows])
        rows.append(
            {
                "feature_group": group_name,
                "feature_count": len(group_rows),
                "status": status,
                "green_features": sum(1 for row in group_rows if row["status"] == "green"),
                "yellow_features": sum(1 for row in group_rows if row["status"] == "yellow"),
                "red_features": sum(1 for row in group_rows if row["status"] == "red"),
            }
        )
    return rows


def _classifier_support_rows(corpus_adequacy: CorpusAdequacyResult) -> list[dict[str, object]]:
    feature_set_rows = _feature_set_summary_rows(corpus_adequacy)
    feature_set_status = {str(row["feature_set"]): str(row["status"]) for row in feature_set_rows}
    corpus_global_status = _worst_status(
        [
            _summary_status_to_color(corpus_adequacy.summary.class_pair_status),
            _summary_status_to_color(corpus_adequacy.summary.class_balance_status),
            _summary_status_to_color(corpus_adequacy.summary.covariate_status),
        ]
    )
    rows: list[dict[str, object]] = []
    for classifier in load_classifier_manifest():
        feature_set_name = str(
            classifier.get("feature_set_id")
            or classifier.get("requires_feature_set")
            or "all_engineered"
        )
        fs_status = feature_set_status.get(feature_set_name, "yellow")
        classifier_status = _worst_status([fs_status, corpus_global_status])
        recommendation = ""
        if fs_status == "red":
            recommendation = f"Improve corpus support for feature set `{feature_set_name}` before relying on this classifier."
        elif corpus_global_status == "red":
            recommendation = "Resolve corpus-wide class-pair, balance, or leakage failures before trusting this classifier."
        elif classifier_status == "yellow":
            recommendation = "Classifier is usable, but corpus warnings remain and should be reviewed."
        rows.append(
            {
                "classifier_id": str(classifier["id"]),
                "family": str(classifier.get("family", "unknown")),
                "feature_set": feature_set_name,
                "produces": " ".join(str(item) for item in classifier.get("produces", [])),
                "feature_set_status": fs_status,
                "corpus_global_status": corpus_global_status,
                "status": classifier_status,
                "ready_for_evaluation": classifier_status != "red",
                "recommendation": recommendation,
            }
        )
    return rows


def analyze_coverage_report(
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    thresholds: CorpusAdequacyThresholds | None = None,
) -> CoverageReportResult:
    corpus_adequacy = analyze_corpus_adequacy(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        thresholds=thresholds,
    )
    feature_set_summary_rows = _feature_set_summary_rows(corpus_adequacy)
    feature_group_rows = _feature_group_rows(corpus_adequacy)
    classifier_support_rows = _classifier_support_rows(corpus_adequacy)
    classifier_support_status = _worst_status([str(row["status"]) for row in classifier_support_rows])
    overall_status = _worst_status(
        [_summary_status_to_color(corpus_adequacy.summary.overall_status), classifier_support_status]
    )
    summary = CoverageReportSummary(
        overall_status=overall_status,
        corpus_status=_summary_status_to_color(corpus_adequacy.summary.overall_status),
        classifier_support_status=classifier_support_status,
        feature_set_count=len(feature_set_summary_rows),
        classifier_count=len(classifier_support_rows),
        green_classifier_count=sum(1 for row in classifier_support_rows if row["status"] == "green"),
        yellow_classifier_count=sum(1 for row in classifier_support_rows if row["status"] == "yellow"),
        red_classifier_count=sum(1 for row in classifier_support_rows if row["status"] == "red"),
    )
    return CoverageReportResult(
        corpus_adequacy=corpus_adequacy,
        feature_set_summary_rows=tuple(feature_set_summary_rows),
        feature_group_rows=tuple(feature_group_rows),
        classifier_support_rows=tuple(classifier_support_rows),
        summary=summary,
    )


def render_coverage_report(result: CoverageReportResult) -> str:
    lines = [
        "# Corpus Coverage Report",
        "",
        "This report summarizes how well the current synthetic corpus covers feature space and the declared classifier space.",
        "",
        "## Summary",
        "",
        f"- Overall status: {result.summary.overall_status}",
        f"- Corpus adequacy status: {result.corpus_adequacy.summary.overall_status}",
        f"- Classifier support status: {result.summary.classifier_support_status}",
        f"- Feature sets covered: {result.summary.feature_set_count}",
        f"- Declared classifiers: {result.summary.classifier_count}",
        f"- Classifier support counts: green={result.summary.green_classifier_count}, yellow={result.summary.yellow_classifier_count}, red={result.summary.red_classifier_count}",
        "",
        "## Feature Set Coverage",
        "",
        "| feature_set | feature_count | history_behavior | status | avg_moderate_or_strong_fraction |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in result.feature_set_summary_rows:
        lines.append(
            f"| {row['feature_set']} | {row['feature_count']} | {row['history_behavior']} | {row['status']} | {row['avg_moderate_or_strong_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Feature Group Coverage",
            "",
            "| feature_group | feature_count | status |",
            "| --- | ---: | --- |",
        ]
    )
    for row in result.feature_group_rows:
        lines.append(f"| {row['feature_group']} | {row['feature_count']} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Classifier Support",
            "",
            "| classifier_id | family | feature_set | feature_set_status | corpus_global_status | status | ready_for_evaluation |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result.classifier_support_rows:
        ready = "yes" if row["ready_for_evaluation"] else "no"
        lines.append(
            f"| {row['classifier_id']} | {row['family']} | {row['feature_set']} | {row['feature_set_status']} | {row['corpus_global_status']} | {row['status']} | {ready} |"
        )
    lines.extend(
        [
            "",
            "## Corpus Adequacy Detail",
            "",
            render_corpus_adequacy_report(result.corpus_adequacy),
        ]
    )
    return "\n".join(lines)


def write_coverage_report_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    thresholds: CorpusAdequacyThresholds | None = None,
) -> CoverageReportArtifacts:
    result = analyze_coverage_report(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        thresholds=thresholds,
    )
    output_root = Path(output_dir)
    run_dir = output_root / "coverage_report_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "coverage_report.md"
    summary_path = run_dir / "coverage_report_summary.json"
    feature_set_summary_path = run_dir / "feature_set_summary.csv"
    feature_group_summary_path = run_dir / "feature_group_summary.csv"
    classifier_support_path = run_dir / "classifier_support.csv"

    report_path.write_text(render_coverage_report(result), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "summary": asdict(result.summary),
                "corpus_adequacy_summary": asdict(result.corpus_adequacy.summary),
                "thresholds": asdict(result.corpus_adequacy.thresholds),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(
        feature_set_summary_path,
        [dict(row) for row in result.feature_set_summary_rows],
        [
            "feature_set",
            "feature_count",
            "history_behavior",
            "description",
            "status",
            "green_features",
            "yellow_features",
            "red_features",
            "avg_moderate_or_strong_fraction",
        ],
    )
    _write_csv(
        feature_group_summary_path,
        [dict(row) for row in result.feature_group_rows],
        ["feature_group", "feature_count", "status", "green_features", "yellow_features", "red_features"],
    )
    _write_csv(
        classifier_support_path,
        [dict(row) for row in result.classifier_support_rows],
        [
            "classifier_id",
            "family",
            "feature_set",
            "produces",
            "feature_set_status",
            "corpus_global_status",
            "status",
            "ready_for_evaluation",
            "recommendation",
        ],
    )
    return CoverageReportArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        feature_set_summary_path=feature_set_summary_path,
        feature_group_summary_path=feature_group_summary_path,
        classifier_support_path=classifier_support_path,
    )
