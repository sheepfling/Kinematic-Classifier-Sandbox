from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..analysis.feature_analysis import (
    FEATURE_SET_MANIFEST_PATH,
    load_feature_registry,
    load_feature_set_manifest,
)
from ..utils.runtime import repo_root
from .adequacy_audit import (
    CorpusAdequacyResult,
    CorpusAdequacyThresholds,
    analyze_corpus_adequacy,
    render_corpus_adequacy_report,
)
from .coverage_contracts import CoverageReportResult, CoverageReportSummary


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}[status]

def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "green"
    return max(statuses, key=_status_rank)

def _summary_status_to_color(status: str) -> str:
    try:
        return {"pass": "green", "warn": "yellow", "fail": "red"}[status]
    except KeyError:
        pass
    ####
    return "gray"
####

CLASSIFIER_MANIFEST_PATH = (
        repo_root()
        / "experiments"
        / "common_1d_classifier_study"
        / "classifier_manifest.json"
)

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
        corpus_adequacy_result: CorpusAdequacyResult | None = None,
) -> CoverageReportResult:
    corpus_adequacy = corpus_adequacy_result or analyze_corpus_adequacy(
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
    report = MarkdownDocument("Corpus Coverage Report")
    report.paragraph(
        "This report summarizes how well the current synthetic corpus covers feature space and the declared classifier space."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Overall status: {result.summary.overall_status}",
            f"Corpus adequacy status: {result.corpus_adequacy.summary.overall_status}",
            f"Classifier support status: {result.summary.classifier_support_status}",
            f"Feature sets covered: {result.summary.feature_set_count}",
            f"Declared classifiers: {result.summary.classifier_count}",
            (
                "Classifier support counts: "
                f"green={result.summary.green_classifier_count}, yellow={result.summary.yellow_classifier_count}, red={result.summary.red_classifier_count}"
            ),
        ]
    )
    report.heading("Feature Set Coverage", level=2)
    report.table(
        ["feature_set", "feature_count", "history_behavior", "status", "avg_moderate_or_strong_fraction"],
        [
            (
                row["feature_set"],
                row["feature_count"],
                row["history_behavior"],
                row["status"],
                f"{row['avg_moderate_or_strong_fraction']:.3f}",
            )
            for row in result.feature_set_summary_rows
        ],
    )
    report.heading("Feature Group Coverage", level=2)
    report.table(
        ["feature_group", "feature_count", "status"],
        [
            (
                row["feature_group"],
                row["feature_count"],
                row["status"],
            )
            for row in result.feature_group_rows
        ],
    )
    report.heading("Classifier Support", level=2)
    report.table(
        [
            "classifier_id",
            "family",
            "feature_set",
            "feature_set_status",
            "corpus_global_status",
            "status",
            "ready_for_evaluation",
        ],
        [
            (
                row["classifier_id"],
                row["family"],
                row["feature_set"],
                row["feature_set_status"],
                row["corpus_global_status"],
                row["status"],
                "yes" if row["ready_for_evaluation"] else "no",
            )
            for row in result.classifier_support_rows
        ],
    )
    report.heading("Corpus Adequacy Detail", level=2)
    report.paragraph(render_corpus_adequacy_report(result.corpus_adequacy))
    return report.text()
