from __future__ import annotations

import csv
from pathlib import Path

import yaml

REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "decision_card.md",
    "static_audit_report.md",
    "static_audit_decision_card.md",
    "class_confusability_matrix.csv",
    "class_pair_diagnostics.csv",
    "class_feature_signature.csv",
    "class_observability.csv",
    "feature_relevance_table.csv",
    "feature_redundancy_matrix.csv",
    "feature_alias_candidates.csv",
    "feature_synergy_candidates.csv",
    "prior_pathology_report.csv",
    "prior_flip_thresholds.csv",
    "static_leakage_provenance_audit.csv",
    "figure_manifest.csv",
    "lane_proof_matrix.md",
    "hero_chart_contact_sheet.png",
    "02b_static_audit_decision_card.png",
    "02c_class_pair_confusability_matrix.png",
    "02e_feature_redundancy_graph.png",
    "02g_prior_pathology_surface.png",
)


def validate_static_admissibility_packet(packet_dir: str | Path, *, repo_root: str | Path | None = None) -> list[str]:
    base = Path(packet_dir)
    repo_base = None if repo_root is None else Path(repo_root)
    if (base / "packet_manifest.yaml").exists():
        manifest_payload = yaml.safe_load((base / "packet_manifest.yaml").read_text(encoding="utf-8")) or {}
        packet_id = str(manifest_payload.get("packet_id", ""))
        if packet_id == "01_static_admissibility":
            return _validate_exemplar_suite_packet(base, repo_root=repo_base)
        if packet_id == "01_static_admissibility_multi_domain_3d":
            return _validate_multi_domain_3d_packet(base, repo_root=repo_base)
    issues: list[str] = []
    for name in REQUIRED_FILES:
        if not (base / name).exists():
            issues.append(f"missing required packet file: {name}")

    if not issues:
        issues.extend(_validate_prior_regime(base / "prior_regime.csv"))
        issues.extend(_validate_class_matrix(base / "class_confusability_matrix.csv"))
        issues.extend(_validate_class_pair_diagnostics(base / "class_pair_diagnostics.csv"))
        issues.extend(_validate_class_feature_surface(base))
        issues.extend(_validate_feature_tables(base))
        issues.extend(_validate_feature_aliases(base / "feature_alias_candidates.csv"))
        issues.extend(_validate_leakage(base / "static_leakage_provenance_audit.csv"))
        issues.extend(_validate_synergy(base / "feature_synergy_candidates.csv"))
        issues.extend(_validate_decision_consistency(base))
        issues.extend(_validate_figure_manifest(base, repo_root=repo_base))
    return issues


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_manifest_path(base: Path, text: str, *, repo_root: Path | None = None) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    if repo_root is not None and text.startswith(("artifacts/", "docs/", "src/", "scripts/", "tests/", "experiments/")):
        return repo_root / path
    return base / path


def _validate_prior_regime(path: Path) -> list[str]:
    rows = _rows(path)
    total = sum(float(row["prior_probability"]) for row in rows)
    zero_classes = [row["class_name"] for row in rows if float(row["prior_probability"]) <= 0.0]
    issues: list[str] = []
    if abs(total - 1.0) > 1.0e-6:
        issues.append(f"prior probabilities must sum to 1.0, found {total:.6f}")
    if any(float(row["prior_probability"]) < 0.0 for row in rows):
        issues.append("prior probabilities must be nonnegative")
    if zero_classes:
        issues.append(f"prior regime assigns zero mass to declared classes: {', '.join(zero_classes)}")
    return issues


def _validate_class_matrix(path: Path) -> list[str]:
    rows = _rows(path)
    classes = [row["class"] for row in rows]
    issues: list[str] = []
    for row in rows:
        for class_name in classes:
            if class_name not in row:
                issues.append(f"class matrix missing column for {class_name}")
                continue
            mirror = next(other for other in rows if other["class"] == class_name)
            left = float(row[class_name] or 0.0)
            right = float(mirror[row["class"]] or 0.0)
            if abs(left - right) > 1.0e-9:
                issues.append(f"class matrix is not symmetric for {row['class']} and {class_name}")
    return issues


def _validate_class_pair_diagnostics(path: Path) -> list[str]:
    rows = _rows(path)
    required = {
        "class_a",
        "class_b",
        "status",
        "exact_shared_vector_count",
        "exact_shared_vector_rate",
        "signature_distance",
        "collision_status",
        "expected_signature_distance",
        "expected_signature_collision_status",
    }
    issues: list[str] = []
    for row in rows:
        missing = required - set(row)
        if missing:
            issues.append(f"class pair diagnostic row missing fields: {', '.join(sorted(missing))}")
            break
        if row["class_a"] == row["class_b"]:
            issues.append("class pair diagnostics must not contain self-pairs")
        if row["collision_status"] == "exact_feature_collision" and int(float(row["exact_shared_vector_count"])) <= 0:
            issues.append("exact feature collision rows must report at least one shared vector")
    return issues


def _validate_class_feature_surface(base: Path) -> list[str]:
    signature_rows = _rows(base / "class_feature_signature.csv")
    observability_rows = _rows(base / "class_observability.csv")
    issues: list[str] = []
    for row in signature_rows:
        for column in ("class_name", "feature", "sample_count", "status"):
            if row.get(column, "") == "":
                issues.append(f"class feature signature row missing `{column}`")
                break
    allowed_statuses = {
        "observable_on_declared_surface",
        "near_collision_warning",
        "exact_collision_bound",
        "unobserved_class",
    }
    for row in observability_rows:
        if row.get("class_name", "") == "":
            issues.append("class observability row missing class_name")
        if row.get("status") not in allowed_statuses:
            issues.append(f"unknown class observability status `{row.get('status', '')}`")
        if row.get("selection_status", "") == "" or row.get("expected_signature_coverage", "") == "":
            issues.append(f"class observability row missing future-signature fields for `{row.get('class_name', '<unknown>')}`")
    return issues


def _validate_feature_aliases(path: Path) -> list[str]:
    rows = _rows(path)
    issues: list[str] = []
    required = {"feature_a", "feature_b", "alias_type", "recommended_action"}
    for row in rows:
        missing = required - set(row)
        if missing:
            issues.append(f"feature alias row missing fields: {', '.join(sorted(missing))}")
            break
        if row["feature_a"] == row["feature_b"]:
            issues.append("feature alias rows must not contain self-pairs")
        if row["alias_type"] == "duplicate" and row["recommended_action"] != "drop_duplicate":
            issues.append("duplicate feature aliases must recommend drop_duplicate")
    return issues


def _validate_feature_tables(base: Path) -> list[str]:
    relevance = _rows(base / "feature_relevance_table.csv")
    redundancy = _rows(base / "feature_redundancy_matrix.csv")
    features = {row["feature"] for row in relevance}
    mentioned = set(features)
    for row in redundancy:
        mentioned.add(row["feature_a"])
        mentioned.add(row["feature_b"])
    missing = sorted(features - mentioned)
    return [f"declared features missing from redundancy output: {', '.join(missing)}"] if missing else []


def _validate_leakage(path: Path) -> list[str]:
    rows = _rows(path)
    required = {"feature", "provenance_tags", "online_available", "label_rule_overlap_flag", "status"}
    issues = []
    for row in rows:
        missing = required - set(row)
        if missing:
            issues.append(f"leakage row missing fields: {', '.join(sorted(missing))}")
        if row.get("online_available", "") == "":
            issues.append(f"leakage row missing online availability for {row.get('feature', '<unknown>')}")
    return issues


def _validate_synergy(path: Path) -> list[str]:
    rows = _rows(path)
    bad = [row for row in rows if row.get("status") not in {"ordinary", "synergy_candidate"}]
    if bad:
        return ["synergy must remain ordinary or synergy_candidate unless ablation-backed"]
    return []


def _validate_decision_consistency(base: Path) -> list[str]:
    decision_text = (base / "decision_card.md").read_text(encoding="utf-8")
    leakage_rows = _rows(base / "static_leakage_provenance_audit.csv")
    has_blocker = any(row["status"] == "blocker" for row in leakage_rows)
    if has_blocker and "promote_to_corpus_explorer" in decision_text:
        return ["static blocker cannot produce promote_to_corpus_explorer"]
    observability_rows = _rows(base / "class_observability.csv")
    if any(row.get("status") == "unobserved_class" for row in observability_rows) and "promote_to_corpus_explorer" in decision_text:
        return ["unobserved declared class cannot produce promote_to_corpus_explorer"]
    return []


def _validate_figure_manifest(base: Path, *, repo_root: Path | None = None) -> list[str]:
    rows = _rows(base / "figure_manifest.csv")
    issues = []
    for row in rows:
        figure_id = row.get("figure_id") or row.get("chart_id") or ""
        figure_path_text = row.get("figure_path") or row.get("packet_path") or row.get("path") or figure_id
        source_path_text = row.get("source_table") or row.get("source_artifact") or row.get("source_path") or ""
        figure = _resolve_manifest_path(base, figure_path_text, repo_root=repo_root)
        source_paths = [Path(part.strip()) for part in source_path_text.split(";") if part.strip()]
        if not figure.exists():
            issues.append(f"manifest figure missing: {figure_path_text}")
        for source in source_paths:
            source_path = _resolve_manifest_path(base, str(source), repo_root=repo_root)
            if not source_path.exists():
                issues.append(f"manifest source missing: {source}")
        if (
            figure_id in {"02f_feature_synergy_map", "02f_feature_synergy_map.png"}
            or figure_path_text.endswith("02f_feature_synergy_map.png")
        ) and "candidate" not in row.get("claim_boundary", ""):
            issues.append("synergy figure must keep candidate claim boundary")
    return issues


def _validate_no_absolute_local_paths(base: Path, relative_paths: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    forbidden_markers = ("/Users/rick/", "file:///Users/rick/")
    for relative_path in relative_paths:
        path = base / relative_path
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            if marker in text:
                issues.append(f"{relative_path}: packet text must not contain local absolute paths")
                break
    return issues


def _validate_exemplar_suite_packet(base: Path, *, repo_root: Path | None = None) -> list[str]:
    manifest_path = base / "packet_manifest.yaml"
    route_matrix_path = base / "source_artifacts" / "exemplar_route_matrix.csv"
    suite_manifest_path = base / "source_artifacts" / "exemplar_suite_manifest.csv"
    fingerprint_path = base / "source_artifacts" / "exemplar_fingerprint_scores.csv"
    card_manifest_path = base / "source_artifacts" / "exemplar_card_manifest.csv"
    hero_manifest_path = base / "hero_chart_manifest.csv"
    if not manifest_path.exists():
        return []
    issues: list[str] = []
    required = (
        "README.md",
        "quickstart.md",
        "packet_manifest.yaml",
        "decision_card.md",
        "validation_report.md",
        "claim_boundary.md",
        "automated_brief.md",
        "hero_chart_manifest.csv",
        "lane_proof_matrix.md",
        "figures/02a_static_bundle_ingestion_spine.png",
        "figures/02a_static_exemplar_suite_routing_matrix.png",
        "figures/02b_static_audit_decision_card.png",
        "figures/02m_static_exemplar_fingerprint_strip.png",
    )
    for name in required:
        if not (base / name).exists():
            issues.append(f"missing required exemplar-suite packet file: {name}")
    if not route_matrix_path.exists() or not suite_manifest_path.exists() or not fingerprint_path.exists() or not card_manifest_path.exists():
        issues.append("exemplar-suite packet missing one or more source artifact tables")
        return issues

    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest_source = manifest_payload.get("suite_manifest")
    declared_exemplars = []
    if manifest_source:
        source_path = _resolve_manifest_path(base, str(manifest_source), repo_root=repo_root)
        if not source_path.exists():
            issues.append(f"suite manifest missing: {manifest_source}")
        else:
            source_payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
            declared_exemplars = list(source_payload.get("epic1_static_admissibility_exemplars", ()))

    route_rows = _rows(route_matrix_path)
    suite_rows = _rows(suite_manifest_path)
    card_rows = _rows(card_manifest_path)
    hero_rows = _rows(hero_manifest_path) if hero_manifest_path.exists() else []
    figure_map = {row.get("chart_id", ""): row for row in hero_rows}

    route_by_id = {row["exemplar_id"]: row for row in route_rows}
    suite_by_id = {row["exemplar_id"]: row for row in suite_rows}
    card_by_id = {row["exemplar_id"]: row for row in card_rows}
    for exemplar in declared_exemplars:
        exemplar_id = str(exemplar["exemplar_id"])
        expected_route = str(exemplar["expected_status"])
        if exemplar_id not in route_by_id:
            issues.append(f"missing route matrix row for exemplar {exemplar_id}")
            continue
        if exemplar_id not in suite_by_id:
            issues.append(f"missing suite manifest row for exemplar {exemplar_id}")
        if exemplar_id not in card_by_id:
            issues.append(f"missing card manifest row for exemplar {exemplar_id}")
            continue
        route_row = route_by_id[exemplar_id]
        actual_route = route_row["actual_route"]
        if route_row["expected_route"] != expected_route:
            issues.append(f"{exemplar_id}: route matrix expected_route does not match suite manifest")
        if actual_route != expected_route:
            issues.append(f"{exemplar_id}: expected_route != actual_route")
        if route_row["validator_result"] != "pass":
            issues.append(f"{exemplar_id}: validator_result must be pass")
        card_png = base / card_by_id[exemplar_id]["card_png"]
        card_md = base / card_by_id[exemplar_id]["card_md"]
        source_bundle = base / card_by_id[exemplar_id]["source_bundle"]
        source_artifacts = base / card_by_id[exemplar_id]["source_artifacts"]
        if not card_png.exists():
            issues.append(f"{exemplar_id}: missing exemplar card figure")
        if not card_md.exists():
            issues.append(f"{exemplar_id}: missing exemplar card markdown")
        if not source_bundle.exists():
            issues.append(f"{exemplar_id}: missing copied source bundle")
        if not source_artifacts.exists():
            issues.append(f"{exemplar_id}: missing copied source artifacts")
        if exemplar_id == "leakage_blocker_family" and actual_route == "promote_to_corpus_explorer":
            issues.append("leakage_blocker_family cannot produce promote status")
        if exemplar_id == "class_overlap_boundary_family" and actual_route == "promote_to_corpus_explorer":
            issues.append("class_overlap_boundary_family cannot produce clean promote status")
        if exemplar_id == "prior_domination_family" and actual_route == "promote_to_corpus_explorer":
            issues.append("prior_domination_family cannot produce clean promote status")
        if exemplar_id == "coverage_thin_cells_family" and route_row["coverage_status"] == "pass":
            issues.append("coverage_thin_cells_family must show a coverage warning")
        if exemplar_id == "redundancy_synergy_family" and route_row["synergy_status"] == "block":
            issues.append("redundancy_synergy_family must keep synergy at candidate status")

    synergy_row = figure_map.get("02f_feature_synergy_map") or figure_map.get("02f_feature_synergy_map.png")
    if synergy_row is not None and "candidate" not in synergy_row.get("claim_boundary", "").lower():
        issues.append("synergy figure must keep candidate claim boundary")
    for row in hero_rows:
        for column in ("chart_id", "path", "evidence_tier", "source_artifact", "claim", "claim_boundary"):
            if not row.get(column):
                issues.append(f"hero_chart_manifest row missing `{column}`")
        path = base / row.get("path", "")
        if not path.exists():
            issues.append(f"hero chart missing: {row.get('path', '')}")
        for source_text in [part.strip() for part in row.get("source_artifact", "").split(";") if part.strip()]:
            source_path = _resolve_manifest_path(base, source_text, repo_root=repo_root)
            if not source_path.exists():
                issues.append(f"hero chart source missing: {source_text}")
    issues.extend(
        _validate_no_absolute_local_paths(
            base,
            (
                "README.md",
                "quickstart.md",
                "decision_card.md",
                "validation_report.md",
                "claim_boundary.md",
                "packet_manifest.yaml",
                "hero_chart_manifest.csv",
                "lane_proof_matrix.md",
                "automated_brief.md",
            ),
        )
    )
    return issues


def _validate_multi_domain_3d_packet(base: Path, *, repo_root: Path | None = None) -> list[str]:
    manifest_path = base / "packet_manifest.yaml"
    bundle_matrix_path = base / "source_artifacts" / "multidomain_bundle_route_matrix.csv"
    diagnostic_matrix_path = base / "source_artifacts" / "multidomain_bundle_diagnostics.csv"
    hero_manifest_path = base / "hero_chart_manifest.csv"
    issues: list[str] = []
    required = (
        "README.md",
        "quickstart.md",
        "decision_card.md",
        "validation_report.md",
        "claim_boundary.md",
        "packet_manifest.yaml",
        "hero_chart_manifest.csv",
        "lane_proof_matrix.md",
        "brief/automated_brief.md",
        "estimator_reliability_report.md",
        "feature_alias_and_redundancy_report.md",
        "latex/multidomain_3d_static_admissibility.tex",
        "figures/MD3D_01_bundle_ingestion_spine.png",
        "figures/MD3D_05_class_feature_excitation_matrix.png",
        "figures/MD3D_07_prior_pathology_surface.png",
        "figures/MD3D_10_unobservable_and_leakage_audit.png",
        "figures/MD3D_11_static_decision_card.png",
        "figures/MD3D_13_estimator_reliability_dashboard.png",
        "figures/MD3D_15_prior_evidence_budget.png",
        "figures/MD3D_19_threshold_subsumption_map.png",
        "figures/MD3D_21_decision_redundancy_matrix.png",
        "source_artifacts/multi_domain_3d_class_schema.csv",
        "source_artifacts/multi_domain_3d_feature_schema.csv",
        "source_artifacts/multi_domain_3d_prior_regimes.csv",
        "source_artifacts/multi_domain_3d_class_feature_signature.csv",
        "source_artifacts/multi_domain_3d_synthetic_samples.csv",
        "source_artifacts/multi_domain_3d_expected_confusions.csv",
        "source_artifacts/multi_domain_3d_expected_synergy_pairs.csv",
        "source_artifacts/multi_domain_3d_blocked_features.csv",
        "source_artifacts/multi_domain_3d_observability_gaps.csv",
        "source_artifacts/static_metric_uncertainty.csv",
        "source_artifacts/pairwise_error_bound_proxy.csv",
        "source_artifacts/prior_evidence_budget.csv",
        "source_artifacts/sample_size_adequacy_report.csv",
        "source_artifacts/metric_assumption_registry.csv",
        "source_artifacts/bound_validity_manifest.yaml",
        "source_artifacts/bootstrap_metric_distributions.parquet",
        "source_artifacts/permutation_null_summary.csv",
        "source_artifacts/feature_alias_candidates.csv",
        "source_artifacts/feature_threshold_subsumption.csv",
        "source_artifacts/feature_functional_equivalence.csv",
        "source_artifacts/feature_decision_redundancy.csv",
        "source_artifacts/feature_redundancy_clusters.csv",
    )
    for name in required:
        if not (base / name).exists():
            issues.append(f"missing required multi-domain packet file: {name}")
    if not manifest_path.exists() or not bundle_matrix_path.exists() or not diagnostic_matrix_path.exists():
        return issues

    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest_payload.get("packet_id") != "01_static_admissibility_multi_domain_3d":
        issues.append("multi-domain packet_manifest.yaml must declare packet_id 01_static_admissibility_multi_domain_3d")

    bundle_rows = _rows(bundle_matrix_path)
    diagnostic_rows = _rows(diagnostic_matrix_path)
    hero_rows = _rows(hero_manifest_path) if hero_manifest_path.exists() else []
    diagnostics_by_id = {row["bundle_id"]: row for row in diagnostic_rows}
    expected_routes = {
        "clean_multidomain_3d_bundle": "promote_to_corpus_explorer",
        "prior_pathology_multidomain_3d_bundle": "revise_prior",
        "redundancy_synergy_multidomain_3d_bundle": "promote_to_corpus_explorer",
        "unobservable_navy_space_bundle": "revise_class_set",
        "leakage_blocker_multidomain_3d_bundle": "reject",
    }
    for row in bundle_rows:
        bundle_id = row["bundle_id"]
        expected_route = expected_routes.get(bundle_id)
        if expected_route is None:
            issues.append(f"unexpected multi-domain bundle row `{bundle_id}`")
            continue
        if row["expected_route"] != expected_route:
            issues.append(f"{bundle_id}: bundle matrix expected_route mismatch")
        if row["actual_route"] != expected_route:
            issues.append(f"{bundle_id}: expected_route != actual_route")
        if row["validator_status"] != "pass":
            issues.append(f"{bundle_id}: validator_status must be pass")
        if not row.get("decision_confidence"):
            issues.append(f"{bundle_id}: decision_confidence must be populated")
        source_bundle = base / str(row["source_bundle"])
        source_run = base / str(row["source_run"])
        if not source_bundle.exists():
            issues.append(f"{bundle_id}: missing source bundle directory")
        if not source_run.exists():
            issues.append(f"{bundle_id}: missing source run directory")
        if bundle_id not in diagnostics_by_id:
            issues.append(f"{bundle_id}: missing diagnostics row")
            continue
        diagnostic_row = diagnostics_by_id[bundle_id]
        if diagnostic_row["expected_route"] != expected_route:
            issues.append(f"{bundle_id}: diagnostics expected_route mismatch")
        if diagnostic_row["validator_result"] != "pass":
            issues.append(f"{bundle_id}: diagnostics validator_result must be pass")
        if bundle_id == "redundancy_synergy_multidomain_3d_bundle" and diagnostic_row["synergy_status"] == "block":
            issues.append("redundancy_synergy_multidomain_3d_bundle must keep synergy at candidate status")
        if bundle_id == "prior_pathology_multidomain_3d_bundle" and diagnostic_row["prior_pathology_status"] == "pass":
            issues.append("prior_pathology_multidomain_3d_bundle must show prior pathology")
        if bundle_id == "unobservable_navy_space_bundle" and diagnostic_row["class_separability_status"] == "pass":
            issues.append("unobservable_navy_space_bundle must not look cleanly separable")
        if bundle_id == "leakage_blocker_multidomain_3d_bundle" and diagnostic_row["leakage_status"] != "block":
            issues.append("leakage_blocker_multidomain_3d_bundle must show leakage block")

    prior_rows = _rows(base / "source_artifacts" / "multi_domain_3d_prior_regimes.csv")
    totals: dict[str, float] = {}
    for row in prior_rows:
        totals.setdefault(row["prior_regime"], 0.0)
        totals[row["prior_regime"]] += float(row["prior_probability"])
    for regime, total in totals.items():
        if abs(total - 1.0) > 1.0e-6:
            issues.append(f"prior regime `{regime}` must sum to 1.0, found {total:.6f}")

    feature_rows = _rows(base / "source_artifacts" / "multi_domain_3d_feature_schema.csv")
    required_feature_columns = {
        "feature_id",
        "feature_group",
        "observable_from_3d_track",
        "online_available",
        "uses_future_window",
        "uses_generator_metadata",
        "uses_identity_or_catalog_lookup",
        "label_rule_overlap",
        "allowed_for_static_audit",
        "leakage_status",
    }
    for row in feature_rows:
        missing = required_feature_columns - set(row)
        if missing:
            issues.append(f"multi-domain feature schema row missing fields: {', '.join(sorted(missing))}")
            break

    class_rows = _rows(base / "source_artifacts" / "multi_domain_3d_class_schema.csv")
    for row in class_rows:
        if not row.get("expected_confusions") or not row.get("decisionability_notes"):
            issues.append(f"class schema row `{row.get('class_id', '<unknown>')}` must include confusions and decisionability notes")

    uncertainty_rows = _rows(base / "source_artifacts" / "static_metric_uncertainty.csv")
    for row in uncertainty_rows:
        for column in ("metric_id", "metric_family", "point_estimate", "ci_low", "ci_high", "ci_method", "n_effective", "stability_status", "evidence_tier"):
            if row.get(column, "") == "":
                issues.append(f"static_metric_uncertainty row missing `{column}`")
                break

    assumption_rows = _rows(base / "source_artifacts" / "metric_assumption_registry.csv")
    for row in assumption_rows:
        if not row.get("metric_id") or not row.get("recommended_evidence_tier"):
            issues.append("metric_assumption_registry rows must include metric_id and recommended_evidence_tier")
            break

    decision_text = (base / "decision_card.md").read_text(encoding="utf-8")
    if "decision_confidence" not in decision_text:
        issues.append("decision_card.md must include decision_confidence")
    readme_text = (base / "README.md").read_text(encoding="utf-8")
    quickstart_text = (base / "quickstart.md").read_text(encoding="utf-8")
    claim_boundary_text = (base / "claim_boundary.md").read_text(encoding="utf-8")
    if "not a full 3D tracking implementation" not in readme_text and "not a full 3D tracking implementation" not in quickstart_text:
        issues.append("MD3D packet must state that it is not a full 3D tracking implementation")
    if "not operational guarantees" not in claim_boundary_text and "not operational guarantees" not in (base / "brief" / "automated_brief.md").read_text(encoding="utf-8"):
        issues.append("MD3D packet must state that static bounds are not operational guarantees")

    alias_rows = _rows(base / "source_artifacts" / "feature_alias_candidates.csv")
    threshold_rows = _rows(base / "source_artifacts" / "feature_threshold_subsumption.csv")
    if not alias_rows:
        issues.append("feature_alias_candidates.csv must include at least one alias row")
    if not threshold_rows:
        issues.append("feature_threshold_subsumption.csv must include at least one threshold row")
    for row in alias_rows:
        if not row.get("alias_type"):
            issues.append("feature_alias_candidates.csv rows must include alias_type")
            break
    for row in threshold_rows:
        for column in ("threshold_gap_over_uncertainty", "retention_confidence", "required_followup"):
            if row.get(column, "") == "":
                issues.append(f"feature_threshold_subsumption row missing `{column}`")
                break
        recommended_action = str(row.get("recommended_action", ""))
        if recommended_action == "retain_pair_specific":
            gap_ratio = float(row.get("threshold_gap_over_uncertainty", "0") or 0.0)
            if gap_ratio < 1.0:
                issues.append(
                    "below-uncertainty threshold retention must be candidate-level or include follow-up"
                )
        if str(row.get("recommended_action", "")).startswith("retain_pair_specific"):
            if row.get("boundary_slice_count", "") == "" or row.get("boundary_slice_class_mix", "") == "":
                issues.append("retain_pair_specific threshold rows must include boundary slice count and class mix")
    decision_rows = _rows(base / "source_artifacts" / "feature_decision_redundancy.csv")
    for row in decision_rows:
        if not row.get("affected_class_pair") and row.get("affected_class_pair") != "global":
            issues.append("feature_decision_redundancy rows must include affected class pair or global scope")
            break

    for row in hero_rows:
        for column in ("chart_id", "path", "evidence_tier", "source_artifact", "claim", "claim_boundary"):
            if not row.get(column):
                issues.append(f"hero_chart_manifest row missing `{column}`")
        path = base / row.get("path", "")
        if not path.exists():
            issues.append(f"hero chart missing: {row.get('path', '')}")
        for source_text in [part.strip() for part in row.get("source_artifact", "").split(";") if part.strip()]:
            source_path = _resolve_manifest_path(base, source_text, repo_root=repo_root)
            if not source_path.exists():
                issues.append(f"hero chart source missing: {source_text}")
        if row.get("chart_id") == "MD3D_09_redundancy_synergy_graph" and "candidate" not in row.get("claim_boundary", "").lower():
            issues.append("MD3D_09_redundancy_synergy_graph must keep candidate claim boundary")
        if row.get("chart_id") in {"MD3D_14_pairwise_error_bound_proxy", "MD3D_15_prior_evidence_budget"} and "proxy" not in row.get("claim_boundary", "").lower():
            issues.append(f"{row.get('chart_id')}: bound chart must keep proxy claim boundary")
        if row.get("chart_id") in {"MD3D_19_threshold_subsumption_map", "MD3D_21_decision_redundancy_matrix"} and "proxy" not in row.get("claim_boundary", "").lower():
            issues.append(f"{row.get('chart_id')}: redundancy chart must keep proxy claim boundary")
    issues.extend(
        _validate_no_absolute_local_paths(
            base,
            (
                "README.md",
                "quickstart.md",
                "decision_card.md",
                "validation_report.md",
                "claim_boundary.md",
                "packet_manifest.yaml",
                "hero_chart_manifest.csv",
                "lane_proof_matrix.md",
                "brief/automated_brief.md",
                "estimator_reliability_report.md",
                "feature_alias_and_redundancy_report.md",
                "latex/multidomain_3d_static_admissibility.tex",
            ),
        )
    )
    return issues
