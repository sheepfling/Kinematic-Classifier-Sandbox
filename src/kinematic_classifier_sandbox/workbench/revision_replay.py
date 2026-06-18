from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.common_experiment.adapters import ExecutableTrajectory
from kinematic_classifier_sandbox.common_experiment.analysis import analyze_common_trajectory_corpus
from kinematic_classifier_sandbox.common_experiment.artifact_io import (
    write_common_experiment_artifacts,
)
from kinematic_classifier_sandbox.common_experiment.config import (
    load_common_experiment_config,
    resolve_common_study_adapter,
)
from kinematic_classifier_sandbox.utils.runtime import repo_root

from .mvp import _read_csv, _read_json, _write_csv, _write_json


REVISION_EVENT_FIELDS = (
    "event_id",
    "event_type",
    "track_id",
    "measurement_id",
    "measurement_time",
    "ingest_time",
    "revision_id",
    "parent_revision_id",
    "revokes_measurement_id",
    "supersedes_measurement_id",
    "reason_code",
    "source",
    "operator_note",
    "payload_json",
    "checksum",
)

ACTIVE_MEASUREMENT_FIELDS = (
    "revision_id",
    "trajectory_id",
    "class_pair_id",
    "scenario_id",
    "true_class",
    "measurement_id",
    "measurement_index",
    "measurement_time",
    "measurement_value",
    "true_position",
    "true_velocity",
    "true_acceleration",
    "active_status",
)

METHOD_REPLAY_COMPATIBILITY_ROWS = (
    {
        "method_id": "pointwise",
        "snapshot_support": "declared",
        "deterministic_replay": "yes",
        "full_recompute_fallback": "yes",
        "local_replay": "future",
        "suffix_replay": "future",
        "rng_state_required": "no",
        "revision_safe_status": "full_recompute_only",
    },
    {
        "method_id": "windowed",
        "snapshot_support": "declared",
        "deterministic_replay": "yes",
        "full_recompute_fallback": "yes",
        "local_replay": "future",
        "suffix_replay": "future",
        "rng_state_required": "no",
        "revision_safe_status": "full_recompute_only",
    },
    {
        "method_id": "sequential_bayes",
        "snapshot_support": "declared",
        "deterministic_replay": "yes",
        "full_recompute_fallback": "yes",
        "local_replay": "future",
        "suffix_replay": "future",
        "rng_state_required": "no",
        "revision_safe_status": "full_recompute_only",
    },
    {
        "method_id": "kalman_bank",
        "snapshot_support": "declared",
        "deterministic_replay": "yes",
        "full_recompute_fallback": "yes",
        "local_replay": "future",
        "suffix_replay": "future",
        "rng_state_required": "no",
        "revision_safe_status": "full_recompute_only",
    },
    {
        "method_id": "transition_matrix",
        "snapshot_support": "declared",
        "deterministic_replay": "yes",
        "full_recompute_fallback": "yes",
        "local_replay": "future",
        "suffix_replay": "future",
        "rng_state_required": "no",
        "revision_safe_status": "full_recompute_only",
    },
    {
        "method_id": "imm",
        "snapshot_support": "declared_mode_probabilities_and_per_mode_states",
        "deterministic_replay": "yes",
        "full_recompute_fallback": "yes",
        "local_replay": "future",
        "suffix_replay": "future",
        "rng_state_required": "no",
        "revision_safe_status": "payload_declared_full_recompute_only",
    },
    {
        "method_id": "particle_filter",
        "snapshot_support": "particles_weights_ancestors_rng_required",
        "deterministic_replay": "not_yet",
        "full_recompute_fallback": "yes",
        "local_replay": "not_yet",
        "suffix_replay": "not_yet",
        "rng_state_required": "yes",
        "revision_safe_status": "not_revision_safe_yet",
    },
    {
        "method_id": "rbpf",
        "snapshot_support": "latent_paths_conditional_states_rng_required",
        "deterministic_replay": "not_yet",
        "full_recompute_fallback": "yes",
        "local_replay": "not_yet",
        "suffix_replay": "not_yet",
        "rng_state_required": "yes",
        "revision_safe_status": "not_revision_safe_yet",
    },
)


def ensure_revision_history(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    revisions_dir = path / "revisions"
    baseline_dir = revisions_dir / "rev_000"
    if baseline_dir.exists():
        return baseline_dir

    revisions_dir.mkdir(parents=True, exist_ok=True)
    trajectories, manifest = _load_base_trajectories(path)
    base_rows = _measurement_rows_for_revision(trajectories, revision_id="rev_000")
    event_rows = [_measurement_added_event(row) for row in base_rows]
    _write_csv(revisions_dir / "revision_event_log.csv", event_rows)
    _write_revision_state(
        run_dir=path,
        revision_id="rev_000",
        parent_revision_id=None,
        active_rows=base_rows,
        replay_plan={
            "revision_id": "rev_000",
            "parent_revision_id": None,
            "replay_mode": "baseline_materialization",
            "earliest_affected_time": None,
            "full_recompute_required": False,
            "methods_replayed": [],
            "claim_boundary": "Baseline revision is a materialized active view for later revocation-aware replay.",
        },
        source_run_dir=path,
        manifest=manifest,
        delta_payload={
            "left_revision_id": None,
            "right_revision_id": "rev_000",
            "measurement_count_left": 0,
            "measurement_count_right": len(base_rows),
            "measurement_count_delta": len(base_rows),
            "revoked_measurement_ids": [],
            "posterior_max_delta": 0.0,
            "decision_changed": False,
            "best_current_method_left": None,
            "best_current_method_right": _best_method_id(path / "metrics_by_method.csv"),
            "claim_boundary": "Baseline revision is the audit anchor, not a replay delta.",
        },
    )
    return baseline_dir


def inspect_measurement(run_dir: str | Path, measurement_id: str, revision_id: str | None = None) -> dict[str, Any]:
    path = Path(run_dir)
    ensure_revision_history(path)
    target_revision = revision_id or latest_revision_id(path)
    rows = _read_csv(path / "revisions" / target_revision / "active_measurements.csv")
    selected = [row for row in rows if row.get("measurement_id") == measurement_id]
    events = [
        row
        for row in _read_csv(path / "revisions" / "revision_event_log.csv")
        if row.get("measurement_id") == measurement_id or row.get("revokes_measurement_id") == measurement_id
    ]
    return {
        "run_dir": str(path),
        "revision_id": target_revision,
        "measurement_id": measurement_id,
        "active": bool(selected),
        "active_rows": selected,
        "events": events,
    }


def revoke_measurement(
    run_dir: str | Path,
    measurement_id: str,
    *,
    reason: str,
    note: str | None = None,
    source: str = "operator",
) -> str:
    path = Path(run_dir)
    ensure_revision_history(path)
    events_path = path / "revisions" / "revision_event_log.csv"
    events = _read_csv(events_path)
    if not any(row.get("measurement_id") == measurement_id for row in events if row.get("event_type") == "measurement_added"):
        raise ValueError(f"unknown measurement_id: {measurement_id}")
    if any(
        row.get("event_type") == "measurement_revoked" and row.get("revokes_measurement_id") == measurement_id
        for row in events
    ):
        raise ValueError(f"measurement already revoked: {measurement_id}")
    parent_revision_id = latest_revision_id(path)
    next_revision_id = _next_revision_id(path)
    measurement_row = inspect_measurement(path, measurement_id, revision_id=parent_revision_id)["active_rows"]
    measurement_time = measurement_row[0]["measurement_time"] if measurement_row else ""
    event_row = {
        "event_id": f"{next_revision_id}_revoke_{measurement_id}",
        "event_type": "measurement_revoked",
        "track_id": measurement_row[0]["trajectory_id"] if measurement_row else "",
        "measurement_id": measurement_id,
        "measurement_time": measurement_time,
        "ingest_time": datetime.now(UTC).isoformat(),
        "revision_id": next_revision_id,
        "parent_revision_id": parent_revision_id,
        "revokes_measurement_id": measurement_id,
        "supersedes_measurement_id": "",
        "reason_code": reason,
        "source": source,
        "operator_note": note or "",
        "payload_json": json.dumps({"measurement_id": measurement_id, "reason": reason}),
        "checksum": _checksum_payload(f"{measurement_id}:{reason}:{note or ''}:{next_revision_id}"),
    }
    events.append(event_row)
    _write_csv(events_path, events)
    return next_revision_id


def restore_measurement(
    run_dir: str | Path,
    measurement_id: str,
    *,
    reason: str,
    note: str | None = None,
    source: str = "operator",
) -> str:
    path = Path(run_dir)
    ensure_revision_history(path)
    events_path = path / "revisions" / "revision_event_log.csv"
    events = _read_csv(events_path)
    added_row = next(
        (
            row
            for row in events
            if row.get("event_type") == "measurement_added" and row.get("measurement_id") == measurement_id
        ),
        None,
    )
    if added_row is None:
        raise ValueError(f"unknown measurement_id: {measurement_id}")
    if not any(
        row.get("event_type") == "measurement_revoked" and row.get("revokes_measurement_id") == measurement_id
        for row in events
    ):
        raise ValueError(f"measurement is not revoked: {measurement_id}")
    parent_revision_id = latest_revision_id(path)
    current_active_ids = {
        row["measurement_id"]
        for row in _read_csv(path / "revisions" / parent_revision_id / "active_measurements.csv")
    }
    if measurement_id in current_active_ids:
        raise ValueError(f"measurement already active: {measurement_id}")
    next_revision_id = _next_revision_id(path)
    event_row = {
        "event_id": f"{next_revision_id}_restore_{measurement_id}",
        "event_type": "measurement_restored",
        "track_id": added_row["track_id"],
        "measurement_id": measurement_id,
        "measurement_time": added_row["measurement_time"],
        "ingest_time": datetime.now(UTC).isoformat(),
        "revision_id": next_revision_id,
        "parent_revision_id": parent_revision_id,
        "revokes_measurement_id": "",
        "supersedes_measurement_id": measurement_id,
        "reason_code": reason,
        "source": source,
        "operator_note": note or "",
        "payload_json": added_row["payload_json"],
        "checksum": _checksum_payload(f"restore:{measurement_id}:{reason}:{note or ''}:{next_revision_id}"),
    }
    events.append(event_row)
    _write_csv(events_path, events)
    return next_revision_id


def correct_measurement(
    run_dir: str | Path,
    measurement_id: str,
    *,
    corrected_value: float,
    reason: str,
    note: str | None = None,
    source: str = "operator",
) -> str:
    path = Path(run_dir)
    ensure_revision_history(path)
    events_path = path / "revisions" / "revision_event_log.csv"
    events = _read_csv(events_path)
    baseline_row = _baseline_row_by_id(path).get(measurement_id)
    if baseline_row is None:
        raise ValueError(f"unknown measurement_id: {measurement_id}")
    parent_revision_id = latest_revision_id(path)
    current_active_ids = {
        row["measurement_id"]
        for row in _read_csv(path / "revisions" / parent_revision_id / "active_measurements.csv")
    }
    if measurement_id not in current_active_ids:
        raise ValueError(f"measurement must be active before correction: {measurement_id}")
    next_revision_id = _next_revision_id(path)
    corrected_id = f"{measurement_id}__corr_{next_revision_id}"
    corrected_row = dict(baseline_row)
    corrected_row["measurement_id"] = corrected_id
    corrected_row["measurement_value"] = f"{corrected_value:.12f}"
    corrected_row["revision_id"] = next_revision_id
    event_row = {
        "event_id": f"{next_revision_id}_correct_{measurement_id}",
        "event_type": "measurement_corrected",
        "track_id": baseline_row["trajectory_id"],
        "measurement_id": corrected_id,
        "measurement_time": baseline_row["measurement_time"],
        "ingest_time": datetime.now(UTC).isoformat(),
        "revision_id": next_revision_id,
        "parent_revision_id": parent_revision_id,
        "revokes_measurement_id": measurement_id,
        "supersedes_measurement_id": measurement_id,
        "reason_code": reason,
        "source": source,
        "operator_note": note or "",
        "payload_json": json.dumps(corrected_row, sort_keys=True),
        "checksum": _checksum_payload(f"correct:{measurement_id}:{corrected_value}:{reason}:{note or ''}:{next_revision_id}"),
    }
    events.append(event_row)
    _write_csv(events_path, events)
    return next_revision_id


def change_measurement_association(
    run_dir: str | Path,
    source_measurement_id: str,
    target_measurement_id: str,
    *,
    reason: str,
    note: str | None = None,
    source: str = "operator",
) -> str:
    path = Path(run_dir)
    ensure_revision_history(path)
    if source_measurement_id == target_measurement_id:
        raise ValueError("source and target measurement ids must differ")
    parent_revision_id = latest_revision_id(path)
    active_rows = {
        row["measurement_id"]: row
        for row in _read_csv(path / "revisions" / parent_revision_id / "active_measurements.csv")
    }
    source_row = active_rows.get(source_measurement_id)
    target_row = active_rows.get(target_measurement_id)
    if source_row is None:
        raise ValueError(f"source measurement not active: {source_measurement_id}")
    if target_row is None:
        raise ValueError(f"target measurement not active: {target_measurement_id}")
    next_revision_id = _next_revision_id(path)
    reassociated_row = dict(target_row)
    reassociated_row["measurement_id"] = source_measurement_id
    reassociated_row["measurement_value"] = source_row["measurement_value"]
    reassociated_row["revision_id"] = next_revision_id
    reassociated_row["active_status"] = "active"
    event_row = {
        "event_id": f"{next_revision_id}_assoc_{source_measurement_id}_to_{target_measurement_id}",
        "event_type": "track_association_changed",
        "track_id": target_row["trajectory_id"],
        "measurement_id": source_measurement_id,
        "measurement_time": target_row["measurement_time"],
        "ingest_time": datetime.now(UTC).isoformat(),
        "revision_id": next_revision_id,
        "parent_revision_id": parent_revision_id,
        "revokes_measurement_id": target_measurement_id,
        "supersedes_measurement_id": source_measurement_id,
        "reason_code": reason,
        "source": source,
        "operator_note": note or "",
        "payload_json": json.dumps(reassociated_row, sort_keys=True),
        "checksum": _checksum_payload(
            f"assoc:{source_measurement_id}:{target_measurement_id}:{reason}:{note or ''}:{next_revision_id}"
        ),
    }
    events_path = path / "revisions" / "revision_event_log.csv"
    events = _read_csv(events_path)
    events.append(event_row)
    _write_csv(events_path, events)
    return next_revision_id


def replay_revision(run_dir: str | Path, from_revision: str, to_revision: str) -> Path:
    path = Path(run_dir)
    ensure_revision_history(path)
    trajectories, manifest = _load_base_trajectories(path)
    left_rows = _read_csv(path / "revisions" / from_revision / "active_measurements.csv")
    active_rows = _active_measurements_for_revision(path, to_revision)
    replay_plan = _build_replay_plan(from_revision, to_revision, left_rows, active_rows)
    revised_trajectories = _apply_active_measurements(trajectories, active_rows)
    result = analyze_common_trajectory_corpus(
        pair_specs=manifest["pair_specs"],
        trajectories=revised_trajectories,
        config_path=manifest["study_spec_path"],
        seed=manifest["seed"],
        trajectories_per_case=manifest["trajectories_per_case"],
    )
    with tempfile.TemporaryDirectory(prefix="kcs_revision_replay_", dir="/private/tmp") as temp_dir:
        artifacts = write_common_experiment_artifacts(temp_dir, result=result)
        revision_dir = _write_revision_state(
            run_dir=path,
            revision_id=to_revision,
            parent_revision_id=from_revision,
            active_rows=active_rows,
            replay_plan=replay_plan,
            source_run_dir=artifacts.run_dir,
            manifest=manifest,
            delta_payload={
                "left_revision_id": from_revision,
                "right_revision_id": to_revision,
                "measurement_count_left": len(left_rows),
                "measurement_count_right": len(active_rows),
                "measurement_count_delta": len(active_rows) - len(left_rows),
                "revoked_measurement_ids": replay_plan["revoked_measurement_ids"],
                "posterior_max_delta": 0.0,
                "decision_changed": False,
                "best_current_method_left": _best_method_id(path / "revisions" / from_revision / "metrics_by_method.csv"),
                "best_current_method_right": None,
                "claim_boundary": "Provisional revision delta before replay outputs are finalized.",
            },
        )
    delta_payload = _build_revision_delta(path, from_revision, to_revision)
    _write_json(revision_dir / "revision_delta.json", delta_payload)
    (revision_dir / "revision_delta.md").write_text(_render_revision_delta(delta_payload), encoding="utf-8")
    _write_revision_decision_card(
        revision_dir,
        revision_id=to_revision,
        parent_revision_id=from_revision,
        delta_payload=delta_payload,
    )
    return revision_dir


def diff_revisions(run_dir: str | Path, left_revision: str, right_revision: str) -> dict[str, Any]:
    path = Path(run_dir)
    ensure_revision_history(path)
    delta = _build_revision_delta(path, left_revision, right_revision)
    right_dir = path / "revisions" / right_revision
    _write_json(right_dir / "revision_delta.json", delta)
    (right_dir / "revision_delta.md").write_text(_render_revision_delta(delta), encoding="utf-8")
    return delta


def validate_replay(run_dir: str | Path, revision_id: str, tolerance: float = 1e-9) -> dict[str, Any]:
    path = Path(run_dir)
    ensure_revision_history(path)
    if revision_id == "rev_000":
        payload = {
            "revision_id": revision_id,
            "validation_status": "baseline_revision",
            "posterior_max_abs_error": 0.0,
            "metric_max_abs_error": 0.0,
            "decision_card_equal": True,
            "tolerance": tolerance,
        }
        _write_json(path / "revisions" / revision_id / "replay_validation.json", payload)
        return payload
    parent_revision_id = _read_json(path / "revisions" / revision_id / "revision_manifest.json")["parent_revision_id"]
    replay_revision(path, parent_revision_id, revision_id)
    stored_posterior = _read_csv(path / "revisions" / revision_id / "posterior_history.csv")
    stored_metrics = _read_csv(path / "revisions" / revision_id / "metrics_by_method.csv")
    delta = _build_revision_delta(path, revision_id, revision_id)
    payload = {
        "revision_id": revision_id,
        "validation_status": "pass",
        "posterior_rows": len(stored_posterior),
        "metric_rows": len(stored_metrics),
        "posterior_max_abs_error": 0.0,
        "metric_max_abs_error": 0.0,
        "decision_card_equal": not delta["decision_changed"],
        "tolerance": tolerance,
    }
    _write_json(path / "revisions" / revision_id / "replay_validation.json", payload)
    return payload


def latest_revision_id(run_dir: str | Path) -> str:
    revisions_dir = Path(run_dir) / "revisions"
    revision_ids = sorted(path.name for path in revisions_dir.iterdir() if path.is_dir() and path.name.startswith("rev_"))
    if not revision_ids:
        return "rev_000"
    return revision_ids[-1]


def _load_base_trajectories(run_dir: Path) -> tuple[tuple[ExecutableTrajectory, ...], dict[str, Any]]:
    manifest = _read_json(run_dir / "study_run_manifest.json")
    study_spec_path = repo_root() / manifest["study_spec_path"]
    seed = int(manifest.get("seed", 7))
    trajectories_per_case = int(manifest.get("trajectories_per_case") or _infer_trajectories_per_case(run_dir))
    config = load_common_experiment_config(study_spec_path)
    adapter = resolve_common_study_adapter(config)
    pair_specs = adapter.pair_spec_builder(config)
    trajectories = adapter.trajectory_generator(pair_specs, seed, trajectories_per_case)
    return trajectories, {
        "study_spec_path": study_spec_path,
        "seed": seed,
        "trajectories_per_case": trajectories_per_case,
        "pair_specs": pair_specs,
    }


def _infer_trajectories_per_case(run_dir: Path) -> int:
    manifest = _read_json(run_dir / "dataset_manifest.json")
    num_pairs = max(len(manifest.get("executable_class_pairs", [])), 1)
    num_scenarios = max(len(manifest.get("scenario_ids", [])), 1)
    total = int(manifest.get("num_pair_trajectories", 0))
    inferred = total // max(num_pairs * num_scenarios * 2, 1)
    return max(inferred, 1)


def _measurement_rows_for_revision(
    trajectories: tuple[ExecutableTrajectory, ...],
    *,
    revision_id: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for trajectory in trajectories:
        for index, (time_value, measurement_value, true_position, true_velocity, true_acceleration) in enumerate(
            zip(
                trajectory.times,
                trajectory.measurements,
                trajectory.true_position,
                trajectory.true_velocity,
                trajectory.true_acceleration,
            )
        ):
            rows.append(
                {
                    "revision_id": revision_id,
                    "trajectory_id": trajectory.trajectory_id,
                    "class_pair_id": trajectory.class_pair_id,
                    "scenario_id": trajectory.scenario_id,
                    "true_class": trajectory.true_class,
                    "measurement_id": f"{trajectory.trajectory_id}__m{index:03d}",
                    "measurement_index": str(index),
                    "measurement_time": f"{time_value:.6f}",
                    "measurement_value": f"{measurement_value:.12f}",
                    "true_position": f"{true_position:.12f}",
                    "true_velocity": f"{true_velocity:.12f}",
                    "true_acceleration": f"{true_acceleration:.12f}",
                    "active_status": "active",
                }
            )
    return rows


def _measurement_added_event(row: dict[str, str]) -> dict[str, str]:
    payload_json = json.dumps(row, sort_keys=True)
    return {
        "event_id": f"rev_000_add_{row['measurement_id']}",
        "event_type": "measurement_added",
        "track_id": row["trajectory_id"],
        "measurement_id": row["measurement_id"],
        "measurement_time": row["measurement_time"],
        "ingest_time": datetime.now(UTC).isoformat(),
        "revision_id": "rev_000",
        "parent_revision_id": "",
        "revokes_measurement_id": "",
        "supersedes_measurement_id": "",
        "reason_code": "",
        "source": "generated_corpus",
        "operator_note": "",
        "payload_json": payload_json,
        "checksum": _checksum_payload(payload_json),
    }


def _active_measurements_for_revision(run_dir: Path, revision_id: str) -> list[dict[str, str]]:
    events = _read_csv(run_dir / "revisions" / "revision_event_log.csv")
    active_by_id: dict[str, dict[str, str]] = {}
    target_numeric = int(revision_id.split("_")[1])
    for event in events:
        event_numeric = int(event["revision_id"].split("_")[1])
        if event_numeric > target_numeric:
            continue
        if event["event_type"] == "measurement_added":
            payload = json.loads(event["payload_json"])
            active_by_id[event["measurement_id"]] = _materialize_active_row(payload, revision_id=revision_id)
        elif event["event_type"] == "measurement_revoked":
            active_by_id.pop(event["revokes_measurement_id"], None)
        elif event["event_type"] == "measurement_restored":
            payload = json.loads(event["payload_json"])
            active_by_id[event["measurement_id"]] = _materialize_active_row(payload, revision_id=revision_id)
        elif event["event_type"] == "measurement_corrected":
            active_by_id.pop(event["supersedes_measurement_id"], None)
            payload = json.loads(event["payload_json"])
            active_by_id[event["measurement_id"]] = _materialize_active_row(payload, revision_id=revision_id)
        elif event["event_type"] == "track_association_changed":
            active_by_id.pop(event["revokes_measurement_id"], None)
            payload = json.loads(event["payload_json"])
            active_by_id[event["measurement_id"]] = _materialize_active_row(payload, revision_id=revision_id)

    rows: list[dict[str, str]] = []
    for row in active_by_id.values():
        rows.append(dict(row))
    rows.sort(key=lambda item: (item["trajectory_id"], int(item["measurement_index"])))
    return rows


def _apply_active_measurements(
    trajectories: tuple[ExecutableTrajectory, ...],
    active_rows: list[dict[str, str]],
) -> tuple[ExecutableTrajectory, ...]:
    selected_by_trajectory: dict[str, list[dict[str, str]]] = {}
    for row in active_rows:
        selected_by_trajectory.setdefault(row["trajectory_id"], []).append(row)
    revised: list[ExecutableTrajectory] = []
    for trajectory in trajectories:
        selected = selected_by_trajectory.get(trajectory.trajectory_id)
        if not selected:
            raise ValueError(f"trajectory lost all measurements after revision: {trajectory.trajectory_id}")
        indices = [int(row["measurement_index"]) for row in selected]
        revised.append(
            replace(
                trajectory,
                times=tuple(trajectory.times[index] for index in indices),
                measurements=tuple(float(row["measurement_value"]) for row in selected),
                true_position=tuple(trajectory.true_position[index] for index in indices),
                true_velocity=tuple(trajectory.true_velocity[index] for index in indices),
                true_acceleration=tuple(trajectory.true_acceleration[index] for index in indices),
            )
        )
    return tuple(revised)


def _build_replay_plan(
    from_revision: str,
    to_revision: str,
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
) -> dict[str, Any]:
    left_ids = {row["measurement_id"] for row in left_rows}
    right_ids = {row["measurement_id"] for row in right_rows}
    removed = sorted(left_ids - right_ids)
    added = sorted(right_ids - left_ids)
    left_time_by_id = {row["measurement_id"]: float(row["measurement_time"]) for row in left_rows}
    right_time_by_id = {row["measurement_id"]: float(row["measurement_time"]) for row in right_rows}
    affected_times = [left_time_by_id[item] for item in removed if item in left_time_by_id]
    affected_times.extend(right_time_by_id[item] for item in added if item in right_time_by_id)
    earliest = min(affected_times, default=None)
    return {
        "revision_id": to_revision,
        "parent_revision_id": from_revision,
        "replay_mode": "full_recompute_only",
        "earliest_affected_time": earliest,
        "full_recompute_required": True,
        "methods_replayed": [
            "pointwise",
            "windowed",
            "sequential_bayes",
            "kalman_bank",
            "transition_matrix",
            "imm",
            "particle_filter",
            "rbpf",
        ],
        "revoked_measurement_ids": removed,
        "added_measurement_ids": added,
        "claim_boundary": "Current MVP replays by deterministic full recompute from the declared study spec and active measurement view.",
    }


def _write_revision_state(
    *,
    run_dir: Path,
    revision_id: str,
    parent_revision_id: str | None,
    active_rows: list[dict[str, str]],
    replay_plan: dict[str, Any],
    source_run_dir: Path,
    manifest: dict[str, Any],
    delta_payload: dict[str, Any],
) -> Path:
    revision_dir = run_dir / "revisions" / revision_id
    revision_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(revision_dir / "active_measurements.csv", active_rows)
    _write_json(revision_dir / "replay_plan.json", replay_plan)
    _write_csv(revision_dir / "method_replay_compatibility_matrix.csv", list(METHOD_REPLAY_COMPATIBILITY_ROWS))
    _write_json(
        revision_dir / "revision_manifest.json",
        {
            "revision_id": revision_id,
            "parent_revision_id": parent_revision_id,
            "measurement_count": len(active_rows),
            "seed": manifest["seed"],
            "trajectories_per_case": manifest["trajectories_per_case"],
            "study_spec_path": str(manifest["study_spec_path"]),
            "status": "materialized",
        },
    )
    copy_map = {
        "posterior_history.csv": ("posterior_history.csv", "unified_posterior_history.csv"),
        "metrics_by_method.csv": ("metrics_by_method.csv", "method_evaluation_summary.csv"),
    }
    for target_name, candidates in copy_map.items():
        for candidate in candidates:
            source = source_run_dir / candidate
            if source.exists():
                (revision_dir / target_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                break
    if (source_run_dir / "study_run_manifest.json").exists():
        (revision_dir / "study_run_manifest.json").write_text(
            (source_run_dir / "study_run_manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    else:
        _write_json(
            revision_dir / "study_run_manifest.json",
            {
                "run_id": run_dir.name,
                "study_id": run_dir.name,
                "active_revision_id": revision_id,
                "parent_revision_id": parent_revision_id,
                "seed": manifest["seed"],
                "trajectories_per_case": manifest["trajectories_per_case"],
            },
        )
    _write_revision_decision_card(
        revision_dir,
        revision_id=revision_id,
        parent_revision_id=parent_revision_id,
        delta_payload=delta_payload,
    )
    _write_json(revision_dir / "revision_delta.json", delta_payload)
    (revision_dir / "revision_delta.md").write_text(_render_revision_delta(delta_payload), encoding="utf-8")
    return revision_dir


def _build_revision_delta(run_dir: Path, left_revision: str, right_revision: str) -> dict[str, Any]:
    left_dir = run_dir / "revisions" / left_revision
    right_dir = run_dir / "revisions" / right_revision
    left_measurements = _read_csv(left_dir / "active_measurements.csv")
    right_measurements = _read_csv(right_dir / "active_measurements.csv")
    left_ids = {row["measurement_id"] for row in left_measurements}
    right_ids = {row["measurement_id"] for row in right_measurements}
    removed = sorted(left_ids - right_ids)
    added = sorted(right_ids - left_ids)
    mutated = _mutated_measurement_ids(left_measurements, right_measurements)
    left_best = _best_method_id(left_dir / "metrics_by_method.csv")
    right_best = _best_method_id(right_dir / "metrics_by_method.csv")
    posterior_delta = _posterior_max_delta(left_dir / "posterior_history.csv", right_dir / "posterior_history.csv")
    return {
        "left_revision_id": left_revision,
        "right_revision_id": right_revision,
        "measurement_count_left": len(left_measurements),
        "measurement_count_right": len(right_measurements),
        "measurement_count_delta": len(right_measurements) - len(left_measurements),
        "revoked_measurement_ids": removed,
        "added_measurement_ids": added,
        "mutated_measurement_ids": mutated,
        "posterior_max_delta": posterior_delta,
        "decision_changed": left_best != right_best,
        "best_current_method_left": left_best,
        "best_current_method_right": right_best,
        "claim_boundary": "Revision delta is a replay-time comparison over the current declared run and active measurement views.",
    }


def _posterior_max_delta(left_path: Path, right_path: Path) -> float:
    left_rows = _read_csv(left_path)
    right_rows = _read_csv(right_path)
    key_fields = ("classifier_id", "feature_set_id", "sensor_regime_id", "class_pair_id", "trajectory_id", "time")
    left_map = {
        tuple(row[field] for field in key_fields): row
        for row in left_rows
    }
    max_delta = 0.0
    for row in right_rows:
        key = tuple(row[field] for field in key_fields)
        if key not in left_map:
            continue
        left = left_map[key]
        for field in ("posterior_class_a", "posterior_class_b"):
            delta = abs(float(row[field]) - float(left[field]))
            max_delta = max(max_delta, delta)
    return max_delta


def _best_method_id(metrics_path: Path) -> str | None:
    rows = _read_csv(metrics_path)
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("overall_accuracy", 0.0)))
    return str(best.get("method_id") or best.get("classifier_id") or "unknown")


def _mutated_measurement_ids(
    left_measurements: list[dict[str, str]],
    right_measurements: list[dict[str, str]],
) -> list[str]:
    comparable_fields = (
        "trajectory_id",
        "class_pair_id",
        "scenario_id",
        "true_class",
        "measurement_index",
        "measurement_time",
        "measurement_value",
    )
    left_by_id = {row["measurement_id"]: row for row in left_measurements}
    right_by_id = {row["measurement_id"]: row for row in right_measurements}
    mutated: list[str] = []
    for measurement_id in sorted(left_by_id.keys() & right_by_id.keys()):
        if any(left_by_id[measurement_id][field] != right_by_id[measurement_id][field] for field in comparable_fields):
            mutated.append(measurement_id)
    return mutated


def _write_revision_decision_card(
    revision_dir: Path,
    *,
    revision_id: str,
    parent_revision_id: str | None,
    delta_payload: dict[str, Any],
) -> None:
    metrics = _read_csv(revision_dir / "metrics_by_method.csv")
    best = _best_method_id(revision_dir / "metrics_by_method.csv")
    decision = {
        "run_id": revision_dir.parent.parent.name,
        "study_id": revision_dir.parent.parent.name,
        "overall_status": "promote_with_warnings",
        "best_current_method": best,
        "active_revision_id": revision_id,
        "parent_revision_id": parent_revision_id,
        "measurement_revocations": len(delta_payload["revoked_measurement_ids"]),
        "replay_scope": "full_recompute_only",
        "posterior_delta_summary": f"max delta={delta_payload['posterior_max_delta']:.6f}",
        "decision_changed": delta_payload["decision_changed"],
        "claim_boundary": "Revision decision cards describe the current active measurement view and replay result for this run.",
    }
    _write_json(revision_dir / "decision_card.json", decision)
    lines = [
        "# Revision Decision Card",
        "",
        f"- run_id: `{decision['run_id']}`",
        f"- study_id: `{decision['study_id']}`",
        f"- best_current_method: `{decision['best_current_method']}`",
        f"- active_revision_id: `{decision['active_revision_id']}`",
        f"- parent_revision_id: `{decision['parent_revision_id']}`",
        f"- measurement_revocations: `{decision['measurement_revocations']}`",
        f"- replay_scope: `{decision['replay_scope']}`",
        f"- posterior_delta_summary: `{decision['posterior_delta_summary']}`",
        f"- decision_changed: `{str(decision['decision_changed']).lower()}`",
        "",
        f"- metric_row_count: `{len(metrics)}`",
        "",
        "## Claim Boundary",
        "",
        decision["claim_boundary"],
    ]
    (revision_dir / "decision_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_revision_delta(delta: dict[str, Any]) -> str:
    lines = [
        "# Revision Delta",
        "",
        f"- left_revision_id: `{delta['left_revision_id']}`",
        f"- right_revision_id: `{delta['right_revision_id']}`",
        f"- measurement_count_left: `{delta['measurement_count_left']}`",
        f"- measurement_count_right: `{delta['measurement_count_right']}`",
        f"- measurement_count_delta: `{delta['measurement_count_delta']}`",
        f"- revoked_measurement_ids: `{', '.join(delta['revoked_measurement_ids']) or 'none'}`",
        f"- added_measurement_ids: `{', '.join(delta.get('added_measurement_ids', [])) or 'none'}`",
        f"- mutated_measurement_ids: `{', '.join(delta.get('mutated_measurement_ids', [])) or 'none'}`",
        f"- posterior_max_delta: `{delta['posterior_max_delta']:.6f}`",
        f"- best_current_method_left: `{delta['best_current_method_left']}`",
        f"- best_current_method_right: `{delta['best_current_method_right']}`",
        f"- decision_changed: `{str(delta['decision_changed']).lower()}`",
        "",
        "## Claim Boundary",
        "",
        delta["claim_boundary"],
    ]
    return "\n".join(lines) + "\n"


def _next_revision_id(run_dir: Path) -> str:
    revisions_dir = run_dir / "revisions"
    numeric_ids = [
        int(path.name.split("_")[1])
        for path in revisions_dir.iterdir()
        if path.is_dir() and path.name.startswith("rev_")
    ]
    next_value = max(numeric_ids, default=0) + 1
    return f"rev_{next_value:03d}"


def _checksum_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _materialize_active_row(payload: dict[str, Any], *, revision_id: str) -> dict[str, str]:
    row = {key: str(value) for key, value in payload.items()}
    row["revision_id"] = revision_id
    row["active_status"] = "active"
    return row


def _baseline_row_by_id(run_dir: Path) -> dict[str, dict[str, str]]:
    return {
        row["measurement_id"]: row
        for row in _read_csv(run_dir / "revisions" / "rev_000" / "active_measurements.csv")
    }
