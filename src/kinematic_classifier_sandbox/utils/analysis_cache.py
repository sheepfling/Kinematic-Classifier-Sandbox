from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .runtime import runtime_root

T = TypeVar("T")
_CACHE_METADATA_VERSION = 1
_VALID_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_CACHE_STATS: dict[str, dict[str, int]] = {}


def analysis_cache_root() -> Path:
    path = runtime_root() / "analysis_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_cache_namespace_root(namespace: str) -> Path:
    validated = _validate_namespace(namespace)
    path = analysis_cache_root() / validated
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_fingerprint(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def stable_cache_key(namespace: str, payload: dict[str, object]) -> str:
    _validate_namespace(namespace)
    normalized = _normalize(payload)
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{namespace}_{digest}"


def load_or_compute_pickled(
    *,
    namespace: str,
    cache_key: str,
    compute: Callable[[], T],
    enabled: bool = True,
    metadata: dict[str, object] | None = None,
) -> T:
    _validate_namespace(namespace)
    if not enabled:
        _record_cache_event(namespace, "disabled")
        return compute()

    cache_dir = analysis_cache_namespace_root(namespace)
    pickle_path = cache_dir / f"{cache_key}.pkl"
    metadata_path = cache_dir / f"{cache_key}.json"

    if pickle_path.exists():
        try:
            with pickle_path.open("rb") as handle:
                cached = pickle.load(handle)
            _record_cache_event(namespace, "hit")
            return cached
        except (pickle.PickleError, EOFError, AttributeError, ValueError):
            _record_cache_event(namespace, "corrupt")
            pickle_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)

    _record_cache_event(namespace, "miss")
    result = compute()
    tmp_path = pickle_path.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(pickle_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    payload = {
        "metadata_version": _CACHE_METADATA_VERSION,
        "namespace": namespace,
        "cache_key": cache_key,
    }
    if metadata:
        payload["metadata"] = _normalize(metadata)
    tmp_metadata_path = metadata_path.with_suffix(".json.tmp")
    try:
        tmp_metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_metadata_path.replace(metadata_path)
    finally:
        tmp_metadata_path.unlink(missing_ok=True)
    return result


def reset_analysis_cache_stats(*, namespace: str | None = None) -> None:
    if namespace is None:
        _CACHE_STATS.clear()
        return
    _CACHE_STATS.pop(_validate_namespace(namespace), None)


def describe_analysis_cache_stats(*, namespace: str | None = None) -> dict[str, object]:
    selected = (
        {namespace: _CACHE_STATS.get(_validate_namespace(namespace), {})}
        if namespace is not None
        else dict(sorted(_CACHE_STATS.items()))
    )
    rows: list[dict[str, object]] = []
    totals = {"hit": 0, "miss": 0, "corrupt": 0, "disabled": 0}
    for name, stats in selected.items():
        row = {
            "namespace": name,
            "hit": int(stats.get("hit", 0)),
            "miss": int(stats.get("miss", 0)),
            "corrupt": int(stats.get("corrupt", 0)),
            "disabled": int(stats.get("disabled", 0)),
        }
        rows.append(row)
        for key in totals:
            totals[key] += int(row[key])
    return {
        "namespace_count": len(rows),
        "hit": totals["hit"],
        "miss": totals["miss"],
        "corrupt": totals["corrupt"],
        "disabled": totals["disabled"],
        "namespaces": rows,
    }


def describe_analysis_cache(*, namespace: str | None = None) -> dict[str, object]:
    root = analysis_cache_root()
    namespaces = [analysis_cache_namespace_root(namespace)] if namespace is not None else sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    rows: list[dict[str, object]] = []
    total_bytes = 0
    total_entries = 0
    for namespace_path in namespaces:
        pickle_paths = sorted(namespace_path.glob("*.pkl"))
        metadata_paths = sorted(namespace_path.glob("*.json"))
        namespace_bytes = sum(path.stat().st_size for path in [*pickle_paths, *metadata_paths] if path.exists())
        entry = {
            "namespace": namespace_path.name,
            "entry_count": len(pickle_paths),
            "metadata_count": len(metadata_paths),
            "bytes": namespace_bytes,
            "path": str(namespace_path),
        }
        rows.append(entry)
        total_bytes += namespace_bytes
        total_entries += len(pickle_paths)
    return {
        "root": str(root),
        "namespace_count": len(rows),
        "entry_count": total_entries,
        "bytes": total_bytes,
        "namespaces": rows,
    }


def clear_analysis_cache(*, namespace: str | None = None) -> dict[str, object]:
    root = analysis_cache_root()
    target = analysis_cache_namespace_root(namespace) if namespace is not None else root
    before = describe_analysis_cache(namespace=namespace)
    if target.exists():
        shutil.rmtree(target)
    return {
        "cleared_namespace": namespace,
        "cleared_path": str(target),
        "cleared_entry_count": before["entry_count"],
        "cleared_bytes": before["bytes"],
    }


def _validate_namespace(namespace: str) -> str:
    if not namespace or not _VALID_NAMESPACE_PATTERN.fullmatch(namespace):
        raise ValueError(f"invalid cache namespace: {namespace!r}")
    return namespace


def _record_cache_event(namespace: str, event: str) -> None:
    stats = _CACHE_STATS.setdefault(_validate_namespace(namespace), {})
    stats[event] = stats.get(event, 0) + 1


def _normalize(value: object) -> object:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item) for item in value)
    return value
