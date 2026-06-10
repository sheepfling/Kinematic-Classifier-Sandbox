from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .runtime import runtime_root

T = TypeVar("T")


def analysis_cache_root() -> Path:
    path = runtime_root() / "analysis_cache"
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
    if not enabled:
        return compute()

    cache_dir = analysis_cache_root() / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = cache_dir / f"{cache_key}.pkl"
    metadata_path = cache_dir / f"{cache_key}.json"

    if pickle_path.exists():
        try:
            with pickle_path.open("rb") as handle:
                return pickle.load(handle)
        except (pickle.PickleError, EOFError, AttributeError, ValueError):
            pass

    result = compute()
    tmp_path = pickle_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(pickle_path)

    payload = {
        "namespace": namespace,
        "cache_key": cache_key,
    }
    if metadata:
        payload["metadata"] = _normalize(metadata)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result


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

