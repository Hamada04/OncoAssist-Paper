"""Generic, deterministic research-artifact integrity primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Callable, Mapping, Sequence
import uuid

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is optional for this module.
    np = None


RUN_LOCK_SCHEMA_VERSION = "research-run-lock-v1"


def jsonable(payload: Any) -> Any:
    """Normalize supported values before strict deterministic JSON encoding."""
    if isinstance(payload, Path):
        return str(payload)
    if np is not None and isinstance(payload, np.ndarray):
        return jsonable(payload.tolist())
    if np is not None and isinstance(payload, np.generic):
        return jsonable(payload.item())
    if isinstance(payload, Mapping):
        return {str(key): jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [jsonable(value) for value in payload]
    return payload


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    """Return deterministic UTF-8 JSON bytes, rejecting NaN and infinity."""
    options: dict[str, Any] = {
        "ensure_ascii": True,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(jsonable(payload), **options) + "\n").encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> str:
    """Atomically publish replaceable JSON state and return its byte hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = canonical_json_bytes(payload)
    temporary_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    replaced = False
    try:
        with temporary_path.open("xb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        replaced = True
    finally:
        if not replaced:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(contents).hexdigest()


def create_immutable_json(path: Path, payload: Any) -> str:
    """Create JSON evidence once; existing evidence is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(contents).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON object does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON object is malformed: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_json_record_stream(
    path: Path,
    *,
    expected_record_count: int | None = None,
    unique_key: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Read compact JSONL or legacy concatenated pretty JSON object records."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON record stream does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    index = 0
    records: list[dict[str, Any]] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text):
            break
        try:
            record, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON record stream is malformed: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSON record stream contains a non-object record: {path}")
        records.append(record)
    if not records:
        raise ValueError(f"JSON record stream contains no records: {path}")
    if expected_record_count is not None and len(records) != expected_record_count:
        raise ValueError(
            f"JSON record stream count {len(records)} does not match expected {expected_record_count}."
        )
    if unique_key is not None:
        seen: set[Any] = set()
        for record in records:
            key = unique_key(record)
            if key is None:
                raise ValueError("JSON record stream uniqueness key must not be None.")
            try:
                duplicate = key in seen
            except TypeError as error:
                raise ValueError("JSON record stream uniqueness key must be hashable.") from error
            if duplicate:
                raise ValueError(f"JSON record stream contains duplicate unique key: {key!r}")
            seen.add(key)
    return records


def write_compact_jsonl(
    path: Path, records: Sequence[Mapping[str, Any]], *, allow_empty: bool = False
) -> None:
    """Write one compact JSON object per physical line and fsync before return."""
    if not records and not allow_empty:
        raise ValueError("JSONL writing requires at least one record.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("JSONL records must be JSON objects.")
            line = json.dumps(
                jsonable(record),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def publish_directory(target: Path, writer: Callable[[Path], None]) -> Path:
    """Publish a newly created directory with one same-parent rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Publication target already exists: {target}")
    temporary = target.parent / f".{target.name}.publishing-{uuid.uuid4().hex}"
    temporary.mkdir()
    writer(temporary)
    os.rename(temporary, target)
    return target


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _lock_path(output_directory: Path) -> Path:
    return output_directory / ".run_lock.json"


def _lifecycle_path(output_directory: Path, event: str, lock_id: str) -> Path:
    return output_directory / "lock_lifecycle" / event / f"{lock_id}.json"


def _pid_is_active(owner: Mapping[str, Any]) -> bool:
    hostname = owner.get("hostname")
    pid = owner.get("pid")
    if hostname != socket.gethostname():
        raise RuntimeError("A lock owned by another hostname cannot be declared stale locally.")
    if type(pid) is not int or pid <= 0:
        raise ValueError("Lock owner PID is invalid; stale recovery is unsafe.")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        raise RuntimeError("Lock owner liveness could not be determined safely.") from error
    return True


def _recover_stale_lock(output_directory: Path, expected_binding: Mapping[str, Any]) -> None:
    lock_path = _lock_path(output_directory)
    guard_path = output_directory / ".run_lock_recovery_guard.json"
    guard_payload = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "acquired_at_utc": _utc_now(),
    }
    create_immutable_json(guard_path, guard_payload)
    try:
        active = read_json_object(lock_path)
        if active.get("binding") != dict(expected_binding):
            raise ValueError("Stale lock binding does not match requested binding.")
        owner = active.get("owner")
        if not isinstance(owner, dict):
            raise ValueError("Stale lock owner information is invalid; recovery is unsafe.")
        if _pid_is_active(owner):
            raise RuntimeError("Existing local lock owner is active and cannot be recovered.")
        lock_id = active.get("lock_id")
        if not isinstance(lock_id, str) or not lock_id:
            raise ValueError("Stale lock identity is invalid; recovery is unsafe.")
        recovered_path = _lifecycle_path(output_directory, "recovered", lock_id)
        recovered_path.parent.mkdir(parents=True, exist_ok=True)
        if recovered_path.exists():
            raise FileExistsError(f"Recovered-lock evidence already exists: {recovered_path}")
        os.rename(lock_path, recovered_path)
    finally:
        try:
            guard_path.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class RunLock:
    path: Path
    lock_id: str
    binding: dict[str, Any]


def acquire_run_lock(output_directory: Path, binding: Mapping[str, Any]) -> RunLock:
    """Acquire an opaque-binding local lock, recovering only provably stale local locks."""
    if not isinstance(binding, Mapping):
        raise ValueError("Run-lock binding must be a JSON object.")
    normalized_binding = jsonable(dict(binding))
    canonical_json_bytes(normalized_binding)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = _lock_path(output_directory)
    for attempt in range(2):
        lock_id = uuid.uuid4().hex
        owner = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "process_identity": f"{socket.gethostname()}:{os.getpid()}",
        }
        payload = {
            "schema_version": RUN_LOCK_SCHEMA_VERSION,
            "lock_id": lock_id,
            "binding": normalized_binding,
            "owner": owner,
            "lifecycle": {"acquired_at_utc": _utc_now()},
        }
        try:
            create_immutable_json(path, payload)
        except FileExistsError:
            if attempt:
                raise FileExistsError(f"Run lock is already held: {path}")
            _recover_stale_lock(output_directory, normalized_binding)
            continue
        atomic_write_json(
            _lifecycle_path(output_directory, "acquired", lock_id),
            {"lock_id": lock_id, "binding": normalized_binding, "owner": owner, "acquired_at_utc": _utc_now()},
        )
        return RunLock(path=path, lock_id=lock_id, binding=normalized_binding)
    raise RuntimeError("Run lock acquisition did not complete.")


def recover_abandoned_run_lock(
    output_directory: Path,
    expected_binding: Mapping[str, Any],
    expected_lock_id: str,
) -> None:
    """Explicitly archive a remote/disconnected lock after operator confirmation.

    Unlike local stale recovery, this function never tries to infer whether a
    remote owner is alive. Calling it is the caller's explicit confirmation
    that the prior run is abandoned.
    """
    if not isinstance(expected_binding, Mapping):
        raise ValueError("Abandoned-lock recovery requires a JSON-object binding.")
    if not isinstance(expected_lock_id, str) or not expected_lock_id:
        raise ValueError("Abandoned-lock recovery requires the prior lock ID.")
    lock_path = _lock_path(output_directory)
    active = read_json_object(lock_path)
    normalized_binding = jsonable(dict(expected_binding))
    if active.get("binding") != normalized_binding or active.get("lock_id") != expected_lock_id:
        raise ValueError("Abandoned-lock recovery binding or lock ID does not match the active lock.")
    intent_path = _lifecycle_path(output_directory, "abandon_requested", expected_lock_id)
    create_immutable_json(
        intent_path,
        {
            "lock_id": expected_lock_id,
            "binding": normalized_binding,
            "requested_at_utc": _utc_now(),
            "requesting_owner": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "process_identity": f"{socket.gethostname()}:{os.getpid()}",
            },
        },
    )
    archived_path = _lifecycle_path(output_directory, "abandoned", expected_lock_id)
    archived_path.parent.mkdir(parents=True, exist_ok=True)
    if archived_path.exists():
        raise FileExistsError(f"Abandoned-lock evidence already exists: {archived_path}")
    os.rename(lock_path, archived_path)


def release_run_lock(lock: RunLock, *, outcome: str) -> bool:
    """Release only the lock held by this lock identity and retain evidence."""
    if not lock.path.exists():
        return False
    active = read_json_object(lock.path)
    if active.get("lock_id") != lock.lock_id:
        raise PermissionError("Run-lock ownership does not match the active lock.")
    atomic_write_json(
        _lifecycle_path(lock.path.parent, "released", lock.lock_id),
        {
            "lock_id": lock.lock_id,
            "binding": lock.binding,
            "outcome": outcome,
            "released_at_utc": _utc_now(),
        },
    )
    lock.path.unlink()
    return True
