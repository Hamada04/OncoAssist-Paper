import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from oncoassist_research import artifacts


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_canonical_json_and_hash_are_order_independent(self) -> None:
        first = {"b": [2, 1], "a": {"x": 1}}
        second = {"a": {"x": 1}, "b": [2, 1]}
        self.assertEqual(artifacts.canonical_json_bytes(first), artifacts.canonical_json_bytes(second))
        self.assertEqual(artifacts.payload_sha256(first), artifacts.payload_sha256(second))
        self.assertNotEqual(artifacts.payload_sha256(first), artifacts.payload_sha256({"a": 2}))

    def test_canonical_json_supports_paths_tuples_and_numpy(self) -> None:
        payload = {"path": Path("result.json"), "tuple": (1, 2), "scalar": np.int64(3), "array": np.array([4, 5])}
        decoded = json.loads(artifacts.canonical_json_bytes(payload))
        self.assertEqual(decoded, {"array": [4, 5], "path": "result.json", "scalar": 3, "tuple": [1, 2]})

    def test_canonical_json_rejects_nan_and_infinity(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    artifacts.canonical_json_bytes({"value": value})

    def test_atomic_json_write_is_valid_replaceable_and_hashed(self) -> None:
        path = self.root / "nested" / "state.json"
        first_hash = artifacts.atomic_write_json(path, {"value": 1})
        self.assertEqual(first_hash, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(artifacts.read_json_object(path), {"value": 1})
        artifacts.atomic_write_json(path, {"value": 2})
        self.assertEqual(artifacts.read_json_object(path), {"value": 2})
        self.assertEqual(list(path.parent.glob("*.tmp-*")), [])

    def test_immutable_json_is_create_once(self) -> None:
        path = self.root / "evidence.json"
        artifacts.create_immutable_json(path, {"first": True})
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            artifacts.create_immutable_json(path, {"first": False})
        self.assertEqual(path.read_bytes(), original)

    def test_strict_json_reader_rejects_missing_malformed_and_non_object(self) -> None:
        with self.assertRaises(FileNotFoundError):
            artifacts.read_json_object(self.root / "missing.json")
        malformed = self.root / "malformed.json"
        malformed.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            artifacts.read_json_object(malformed)
        scalar = self.root / "scalar.json"
        scalar.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            artifacts.read_json_object(scalar)

    def test_compact_jsonl_writer_and_reader_contract(self) -> None:
        path = self.root / "records.jsonl"
        records = [{"b": 2, "a": 1}, {"b": 4, "a": 3}]
        artifacts.write_compact_jsonl(path, records)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        self.assertEqual([json.loads(line) for line in lines], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        self.assertEqual(artifacts.read_json_record_stream(path), [{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    def test_record_stream_supports_legacy_and_rejects_invalid_cases(self) -> None:
        legacy = self.root / "legacy.jsonl"
        legacy.write_text('{\n  "id": 1\n}\n{\n  "id": 2\n}\n', encoding="utf-8")
        self.assertEqual(artifacts.read_json_record_stream(legacy), [{"id": 1}, {"id": 2}])
        with self.assertRaises(ValueError):
            artifacts.read_json_record_stream(legacy, expected_record_count=1)
        with self.assertRaises(ValueError):
            artifacts.read_json_record_stream(legacy, unique_key=lambda record: 1)
        malformed = self.root / "broken.jsonl"
        malformed.write_text('{"id": 1}\n{', encoding="utf-8")
        with self.assertRaises(ValueError):
            artifacts.read_json_record_stream(malformed)
        with self.assertRaises(ValueError):
            artifacts.write_compact_jsonl(self.root / "empty.jsonl", [])

    def test_atomic_directory_publication(self) -> None:
        target = self.root / "published"
        result = artifacts.publish_directory(target, lambda temporary: (temporary / "evidence.txt").write_text("ok", encoding="utf-8"))
        self.assertEqual(result, target)
        self.assertEqual((target / "evidence.txt").read_text(encoding="utf-8"), "ok")
        with self.assertRaises(FileExistsError):
            artifacts.publish_directory(target, lambda temporary: None)

    def test_directory_writer_failure_preserves_temporary_evidence_without_target(self) -> None:
        target = self.root / "failed"
        def fail_writer(temporary: Path) -> None:
            (temporary / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("writer failed")
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            artifacts.publish_directory(target, fail_writer)
        self.assertFalse(target.exists())
        temporary = list(self.root.glob(".failed.publishing-*"))
        self.assertEqual(len(temporary), 1)
        self.assertTrue((temporary[0] / "partial.txt").is_file())

    def test_run_lock_acquire_release_and_active_lock_rejection(self) -> None:
        binding = {"study": "synthetic", "run": 1}
        lock = artifacts.acquire_run_lock(self.root, binding)
        active = artifacts.read_json_object(lock.path)
        self.assertEqual(active["lock_id"], lock.lock_id)
        self.assertEqual(active["binding"], binding)
        self.assertIn("hostname", active["owner"])
        with self.assertRaises(RuntimeError):
            artifacts.acquire_run_lock(self.root, binding)
        self.assertTrue(artifacts.release_run_lock(lock, outcome="completed"))
        self.assertFalse(lock.path.exists())
        release = artifacts.read_json_object(self.root / "lock_lifecycle" / "released" / f"{lock.lock_id}.json")
        self.assertEqual(release["outcome"], "completed")

    def test_run_lock_wrong_owner_release_is_rejected(self) -> None:
        lock = artifacts.acquire_run_lock(self.root, {"run": 1})
        wrong = artifacts.RunLock(lock.path, "wrong", lock.binding)
        with self.assertRaises(PermissionError):
            artifacts.release_run_lock(wrong, outcome="failed")
        artifacts.release_run_lock(lock, outcome="completed")

    def _write_stale_lock(self, binding: dict, owner: dict) -> Path:
        path = self.root / ".run_lock.json"
        artifacts.atomic_write_json(path, {"schema_version": artifacts.RUN_LOCK_SCHEMA_VERSION, "lock_id": "stale-id", "binding": binding, "owner": owner, "lifecycle": {"acquired_at_utc": "2026-01-01T00:00:00Z"}})
        return path

    def test_stale_lock_recovery_preserves_evidence(self) -> None:
        binding = {"run": 1}
        self._write_stale_lock(binding, {"hostname": "local", "pid": 123, "process_identity": "local:123"})
        with patch("oncoassist_research.artifacts._pid_is_active", return_value=False):
            lock = artifacts.acquire_run_lock(self.root, binding)
        self.assertTrue((self.root / "lock_lifecycle" / "recovered" / "stale-id.json").is_file())
        artifacts.release_run_lock(lock, outcome="completed")

    def test_stale_recovery_fails_safely_for_binding_host_and_pid(self) -> None:
        self._write_stale_lock({"run": 2}, {"hostname": "local", "pid": 123, "process_identity": "local:123"})
        with patch("oncoassist_research.artifacts._pid_is_active", return_value=False):
            with self.assertRaises(ValueError):
                artifacts.acquire_run_lock(self.root, {"run": 1})
        os.remove(self.root / ".run_lock.json")
        self._write_stale_lock({"run": 1}, {"hostname": "remote", "pid": 123, "process_identity": "remote:123"})
        with self.assertRaises(RuntimeError):
            artifacts.acquire_run_lock(self.root, {"run": 1})
        os.remove(self.root / ".run_lock.json")
        local_hostname = artifacts.socket.gethostname()
        self._write_stale_lock({"run": 1}, {"hostname": local_hostname, "pid": "bad", "process_identity": f"{local_hostname}:bad"})
        with self.assertRaises(ValueError):
            artifacts.acquire_run_lock(self.root, {"run": 1})

    def test_explicit_abandoned_remote_lock_recovery_requires_exact_identity(self) -> None:
        binding = {"study": "remote"}
        self._write_stale_lock(binding, {"hostname": "remote", "pid": 123, "process_identity": "remote:123"})
        with self.assertRaises(RuntimeError):
            artifacts.acquire_run_lock(self.root, binding)
        with self.assertRaises(ValueError):
            artifacts.recover_abandoned_run_lock(self.root, binding, "wrong")
        artifacts.recover_abandoned_run_lock(self.root, binding, "stale-id")
        self.assertTrue((self.root / "lock_lifecycle" / "abandon_requested" / "stale-id.json").is_file())
        self.assertTrue((self.root / "lock_lifecycle" / "abandoned" / "stale-id.json").is_file())
        lock = artifacts.acquire_run_lock(self.root, binding)
        artifacts.release_run_lock(lock, outcome="completed")


if __name__ == "__main__":
    unittest.main()
