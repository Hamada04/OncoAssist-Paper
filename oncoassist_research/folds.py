"""Deterministic patient-level cross-validation manifest infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


OUTER_MANIFEST_SCHEMA_VERSION = "research-outer-repeated-stratified-kfold-v1"
INNER_MANIFEST_SCHEMA_VERSION = "research-nested-inner-stratified-kfold-v1"


@dataclass(frozen=True)
class FoldProtocol:
    """Explicit deterministic cross-validation settings for a research run."""

    outer_n_splits: int
    outer_n_repeats: int
    outer_random_state: int
    inner_n_splits: int
    inner_random_state_base: int

    def __post_init__(self) -> None:
        for name in (
            "outer_n_splits",
            "outer_n_repeats",
            "outer_random_state",
            "inner_n_splits",
            "inner_random_state_base",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer and not a boolean.")
        if self.outer_n_splits < 2:
            raise ValueError("outer_n_splits must be at least 2.")
        if self.outer_n_repeats < 1:
            raise ValueError("outer_n_repeats must be at least 1.")
        if self.inner_n_splits < 2:
            raise ValueError("inner_n_splits must be at least 2.")

    def as_dict(self) -> dict[str, int]:
        return {
            "outer_n_splits": self.outer_n_splits,
            "outer_n_repeats": self.outer_n_repeats,
            "outer_random_state": self.outer_random_state,
            "inner_n_splits": self.inner_n_splits,
            "inner_random_state_base": self.inner_random_state_base,
        }


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def manifest_identity_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical content identity for a validated fold manifest."""
    if not isinstance(manifest, Mapping):
        raise ValueError("Fold manifest identity requires a JSON object mapping.")
    return hashlib.sha256(_canonical_json_bytes(dict(manifest))).hexdigest()


def _sample_id_list_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(sample_ids))).hexdigest()


def _class_counts(labels: Sequence[int] | np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    observed = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return {"0": observed.get("0", 0), "1": observed.get("1", 0)}


def _label_by_sample_id(
    sample_ids: Sequence[str], y_binary: Sequence[int] | np.ndarray
) -> dict[str, int]:
    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    labels = np.asarray(y_binary, dtype=int)
    if normalized_ids != sorted(normalized_ids):
        raise ValueError("Fold generation requires lexicographically ordered SAMPLE_ID values.")
    if len(normalized_ids) != len(labels):
        raise ValueError("SAMPLE_ID and binary-label counts differ for fold generation.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Fold generation requires unique SAMPLE_ID values.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Fold generation requires binary labels 0 and 1.")
    return dict(zip(normalized_ids, labels.tolist()))


def _partition_summary(
    sample_ids: Sequence[str], label_by_sample_id: Mapping[str, int]
) -> dict[str, Any]:
    labels = np.asarray([label_by_sample_id[sample_id] for sample_id in sample_ids])
    return {"sample_count": len(sample_ids), "class_counts": _class_counts(labels)}


def _outer_manifest_protocol(protocol: FoldProtocol) -> dict[str, Any]:
    return {
        "splitter": "RepeatedStratifiedKFold",
        **protocol.as_dict(),
    }


def _inner_manifest_protocol(protocol: FoldProtocol) -> dict[str, Any]:
    return {
        "splitter": "StratifiedKFold",
        "shuffle": True,
        **protocol.as_dict(),
        "random_state_derivation": "inner_random_state_base + repeat_id * 100 + fold_id",
    }


def _build_outer_fold_records(
    sample_ids: Sequence[str], y_binary: Sequence[int] | np.ndarray, protocol: FoldProtocol
) -> list[dict[str, Any]]:
    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    labels = np.asarray(y_binary, dtype=int)
    label_by_sample_id = _label_by_sample_id(normalized_ids, labels)
    splitter = RepeatedStratifiedKFold(
        n_splits=protocol.outer_n_splits,
        n_repeats=protocol.outer_n_repeats,
        random_state=protocol.outer_random_state,
    )
    folds: list[dict[str, Any]] = []
    for split_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        train_ids = sorted(normalized_ids[index] for index in train_indices)
        test_ids = sorted(normalized_ids[index] for index in test_indices)
        folds.append(
            {
                "repeat_id": split_index // protocol.outer_n_splits,
                "fold_id": split_index % protocol.outer_n_splits,
                "train_sample_ids": train_ids,
                "test_sample_ids": test_ids,
                "train_partition": _partition_summary(train_ids, label_by_sample_id),
                "test_partition": _partition_summary(test_ids, label_by_sample_id),
            }
        )
    return folds


def build_outer_data_fingerprint(data: Mapping[str, Any]) -> dict[str, Any]:
    """Bind manifests to Step 1 audit metadata without reading CSV files."""
    sample_ids = [str(sample_id) for sample_id in data["sample_ids"]]
    files = data["audit_summary"]["files"]
    return {
        "csv_sha256": {
            modality: files[modality]["sha256"] for modality in ("mGE", "mDM", "CNA")
        },
        "sample_count": len(sample_ids),
        "label_mapping": json.loads(
            _canonical_json_bytes(data["label_mapping"]).decode("utf-8")
        ),
        "ordered_sample_ids_canonical_json_sha256": _sample_id_list_sha256(sample_ids),
    }


def build_outer_fold_manifest(
    sample_ids: Sequence[str],
    y_binary: Sequence[int] | np.ndarray,
    data_fingerprint: Mapping[str, Any],
    protocol: FoldProtocol,
) -> dict[str, Any]:
    return {
        "schema_version": OUTER_MANIFEST_SCHEMA_VERSION,
        "protocol": _outer_manifest_protocol(protocol),
        "data_fingerprint": dict(data_fingerprint),
        "folds": _build_outer_fold_records(sample_ids, y_binary, protocol),
    }


def _validate_partition(
    pair: tuple[int, int],
    record: Mapping[str, Any],
    label_by_sample_id: Mapping[str, int],
    current_ids: set[str],
) -> tuple[list[str], list[str]]:
    train_ids = record.get("train_sample_ids")
    test_ids = record.get("test_sample_ids")
    if not isinstance(train_ids, list) or not isinstance(test_ids, list):
        raise ValueError(f"Outer-fold manifest {pair} must contain train and test SAMPLE_ID lists.")
    if train_ids != sorted(train_ids) or test_ids != sorted(test_ids):
        raise ValueError(f"Outer-fold manifest {pair} SAMPLE_ID lists must be sorted.")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError(f"Outer-fold manifest {pair} contains duplicate IDs within a partition.")
    train_set, test_set = set(train_ids), set(test_ids)
    if not train_set.issubset(current_ids) or not test_set.issubset(current_ids):
        raise ValueError(f"Outer-fold manifest {pair} contains IDs absent from current data.")
    if train_set.intersection(test_set):
        raise ValueError(f"Outer-fold manifest {pair} has train/test SAMPLE_ID overlap.")
    if train_set.union(test_set) != current_ids:
        raise ValueError(f"Outer-fold manifest {pair} does not partition all current SAMPLE_ID values.")
    expected_train = _partition_summary(train_ids, label_by_sample_id)
    expected_test = _partition_summary(test_ids, label_by_sample_id)
    if record.get("train_partition") != expected_train:
        raise ValueError(f"Outer-fold manifest {pair} train counts do not match current labels.")
    if record.get("test_partition") != expected_test:
        raise ValueError(f"Outer-fold manifest {pair} test counts do not match current labels.")
    if 0 in expected_train["class_counts"].values() or 0 in expected_test["class_counts"].values():
        raise ValueError(f"Outer-fold manifest {pair} does not contain both classes in each partition.")
    return train_ids, test_ids


def validate_outer_fold_manifest(
    manifest: Mapping[str, Any],
    sample_ids: Sequence[str],
    y_binary: Sequence[int] | np.ndarray,
    data_fingerprint: Mapping[str, Any],
    protocol: FoldProtocol,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("Outer-fold manifest must be a JSON object.")
    if manifest.get("schema_version") != OUTER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Outer-fold manifest schema version does not match this protocol.")
    if manifest.get("protocol") != _outer_manifest_protocol(protocol):
        raise ValueError("Outer-fold manifest protocol configuration does not match.")
    if manifest.get("data_fingerprint") != dict(data_fingerprint):
        raise ValueError("Outer-fold manifest data fingerprint does not match current data.")

    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    label_by_sample_id = _label_by_sample_id(normalized_ids, y_binary)
    current_ids = set(normalized_ids)
    folds = manifest.get("folds")
    expected_count = protocol.outer_n_splits * protocol.outer_n_repeats
    if not isinstance(folds, list) or len(folds) != expected_count:
        raise ValueError(f"Outer-fold manifest must contain exactly {expected_count} fold records.")

    seen_pairs: set[tuple[int, int]] = set()
    test_ids_by_repeat = {repeat_id: [] for repeat_id in range(protocol.outer_n_repeats)}
    for record in folds:
        if not isinstance(record, Mapping):
            raise ValueError("Outer-fold manifest contains a non-object fold record.")
        repeat_id, fold_id = record.get("repeat_id"), record.get("fold_id")
        if type(repeat_id) is not int or type(fold_id) is not int:
            raise ValueError("Outer-fold repeat_id and fold_id must be integers.")
        if repeat_id not in range(protocol.outer_n_repeats) or fold_id not in range(protocol.outer_n_splits):
            raise ValueError("Outer-fold manifest contains an out-of-range repeat/fold identifier.")
        pair = (repeat_id, fold_id)
        if pair in seen_pairs:
            raise ValueError(f"Outer-fold manifest duplicates repeat/fold pair {pair}.")
        seen_pairs.add(pair)
        _, test_ids = _validate_partition(pair, record, label_by_sample_id, current_ids)
        test_ids_by_repeat[repeat_id].extend(test_ids)

    expected_pairs = {
        (repeat_id, fold_id)
        for repeat_id in range(protocol.outer_n_repeats)
        for fold_id in range(protocol.outer_n_splits)
    }
    if seen_pairs != expected_pairs:
        raise ValueError("Outer-fold manifest does not contain every fold for every repeat.")

    coverage: dict[str, dict[str, Any]] = {}
    for repeat_id, repeat_test_ids in test_ids_by_repeat.items():
        if len(repeat_test_ids) != len(current_ids):
            raise ValueError(
                f"Outer-fold manifest repeat {repeat_id} does not contain exactly one test entry per sample."
            )
        if len(set(repeat_test_ids)) != len(current_ids) or set(repeat_test_ids) != current_ids:
            raise ValueError(
                f"Outer-fold manifest repeat {repeat_id} does not cover every sample exactly once in test folds."
            )
        coverage[str(repeat_id)] = {
            "passed": True,
            "unique_test_sample_count": len(set(repeat_test_ids)),
        }

    expected_folds = build_outer_fold_manifest(
        normalized_ids, y_binary, data_fingerprint, protocol
    )["folds"]
    if folds != expected_folds:
        raise ValueError("Outer-fold contents do not match the deterministic protocol rebuild.")
    return {
        "passed": True,
        "fold_count": len(folds),
        "sample_count": len(current_ids),
        "per_repeat_test_coverage": coverage,
        "deterministic_rebuild_matches": True,
    }


def _write_manifest(manifest: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not path.parent.exists():
        raise FileNotFoundError(f"Manifest parent directory does not exist: {path.parent}")
    payload = _canonical_json_bytes(manifest)
    path.write_bytes(payload)
    return {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_size_bytes": len(payload),
    }


def write_outer_fold_manifest(manifest: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return _write_manifest(manifest, path)


def _load_manifest(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return manifest


def load_outer_fold_manifest(path: Path) -> dict[str, Any]:
    return _load_manifest(path, "Outer-fold manifest")


def derive_inner_fold_seed(
    repeat_id: int, fold_id: int, protocol: FoldProtocol
) -> int:
    if type(repeat_id) is not int or type(fold_id) is not int:
        raise ValueError("Inner-fold seed derivation requires integer outer fold identifiers.")
    if repeat_id not in range(protocol.outer_n_repeats) or fold_id not in range(protocol.outer_n_splits):
        raise ValueError("Inner-fold seed derivation requires a valid outer repeat/fold pair.")
    return protocol.inner_random_state_base + repeat_id * 100 + fold_id


def _build_inner_fold_records(
    repeat_id: int,
    fold_id: int,
    outer_train_sample_ids: Sequence[str],
    label_by_sample_id: Mapping[str, int],
    protocol: FoldProtocol,
) -> list[dict[str, Any]]:
    train_ids = list(outer_train_sample_ids)
    if train_ids != sorted(train_ids) or len(train_ids) != len(set(train_ids)):
        raise ValueError("Inner-fold construction requires sorted unique outer-training SAMPLE_ID values.")
    labels = np.asarray([label_by_sample_id[sample_id] for sample_id in train_ids], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Inner-fold construction requires both classes in each outer-training partition.")
    seed = derive_inner_fold_seed(repeat_id, fold_id, protocol)
    splitter = StratifiedKFold(
        n_splits=protocol.inner_n_splits, shuffle=True, random_state=seed
    )
    records: list[dict[str, Any]] = []
    for inner_fold_id, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        inner_train_ids = sorted(train_ids[index] for index in train_indices)
        validation_ids = sorted(train_ids[index] for index in validation_indices)
        records.append(
            {
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "inner_fold_id": inner_fold_id,
                "inner_seed": seed,
                "inner_train_sample_ids": inner_train_ids,
                "inner_validation_sample_ids": validation_ids,
                "inner_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(inner_train_ids),
                "inner_validation_sample_ids_canonical_json_sha256": _sample_id_list_sha256(validation_ids),
                "inner_train_partition": _partition_summary(inner_train_ids, label_by_sample_id),
                "inner_validation_partition": _partition_summary(validation_ids, label_by_sample_id),
            }
        )
    return records


def build_inner_fold_manifest(
    outer_manifest: Mapping[str, Any],
    outer_manifest_sha256: str,
    sample_ids: Sequence[str],
    y_binary: Sequence[int] | np.ndarray,
    data_fingerprint: Mapping[str, Any],
    protocol: FoldProtocol,
) -> dict[str, Any]:
    label_by_sample_id = _label_by_sample_id(sample_ids, y_binary)
    outer_folds: list[dict[str, Any]] = []
    for outer_record in outer_manifest["folds"]:
        repeat_id, fold_id = outer_record["repeat_id"], outer_record["fold_id"]
        outer_train_ids = list(outer_record["train_sample_ids"])
        outer_test_ids = list(outer_record["test_sample_ids"])
        outer_folds.append(
            {
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "outer_train_sample_count": len(outer_train_ids),
                "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_train_ids),
                "outer_test_sample_count": len(outer_test_ids),
                "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_test_ids),
                "inner_folds": _build_inner_fold_records(
                    repeat_id, fold_id, outer_train_ids, label_by_sample_id, protocol
                ),
            }
        )
    return {
        "schema_version": INNER_MANIFEST_SCHEMA_VERSION,
        "protocol": _inner_manifest_protocol(protocol),
        "data_fingerprint": dict(data_fingerprint),
        "outer_manifest_binding": {
            "outer_manifest_schema_version": OUTER_MANIFEST_SCHEMA_VERSION,
            "outer_manifest_sha256": outer_manifest_sha256,
        },
        "outer_folds": outer_folds,
    }


def validate_inner_fold_manifest(
    manifest: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    outer_manifest_sha256: str,
    sample_ids: Sequence[str],
    y_binary: Sequence[int] | np.ndarray,
    data_fingerprint: Mapping[str, Any],
    protocol: FoldProtocol,
) -> dict[str, Any]:
    validate_outer_fold_manifest(
        outer_manifest, sample_ids, y_binary, data_fingerprint, protocol
    )
    if not isinstance(manifest, Mapping):
        raise ValueError("Inner-fold manifest must be a JSON object.")
    if manifest.get("schema_version") != INNER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Inner-fold manifest schema version does not match this protocol.")
    if manifest.get("protocol") != _inner_manifest_protocol(protocol):
        raise ValueError("Inner-fold manifest splitter configuration does not match.")
    if manifest.get("data_fingerprint") != dict(data_fingerprint):
        raise ValueError("Inner-fold manifest data fingerprint does not match current data.")
    expected_binding = {
        "outer_manifest_schema_version": OUTER_MANIFEST_SCHEMA_VERSION,
        "outer_manifest_sha256": outer_manifest_sha256,
    }
    if manifest.get("outer_manifest_binding") != expected_binding:
        raise ValueError("Inner-fold manifest outer-manifest binding does not match.")

    label_by_sample_id = _label_by_sample_id(sample_ids, y_binary)
    outer_by_pair = {
        (record["repeat_id"], record["fold_id"]): record
        for record in outer_manifest["folds"]
    }
    records = manifest.get("outer_folds")
    expected_outer_count = protocol.outer_n_splits * protocol.outer_n_repeats
    if not isinstance(records, list) or len(records) != expected_outer_count:
        raise ValueError(
            f"Inner-fold manifest must contain exactly {expected_outer_count} outer records."
        )

    seen_pairs: set[tuple[int, int]] = set()
    seen_inner: set[tuple[int, int, int]] = set()
    coverage: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Inner-fold manifest contains a non-object outer record.")
        pair = (record.get("repeat_id"), record.get("fold_id"))
        if pair not in outer_by_pair or pair in seen_pairs:
            raise ValueError("Inner-fold manifest outer repeat/fold records are invalid or duplicated.")
        seen_pairs.add(pair)
        outer_record = outer_by_pair[pair]
        outer_train_ids = outer_record["train_sample_ids"]
        outer_test_ids = outer_record["test_sample_ids"]
        expected_metadata = {
            "outer_train_sample_count": len(outer_train_ids),
            "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_train_ids),
            "outer_test_sample_count": len(outer_test_ids),
            "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_test_ids),
        }
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"Inner-fold manifest outer metadata does not match outer fold {pair}.")
        inner_records = record.get("inner_folds")
        if not isinstance(inner_records, list) or len(inner_records) != protocol.inner_n_splits:
            raise ValueError(
                f"Inner-fold manifest outer fold {pair} must contain exactly {protocol.inner_n_splits} inner records."
            )
        outer_train_set, outer_test_set = set(outer_train_ids), set(outer_test_ids)
        validation_coverage: list[str] = []
        seen_inner_ids: set[int] = set()
        for inner in inner_records:
            if not isinstance(inner, Mapping):
                raise ValueError(f"Inner-fold manifest {pair} contains a non-object inner record.")
            inner_fold_id = inner.get("inner_fold_id")
            if (
                inner.get("repeat_id") != pair[0]
                or inner.get("fold_id") != pair[1]
                or type(inner_fold_id) is not int
                or inner_fold_id not in range(protocol.inner_n_splits)
                or inner_fold_id in seen_inner_ids
            ):
                raise ValueError(f"Inner-fold manifest {pair} has invalid or duplicated inner_fold_id values.")
            seen_inner_ids.add(inner_fold_id)
            triple = (pair[0], pair[1], inner_fold_id)
            if triple in seen_inner:
                raise ValueError("Inner-fold manifest duplicates an outer/inner fold record.")
            seen_inner.add(triple)
            if inner.get("inner_seed") != derive_inner_fold_seed(pair[0], pair[1], protocol):
                raise ValueError(f"Inner-fold manifest {pair} has an incorrect derived inner seed.")
            train_ids = inner.get("inner_train_sample_ids")
            validation_ids = inner.get("inner_validation_sample_ids")
            if (
                not isinstance(train_ids, list)
                or not isinstance(validation_ids, list)
                or train_ids != sorted(train_ids)
                or validation_ids != sorted(validation_ids)
            ):
                raise ValueError(f"Inner-fold manifest {pair} SAMPLE_ID lists must be sorted lists.")
            train_set, validation_set = set(train_ids), set(validation_ids)
            if len(train_set) != len(train_ids) or len(validation_set) != len(validation_ids):
                raise ValueError(f"Inner-fold manifest {pair} contains duplicated inner SAMPLE_ID values.")
            if not train_set.issubset(outer_train_set) or not validation_set.issubset(outer_train_set):
                raise ValueError(f"Inner-fold manifest {pair} contains IDs outside the outer-training partition.")
            if train_set.intersection(outer_test_set) or validation_set.intersection(outer_test_set):
                raise ValueError(f"Inner-fold manifest {pair} contains outer-test SAMPLE_ID values.")
            if train_set.intersection(validation_set) or train_set.union(validation_set) != outer_train_set:
                raise ValueError(f"Inner-fold manifest {pair} inner train/validation partitions are invalid.")
            if (
                inner.get("inner_train_sample_ids_canonical_json_sha256")
                != _sample_id_list_sha256(train_ids)
                or inner.get("inner_validation_sample_ids_canonical_json_sha256")
                != _sample_id_list_sha256(validation_ids)
            ):
                raise ValueError(f"Inner-fold manifest {pair} SAMPLE_ID hashes do not match.")
            for name, ids in (
                ("inner_train_partition", train_ids),
                ("inner_validation_partition", validation_ids),
            ):
                expected_partition = _partition_summary(ids, label_by_sample_id)
                if inner.get(name) != expected_partition or 0 in expected_partition["class_counts"].values():
                    raise ValueError(f"Inner-fold manifest {pair} {name} counts do not match current labels.")
            validation_coverage.extend(validation_ids)
        if (
            seen_inner_ids != set(range(protocol.inner_n_splits))
            or len(validation_coverage) != len(outer_train_ids)
            or len(set(validation_coverage)) != len(outer_train_ids)
            or set(validation_coverage) != outer_train_set
        ):
            raise ValueError(
                f"Inner-fold manifest {pair} does not cover every outer-training patient exactly once in validation."
            )
        coverage[f"{pair[0]}:{pair[1]}"] = {
            "passed": True,
            "validation_sample_count": len(validation_coverage),
        }

    if seen_pairs != set(outer_by_pair) or len(seen_inner) != expected_outer_count * protocol.inner_n_splits:
        raise ValueError("Inner-fold manifest does not contain the complete deterministic outer/inner record set.")
    expected = build_inner_fold_manifest(
        outer_manifest,
        outer_manifest_sha256,
        sample_ids,
        y_binary,
        data_fingerprint,
        protocol,
    )
    if dict(manifest) != expected:
        raise ValueError("Inner-fold manifest contents do not match the deterministic protocol rebuild.")
    return {
        "passed": True,
        "outer_record_count": len(records),
        "inner_record_count": len(seen_inner),
        "per_outer_validation_coverage": coverage,
        "deterministic_rebuild_matches": True,
    }


def write_inner_fold_manifest(manifest: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return _write_manifest(manifest, path)


def load_inner_fold_manifest(path: Path) -> dict[str, Any]:
    return _load_manifest(path, "Inner-fold manifest")
