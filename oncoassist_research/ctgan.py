"""Primary minority-only CTGAN augmentation through an isolated CPU worker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_json, payload_sha256, read_json_object


CTGAN_EXECUTION_BACKEND = "isolated_cpu_subprocess_v1"
CTGAN_STRATEGY = "minority_only_ctgan"
WORKER_SCHEMA_VERSION = "research-minority-ctgan-worker-v1"
_FORBIDDEN_NAME_TOKENS = ("class", "sample_id", "target", "label", "is_synthetic", "synthetic", "real")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update((json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class CTGANConfig:
    epochs: int
    pac: int
    verbose: bool

    def __post_init__(self) -> None:
        if type(self.epochs) is not int or self.epochs < 1 or type(self.pac) is not int or self.pac < 1 or type(self.verbose) is not bool:
            raise ValueError("CTGAN epochs/pac must be positive integers and verbose must be boolean.")


@dataclass(frozen=True)
class MinorityCTGANInput:
    minority_features: np.ndarray
    minority_label: int
    majority_label: int
    needed_synthetic_count: int
    training_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class AugmentedTrainingSet:
    features: np.ndarray
    labels: np.ndarray
    record_ids: tuple[str, ...]
    is_synthetic: np.ndarray
    feature_names: tuple[str, ...]
    evidence: Mapping[str, Any]


def derive_ctgan_batch_size(fit_row_count: int, pac: int, max_batch_size: int = 500) -> int:
    if type(fit_row_count) is not int or type(pac) is not int or type(max_batch_size) is not int or fit_row_count < 1 or pac < 1 or max_batch_size < pac:
        raise ValueError("CTGAN batch-size inputs are invalid.")
    batch_size = (min(max_batch_size, fit_row_count) // pac) * pac
    if batch_size < pac or batch_size % pac or batch_size % 2:
        raise ValueError("Derived CTGAN batch size must be even, at least pac, and divisible by pac.")
    return batch_size


def _feature_schema(feature_names: Sequence[str], expected_width: int, expected_hash: str) -> tuple[str, ...]:
    names = tuple(feature_names)
    if len(names) != expected_width or len(names) != len(set(names)) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("CTGAN feature schema is invalid.")
    if any(token in name.lower() for name in names for token in _FORBIDDEN_NAME_TOKENS):
        raise ValueError("CTGAN feature names contain a forbidden target or identifier token.")
    if payload_sha256(list(names)) != expected_hash:
        raise ValueError("CTGAN feature-name hash does not match the ordered schema.")
    return names


def extract_minority_training_input(
    fused_training: np.ndarray,
    training_labels: Sequence[int] | np.ndarray,
    training_sample_ids: Sequence[str],
    feature_names: Sequence[str],
    feature_names_sha256: str,
) -> MinorityCTGANInput:
    matrix = np.asarray(fused_training)
    labels = np.asarray(training_labels)
    if matrix.ndim != 2 or matrix.dtype == object or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("CTGAN requires a finite numeric fused training matrix.")
    if labels.ndim != 1 or len(labels) != len(matrix) or not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise ValueError("CTGAN labels must be finite numeric values aligned to training rows.")
    ids = tuple(str(item) for item in training_sample_ids)
    if len(ids) != len(matrix) or len(ids) != len(set(ids)) or any(not item.strip() for item in ids):
        raise ValueError("CTGAN training SAMPLE_ID values are invalid.")
    names = _feature_schema(feature_names, matrix.shape[1], feature_names_sha256)
    values, counts = np.unique(labels.astype(int), return_counts=True)
    if len(values) != 2:
        raise ValueError("Minority-only CTGAN requires exactly two observed classes.")
    if counts[0] == counts[1]:
        raise ValueError("Minority-only CTGAN requires strict class imbalance.")
    minority_index, majority_index = int(np.argmin(counts)), int(np.argmax(counts))
    minority_label, majority_label = int(values[minority_index]), int(values[majority_index])
    minority = np.ascontiguousarray(matrix[labels.astype(int) == minority_label].astype(np.float32, copy=True))
    needed = int(counts[majority_index] - counts[minority_index])
    evidence = {"strategy": CTGAN_STRATEGY, "real_training_row_count": len(matrix), "training_ids_sha256": payload_sha256(list(ids)), "training_labels_sha256": _array_sha256(labels.astype(int, copy=True)), "feature_names_sha256": feature_names_sha256, "raw_fused_training_sha256": _array_sha256(np.ascontiguousarray(matrix.astype(np.float32, copy=True))), "minority_label": minority_label, "majority_label": majority_label, "minority_count": int(counts[minority_index]), "majority_count": int(counts[majority_index]), "needed_synthetic_count": needed, "minority_training_sha256": _array_sha256(minority), "raw_fused_latent_input_only": True}
    return MinorityCTGANInput(_readonly_copy(minority), minority_label, majority_label, needed, ids, names, evidence)


def _run_worker(minority: MinorityCTGANInput, config: CTGANConfig, batch_size: int, seed: int) -> tuple[np.ndarray, Mapping[str, Any]]:
    if type(seed) is not int or seed < 0:
        raise ValueError("CTGAN seed must be a non-negative integer.")
    worker = Path(__file__).with_name("ctgan_worker.py")
    request = {"schema_version": WORKER_SCHEMA_VERSION, "strategy": CTGAN_STRATEGY, "feature_names": list(minority.feature_names), "feature_names_sha256": payload_sha256(list(minority.feature_names)), "ctgan_config": {"epochs": config.epochs, "pac": config.pac, "verbose": config.verbose, "batch_size": batch_size}, "requested_synthetic_rows": minority.needed_synthetic_count, "seed": seed, "input_minority_features_sha256": _array_sha256(minority.minority_features)}
    with tempfile.TemporaryDirectory(prefix="research-minority-ctgan-") as directory_name:
        directory = Path(directory_name); request_path = directory / "request.json"; input_path = directory / "input.npz"; output_path = directory / "synthetic.npz"; response_path = directory / "response.json"
        atomic_write_json(request_path, request); np.savez_compressed(input_path, features=minority.minority_features)
        environment = os.environ.copy(); environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONFAULTHANDLER": "1"})
        completed = subprocess.run([sys.executable, str(worker), str(request_path), str(input_path), str(output_path), str(response_path)], cwd=str(directory), env=environment, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Minority CTGAN worker failed; no fallback exists. stderr: {completed.stderr[-2000:]}")
        response = read_json_object(response_path)
        with np.load(output_path) as output:
            synthetic = output["synthetic"]
    required = {"schema_version": WORKER_SCHEMA_VERSION, "strategy": CTGAN_STRATEGY, "feature_names": list(minority.feature_names), "feature_names_sha256": request["feature_names_sha256"], "requested_synthetic_rows": minority.needed_synthetic_count, "returned_synthetic_rows": minority.needed_synthetic_count, "input_minority_features_sha256": request["input_minority_features_sha256"]}
    if any(response.get(key) != value for key, value in required.items()):
        raise RuntimeError("Minority CTGAN worker response contract differs; no fallback exists.")
    if response.get("constructor_configuration", {}).get("batch_size") != batch_size or response.get("constructor_configuration", {}).get("enable_gpu") is not False:
        raise RuntimeError("Minority CTGAN worker configuration differs; no fallback exists.")
    execution = response.get("execution_evidence")
    if not isinstance(execution, dict) or execution.get("ctgan_execution_backend") != CTGAN_EXECUTION_BACKEND or execution.get("tensorflow_present_in_worker") is not False or execution.get("ctgan_gpu_enabled") is not False:
        raise RuntimeError("Minority CTGAN worker isolation evidence is invalid; no fallback exists.")
    synthetic = np.ascontiguousarray(np.asarray(synthetic, dtype=np.float32))
    if synthetic.shape != (minority.needed_synthetic_count, len(minority.feature_names)) or not np.isfinite(synthetic).all() or response.get("synthetic_sha256") != _array_sha256(synthetic):
        raise RuntimeError("Minority CTGAN synthetic output is invalid; no fallback exists.")
    return synthetic, response


def fit_and_sample_minority_ctgan(minority: MinorityCTGANInput, config: CTGANConfig, seed: int) -> tuple[np.ndarray, Mapping[str, Any]]:
    if not isinstance(minority, MinorityCTGANInput) or not isinstance(config, CTGANConfig):
        raise TypeError("Minority CTGAN requires MinorityCTGANInput and CTGANConfig.")
    batch_size = derive_ctgan_batch_size(len(minority.minority_features), config.pac)
    return _run_worker(minority, config, batch_size, seed)


def probe_isolated_ctgan_worker(config: CTGANConfig) -> Mapping[str, Any]:
    """Verify the isolated CTGAN runtime without fitting or sampling data."""
    if not isinstance(config, CTGANConfig):
        raise TypeError("CTGAN worker probe requires CTGANConfig.")
    worker = Path(__file__).with_name("ctgan_worker.py")
    with tempfile.TemporaryDirectory(prefix="research-minority-ctgan-probe-") as directory_name:
        directory = Path(directory_name)
        response_path = directory / "response.json"
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONFAULTHANDLER": "1",
            }
        )
        completed = subprocess.run(
            [sys.executable, str(worker), "--preflight", str(response_path)],
            cwd=str(directory),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Minority CTGAN worker preflight failed. stderr: {completed.stderr[-2000:]}")
        response = read_json_object(response_path)
    required = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "preflight_only": True,
        "ctgan_execution_backend": CTGAN_EXECUTION_BACKEND,
        "cuda_visible_devices": "",
        "ctgan_gpu_enabled": False,
    }
    if any(response.get(key) != value for key, value in required.items()):
        raise RuntimeError("Minority CTGAN worker preflight response differs from the required CPU contract.")
    parameters = response.get("constructor_parameters")
    if not isinstance(parameters, list) or {"metadata", "epochs", "batch_size", "pac", "verbose"}.difference(parameters):
        raise RuntimeError("Minority CTGAN worker constructor API is incompatible; no fallback exists.")
    versions = response.get("versions")
    if not isinstance(versions, Mapping):
        raise RuntimeError("Minority CTGAN worker preflight did not report dependency versions.")
    return response


def augment_with_minority_ctgan(
    fused_training: np.ndarray, training_labels: Sequence[int] | np.ndarray, training_sample_ids: Sequence[str], feature_names: Sequence[str], feature_names_sha256: str, config: CTGANConfig, seed: int, synthetic_namespace: str
) -> AugmentedTrainingSet:
    if not isinstance(synthetic_namespace, str) or not synthetic_namespace or not re.fullmatch(r"[A-Za-z0-9._:-]+", synthetic_namespace):
        raise ValueError("synthetic_namespace must be a non-empty safe string.")
    real_features = np.ascontiguousarray(np.asarray(fused_training, dtype=np.float32).copy())
    real_labels = np.asarray(training_labels, dtype=int).copy()
    minority = extract_minority_training_input(real_features, real_labels, training_sample_ids, feature_names, feature_names_sha256)
    synthetic, worker_evidence = fit_and_sample_minority_ctgan(minority, config, seed)
    if not np.array_equal(real_features, np.asarray(fused_training, dtype=np.float32)) or not np.array_equal(real_labels, np.asarray(training_labels, dtype=int)):
        raise AssertionError("Minority CTGAN changed real training inputs.")
    synthetic_ids = tuple(f"SYNTHETIC:{synthetic_namespace}:MINORITY:{minority.minority_label}:{index:06d}" for index in range(len(synthetic)))
    if len(set(synthetic_ids)) != len(synthetic_ids) or set(synthetic_ids).intersection(minority.training_ids):
        raise AssertionError("Synthetic CTGAN record IDs are invalid.")
    features = np.vstack([real_features, synthetic]).astype(np.float32, copy=False)
    labels = np.concatenate([real_labels, np.full(len(synthetic), minority.minority_label, dtype=int)])
    mask = np.concatenate([np.zeros(len(real_features), dtype=bool), np.ones(len(synthetic), dtype=bool)])
    counts = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    if counts.get(minority.minority_label) != counts.get(minority.majority_label):
        raise AssertionError("Minority CTGAN augmentation did not exactly balance classes.")
    evidence = {**minority.evidence, "ctgan_config": {"epochs": config.epochs, "pac": config.pac, "verbose": config.verbose}, "derived_batch_size": derive_ctgan_batch_size(len(minority.minority_features), config.pac), "requested_seed": seed, "execution_backend": CTGAN_EXECUTION_BACKEND, "worker": worker_evidence, "synthetic_sha256": _array_sha256(synthetic), "augmented_features_sha256": _array_sha256(features), "augmented_labels_sha256": _array_sha256(labels), "synthetic_ids_sha256": payload_sha256(list(synthetic_ids)), "real_rows_first": True, "synthetic_rows_second": True, "real_training_unchanged": True, "heldout_supplied_to_ctgan": False, "fallback_exists": False, "exact_regeneration_guaranteed": False, "final_class_counts": {str(key): value for key, value in counts.items()}}
    return AugmentedTrainingSet(_readonly_copy(features), _readonly_copy(labels), tuple(minority.training_ids) + synthetic_ids, _readonly_copy(mask), minority.feature_names, evidence)
