"""Controlled Primary V1 preflight, execution orchestration, and publications.

Scientific fitting remains exclusively inside the frozen public API chain. This
module supplies canonical populations, immutable publication boundaries, and
fail-closed resume control without implementing scientific algorithms itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

from . import artifacts
from .ctgan import CTGANConfig, derive_ctgan_batch_size, probe_isolated_ctgan_worker
from .calibration import cross_fit_sigmoid_calibration, fit_final_sigmoid_calibrator
from .data import load_and_align_multiomics
from .folds import (
    build_inner_fold_manifest,
    build_outer_data_fingerprint,
    build_outer_fold_manifest,
    manifest_identity_sha256,
    validate_inner_fold_manifest,
    validate_outer_fold_manifest,
)
from .protocol import (
    PRIMARY_MODALITIES,
    PrimaryProtocolV1,
    PrimarySeedManifest,
    PrimaryV1RunProvenance,
    create_primary_v1_run_provenance,
    ordered_patient_ids_sha256,
    validate_primary_v1_run_provenance,
)
from .search import build_primary_candidates, run_primary_inner_search, validate_primary_search_result
from .thresholds import select_operational_threshold
from .final_refit import (
    FrozenOuterPredictions,
    OuterPrediction,
    fit_final_primary_model,
    score_outer_test,
    validate_final_primary_model_bundle,
    validate_frozen_outer_predictions,
)
from .outer_evaluation import evaluate_outer_test_predictions


RUNNER_SCHEMA_VERSION = "controlled-primary-v1-runner-v1"
STUDY_BINDING_SCHEMA_VERSION = "controlled-primary-v1-study-binding-v1"
PREFLIGHT_SCHEMA_VERSION = "controlled-primary-v1-preflight-v1"
SCORING_RECEIPT_SCHEMA_VERSION = "controlled-primary-v1-scoring-receipt-v1"
EVALUATION_PUBLICATION_SCHEMA_VERSION = "controlled-primary-v1-evaluation-publication-v1"
EXPECTED_REFERENCE_SHA256 = "637EC5378B2C10EE1576136141A21D273D668B89255A7102C8C0D6590F9BC074"
FEATURE_PROVENANCE_STATUS = "UNKNOWN"
_STUDY_STATES = {"NEW", "PREFLIGHTED", "RUNNING", "INTERRUPTED", "FAILED", "COMPLETE"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _coordinate_key(repeat_id: int, fold_id: int) -> str:
    return f"{repeat_id}:{fold_id}"


def _coordinate_payload(repeat_id: int, fold_id: int) -> dict[str, int]:
    return {"repeat_id": repeat_id, "fold_id": fold_id}


def _validate_coordinate(repeat_id: int, fold_id: int) -> None:
    if type(repeat_id) is not int or type(fold_id) is not int or repeat_id not in range(5) or fold_id not in range(5):
        raise ValueError("Coordinate is outside the frozen five-repeat/five-fold protocol.")


def primary_coordinate_plan(protocol: PrimaryProtocolV1) -> tuple[tuple[int, int], ...]:
    """Return the complete protocol-derived 25-coordinate plan."""
    if not isinstance(protocol, PrimaryProtocolV1):
        raise TypeError("Coordinate planning requires PrimaryProtocolV1.")
    plan = tuple(
        (repeat_id, fold_id)
        for repeat_id in range(protocol.outer_n_repeats)
        for fold_id in range(protocol.outer_n_splits)
    )
    if len(plan) != 25 or len(set(plan)) != 25:
        raise AssertionError("Frozen Primary V1 coordinate plan must contain exactly 25 unique coordinates.")
    return plan


@dataclass(frozen=True)
class ControlledRunnerConfig:
    rna_path: Path
    dna_path: Path
    cna_path: Path
    output_root: Path
    run_id: str
    root_seed: int
    ae_device_policy: str
    reference_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip() or type(self.root_seed) is not int or self.root_seed < 0:
            raise ValueError("Runner run_id and root_seed are invalid.")
        if self.ae_device_policy not in {"cpu", "gpu"}:
            raise ValueError("AE device policy must be explicitly 'cpu' or 'gpu'.")
        for name in ("rna_path", "dna_path", "cna_path", "output_root"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path.")

    @property
    def immutable_reference_path(self) -> Path:
        if self.reference_path is not None:
            return self.reference_path
        return Path(__file__).resolve().parent.parent / "reference" / "current_working_source.py"


@dataclass(frozen=True)
class StudyBinding:
    payload: Mapping[str, Any]

    @property
    def study_identity_sha256(self) -> str:
        return artifacts.payload_sha256(dict(self.payload))

    def as_json(self) -> dict[str, Any]:
        return {**dict(self.payload), "study_identity_sha256": self.study_identity_sha256}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "StudyBinding":
        if not isinstance(value, Mapping):
            raise ValueError("Study binding must be a JSON object.")
        payload = dict(value)
        identity = payload.pop("study_identity_sha256", None)
        binding = cls(payload)
        if identity != binding.study_identity_sha256:
            raise ValueError("Study binding identity does not verify.")
        return binding


@dataclass(frozen=True)
class PreparedStudy:
    config: ControlledRunnerConfig
    aligned_data: Mapping[str, Any]
    protocol: PrimaryProtocolV1
    provenance: PrimaryV1RunProvenance
    seed_manifest: PrimarySeedManifest
    fold_protocol: Any
    outer_manifest: Mapping[str, Any]
    inner_manifest: Mapping[str, Any]
    binding: StudyBinding
    preflight: Mapping[str, Any]

    @property
    def study_directory(self) -> Path:
        return self.config.output_root / self.binding.study_identity_sha256


@dataclass(frozen=True)
class ScoringPublicationReceipt:
    """Portable, model-weight-free authority to be published by STEP 11B-B."""

    study_identity_sha256: str
    repeat_id: int
    fold_id: int
    run_provenance_identity_sha256: str
    immutable_reference_sha256: str
    protocol_sha256: str
    seed_manifest_identity_sha256: str
    outer_manifest_identity_sha256: str
    inner_manifest_identity_sha256: str
    fold_authority_identity_sha256: str
    search_selection_identity_sha256: str
    selected_candidate_identity_sha256: str
    selected_oof_predictions_sha256: str
    cross_fitted_calibration_sha256: str
    final_sigmoid_identity_sha256: str
    threshold_identity_sha256: str
    final_model_identity_sha256: str
    frozen_model_state_sha256: str
    aligned_data_content_identity_sha256: str
    ordered_outer_test_ids_sha256: str
    frozen_prediction_hash: str

    def __post_init__(self) -> None:
        _validate_coordinate(self.repeat_id, self.fold_id)
        for name, value in self._identity_values().items():
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Scoring receipt {name} must be a SHA-256 identity.")

    def _identity_values(self) -> dict[str, str]:
        return {
            "study_identity_sha256": self.study_identity_sha256,
            "run_provenance_identity_sha256": self.run_provenance_identity_sha256,
            "immutable_reference_sha256": self.immutable_reference_sha256,
            "protocol_sha256": self.protocol_sha256,
            "seed_manifest_identity_sha256": self.seed_manifest_identity_sha256,
            "outer_manifest_identity_sha256": self.outer_manifest_identity_sha256,
            "inner_manifest_identity_sha256": self.inner_manifest_identity_sha256,
            "fold_authority_identity_sha256": self.fold_authority_identity_sha256,
            "search_selection_identity_sha256": self.search_selection_identity_sha256,
            "selected_candidate_identity_sha256": self.selected_candidate_identity_sha256,
            "selected_oof_predictions_sha256": self.selected_oof_predictions_sha256,
            "cross_fitted_calibration_sha256": self.cross_fitted_calibration_sha256,
            "final_sigmoid_identity_sha256": self.final_sigmoid_identity_sha256,
            "threshold_identity_sha256": self.threshold_identity_sha256,
            "final_model_identity_sha256": self.final_model_identity_sha256,
            "frozen_model_state_sha256": self.frozen_model_state_sha256,
            "aligned_data_content_identity_sha256": self.aligned_data_content_identity_sha256,
            "ordered_outer_test_ids_sha256": self.ordered_outer_test_ids_sha256,
            "frozen_prediction_hash": self.frozen_prediction_hash,
        }

    def content(self) -> dict[str, Any]:
        return {"schema_version": SCORING_RECEIPT_SCHEMA_VERSION, "coordinate": _coordinate_payload(self.repeat_id, self.fold_id), **self._identity_values()}

    @property
    def scoring_publication_identity_sha256(self) -> str:
        return artifacts.payload_sha256(self.content())

    def as_json(self) -> dict[str, Any]:
        return {**self.content(), "scoring_publication_identity_sha256": self.scoring_publication_identity_sha256}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ScoringPublicationReceipt":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCORING_RECEIPT_SCHEMA_VERSION:
            raise ValueError("Scoring publication schema is invalid.")
        coordinate = value.get("coordinate")
        if not isinstance(coordinate, Mapping):
            raise ValueError("Scoring publication coordinate is invalid.")
        fields = {
            name: value.get(name)
            for name in cls.__dataclass_fields__
            if name not in {"repeat_id", "fold_id"}
        }
        receipt = cls(repeat_id=coordinate.get("repeat_id"), fold_id=coordinate.get("fold_id"), **fields)
        if value != receipt.as_json():
            raise ValueError("Scoring publication content or identity does not verify.")
        return receipt


def _package_versions() -> dict[str, str]:
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "sklearn",
        "tensorflow": "tensorflow",
        "torch": "torch",
        "sdv": "sdv",
        "ctgan": "ctgan",
    }
    versions: dict[str, str] = {}
    for name, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            versions[name] = "unavailable"
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _runtime_environment(ae_device_policy: str) -> dict[str, Any]:
    versions = _package_versions()
    if versions["tensorflow"] == "unavailable":
        raise RuntimeError("TensorFlow is required by the frozen Primary V1 autoencoder implementation.")
    visible_devices: list[str] = []
    cpu_inventory: list[str] = []
    gpu_inventory: list[str] = []
    try:
        tensorflow = importlib.import_module("tensorflow")
        visible_devices = [device.name for device in tensorflow.config.get_visible_devices()]
        cpu_inventory = [device.name for device in tensorflow.config.list_physical_devices("CPU")]
        gpu_inventory = [device.name for device in tensorflow.config.list_physical_devices("GPU")]
    except (AttributeError, RuntimeError):
        visible_devices = []
        cpu_inventory = []
        gpu_inventory = []
    if ae_device_policy == "cpu" and not cpu_inventory:
        raise RuntimeError("AE device policy 'cpu' was requested but TensorFlow reports no CPU device.")
    if ae_device_policy == "gpu" and not gpu_inventory:
        raise RuntimeError("AE device policy 'gpu' was requested but no TensorFlow GPU is available.")
    compatibility = {
        "schema_version": "controlled-primary-v1-environment-v1",
        "python": sys.version.split()[0],
        "package_versions": versions,
        "ae_device_policy": ae_device_policy,
    }
    return {
        **compatibility,
        "environment_compatibility_identity_sha256": artifacts.payload_sha256(compatibility),
        "visible_tensorflow_devices": visible_devices,
        "cpu_inventory": cpu_inventory,
        "gpu_inventory_informational": gpu_inventory,
    }


def _filesystem_capability_probe(output_root: Path) -> dict[str, bool]:
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / f".controlled-runner-probe-{uuid.uuid4().hex}"
    probe.mkdir()
    try:
        immutable = probe / "immutable.json"
        artifacts.create_immutable_json(immutable, {"probe": "immutable"})
        try:
            artifacts.create_immutable_json(immutable, {"probe": "overwrite"})
        except FileExistsError:
            create_once = True
        else:
            create_once = False
        replaced = probe / "replaceable.json"
        artifacts.atomic_write_json(replaced, {"value": 1})
        artifacts.atomic_write_json(replaced, {"value": 2})
        atomic_replace = artifacts.read_json_object(replaced) == {"value": 2}
        published = probe / "published"
        artifacts.publish_directory(published, lambda temporary: artifacts.create_immutable_json(temporary / "evidence.json", {"probe": "directory"}))
        directory_publication = (published / "evidence.json").is_file()
        lock = artifacts.acquire_run_lock(probe / "locking", {"probe": "controlled-runner"})
        locking = artifacts.release_run_lock(lock, outcome="preflight_probe")
        result = {
            "create_once_files": create_once,
            "atomic_replace": atomic_replace,
            "same_parent_directory_publication": directory_publication,
            "locking": locking,
        }
        if not all(result.values()):
            raise RuntimeError("Output root did not satisfy required artifact filesystem capabilities.")
        return result
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def _fold_authority_identity(
    protocol: PrimaryProtocolV1,
    seed_manifest: PrimarySeedManifest,
    fold_protocol: Any,
    outer_identity: str,
    inner_identity: str,
    provenance_identity: str,
    repeat_id: int,
    fold_id: int,
) -> str:
    return artifacts.payload_sha256(
        {
            "schema_version": "primary-fold-authority-v1",
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.identity_sha256,
            "run_provenance_identity_sha256": provenance_identity,
            "seed_manifest_identity_sha256": seed_manifest.identity_sha256,
            "fold_protocol_identity_sha256": protocol.fold_protocol_identity_sha256(fold_protocol, seed_manifest, provenance_identity),
            "outer_manifest_identity_sha256": outer_identity,
            "inner_manifest_identity_sha256": inner_identity,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
        }
    )


def _validate_ctgan_feasibility(
    outer_manifest: Mapping[str, Any],
    inner_manifest: Mapping[str, Any],
    labels_by_id: Mapping[str, int],
    config: CTGANConfig,
) -> dict[str, Any]:
    partitions: list[dict[str, Any]] = []
    for record in outer_manifest["folds"]:
        ids = record["train_sample_ids"]
        counts = np.bincount(np.asarray([labels_by_id[item] for item in ids], dtype=int), minlength=2)
        if not counts[0] or not counts[1] or counts[0] == counts[1]:
            raise ValueError("Mandatory minority-only CTGAN requires strict imbalance in every final outer-training partition.")
        minority = int(min(counts))
        partitions.append({"kind": "final_outer_training", "coordinate": _coordinate_payload(record["repeat_id"], record["fold_id"]), "minority_count": minority, "derived_batch_size": derive_ctgan_batch_size(minority, config.pac)})
    for outer in inner_manifest["outer_folds"]:
        for inner in outer["inner_folds"]:
            ids = inner["inner_train_sample_ids"]
            counts = np.bincount(np.asarray([labels_by_id[item] for item in ids], dtype=int), minlength=2)
            if not counts[0] or not counts[1] or counts[0] == counts[1]:
                raise ValueError("Mandatory minority-only CTGAN requires strict imbalance in every inner-training partition.")
            minority = int(min(counts))
            partitions.append({"kind": "inner_training", "coordinate": _coordinate_payload(outer["repeat_id"], outer["fold_id"]), "inner_fold_id": inner["inner_fold_id"], "minority_count": minority, "derived_batch_size": derive_ctgan_batch_size(minority, config.pac)})
    return {"pac": config.pac, "partition_count": len(partitions), "partitions": partitions}


def prepare_study(config: ControlledRunnerConfig) -> PreparedStudy:
    """Run all zero-fit checks and construct immutable study authority in memory."""
    reference_hash = _sha256_file(config.immutable_reference_path)
    if reference_hash != EXPECTED_REFERENCE_SHA256:
        raise ValueError("Immutable reference SHA-256 does not match the required controlled baseline.")
    protocol = PrimaryProtocolV1()
    if protocol.feature_provenance_status != FEATURE_PROVENANCE_STATUS:
        raise ValueError("Frozen Primary V1 feature provenance status differs from UNKNOWN.")
    aligned_data = load_and_align_multiomics(config.rna_path, config.dna_path, config.cna_path)
    provenance = create_primary_v1_run_provenance(run_id=config.run_id, root_seed=config.root_seed, protocol=protocol, aligned_data=aligned_data)
    validate_primary_v1_run_provenance(provenance, protocol=protocol, aligned_data=aligned_data)
    candidates = build_primary_candidates({"mGE": len(aligned_data["feature_columns"]["rna"]), "mDM": len(aligned_data["feature_columns"]["dna"]), "mCNA": len(aligned_data["feature_columns"]["cna"])}, protocol)
    if len(candidates) != 9 or any(candidate.latent_dimensions[modality] < 2 or candidate.latent_dimensions[modality] >= len(aligned_data["feature_columns"][{"mGE": "rna", "mDM": "dna", "mCNA": "cna"}[modality]]) for candidate in candidates for modality in PRIMARY_MODALITIES):
        raise ValueError("Primary candidate grid is not exactly the frozen compressive nine-candidate grid.")
    seed_manifest = PrimarySeedManifest.generate_primary(provenance, candidates)
    protocol.validate_primary_seed_manifest(seed_manifest, provenance)
    fold_protocol = protocol.make_fold_protocol(seed_manifest, provenance)
    data_fingerprint = build_outer_data_fingerprint(aligned_data)
    outer_manifest = build_outer_fold_manifest(aligned_data["sample_ids"], aligned_data["y_binary"], data_fingerprint, fold_protocol)
    validate_outer_fold_manifest(outer_manifest, aligned_data["sample_ids"], aligned_data["y_binary"], data_fingerprint, fold_protocol)
    outer_identity = manifest_identity_sha256(outer_manifest)
    inner_manifest = build_inner_fold_manifest(outer_manifest, outer_identity, aligned_data["sample_ids"], aligned_data["y_binary"], data_fingerprint, fold_protocol)
    validate_inner_fold_manifest(inner_manifest, outer_manifest, outer_identity, aligned_data["sample_ids"], aligned_data["y_binary"], data_fingerprint, fold_protocol)
    inner_identity = manifest_identity_sha256(inner_manifest)
    plan = primary_coordinate_plan(protocol)
    labels_by_id = dict(zip(aligned_data["sample_ids"], np.asarray(aligned_data["y_binary"], dtype=int).tolist()))
    ctgan_config = protocol.make_ctgan_config()
    feasibility = _validate_ctgan_feasibility(outer_manifest, inner_manifest, labels_by_id, ctgan_config)
    worker_probe = probe_isolated_ctgan_worker(ctgan_config)
    environment = _runtime_environment(config.ae_device_policy)
    filesystem = _filesystem_capability_probe(config.output_root)
    fold_authorities = {
        _coordinate_key(repeat_id, fold_id): _fold_authority_identity(protocol, seed_manifest, fold_protocol, outer_identity, inner_identity, provenance.identity_sha256, repeat_id, fold_id)
        for repeat_id, fold_id in plan
    }
    binding_payload = {
        "schema_version": STUDY_BINDING_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "immutable_reference_sha256": reference_hash,
        "protocol_identity_sha256": protocol.identity_sha256,
        "run_provenance_identity_sha256": provenance.identity_sha256,
        "aligned_data_content_identity_sha256": provenance.aligned_data_content_identity_sha256,
        "feature_schema_sha256": dict(provenance.feature_schema_sha256),
        "seed_manifest_identity_sha256": seed_manifest.identity_sha256,
        "fold_protocol_identity_sha256": protocol.fold_protocol_identity_sha256(fold_protocol, seed_manifest, provenance),
        "outer_manifest_identity_sha256": outer_identity,
        "inner_manifest_identities_sha256": {key: inner_identity for key in fold_authorities},
        "coordinate_fold_authority_sha256": fold_authorities,
        "coordinates": [_coordinate_payload(repeat_id, fold_id) for repeat_id, fold_id in plan],
        "feature_provenance_status": FEATURE_PROVENANCE_STATUS,
        "ae_device_policy": config.ae_device_policy,
        "environment_compatibility_identity_sha256": environment["environment_compatibility_identity_sha256"],
    }
    binding = StudyBinding(binding_payload)
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "study_identity_sha256": binding.study_identity_sha256,
        "immutable_reference_sha256": reference_hash,
        "feature_provenance_status": FEATURE_PROVENANCE_STATUS,
        "environment": environment,
        "filesystem_capabilities": filesystem,
        "ctgan_worker_probe": dict(worker_probe),
        "ctgan_feasibility": feasibility,
        "candidate_count": len(candidates),
        "coordinate_count": len(plan),
        "outer_manifest_identity_sha256": outer_identity,
        "inner_manifest_identity_sha256": inner_identity,
    }
    return PreparedStudy(config, aligned_data, protocol, provenance, seed_manifest, fold_protocol, outer_manifest, inner_manifest, binding, preflight)


def _study_lock_binding(binding: StudyBinding) -> dict[str, str]:
    return {"schema_version": "controlled-primary-v1-study-lock-v1", "study_identity_sha256": binding.study_identity_sha256}


def acquire_study_lock(study_directory: Path, binding: StudyBinding) -> artifacts.RunLock:
    return artifacts.acquire_run_lock(study_directory, _study_lock_binding(binding))


def recover_abandoned_study_lock(study_directory: Path, binding: StudyBinding, expected_lock_id: str) -> None:
    artifacts.recover_abandoned_run_lock(study_directory, _study_lock_binding(binding), expected_lock_id)


def initialize_study(prepared: PreparedStudy) -> Path:
    """Create only immutable study authority and a reconstructable state cache."""
    directory = prepared.study_directory
    if directory.exists():
        validate_study_directory(directory, prepared.binding)
        return directory
    directory.mkdir(parents=True)
    artifacts.create_immutable_json(directory / "study_binding.json", prepared.binding.as_json())
    manifests = directory / "fold_manifests"
    manifests.mkdir()
    artifacts.create_immutable_json(manifests / "outer.json", prepared.outer_manifest)
    artifacts.create_immutable_json(manifests / "inner.json", prepared.inner_manifest)
    artifacts.create_immutable_json(directory / "preflight.json", dict(prepared.preflight))
    (directory / "failures").mkdir()
    (directory / "outer_folds").mkdir()
    for repeat_id, fold_id in primary_coordinate_plan(prepared.protocol):
        _coordinate_directory(directory, repeat_id, fold_id).mkdir(parents=True)
    write_runtime_state(directory, prepared.binding)
    return directory


def validate_study_directory(study_directory: Path, binding: StudyBinding) -> None:
    persisted = StudyBinding.from_json(artifacts.read_json_object(study_directory / "study_binding.json"))
    if persisted.as_json() != binding.as_json():
        raise ValueError("Existing study directory binding is incompatible with this controlled study.")
    for path, expected_identity in (
        (study_directory / "fold_manifests" / "outer.json", binding.payload["outer_manifest_identity_sha256"]),
        (study_directory / "fold_manifests" / "inner.json", next(iter(binding.payload["inner_manifest_identities_sha256"].values()))),
    ):
        payload = artifacts.read_json_object(path)
        if manifest_identity_sha256(payload) != expected_identity:
            raise ValueError("Persisted fold manifest identity differs from the study binding.")
    preflight = artifacts.read_json_object(study_directory / "preflight.json")
    if preflight.get("study_identity_sha256") != binding.study_identity_sha256 or preflight.get("environment", {}).get("environment_compatibility_identity_sha256") != binding.payload["environment_compatibility_identity_sha256"]:
        raise ValueError("Persisted preflight evidence differs from the study binding.")


def _coordinate_directory(study_directory: Path, repeat_id: int, fold_id: int) -> Path:
    return study_directory / "outer_folds" / f"repeat-{repeat_id:02d}" / f"fold-{fold_id:02d}"


def _validate_scoring_directory(path: Path, binding: StudyBinding, repeat_id: int, fold_id: int) -> ScoringPublicationReceipt:
    required = {"publication.json", "frozen_predictions.json", "selected_search_summary.json", "calibration_threshold_summary.json", "final_refit_summary.json"}
    present = {item.name for item in path.iterdir()} if path.is_dir() else set()
    if present != required:
        raise ValueError("Scoring publication is incomplete, malformed, or contains unexpected evidence.")
    receipt = ScoringPublicationReceipt.from_json(artifacts.read_json_object(path / "publication.json"))
    if (
        receipt.study_identity_sha256 != binding.study_identity_sha256
        or (receipt.repeat_id, receipt.fold_id) != (repeat_id, fold_id)
        or receipt.run_provenance_identity_sha256 != binding.payload["run_provenance_identity_sha256"]
        or receipt.immutable_reference_sha256 != binding.payload["immutable_reference_sha256"]
        or receipt.protocol_sha256 != binding.payload["protocol_identity_sha256"]
        or receipt.seed_manifest_identity_sha256 != binding.payload["seed_manifest_identity_sha256"]
        or receipt.outer_manifest_identity_sha256 != binding.payload["outer_manifest_identity_sha256"]
        or receipt.inner_manifest_identity_sha256 != binding.payload["inner_manifest_identities_sha256"].get(_coordinate_key(repeat_id, fold_id))
        or receipt.fold_authority_identity_sha256 != binding.payload["coordinate_fold_authority_sha256"].get(_coordinate_key(repeat_id, fold_id))
        or receipt.aligned_data_content_identity_sha256 != binding.payload["aligned_data_content_identity_sha256"]
    ):
        raise ValueError("Scoring publication does not belong to this exact study coordinate.")
    for name in required.difference({"publication.json"}):
        artifacts.read_json_object(path / name)
    search = artifacts.read_json_object(path / "selected_search_summary.json")
    calibration = artifacts.read_json_object(path / "calibration_threshold_summary.json")
    final_refit = artifacts.read_json_object(path / "final_refit_summary.json")
    if (
        search.get("search_selection_identity_sha256") != receipt.search_selection_identity_sha256
        or search.get("selected_candidate_identity_sha256") != receipt.selected_candidate_identity_sha256
        or search.get("selected_oof_predictions_sha256") != receipt.selected_oof_predictions_sha256
        or calibration.get("cross_fitted_calibration_sha256") != receipt.cross_fitted_calibration_sha256
        or calibration.get("final_sigmoid_identity_sha256") != receipt.final_sigmoid_identity_sha256
        or calibration.get("threshold_identity_sha256") != receipt.threshold_identity_sha256
        or final_refit.get("candidate_identity_sha256") != receipt.selected_candidate_identity_sha256
        or final_refit.get("final_model_identity_sha256") != receipt.final_model_identity_sha256
        or final_refit.get("frozen_model_state_sha256") != receipt.frozen_model_state_sha256
    ):
        raise ValueError("Scoring publication summaries do not match their immutable receipt.")
    return receipt


def _validate_evaluation_directory(path: Path, binding: StudyBinding, receipt: ScoringPublicationReceipt, repeat_id: int, fold_id: int) -> None:
    required = {"publication.json", "evaluation.json"}
    present = {item.name for item in path.iterdir()} if path.is_dir() else set()
    if present != required:
        raise ValueError("Evaluation publication is incomplete, malformed, or contains unexpected evidence.")
    publication = artifacts.read_json_object(path / "publication.json")
    evaluation = artifacts.read_json_object(path / "evaluation.json")
    content = {
        "schema_version": EVALUATION_PUBLICATION_SCHEMA_VERSION,
        "study_identity_sha256": binding.study_identity_sha256,
        "coordinate": _coordinate_payload(repeat_id, fold_id),
        "scoring_publication_identity_sha256": receipt.scoring_publication_identity_sha256,
        "evaluation_payload_sha256": artifacts.payload_sha256(evaluation),
    }
    expected = {**content, "evaluation_publication_identity_sha256": artifacts.payload_sha256(content)}
    if publication != expected:
        raise ValueError("Evaluation publication does not verify against the scoring publication.")


def classify_coordinate(study_directory: Path, binding: StudyBinding, repeat_id: int, fold_id: int) -> dict[str, str]:
    _validate_coordinate(repeat_id, fold_id)
    coordinate = _coordinate_directory(study_directory, repeat_id, fold_id)
    scoring = coordinate / "scoring"
    evaluation = coordinate / "evaluation"
    if not scoring.exists():
        if evaluation.exists():
            raise ValueError("Evaluation evidence exists without a scoring publication.")
        return {"state": "PENDING", "resume_classification": "FULL_COORDINATE_RECOMPUTE"}
    receipt, _ = _load_scoring_publication(study_directory, binding, repeat_id, fold_id)
    if not evaluation.exists():
        return {"state": "SCORING_PUBLISHED", "resume_classification": "EVALUATION_ONLY_RESUME"}
    _validate_evaluation_directory(evaluation, binding, receipt, repeat_id, fold_id)
    return {"state": "EVALUATION_PUBLISHED", "resume_classification": "COMPLETE"}


def reconstruct_runtime_state(study_directory: Path, binding: StudyBinding) -> dict[str, Any]:
    """Derive completion only from immutable evidence, never from cached state."""
    validate_study_directory(study_directory, binding)
    coordinates = []
    for coordinate in binding.payload["coordinates"]:
        repeat_id, fold_id = coordinate["repeat_id"], coordinate["fold_id"]
        coordinates.append({**coordinate, **classify_coordinate(study_directory, binding, repeat_id, fold_id)})
    complete = sum(item["resume_classification"] == "COMPLETE" for item in coordinates)
    derived_state = "COMPLETE" if complete == len(coordinates) else ("PREFLIGHTED" if complete == 0 and all(item["state"] == "PENDING" for item in coordinates) else "RUNNING")
    return {
        "schema_version": "controlled-primary-v1-runtime-state-v1",
        "study_identity_sha256": binding.study_identity_sha256,
        "derived_state": derived_state,
        "coordinate_count": len(coordinates),
        "complete_coordinate_count": complete,
        "coordinates": coordinates,
    }


def write_runtime_state(
    study_directory: Path,
    binding: StudyBinding,
    lifecycle_state: str | None = None,
    executing_coordinate: tuple[int, int] | None = None,
) -> dict[str, Any]:
    state = reconstruct_runtime_state(study_directory, binding) if (study_directory / "study_binding.json").is_file() else {
        "schema_version": "controlled-primary-v1-runtime-state-v1",
        "study_identity_sha256": binding.study_identity_sha256,
        "derived_state": "NEW",
        "coordinate_count": 25,
        "complete_coordinate_count": 0,
        "coordinates": [],
    }
    if lifecycle_state is not None:
        if lifecycle_state not in _STUDY_STATES:
            raise ValueError("Unknown controlled-runner lifecycle state.")
        state["recorded_lifecycle_state"] = lifecycle_state
    if executing_coordinate is not None:
        repeat_id, fold_id = executing_coordinate
        _validate_coordinate(repeat_id, fold_id)
        for coordinate in state["coordinates"]:
            if (coordinate["repeat_id"], coordinate["fold_id"]) == executing_coordinate:
                if coordinate["resume_classification"] == "COMPLETE":
                    raise ValueError("A completed coordinate cannot be marked executing.")
                coordinate["state"] = "EXECUTING"
                break
        else:
            raise ValueError("Executing coordinate is absent from the frozen coordinate plan.")
    artifacts.atomic_write_json(study_directory / "runtime_state.json", state)
    return state


def record_failure(
    study_directory: Path,
    binding: StudyBinding,
    *,
    stage: str,
    repeat_id: int | None,
    fold_id: int | None,
    error: BaseException,
) -> Path:
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("Failure evidence requires a stage.")
    if (repeat_id is None) != (fold_id is None):
        raise ValueError("Failure evidence coordinate must be complete or absent.")
    coordinate = None
    fold_identity = None
    if repeat_id is not None and fold_id is not None:
        _validate_coordinate(repeat_id, fold_id)
        coordinate = _coordinate_payload(repeat_id, fold_id)
        fold_identity = binding.payload["coordinate_fold_authority_sha256"][_coordinate_key(repeat_id, fold_id)]
    sanitized = " ".join(str(error).split())[:500]
    trace_hash = artifacts.payload_sha256({"traceback": traceback.format_exc()})
    attempt_id = uuid.uuid4().hex
    path = study_directory / "failures" / f"{attempt_id}.json"
    artifacts.create_immutable_json(path, {
        "schema_version": "controlled-primary-v1-failure-v1",
        "attempt_id": attempt_id,
        "timestamp_utc": _utc_now(),
        "stage": stage,
        "coordinate": coordinate,
        "exception_type": type(error).__name__,
        "sanitized_message": sanitized,
        "traceback_sha256": trace_hash,
        "study_identity_sha256": binding.study_identity_sha256,
        "fold_authority_identity_sha256": fold_identity,
        "environment_compatibility_identity_sha256": binding.payload["environment_compatibility_identity_sha256"],
    })
    return path


@dataclass(frozen=True)
class CoordinateExecutionResult:
    repeat_id: int
    fold_id: int
    resume_classification: str
    resulting_classification: str
    scoring_publication_identity_sha256: str | None
    evaluation_publication_identity_sha256: str | None


def _canonical_coordinate_records(prepared: PreparedStudy, repeat_id: int, fold_id: int) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    outer = [record for record in prepared.outer_manifest["folds"] if (record["repeat_id"], record["fold_id"]) == (repeat_id, fold_id)]
    inner = [record for record in prepared.inner_manifest["outer_folds"] if (record["repeat_id"], record["fold_id"]) == (repeat_id, fold_id)]
    if len(outer) != 1 or len(inner) != 1:
        raise ValueError("Canonical fold manifests do not contain exactly one requested coordinate.")
    return outer[0], inner[0]


def _canonical_modalities(prepared: PreparedStudy, sample_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    ids = tuple(str(value) for value in sample_ids)
    matrices: dict[str, Any] = {}
    contracts: dict[str, tuple[str, ...]] = {}
    mapping = {"mGE": ("X_rna", "rna"), "mDM": ("X_dna", "dna"), "mCNA": ("X_cna", "cna")}
    for modality in PRIMARY_MODALITIES:
        matrix_key, feature_key = mapping[modality]
        names = tuple(prepared.aligned_data["feature_columns"][feature_key])
        matrices[modality] = prepared.aligned_data[matrix_key].loc[list(ids), list(names)].copy()
        contracts[modality] = names
    return matrices, contracts


def _canonical_labels(prepared: PreparedStudy, sample_ids: Sequence[str]) -> np.ndarray:
    labels_by_id = dict(zip(prepared.aligned_data["sample_ids"], np.asarray(prepared.aligned_data["y_binary"], dtype=int).tolist()))
    try:
        values = np.asarray([labels_by_id[str(sample_id)] for sample_id in sample_ids], dtype=int)
    except KeyError as error:
        raise ValueError("Canonical coordinate contains a SAMPLE_ID absent from aligned data.") from error
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("Canonical coordinate labels are invalid.")
    return values


def _frozen_predictions_json(frozen: FrozenOuterPredictions) -> dict[str, Any]:
    validate_frozen_outer_predictions(frozen)
    return {
        "schema_version": "controlled-primary-v1-frozen-predictions-v1",
        "predictions": [asdict(prediction) for prediction in frozen.predictions],
        "patient_ids_hash": frozen.patient_ids_hash,
        "prediction_hash": frozen.prediction_hash,
        "evidence": dict(frozen.evidence),
        "candidate_id": frozen.candidate_id,
        "candidate_identity_sha256": frozen.candidate_identity_sha256,
        "final_model_identity_sha256": frozen.final_model_identity_sha256,
        "search_selection_identity_sha256": frozen.search_selection_identity_sha256,
    }


def _frozen_predictions_from_json(payload: Mapping[str, Any]) -> FrozenOuterPredictions:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "controlled-primary-v1-frozen-predictions-v1":
        raise ValueError("Frozen prediction publication schema is invalid.")
    values = payload.get("predictions")
    if not isinstance(values, list):
        raise ValueError("Frozen prediction publication records are invalid.")
    try:
        frozen = FrozenOuterPredictions(
            tuple(OuterPrediction(**dict(record)) for record in values),
            payload["patient_ids_hash"],
            payload["prediction_hash"],
            payload["evidence"],
            payload["candidate_id"],
            payload["candidate_identity_sha256"],
            payload["final_model_identity_sha256"],
            payload["search_selection_identity_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Frozen prediction publication cannot be reconstructed.") from error
    if payload != _frozen_predictions_json(frozen):
        raise ValueError("Frozen prediction publication contains unexpected or altered content.")
    return frozen


def _search_summary(search_result: Any) -> dict[str, Any]:
    selected = search_result.selected_search
    return {
        "schema_version": "controlled-primary-v1-search-summary-v1",
        "repeat_id": search_result.context.repeat_id,
        "fold_id": search_result.context.fold_id,
        "search_selection_identity_sha256": selected.search_selection_identity_sha256,
        "selected_candidate": asdict(selected.selected_candidate),
        "selected_candidate_identity_sha256": selected.selected_candidate_identity_sha256,
        "selected_oof_predictions_sha256": selected.selected_oof_predictions_sha256,
        "all_candidate_summaries": [
            {
                "candidate_id": summary.candidate_id,
                "candidate_identity_sha256": summary.candidate_identity_sha256,
                "mean_inner_auprc": summary.mean_inner_auprc,
                "mean_inner_auroc": summary.mean_inner_auroc,
                "inner_auprc_sd": summary.inner_auprc_sd,
                "identity_sha256": summary.identity_sha256,
            }
            for summary in selected.all_candidate_summaries
        ],
    }


def _calibration_threshold_summary(calibration: Any, final_calibrator: Any, threshold: Any) -> dict[str, Any]:
    return {
        "schema_version": "controlled-primary-v1-calibration-threshold-summary-v1",
        "cross_fitted_calibration_sha256": calibration.cross_fitted_calibration_sha256,
        "final_sigmoid_identity_sha256": final_calibrator.final_calibrator_identity_sha256,
        "threshold_identity_sha256": threshold.threshold_identity_sha256,
        "threshold": float(threshold.threshold),
        "threshold_metrics": asdict(threshold.metrics),
    }


def _final_refit_summary(bundle: Any) -> dict[str, Any]:
    return {
        "schema_version": "controlled-primary-v1-final-refit-summary-v1",
        "candidate_id": bundle.candidate_id,
        "candidate_identity_sha256": bundle.candidate_identity_sha256,
        "final_model_identity_sha256": bundle.final_model_identity_sha256,
        "frozen_model_state_sha256": bundle.evidence["frozen_model_state_sha256"],
        "evidence": dict(bundle.evidence),
    }


def _make_scoring_receipt(prepared: PreparedStudy, search_result: Any, calibration: Any, final_calibrator: Any, threshold: Any, bundle: Any, frozen: FrozenOuterPredictions) -> ScoringPublicationReceipt:
    context, selected = search_result.context, search_result.selected_search
    return ScoringPublicationReceipt(
        prepared.binding.study_identity_sha256,
        context.repeat_id,
        context.fold_id,
        prepared.provenance.identity_sha256,
        prepared.binding.payload["immutable_reference_sha256"],
        prepared.protocol.identity_sha256,
        prepared.seed_manifest.identity_sha256,
        context.outer_manifest_identity_sha256,
        context.inner_manifest_identity_sha256,
        context.fold_authority_identity_sha256,
        selected.search_selection_identity_sha256,
        selected.selected_candidate_identity_sha256,
        selected.selected_oof_predictions_sha256,
        calibration.cross_fitted_calibration_sha256,
        final_calibrator.final_calibrator_identity_sha256,
        threshold.threshold_identity_sha256,
        bundle.final_model_identity_sha256,
        bundle.evidence["frozen_model_state_sha256"],
        prepared.provenance.aligned_data_content_identity_sha256,
        ordered_patient_ids_sha256(context.outer_testing_ids),
        frozen.prediction_hash,
    )


def _load_scoring_publication(study_directory: Path, binding: StudyBinding, repeat_id: int, fold_id: int) -> tuple[ScoringPublicationReceipt, FrozenOuterPredictions]:
    receipt = _validate_scoring_directory(_coordinate_directory(study_directory, repeat_id, fold_id) / "scoring", binding, repeat_id, fold_id)
    frozen = _frozen_predictions_from_json(artifacts.read_json_object(_coordinate_directory(study_directory, repeat_id, fold_id) / "scoring" / "frozen_predictions.json"))
    if (
        frozen.prediction_hash != receipt.frozen_prediction_hash
        or frozen.patient_ids_hash != receipt.ordered_outer_test_ids_sha256
        or frozen.candidate_identity_sha256 != receipt.selected_candidate_identity_sha256
        or frozen.final_model_identity_sha256 != receipt.final_model_identity_sha256
        or frozen.search_selection_identity_sha256 != receipt.search_selection_identity_sha256
        or frozen.evidence.get("frozen_model_state_sha256") != receipt.frozen_model_state_sha256
    ):
        raise ValueError("Frozen predictions do not verify against the scoring publication receipt.")
    return receipt, frozen


def _publish_scoring(
    prepared: PreparedStudy,
    receipt: ScoringPublicationReceipt,
    frozen: FrozenOuterPredictions,
    search_result: Any,
    calibration: Any,
    final_calibrator: Any,
    threshold: Any,
    bundle: Any,
) -> None:
    target = _coordinate_directory(prepared.study_directory, receipt.repeat_id, receipt.fold_id) / "scoring"
    documents = {
        "publication.json": receipt.as_json(),
        "frozen_predictions.json": _frozen_predictions_json(frozen),
        "selected_search_summary.json": _search_summary(search_result),
        "calibration_threshold_summary.json": _calibration_threshold_summary(calibration, final_calibrator, threshold),
        "final_refit_summary.json": _final_refit_summary(bundle),
    }
    artifacts.publish_directory(target, lambda temporary: [artifacts.create_immutable_json(temporary / name, payload) for name, payload in documents.items()])


def _evaluation_payload(prepared: PreparedStudy, receipt: ScoringPublicationReceipt, frozen: FrozenOuterPredictions, evaluation: Any) -> dict[str, Any]:
    context_record, _ = _canonical_coordinate_records(prepared, receipt.repeat_id, receipt.fold_id)
    labels = _canonical_labels(prepared, context_record["test_sample_ids"])
    search_summary = artifacts.read_json_object(
        _coordinate_directory(prepared.study_directory, receipt.repeat_id, receipt.fold_id) / "scoring" / "selected_search_summary.json"
    )
    return {
        "schema_version": "controlled-primary-v1-evaluation-result-v1",
        "repeat_id": receipt.repeat_id,
        "fold_id": receipt.fold_id,
        "outer_evaluation": asdict(evaluation),
        "aggregation_input": {
            "repeat_id": receipt.repeat_id,
            "fold_id": receipt.fold_id,
            "ordered_outer_test_ids": list(context_record["test_sample_ids"]),
            "outer_test_class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
            "selected_candidate": search_summary["selected_candidate"],
            "selected_candidate_identity_sha256": receipt.selected_candidate_identity_sha256,
            "all_candidate_search_summaries": search_summary["all_candidate_summaries"],
            "search_selection_identity_sha256": receipt.search_selection_identity_sha256,
            "raw_score_auprc": evaluation.primary_ranking_metrics.auprc,
            "raw_score_auroc": evaluation.primary_ranking_metrics.auroc,
            "frozen_threshold_metrics": asdict(evaluation.operational_metrics),
            "brier_score": evaluation.brier_score,
            "log_loss": evaluation.log_loss,
            "calibrated_probability_diagnostic": dict(evaluation.secondary_diagnostics),
            "cross_fitted_calibration_sha256": receipt.cross_fitted_calibration_sha256,
            "threshold_identity_sha256": receipt.threshold_identity_sha256,
            "final_model_identity_sha256": receipt.final_model_identity_sha256,
            "frozen_prediction_hash": frozen.prediction_hash,
            "scoring_publication_identity_sha256": receipt.scoring_publication_identity_sha256,
        },
    }


def _publish_evaluation(prepared: PreparedStudy, receipt: ScoringPublicationReceipt, frozen: FrozenOuterPredictions, evaluation: Any) -> str:
    payload = _evaluation_payload(prepared, receipt, frozen, evaluation)
    content = {
        "schema_version": EVALUATION_PUBLICATION_SCHEMA_VERSION,
        "study_identity_sha256": prepared.binding.study_identity_sha256,
        "coordinate": _coordinate_payload(receipt.repeat_id, receipt.fold_id),
        "scoring_publication_identity_sha256": receipt.scoring_publication_identity_sha256,
        "evaluation_payload_sha256": artifacts.payload_sha256(payload),
    }
    publication = {**content, "evaluation_publication_identity_sha256": artifacts.payload_sha256(content)}
    target = _coordinate_directory(prepared.study_directory, receipt.repeat_id, receipt.fold_id) / "evaluation"
    artifacts.publish_directory(target, lambda temporary: [artifacts.create_immutable_json(temporary / "publication.json", publication), artifacts.create_immutable_json(temporary / "evaluation.json", payload)])
    return publication["evaluation_publication_identity_sha256"]


def _validate_search_coordinate(prepared: PreparedStudy, repeat_id: int, fold_id: int, search_result: Any) -> None:
    validate_primary_search_result(search_result, run_provenance=prepared.provenance, aligned_data=prepared.aligned_data)
    outer, inner = _canonical_coordinate_records(prepared, repeat_id, fold_id)
    context = search_result.context
    expected_inner = tuple(
        {
            "inner_fold_id": record["inner_fold_id"],
            "inner_train_sample_ids": tuple(record["inner_train_sample_ids"]),
            "inner_validation_sample_ids": tuple(record["inner_validation_sample_ids"]),
        }
        for record in sorted(inner["inner_folds"], key=lambda value: value["inner_fold_id"])
    )
    if (
        (context.repeat_id, context.fold_id) != (repeat_id, fold_id)
        or context.outer_training_ids != tuple(outer["train_sample_ids"])
        or context.outer_testing_ids != tuple(outer["test_sample_ids"])
        or context.inner_folds != expected_inner
        or context.outer_manifest_identity_sha256 != prepared.binding.payload["outer_manifest_identity_sha256"]
        or context.inner_manifest_identity_sha256 != prepared.binding.payload["inner_manifest_identities_sha256"][_coordinate_key(repeat_id, fold_id)]
        or context.fold_authority_identity_sha256 != prepared.binding.payload["coordinate_fold_authority_sha256"][_coordinate_key(repeat_id, fold_id)]
        or context.run_provenance_identity_sha256 != prepared.provenance.identity_sha256
    ):
        raise ValueError("Public search result does not belong to the canonical controlled coordinate.")


def _evaluate_published_coordinate(prepared: PreparedStudy, repeat_id: int, fold_id: int) -> tuple[ScoringPublicationReceipt, str]:
    receipt, frozen = _load_scoring_publication(prepared.study_directory, prepared.binding, repeat_id, fold_id)
    outer, _ = _canonical_coordinate_records(prepared, repeat_id, fold_id)
    test_ids = tuple(outer["test_sample_ids"])
    labels = _canonical_labels(prepared, test_ids)
    evaluation = evaluate_outer_test_predictions(
        frozen,
        labels,
        test_ids,
        run_provenance=prepared.provenance,
        aligned_data=prepared.aligned_data,
        scoring_publication_receipt=receipt.as_json(),
    )
    return receipt, _publish_evaluation(prepared, receipt, frozen, evaluation)


def execute_coordinate(prepared: PreparedStudy, coordinate: tuple[int, int]) -> CoordinateExecutionResult:
    """Execute one canonical coordinate through public scientific authorities only."""
    repeat_id, fold_id = coordinate
    _validate_coordinate(repeat_id, fold_id)
    if coordinate not in primary_coordinate_plan(prepared.protocol):
        raise ValueError("Coordinate is absent from the frozen Primary V1 plan.")
    validate_study_directory(prepared.study_directory, prepared.binding)
    classification = classify_coordinate(prepared.study_directory, prepared.binding, repeat_id, fold_id)["resume_classification"]
    if classification == "COMPLETE":
        receipt, _ = _load_scoring_publication(prepared.study_directory, prepared.binding, repeat_id, fold_id)
        evaluation_publication = artifacts.read_json_object(_coordinate_directory(prepared.study_directory, repeat_id, fold_id) / "evaluation" / "publication.json")
        return CoordinateExecutionResult(repeat_id, fold_id, classification, "COMPLETE", receipt.scoring_publication_identity_sha256, evaluation_publication["evaluation_publication_identity_sha256"])
    try:
        if classification == "EVALUATION_ONLY_RESUME":
            receipt, evaluation_identity = _evaluate_published_coordinate(prepared, repeat_id, fold_id)
            write_runtime_state(prepared.study_directory, prepared.binding, "RUNNING")
            return CoordinateExecutionResult(repeat_id, fold_id, classification, "COMPLETE", receipt.scoring_publication_identity_sha256, evaluation_identity)
        write_runtime_state(prepared.study_directory, prepared.binding, "RUNNING", coordinate)
        search_result = run_primary_inner_search(
            run_provenance=prepared.provenance,
            aligned_data=prepared.aligned_data,
            repeat_id=repeat_id,
            fold_id=fold_id,
            ae_training_config=prepared.protocol.make_autoencoder_training_config(),
            ctgan_config=prepared.protocol.make_ctgan_config(),
            ae_validation_fraction=prepared.protocol.ae_validation_fraction,
            synthetic_namespace_prefix=f"controlled:{prepared.binding.study_identity_sha256[:12]}:r{repeat_id:02d}:f{fold_id:02d}",
        )
        _validate_search_coordinate(prepared, repeat_id, fold_id, search_result)
        calibration = cross_fit_sigmoid_calibration(search_result, run_provenance=prepared.provenance, aligned_data=prepared.aligned_data)
        threshold = select_operational_threshold(calibration, search_result=search_result, run_provenance=prepared.provenance, aligned_data=prepared.aligned_data)
        final_calibrator = fit_final_sigmoid_calibrator(search_result.selected_search.selected_oof_predictions, selected_search=search_result.selected_search, context=search_result.context)
        train_ids, test_ids = search_result.context.outer_training_ids, search_result.context.outer_testing_ids
        train_modalities, contracts = _canonical_modalities(prepared, train_ids)
        test_modalities, _ = _canonical_modalities(prepared, test_ids)
        bundle = fit_final_primary_model(
            train_modalities,
            train_ids,
            _canonical_labels(prepared, train_ids),
            test_modalities,
            test_ids,
            contracts,
            prepared.protocol.make_autoencoder_training_config(),
            prepared.protocol.make_ctgan_config(),
            search_result.context.seed_manifest.final_refit_seed_book(),
            final_calibrator,
            threshold,
            search_result=search_result,
            cross_fitted_calibration=calibration,
            protocol=prepared.protocol,
            synthetic_namespace=f"controlled:{prepared.binding.study_identity_sha256[:12]}:r{repeat_id:02d}:f{fold_id:02d}:final",
            run_provenance=prepared.provenance,
            aligned_data=prepared.aligned_data,
            ae_validation_fraction=prepared.protocol.ae_validation_fraction,
        )
        validate_final_primary_model_bundle(bundle, search_result=search_result, run_provenance=prepared.provenance, aligned_data=prepared.aligned_data)
        frozen = score_outer_test(bundle, test_modalities, test_ids, search_result=search_result, run_provenance=prepared.provenance, aligned_data=prepared.aligned_data)
        validate_frozen_outer_predictions(frozen)
        receipt = _make_scoring_receipt(prepared, search_result, calibration, final_calibrator, threshold, bundle, frozen)
        _publish_scoring(prepared, receipt, frozen, search_result, calibration, final_calibrator, threshold, bundle)
        _, evaluation_identity = _evaluate_published_coordinate(prepared, repeat_id, fold_id)
        write_runtime_state(prepared.study_directory, prepared.binding, "RUNNING")
        return CoordinateExecutionResult(repeat_id, fold_id, classification, "COMPLETE", receipt.scoring_publication_identity_sha256, evaluation_identity)
    except Exception as error:
        record_failure(prepared.study_directory, prepared.binding, stage="coordinate_execution", repeat_id=repeat_id, fold_id=fold_id, error=error)
        write_runtime_state(prepared.study_directory, prepared.binding, "FAILED")
        raise


def run_study(prepared: PreparedStudy) -> tuple[CoordinateExecutionResult, ...]:
    """Sequentially run only the work required by immutable coordinate state."""
    validate_study_directory(prepared.study_directory, prepared.binding)
    lock = acquire_study_lock(prepared.study_directory, prepared.binding)
    outcome = "completed"
    results: list[CoordinateExecutionResult] = []
    try:
        for coordinate in primary_coordinate_plan(prepared.protocol):
            results.append(execute_coordinate(prepared, coordinate))
        write_runtime_state(prepared.study_directory, prepared.binding, "COMPLETE")
        return tuple(results)
    except Exception:
        outcome = "failed"
        raise
    finally:
        artifacts.release_run_lock(lock, outcome=outcome)
