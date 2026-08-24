"""Frozen Primary Proposed Method v1 identities, modality binding, and seeds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import payload_sha256
from .folds import FoldProtocol, build_outer_data_fingerprint


PRIMARY_MODALITIES = ("mGE", "mDM", "mCNA")


def _patient_ids(values: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(str(value) for value in values)
    if not ids or len(ids) != len(set(ids)) or any(not value.strip() for value in ids):
        raise ValueError("Patient IDs must be non-empty, unique, and non-blank.")
    return ids


def patient_set_sha256(values: Sequence[str]) -> str:
    """Order-independent identity for membership comparisons only."""
    return payload_sha256(sorted(_patient_ids(values)))


def ordered_patient_ids_sha256(values: Sequence[str]) -> str:
    """Order-sensitive identity for aligned matrices and prediction records."""
    return payload_sha256(list(_patient_ids(values)))


@dataclass(frozen=True)
class ModalityBinding:
    canonical_modality: str
    matrix_key: str
    feature_key: str
    source_schema_sha256: str
    source_audit_identity: str | None = None


@dataclass(frozen=True)
class ModalityAdapter:
    bindings: Mapping[str, ModalityBinding]

    def __post_init__(self) -> None:
        expected = {"mGE": ("X_rna", "rna"), "mDM": ("X_dna", "dna"), "mCNA": ("X_cna", "cna")}
        if set(self.bindings) != set(PRIMARY_MODALITIES):
            raise ValueError("ModalityAdapter must contain exactly mGE, mDM, and mCNA.")
        copied: dict[str, ModalityBinding] = {}
        for modality in PRIMARY_MODALITIES:
            binding = self.bindings[modality]
            if not isinstance(binding, ModalityBinding) or binding.canonical_modality != modality:
                raise ValueError("ModalityAdapter binding does not match its canonical modality.")
            if (binding.matrix_key, binding.feature_key) != expected[modality]:
                raise ValueError("ModalityAdapter rejects swapped or unexpected data mappings.")
            if not isinstance(binding.source_schema_sha256, str) or len(binding.source_schema_sha256) != 64:
                raise ValueError("ModalityAdapter requires a source feature-schema SHA-256.")
            copied[modality] = binding
        if len({(item.matrix_key, item.feature_key) for item in copied.values()}) != 3:
            raise ValueError("ModalityAdapter mappings must be unique.")
        object.__setattr__(self, "bindings", MappingProxyType(copied))

    @classmethod
    def from_aligned_data(cls, aligned_data: Mapping[str, Any]) -> "ModalityAdapter":
        if not isinstance(aligned_data, Mapping) or not isinstance(aligned_data.get("feature_columns"), Mapping):
            raise ValueError("ModalityAdapter requires validated aligned data and feature columns.")
        feature_columns = aligned_data["feature_columns"]
        expected = {"mGE": ("X_rna", "rna"), "mDM": ("X_dna", "dna"), "mCNA": ("X_cna", "cna")}
        bindings = {}
        for modality, (matrix_key, feature_key) in expected.items():
            matrix = aligned_data.get(matrix_key)
            names = feature_columns.get(feature_key)
            if matrix is None or names is None or tuple(matrix.columns) != tuple(names):
                raise ValueError("Aligned data does not match the explicit modality adapter contract.")
            bindings[modality] = ModalityBinding(modality, matrix_key, feature_key, payload_sha256(list(names)))
        return cls(bindings)

    def payload(self) -> dict[str, Any]:
        return {modality: {"matrix_key": self.bindings[modality].matrix_key, "feature_key": self.bindings[modality].feature_key, "source_schema_sha256": self.bindings[modality].source_schema_sha256, "source_audit_identity": self.bindings[modality].source_audit_identity} for modality in PRIMARY_MODALITIES}

    @property
    def identity_sha256(self) -> str:
        return payload_sha256(self.payload())


@dataclass(frozen=True)
class PrimaryProtocolV1:
    protocol_id: str = "primary-protocol-v1"
    outer_n_splits: int = 5
    outer_n_repeats: int = 5
    inner_n_splits: int = 3
    modalities: tuple[str, ...] = PRIMARY_MODALITIES
    ae_hidden_width: int = 128
    ae_max_epochs: int = 50
    ae_batch_size: int = 32
    ae_patience: int = 10
    ae_validation_fraction: float = 0.20
    ae_ratios: tuple[float, ...] = (0.25, 0.50, 0.75)
    logistic_cs: tuple[float, ...] = (0.1, 1.0, 10.0)
    augmentation: str = "minority_only_ctgan"
    ctgan_epochs: int = 300
    ctgan_pac: int = 10
    ctgan_verbose: bool = False
    class_weight: None = None
    calibration: str = "sigmoid"
    threshold_objective: str = "balanced_accuracy"
    feature_provenance_status: str = "UNKNOWN"

    def __post_init__(self) -> None:
        expected = PrimaryProtocolV1.__dataclass_fields__
        required = {"protocol_id": "primary-protocol-v1", "outer_n_splits": 5, "outer_n_repeats": 5, "inner_n_splits": 3, "modalities": PRIMARY_MODALITIES, "ae_hidden_width": 128, "ae_max_epochs": 50, "ae_batch_size": 32, "ae_patience": 10, "ae_validation_fraction": .20, "ae_ratios": (0.25, 0.50, 0.75), "logistic_cs": (0.1, 1.0, 10.0), "augmentation": "minority_only_ctgan", "ctgan_epochs": 300, "ctgan_pac": 10, "ctgan_verbose": False, "class_weight": None, "calibration": "sigmoid", "threshold_objective": "balanced_accuracy", "feature_provenance_status": "UNKNOWN"}
        if any(getattr(self, name) != value for name, value in required.items()):
            raise ValueError("PrimaryProtocolV1 values are frozen and cannot be changed.")

    def payload(self) -> dict[str, Any]:
        return {"schema_version": "primary-protocol-v1", "protocol_id": self.protocol_id, "outer_n_splits": 5, "outer_n_repeats": 5, "inner_n_splits": 3, "modalities": list(self.modalities), "ae": {"hidden_width": 128, "max_epochs": 50, "batch_size": 32, "patience": 10, "validation_fraction": .20}, "ae_ratios": list(self.ae_ratios), "logistic_cs": list(self.logistic_cs), "augmentation": self.augmentation, "ctgan": {"epochs": 300, "pac": 10, "verbose": False}, "class_weight": None, "calibration": self.calibration, "threshold_objective": self.threshold_objective, "selection_rule": ["mean_inner_fold_auprc", "mean_inner_fold_auroc", "lowest_inner_fold_auprc_std", "smaller_fused_latent_width", "smaller_logistic_c", "canonical_candidate_id"], "threshold_tie_break": ["highest_balanced_accuracy", "highest_sensitivity_high_tmb", "closest_to_0.5", "smaller_threshold"], "threshold_candidates": "sorted({0.0} union unique_cross_fitted_probabilities)", "prediction_convention": "calibrated_probability >= threshold", "feature_provenance_status": "UNKNOWN"}

    @property
    def identity_sha256(self) -> str:
        return payload_sha256(self.payload())

    @property
    def seed_policy_identity_sha256(self) -> str:
        """Identity of the prospective Primary V1 seed derivation contract."""
        return payload_sha256({"schema_version": "primary-seed-policy-v1", "protocol_sha256": self.identity_sha256, "derivation": "sha256(canonical-json({root_seed,binding}))[:4]-big-endian", "component_keys": sorted(PrimarySeedManifest.primary_expected_keys()), "fold_settings": {"outer_n_splits": 5, "outer_n_repeats": 5, "inner_n_splits": 3}})

    def validate_fold_protocol(self, fold_protocol: FoldProtocol) -> None:
        if not isinstance(fold_protocol, FoldProtocol) or (fold_protocol.outer_n_splits, fold_protocol.outer_n_repeats, fold_protocol.inner_n_splits) != (5, 5, 3):
            raise ValueError("PrimaryProtocolV1 requires exactly 5 outer folds, 5 repeats, and 3 inner folds.")

    def make_fold_protocol(self, seed_manifest: "PrimarySeedManifest", provenance: "PrimaryV1RunProvenance") -> FoldProtocol:
        """Build the generic deterministic fold configuration from Primary V1 root seeds."""
        self.validate_primary_seed_manifest(seed_manifest, provenance)
        return FoldProtocol(5, 5, seed_manifest.require("outer:fold_generation"), 3, seed_manifest.require("inner:fold_generation"))

    def fold_protocol_identity_sha256(self, fold_protocol: FoldProtocol, seed_manifest: "PrimarySeedManifest" | str, provenance: "PrimaryV1RunProvenance" | str | None = None) -> str:
        self.validate_fold_protocol(fold_protocol)
        manifest_identity = seed_manifest if isinstance(seed_manifest, str) else seed_manifest.identity_sha256
        if not isinstance(manifest_identity, str) or len(manifest_identity) != 64:
            raise ValueError("Primary fold binding requires a seed-manifest SHA-256.")
        if provenance is not None:
            if isinstance(provenance, PrimaryV1RunProvenance):
                provenance_identity = provenance.identity_sha256
            elif isinstance(provenance, str) and len(provenance) == 64:
                provenance_identity = provenance
            else:
                raise TypeError("Primary fold binding requires PrimaryV1RunProvenance.")
        else:
            provenance_identity = None
        return payload_sha256({"schema_version": "primary-fold-protocol-binding-v1", "protocol_sha256": self.identity_sha256, "seed_manifest_identity_sha256": manifest_identity, "run_provenance_identity_sha256": provenance_identity, "fold_protocol": fold_protocol.as_dict()})

    def validate_primary_seed_manifest(self, seed_manifest: "PrimarySeedManifest", provenance: "PrimaryV1RunProvenance") -> None:
        if not isinstance(seed_manifest, PrimarySeedManifest):
            raise TypeError("PrimaryProtocolV1 requires PrimarySeedManifest.")
        expected = PrimarySeedManifest.primary_expected_keys()
        if set(seed_manifest.seeds) != expected:
            raise ValueError("PrimarySeedManifest does not contain exactly the Primary V1 seed material.")
        seed_manifest.validate_primary_derivation(self, provenance)

    def make_autoencoder_training_config(self) -> Any:
        from .autoencoder import AutoencoderTrainingConfig
        return AutoencoderTrainingConfig(50, 32, 10)

    def make_ctgan_config(self) -> Any:
        from .ctgan import CTGANConfig
        return CTGANConfig(300, 10, False)

    def validate_autoencoder_training_config(self, config: Any, validation_fraction: float) -> None:
        expected = self.make_autoencoder_training_config()
        if config != expected or float(validation_fraction) != .20:
            raise ValueError("PrimaryProtocolV1 rejects non-V1 AE training configuration.")

    def validate_ctgan_config(self, config: Any) -> None:
        if config != self.make_ctgan_config():
            raise ValueError("PrimaryProtocolV1 rejects non-V1 CTGAN configuration.")


_PRIMARY_MODALITY_MAPPING = {
    "mGE": {"matrix_key": "X_rna", "feature_key": "rna", "source_audit_key": "mGE"},
    "mDM": {"matrix_key": "X_dna", "feature_key": "dna", "source_audit_key": "mDM"},
    "mCNA": {"matrix_key": "X_cna", "feature_key": "cna", "source_audit_key": "CNA"},
}


def aligned_matrix_content_sha256(
    modality: str,
    frame: pd.DataFrame,
    sample_ids: Sequence[str],
    feature_names: Sequence[str],
) -> str:
    """Hash canonical aligned numeric matrix content without pandas internals."""
    ids = _patient_ids(sample_ids)
    names = tuple(str(name) for name in feature_names)
    if (
        modality not in PRIMARY_MODALITIES
        or not isinstance(frame, pd.DataFrame)
        or tuple(str(value) for value in frame.index) != ids
        or tuple(str(value) for value in frame.columns) != names
        or frame.shape != (len(ids), len(names))
        or len(names) != len(set(names))
    ):
        raise ValueError("Canonical aligned matrix does not match its modality, IDs, or feature schema.")
    values: list[list[int | float | None]] = []
    dtypes: list[str] = []
    for column_index, dtype in enumerate(frame.dtypes):
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            raise ValueError("Canonical aligned matrix values must use non-boolean numeric dtypes.")
        dtypes.append(str(dtype))
        for row_index in range(len(ids)):
            if column_index == 0:
                values.append([])
            value = frame.iat[row_index, column_index]
            if pd.isna(value):
                values[row_index].append(None)
                continue
            numeric = value.item() if isinstance(value, np.generic) else value
            if isinstance(numeric, bool) or not isinstance(numeric, Real) or not np.isfinite(numeric):
                raise ValueError("Canonical aligned matrix values must be finite numeric values or missing.")
            values[row_index].append(int(numeric) if isinstance(numeric, (int, np.integer)) else float(numeric))
    return payload_sha256({
        "schema_version": "primary-v1-aligned-matrix-content-v1",
        "modality": modality,
        "ordered_sample_ids": list(ids),
        "ordered_feature_names": list(names),
        "column_dtypes": dtypes,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "missing_value_encoding": "json-null",
        "values": values,
    })


def _primary_run_material(protocol: PrimaryProtocolV1, aligned_data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, PrimaryProtocolV1) or not isinstance(aligned_data, Mapping):
        raise TypeError("Primary run provenance requires PrimaryProtocolV1 and canonical aligned data.")
    adapter = ModalityAdapter.from_aligned_data(aligned_data)
    try:
        sample_ids = tuple(str(value) for value in aligned_data["sample_ids"])
        labels = tuple(int(value) for value in aligned_data["y_binary"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Primary run provenance requires aligned sample IDs and binary labels.") from error
    if not sample_ids or len(sample_ids) != len(set(sample_ids)) or sample_ids != tuple(sorted(sample_ids)) or len(labels) != len(sample_ids) or set(labels) != {0, 1}:
        raise ValueError("Primary run provenance requires ordered unique binary aligned data.")
    fingerprint = build_outer_data_fingerprint(aligned_data)
    files = aligned_data.get("audit_summary", {}).get("files", {})
    source_hashes: dict[str, str] = {}
    for modality, mapping in _PRIMARY_MODALITY_MAPPING.items():
        audit = files.get(mapping["source_audit_key"])
        if not isinstance(audit, Mapping) or not isinstance(audit.get("sha256"), str) or len(audit["sha256"]) != 64:
            raise ValueError("Primary run provenance requires source-file SHA-256 audit metadata.")
        # Production loader audits contain a source path; synthetic fixtures must say so.
        if "path" not in audit and audit.get("provenance") != "synthetic_test_fixture":
            raise ValueError("Source-file hashes require loader audit metadata or an explicit synthetic test audit.")
        source_hashes[modality] = audit["sha256"]
    feature_hashes = {modality: adapter.bindings[modality].source_schema_sha256 for modality in PRIMARY_MODALITIES}
    labels_hash = payload_sha256([{"sample_id": sample_id, "true_label": label} for sample_id, label in zip(sample_ids, labels)])
    modality_content_hashes = {
        modality: aligned_matrix_content_sha256(
            modality,
            aligned_data[adapter.bindings[modality].matrix_key],
            sample_ids,
            aligned_data["feature_columns"][adapter.bindings[modality].feature_key],
        )
        for modality in PRIMARY_MODALITIES
    }
    aligned_data_content_identity_sha256 = payload_sha256({
        "schema_version": "primary-v1-aligned-data-content-v1",
        "ordered_sample_ids_sha256": ordered_patient_ids_sha256(sample_ids),
        "binary_labels_sha256": labels_hash,
        "modality_mapping": _PRIMARY_MODALITY_MAPPING,
        "feature_schema_sha256": feature_hashes,
        "modality_content_sha256": modality_content_hashes,
    })
    payload = {
        "schema_version": "primary-v1-data-integrity-v1",
        "protocol_sha256": protocol.identity_sha256,
        "source_file_sha256": source_hashes,
        "aligned_data_fingerprint": fingerprint,
        "ordered_sample_ids_sha256": ordered_patient_ids_sha256(sample_ids),
        "patient_set_sha256": patient_set_sha256(sample_ids),
        "binary_labels_sha256": labels_hash,
        "feature_schema_sha256": feature_hashes,
        "modality_mapping": _PRIMARY_MODALITY_MAPPING,
        "modality_adapter_sha256": adapter.identity_sha256,
        "modality_content_sha256": modality_content_hashes,
        "aligned_data_content_identity_sha256": aligned_data_content_identity_sha256,
        "cohort_size": len(sample_ids),
    }
    return {**payload, "data_integrity_sha256": payload_sha256(payload)}


@dataclass(frozen=True)
class PrimaryV1RunProvenance:
    """Declared prospective run identity, independently checked against aligned data."""

    schema_version: str
    run_id: str
    root_seed: int
    protocol_id: str
    protocol_sha256: str
    seed_policy_identity_sha256: str
    data_integrity_sha256: str
    source_file_sha256: Mapping[str, str]
    aligned_data_fingerprint: Mapping[str, Any]
    ordered_sample_ids_sha256: str
    patient_set_sha256: str
    binary_labels_sha256: str
    feature_schema_sha256: Mapping[str, str]
    modality_mapping: Mapping[str, Mapping[str, str]]
    modality_content_sha256: Mapping[str, str]
    aligned_data_content_identity_sha256: str
    cohort_size: int

    def __post_init__(self) -> None:
        if self.schema_version != "primary-v1-run-provenance-v1" or not isinstance(self.run_id, str) or not self.run_id.strip() or type(self.root_seed) is not int or self.root_seed < 0 or type(self.cohort_size) is not int or self.cohort_size < 1:
            raise ValueError("Primary V1 run provenance fields are invalid.")
        if set(self.source_file_sha256) != set(PRIMARY_MODALITIES) or set(self.feature_schema_sha256) != set(PRIMARY_MODALITIES) or set(self.modality_mapping) != set(PRIMARY_MODALITIES) or set(self.modality_content_sha256) != set(PRIMARY_MODALITIES):
            raise ValueError("Primary V1 run provenance modalities are invalid.")
        hashes = (self.protocol_sha256, self.seed_policy_identity_sha256, self.data_integrity_sha256, self.ordered_sample_ids_sha256, self.patient_set_sha256, self.binary_labels_sha256, self.aligned_data_content_identity_sha256, *self.source_file_sha256.values(), *self.feature_schema_sha256.values(), *self.modality_content_sha256.values())
        if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
            raise ValueError("Primary V1 run provenance requires SHA-256 identities.")
        object.__setattr__(self, "source_file_sha256", MappingProxyType(dict(self.source_file_sha256)))
        object.__setattr__(self, "aligned_data_fingerprint", MappingProxyType(dict(self.aligned_data_fingerprint)))
        object.__setattr__(self, "feature_schema_sha256", MappingProxyType(dict(self.feature_schema_sha256)))
        object.__setattr__(self, "modality_mapping", MappingProxyType({key: MappingProxyType(dict(value)) for key, value in self.modality_mapping.items()}))
        object.__setattr__(self, "modality_content_sha256", MappingProxyType(dict(self.modality_content_sha256)))

    @property
    def payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "run_id": self.run_id, "root_seed": self.root_seed, "protocol_id": self.protocol_id, "protocol_sha256": self.protocol_sha256, "seed_policy_identity_sha256": self.seed_policy_identity_sha256, "data_integrity_sha256": self.data_integrity_sha256, "source_file_sha256": dict(self.source_file_sha256), "aligned_data_fingerprint": dict(self.aligned_data_fingerprint), "ordered_sample_ids_sha256": self.ordered_sample_ids_sha256, "patient_set_sha256": self.patient_set_sha256, "binary_labels_sha256": self.binary_labels_sha256, "feature_schema_sha256": dict(self.feature_schema_sha256), "modality_mapping": {key: dict(value) for key, value in self.modality_mapping.items()}, "modality_content_sha256": dict(self.modality_content_sha256), "aligned_data_content_identity_sha256": self.aligned_data_content_identity_sha256, "cohort_size": self.cohort_size}

    @property
    def identity_sha256(self) -> str:
        return payload_sha256(self.payload)

    @property
    def seed_binding_identity_sha256(self) -> str:
        """Run-label-independent material that controls stochastic realization."""
        return payload_sha256({"schema_version": self.schema_version, "root_seed": self.root_seed, "protocol_sha256": self.protocol_sha256, "seed_policy_identity_sha256": self.seed_policy_identity_sha256, "data_integrity_sha256": self.data_integrity_sha256})


def create_primary_v1_run_provenance(*, run_id: str, root_seed: int, protocol: PrimaryProtocolV1, aligned_data: Mapping[str, Any]) -> PrimaryV1RunProvenance:
    material = _primary_run_material(protocol, aligned_data)
    return PrimaryV1RunProvenance("primary-v1-run-provenance-v1", run_id, root_seed, protocol.protocol_id, protocol.identity_sha256, protocol.seed_policy_identity_sha256, material["data_integrity_sha256"], material["source_file_sha256"], material["aligned_data_fingerprint"], material["ordered_sample_ids_sha256"], material["patient_set_sha256"], material["binary_labels_sha256"], material["feature_schema_sha256"], material["modality_mapping"], material["modality_content_sha256"], material["aligned_data_content_identity_sha256"], material["cohort_size"])


def validate_primary_v1_run_provenance(provenance: PrimaryV1RunProvenance, *, protocol: PrimaryProtocolV1, aligned_data: Mapping[str, Any]) -> None:
    if not isinstance(provenance, PrimaryV1RunProvenance):
        raise TypeError("Primary V1 run provenance is required.")
    expected = create_primary_v1_run_provenance(run_id=provenance.run_id, root_seed=provenance.root_seed, protocol=protocol, aligned_data=aligned_data)
    if provenance != expected:
        raise ValueError("Primary V1 run provenance does not match canonical protocol/data inputs.")


def candidate_identity_payload(candidate_id: str, ae_ratio: float, latent_dimensions: Mapping[str, int], fused_latent_width: int, logistic_c: float, augmentation: str, class_weight: None, protocol: PrimaryProtocolV1 | None = None) -> dict[str, Any]:
    frozen = protocol or PrimaryProtocolV1()
    if set(latent_dimensions) != set(PRIMARY_MODALITIES):
        raise ValueError("Candidate identity requires exactly the frozen modalities.")
    return {"schema_version": "primary-candidate-v1", "protocol_id": frozen.protocol_id, "protocol_hash": frozen.identity_sha256, "candidate_id": candidate_id, "ae_ratio": float(ae_ratio), "latent_dimensions": {key: int(latent_dimensions[key]) for key in PRIMARY_MODALITIES}, "fused_latent_width": int(fused_latent_width), "logistic_c": float(logistic_c), "augmentation": augmentation, "class_weight": class_weight}


def candidate_identity_sha256(**kwargs: Any) -> str:
    return payload_sha256(candidate_identity_payload(**kwargs))


def derive_seed(root_seed: int, binding: Mapping[str, Any]) -> int:
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer and not a boolean.")
    digest = hashlib.sha256(payload_sha256({"root_seed": root_seed, "binding": dict(binding)}).encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class PrimarySeedManifest:
    root_seed: int
    binding: Mapping[str, Any]
    seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.root_seed) is not int or self.root_seed < 0 or not isinstance(self.binding, Mapping) or not isinstance(self.seeds, Mapping):
            raise ValueError("PrimarySeedManifest inputs are invalid.")
        copied = {str(key): value for key, value in self.seeds.items()}
        if not copied or any(type(value) is not int or value < 0 for value in copied.values()):
            raise ValueError("PrimarySeedManifest requires explicit non-negative integer seeds.")
        object.__setattr__(self, "binding", MappingProxyType(dict(self.binding)))
        object.__setattr__(self, "seeds", MappingProxyType(copied))

    @property
    def identity_sha256(self) -> str:
        return payload_sha256({"schema_version": "primary-seed-manifest-v1", "derivation": "primary-seed-derivation-v1", "root_seed": self.root_seed, "binding": dict(self.binding), "seeds": dict(self.seeds)})

    @classmethod
    def primary_expected_keys(cls) -> set[str]:
        keys = {"outer:fold_generation", "inner:fold_generation", "final:ae_split", "final:ctgan", "final:lr"}
        keys.update(f"final:ae_select:{name}" for name in PRIMARY_MODALITIES)
        keys.update(f"final:ae_refit:{name}" for name in PRIMARY_MODALITIES)
        for fold_id in range(3):
            for ratio in PrimaryProtocolV1().ae_ratios:
                keys.update((f"ae_split:{fold_id}:{ratio}", f"ctgan:{fold_id}:{ratio}"))
                keys.update(f"ae_select:{fold_id}:{ratio}:{name}" for name in PRIMARY_MODALITIES)
                keys.update(f"ae_refit:{fold_id}:{ratio}:{name}" for name in PRIMARY_MODALITIES)
                keys.update(f"lr:{fold_id}:{ratio}:{c_value}" for c_value in PrimaryProtocolV1().logistic_cs)
        return keys

    @classmethod
    def generate(cls, root_seed: int, binding: Mapping[str, Any], keys: Sequence[str]) -> "PrimarySeedManifest":
        keys_tuple = tuple(str(key) for key in keys)
        if not keys_tuple or len(keys_tuple) != len(set(keys_tuple)):
            raise ValueError("PrimarySeedManifest seed keys must be non-empty and unique.")
        return cls(root_seed, dict(binding), {key: derive_seed(root_seed, {"manifest_binding": dict(binding), "component": key}) for key in keys_tuple})

    @classmethod
    def generate_primary(cls, provenance: PrimaryV1RunProvenance, candidates: Sequence[Any]) -> "PrimarySeedManifest":
        """Materialize every primary-search and final-refit seed before fitting begins."""
        if not isinstance(provenance, PrimaryV1RunProvenance):
            raise TypeError("PrimarySeedManifest requires PrimaryV1RunProvenance.")
        expected_pairs = {(ratio, c_value) for ratio in PrimaryProtocolV1().ae_ratios for c_value in PrimaryProtocolV1().logistic_cs}
        actual_pairs = {(getattr(candidate, "ae_ratio", None), getattr(candidate, "logistic_c", None)) for candidate in candidates}
        if actual_pairs != expected_pairs or len(tuple(candidates)) != 9:
            raise ValueError("PrimarySeedManifest requires the complete nine-candidate Primary V1 grid.")
        binding = {"schema_version": "primary-seed-binding-v1", "protocol_sha256": provenance.protocol_sha256, "seed_policy_identity_sha256": provenance.seed_policy_identity_sha256, "run_seed_binding_identity_sha256": provenance.seed_binding_identity_sha256, "data_integrity_sha256": provenance.data_integrity_sha256}
        return cls.generate(provenance.root_seed, binding, sorted(cls.primary_expected_keys()))

    def validate_primary_derivation(self, protocol: PrimaryProtocolV1, provenance: PrimaryV1RunProvenance) -> None:
        if provenance.protocol_id != protocol.protocol_id or provenance.protocol_sha256 != protocol.identity_sha256 or provenance.seed_policy_identity_sha256 != protocol.seed_policy_identity_sha256:
            raise ValueError("PrimarySeedManifest provenance protocol binding differs.")
        expected = self.generate_primary(provenance, [type("Candidate", (), {"ae_ratio": ratio, "logistic_c": c_value})() for ratio in protocol.ae_ratios for c_value in protocol.logistic_cs])
        if self != expected:
            raise ValueError("PrimarySeedManifest does not regenerate from its run provenance.")

    def require(self, key: str) -> int:
        value = self.seeds.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"PrimarySeedManifest is missing explicit seed: {key}")
        return value

    def primary_search_seed_book(self) -> Any:
        from .search import PrimarySearchSeedBook
        return PrimarySearchSeedBook(dict(self.seeds), self.identity_sha256)

    def final_refit_seed_book(self) -> Any:
        from .final_refit import FinalRefitSeedBook
        return FinalRefitSeedBook(self.require("final:ae_split"), {name: self.require(f"final:ae_select:{name}") for name in PRIMARY_MODALITIES}, {name: self.require(f"final:ae_refit:{name}") for name in PRIMARY_MODALITIES}, self.require("final:ctgan"), self.require("final:lr"))
