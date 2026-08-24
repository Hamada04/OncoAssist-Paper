"""Production shared AE/latent/CTGAN construction for one primary inner fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import payload_sha256
from .autoencoder import AutoencoderArchitecture, AutoencoderTrainingConfig, build_autoencoder_split, refit_selected_epoch_modality, select_epoch_for_modality
from .ctgan import CTGANConfig, augment_with_minority_ctgan
from .latent import FUSION_MODALITY_ORDER, build_latent_feature_names, fuse_fitted_encoders
from .protocol import ModalityAdapter, PrimaryProtocolV1, PrimarySeedManifest, PrimaryV1RunProvenance, ordered_patient_ids_sha256, patient_set_sha256, validate_primary_v1_run_provenance
from .folds import FoldProtocol
from .search import PrimarySearchSeedBook, PrimarySharedBuilderBinding, SharedInnerRepresentation


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(result) != len(set(result)) or any(not value.strip() for value in result):
        raise ValueError(f"{name} IDs must be non-empty, unique, and non-blank.")
    return result


def make_primary_inner_builder(
    aligned_data: Mapping[str, Any],
    modality_adapter: ModalityAdapter,
    outer_training_ids: Sequence[str],
    outer_training_labels: Sequence[int] | np.ndarray,
    ae_training_config: AutoencoderTrainingConfig,
    ctgan_config: CTGANConfig,
    *,
    protocol: PrimaryProtocolV1,
    seed_manifest: PrimarySeedManifest,
    run_provenance: PrimaryV1RunProvenance,
    fold_protocol: FoldProtocol,
    ae_validation_fraction: float,
    synthetic_namespace_prefix: str,
    protocol_hash: str,
    outer_fold_identity: Mapping[str, Any],
) -> PrimarySharedBuilderBinding:
    """Bind validated raw inputs once; returned builder makes one ratio/fold representation."""
    if not isinstance(protocol, PrimaryProtocolV1) or protocol_hash != protocol.identity_sha256:
        raise ValueError("Primary inner builder requires its exact PrimaryProtocolV1 identity.")
    validate_primary_v1_run_provenance(run_provenance, protocol=protocol, aligned_data=aligned_data)
    protocol.validate_primary_seed_manifest(seed_manifest, run_provenance)
    if fold_protocol != protocol.make_fold_protocol(seed_manifest, run_provenance):
        raise ValueError("Primary inner builder requires the seed-manifest-bound FoldProtocol.")
    protocol.validate_autoencoder_training_config(ae_training_config, ae_validation_fraction)
    protocol.validate_ctgan_config(ctgan_config)
    outer_ids = _ids(outer_training_ids, "outer training")
    labels = np.asarray(outer_training_labels, dtype=int)
    if len(labels) != len(outer_ids) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Primary inner builder requires aligned outer-training binary labels.")
    if not isinstance(modality_adapter, ModalityAdapter) or not isinstance(aligned_data, Mapping):
        raise TypeError("Primary inner builder requires aligned data and ModalityAdapter.")
    frames: dict[str, pd.DataFrame] = {}
    contracts: dict[str, tuple[str, ...]] = {}
    for modality in FUSION_MODALITY_ORDER:
        binding = modality_adapter.bindings[modality]
        frame = aligned_data.get(binding.matrix_key)
        names = tuple(aligned_data.get("feature_columns", {}).get(binding.feature_key, ()))
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != names or payload_sha256(list(names)) != binding.source_schema_sha256:
            raise ValueError("Aligned data differs from its declared modality adapter schema.")
        if not set(outer_ids).issubset({str(value) for value in frame.index}):
            raise ValueError("Outer-training IDs are absent from an aligned modality matrix.")
        frames[modality], contracts[modality] = frame, names
    label_by_id = dict(zip(outer_ids, labels.tolist()))

    def builder(ratio: float, dimensions: Mapping[str, int], inner_fold: Mapping[str, Any], seed_book: PrimarySearchSeedBook) -> SharedInnerRepresentation:
        inner_id = inner_fold.get("inner_fold_id")
        train_ids = _ids(inner_fold.get("inner_train_sample_ids", ()), "inner training")
        validation_ids = _ids(inner_fold.get("inner_validation_sample_ids", ()), "inner validation")
        if set(train_ids).intersection(validation_ids) or set(train_ids).union(validation_ids) != set(outer_ids):
            raise ValueError("Inner fold must partition exactly the bound outer-training patients.")
        train_labels = np.asarray([label_by_id[item] for item in train_ids], dtype=int)
        validation_labels = np.asarray([label_by_id[item] for item in validation_ids], dtype=int)
        if set(train_labels.tolist()) != {0, 1}:
            raise ValueError("Inner training partition requires both binary classes.")
        split_seed = seed_book.require(f"ae_split:{inner_id}:{ratio}")
        split = build_autoencoder_split(train_ids, train_labels, ae_validation_fraction, split_seed)
        refits, selections = {}, {}
        for modality in FUSION_MODALITY_ORDER:
            names = contracts[modality]
            if type(dimensions.get(modality)) is not int or dimensions[modality] < 2:
                raise ValueError("Requested latent dimensions are invalid.")
            architecture = AutoencoderArchitecture(modality, len(names), 128, dimensions[modality])
            train_frame = frames[modality].loc[list(train_ids), list(names)].copy()
            validation_frame = frames[modality].loc[list(validation_ids), list(names)].copy()
            select_seed = seed_book.require(f"ae_select:{inner_id}:{ratio}:{modality}")
            refit_seed = seed_book.require(f"ae_refit:{inner_id}:{ratio}:{modality}")
            selection = select_epoch_for_modality(train_frame, train_ids, names, split, architecture, ae_training_config, select_seed)
            refits[modality] = refit_selected_epoch_modality(train_frame, train_ids, validation_frame, validation_ids, names, architecture, ae_training_config, selection.selected_epoch_count, refit_seed)
            selections[modality] = selection
        fused = fuse_fitted_encoders(refits, train_ids, validation_ids)
        if fused.feature_names != build_latent_feature_names(dimensions):
            raise AssertionError("Primary inner fusion schema differs from requested ratio dimensions.")
        ctgan_seed = seed_book.require(f"ctgan:{inner_id}:{ratio}")
        augmentation = augment_with_minority_ctgan(fused.training, train_labels, train_ids, fused.feature_names, fused.evidence["feature_names_sha256"], ctgan_config, ctgan_seed, f"{synthetic_namespace_prefix}:inner_{inner_id}:ratio_{ratio}")
        evidence = {"protocol_hash": protocol_hash, "outer_fold_identity": dict(outer_fold_identity), "inner_fold_id": inner_id, "ae_ratio": ratio, "latent_dimensions": dict(dimensions), "inner_training_ordered_patient_ids_sha256": ordered_patient_ids_sha256(train_ids), "inner_validation_ordered_patient_ids_sha256": ordered_patient_ids_sha256(validation_ids), "inner_training_labels_sha256": payload_sha256(train_labels.tolist()), "modality_adapter": modality_adapter.payload(), "modality_adapter_sha256": modality_adapter.identity_sha256, "source_feature_schema_hashes": {name: modality_adapter.bindings[name].source_schema_sha256 for name in FUSION_MODALITY_ORDER}, "shared_ae_split": dict(split.evidence), "per_modality": {name: {"selection": dict(selections[name].evidence), "refit": dict(refits[name].evidence)} for name in FUSION_MODALITY_ORDER}, "fusion": dict(fused.evidence), "ctgan": dict(augmentation.evidence), "ctgan_seed": ctgan_seed, "validation_excluded_from_fit": True}
        return SharedInnerRepresentation(augmentation, fused.heldout, validation_ids, fused.feature_names, validation_labels, evidence)

    return PrimarySharedBuilderBinding("primary-shared-builder-v1", "primary-inner-v1", protocol.protocol_id, protocol.identity_sha256, patient_set_sha256(outer_ids), modality_adapter.identity_sha256, fold_protocol, protocol.fold_protocol_identity_sha256(fold_protocol, seed_manifest, run_provenance), seed_manifest.identity_sha256, run_provenance.identity_sha256, builder)
