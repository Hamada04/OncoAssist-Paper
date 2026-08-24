"""Deterministic three-modality latent fusion without scientific modeling steps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import payload_sha256
from .autoencoder import FittedEncoder


FUSION_MODALITY_ORDER = ("mGE", "mDM", "mCNA")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(payload_sha256({"dtype": str(value.dtype), "shape": list(value.shape)}).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


def _dimensions(latent_dimensions: Mapping[str, int]) -> dict[str, int]:
    if set(latent_dimensions) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Latent dimensions must contain exactly mGE, mDM, and mCNA.")
    result: dict[str, int] = {}
    for modality in FUSION_MODALITY_ORDER:
        value = latent_dimensions[modality]
        if type(value) is not int or value < 1:
            raise ValueError("Latent dimensions must be positive integers.")
        result[modality] = value
    return result


def build_latent_feature_names(latent_dimensions: Mapping[str, int]) -> tuple[str, ...]:
    dimensions = _dimensions(latent_dimensions)
    names: list[str] = []
    for modality in FUSION_MODALITY_ORDER:
        width = max(3, len(str(dimensions[modality] - 1)))
        names.extend(f"{modality}_z{index:0{width}d}" for index in range(dimensions[modality]))
    if len(names) != len(set(names)):
        raise AssertionError("Latent feature names must be unique.")
    return tuple(names)


def build_latent_slices(latent_dimensions: Mapping[str, int]) -> dict[str, tuple[int, int]]:
    dimensions = _dimensions(latent_dimensions)
    start = 0
    slices: dict[str, tuple[int, int]] = {}
    for modality in FUSION_MODALITY_ORDER:
        stop = start + dimensions[modality]
        slices[modality] = (start, stop)
        start = stop
    return slices


@dataclass(frozen=True)
class FusedLatentRepresentation:
    training: np.ndarray
    heldout: np.ndarray
    training_sample_ids: tuple[str, ...]
    heldout_sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    modality_slices: Mapping[str, tuple[int, int]]
    latent_dimensions: Mapping[str, int]
    evidence: Mapping[str, Any]


def _validate_ids(sample_ids: Sequence[str], expected_rows: int, name: str) -> tuple[str, ...]:
    ids = tuple(str(item) for item in sample_ids)
    if len(ids) != expected_rows or len(ids) != len(set(ids)) or any(not item.strip() for item in ids):
        raise ValueError(f"{name} SAMPLE_ID values must be non-blank, unique, and row-aligned.")
    return ids


def _validate_latents(
    encoder: FittedEncoder, modality: str, training_ids: tuple[str, ...], heldout_ids: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(encoder, FittedEncoder):
        raise TypeError(f"{modality} must be a FittedEncoder.")
    architecture = encoder.evidence.get("architecture") if isinstance(encoder.evidence, Mapping) else None
    if not isinstance(architecture, Mapping) or type(architecture.get("latent_dim")) is not int:
        raise ValueError(f"{modality} encoder evidence lacks latent_dim.")
    latent_dim = architecture["latent_dim"]
    training = np.asarray(encoder.training_latents)
    heldout = np.asarray(encoder.heldout_latents)
    for name, matrix, ids in (("training", training, training_ids), ("heldout", heldout, heldout_ids)):
        if matrix.ndim != 2 or matrix.dtype == object or not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(f"{modality} {name} latents must be numeric two-dimensional arrays.")
        if matrix.shape != (len(ids), latent_dim) or not np.isfinite(matrix).all():
            raise ValueError(f"{modality} {name} latents have invalid shape or values.")
    for key, ids in (("complete_training_sample_ids_sha256", training_ids), ("heldout_sample_ids_sha256", heldout_ids)):
        recorded = encoder.evidence.get(key)
        if recorded is not None and recorded != payload_sha256(list(ids)):
            raise ValueError(f"{modality} encoder evidence does not match supplied patient IDs.")
    return training, heldout, latent_dim


def fuse_fitted_encoders(
    encoders: Mapping[str, FittedEncoder],
    training_sample_ids: Sequence[str],
    heldout_sample_ids: Sequence[str],
) -> FusedLatentRepresentation:
    if set(encoders) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Fusion requires exactly mGE, mDM, and mCNA encoders.")
    first = encoders["mGE"]
    training_ids = _validate_ids(training_sample_ids, len(first.training_latents), "training")
    heldout_ids = _validate_ids(heldout_sample_ids, len(first.heldout_latents), "heldout")
    training_parts: list[np.ndarray] = []
    heldout_parts: list[np.ndarray] = []
    dimensions: dict[str, int] = {}
    for modality in FUSION_MODALITY_ORDER:
        training, heldout, width = _validate_latents(encoders[modality], modality, training_ids, heldout_ids)
        training_parts.append(training)
        heldout_parts.append(heldout)
        dimensions[modality] = width
    training = np.concatenate(training_parts, axis=1).astype(np.float32, copy=True)
    heldout = np.concatenate(heldout_parts, axis=1).astype(np.float32, copy=True)
    feature_names = build_latent_feature_names(dimensions)
    slices = build_latent_slices(dimensions)
    if training.shape != (len(training_ids), len(feature_names)) or heldout.shape != (len(heldout_ids), len(feature_names)):
        raise AssertionError("Fused latent output shape is invalid.")
    if not np.isfinite(training).all() or not np.isfinite(heldout).all():
        raise ValueError("Fused latent output contains non-finite values.")
    recovery: dict[str, dict[str, Any]] = {}
    for modality, source_train, source_heldout in zip(FUSION_MODALITY_ORDER, training_parts, heldout_parts):
        start, stop = slices[modality]
        expected_train = source_train.astype(np.float32, copy=False)
        expected_heldout = source_heldout.astype(np.float32, copy=False)
        exact = source_train.dtype == np.float32 and source_heldout.dtype == np.float32
        matches = (
            np.array_equal(training[:, start:stop], expected_train) and np.array_equal(heldout[:, start:stop], expected_heldout)
            if exact
            else np.allclose(training[:, start:stop], expected_train, rtol=1e-6, atol=1e-6)
            and np.allclose(heldout[:, start:stop], expected_heldout, rtol=1e-6, atol=1e-6)
        )
        if not matches:
            raise AssertionError(f"{modality} latent slice recovery failed.")
        recovery[modality] = {"passed": True, "comparison": "exact" if exact else "float32_allclose"}
    evidence = {
        "modality_order": list(FUSION_MODALITY_ORDER),
        "latent_dimensions": dimensions,
        "modality_slices": {name: list(value) for name, value in slices.items()},
        "feature_names_sha256": payload_sha256(list(feature_names)),
        "training_sample_ids_sha256": payload_sha256(list(training_ids)),
        "heldout_sample_ids_sha256": payload_sha256(list(heldout_ids)),
        "source_latent_shapes": {name: {"training": list(part.shape), "heldout": list(hold.shape)} for name, part, hold in zip(FUSION_MODALITY_ORDER, training_parts, heldout_parts)},
        "fused_shapes": {"training": list(training.shape), "heldout": list(heldout.shape)},
        "fused_dtype": str(training.dtype),
        "fused_training_sha256": _array_sha256(training),
        "fused_heldout_sha256": _array_sha256(heldout),
        "finite_checks": {"training": True, "heldout": True},
        "slice_recovery": recovery,
        "target_identifier_excluded": True,
    }
    return FusedLatentRepresentation(_readonly_copy(training), _readonly_copy(heldout), training_ids, heldout_ids, feature_names, slices, dimensions, evidence)
