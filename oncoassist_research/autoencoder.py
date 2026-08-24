"""Leakage-safe, fold-local modality autoencoders with named bottlenecks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

from .artifacts import payload_sha256
from .preprocessing import (
    FittedPreprocessor,
    fit_preprocessor,
    transform_with_preprocessor,
)


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


@dataclass(frozen=True)
class AutoencoderArchitecture:
    modality: str
    input_dim: int
    hidden_dim: int
    latent_dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ValueError("modality must be a non-empty string.")
        _positive_int("input_dim", self.input_dim)
        _positive_int("hidden_dim", self.hidden_dim)
        _positive_int("latent_dim", self.latent_dim, minimum=2)
        if self.hidden_dim != 128:
            raise ValueError("hidden_dim must be the corrected architecture anchor of 128.")
        if self.latent_dim >= self.input_dim:
            raise ValueError("latent_dim must be smaller than input_dim.")
        if self.latent_dim >= self.hidden_dim:
            raise ValueError("latent_dim must be smaller than hidden_dim.")

    def evidence(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "hidden_activation": "relu",
            "bottleneck_activation": "linear",
            "reconstruction_activation": "linear",
            "optimizer": "adam",
            "loss": "mse",
        }


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    max_epochs: int
    batch_size: int
    patience: int

    def __post_init__(self) -> None:
        _positive_int("max_epochs", self.max_epochs)
        _positive_int("batch_size", self.batch_size)
        _positive_int("patience", self.patience, minimum=0)


@dataclass(frozen=True)
class AutoencoderSplit:
    fit_sample_ids: tuple[str, ...]
    stop_sample_ids: tuple[str, ...]
    split_seed: int
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class EpochSelectionResult:
    selected_epoch_count: int
    evidence: Mapping[str, Any]


@dataclass
class FittedEncoder:
    preprocessor: FittedPreprocessor
    autoencoder: Any
    encoder: Any
    training_latents: np.ndarray
    heldout_latents: np.ndarray
    evidence: Mapping[str, Any]


def latent_dim_from_ratio(input_dim: int, ratio: float) -> int:
    _positive_int("input_dim", input_dim)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("ratio must be a finite positive number.")
    if not math.isfinite(float(ratio)) or float(ratio) <= 0.0:
        raise ValueError("ratio must be a finite positive number.")
    latent_dim = int(math.ceil(input_dim * float(ratio)))
    if latent_dim < 2 or latent_dim >= input_dim:
        raise ValueError("ratio must produce a compressive latent width of at least 2.")
    return latent_dim


def configure_deterministic_seed(model_seed: int) -> dict[str, Any]:
    """Request deterministic execution settings without promising bit-identical results."""
    _positive_int("model_seed", model_seed, minimum=0)
    tf.keras.backend.clear_session()
    random.seed(model_seed)
    np.random.seed(model_seed)
    tf.keras.utils.set_random_seed(model_seed)
    try:
        tf.config.experimental.enable_op_determinism()
        enabled = True
    except (AttributeError, RuntimeError):
        enabled = False
    return {
        "model_seed": model_seed,
        "deterministic_settings_requested": True,
        "deterministic_operations_enabled": enabled,
    }


def build_autoencoder(
    architecture: AutoencoderArchitecture,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.layers.Input(
        shape=(architecture.input_dim,), name=f"{architecture.modality}_input"
    )
    encoded = tf.keras.layers.Dense(
        architecture.hidden_dim,
        activation="relu",
        name=f"{architecture.modality}_encoder_hidden",
    )(inputs)
    bottleneck_name = f"{architecture.modality}_bottleneck"
    bottleneck = tf.keras.layers.Dense(
        architecture.latent_dim, activation="linear", name=bottleneck_name
    )(encoded)
    decoded = tf.keras.layers.Dense(
        architecture.hidden_dim,
        activation="relu",
        name=f"{architecture.modality}_decoder_hidden",
    )(bottleneck)
    reconstruction = tf.keras.layers.Dense(
        architecture.input_dim,
        activation="linear",
        name=f"{architecture.modality}_reconstruction",
    )(decoded)
    autoencoder = tf.keras.Model(
        inputs=inputs,
        outputs=reconstruction,
        name=f"{architecture.modality}_autoencoder",
    )
    autoencoder.compile(optimizer="adam", loss="mse")
    encoder = tf.keras.Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer(bottleneck_name).output,
        name=f"{architecture.modality}_encoder",
    )
    return autoencoder, encoder


def _normalized_ids(sample_ids: Sequence[str], expected_count: int) -> tuple[str, ...]:
    ids = tuple(str(sample_id) for sample_id in sample_ids)
    if len(ids) != expected_count:
        raise ValueError("SAMPLE_ID and label/data counts must match.")
    if len(set(ids)) != len(ids) or any(not sample_id.strip() for sample_id in ids):
        raise ValueError("SAMPLE_ID values must be unique and non-blank.")
    return ids


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))}


def build_autoencoder_split(
    training_sample_ids: Sequence[str],
    labels_for_stratification: Sequence[int] | np.ndarray,
    validation_fraction: float,
    split_seed: int,
) -> AutoencoderSplit:
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)):
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    _positive_int("split_seed", split_seed, minimum=0)
    labels = np.asarray(labels_for_stratification, dtype=int)
    ids = _normalized_ids(training_sample_ids, len(labels))
    if not np.isin(labels, [0, 1]).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("Autoencoder split requires both binary classes for stratification.")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=float(validation_fraction), random_state=split_seed
    )
    try:
        fit_indices, stop_indices = next(splitter.split(np.zeros(len(labels)), labels))
    except ValueError as error:
        raise ValueError("Unable to create a stratified autoencoder fit/stop split.") from error
    fit_order = np.random.default_rng(split_seed).permutation(fit_indices)
    fit_ids = tuple(ids[index] for index in fit_order)
    stop_ids = tuple(ids[index] for index in stop_indices)
    if set(fit_ids).intersection(stop_ids) or set(fit_ids).union(stop_ids) != set(ids):
        raise AssertionError("Autoencoder fit/stop split does not partition declared training IDs.")
    labels_by_id = dict(zip(ids, labels.tolist()))
    evidence = {
        "split_seed": split_seed,
        "fit_sample_count": len(fit_ids),
        "fit_sample_ids_sha256": payload_sha256(list(fit_ids)),
        "stop_sample_count": len(stop_ids),
        "stop_sample_ids_sha256": payload_sha256(list(stop_ids)),
        "fit_class_counts": _class_counts(np.asarray([labels_by_id[item] for item in fit_ids])),
        "stop_class_counts": _class_counts(np.asarray([labels_by_id[item] for item in stop_ids])),
        "labels_used_only_for": "stratification",
    }
    return AutoencoderSplit(fit_ids, stop_ids, split_seed, evidence)


def _select_raw_rows(
    data_df: pd.DataFrame, declared_ids: Sequence[str], selected_ids: Sequence[str]
) -> pd.DataFrame:
    declared = _normalized_ids(declared_ids, len(data_df))
    if tuple(str(index) for index in data_df.index.tolist()) != declared:
        raise ValueError("Raw DataFrame index must match declared training SAMPLE_ID order.")
    selected = data_df.loc[list(selected_ids)].copy()
    if tuple(str(index) for index in selected.index.tolist()) != tuple(selected_ids):
        raise AssertionError("Raw row selection did not preserve requested SAMPLE_ID order.")
    return selected


def _validate_split_for_training(split: AutoencoderSplit, training_ids: Sequence[str]) -> None:
    if not isinstance(split, AutoencoderSplit):
        raise TypeError("Epoch selection requires an AutoencoderSplit.")
    declared = set(training_ids)
    if set(split.fit_sample_ids).intersection(split.stop_sample_ids):
        raise ValueError("Autoencoder split fit/stop SAMPLE_ID values overlap.")
    if set(split.fit_sample_ids).union(split.stop_sample_ids) != declared:
        raise ValueError("Autoencoder split does not partition declared training SAMPLE_ID values.")


def _fit_temporary_model(
    autoencoder: tf.keras.Model,
    fit_matrix: np.ndarray,
    stop_matrix: np.ndarray,
    config: AutoencoderTrainingConfig,
) -> Any:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=config.patience, restore_best_weights=True
    )
    return autoencoder.fit(
        fit_matrix,
        fit_matrix,
        validation_data=(stop_matrix, stop_matrix),
        epochs=config.max_epochs,
        batch_size=min(config.batch_size, len(fit_matrix)),
        callbacks=[early_stopping],
        shuffle=False,
        verbose=0,
    )


def select_epoch_for_modality(
    training_df: pd.DataFrame,
    training_sample_ids: Sequence[str],
    feature_names: Sequence[str],
    split: AutoencoderSplit,
    architecture: AutoencoderArchitecture,
    training_config: AutoencoderTrainingConfig,
    model_seed: int,
) -> EpochSelectionResult:
    """Select an epoch using temporary AE-fit-only preprocessing and AE-stop loss."""
    ids = _normalized_ids(training_sample_ids, len(training_df))
    _validate_split_for_training(split, ids)
    if architecture.input_dim != len(feature_names):
        raise ValueError("Autoencoder architecture input_dim does not match feature names.")
    fit_raw = _select_raw_rows(training_df, ids, split.fit_sample_ids)
    stop_raw = _select_raw_rows(training_df, ids, split.stop_sample_ids)
    temporary_preprocessor = fit_preprocessor(fit_raw, split.fit_sample_ids, feature_names)
    fit_partition = transform_with_preprocessor(
        temporary_preprocessor, fit_raw, split.fit_sample_ids, feature_names
    )
    stop_partition = transform_with_preprocessor(
        temporary_preprocessor, stop_raw, split.stop_sample_ids, feature_names
    )
    deterministic = configure_deterministic_seed(model_seed)
    autoencoder, _ = build_autoencoder(architecture)
    history = _fit_temporary_model(
        autoencoder, fit_partition.matrix, stop_partition.matrix, training_config
    )
    losses = [float(value) for value in history.history.get("loss", [])]
    validation_losses = [float(value) for value in history.history.get("val_loss", [])]
    validation_array = np.asarray(validation_losses, dtype=float)
    if validation_array.ndim != 1 or not len(validation_array) or not np.isfinite(validation_array).all():
        raise ValueError("Temporary epoch selection requires finite validation-loss history.")
    selected = int(np.argmin(validation_array) + 1)
    if not 1 <= selected <= training_config.max_epochs:
        raise ValueError("Selected epoch is outside the configured epoch budget.")
    reconstructed_fit = autoencoder.predict(fit_partition.matrix, verbose=0)
    reconstructed_stop = autoencoder.predict(stop_partition.matrix, verbose=0)
    evidence = {
        "modality": architecture.modality,
        "architecture": architecture.evidence(),
        **split.evidence,
        **deterministic,
        "model_seed": model_seed,
        "epochs_requested": training_config.max_epochs,
        "epochs_ran": len(losses),
        "validation_loss_history": validation_losses,
        "selected_epoch_count": selected,
        "best_validation_loss": float(validation_array[selected - 1]),
        "fit_reconstruction_mse": float(np.mean(np.square(reconstructed_fit - fit_partition.matrix))),
        "stop_reconstruction_mse": float(np.mean(np.square(reconstructed_stop - stop_partition.matrix))),
        "temporary_preprocessing_fit_sample_ids_sha256": temporary_preprocessor.fit_sample_ids_sha256,
        "temporary_preprocessing_metadata": dict(temporary_preprocessor.metadata),
        "temporary_model_used_only_for_epoch_selection": True,
    }
    return EpochSelectionResult(selected, evidence)


def _fit_final_model(
    autoencoder: tf.keras.Model,
    training_matrix: np.ndarray,
    selected_epoch_count: int,
    config: AutoencoderTrainingConfig,
) -> Any:
    return autoencoder.fit(
        training_matrix,
        training_matrix,
        epochs=selected_epoch_count,
        batch_size=min(config.batch_size, len(training_matrix)),
        shuffle=False,
        verbose=0,
    )


def refit_selected_epoch_modality(
    training_df: pd.DataFrame,
    training_sample_ids: Sequence[str],
    heldout_df: pd.DataFrame,
    heldout_sample_ids: Sequence[str],
    feature_names: Sequence[str],
    architecture: AutoencoderArchitecture,
    training_config: AutoencoderTrainingConfig,
    selected_epoch_count: int,
    model_seed: int,
) -> FittedEncoder:
    """Freshly preprocess and train a selected-epoch AE on complete training rows only."""
    _positive_int("selected_epoch_count", selected_epoch_count)
    if selected_epoch_count > training_config.max_epochs:
        raise ValueError("selected_epoch_count exceeds the configured epoch budget.")
    training_ids = _normalized_ids(training_sample_ids, len(training_df))
    heldout_ids = _normalized_ids(heldout_sample_ids, len(heldout_df))
    if architecture.input_dim != len(feature_names):
        raise ValueError("Autoencoder architecture input_dim does not match feature names.")
    fresh_preprocessor = fit_preprocessor(training_df, training_ids, feature_names)
    training_partition = transform_with_preprocessor(
        fresh_preprocessor, training_df, training_ids, feature_names
    )
    heldout_partition = transform_with_preprocessor(
        fresh_preprocessor, heldout_df, heldout_ids, feature_names
    )
    deterministic = configure_deterministic_seed(model_seed)
    autoencoder, encoder = build_autoencoder(architecture)
    _fit_final_model(autoencoder, training_partition.matrix, selected_epoch_count, training_config)
    training_latents = np.asarray(encoder.predict(training_partition.matrix, verbose=0), dtype=np.float32)
    heldout_latents = np.asarray(encoder.predict(heldout_partition.matrix, verbose=0), dtype=np.float32)
    expected_training_shape = (len(training_ids), architecture.latent_dim)
    expected_heldout_shape = (len(heldout_ids), architecture.latent_dim)
    if training_latents.shape != expected_training_shape or heldout_latents.shape != expected_heldout_shape:
        raise ValueError("Fresh autoencoder refit produced invalid latent shapes.")
    if not np.isfinite(training_latents).all() or not np.isfinite(heldout_latents).all():
        raise ValueError("Fresh autoencoder refit produced non-finite latent values.")
    evidence = {
        "modality": architecture.modality,
        "architecture": architecture.evidence(),
        **deterministic,
        "model_seed": model_seed,
        "selected_epoch_count": selected_epoch_count,
        "complete_training_sample_count": len(training_ids),
        "complete_training_sample_ids_sha256": payload_sha256(list(training_ids)),
        "heldout_sample_count": len(heldout_ids),
        "heldout_sample_ids_sha256": payload_sha256(list(heldout_ids)),
        "preprocessing_fit_sample_ids_sha256": fresh_preprocessor.fit_sample_ids_sha256,
        "latent_dim": architecture.latent_dim,
        "epochs_trained": selected_epoch_count,
        "validation_data_used": False,
        "early_stopping_used": False,
        "heldout_supplied_to_fit": False,
        "fresh_model_refit": True,
    }
    return FittedEncoder(
        preprocessor=fresh_preprocessor,
        autoencoder=autoencoder,
        encoder=encoder,
        training_latents=training_latents,
        heldout_latents=heldout_latents,
        evidence=evidence,
    )
