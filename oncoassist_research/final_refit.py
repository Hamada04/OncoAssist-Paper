"""Final selected-candidate refit and label-free frozen outer-test scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import payload_sha256
from .autoencoder import (
    AutoencoderArchitecture,
    AutoencoderTrainingConfig,
    FittedEncoder,
    build_autoencoder_split,
    latent_dim_from_ratio,
    refit_selected_epoch_modality,
    select_epoch_for_modality,
)
from .calibration import CalibratedOOFPrediction, CrossFittedCalibrationResult, FinalSigmoidCalibrator, apply_final_sigmoid_calibrator, validate_cross_fitted_calibration_result, validate_final_sigmoid_calibrator
from .classifiers import FittedLogisticClassifier, LogisticRegressionConfig, fit_logistic_classifier, score_logistic_classifier
from .ctgan import CTGANConfig, augment_with_minority_ctgan
from .latent import FUSION_MODALITY_ORDER, build_latent_feature_names, fuse_fitted_encoders
from .preprocessing import transform_with_preprocessor
from .search import LOGISTIC_CS, RATIOS, PrimaryCandidate, PrimarySearchResult, validate_primary_search_result
from .protocol import PrimaryProtocolV1, PrimaryV1RunProvenance, aligned_matrix_content_sha256, patient_set_sha256, ordered_patient_ids_sha256
from .thresholds import OperationalThresholdResult, validate_operational_threshold_result


_SCHEMA_VERSION = "final-outer-refit-v1"


def _ids(values: Sequence[str], expected_count: int, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) != expected_count or len(result) != len(set(result)) or any(not value.strip() for value in result):
        raise ValueError(f"{name} SAMPLE_ID values must be unique, non-blank, and row-aligned.")
    return result


def _seed(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be an explicit non-negative integer seed.")
    return value


def _mapping_of_modality_seeds(name: str, values: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(values, Mapping) or set(values) != set(FUSION_MODALITY_ORDER):
        raise ValueError(f"{name} must contain exactly mGE, mDM, and mCNA.")
    return {modality: _seed(f"{name}[{modality}]", values[modality]) for modality in FUSION_MODALITY_ORDER}


def _candidate_payload(candidate: PrimaryCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "ae_ratio": candidate.ae_ratio,
        "latent_dimensions": dict(candidate.latent_dimensions),
        "fused_latent_width": candidate.fused_latent_width,
        "logistic_c": candidate.logistic_c,
        "augmentation": candidate.augmentation,
        "class_weight": candidate.class_weight,
    }


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(payload_sha256({"dtype": str(array.dtype), "shape": list(array.shape)}).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    """Hash raw values, dtypes, columns, and order without serializing missing values."""
    digest = hashlib.sha256()
    digest.update(payload_sha256({"columns": list(frame.columns), "dtypes": [str(value) for value in frame.dtypes], "shape": list(frame.shape)}).encode("ascii"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _fitted_array_payload(name: str, value: Any, feature_count: int) -> dict[str, Any]:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.shape != (feature_count,)
        or array.dtype == object
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"Fitted preprocessing {name} must be finite and match the feature count.")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
        "sha256": _array_hash(array),
    }


def _fitted_preprocessor_state(refit: FittedEncoder) -> dict[str, Any]:
    preprocessor = refit.preprocessor
    feature_names = tuple(preprocessor.feature_names)
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("Fitted preprocessing feature names are invalid.")
    pipeline = getattr(preprocessor, "_pipeline", None)
    steps = getattr(pipeline, "named_steps", None)
    if not isinstance(steps, Mapping) or set(steps) != {"imputer", "scaler"}:
        raise ValueError("Fitted preprocessing pipeline must contain only imputer and scaler steps.")
    imputer, scaler = steps["imputer"], steps["scaler"]
    feature_count = len(feature_names)
    missing_values = getattr(imputer, "missing_values", None)
    if (
        getattr(imputer, "strategy", None) != "median"
        or not isinstance(missing_values, (int, float, np.number))
        or not np.isnan(missing_values)
        or getattr(imputer, "add_indicator", None) is not False
        or getattr(imputer, "keep_empty_features", None) is not False
        or getattr(imputer, "n_features_in_", None) != feature_count
        or tuple(str(value) for value in getattr(imputer, "feature_names_in_", ())) != feature_names
        or getattr(scaler, "with_mean", None) is not True
        or getattr(scaler, "with_std", None) is not True
        or getattr(scaler, "n_features_in_", None) != feature_count
    ):
        raise ValueError("Fitted preprocessing state differs from the required median-imputation/scaling contract.")
    return {
        "feature_names": list(feature_names),
        "imputer": {
            "strategy": imputer.strategy,
            "missing_values": "NaN",
            "add_indicator": imputer.add_indicator,
            "keep_empty_features": imputer.keep_empty_features,
            "n_features_in": int(imputer.n_features_in_),
            "statistics": _fitted_array_payload("imputer statistics_", imputer.statistics_, feature_count),
        },
        "scaler": {
            "with_mean": scaler.with_mean,
            "with_std": scaler.with_std,
            "n_features_in": int(scaler.n_features_in_),
            "mean": _fitted_array_payload("scaler mean_", scaler.mean_, feature_count),
            "var": _fitted_array_payload("scaler var_", scaler.var_, feature_count),
            "scale": _fitted_array_payload("scaler scale_", scaler.scale_, feature_count),
        },
    }


def _encoder_state(refit: FittedEncoder) -> dict[str, Any]:
    weights_getter = getattr(refit.encoder, "get_weights", None)
    weights = weights_getter() if callable(weights_getter) else ()
    return {
        "preprocessor_fit_ids_sha256": refit.preprocessor.fit_sample_ids_sha256,
        "preprocessor_metadata_sha256": payload_sha256(dict(refit.preprocessor.metadata)),
        "preprocessor_fitted_state": _fitted_preprocessor_state(refit),
        "encoder_type": f"{type(refit.encoder).__module__}.{type(refit.encoder).__qualname__}",
        "encoder_weight_hashes": [_array_hash(weight) for weight in weights],
    }


def _frozen_model_state_hash(refits: Mapping[str, FittedEncoder], classifier: FittedLogisticClassifier, calibrator: FinalSigmoidCalibrator, threshold: OperationalThresholdResult) -> str:
    return payload_sha256({
        "encoders": {modality: _encoder_state(refits[modality]) for modality in FUSION_MODALITY_ORDER},
        "classifier": {"feature_names": list(classifier.feature_names), "scaler_mean": _array_hash(classifier.scaler.mean_), "scaler_scale": _array_hash(classifier.scaler.scale_), "model_coefficients": _array_hash(classifier.model.coef_), "model_intercept": _array_hash(classifier.model.intercept_), "model_classes": _array_hash(classifier.model.classes_)},
        "calibrator": {"coefficients": _array_hash(calibrator.model.coef_), "intercept": _array_hash(calibrator.model.intercept_), "classes": _array_hash(calibrator.model.classes_)},
        "threshold": float(threshold.threshold),
    })


def _final_model_identity(
    *,
    protocol: PrimaryProtocolV1,
    search_selection_identity_sha256: str,
    candidate: PrimaryCandidate,
    selected_oof_predictions_sha256: str,
    cross_fitted_calibration_sha256: str,
    final_calibrator_identity_sha256: str,
    threshold_identity_sha256: str,
    outer_training_ids_sha256: str,
    outer_training_ordered_ids_sha256: str,
    expected_outer_test_ordered_ids_sha256: str,
    canonical_outer_training_input_identity_sha256: str,
    fused_feature_hash: str,
    frozen_model_state_sha256: str,
) -> str:
    """Hash every authority, selected configuration, and fitted state needed for scoring."""
    return payload_sha256({
        "schema_version": _SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.identity_sha256,
        "search_selection_identity_sha256": search_selection_identity_sha256,
        "candidate_id": candidate.candidate_id,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "selected_candidate": _candidate_payload(candidate),
        "selected_oof_predictions_sha256": selected_oof_predictions_sha256,
        "cross_fitted_calibration_sha256": cross_fitted_calibration_sha256,
        "final_calibrator_identity_sha256": final_calibrator_identity_sha256,
        "threshold_identity_sha256": threshold_identity_sha256,
        "outer_training_patient_set_sha256": outer_training_ids_sha256,
        "outer_training_ordered_patient_ids_sha256": outer_training_ordered_ids_sha256,
        "expected_outer_test_ordered_patient_ids_sha256": expected_outer_test_ordered_ids_sha256,
        "canonical_outer_training_input_identity_sha256": canonical_outer_training_input_identity_sha256,
        "ae_ratio": candidate.ae_ratio,
        "latent_dimensions": dict(candidate.latent_dimensions),
        "fused_latent_width": candidate.fused_latent_width,
        "logistic_c": candidate.logistic_c,
        "fused_feature_names_sha256": fused_feature_hash,
        "frozen_model_state_sha256": frozen_model_state_sha256,
    })


@dataclass(frozen=True)
class FinalRefitSeedBook:
    ae_split_seed: int
    epoch_selection_model_seeds: Mapping[str, int]
    final_ae_model_seeds: Mapping[str, int]
    ctgan_seed: int
    logistic_seed: int

    def __post_init__(self) -> None:
        _seed("ae_split_seed", self.ae_split_seed)
        object.__setattr__(self, "epoch_selection_model_seeds", MappingProxyType(_mapping_of_modality_seeds("epoch_selection_model_seeds", self.epoch_selection_model_seeds)))
        object.__setattr__(self, "final_ae_model_seeds", MappingProxyType(_mapping_of_modality_seeds("final_ae_model_seeds", self.final_ae_model_seeds)))
        _seed("ctgan_seed", self.ctgan_seed)
        _seed("logistic_seed", self.logistic_seed)

    def evidence(self) -> dict[str, Any]:
        return {
            "ae_split_seed": self.ae_split_seed,
            "epoch_selection_model_seeds": _mapping_of_modality_seeds("epoch_selection_model_seeds", self.epoch_selection_model_seeds),
            "final_ae_model_seeds": _mapping_of_modality_seeds("final_ae_model_seeds", self.final_ae_model_seeds),
            "ctgan_seed": self.ctgan_seed,
            "logistic_seed": self.logistic_seed,
        }


@dataclass
class FinalPrimaryModelBundle:
    selected_candidate: PrimaryCandidate
    seed_identity: Mapping[str, Any]
    modality_refits: Mapping[str, FittedEncoder]
    fused_feature_names: tuple[str, ...]
    fused_feature_hash: str
    fused_latent_dimensions: Mapping[str, int]
    ctgan_evidence: Mapping[str, Any]
    classifier: FittedLogisticClassifier
    cross_fitted_calibration: CrossFittedCalibrationResult
    final_calibrator: FinalSigmoidCalibrator
    threshold: OperationalThresholdResult
    outer_training_ids_hash: str
    expected_outer_test_ids_hash: str
    evidence: Mapping[str, Any]
    candidate_id: str
    candidate_identity_sha256: str
    final_model_identity_sha256: str
    search_selection_identity_sha256: str


@dataclass(frozen=True)
class OuterPrediction:
    sample_id: str
    raw_decision_score: float
    uncalibrated_probability: float
    calibrated_probability: float
    threshold: float
    predicted_label: int
    candidate_id: str
    candidate_identity_sha256: str
    final_model_identity_sha256: str


@dataclass(frozen=True)
class FrozenOuterPredictions:
    predictions: tuple[OuterPrediction, ...]
    patient_ids_hash: str
    prediction_hash: str
    evidence: Mapping[str, Any]
    candidate_id: str
    candidate_identity_sha256: str
    final_model_identity_sha256: str
    search_selection_identity_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


def _validate_candidate(candidate: PrimaryCandidate, feature_contracts: Mapping[str, Sequence[str]]) -> dict[str, int]:
    if not isinstance(candidate, PrimaryCandidate):
        raise TypeError("Final refit requires one selected PrimaryCandidate.")
    if candidate.augmentation != "minority_only_ctgan" or candidate.class_weight is not None:
        raise ValueError("Final refit accepts only minority_only_ctgan with class_weight=None.")
    if candidate.ae_ratio not in RATIOS or candidate.logistic_c not in LOGISTIC_CS:
        raise ValueError("Selected candidate is outside the frozen primary candidate grid.")
    if set(feature_contracts) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Feature contracts must contain exactly mGE, mDM, and mCNA.")
    dimensions: dict[str, int] = {}
    for modality in FUSION_MODALITY_ORDER:
        names = tuple(feature_contracts[modality])
        if not names or len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{modality} feature contract is invalid.")
        expected = latent_dim_from_ratio(len(names), candidate.ae_ratio)
        if candidate.latent_dimensions.get(modality) != expected:
            raise ValueError("Selected candidate latent dimensions do not match ratio and feature counts.")
        dimensions[modality] = expected
    if set(candidate.latent_dimensions) != set(FUSION_MODALITY_ORDER) or candidate.fused_latent_width != sum(dimensions.values()):
        raise ValueError("Selected candidate fused latent width is invalid.")
    canonical_id = (
        f"primary:ratio_{candidate.ae_ratio:.2f}:mGE_{dimensions['mGE']}:mDM_{dimensions['mDM']}:"
        f"mCNA_{dimensions['mCNA']}:width_{sum(dimensions.values())}:C_{candidate.logistic_c:g}"
    )
    if candidate.candidate_id != canonical_id:
        raise ValueError("Selected candidate ID does not match its frozen primary configuration.")
    return dimensions


def _validate_modalities(modalities: Mapping[str, pd.DataFrame], ids: tuple[str, ...], feature_contracts: Mapping[str, Sequence[str]], name: str) -> None:
    if not isinstance(modalities, Mapping) or set(modalities) != set(FUSION_MODALITY_ORDER):
        raise ValueError(f"{name} modalities must contain exactly mGE, mDM, and mCNA.")
    for modality in FUSION_MODALITY_ORDER:
        frame = modalities[modality]
        if not isinstance(frame, pd.DataFrame) or tuple(str(value) for value in frame.index) != ids:
            raise ValueError(f"{name} {modality} DataFrame index must match declared SAMPLE_ID order.")
        if tuple(frame.columns) != tuple(feature_contracts[modality]):
            raise ValueError(f"{name} {modality} columns differ from its feature contract.")


def _canonical_final_refit_training_inputs(
    aligned_data: Mapping[str, Any],
    context: Any,
    run_provenance: PrimaryV1RunProvenance,
) -> tuple[np.ndarray, dict[str, pd.DataFrame], dict[str, tuple[str, ...]], dict[str, Any]]:
    """Derive the only training inputs final refit may use from validated authority."""
    try:
        all_ids = tuple(str(value) for value in aligned_data["sample_ids"])
        all_labels = np.asarray(aligned_data["y_binary"], dtype=int)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Canonical aligned data lacks final-refit labels.") from error
    if len(all_ids) != len(all_labels) or len(set(all_ids)) != len(all_ids):
        raise ValueError("Canonical aligned data labels are not SAMPLE_ID aligned.")
    label_by_id = dict(zip(all_ids, all_labels.tolist()))
    training_ids = context.outer_training_ids
    if any(sample_id not in label_by_id for sample_id in training_ids):
        raise ValueError("Official outer-training IDs are absent from canonical aligned labels.")
    training_labels = np.asarray([label_by_id[sample_id] for sample_id in training_ids], dtype=int)
    contracts: dict[str, tuple[str, ...]] = {}
    modalities: dict[str, pd.DataFrame] = {}
    modality_hashes: dict[str, str] = {}
    for modality in FUSION_MODALITY_ORDER:
        binding = context.modality_adapter.bindings[modality]
        names = tuple(aligned_data.get("feature_columns", {}).get(binding.feature_key, ()))
        frame = aligned_data.get(binding.matrix_key)
        if not names or not isinstance(frame, pd.DataFrame):
            raise ValueError("Canonical aligned data lacks a final-refit modality contract.")
        canonical = frame.loc[list(training_ids), list(names)].copy()
        contracts[modality] = names
        modalities[modality] = canonical
        modality_hashes[modality] = aligned_matrix_content_sha256(modality, canonical, training_ids, names)
    labels_hash = payload_sha256([
        {"sample_id": sample_id, "true_label": int(label)}
        for sample_id, label in zip(training_ids, training_labels)
    ])
    contracts_hash = payload_sha256({modality: list(contracts[modality]) for modality in FUSION_MODALITY_ORDER})
    input_identity = payload_sha256({
        "schema_version": "primary-v1-final-refit-canonical-training-input-v1",
        "aligned_data_content_identity_sha256": run_provenance.aligned_data_content_identity_sha256,
        "ordered_training_sample_ids_sha256": ordered_patient_ids_sha256(training_ids),
        "training_labels_sha256": labels_hash,
        "feature_contracts_sha256": contracts_hash,
        "modality_content_sha256": modality_hashes,
    })
    return training_labels, modalities, contracts, {
        "canonical_aligned_data_content_identity_sha256": run_provenance.aligned_data_content_identity_sha256,
        "canonical_outer_training_labels_sha256": labels_hash,
        "canonical_outer_training_modality_content_sha256": modality_hashes,
        "canonical_outer_training_feature_contracts_sha256": contracts_hash,
        "canonical_outer_training_input_identity_sha256": input_identity,
    }


def _validate_canonical_final_refit_training_evidence(
    outer_training_modalities: Mapping[str, pd.DataFrame],
    outer_training_sample_ids: Sequence[str],
    outer_training_labels: Sequence[int] | np.ndarray,
    feature_name_contracts: Mapping[str, Sequence[str]],
    context: Any,
    canonical_labels: np.ndarray,
    canonical_contracts: Mapping[str, tuple[str, ...]],
    canonical_evidence: Mapping[str, Any],
) -> None:
    submitted_labels = np.asarray(outer_training_labels, dtype=int)
    training_ids = _ids(outer_training_sample_ids, len(submitted_labels), "outer training")
    if tuple(training_ids) != context.outer_training_ids:
        raise ValueError("Final refit cohort differs from its trusted search context.")
    if submitted_labels.ndim != 1 or not np.array_equal(submitted_labels, canonical_labels):
        raise ValueError("Final refit training labels differ from canonical aligned data.")
    if not isinstance(feature_name_contracts, Mapping) or set(feature_name_contracts) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Final refit feature contracts are invalid.")
    if any(tuple(feature_name_contracts[modality]) != canonical_contracts[modality] for modality in FUSION_MODALITY_ORDER):
        raise ValueError("Final refit feature contracts differ from canonical aligned data.")
    _validate_modalities(outer_training_modalities, training_ids, canonical_contracts, "outer training")
    for modality in FUSION_MODALITY_ORDER:
        actual = aligned_matrix_content_sha256(modality, outer_training_modalities[modality], training_ids, canonical_contracts[modality])
        if actual != canonical_evidence["canonical_outer_training_modality_content_sha256"][modality]:
            raise ValueError("Final refit training modality values differ from canonical aligned data.")


def _validate_frozen_calibration(final_calibrator: FinalSigmoidCalibrator, threshold: OperationalThresholdResult, outer_training_ids_hash: str) -> None:
    if not isinstance(final_calibrator, FinalSigmoidCalibrator) or not isinstance(threshold, OperationalThresholdResult):
        raise TypeError("Final refit requires already-created final calibrator and operational threshold.")
    if final_calibrator.evidence.get("deployment_probability_source") != "final_sigmoid_calibrator_all_oof_refit":
        raise ValueError("Final calibrator evidence does not describe all-OOF deployment fitting.")
    if threshold.evidence.get("probability_source") != "cross_fitted_sigmoid_oof":
        raise ValueError("Operational threshold was not selected from cross-fitted sigmoid OOF probabilities.")
    if final_calibrator.evidence.get("outer_test_used") is not False or threshold.evidence.get("outer_test_used") is not False:
        raise ValueError("Frozen calibration or threshold evidence indicates outer-test use.")
    if final_calibrator.evidence.get("all_oof_patient_set_sha256") != outer_training_ids_hash:
        raise ValueError("Final calibrator OOF patients do not match complete outer-training patients.")
    if threshold.evidence.get("patient_set_sha256") != outer_training_ids_hash:
        raise ValueError("Operational threshold OOF patients do not match complete outer-training patients.")
    protocol = PrimaryProtocolV1()
    if final_calibrator.protocol_id != protocol.protocol_id or final_calibrator.protocol_sha256 != protocol.identity_sha256 or threshold.protocol_id != protocol.protocol_id or threshold.protocol_sha256 != protocol.identity_sha256 or final_calibrator.outer_training_patient_set_sha256 != outer_training_ids_hash or threshold.outer_training_patient_set_sha256 != outer_training_ids_hash:
        raise ValueError("Calibration or threshold protocol/cohort binding differs.")
    if not np.isfinite(float(threshold.threshold)) or not 0.0 <= float(threshold.threshold) <= 1.0:
        raise ValueError("Operational threshold must be finite and in [0, 1].")


def fit_final_primary_model(
    outer_training_modalities: Mapping[str, pd.DataFrame],
    outer_training_sample_ids: Sequence[str],
    outer_training_labels: Sequence[int] | np.ndarray,
    raw_outer_test_modalities: Mapping[str, pd.DataFrame],
    expected_outer_test_sample_ids: Sequence[str],
    feature_name_contracts: Mapping[str, Sequence[str]],
    ae_training_config: AutoencoderTrainingConfig,
    ctgan_config: CTGANConfig,
    seed_book: FinalRefitSeedBook,
    final_calibrator: FinalSigmoidCalibrator,
    threshold: OperationalThresholdResult,
    *,
    search_result: PrimarySearchResult,
    cross_fitted_calibration: CrossFittedCalibrationResult,
    protocol: PrimaryProtocolV1,
    synthetic_namespace: str,
    run_provenance: PrimaryV1RunProvenance,
    aligned_data: Mapping[str, Any],
    ae_validation_fraction: float = 0.20,
    binding_evidence: Mapping[str, Any] | None = None,
) -> FinalPrimaryModelBundle:
    """Refit one frozen primary candidate; outer-test labels are intentionally absent."""
    if not isinstance(protocol, PrimaryProtocolV1) or not isinstance(search_result, PrimarySearchResult) or search_result.context.protocol != protocol or not isinstance(seed_book, FinalRefitSeedBook) or not isinstance(ae_training_config, AutoencoderTrainingConfig) or not isinstance(ctgan_config, CTGANConfig):
        raise TypeError("Final refit configuration contracts are invalid.")
    validate_primary_search_result(search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    context, selected_search = search_result.context, search_result.selected_search
    selected_candidate = selected_search.selected_candidate
    protocol.validate_autoencoder_training_config(ae_training_config, ae_validation_fraction)
    protocol.validate_ctgan_config(ctgan_config)
    canonical_labels, canonical_modalities, canonical_contracts, canonical_evidence = _canonical_final_refit_training_inputs(aligned_data, context, run_provenance)
    _validate_canonical_final_refit_training_evidence(
        outer_training_modalities,
        outer_training_sample_ids,
        outer_training_labels,
        feature_name_contracts,
        context,
        canonical_labels,
        canonical_contracts,
        canonical_evidence,
    )
    dimensions = _validate_candidate(selected_candidate, canonical_contracts)
    training_labels = canonical_labels
    training_ids = context.outer_training_ids
    outer_training_modalities = canonical_modalities
    feature_name_contracts = canonical_contracts
    test_ids = _ids(expected_outer_test_sample_ids, len(expected_outer_test_sample_ids), "outer test")
    if set(training_ids).intersection(test_ids):
        raise ValueError("Outer training and outer test SAMPLE_ID values must be disjoint.")
    if not np.isin(training_labels, [0, 1]).all() or set(training_labels.tolist()) != {0, 1}:
        raise ValueError("Final refit requires both real outer-training binary classes.")
    _validate_modalities(raw_outer_test_modalities, test_ids, feature_name_contracts, "outer test")
    training_ids_hash = patient_set_sha256(training_ids)
    if tuple(test_ids) != context.outer_testing_ids:
        raise ValueError("Final refit outer-test cohort differs from its trusted search context.")
    if seed_book != context.seed_manifest.final_refit_seed_book():
        raise ValueError("Final refit seed book differs from regenerated Primary V1 seed manifest.")
    validate_cross_fitted_calibration_result(cross_fitted_calibration, search_result=search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    validate_final_sigmoid_calibrator(final_calibrator, selected_search=selected_search, context=context)
    validate_operational_threshold_result(threshold, calibration=cross_fitted_calibration, search_result=search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    _validate_frozen_calibration(final_calibrator, threshold, training_ids_hash)
    if selected_search.selected_candidate != selected_candidate or selected_search.selected_candidate_id != selected_candidate.candidate_id or selected_search.selected_candidate_identity_sha256 != selected_candidate.candidate_identity_sha256 or final_calibrator.candidate_id != selected_candidate.candidate_id or threshold.candidate_id != selected_candidate.candidate_id or final_calibrator.candidate_identity_sha256 != selected_candidate.candidate_identity_sha256 or threshold.candidate_identity_sha256 != selected_candidate.candidate_identity_sha256 or final_calibrator.selected_oof_predictions_sha256 != threshold.selected_oof_predictions_sha256 or final_calibrator.selected_oof_predictions_sha256 != selected_search.selected_oof_predictions_sha256 or final_calibrator.search_selection_identity_sha256 != selected_search.search_selection_identity_sha256 or threshold.search_selection_identity_sha256 != selected_search.search_selection_identity_sha256:
        raise ValueError("Final refit candidate/calibration/threshold provenance chain differs.")
    if binding_evidence is not None and binding_evidence.get("selected_candidate_id") not in {None, selected_candidate.candidate_id}:
        raise ValueError("Caller binding evidence candidate ID differs from the selected candidate.")
    split = build_autoencoder_split(training_ids, training_labels, ae_validation_fraction, seed_book.ae_split_seed)
    epoch_results: dict[str, Any] = {}
    refits: dict[str, FittedEncoder] = {}
    for modality in FUSION_MODALITY_ORDER:
        architecture = AutoencoderArchitecture(modality, len(feature_name_contracts[modality]), 128, dimensions[modality])
        selection = select_epoch_for_modality(
            outer_training_modalities[modality], training_ids, feature_name_contracts[modality], split,
            architecture, ae_training_config, seed_book.epoch_selection_model_seeds[modality],
        )
        refit = refit_selected_epoch_modality(
            outer_training_modalities[modality], training_ids, raw_outer_test_modalities[modality], test_ids,
            feature_name_contracts[modality], architecture, ae_training_config, selection.selected_epoch_count,
            seed_book.final_ae_model_seeds[modality],
        )
        epoch_results[modality] = selection
        refits[modality] = refit
    fused = fuse_fitted_encoders(refits, training_ids, test_ids)
    if fused.latent_dimensions != dimensions or fused.feature_names != build_latent_feature_names(dimensions):
        raise AssertionError("Final latent fusion differs from selected candidate schema.")
    augmented = augment_with_minority_ctgan(
        fused.training, training_labels, training_ids, fused.feature_names, fused.evidence["feature_names_sha256"],
        ctgan_config, seed_book.ctgan_seed, synthetic_namespace,
    )
    classifier = fit_logistic_classifier(augmented, LogisticRegressionConfig(selected_candidate.logistic_c), seed_book.logistic_seed)
    if classifier.feature_names != fused.feature_names:
        raise AssertionError("Final classifier feature schema differs from final fused schema.")
    seed_identity = seed_book.evidence()
    modality_evidence = {
        modality: {
            "selected_epoch_count": epoch_results[modality].selected_epoch_count,
            "temporary_ae_fit_ids_sha256": epoch_results[modality].evidence.get("fit_sample_ids_sha256"),
            "temporary_ae_stop_ids_sha256": epoch_results[modality].evidence.get("stop_sample_ids_sha256"),
            "temporary_preprocessing_fit_ids_sha256": epoch_results[modality].evidence.get("temporary_preprocessing_fit_sample_ids_sha256"),
            "epoch_selection_seed": seed_book.epoch_selection_model_seeds[modality],
            "final_fresh_ae_seed": seed_book.final_ae_model_seeds[modality],
            "final_preprocessing_fit_ids_sha256": refits[modality].evidence.get("preprocessing_fit_sample_ids_sha256"),
            "architecture": refits[modality].evidence.get("architecture"),
            "validation_data_used_final": False,
            "early_stopping_used_final": False,
            "outer_test_used_for_fit": False,
            "fresh_refit": True,
        }
        for modality in FUSION_MODALITY_ORDER
    }
    evidence = {
        "schema_version": _SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.identity_sha256,
        "run_provenance_identity_sha256": run_provenance.identity_sha256,
        "selected_candidate": _candidate_payload(selected_candidate),
        "seed_identity": seed_identity,
        "seed_identity_sha256": payload_sha256(seed_identity),
        "outer_training_ids_sha256": training_ids_hash,
        "outer_training_ordered_patient_ids_sha256": ordered_patient_ids_sha256(training_ids),
        **canonical_evidence,
        "expected_outer_test_ordered_patient_ids_sha256": ordered_patient_ids_sha256(test_ids),
        "expected_outer_test_modality_hashes": {modality: _frame_hash(raw_outer_test_modalities[modality]) for modality in FUSION_MODALITY_ORDER},
        "feature_contracts_sha256": payload_sha256({name: list(feature_name_contracts[name]) for name in FUSION_MODALITY_ORDER}),
        "ae_split": dict(split.evidence),
        "modality_epoch_evidence": modality_evidence,
        "fused_latent": dict(fused.evidence),
        "ctgan": dict(augmented.evidence),
        "classifier": dict(classifier.evidence),
        "final_calibrator_evidence": dict(final_calibrator.evidence),
        "final_calibrator_evidence_sha256": payload_sha256(dict(final_calibrator.evidence)),
        "candidate_id": selected_candidate.candidate_id,
        "candidate_identity_sha256": selected_candidate.candidate_identity_sha256,
        "selected_ae_ratio": selected_candidate.ae_ratio,
        "selected_latent_dimensions": dict(dimensions),
        "selected_fused_latent_width": selected_candidate.fused_latent_width,
        "selected_logistic_c": selected_candidate.logistic_c,
        "search_selection_identity_sha256": selected_search.search_selection_identity_sha256,
        "selected_oof_predictions_sha256": final_calibrator.selected_oof_predictions_sha256,
        "cross_fitted_calibration_sha256": threshold.cross_fitted_calibration_sha256,
        "final_calibrator_identity_sha256": final_calibrator.final_calibrator_identity_sha256,
        "threshold_identity_sha256": threshold.threshold_identity_sha256,
        "operational_threshold": {"threshold": float(threshold.threshold), "metrics": asdict(threshold.metrics), "evidence": dict(threshold.evidence)},
        "operational_threshold_evidence_sha256": payload_sha256({"threshold": float(threshold.threshold), "metrics": asdict(threshold.metrics), "evidence": dict(threshold.evidence)}),
        "threshold_probability_source": "cross_fitted_sigmoid_oof",
        "deployment_probability_source": "final_sigmoid_calibrator_all_oof_refit",
        "threshold_transfer_exact_equivalence_guaranteed": False,
        "threshold_is_clinically_validated": False,
        "binding_evidence": dict(binding_evidence or {}),
        "scientific_state_frozen_before_outer_scoring": True,
        "outer_test_labels_seen": False,
        "final_refit_outer_test_latents_created_by_step5": True,
        "outer_test_classifier_scoring_performed": False,
    }
    evidence["frozen_model_state_sha256"] = _frozen_model_state_hash(refits, classifier, final_calibrator, threshold)
    evidence["final_model_identity_sha256"] = _final_model_identity(
        protocol=protocol,
        search_selection_identity_sha256=selected_search.search_selection_identity_sha256,
        candidate=selected_candidate,
        selected_oof_predictions_sha256=selected_search.selected_oof_predictions_sha256,
        cross_fitted_calibration_sha256=cross_fitted_calibration.cross_fitted_calibration_sha256,
        final_calibrator_identity_sha256=final_calibrator.final_calibrator_identity_sha256,
        threshold_identity_sha256=threshold.threshold_identity_sha256,
        outer_training_ids_sha256=training_ids_hash,
        outer_training_ordered_ids_sha256=ordered_patient_ids_sha256(training_ids),
        expected_outer_test_ordered_ids_sha256=ordered_patient_ids_sha256(test_ids),
        canonical_outer_training_input_identity_sha256=canonical_evidence["canonical_outer_training_input_identity_sha256"],
        fused_feature_hash=fused.evidence["feature_names_sha256"],
        frozen_model_state_sha256=evidence["frozen_model_state_sha256"],
    )
    bundle = FinalPrimaryModelBundle(selected_candidate, seed_identity, refits, fused.feature_names, fused.evidence["feature_names_sha256"], dimensions, augmented.evidence, classifier, cross_fitted_calibration, final_calibrator, threshold, training_ids_hash, patient_set_sha256(test_ids), evidence, selected_candidate.candidate_id, selected_candidate.candidate_identity_sha256, evidence["final_model_identity_sha256"], selected_search.search_selection_identity_sha256)
    validate_final_primary_model_bundle(bundle, search_result=search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    return bundle


def _prediction_payload(prediction: OuterPrediction) -> dict[str, Any]:
    return {
        "sample_id": prediction.sample_id,
        "raw_decision_score": prediction.raw_decision_score,
        "uncalibrated_probability": prediction.uncalibrated_probability,
        "calibrated_probability": prediction.calibrated_probability,
        "threshold": prediction.threshold,
        "predicted_label": prediction.predicted_label,
        "candidate_id": prediction.candidate_id,
        "candidate_identity_sha256": prediction.candidate_identity_sha256,
        "final_model_identity_sha256": prediction.final_model_identity_sha256,
    }


def validate_frozen_outer_predictions(frozen_predictions: FrozenOuterPredictions) -> None:
    if not isinstance(frozen_predictions, FrozenOuterPredictions) or not frozen_predictions.predictions:
        raise TypeError("A non-empty FrozenOuterPredictions payload is required.")
    predictions = tuple(frozen_predictions.predictions)
    ids = _ids([item.sample_id for item in predictions], len(predictions), "frozen prediction")
    if frozen_predictions.patient_ids_hash != ordered_patient_ids_sha256(ids):
        raise ValueError("Frozen prediction patient ID hash does not verify.")
    candidate_ids = {item.candidate_id for item in predictions}; candidate_hashes = {item.candidate_identity_sha256 for item in predictions}; model_hashes = {item.final_model_identity_sha256 for item in predictions}
    if any(not isinstance(item, OuterPrediction) or not np.isfinite(item.raw_decision_score) or not np.isfinite(item.uncalibrated_probability) or not np.isfinite(item.calibrated_probability) or not np.isfinite(item.threshold) or not 0.0 <= item.uncalibrated_probability <= 1.0 or not 0.0 <= item.calibrated_probability <= 1.0 or not 0.0 <= item.threshold <= 1.0 or item.predicted_label not in {0, 1} or item.predicted_label != int(item.calibrated_probability >= item.threshold) for item in predictions) or len(candidate_ids) != 1 or len(candidate_hashes) != 1 or len(model_hashes) != 1 or frozen_predictions.candidate_id not in candidate_ids or frozen_predictions.candidate_identity_sha256 not in candidate_hashes or frozen_predictions.final_model_identity_sha256 not in model_hashes:
        raise ValueError("Frozen prediction records are invalid.")
    if frozen_predictions.prediction_hash != payload_sha256([_prediction_payload(item) for item in predictions]):
        raise ValueError("Frozen prediction hash does not verify.")
    if frozen_predictions.evidence.get("prediction_state_frozen") is not True or frozen_predictions.evidence.get("outer_labels_seen") is not False or not isinstance(frozen_predictions.evidence.get("run_provenance_identity_sha256"), str) or len(frozen_predictions.evidence["run_provenance_identity_sha256"]) != 64 or not isinstance(frozen_predictions.evidence.get("aligned_data_content_identity_sha256"), str) or len(frozen_predictions.evidence["aligned_data_content_identity_sha256"]) != 64:
        raise ValueError("Frozen prediction evidence does not establish label-free state.")


def validate_final_primary_model_bundle(
    bundle: FinalPrimaryModelBundle,
    *,
    search_result: PrimarySearchResult,
    run_provenance: PrimaryV1RunProvenance,
    aligned_data: Mapping[str, Any],
) -> None:
    """Verify a frozen final model against its official search authority without fitting."""
    if not isinstance(bundle, FinalPrimaryModelBundle):
        raise TypeError("FinalPrimaryModelBundle is required.")
    validate_primary_search_result(search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    context, selected_search = search_result.context, search_result.selected_search
    candidate, protocol = selected_search.selected_candidate, context.protocol
    _, _, _, canonical_evidence = _canonical_final_refit_training_inputs(aligned_data, context, run_provenance)
    evidence = bundle.evidence
    if not isinstance(evidence, Mapping):
        raise ValueError("Final model bundle evidence is invalid.")
    if (
        bundle.selected_candidate != candidate
        or bundle.candidate_id != candidate.candidate_id
        or bundle.candidate_identity_sha256 != candidate.candidate_identity_sha256
        or bundle.search_selection_identity_sha256 != selected_search.search_selection_identity_sha256
    ):
        raise ValueError("Final model bundle candidate/search authority differs.")
    if set(bundle.modality_refits) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Final model bundle modalities differ from the frozen fusion order.")
    contracts: dict[str, tuple[str, ...]] = {}
    for modality in FUSION_MODALITY_ORDER:
        refit = bundle.modality_refits[modality]
        if not isinstance(refit, FittedEncoder):
            raise TypeError(f"Final model bundle {modality} refit is invalid.")
        names = tuple(refit.preprocessor.feature_names)
        architecture = AutoencoderArchitecture(modality, len(names), 128, candidate.latent_dimensions[modality])
        if refit.evidence.get("architecture") != architecture.evidence() or refit.evidence.get("complete_training_sample_ids_sha256") != ordered_patient_ids_sha256(context.outer_training_ids):
            raise ValueError(f"Final model bundle {modality} refit differs from the selected configuration.")
        contracts[modality] = names
    dimensions = _validate_candidate(candidate, contracts)
    expected_feature_names = build_latent_feature_names(dimensions)
    if (
        dict(bundle.fused_latent_dimensions) != dimensions
        or bundle.fused_feature_names != expected_feature_names
        or bundle.fused_feature_hash != payload_sha256(list(expected_feature_names))
    ):
        raise ValueError("Final model bundle fused latent schema differs from the selected candidate.")
    classifier = bundle.classifier
    model = getattr(classifier, "model", None)
    if (
        not isinstance(classifier, FittedLogisticClassifier)
        or classifier.feature_names != expected_feature_names
        or classifier.evidence.get("C") != float(candidate.logistic_c)
        or classifier.evidence.get("solver") != "liblinear"
        or classifier.evidence.get("penalty") != "l2"
        or classifier.evidence.get("max_iter") != 1000
        or classifier.evidence.get("class_weight") is not None
        or float(getattr(model, "C", np.nan)) != float(candidate.logistic_c)
        or getattr(model, "solver", None) != "liblinear"
        or getattr(model, "penalty", None) != "l2"
        or getattr(model, "max_iter", None) != 1000
        or getattr(model, "class_weight", object()) is not None
    ):
        raise ValueError("Final classifier configuration differs from the official selected candidate.")
    outer_training_ids_hash = patient_set_sha256(context.outer_training_ids)
    selected_oof_hash = selected_search.selected_oof_predictions_sha256
    calibration = bundle.cross_fitted_calibration
    calibration_records = tuple(getattr(calibration, "predictions", ()))
    trusted_oof_by_id = {record.sample_id: record for record in selected_search.selected_oof_predictions}
    if (
        not isinstance(calibration, CrossFittedCalibrationResult)
        or not calibration_records
        or any(not isinstance(record, CalibratedOOFPrediction) for record in calibration_records)
        or calibration_records != tuple(sorted(calibration_records, key=lambda item: item.sample_id))
        or {record.sample_id for record in calibration_records} != set(trusted_oof_by_id)
        or any(
            record.sample_id not in trusted_oof_by_id
            or record.inner_fold_id != trusted_oof_by_id[record.sample_id].inner_fold_id
            or record.true_label != trusted_oof_by_id[record.sample_id].true_label
            or record.decision_score != trusted_oof_by_id[record.sample_id].decision_score
            or record.candidate_id != candidate.candidate_id
            or record.candidate_identity_sha256 != candidate.candidate_identity_sha256
            or record.selected_oof_predictions_sha256 != selected_oof_hash
            or not np.isfinite(record.cross_fitted_probability)
            or not 0.0 <= record.cross_fitted_probability <= 1.0
            for record in calibration_records
        )
    ):
        raise ValueError("Final model bundle cross-fitted calibration evidence differs from selected OOF evidence.")
    actual_cross_fitted_calibration_sha256 = payload_sha256([
        {
            "sample_id": record.sample_id,
            "inner_fold_id": record.inner_fold_id,
            "true_label": record.true_label,
            "decision_score": record.decision_score,
            "cross_fitted_probability": record.cross_fitted_probability,
            "candidate_id": record.candidate_id,
            "candidate_identity_sha256": record.candidate_identity_sha256,
            "selected_oof_predictions_sha256": record.selected_oof_predictions_sha256,
        }
        for record in calibration_records
    ])
    if (
        calibration.candidate_id != candidate.candidate_id
        or calibration.candidate_identity_sha256 != candidate.candidate_identity_sha256
        or calibration.selected_oof_predictions_sha256 != selected_oof_hash
        or calibration.search_selection_identity_sha256 != selected_search.search_selection_identity_sha256
        or calibration.protocol_id != protocol.protocol_id
        or calibration.protocol_sha256 != protocol.identity_sha256
        or calibration.outer_training_patient_set_sha256 != outer_training_ids_hash
        or calibration.cross_fitted_calibration_sha256 != actual_cross_fitted_calibration_sha256
        or calibration.evidence.get("cross_fitted_calibration_sha256") != actual_cross_fitted_calibration_sha256
    ):
        raise ValueError("Final model bundle cross-fitted calibration identity does not recompute.")
    calibrator, threshold = bundle.final_calibrator, bundle.threshold
    if (
        not isinstance(calibrator, FinalSigmoidCalibrator)
        or not isinstance(threshold, OperationalThresholdResult)
        or bundle.outer_training_ids_hash != outer_training_ids_hash
        or calibrator.candidate_id != candidate.candidate_id
        or threshold.candidate_id != candidate.candidate_id
        or calibrator.candidate_identity_sha256 != candidate.candidate_identity_sha256
        or threshold.candidate_identity_sha256 != candidate.candidate_identity_sha256
        or calibrator.selected_oof_predictions_sha256 != selected_oof_hash
        or threshold.selected_oof_predictions_sha256 != selected_oof_hash
        or calibrator.search_selection_identity_sha256 != selected_search.search_selection_identity_sha256
        or threshold.search_selection_identity_sha256 != selected_search.search_selection_identity_sha256
        or calibrator.protocol_id != protocol.protocol_id
        or calibrator.protocol_sha256 != protocol.identity_sha256
        or threshold.protocol_id != protocol.protocol_id
        or threshold.protocol_sha256 != protocol.identity_sha256
        or calibrator.outer_training_patient_set_sha256 != outer_training_ids_hash
        or threshold.outer_training_patient_set_sha256 != outer_training_ids_hash
        or calibrator.run_provenance_identity_sha256 != run_provenance.identity_sha256
        or threshold.run_provenance_identity_sha256 != run_provenance.identity_sha256
        or threshold.cross_fitted_calibration_sha256 != actual_cross_fitted_calibration_sha256
        or threshold.cross_fitted_calibration_sha256 != evidence.get("cross_fitted_calibration_sha256")
    ):
        raise ValueError("Final model bundle calibration/threshold authority differs.")
    calibrator_model = calibrator.model
    try:
        calibrator_coefficient = float(calibrator_model.coef_[0, 0])
        calibrator_intercept = float(calibrator_model.intercept_[0])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("Final sigmoid calibrator model state is invalid.") from error
    expected_calibrator_identity = payload_sha256({
        "protocol_sha256": protocol.identity_sha256,
        "run_provenance_identity_sha256": run_provenance.identity_sha256,
        "candidate_id": candidate.candidate_id,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "selected_oof_predictions_sha256": selected_oof_hash,
        "search_selection_identity_sha256": selected_search.search_selection_identity_sha256,
        "coefficient": calibrator_coefficient,
        "intercept": calibrator_intercept,
        "solver": "lbfgs",
        "max_iter": 1000,
    })
    expected_threshold_identity = payload_sha256({
        "protocol_sha256": protocol.identity_sha256,
        "run_provenance_identity_sha256": run_provenance.identity_sha256,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "selected_oof_predictions_sha256": selected_oof_hash,
        "search_selection_identity_sha256": selected_search.search_selection_identity_sha256,
        "cross_fitted_calibration_sha256": threshold.cross_fitted_calibration_sha256,
        "threshold": float(threshold.threshold),
    })
    if (
        getattr(calibrator_model, "solver", None) != "lbfgs"
        or getattr(calibrator_model, "max_iter", None) != 1000
        or calibrator.final_calibrator_identity_sha256 != expected_calibrator_identity
        or calibrator.evidence.get("final_calibrator_identity_sha256") != expected_calibrator_identity
        or threshold.threshold_identity_sha256 != expected_threshold_identity
        or threshold.evidence.get("threshold_identity_sha256") != expected_threshold_identity
    ):
        raise ValueError("Final model bundle calibrator or threshold identity does not recompute.")
    expected_evidence = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.identity_sha256,
        "candidate_id": candidate.candidate_id,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "selected_ae_ratio": candidate.ae_ratio,
        "selected_latent_dimensions": dimensions,
        "selected_fused_latent_width": candidate.fused_latent_width,
        "selected_logistic_c": candidate.logistic_c,
        "search_selection_identity_sha256": selected_search.search_selection_identity_sha256,
        "selected_oof_predictions_sha256": selected_oof_hash,
        "final_calibrator_identity_sha256": calibrator.final_calibrator_identity_sha256,
        "threshold_identity_sha256": threshold.threshold_identity_sha256,
        "outer_training_ids_sha256": outer_training_ids_hash,
        "outer_training_ordered_patient_ids_sha256": ordered_patient_ids_sha256(context.outer_training_ids),
        **canonical_evidence,
    }
    if evidence.get("run_provenance_identity_sha256") != run_provenance.identity_sha256 or any(evidence.get(key) != value for key, value in expected_evidence.items()):
        raise ValueError("Final model bundle evidence differs from official selected configuration.")
    if not isinstance(evidence.get("fused_latent"), Mapping) or evidence["fused_latent"].get("feature_names_sha256") != bundle.fused_feature_hash:
        raise ValueError("Final model bundle fused latent evidence differs from its schema.")
    if bundle.ctgan_evidence.get("strategy") != "minority_only_ctgan" or bundle.ctgan_evidence.get("training_ids_sha256") != ordered_patient_ids_sha256(context.outer_training_ids):
        raise ValueError("Final model bundle CTGAN evidence differs from outer-training-only augmentation.")
    _validate_frozen_calibration(calibrator, threshold, outer_training_ids_hash)
    actual_state_hash = _frozen_model_state_hash(bundle.modality_refits, classifier, calibrator, threshold)
    if evidence.get("frozen_model_state_sha256") != actual_state_hash:
        raise ValueError("Frozen final model state changed after final refit.")
    expected_identity = _final_model_identity(
        protocol=protocol,
        search_selection_identity_sha256=selected_search.search_selection_identity_sha256,
        candidate=candidate,
        selected_oof_predictions_sha256=selected_oof_hash,
        cross_fitted_calibration_sha256=actual_cross_fitted_calibration_sha256,
        final_calibrator_identity_sha256=calibrator.final_calibrator_identity_sha256,
        threshold_identity_sha256=threshold.threshold_identity_sha256,
        outer_training_ids_sha256=outer_training_ids_hash,
        outer_training_ordered_ids_sha256=ordered_patient_ids_sha256(context.outer_training_ids),
        expected_outer_test_ordered_ids_sha256=evidence.get("expected_outer_test_ordered_patient_ids_sha256"),
        canonical_outer_training_input_identity_sha256=evidence.get("canonical_outer_training_input_identity_sha256"),
        fused_feature_hash=bundle.fused_feature_hash,
        frozen_model_state_sha256=actual_state_hash,
    )
    if bundle.final_model_identity_sha256 != expected_identity or evidence.get("final_model_identity_sha256") != expected_identity:
        raise ValueError("Final model identity does not recompute from official authority and fitted state.")


def score_outer_test(
    frozen_final_model: FinalPrimaryModelBundle,
    raw_outer_test_modalities: Mapping[str, pd.DataFrame],
    outer_test_sample_ids: Sequence[str],
    *,
    search_result: PrimarySearchResult,
    run_provenance: PrimaryV1RunProvenance,
    aligned_data: Mapping[str, Any],
) -> FrozenOuterPredictions:
    """Apply only frozen state. This API deliberately has no label argument."""
    if not isinstance(frozen_final_model, FinalPrimaryModelBundle):
        raise TypeError("Outer scoring requires FinalPrimaryModelBundle.")
    validate_final_primary_model_bundle(frozen_final_model, search_result=search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    if frozen_final_model.evidence.get("scientific_state_frozen_before_outer_scoring") is not True:
        raise ValueError("Final model bundle is not frozen for outer scoring.")
    ids = _ids(outer_test_sample_ids, len(outer_test_sample_ids), "outer test")
    if patient_set_sha256(ids) != frozen_final_model.expected_outer_test_ids_hash:
        raise ValueError("Outer scoring SAMPLE_ID values differ from the frozen expected outer test set.")
    contracts = {modality: tuple(frozen_final_model.modality_refits[modality].preprocessor.feature_names) for modality in FUSION_MODALITY_ORDER}
    _validate_modalities(raw_outer_test_modalities, ids, contracts, "outer test")
    expected_hashes = frozen_final_model.evidence.get("expected_outer_test_modality_hashes")
    actual_hashes = {modality: _frame_hash(raw_outer_test_modalities[modality]) for modality in FUSION_MODALITY_ORDER}
    if expected_hashes != actual_hashes:
        raise ValueError("Outer-test raw modality values differ from the frozen final-refit input.")
    parts: list[np.ndarray] = []
    for modality in FUSION_MODALITY_ORDER:
        refit = frozen_final_model.modality_refits[modality]
        partition = transform_with_preprocessor(refit.preprocessor, raw_outer_test_modalities[modality], ids, contracts[modality])
        latent = np.asarray(refit.encoder.predict(partition.matrix, verbose=0), dtype=np.float32)
        expected = (len(ids), frozen_final_model.fused_latent_dimensions[modality])
        if latent.shape != expected or not np.isfinite(latent).all():
            raise ValueError(f"Frozen {modality} encoder produced invalid outer-test latents.")
        parts.append(latent)
    fused = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if fused.shape != (len(ids), len(frozen_final_model.fused_feature_names)) or not np.isfinite(fused).all():
        raise ValueError("Frozen outer-test fusion is invalid.")
    scores = score_logistic_classifier(frozen_final_model.classifier, fused, ids, frozen_final_model.fused_feature_names)
    calibrated, calibrated_ids = apply_final_sigmoid_calibrator(frozen_final_model.final_calibrator, scores.decision_scores, ids)
    if calibrated_ids != ids:
        raise AssertionError("Calibrator did not preserve outer-test patient order.")
    frozen_threshold = float(frozen_final_model.threshold.threshold)
    predictions = tuple(
        OuterPrediction(sample_id, float(decision), float(probability), float(calibration), frozen_threshold, int(calibration >= frozen_threshold), frozen_final_model.candidate_id, frozen_final_model.candidate_identity_sha256, frozen_final_model.final_model_identity_sha256)
        for sample_id, decision, probability, calibration in zip(ids, scores.decision_scores, scores.probabilities, calibrated)
    )
    evidence = {
        "schema_version": _SCHEMA_VERSION,
        "protocol_id": search_result.context.protocol.protocol_id,
        "protocol_sha256": search_result.context.protocol.identity_sha256,
        "candidate_id": frozen_final_model.selected_candidate.candidate_id,
        "candidate_identity_sha256": frozen_final_model.candidate_identity_sha256,
        "ordered_patient_ids_sha256": ordered_patient_ids_sha256(ids),
        "final_model_identity_sha256": frozen_final_model.evidence["final_model_identity_sha256"],
        "search_selection_identity_sha256": frozen_final_model.search_selection_identity_sha256,
        "run_provenance_identity_sha256": run_provenance.identity_sha256,
        "aligned_data_content_identity_sha256": run_provenance.aligned_data_content_identity_sha256,
        "outer_manifest_identity_sha256": search_result.context.outer_manifest_identity_sha256,
        "inner_manifest_identity_sha256": search_result.context.inner_manifest_identity_sha256,
        "fold_authority_identity_sha256": search_result.context.fold_authority_identity_sha256,
        "frozen_model_state_sha256": frozen_final_model.evidence["frozen_model_state_sha256"],
        "fused_feature_names_sha256": frozen_final_model.fused_feature_hash,
        "score_source": "frozen_logistic_decision_function",
        "uncalibrated_probability_source": "frozen_logistic_predict_proba_class_1",
        "calibrated_probability_source": "final_sigmoid_calibrator_all_oof_refit",
        "threshold_source": "cross_fitted_sigmoid_oof",
        "prediction_convention": "calibrated_probability >= threshold",
        "prediction_state_frozen": True,
        "outer_labels_seen": False,
        "persistent_prediction_publication_required_before_runner_evaluation": True,
        "fit_calls_performed": 0,
        "outer_test_encoder_passes": 1,
        "outer_test_classifier_scoring_passes": 1,
    }
    result = FrozenOuterPredictions(predictions, ordered_patient_ids_sha256(ids), payload_sha256([_prediction_payload(item) for item in predictions]), evidence, frozen_final_model.candidate_id, frozen_final_model.candidate_identity_sha256, frozen_final_model.final_model_identity_sha256, frozen_final_model.search_selection_identity_sha256)
    validate_frozen_outer_predictions(result)
    return result
