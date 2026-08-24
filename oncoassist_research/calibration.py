"""Content-validated sigmoid calibration for selected Primary V1 OOF scores."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from .artifacts import payload_sha256
from .protocol import PrimaryV1RunProvenance, ordered_patient_ids_sha256, patient_set_sha256
from .search import OOFPrediction, PrimarySearchContext, PrimarySearchResult, SelectedPrimarySearchArtifact, validate_primary_search_result, validate_selected_primary_search_artifact

@dataclass(frozen=True)
class SigmoidCalibrationConfig:
    solver: str = "lbfgs"
    max_iter: int = 1000
    def __post_init__(self):
        if self.solver != "lbfgs" or type(self.max_iter) is not int or self.max_iter < 1: raise ValueError("Primary sigmoid calibration requires solver='lbfgs' and positive max_iter.")

@dataclass(frozen=True)
class CalibratedOOFPrediction:
    sample_id: str; inner_fold_id: int; true_label: int; decision_score: float; cross_fitted_probability: float; candidate_id: str; candidate_identity_sha256: str; selected_oof_predictions_sha256: str

@dataclass(frozen=True)
class CrossFittedCalibrationResult:
    predictions: tuple[CalibratedOOFPrediction, ...]; fold_evidence: Mapping[int, Mapping[str, Any]]; diagnostics: Mapping[str, Any]; evidence: Mapping[str, Any]; candidate_id: str; candidate_identity_sha256: str; selected_oof_predictions_sha256: str; cross_fitted_calibration_sha256: str; search_selection_identity_sha256: str; protocol_id: str; protocol_sha256: str; outer_training_patient_set_sha256: str; run_provenance_identity_sha256: str

@dataclass
class FinalSigmoidCalibrator:
    model: LogisticRegression; evidence: Mapping[str, Any]; candidate_id: str; candidate_identity_sha256: str; selected_oof_predictions_sha256: str; final_calibrator_identity_sha256: str; search_selection_identity_sha256: str; protocol_id: str; protocol_sha256: str; outer_training_patient_set_sha256: str; run_provenance_identity_sha256: str

def _oof_hash(records): return payload_sha256([{"sample_id": x.sample_id, "inner_fold_id": x.inner_fold_id, "true_label": x.true_label, "decision_score": x.decision_score, "uncalibrated_probability": x.uncalibrated_probability, "candidate_id": x.candidate_id, "ae_ratio": x.ae_ratio, "logistic_c": x.logistic_c, "candidate_identity_sha256": x.candidate_identity_sha256} for x in sorted(records, key=lambda x: x.sample_id)])
def _fit(scores, labels, config):
    model = LogisticRegression(penalty=None, solver=config.solver, max_iter=config.max_iter, fit_intercept=True); model.fit(scores.reshape(-1, 1), labels); return model
def _probability(model, scores): return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, int(np.flatnonzero(model.classes_ == 1)[0])], dtype=float)
def _records(records, context):
    values = tuple(sorted(records, key=lambda x: x.sample_id)); ids = tuple(x.sample_id for x in values)
    if not values or len(ids) != len(set(ids)) or set(ids) != set(context.outer_training_ids) or any(not isinstance(x, OOFPrediction) or x.sample_id.startswith("SYNTHETIC:") or x.sample_id not in context.label_by_patient or x.true_label != context.label_by_patient[x.sample_id] or x.inner_fold_id != context.validation_fold_by_patient[x.sample_id] or not np.isfinite(x.decision_score) for x in values): raise ValueError("Calibration records are not selected real OOF evidence for the trusted cohort.")
    return values
def _selection(records, selected, context):
    validate_selected_primary_search_artifact(selected, context)
    if tuple(records) != selected.selected_oof_predictions or _oof_hash(records) != selected.selected_oof_predictions_sha256: raise ValueError("Calibration must consume exact selected-search OOF content.")

def _cross_fit(records, config, selected, context):
    values = _records(records, context); _selection(values, selected, context); candidate_id = values[0].candidate_id; candidate_hash = values[0].candidate_identity_sha256
    if any(x.candidate_id != candidate_id or x.candidate_identity_sha256 != candidate_hash for x in values): raise ValueError("Calibration requires one selected candidate.")
    predictions = []; evidence = {}
    for fold in range(3):
        held = tuple(x for x in values if context.validation_fold_by_patient[x.sample_id] == fold); train = tuple(x for x in values if context.validation_fold_by_patient[x.sample_id] != fold)
        if not held or not train: raise ValueError("Calibration requires exact three-fold OOF structure.")
        model = _fit(np.asarray([x.decision_score for x in train]), np.asarray([x.true_label for x in train]), config); probabilities = _probability(model, np.asarray([x.decision_score for x in held]))
        evidence[fold] = {"heldout_inner_fold_id": fold, "training_ordered_patient_ids_sha256": ordered_patient_ids_sha256([x.sample_id for x in train]), "validation_ordered_patient_ids_sha256": ordered_patient_ids_sha256([x.sample_id for x in held]), "coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]), "heldout_fold_excluded_from_fit": True}
        predictions.extend(CalibratedOOFPrediction(x.sample_id, fold, x.true_label, x.decision_score, float(p), candidate_id, candidate_hash, selected.selected_oof_predictions_sha256) for x, p in zip(held, probabilities))
    predictions = tuple(sorted(predictions, key=lambda x: x.sample_id)); labels = np.asarray([x.true_label for x in predictions]); probabilities = np.asarray([x.cross_fitted_probability for x in predictions]); config_hash = payload_sha256({"solver": config.solver, "max_iter": config.max_iter})
    identity = payload_sha256([{"sample_id": x.sample_id, "inner_fold_id": x.inner_fold_id, "true_label": x.true_label, "decision_score": x.decision_score, "cross_fitted_probability": x.cross_fitted_probability, "candidate_id": candidate_id, "candidate_identity_sha256": candidate_hash, "selected_oof_predictions_sha256": selected.selected_oof_predictions_sha256} for x in predictions])
    all_evidence = {"protocol_id": selected.protocol_id, "protocol_sha256": selected.protocol_sha256, "run_provenance_identity_sha256": context.run_provenance_identity_sha256, "calibration_config_identity_sha256": config_hash, "patient_set_sha256": patient_set_sha256([x.sample_id for x in predictions]), "ordered_patient_ids_sha256": ordered_patient_ids_sha256([x.sample_id for x in predictions]), "cross_fitted_calibration_sha256": identity, "search_selection_identity_sha256": selected.search_selection_identity_sha256, "outer_test_used": False, "threshold_probability_source": "cross_fitted_sigmoid_oof", "threshold_is_clinically_validated": False}
    diagnostics = {"brier_score": float(brier_score_loss(labels, probabilities)), "log_loss": float(log_loss(labels, probabilities, labels=[0, 1]))}
    return CrossFittedCalibrationResult(predictions, evidence, diagnostics, all_evidence, candidate_id, candidate_hash, selected.selected_oof_predictions_sha256, identity, selected.search_selection_identity_sha256, selected.protocol_id, selected.protocol_sha256, selected.outer_training_patient_set_sha256, context.run_provenance_identity_sha256)

def cross_fit_sigmoid_calibration(search_result: PrimarySearchResult, config: SigmoidCalibrationConfig = SigmoidCalibrationConfig(), *, run_provenance: PrimaryV1RunProvenance, aligned_data: Mapping[str, Any]) -> CrossFittedCalibrationResult:
    validate_primary_search_result(search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    return _cross_fit(search_result.selected_search.selected_oof_predictions, config, search_result.selected_search, search_result.context)

def validate_cross_fitted_calibration_result(calibration: CrossFittedCalibrationResult, *, search_result: PrimarySearchResult, run_provenance: PrimaryV1RunProvenance, aligned_data: Mapping[str, Any]) -> None:
    if not isinstance(calibration, CrossFittedCalibrationResult): raise TypeError("CrossFittedCalibrationResult is required.")
    validate_primary_search_result(search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    expected = _cross_fit(search_result.selected_search.selected_oof_predictions, SigmoidCalibrationConfig(), search_result.selected_search, search_result.context)
    if calibration != expected: raise ValueError("Cross-fitted calibration content does not recompute from selected OOF evidence.")

def fit_final_sigmoid_calibrator(records: Sequence[OOFPrediction], config: SigmoidCalibrationConfig = SigmoidCalibrationConfig(), *, selected_search: SelectedPrimarySearchArtifact, context: PrimarySearchContext) -> FinalSigmoidCalibrator:
    values = _records(records, context); _selection(values, selected_search, context); scores = np.asarray([x.decision_score for x in values]); labels = np.asarray([x.true_label for x in values]); model = _fit(scores, labels, config); candidate_id = values[0].candidate_id; candidate_hash = values[0].candidate_identity_sha256
    identity = payload_sha256({"protocol_sha256": selected_search.protocol_sha256, "run_provenance_identity_sha256": context.run_provenance_identity_sha256, "candidate_id": candidate_id, "candidate_identity_sha256": candidate_hash, "selected_oof_predictions_sha256": selected_search.selected_oof_predictions_sha256, "search_selection_identity_sha256": selected_search.search_selection_identity_sha256, "coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]), "solver": config.solver, "max_iter": config.max_iter})
    evidence = {"protocol_id": selected_search.protocol_id, "protocol_sha256": selected_search.protocol_sha256, "run_provenance_identity_sha256": context.run_provenance_identity_sha256, "calibration_config_identity_sha256": payload_sha256({"solver": config.solver, "max_iter": config.max_iter}), "all_oof_patient_set_sha256": patient_set_sha256([x.sample_id for x in values]), "all_oof_ordered_patient_ids_sha256": ordered_patient_ids_sha256([x.sample_id for x in values]), "selected_oof_predictions_sha256": selected_search.selected_oof_predictions_sha256, "search_selection_identity_sha256": selected_search.search_selection_identity_sha256, "final_calibrator_identity_sha256": identity, "coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]), "solver": config.solver, "max_iter": config.max_iter, "outer_test_used": False, "deployment_probability_source": "final_sigmoid_calibrator_all_oof_refit", "threshold_is_clinically_validated": False}
    return FinalSigmoidCalibrator(model, evidence, candidate_id, candidate_hash, selected_search.selected_oof_predictions_sha256, identity, selected_search.search_selection_identity_sha256, selected_search.protocol_id, selected_search.protocol_sha256, selected_search.outer_training_patient_set_sha256, context.run_provenance_identity_sha256)

def validate_final_sigmoid_calibrator(calibrator: FinalSigmoidCalibrator, *, selected_search: SelectedPrimarySearchArtifact, context: PrimarySearchContext) -> None:
    if not isinstance(calibrator, FinalSigmoidCalibrator): raise TypeError("FinalSigmoidCalibrator is required.")
    expected = fit_final_sigmoid_calibrator(selected_search.selected_oof_predictions, selected_search=selected_search, context=context)
    if calibrator.candidate_id != expected.candidate_id or calibrator.candidate_identity_sha256 != expected.candidate_identity_sha256 or calibrator.selected_oof_predictions_sha256 != expected.selected_oof_predictions_sha256 or calibrator.final_calibrator_identity_sha256 != expected.final_calibrator_identity_sha256 or calibrator.evidence != expected.evidence or calibrator.run_provenance_identity_sha256 != expected.run_provenance_identity_sha256 or not np.array_equal(calibrator.model.coef_, expected.model.coef_) or not np.array_equal(calibrator.model.intercept_, expected.model.intercept_): raise ValueError("Final sigmoid calibrator content does not recompute.")

def apply_final_sigmoid_calibrator(fitted: FinalSigmoidCalibrator, raw_decision_scores: Sequence[float], sample_ids: Sequence[str]):
    ids = tuple(str(x) for x in sample_ids); scores = np.asarray(raw_decision_scores, dtype=float)
    if not isinstance(fitted, FinalSigmoidCalibrator) or len(ids) != len(set(ids)) or len(ids) != len(scores): raise ValueError("Final sigmoid calibration application inputs are invalid.")
    result = np.array(_probability(fitted.model, scores), copy=True); result.setflags(write=False); return result, ids
