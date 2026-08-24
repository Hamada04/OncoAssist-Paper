"""Content-validated operational threshold selection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from .artifacts import payload_sha256
from .calibration import CrossFittedCalibrationResult, validate_cross_fitted_calibration_result
from .metrics import BinaryMetrics, compute_binary_metrics
from .protocol import PrimaryV1RunProvenance, ordered_patient_ids_sha256, patient_set_sha256
from .search import PrimarySearchResult

@dataclass(frozen=True)
class OperationalThresholdResult:
    threshold: float; metrics: BinaryMetrics; evidence: Mapping[str, Any]; candidate_id: str; candidate_identity_sha256: str; selected_oof_predictions_sha256: str; cross_fitted_calibration_sha256: str; threshold_identity_sha256: str; search_selection_identity_sha256: str; protocol_id: str; protocol_sha256: str; outer_training_patient_set_sha256: str; run_provenance_identity_sha256: str
def threshold_candidates(probabilities: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)): raise ValueError("Threshold candidates require finite probabilities in [0,1].")
    return tuple(sorted({0.0, *(float(x) for x in values)}))
def select_operational_threshold(calibration: CrossFittedCalibrationResult, *, search_result: PrimarySearchResult, run_provenance: PrimaryV1RunProvenance, aligned_data: Mapping[str, Any]) -> OperationalThresholdResult:
    validate_cross_fitted_calibration_result(calibration, search_result=search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    records = calibration.predictions; ids = tuple(x.sample_id for x in records); labels = np.asarray([x.true_label for x in records]); probabilities = np.asarray([x.cross_fitted_probability for x in records]); candidates = threshold_candidates(probabilities); threshold, metrics = min(((x, compute_binary_metrics(labels, probabilities, x, "probability")) for x in candidates), key=lambda x: (-x[1].balanced_accuracy, -x[1].sensitivity, abs(x[0] - .5), x[0]))
    identity = payload_sha256({"protocol_sha256": calibration.protocol_sha256, "run_provenance_identity_sha256": calibration.run_provenance_identity_sha256, "candidate_identity_sha256": calibration.candidate_identity_sha256, "selected_oof_predictions_sha256": calibration.selected_oof_predictions_sha256, "search_selection_identity_sha256": calibration.search_selection_identity_sha256, "cross_fitted_calibration_sha256": calibration.cross_fitted_calibration_sha256, "threshold": float(threshold)})
    evidence = {"protocol_id": calibration.protocol_id, "protocol_sha256": calibration.protocol_sha256, "run_provenance_identity_sha256": calibration.run_provenance_identity_sha256, "patient_set_sha256": patient_set_sha256(ids), "ordered_patient_ids_sha256": ordered_patient_ids_sha256(ids), "candidate_generation": "sorted({0.0} union unique_cross_fitted_probabilities)", "objective": "balanced_accuracy", "probability_source": "cross_fitted_sigmoid_oof", "cross_fitted_calibration_sha256": calibration.cross_fitted_calibration_sha256, "search_selection_identity_sha256": calibration.search_selection_identity_sha256, "threshold_identity_sha256": identity, "outer_test_used": False, "threshold_is_clinically_validated": False}
    return OperationalThresholdResult(threshold, metrics, evidence, calibration.candidate_id, calibration.candidate_identity_sha256, calibration.selected_oof_predictions_sha256, calibration.cross_fitted_calibration_sha256, identity, calibration.search_selection_identity_sha256, calibration.protocol_id, calibration.protocol_sha256, calibration.outer_training_patient_set_sha256, calibration.run_provenance_identity_sha256)
def validate_operational_threshold_result(result: OperationalThresholdResult, *, calibration: CrossFittedCalibrationResult, search_result: PrimarySearchResult, run_provenance: PrimaryV1RunProvenance, aligned_data: Mapping[str, Any]) -> None:
    if not isinstance(result, OperationalThresholdResult): raise TypeError("OperationalThresholdResult is required.")
    expected = select_operational_threshold(calibration, search_result=search_result, run_provenance=run_provenance, aligned_data=aligned_data)
    if result != expected: raise ValueError("Operational threshold content does not recompute from validated calibration.")
