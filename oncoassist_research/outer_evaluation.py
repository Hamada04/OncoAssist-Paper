"""Label-gated evaluation of already frozen outer-test predictions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from .artifacts import payload_sha256
from .final_refit import FinalPrimaryModelBundle, FrozenOuterPredictions, validate_final_primary_model_bundle, validate_frozen_outer_predictions
from .metrics import BinaryMetrics, RankingMetrics, compute_binary_metrics, compute_ranking_metrics
from .protocol import PrimaryProtocolV1, PrimaryV1RunProvenance, ordered_patient_ids_sha256, patient_set_sha256, validate_primary_v1_run_provenance
from .search import PrimarySearchResult, validate_primary_search_result


@dataclass(frozen=True)
class OuterEvaluation:
    primary_ranking_metrics: RankingMetrics
    operational_metrics: BinaryMetrics
    brier_score: float
    log_loss: float
    secondary_diagnostics: Mapping[str, Any]
    evidence: Mapping[str, Any]


def _canonical_outer_test_labels(aligned_data: Mapping[str, Any], outer_test_ids: Sequence[str]) -> np.ndarray:
    """Reconstruct label-gated evaluation targets from validated aligned data."""
    try:
        sample_ids = tuple(str(value) for value in aligned_data["sample_ids"])
        labels = np.asarray(aligned_data["y_binary"], dtype=int)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Canonical aligned data lacks outer-evaluation labels.") from error
    if len(sample_ids) != len(labels) or len(set(sample_ids)) != len(sample_ids) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Canonical aligned data labels are invalid for outer evaluation.")
    label_by_id = dict(zip(sample_ids, labels.tolist()))
    ids = tuple(str(value) for value in outer_test_ids)
    if any(sample_id not in label_by_id for sample_id in ids):
        raise ValueError("Official outer-test IDs are absent from canonical aligned labels.")
    return np.asarray([label_by_id[sample_id] for sample_id in ids], dtype=int)


def evaluate_outer_test_predictions(
    frozen_predictions: FrozenOuterPredictions,
    outer_test_labels: Sequence[int] | np.ndarray,
    expected_outer_test_ids: Sequence[str],
    *,
    run_provenance: PrimaryV1RunProvenance,
    aligned_data: Mapping[str, Any],
    search_result: PrimarySearchResult | None = None,
    frozen_final_model: FinalPrimaryModelBundle | None = None,
    scoring_publication_receipt: Mapping[str, Any] | None = None,
) -> OuterEvaluation:
    """Evaluate immutable predictions; this is the sole Step 9 label-accepting API."""
    validate_frozen_outer_predictions(frozen_predictions)
    predictions = frozen_predictions.predictions
    ids = tuple(item.sample_id for item in predictions)
    expected = tuple(str(value) for value in expected_outer_test_ids)
    evidence = frozen_predictions.evidence
    if expected != ids or len(expected) != len(set(expected)) or any(not value.strip() for value in expected):
        raise ValueError("Evaluation expected SAMPLE_ID order must exactly match frozen predictions.")
    if scoring_publication_receipt is not None:
        if search_result is not None or frozen_final_model is not None:
            raise ValueError("Published-scoring evaluation must not mix live model and receipt authority.")
        protocol = PrimaryProtocolV1()
        validate_primary_v1_run_provenance(run_provenance, protocol=protocol, aligned_data=aligned_data)
        receipt = dict(scoring_publication_receipt)
        if (
            receipt.get("schema_version") != "controlled-primary-v1-scoring-receipt-v1"
            or receipt.get("run_provenance_identity_sha256") != run_provenance.identity_sha256
            or receipt.get("protocol_sha256") != protocol.identity_sha256
            or receipt.get("aligned_data_content_identity_sha256") != run_provenance.aligned_data_content_identity_sha256
            or receipt.get("search_selection_identity_sha256") != frozen_predictions.search_selection_identity_sha256
            or receipt.get("selected_candidate_identity_sha256") != frozen_predictions.candidate_identity_sha256
            or receipt.get("final_model_identity_sha256") != frozen_predictions.final_model_identity_sha256
            or receipt.get("frozen_model_state_sha256") != evidence.get("frozen_model_state_sha256")
            or receipt.get("ordered_outer_test_ids_sha256") != ordered_patient_ids_sha256(expected)
            or receipt.get("frozen_prediction_hash") != frozen_predictions.prediction_hash
            or evidence.get("run_provenance_identity_sha256") != run_provenance.identity_sha256
            or evidence.get("protocol_sha256") != protocol.identity_sha256
            or evidence.get("search_selection_identity_sha256") != receipt.get("search_selection_identity_sha256")
            or evidence.get("candidate_identity_sha256") != receipt.get("selected_candidate_identity_sha256")
            or evidence.get("final_model_identity_sha256") != receipt.get("final_model_identity_sha256")
            or evidence.get("aligned_data_content_identity_sha256") != run_provenance.aligned_data_content_identity_sha256
            or frozen_predictions.patient_ids_hash != ordered_patient_ids_sha256(expected)
        ):
            raise ValueError("Frozen predictions do not belong to the published scoring authority.")
        outer_manifest_identity = receipt.get("outer_manifest_identity_sha256")
        fold_authority_identity = receipt.get("fold_authority_identity_sha256")
        frozen_model_state = receipt.get("frozen_model_state_sha256")
    else:
        if search_result is None or frozen_final_model is None:
            raise TypeError("Evaluation requires either live search/final-model authority or a scoring receipt.")
        validate_primary_v1_run_provenance(run_provenance, protocol=search_result.context.protocol, aligned_data=aligned_data)
        validate_primary_search_result(search_result, run_provenance=run_provenance, aligned_data=aligned_data)
        validate_final_primary_model_bundle(frozen_final_model, search_result=search_result, run_provenance=run_provenance, aligned_data=aligned_data)
        context, selected = search_result.context, search_result.selected_search
        if (
            expected != context.outer_testing_ids
            or evidence.get("run_provenance_identity_sha256") != run_provenance.identity_sha256
            or evidence.get("protocol_id") != context.protocol.protocol_id
            or evidence.get("protocol_sha256") != context.protocol.identity_sha256
            or evidence.get("search_selection_identity_sha256") != selected.search_selection_identity_sha256
            or evidence.get("candidate_id") != selected.selected_candidate_id
            or evidence.get("candidate_identity_sha256") != selected.selected_candidate_identity_sha256
            or evidence.get("final_model_identity_sha256") != frozen_final_model.final_model_identity_sha256
            or evidence.get("outer_manifest_identity_sha256") != context.outer_manifest_identity_sha256
            or evidence.get("inner_manifest_identity_sha256") != context.inner_manifest_identity_sha256
            or evidence.get("fold_authority_identity_sha256") != context.fold_authority_identity_sha256
            or evidence.get("frozen_model_state_sha256") != frozen_final_model.evidence.get("frozen_model_state_sha256")
            or evidence.get("aligned_data_content_identity_sha256") != run_provenance.aligned_data_content_identity_sha256
            or frozen_predictions.candidate_id != selected.selected_candidate_id
            or frozen_predictions.candidate_identity_sha256 != selected.selected_candidate_identity_sha256
            or frozen_predictions.final_model_identity_sha256 != frozen_final_model.final_model_identity_sha256
            or frozen_predictions.search_selection_identity_sha256 != selected.search_selection_identity_sha256
            or frozen_predictions.patient_ids_hash != ordered_patient_ids_sha256(context.outer_testing_ids)
            or patient_set_sha256(ids) != frozen_final_model.expected_outer_test_ids_hash
        ):
            raise ValueError("Frozen predictions do not belong to the canonical run/final-model authority.")
        outer_manifest_identity = context.outer_manifest_identity_sha256
        fold_authority_identity = context.fold_authority_identity_sha256
        frozen_model_state = frozen_final_model.evidence["frozen_model_state_sha256"]
    canonical_labels = _canonical_outer_test_labels(aligned_data, expected)
    submitted_labels = np.asarray(outer_test_labels)
    if (
        submitted_labels.ndim != 1
        or len(submitted_labels) != len(ids)
        or not np.array_equal(submitted_labels, canonical_labels)
        or set(canonical_labels.tolist()) != {0, 1}
    ):
        raise ValueError("Outer evaluation labels must exactly match canonical ordered outer-test labels.")
    labels = canonical_labels
    raw_scores = np.asarray([item.raw_decision_score for item in predictions], dtype=float)
    calibrated = np.asarray([item.calibrated_probability for item in predictions], dtype=float)
    frozen_labels = np.asarray([item.predicted_label for item in predictions], dtype=int)
    threshold = float(predictions[0].threshold)
    if any(item.threshold != threshold for item in predictions):
        raise ValueError("Frozen predictions do not share one operational threshold.")
    if evidence.get("candidate_id") != frozen_predictions.candidate_id or evidence.get("candidate_identity_sha256") != frozen_predictions.candidate_identity_sha256 or evidence.get("final_model_identity_sha256") != frozen_predictions.final_model_identity_sha256 or evidence.get("search_selection_identity_sha256") != frozen_predictions.search_selection_identity_sha256:
        raise ValueError("Frozen prediction evidence identity does not match prediction records.")
    primary = compute_ranking_metrics(labels, raw_scores)
    operational = replace(
        compute_binary_metrics(labels, frozen_labels, 0.5, "decision_score"),
        threshold=threshold,
        score_kind="frozen_predicted_label",
        label="frozen_operational_threshold_metrics",
    )
    secondary = {
        "secondary_calibrated_probability_ranking_diagnostic": {
            "auprc": compute_ranking_metrics(labels, calibrated).auprc,
            "auroc": compute_ranking_metrics(labels, calibrated).auroc,
            "not_primary": True,
        }
    }
    evidence = {
        "prediction_hash": frozen_predictions.prediction_hash,
        "label_hash": payload_sha256(labels.astype(int).tolist()),
        "patient_ids_hash": frozen_predictions.patient_ids_hash,
        "candidate_id": frozen_predictions.candidate_id,
        "candidate_identity_sha256": frozen_predictions.candidate_identity_sha256,
        "final_model_identity_sha256": frozen_predictions.final_model_identity_sha256,
        "search_selection_identity_sha256": frozen_predictions.search_selection_identity_sha256,
        "run_provenance_identity_sha256": frozen_predictions.evidence["run_provenance_identity_sha256"],
        "outer_manifest_identity_sha256": outer_manifest_identity,
        "fold_authority_identity_sha256": fold_authority_identity,
        "frozen_model_state_sha256": frozen_model_state,
        "threshold": threshold,
        "discrimination_score_source": "raw_logistic_decision_score",
        "operational_label_source": "frozen_calibrated_probability_threshold_prediction",
        "calibration_probability_source": "final_sigmoid_calibrator",
        "outer_test_labels_used_only_for_evaluation": True,
        "post_test_adaptation_performed": False,
    }
    return OuterEvaluation(
        primary,
        operational,
        float(brier_score_loss(labels, calibrated)),
        float(log_loss(labels, calibrated, labels=[0, 1])),
        secondary,
        evidence,
    )
