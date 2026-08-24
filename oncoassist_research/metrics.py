"""Binary High-TMB metrics with explicit ranking and threshold contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class RankingMetrics:
    auprc: float
    auroc: float


@dataclass(frozen=True)
class BinaryMetrics:
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    accuracy: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int
    threshold: float
    score_kind: str
    label: str = "diagnostic_uncalibrated_threshold_metrics"


def _validated(y_true: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    labels, values = np.asarray(y_true), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1 or not len(labels) or len(labels) != len(values):
        raise ValueError("Binary metrics require non-empty aligned one-dimensional labels and scores.")
    if not np.isin(labels, [0, 1]).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("Binary ranking metrics require both labels 0 and 1.")
    if not np.isfinite(values).all():
        raise ValueError("Binary metrics require finite scores.")
    return labels.astype(int), values


def compute_ranking_metrics(y_true: Sequence[int], decision_scores: Sequence[float]) -> RankingMetrics:
    labels, scores = _validated(y_true, decision_scores)
    return RankingMetrics(float(average_precision_score(labels, scores, pos_label=1)), float(roc_auc_score(labels, scores)))


def compute_binary_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float, score_kind: str) -> BinaryMetrics:
    if score_kind not in {"decision_score", "probability"} or not np.isfinite(float(threshold)):
        raise ValueError("Threshold metrics require an explicit finite threshold and score kind.")
    labels, values = _validated(y_true, scores)
    predicted = (values >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel().tolist()
    return BinaryMetrics(float(balanced_accuracy_score(labels, predicted)), float(recall_score(labels, predicted, pos_label=1, zero_division=0)), float(tn / (tn + fp)) if tn + fp else 0.0, float(precision_score(labels, predicted, pos_label=1, zero_division=0)), float(f1_score(labels, predicted, pos_label=1, zero_division=0)), float(accuracy_score(labels, predicted)), float(matthews_corrcoef(labels, predicted)), int(tn), int(fp), int(fn), int(tp), float(threshold), score_kind)
