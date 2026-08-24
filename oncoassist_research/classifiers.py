"""Primary Logistic Regression on augmented fused latent features only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .ctgan import AugmentedTrainingSet


_FORBIDDEN = ("class", "sample_id", "target", "label", "synthetic", "real")


def _readonly(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True); copied.setflags(write=False); return copied


def _validate_features(features: np.ndarray, ids: Sequence[str], names: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    matrix = np.asarray(features)
    record_ids, feature_names = tuple(str(item) for item in ids), tuple(names)
    if matrix.ndim != 2 or matrix.dtype == object or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all(): raise ValueError("Classifier features must be finite numeric two-dimensional arrays.")
    if len(record_ids) != len(matrix) or len(record_ids) != len(set(record_ids)) or any(not item.strip() for item in record_ids): raise ValueError("Classifier record IDs are invalid.")
    if len(feature_names) != matrix.shape[1] or len(feature_names) != len(set(feature_names)) or any(not isinstance(name, str) or not name.strip() or any(token in name.lower() for token in _FORBIDDEN) for name in feature_names): raise ValueError("Classifier feature schema is invalid.")
    return matrix.astype(np.float32, copy=True), record_ids, feature_names


@dataclass(frozen=True)
class LogisticRegressionConfig:
    C: float
    def __post_init__(self) -> None:
        if isinstance(self.C, bool) or not isinstance(self.C, (int, float)) or not np.isfinite(float(self.C)) or float(self.C) <= 0: raise ValueError("Logistic Regression C must be finite and positive.")


@dataclass
class FittedLogisticClassifier:
    scaler: StandardScaler
    model: LogisticRegression
    feature_names: tuple[str, ...]
    training_record_ids: tuple[str, ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class LogisticScores:
    sample_ids: tuple[str, ...]
    decision_scores: np.ndarray
    probabilities: np.ndarray
    evidence: Mapping[str, Any]


def fit_logistic_classifier(training: AugmentedTrainingSet, config: LogisticRegressionConfig, model_seed: int) -> FittedLogisticClassifier:
    if not isinstance(training, AugmentedTrainingSet) or type(model_seed) is not int or model_seed < 0: raise ValueError("Classifier training requires augmented data and a non-negative integer seed.")
    features, ids, names = _validate_features(training.features, training.record_ids, training.feature_names)
    labels = np.asarray(training.labels)
    if labels.ndim != 1 or len(labels) != len(features) or not np.isin(labels, [0,1]).all() or set(labels.tolist()) != {0,1}: raise ValueError("Classifier training requires both binary classes.")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True); scaled = scaler.fit_transform(features)
    model = LogisticRegression(C=float(config.C), solver="liblinear", penalty="l2", max_iter=1000, class_weight=None, random_state=model_seed)
    model.fit(scaled, labels.astype(int))
    if set(model.classes_.tolist()) != {0,1}: raise ValueError("Classifier classes must be exactly binary.")
    return FittedLogisticClassifier(scaler, model, names, ids, {"C": float(config.C), "solver": "liblinear", "penalty": "l2", "max_iter": 1000, "class_weight": None, "model_seed": model_seed, "scaler_fit_row_count": len(features), "scaler_fit_includes_synthetic_rows": bool(np.asarray(training.is_synthetic).any())})


def score_logistic_classifier(fitted: FittedLogisticClassifier, features: np.ndarray, sample_ids: Sequence[str], expected_feature_names: Sequence[str]) -> LogisticScores:
    if not isinstance(fitted, FittedLogisticClassifier): raise TypeError("Scoring requires FittedLogisticClassifier.")
    matrix, ids, names = _validate_features(features, sample_ids, expected_feature_names)
    if names != fitted.feature_names: raise ValueError("Classifier scoring feature schema/order differs from fit.")
    scaled = fitted.scaler.transform(matrix); decisions = np.asarray(fitted.model.decision_function(scaled), dtype=float)
    index = int(np.flatnonzero(fitted.model.classes_ == 1)[0]); probabilities = np.asarray(fitted.model.predict_proba(scaled)[:, index], dtype=float)
    if decisions.shape != (len(ids),) or probabilities.shape != (len(ids),) or not np.isfinite(decisions).all() or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)): raise ValueError("Classifier scores are invalid.")
    return LogisticScores(ids, _readonly(decisions), _readonly(probabilities), {"canonical_raw_score": "decision_function", "diagnostic_probability": "predict_proba_class_1", "positive_class": 1, "model_classes": fitted.model.classes_.tolist()})
