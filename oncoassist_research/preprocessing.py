"""Leakage-resistant fold-local median-imputation and scaling infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import payload_sha256
from .data import FORBIDDEN_FEATURE_COLUMNS


@dataclass(frozen=True)
class FittedPreprocessor:
    """Training-bound preprocessing state; callers must not refit ``_pipeline``."""

    feature_names: tuple[str, ...]
    fit_sample_ids: tuple[str, ...]
    fit_sample_ids_sha256: str
    metadata: Mapping[str, Any]
    _pipeline: Pipeline = field(repr=False, compare=False)


@dataclass(frozen=True)
class PreprocessedPartition:
    """A transformed partition retaining its original patient and feature order."""

    matrix: np.ndarray
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]


def _build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _normalized_sample_ids(sample_ids: Sequence[str], expected_count: int) -> tuple[str, ...]:
    normalized = tuple(str(sample_id) for sample_id in sample_ids)
    if len(normalized) != expected_count:
        raise ValueError("DataFrame row count and SAMPLE_ID count must match.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("SAMPLE_ID values must be unique within a preprocessing partition.")
    if any(not sample_id.strip() for sample_id in normalized):
        raise ValueError("SAMPLE_ID values must not be blank.")
    return normalized


def _normalized_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(feature_name) for feature_name in feature_names)
    if not names:
        raise ValueError("Preprocessing requires at least one biological feature.")
    if len(set(names)) != len(names):
        raise ValueError("Preprocessing feature names must be unique.")
    forbidden = FORBIDDEN_FEATURE_COLUMNS.intersection(names)
    if forbidden:
        raise ValueError(f"Preprocessing feature names contain forbidden columns: {sorted(forbidden)}")
    return names


def _validate_and_numeric_frame(
    data_df: pd.DataFrame,
    sample_ids: Sequence[str],
    feature_names: Sequence[str],
    *,
    reject_empty: bool,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("Preprocessing input must be a pandas DataFrame.")
    if reject_empty and data_df.empty:
        raise ValueError("Preprocessing training data must not be empty.")
    ids = _normalized_sample_ids(sample_ids, len(data_df))
    names = _normalized_feature_names(feature_names)
    index_ids = tuple(str(index) for index in data_df.index.tolist())
    if index_ids != ids:
        raise ValueError("DataFrame index order must exactly match supplied SAMPLE_ID values.")
    if data_df.columns.duplicated().any():
        raise ValueError("Preprocessing input DataFrame contains duplicate feature columns.")
    forbidden_columns = FORBIDDEN_FEATURE_COLUMNS.intersection(data_df.columns)
    if forbidden_columns:
        raise ValueError(
            f"Preprocessing input contains forbidden feature columns: {sorted(forbidden_columns)}"
        )
    actual_names = tuple(str(column) for column in data_df.columns.tolist())
    if actual_names != names:
        raise ValueError("Preprocessing input feature schema/order does not match expected features.")

    numeric = pd.DataFrame(index=data_df.index)
    for feature_name in names:
        try:
            numeric[feature_name] = pd.to_numeric(data_df[feature_name], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Biological feature '{feature_name}' contains non-numeric values."
            ) from error
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Biological features contain infinite values.")
    return numeric, ids, names


def fit_preprocessor(
    training_df: pd.DataFrame,
    training_sample_ids: Sequence[str],
    feature_names: Sequence[str],
) -> FittedPreprocessor:
    """Fit median imputation and scaling only on the explicit training partition."""
    numeric_training, fit_ids, names = _validate_and_numeric_frame(
        training_df, training_sample_ids, feature_names, reject_empty=True
    )
    entirely_missing = numeric_training.columns[numeric_training.isna().all()].tolist()
    if entirely_missing:
        raise ValueError(
            f"Training features are entirely missing: {entirely_missing}"
        )
    pipeline = _build_preprocessor()
    transformed = pipeline.fit_transform(numeric_training)
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessing produced non-finite training values.")
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    zero_variance = np.flatnonzero(np.isclose(scaler.var_, 0.0))
    fit_hash = payload_sha256(list(fit_ids))
    metadata = {
        "fit_sample_count": len(fit_ids),
        "feature_count": len(names),
        "feature_names": list(names),
        "fit_sample_ids_sha256": fit_hash,
        "raw_training_missing_value_count": int(numeric_training.isna().sum().sum()),
        "imputer_medians": {
            name: float(value) for name, value in zip(names, imputer.statistics_)
        },
        "scaler_means": {
            name: float(value) for name, value in zip(names, scaler.mean_)
        },
        "scaler_variances": {
            name: float(value) for name, value in zip(names, scaler.var_)
        },
        "scaler_scales": {
            name: float(value) for name, value in zip(names, scaler.scale_)
        },
        "zero_variance_feature_names": [names[index] for index in zero_variance],
    }
    return FittedPreprocessor(
        feature_names=names,
        fit_sample_ids=fit_ids,
        fit_sample_ids_sha256=fit_hash,
        metadata=metadata,
        _pipeline=pipeline,
    )


def transform_with_preprocessor(
    fitted: FittedPreprocessor,
    data_df: pd.DataFrame,
    sample_ids: Sequence[str],
    expected_feature_names: Sequence[str],
) -> PreprocessedPartition:
    """Transform a validated partition without fitting any preprocessing state."""
    if not isinstance(fitted, FittedPreprocessor):
        raise TypeError("Transform requires a FittedPreprocessor instance.")
    numeric_data, ids, names = _validate_and_numeric_frame(
        data_df, sample_ids, expected_feature_names, reject_empty=False
    )
    if names != fitted.feature_names:
        raise ValueError("Transform feature contract does not match fitted preprocessing state.")
    transformed = np.asarray(fitted._pipeline.transform(numeric_data), dtype=np.float32)
    if transformed.ndim != 2 or transformed.shape != (len(ids), len(names)):
        raise ValueError("Preprocessing transform produced an invalid output shape.")
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessing transform produced non-finite values.")
    return PreprocessedPartition(
        matrix=transformed,
        sample_ids=ids,
        feature_names=names,
    )
