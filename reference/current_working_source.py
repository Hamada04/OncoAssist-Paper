import argparse
from contextvars import ContextVar
import copy
import csv
import hashlib
import importlib.metadata
import importlib
import inspect
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    pairwise_distances,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input


np.random.seed(42)
tf.random.set_seed(42)


# =========================================
# SECTION 1: Imports and configuration
# =========================================
# This script refactors the doctor's notebook into a clean, non-leaky,
# competition-ready training and inference workflow.


# ==================================================
# ADDED BLOCK -- DATA + LABEL REVIEWED TRAINING PIPELINE
# Purpose: clean and reproducible baseline replacing fragmented notebook cells
# Where to place: standalone script for this model-only project
# ==================================================


"""
Purpose of this function

This function:

loads the 3 omics files
checks required columns
finds common samples across all files
aligns them in the same order
separates features and labels
converts labels into binary format

This is one of the most important functions in the script.
"""


REQUIRED_TRAINING_COLUMNS = {"SAMPLE_ID", "CLASS"}
FORBIDDEN_FEATURE_COLUMNS = {"SAMPLE_ID", "CLASS"}
PROJECT_LABEL_MAPPING = {
    "raw_to_binary": {"1": 0, "2": 1},
    "binary_to_tmb": {0: "Low-TMB", 1: "High-TMB"},
    "raw_semantics": {"1": "Low-TMB", "2": "High-TMB"},
    "mapping_basis": "Project-configured mapping based on the released BLCA dataset; no numeric clinical TMB cutoff is asserted.",
}
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3
OUTER_RANDOM_STATE = 42
OUTER_MANIFEST_SCHEMA_VERSION = "outer-repeated-stratified-kfold-v1"
INNER_N_SPLITS = 3
INNER_RANDOM_STATE_BASE = 4200
INNER_MANIFEST_SCHEMA_VERSION = "nested-inner-stratified-kfold-v1"
NESTED_SEARCH_SPACE_SCHEMA_VERSION = "nested-search-space-v1"
AE_EARLY_STOPPING_VALIDATION_FRACTION = 0.20
AE_EARLY_STOPPING_RANDOM_STATE_BASE = 5200
AE_OUTER_REFIT_EARLY_STOPPING_RANDOM_STATE_BASE = 6200


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_and_validate_training_table(
    modality: str, path: Path
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{modality} training file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"{modality} training file is empty or has no header: {path}")
    duplicate_headers = sorted({column for column in header if header.count(column) > 1})
    if duplicate_headers:
        raise ValueError(
            f"{modality} training file has duplicate column names: {duplicate_headers}"
        )

    df = pd.read_csv(path, dtype={"SAMPLE_ID": "string"})
    if df.columns.duplicated().any():
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"{modality} training file has duplicate column names: {duplicate_columns}"
        )

    missing_required = sorted(REQUIRED_TRAINING_COLUMNS.difference(df.columns))
    if missing_required:
        raise ValueError(
            f"{modality} training file is missing required columns: {missing_required}"
        )

    blank_sample_ids = df["SAMPLE_ID"].isna() | df["SAMPLE_ID"].str.strip().eq("")
    if blank_sample_ids.any():
        raise ValueError(
            f"{modality} training file has {int(blank_sample_ids.sum())} null or blank SAMPLE_ID values."
        )
    duplicate_id_count = int(df["SAMPLE_ID"].duplicated(keep=False).sum())
    if duplicate_id_count:
        duplicate_ids = df.loc[df["SAMPLE_ID"].duplicated(keep=False), "SAMPLE_ID"]
        raise ValueError(
            f"{modality} training file has {duplicate_id_count} duplicate SAMPLE_ID rows: "
            f"{duplicate_ids.drop_duplicates().tolist()[:10]}"
        )

    try:
        class_values = pd.to_numeric(df["CLASS"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{modality} CLASS values must be numeric 1 or 2.") from error
    if class_values.isna().any() or not np.isfinite(class_values.to_numpy(dtype=float)).all():
        raise ValueError(f"{modality} CLASS contains missing or infinite values.")
    if not class_values.isin([1, 2]).all():
        invalid_labels = sorted(class_values[~class_values.isin([1, 2])].unique().tolist())
        raise ValueError(
            f"{modality} CLASS must contain only released BLCA labels 1 and 2. "
            f"Found invalid values: {invalid_labels}"
        )
    df["CLASS"] = class_values.astype(int)

    feature_columns = [
        column for column in df.columns if column not in FORBIDDEN_FEATURE_COLUMNS
    ]
    if not feature_columns:
        raise ValueError(f"{modality} training file has no biological feature columns.")

    numeric_features = pd.DataFrame(index=df.index)
    for column in feature_columns:
        try:
            numeric_features[column] = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{modality} biological feature '{column}' contains non-numeric values."
            ) from error
    if np.isinf(numeric_features.to_numpy(dtype=float)).any():
        raise ValueError(f"{modality} biological features contain infinite values.")
    df.loc[:, feature_columns] = numeric_features

    audit = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "row_count": int(len(df)),
        "unique_sample_id_count": int(df["SAMPLE_ID"].nunique()),
        "feature_count": int(len(feature_columns)),
        "missing_value_count": int(numeric_features.isna().sum().sum()),
        "duplicate_id_count": duplicate_id_count,
    }
    return df, feature_columns, audit


def _assert_target_leakage_guards(
    feature_columns: Dict[str, List[str]], feature_matrices: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    for modality, columns in feature_columns.items():
        forbidden_columns = FORBIDDEN_FEATURE_COLUMNS.intersection(columns)
        if forbidden_columns:
            raise AssertionError(
                f"Target-leakage guard failed for {modality} feature list: "
                f"{sorted(forbidden_columns)}"
            )
    for modality, matrix in feature_matrices.items():
        forbidden_columns = FORBIDDEN_FEATURE_COLUMNS.intersection(matrix.columns)
        if forbidden_columns:
            raise AssertionError(
                f"Target-leakage guard failed for {modality} model input: "
                f"{sorted(forbidden_columns)}"
            )
    return {
        "passed": True,
        "excluded_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "checked_model_inputs": sorted(feature_matrices),
    }


def load_and_align_multiomics(
    rna_path: Path,
    dna_path: Path,
    cna_path: Path,
) -> Dict[str, Any]:
    """Load and validate the three released BLCA training tables."""
    table_specs = {
        "mGE": (rna_path, "rna"),
        "mDM": (dna_path, "dna"),
        "CNA": (cna_path, "cna"),
    }
    tables = {}
    feature_columns = {}
    audit_files = {}
    sample_id_sets = {}

    for modality, (path, feature_key) in table_specs.items():
        table, columns, audit = _read_and_validate_training_table(modality, path)
        tables[modality] = table
        feature_columns[feature_key] = columns
        audit_files[modality] = audit
        sample_id_sets[modality] = set(table["SAMPLE_ID"].tolist())

    reference_ids = sample_id_sets["mGE"]
    mismatched_modalities = {
        modality: {
            "missing_from_modality": sorted(reference_ids.difference(sample_ids))[:10],
            "unexpected_in_modality": sorted(sample_ids.difference(reference_ids))[:10],
        }
        for modality, sample_ids in sample_id_sets.items()
        if sample_ids != reference_ids
    }
    if mismatched_modalities:
        raise ValueError(
            "Training CSV SAMPLE_ID sets must match exactly; no patients may be discarded. "
            f"Differences: {mismatched_modalities}"
        )

    aligned_ids = sorted(reference_ids)
    aligned_tables = {
        modality: table.set_index("SAMPLE_ID").loc[aligned_ids]
        for modality, table in tables.items()
    }

    rna_labels = aligned_tables["mGE"]["CLASS"]
    label_mismatches = (
        (rna_labels != aligned_tables["mDM"]["CLASS"])
        | (rna_labels != aligned_tables["CNA"]["CLASS"])
    )
    if label_mismatches.any():
        raise ValueError(
            "CLASS must agree across mGE, mDM, and CNA for every aligned patient. "
            f"Mismatched SAMPLE_ID values: {label_mismatches[label_mismatches].index.tolist()[:10]}"
        )

    X_rna = aligned_tables["mGE"].loc[:, feature_columns["rna"]]
    X_dna = aligned_tables["mDM"].loc[:, feature_columns["dna"]]
    X_cna = aligned_tables["CNA"].loc[:, feature_columns["cna"]]
    leakage_guard = _assert_target_leakage_guards(
        feature_columns,
        {"rna": X_rna, "dna": X_dna, "cna": X_cna},
    )

    y_raw = rna_labels.to_numpy(dtype=int)
    y_binary = (y_raw == 2).astype(int)
    raw_class_counts = {
        str(label): int(count)
        for label, count in pd.Series(y_raw).value_counts().sort_index().items()
    }
    binary_class_counts = {
        str(label): int(count)
        for label, count in pd.Series(y_binary).value_counts().sort_index().items()
    }

    return {
        "X_rna": X_rna,
        "X_dna": X_dna,
        "X_cna": X_cna,
        "y_binary": y_binary,
        "sample_ids": aligned_ids,
        "feature_columns": feature_columns,
        "label_mapping": PROJECT_LABEL_MAPPING,
        "audit_summary": {
            "files": audit_files,
            "raw_class_counts": raw_class_counts,
            "binary_class_counts": binary_class_counts,
            "label_mapping": PROJECT_LABEL_MAPPING,
            "sample_alignment": {
                "passed": True,
                "aligned_sample_id_count": len(aligned_ids),
                "ordering": "lexicographic SAMPLE_ID order",
            },
            "target_leakage_guard": leakage_guard,
        },
    }


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _class_counts(labels: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    observed = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return {"0": observed.get("0", 0), "1": observed.get("1", 0)}


def _label_by_sample_id(
    sample_ids: List[str], y_binary: np.ndarray
) -> Dict[str, int]:
    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    labels = np.asarray(y_binary, dtype=int)
    if normalized_ids != sorted(normalized_ids):
        raise ValueError("Outer-fold manifest requires lexicographically ordered SAMPLE_ID values.")
    if len(normalized_ids) != len(labels):
        raise ValueError("SAMPLE_ID and binary-label counts differ for outer-fold manifest.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Outer-fold manifest requires unique SAMPLE_ID values.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Outer-fold manifest requires binary labels 0 and 1.")
    return dict(zip(normalized_ids, labels.tolist()))


def _partition_summary(
    sample_ids: List[str], label_by_sample_id: Dict[str, int]
) -> Dict[str, Any]:
    labels = np.asarray([label_by_sample_id[sample_id] for sample_id in sample_ids])
    return {
        "sample_count": len(sample_ids),
        "class_counts": _class_counts(labels),
    }


def _build_outer_fold_records(
    sample_ids: List[str], y_binary: np.ndarray
) -> List[Dict[str, Any]]:
    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    labels = np.asarray(y_binary, dtype=int)
    label_by_sample_id = _label_by_sample_id(normalized_ids, labels)
    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        n_repeats=OUTER_N_REPEATS,
        random_state=OUTER_RANDOM_STATE,
    )
    folds = []
    for split_index, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        train_sample_ids = sorted(normalized_ids[index] for index in train_idx)
        test_sample_ids = sorted(normalized_ids[index] for index in test_idx)
        folds.append(
            {
                "repeat_id": split_index // OUTER_N_SPLITS,
                "fold_id": split_index % OUTER_N_SPLITS,
                "train_sample_ids": train_sample_ids,
                "test_sample_ids": test_sample_ids,
                "train_partition": _partition_summary(
                    train_sample_ids, label_by_sample_id
                ),
                "test_partition": _partition_summary(
                    test_sample_ids, label_by_sample_id
                ),
            }
        )
    return folds


def build_outer_data_fingerprint(data: Dict[str, Any]) -> Dict[str, Any]:
    sample_ids = [str(sample_id) for sample_id in data["sample_ids"]]
    files = data["audit_summary"]["files"]
    return {
        "csv_sha256": {
            modality: files[modality]["sha256"] for modality in ["mGE", "mDM", "CNA"]
        },
        "sample_count": len(sample_ids),
        "label_mapping": json.loads(
            _canonical_json_bytes(data["label_mapping"]).decode("utf-8")
        ),
        "ordered_sample_ids_canonical_json_sha256": hashlib.sha256(
            _canonical_json_bytes(sample_ids)
        ).hexdigest(),
    }


def build_outer_fold_manifest(
    sample_ids: List[str], y_binary: np.ndarray, data_fingerprint: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": OUTER_MANIFEST_SCHEMA_VERSION,
        "protocol": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": OUTER_N_SPLITS,
            "n_repeats": OUTER_N_REPEATS,
            "random_state": OUTER_RANDOM_STATE,
        },
        "data_fingerprint": data_fingerprint,
        "folds": _build_outer_fold_records(sample_ids, y_binary),
    }


def validate_outer_fold_manifest(
    manifest: Dict[str, Any],
    sample_ids: List[str],
    y_binary: np.ndarray,
    data_fingerprint: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Outer-fold manifest must be a JSON object.")
    expected_protocol = {
        "splitter": "RepeatedStratifiedKFold",
        "n_splits": OUTER_N_SPLITS,
        "n_repeats": OUTER_N_REPEATS,
        "random_state": OUTER_RANDOM_STATE,
    }
    if manifest.get("schema_version") != OUTER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Outer-fold manifest schema version does not match this protocol.")
    if manifest.get("protocol") != expected_protocol:
        raise ValueError("Outer-fold manifest protocol configuration does not match.")
    if manifest.get("data_fingerprint") != data_fingerprint:
        raise ValueError("Outer-fold manifest data fingerprint does not match current data.")

    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    label_by_sample_id = _label_by_sample_id(normalized_ids, y_binary)
    current_id_set = set(normalized_ids)
    folds = manifest.get("folds")
    expected_fold_count = OUTER_N_SPLITS * OUTER_N_REPEATS
    if not isinstance(folds, list) or len(folds) != expected_fold_count:
        raise ValueError(
            f"Outer-fold manifest must contain exactly {expected_fold_count} fold records."
        )

    seen_pairs = set()
    test_ids_by_repeat = {repeat_id: [] for repeat_id in range(OUTER_N_REPEATS)}
    for record in folds:
        if not isinstance(record, dict):
            raise ValueError("Outer-fold manifest contains a non-object fold record.")
        repeat_id = record.get("repeat_id")
        fold_id = record.get("fold_id")
        if not isinstance(repeat_id, int) or not isinstance(fold_id, int):
            raise ValueError("Outer-fold repeat_id and fold_id must be integers.")
        if repeat_id not in range(OUTER_N_REPEATS) or fold_id not in range(OUTER_N_SPLITS):
            raise ValueError("Outer-fold manifest contains an out-of-range repeat/fold identifier.")
        pair = (repeat_id, fold_id)
        if pair in seen_pairs:
            raise ValueError(f"Outer-fold manifest duplicates repeat/fold pair {pair}.")
        seen_pairs.add(pair)

        train_ids = record.get("train_sample_ids")
        test_ids = record.get("test_sample_ids")
        if not isinstance(train_ids, list) or not isinstance(test_ids, list):
            raise ValueError(f"Outer-fold manifest {pair} must contain train and test SAMPLE_ID lists.")
        if train_ids != sorted(train_ids) or test_ids != sorted(test_ids):
            raise ValueError(f"Outer-fold manifest {pair} SAMPLE_ID lists must be sorted.")
        if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
            raise ValueError(f"Outer-fold manifest {pair} contains duplicate IDs within a partition.")
        train_set = set(train_ids)
        test_set = set(test_ids)
        if not train_set.issubset(current_id_set) or not test_set.issubset(current_id_set):
            raise ValueError(f"Outer-fold manifest {pair} contains IDs absent from current data.")
        if train_set.intersection(test_set):
            raise ValueError(f"Outer-fold manifest {pair} has train/test SAMPLE_ID overlap.")
        if train_set.union(test_set) != current_id_set:
            raise ValueError(f"Outer-fold manifest {pair} does not partition all current SAMPLE_ID values.")

        expected_train_summary = _partition_summary(train_ids, label_by_sample_id)
        expected_test_summary = _partition_summary(test_ids, label_by_sample_id)
        if record.get("train_partition") != expected_train_summary:
            raise ValueError(f"Outer-fold manifest {pair} train counts do not match current labels.")
        if record.get("test_partition") != expected_test_summary:
            raise ValueError(f"Outer-fold manifest {pair} test counts do not match current labels.")
        if 0 in expected_train_summary["class_counts"].values() or 0 in expected_test_summary["class_counts"].values():
            raise ValueError(f"Outer-fold manifest {pair} does not contain both classes in each partition.")
        test_ids_by_repeat[repeat_id].extend(test_ids)

    expected_pairs = {
        (repeat_id, fold_id)
        for repeat_id in range(OUTER_N_REPEATS)
        for fold_id in range(OUTER_N_SPLITS)
    }
    if seen_pairs != expected_pairs:
        raise ValueError("Outer-fold manifest does not contain five folds for every repeat.")

    per_repeat_test_coverage = {}
    for repeat_id, repeat_test_ids in test_ids_by_repeat.items():
        if len(repeat_test_ids) != len(current_id_set):
            raise ValueError(
                f"Outer-fold manifest repeat {repeat_id} does not contain exactly one test entry per sample."
            )
        if len(set(repeat_test_ids)) != len(current_id_set) or set(repeat_test_ids) != current_id_set:
            raise ValueError(
                f"Outer-fold manifest repeat {repeat_id} does not cover every sample exactly once in test folds."
            )
        per_repeat_test_coverage[str(repeat_id)] = {
            "passed": True,
            "unique_test_sample_count": len(set(repeat_test_ids)),
        }

    expected_folds = build_outer_fold_manifest(
        normalized_ids, y_binary, data_fingerprint
    )["folds"]
    if folds != expected_folds:
        raise ValueError(
            "Outer-fold manifest fold contents do not match the deterministic protocol rebuild."
        )

    return {
        "passed": True,
        "fold_count": len(folds),
        "sample_count": len(current_id_set),
        "per_repeat_test_coverage": per_repeat_test_coverage,
        "deterministic_rebuild_matches": True,
    }


def write_outer_fold_manifest(manifest: Dict[str, Any], path: Path) -> Dict[str, Any]:
    if not path.parent.exists():
        raise FileNotFoundError(f"Manifest parent directory does not exist: {path.parent}")
    payload = _canonical_json_bytes(manifest)
    path.write_bytes(payload)
    return {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_size_bytes": len(payload),
    }


def load_outer_fold_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Outer-fold manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Outer-fold manifest is not valid JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Outer-fold manifest must be a JSON object.")
    return manifest


def _inner_manifest_protocol() -> Dict[str, Any]:
    return {
        "splitter": "StratifiedKFold",
        "n_splits": INNER_N_SPLITS,
        "shuffle": True,
        "inner_random_state_base": INNER_RANDOM_STATE_BASE,
        "random_state_derivation": "INNER_RANDOM_STATE_BASE + repeat_id * 100 + fold_id",
    }


def derive_inner_fold_seed(repeat_id: int, fold_id: int) -> int:
    if repeat_id not in range(OUTER_N_REPEATS) or fold_id not in range(OUTER_N_SPLITS):
        raise ValueError("Inner-fold seed derivation requires a valid outer repeat/fold pair.")
    return INNER_RANDOM_STATE_BASE + repeat_id * 100 + fold_id


def _build_inner_fold_records(
    repeat_id: int, fold_id: int, outer_train_sample_ids: List[str], label_by_sample_id: Dict[str, int]
) -> List[Dict[str, Any]]:
    train_ids = list(outer_train_sample_ids)
    if train_ids != sorted(train_ids) or len(train_ids) != len(set(train_ids)):
        raise ValueError("Inner-fold construction requires sorted unique outer-training SAMPLE_ID values.")
    labels = np.asarray([label_by_sample_id[sample_id] for sample_id in train_ids], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Inner-fold construction requires both classes in each outer-training partition.")
    seed = derive_inner_fold_seed(repeat_id, fold_id)
    splitter = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
    records = []
    for inner_fold_id, (inner_train_index, inner_validation_index) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        inner_train_ids = sorted(train_ids[index] for index in inner_train_index)
        inner_validation_ids = sorted(train_ids[index] for index in inner_validation_index)
        records.append({
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "inner_fold_id": inner_fold_id,
            "inner_seed": seed,
            "inner_train_sample_ids": inner_train_ids,
            "inner_validation_sample_ids": inner_validation_ids,
            "inner_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(inner_train_ids),
            "inner_validation_sample_ids_canonical_json_sha256": _sample_id_list_sha256(inner_validation_ids),
            "inner_train_partition": _partition_summary(inner_train_ids, label_by_sample_id),
            "inner_validation_partition": _partition_summary(inner_validation_ids, label_by_sample_id),
        })
    return records


def build_inner_fold_manifest(
    outer_manifest: Dict[str, Any], outer_manifest_sha256: str, sample_ids: List[str], y_binary: np.ndarray, data_fingerprint: Dict[str, Any]
) -> Dict[str, Any]:
    label_by_sample_id = _label_by_sample_id([str(sample_id) for sample_id in sample_ids], y_binary)
    outer_records = []
    for outer_record in outer_manifest["folds"]:
        repeat_id, fold_id = outer_record["repeat_id"], outer_record["fold_id"]
        outer_train_ids = list(outer_record["train_sample_ids"])
        outer_test_ids = list(outer_record["test_sample_ids"])
        outer_records.append({
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "outer_train_sample_count": len(outer_train_ids),
            "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_train_ids),
            "outer_test_sample_count": len(outer_test_ids),
            "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_test_ids),
            "inner_folds": _build_inner_fold_records(repeat_id, fold_id, outer_train_ids, label_by_sample_id),
        })
    return {
        "schema_version": INNER_MANIFEST_SCHEMA_VERSION,
        "protocol": _inner_manifest_protocol(),
        "data_fingerprint": data_fingerprint,
        "outer_manifest_binding": {
            "outer_manifest_schema_version": OUTER_MANIFEST_SCHEMA_VERSION,
            "outer_manifest_sha256": outer_manifest_sha256,
        },
        "outer_folds": outer_records,
    }


def validate_inner_fold_manifest(
    manifest: Dict[str, Any], outer_manifest: Dict[str, Any], outer_manifest_sha256: str, sample_ids: List[str], y_binary: np.ndarray, data_fingerprint: Dict[str, Any]
) -> Dict[str, Any]:
    validate_outer_fold_manifest(outer_manifest, sample_ids, y_binary, data_fingerprint)
    if not isinstance(manifest, dict):
        raise ValueError("Inner-fold manifest must be a JSON object.")
    if manifest.get("schema_version") != INNER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Inner-fold manifest schema version does not match this protocol.")
    if manifest.get("protocol") != _inner_manifest_protocol():
        raise ValueError("Inner-fold manifest splitter configuration does not match.")
    if manifest.get("data_fingerprint") != data_fingerprint:
        raise ValueError("Inner-fold manifest data fingerprint does not match current data.")
    expected_binding = {"outer_manifest_schema_version": OUTER_MANIFEST_SCHEMA_VERSION, "outer_manifest_sha256": outer_manifest_sha256}
    if manifest.get("outer_manifest_binding") != expected_binding:
        raise ValueError("Inner-fold manifest outer-manifest binding does not match.")
    label_by_sample_id = _label_by_sample_id([str(sample_id) for sample_id in sample_ids], y_binary)
    outer_by_pair = {(record["repeat_id"], record["fold_id"]): record for record in outer_manifest["folds"]}
    records = manifest.get("outer_folds")
    if not isinstance(records, list) or len(records) != OUTER_N_SPLITS * OUTER_N_REPEATS:
        raise ValueError("Inner-fold manifest must contain exactly 15 outer records.")
    seen_pairs, seen_inner_records, coverage = set(), set(), {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Inner-fold manifest contains a non-object outer record.")
        pair = (record.get("repeat_id"), record.get("fold_id"))
        if pair not in outer_by_pair or pair in seen_pairs:
            raise ValueError("Inner-fold manifest outer repeat/fold records are invalid or duplicated.")
        seen_pairs.add(pair)
        outer_record = outer_by_pair[pair]
        outer_train_ids, outer_test_ids = outer_record["train_sample_ids"], outer_record["test_sample_ids"]
        expected_outer_metadata = {
            "outer_train_sample_count": len(outer_train_ids),
            "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_train_ids),
            "outer_test_sample_count": len(outer_test_ids),
            "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(outer_test_ids),
        }
        if any(record.get(key) != value for key, value in expected_outer_metadata.items()):
            raise ValueError(f"Inner-fold manifest outer metadata does not match outer fold {pair}.")
        inner_records = record.get("inner_folds")
        if not isinstance(inner_records, list) or len(inner_records) != INNER_N_SPLITS:
            raise ValueError(f"Inner-fold manifest outer fold {pair} must contain exactly three inner records.")
        outer_train_set, outer_test_set, validation_ids = set(outer_train_ids), set(outer_test_ids), []
        seen_inner_ids = set()
        for inner in inner_records:
            inner_fold_id = inner.get("inner_fold_id")
            if inner.get("repeat_id") != pair[0] or inner.get("fold_id") != pair[1] or inner_fold_id not in range(INNER_N_SPLITS) or inner_fold_id in seen_inner_ids:
                raise ValueError(f"Inner-fold manifest {pair} has invalid or duplicated inner_fold_id values.")
            seen_inner_ids.add(inner_fold_id)
            triple = (pair[0], pair[1], inner_fold_id)
            if triple in seen_inner_records:
                raise ValueError("Inner-fold manifest duplicates an outer/inner fold record.")
            seen_inner_records.add(triple)
            if inner.get("inner_seed") != derive_inner_fold_seed(*pair):
                raise ValueError(f"Inner-fold manifest {pair} has an incorrect derived inner seed.")
            train_ids, validation_ids_for_fold = inner.get("inner_train_sample_ids"), inner.get("inner_validation_sample_ids")
            if not isinstance(train_ids, list) or not isinstance(validation_ids_for_fold, list) or train_ids != sorted(train_ids) or validation_ids_for_fold != sorted(validation_ids_for_fold):
                raise ValueError(f"Inner-fold manifest {pair} SAMPLE_ID lists must be sorted lists.")
            train_set, validation_set = set(train_ids), set(validation_ids_for_fold)
            if len(train_set) != len(train_ids) or len(validation_set) != len(validation_ids_for_fold):
                raise ValueError(f"Inner-fold manifest {pair} contains duplicated inner SAMPLE_ID values.")
            if not train_set.issubset(outer_train_set) or not validation_set.issubset(outer_train_set):
                raise ValueError(f"Inner-fold manifest {pair} contains IDs outside the outer-training partition.")
            if train_set.intersection(outer_test_set) or validation_set.intersection(outer_test_set):
                raise ValueError(f"Inner-fold manifest {pair} contains outer-test SAMPLE_ID values.")
            if train_set.intersection(validation_set) or train_set.union(validation_set) != outer_train_set:
                raise ValueError(f"Inner-fold manifest {pair} inner train/validation partitions are invalid.")
            if inner.get("inner_train_sample_ids_canonical_json_sha256") != _sample_id_list_sha256(train_ids) or inner.get("inner_validation_sample_ids_canonical_json_sha256") != _sample_id_list_sha256(validation_ids_for_fold):
                raise ValueError(f"Inner-fold manifest {pair} SAMPLE_ID hashes do not match.")
            for partition_name, partition_ids in [("inner_train_partition", train_ids), ("inner_validation_partition", validation_ids_for_fold)]:
                expected_partition = _partition_summary(partition_ids, label_by_sample_id)
                if inner.get(partition_name) != expected_partition or 0 in expected_partition["class_counts"].values():
                    raise ValueError(f"Inner-fold manifest {pair} {partition_name} counts do not match current labels.")
            validation_ids.extend(validation_ids_for_fold)
        if seen_inner_ids != set(range(INNER_N_SPLITS)) or len(validation_ids) != len(outer_train_ids) or len(set(validation_ids)) != len(outer_train_ids) or set(validation_ids) != outer_train_set:
            raise ValueError(f"Inner-fold manifest {pair} does not cover every outer-training patient exactly once in validation.")
        coverage[f"{pair[0]}:{pair[1]}"] = {"passed": True, "validation_sample_count": len(validation_ids)}
    expected_pairs = set(outer_by_pair)
    if seen_pairs != expected_pairs or len(seen_inner_records) != OUTER_N_SPLITS * OUTER_N_REPEATS * INNER_N_SPLITS:
        raise ValueError("Inner-fold manifest does not contain the complete deterministic outer/inner record set.")
    expected = build_inner_fold_manifest(outer_manifest, outer_manifest_sha256, sample_ids, y_binary, data_fingerprint)
    if manifest != expected:
        raise ValueError("Inner-fold manifest contents do not match the deterministic protocol rebuild.")
    return {"passed": True, "outer_record_count": len(records), "inner_record_count": len(seen_inner_records), "per_outer_validation_coverage": coverage, "deterministic_rebuild_matches": True}


def write_inner_fold_manifest(manifest: Dict[str, Any], path: Path) -> Dict[str, Any]:
    if not path.parent.exists():
        raise FileNotFoundError(f"Manifest parent directory does not exist: {path.parent}")
    payload = _canonical_json_bytes(manifest)
    path.write_bytes(payload)
    return {"manifest_path": str(path), "manifest_sha256": hashlib.sha256(payload).hexdigest(), "manifest_size_bytes": len(payload)}


def load_inner_fold_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Inner-fold manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Inner-fold manifest is not valid JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Inner-fold manifest must be a JSON object.")
    return manifest


FOLD_MODALITIES = {
    "mGE": {"matrix_key": "X_rna", "feature_key": "rna"},
    "mDM": {"matrix_key": "X_dna", "feature_key": "dna"},
    "mCNA": {"matrix_key": "X_cna", "feature_key": "cna"},
}


def load_validated_outer_fold_manifest(
    data: Dict[str, Any], manifest_path: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data_fingerprint = build_outer_data_fingerprint(data)
    manifest = load_outer_fold_manifest(manifest_path)
    validation = validate_outer_fold_manifest(
        manifest, data["sample_ids"], data["y_binary"], data_fingerprint
    )
    return manifest, validation


def _feature_name_sha256(feature_names: List[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(feature_names)).hexdigest()


def _sample_id_list_sha256(sample_ids: List[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(sample_ids)).hexdigest()


def materialize_outer_fold(
    data: Dict[str, Any], manifest: Dict[str, Any], repeat_id: int, fold_id: int
) -> Dict[str, Any]:
    matching_records = [
        record
        for record in manifest["folds"]
        if record["repeat_id"] == repeat_id and record["fold_id"] == fold_id
    ]
    if len(matching_records) != 1:
        raise ValueError(
            f"Requested outer fold repeat_id={repeat_id}, fold_id={fold_id} must exist exactly once."
        )

    record = matching_records[0]
    train_sample_ids = list(record["train_sample_ids"])
    test_sample_ids = list(record["test_sample_ids"])
    train_set = set(train_sample_ids)
    test_set = set(test_sample_ids)
    all_sample_ids = set(data["sample_ids"])
    if train_set.intersection(test_set):
        raise AssertionError("Materialized fold has train/test SAMPLE_ID overlap.")
    if train_set.union(test_set) != all_sample_ids:
        raise AssertionError("Materialized fold does not partition all current SAMPLE_ID values.")

    labels = pd.Series(
        data["y_binary"], index=pd.Index(data["sample_ids"], name="SAMPLE_ID")
    )
    y_train = labels.loc[train_sample_ids].to_numpy(dtype=int)
    y_test = labels.loc[test_sample_ids].to_numpy(dtype=int)
    if _class_counts(y_train) != record["train_partition"]["class_counts"]:
        raise AssertionError("Materialized train labels do not match manifest class counts.")
    if _class_counts(y_test) != record["test_partition"]["class_counts"]:
        raise AssertionError("Materialized test labels do not match manifest class counts.")

    modalities = {}
    leakage_inputs = {}
    for modality, keys in FOLD_MODALITIES.items():
        matrix = data[keys["matrix_key"]]
        expected_features = list(data["feature_columns"][keys["feature_key"]])
        if matrix.index.tolist() != data["sample_ids"]:
            raise AssertionError(f"{modality} matrix index does not match Phase 1 SAMPLE_ID order.")
        if matrix.columns.tolist() != expected_features:
            raise AssertionError(f"{modality} matrix feature names do not match the Phase 1 schema.")
        train_df = matrix.loc[train_sample_ids, expected_features].copy()
        test_df = matrix.loc[test_sample_ids, expected_features].copy()
        if train_df.index.tolist() != train_sample_ids or test_df.index.tolist() != test_sample_ids:
            raise AssertionError(f"{modality} materialization did not preserve manifest SAMPLE_ID order.")
        if train_df.columns.tolist() != test_df.columns.tolist():
            raise AssertionError(f"{modality} train/test feature order differs.")
        feature_hash = _feature_name_sha256(expected_features)
        modalities[modality] = {
            "train_df": train_df,
            "test_df": test_df,
            "metadata": {
                "modality": modality,
                "train_sample_ids": train_sample_ids,
                "test_sample_ids": test_sample_ids,
                "feature_names": expected_features,
                "feature_name_sha256": feature_hash,
            },
        }
        leakage_inputs[f"{modality}_train"] = train_df
        leakage_inputs[f"{modality}_test"] = test_df

    leakage_guard = _assert_target_leakage_guards(
        {modality: value["metadata"]["feature_names"] for modality, value in modalities.items()},
        leakage_inputs,
    )
    return {
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "train_sample_ids": train_sample_ids,
        "test_sample_ids": test_sample_ids,
        "y_train": y_train,
        "y_test": y_test,
        "train_class_counts": _class_counts(y_train),
        "test_class_counts": _class_counts(y_test),
        "modalities": modalities,
        "checks": {
            "cross_omics_alignment": {
                "passed": True,
                "train_sample_id_order_matches": True,
                "test_sample_id_order_matches": True,
            },
            "train_test_overlap": {"passed": True},
            "target_leakage_guard": leakage_guard,
        },
    }


# =========================================
# SECTION 2: Safe preprocessing utilities
# =========================================
# What: per-modality impute + scale pipelines
# Why: avoid leakage and ensure stable latent learning
# Input: training DataFrames per modality
# Output: fitted sklearn pipelines

"""
This function creates a preprocessing pipeline that will be used for each omics dataset:

RNA
DNA methylation
CNA

So instead of preprocessing each dataset manually, the script uses one reusable function.
"""


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def fit_fold_preprocessors(materialized_fold: Dict[str, Any]) -> Dict[str, Any]:
    transformed_modalities = {}
    for modality, materialized in materialized_fold["modalities"].items():
        train_df = materialized["train_df"]
        test_df = materialized["test_df"]
        feature_names = materialized["metadata"]["feature_names"]
        entirely_missing = train_df.columns[train_df.isna().all()].tolist()
        if entirely_missing:
            raise ValueError(
                f"{modality} training features are entirely missing: {entirely_missing}"
            )

        preprocessor = build_preprocessor()
        transformed_train = preprocessor.fit_transform(train_df)
        transformed_test = preprocessor.transform(test_df)
        if not np.isfinite(transformed_train).all() or not np.isfinite(transformed_test).all():
            raise ValueError(f"{modality} preprocessing produced non-finite transformed values.")

        imputer = preprocessor.named_steps["imputer"]
        scaler = preprocessor.named_steps["scaler"]
        zero_variance_indices = np.flatnonzero(np.isclose(scaler.var_, 0.0))
        transformed_modalities[modality] = {
            "train": transformed_train,
            "test": transformed_test,
            "preprocessor": preprocessor,
            "metadata": {
                **materialized["metadata"],
                "raw_train_missing_value_count": int(train_df.isna().sum().sum()),
                "raw_test_missing_value_count": int(test_df.isna().sum().sum()),
                "imputer_medians": {
                    feature: float(value)
                    for feature, value in zip(feature_names, imputer.statistics_)
                },
                "scaler_means": {
                    feature: float(value)
                    for feature, value in zip(feature_names, scaler.mean_)
                },
                "scaler_scales": {
                    feature: float(value)
                    for feature, value in zip(feature_names, scaler.scale_)
                },
                "zero_variance_feature_names": [
                    feature_names[index] for index in zero_variance_indices
                ],
            },
        }
    return transformed_modalities


def _transformed_train_summary(transformed_train: np.ndarray) -> Dict[str, float]:
    feature_means = np.mean(transformed_train, axis=0)
    feature_stds = np.std(transformed_train, axis=0, ddof=0)
    return {
        "feature_mean_min": float(np.min(feature_means)),
        "feature_mean_max": float(np.max(feature_means)),
        "max_abs_feature_mean": float(np.max(np.abs(feature_means))),
        "feature_std_min_ddof0": float(np.min(feature_stds)),
        "feature_std_max_ddof0": float(np.max(feature_stds)),
    }


def build_fold_preprocessing_audit(
    manifest_validation: Dict[str, Any],
    materialized_fold: Dict[str, Any],
    transformed_modalities: Dict[str, Any],
) -> Dict[str, Any]:
    modality_audit = {}
    for modality, transformed in transformed_modalities.items():
        metadata = transformed["metadata"]
        train_matrix = transformed["train"]
        test_matrix = transformed["test"]
        modality_audit[modality] = {
            "train_shape": list(train_matrix.shape),
            "test_shape": list(test_matrix.shape),
            "feature_count": len(metadata["feature_names"]),
            "feature_name_sha256": metadata["feature_name_sha256"],
            "raw_missing_value_counts": {
                "train": metadata["raw_train_missing_value_count"],
                "test": metadata["raw_test_missing_value_count"],
            },
            "imputer_medians": metadata["imputer_medians"],
            "scaler_means": metadata["scaler_means"],
            "scaler_scales": metadata["scaler_scales"],
            "zero_variance_feature_count": len(metadata["zero_variance_feature_names"]),
            "zero_variance_feature_names": metadata["zero_variance_feature_names"],
            "finite_transformed_outputs": {
                "train": bool(np.isfinite(train_matrix).all()),
                "test": bool(np.isfinite(test_matrix).all()),
            },
            "transformed_training_summary": _transformed_train_summary(train_matrix),
            "fit_sample_count": len(metadata["train_sample_ids"]),
            "transform_sample_count": len(metadata["test_sample_ids"]),
            "fit_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                metadata["train_sample_ids"]
            ),
            "transform_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                metadata["test_sample_ids"]
            ),
            "fit_scope": "train_only",
            "test_operation": "transform_only",
        }
    return {
        "action": "audited_preprocessing",
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": materialized_fold["repeat_id"],
            "fold_id": materialized_fold["fold_id"],
            "train_class_counts": materialized_fold["train_class_counts"],
            "test_class_counts": materialized_fold["test_class_counts"],
            "cross_omics_alignment": materialized_fold["checks"]["cross_omics_alignment"],
            "train_test_overlap": materialized_fold["checks"]["train_test_overlap"],
            "target_leakage_guard": materialized_fold["checks"]["target_leakage_guard"],
        },
        "modalities": modality_audit,
    }


# =========================================
# SECTION 2B: Fold-scoped autoencoder audit utilities
# =========================================


def _validate_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _validate_fold_autoencoder_config(
    modality: str, input_dim: int, config: Dict[str, Any]
) -> Dict[str, Any]:
    if modality not in FOLD_MODALITIES:
        raise ValueError(f"Unknown fold autoencoder modality: {modality}")
    if not isinstance(config, dict):
        raise ValueError(f"{modality} autoencoder configuration must be an object.")
    input_width = _validate_positive_integer(f"{modality} input_dim", input_dim)
    hidden_dims = config.get("hidden_dims")
    if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
        raise ValueError(f"{modality} hidden_dims must be a non-empty list of positive integers.")
    normalized_hidden_dims = [
        _validate_positive_integer(f"{modality} hidden_dims[{index}]", value)
        for index, value in enumerate(hidden_dims)
    ]
    latent_dim = _validate_positive_integer(f"{modality} latent_dim", config.get("latent_dim"))
    if latent_dim >= input_width:
        raise ValueError(
            f"{modality} latent_dim must be smaller than input_dim ({input_width})."
        )
    if latent_dim >= min(normalized_hidden_dims):
        raise ValueError(
            f"{modality} latent_dim must be smaller than every hidden dimension."
        )
    return {"hidden_dims": normalized_hidden_dims, "latent_dim": latent_dim}


def _validate_autoencoder_training_config(config: Dict[str, Any]) -> Dict[str, int]:
    if not isinstance(config, dict):
        raise ValueError("Autoencoder training configuration must be an object.")
    return {
        "epochs": _validate_positive_integer("ae_epochs", config.get("epochs")),
        "batch_size": _validate_positive_integer("ae_batch_size", config.get("batch_size")),
        "patience": _validate_nonnegative_integer("ae_patience", config.get("patience")),
        "seed": _validate_nonnegative_integer("ae_seed", config.get("seed")),
    }


def _parse_hidden_dims(value: str) -> List[int]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("--ae-hidden-dims must be a comma-separated list of positive integers.")
    try:
        hidden_dims = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError(
            "--ae-hidden-dims must be a comma-separated list of positive integers."
        ) from error
    if any(dimension <= 0 for dimension in hidden_dims):
        raise ValueError("--ae-hidden-dims must contain only positive integers.")
    return hidden_dims


def build_inner_autoencoder_split(
    materialized_fold: Dict[str, Any], validation_fraction: float, seed: int
) -> Dict[str, Any]:
    if not isinstance(validation_fraction, (float, int)) or not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("inner_validation_fraction must be strictly between 0 and 1.")
    split_seed = _validate_nonnegative_integer("inner_split_seed", seed)
    outer_train_ids = list(materialized_fold["train_sample_ids"])
    outer_train_labels = np.asarray(materialized_fold["y_train"], dtype=int)
    if len(outer_train_ids) != len(outer_train_labels):
        raise ValueError("Outer-training SAMPLE_ID and label counts differ for the inner split.")

    try:
        inner_train_ids, inner_validation_ids, inner_train_y, inner_validation_y = train_test_split(
            outer_train_ids,
            outer_train_labels,
            test_size=float(validation_fraction),
            random_state=split_seed,
            stratify=outer_train_labels,
        )
    except ValueError as error:
        raise ValueError("Unable to create a stratified inner autoencoder split.") from error

    order_indices = np.random.default_rng(split_seed).permutation(len(inner_train_ids))
    training_order_ids = [inner_train_ids[index] for index in order_indices]
    label_by_id = dict(zip(outer_train_ids, outer_train_labels.tolist()))
    ordered_train_y = np.asarray([label_by_id[sample_id] for sample_id in training_order_ids])
    if set(training_order_ids).intersection(inner_validation_ids):
        raise AssertionError("Inner autoencoder train/validation SAMPLE_ID lists overlap.")
    if set(training_order_ids).union(inner_validation_ids) != set(outer_train_ids):
        raise AssertionError("Inner autoencoder split does not partition outer-training SAMPLE_ID values.")

    return {
        "inner_train_sample_ids": training_order_ids,
        "inner_validation_sample_ids": list(inner_validation_ids),
        "inner_train_class_counts": _class_counts(ordered_train_y),
        "inner_validation_class_counts": _class_counts(inner_validation_y),
        "inner_split_seed": split_seed,
        "training_order_seed": split_seed,
        "inner_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
            training_order_ids
        ),
        "inner_validation_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
            list(inner_validation_ids)
        ),
        "training_order_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
            training_order_ids
        ),
        "labels_used_only_for": "stratification",
    }


def _select_preprocessed_rows_by_sample_id(
    matrix: np.ndarray, sample_ids: List[str], selected_sample_ids: List[str], context: str
) -> np.ndarray:
    if matrix.shape[0] != len(sample_ids):
        raise AssertionError(f"{context} matrix row count does not match its SAMPLE_ID list.")
    indexed_matrix = pd.DataFrame(matrix, index=pd.Index(sample_ids, name="SAMPLE_ID"))
    selected = indexed_matrix.loc[selected_sample_ids]
    if selected.index.tolist() != selected_sample_ids:
        raise AssertionError(f"{context} did not preserve requested SAMPLE_ID order.")
    return selected.to_numpy(dtype=np.float32, copy=True)


def _set_fold_autoencoder_seed(seed: int) -> bool:
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
        return True
    except (AttributeError, RuntimeError):
        return False


def build_fold_autoencoder(
    modality: str, input_dim: int, hidden_dims: List[int], latent_dim: int
) -> Tuple[Model, Model]:
    architecture = _validate_fold_autoencoder_config(
        modality, input_dim, {"hidden_dims": hidden_dims, "latent_dim": latent_dim}
    )
    inputs = Input(shape=(input_dim,), name=f"{modality}_input")
    encoded = inputs
    for index, hidden_dim in enumerate(architecture["hidden_dims"]):
        encoded = Dense(
            hidden_dim, activation="relu", name=f"{modality}_encoder_hidden_{index}"
        )(encoded)
    bottleneck_name = f"{modality}_bottleneck"
    bottleneck = Dense(
        architecture["latent_dim"], activation="linear", name=bottleneck_name
    )(encoded)
    decoded = bottleneck
    for index, hidden_dim in enumerate(reversed(architecture["hidden_dims"])):
        decoded = Dense(
            hidden_dim, activation="relu", name=f"{modality}_decoder_hidden_{index}"
        )(decoded)
    reconstruction = Dense(
        input_dim, activation="linear", name=f"{modality}_reconstruction"
    )(decoded)
    autoencoder = Model(inputs=inputs, outputs=reconstruction, name=f"{modality}_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    encoder = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer(bottleneck_name).output,
        name=f"{modality}_encoder",
    )
    return autoencoder, encoder


def fit_fold_modality_autoencoder(
    inner_train_matrix: np.ndarray,
    inner_validation_matrix: np.ndarray,
    modality: str,
    architecture_config: Dict[str, Any],
    training_config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    if inner_train_matrix.ndim != 2 or inner_validation_matrix.ndim != 2:
        raise ValueError(f"{modality} autoencoder inputs must be two-dimensional matrices.")
    if inner_train_matrix.shape[0] == 0 or inner_validation_matrix.shape[0] == 0:
        raise ValueError(f"{modality} autoencoder inner train/validation matrices must be non-empty.")
    if inner_train_matrix.shape[1] != inner_validation_matrix.shape[1]:
        raise ValueError(f"{modality} autoencoder inner train/validation feature counts differ.")
    if not np.isfinite(inner_train_matrix).all() or not np.isfinite(inner_validation_matrix).all():
        raise ValueError(f"{modality} autoencoder inputs must be finite.")

    architecture = _validate_fold_autoencoder_config(
        modality, inner_train_matrix.shape[1], architecture_config
    )
    training = _validate_autoencoder_training_config(training_config)
    model_seed = _validate_nonnegative_integer(f"{modality} autoencoder seed", seed)
    deterministic_ops_enabled = _set_fold_autoencoder_seed(model_seed)
    autoencoder, encoder = build_fold_autoencoder(
        modality, inner_train_matrix.shape[1], architecture["hidden_dims"], architecture["latent_dim"]
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=training["patience"], restore_best_weights=True
    )
    history = autoencoder.fit(
        inner_train_matrix,
        inner_train_matrix,
        validation_data=(inner_validation_matrix, inner_validation_matrix),
        epochs=training["epochs"],
        batch_size=min(training["batch_size"], len(inner_train_matrix)),
        callbacks=[early_stopping],
        shuffle=False,
        verbose=0,
    )
    loss_history = [float(value) for value in history.history.get("loss", [])]
    validation_loss_history = [float(value) for value in history.history.get("val_loss", [])]
    best_history = validation_loss_history if validation_loss_history else loss_history
    best_epoch = int(np.argmin(best_history) + 1) if best_history else 0
    reconstructed_train = autoencoder.predict(inner_train_matrix, verbose=0)
    reconstructed_validation = autoencoder.predict(inner_validation_matrix, verbose=0)
    return {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "metadata": {
            "modality": modality,
            "model_seed": model_seed,
            "deterministic_operations_enabled": deterministic_ops_enabled,
            "architecture": {
                "input_dim": int(inner_train_matrix.shape[1]),
                "hidden_dims": architecture["hidden_dims"],
                "latent_dim": architecture["latent_dim"],
                "bottleneck_layer_name": f"{modality}_bottleneck",
                "hidden_activation": "relu",
                "reconstruction_activation": "linear",
                "loss": "mse",
            },
            "epochs_requested": training["epochs"],
            "epochs_ran": len(loss_history),
            "best_epoch": best_epoch,
            "loss_history": loss_history,
            "validation_loss_history": validation_loss_history,
            "inner_train_reconstruction_mse": float(
                np.mean(np.square(reconstructed_train - inner_train_matrix))
            ),
            "inner_validation_reconstruction_mse": float(
                np.mean(np.square(reconstructed_validation - inner_validation_matrix))
            ),
        },
    }


def fit_fold_autoencoders(
    materialized_fold: Dict[str, Any],
    preprocessed_modalities: Dict[str, Any],
    modality_configs: Dict[str, Dict[str, Any]],
    training_config: Dict[str, Any],
    inner_validation_fraction: float,
    inner_split_seed: int,
) -> Dict[str, Any]:
    if set(modality_configs) != set(FOLD_MODALITIES):
        raise ValueError("Autoencoder configuration must contain exactly mGE, mDM, and mCNA.")
    training = _validate_autoencoder_training_config(training_config)
    inner_split = build_inner_autoencoder_split(
        materialized_fold, inner_validation_fraction, inner_split_seed
    )
    fitted_modalities = {}
    for modality_index, modality in enumerate(FOLD_MODALITIES):
        transformed = preprocessed_modalities[modality]
        metadata = transformed["metadata"]
        outer_train_ids = list(materialized_fold["train_sample_ids"])
        outer_test_ids = list(materialized_fold["test_sample_ids"])
        if metadata["train_sample_ids"] != outer_train_ids or metadata["test_sample_ids"] != outer_test_ids:
            raise AssertionError(f"{modality} preprocessed SAMPLE_ID metadata does not match the fold.")
        outer_train_matrix = _select_preprocessed_rows_by_sample_id(
            transformed["train"], outer_train_ids, outer_train_ids, f"{modality} outer train"
        )
        inner_train_matrix = _select_preprocessed_rows_by_sample_id(
            transformed["train"],
            outer_train_ids,
            inner_split["inner_train_sample_ids"],
            f"{modality} inner train",
        )
        inner_validation_matrix = _select_preprocessed_rows_by_sample_id(
            transformed["train"],
            outer_train_ids,
            inner_split["inner_validation_sample_ids"],
            f"{modality} inner validation",
        )
        model_seed = training["seed"] + modality_index
        fitted = fit_fold_modality_autoencoder(
            inner_train_matrix,
            inner_validation_matrix,
            modality,
            modality_configs[modality],
            training,
            model_seed,
        )

        # Outer-test matrices are accessed only after model fitting has completed.
        outer_test_matrix = _select_preprocessed_rows_by_sample_id(
            transformed["test"], outer_test_ids, outer_test_ids, f"{modality} outer test"
        )
        outer_train_latent = fitted["encoder"].predict(outer_train_matrix, verbose=0)
        outer_test_latent = fitted["encoder"].predict(outer_test_matrix, verbose=0)
        if not np.isfinite(outer_train_latent).all() or not np.isfinite(outer_test_latent).all():
            raise ValueError(f"{modality} autoencoder produced non-finite latent values.")
        latent_dim = fitted["metadata"]["architecture"]["latent_dim"]
        if outer_train_latent.shape != (len(outer_train_ids), latent_dim):
            raise AssertionError(f"{modality} outer-training latent shape is invalid.")
        if outer_test_latent.shape != (len(outer_test_ids), latent_dim):
            raise AssertionError(f"{modality} outer-test latent shape is invalid.")
        fitted_modalities[modality] = {
            "autoencoder": fitted["autoencoder"],
            "encoder": fitted["encoder"],
            "outer_train_latent": outer_train_latent,
            "outer_test_latent": outer_test_latent,
            "metadata": {
                **metadata,
                **fitted["metadata"],
                "outer_train_sample_ids": outer_train_ids,
                "outer_test_sample_ids": outer_test_ids,
                "inner_train_sample_ids": list(inner_split["inner_train_sample_ids"]),
                "inner_validation_sample_ids": list(inner_split["inner_validation_sample_ids"]),
            },
        }
    if len({id(value["autoencoder"]) for value in fitted_modalities.values()}) != len(FOLD_MODALITIES):
        raise AssertionError("Each modality must use an independent autoencoder model.")
    return {
        "repeat_id": materialized_fold["repeat_id"],
        "fold_id": materialized_fold["fold_id"],
        "inner_split": inner_split,
        "training_config": training,
        "modalities": fitted_modalities,
    }


def _keras_runtime_version() -> Optional[str]:
    version = getattr(tf.keras, "__version__", None)
    if version is not None:
        return str(version)
    version_function = getattr(tf.keras, "version", None)
    return str(version_function()) if callable(version_function) else None


def build_fold_autoencoder_audit(
    manifest_validation: Dict[str, Any],
    materialized_fold: Dict[str, Any],
    autoencoder_result: Dict[str, Any],
) -> Dict[str, Any]:
    inner_split = autoencoder_result["inner_split"]
    inner_split_audit = {
        key: inner_split[key]
        for key in [
            "inner_train_class_counts",
            "inner_validation_class_counts",
            "inner_split_seed",
            "training_order_seed",
            "inner_train_sample_ids_canonical_json_sha256",
            "inner_validation_sample_ids_canonical_json_sha256",
            "training_order_sample_ids_canonical_json_sha256",
            "labels_used_only_for",
        ]
    }
    modalities = {}
    for modality, fitted in autoencoder_result["modalities"].items():
        metadata = fitted["metadata"]
        modalities[modality] = {
            "feature_name_sha256": metadata["feature_name_sha256"],
            "feature_count": len(metadata["feature_names"]),
            "model_seed": metadata["model_seed"],
            "architecture": metadata["architecture"],
            "training": {
                key: metadata[key]
                for key in [
                    "epochs_requested",
                    "epochs_ran",
                    "best_epoch",
                    "loss_history",
                    "validation_loss_history",
                    "inner_train_reconstruction_mse",
                    "inner_validation_reconstruction_mse",
                ]
            },
            "latents": {
                "source": "named_bottleneck",
                "outer_train_shape": list(fitted["outer_train_latent"].shape),
                "outer_test_shape": list(fitted["outer_test_latent"].shape),
                "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                    metadata["outer_train_sample_ids"]
                ),
                "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                    metadata["outer_test_sample_ids"]
                ),
                "finite": {
                    "outer_train": bool(np.isfinite(fitted["outer_train_latent"]).all()),
                    "outer_test": bool(np.isfinite(fitted["outer_test_latent"]).all()),
                },
            },
            "deterministic_operations_enabled": metadata["deterministic_operations_enabled"],
        }
    return {
        "action": "audited_autoencoder",
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": materialized_fold["repeat_id"],
            "fold_id": materialized_fold["fold_id"],
            "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                materialized_fold["train_sample_ids"]
            ),
            "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                materialized_fold["test_sample_ids"]
            ),
            "outer_train_class_counts": materialized_fold["train_class_counts"],
            "outer_test_class_counts": materialized_fold["test_class_counts"],
            "cross_omics_alignment": materialized_fold["checks"]["cross_omics_alignment"],
            "train_test_overlap": materialized_fold["checks"]["train_test_overlap"],
        },
        "inner_split": inner_split_audit,
        "runtime": {
            "tensorflow_version": tf.__version__,
            "keras_version": _keras_runtime_version(),
        },
        "modalities": modalities,
    }


# =========================================
# SECTION 2C: Fold-scoped latent fusion utilities
# =========================================


FUSION_MODALITY_ORDER = ("mGE", "mDM", "mCNA")


def _validate_fold_latent_matrix(
    modality: str,
    matrix: Any,
    stored_sample_ids: Any,
    expected_sample_ids: List[str],
    expected_latent_dim: Any,
    partition: str,
) -> np.ndarray:
    if modality not in FUSION_MODALITY_ORDER:
        raise ValueError(f"Unknown fusion modality: {modality}")
    if not isinstance(matrix, np.ndarray) or isinstance(matrix, pd.DataFrame):
        raise ValueError(f"{modality} {partition} latent matrix must be a numeric NumPy array.")
    if matrix.ndim != 2:
        raise ValueError(f"{modality} {partition} latent matrix must be two-dimensional.")
    if matrix.dtype == object or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{modality} {partition} latent matrix must have a numeric non-object dtype.")
    if not isinstance(stored_sample_ids, list):
        raise ValueError(f"{modality} latent metadata must include an ordered {partition} SAMPLE_ID list.")
    if stored_sample_ids != expected_sample_ids:
        raise ValueError(
            f"{modality} {partition} latent SAMPLE_ID order does not match the materialized fold."
        )
    if matrix.shape[0] != len(expected_sample_ids):
        raise ValueError(
            f"{modality} {partition} latent row count does not match its SAMPLE_ID partition."
        )
    latent_dim = _validate_positive_integer(
        f"{modality} recorded latent dimension", expected_latent_dim
    )
    if matrix.shape[1] != latent_dim:
        raise ValueError(
            f"{modality} {partition} latent width does not match its recorded latent dimension."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{modality} {partition} latent matrix contains non-finite values.")
    return matrix


def build_fold_latent_feature_names(latent_dimensions: Dict[str, int]) -> List[str]:
    if set(latent_dimensions) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Latent dimensions must contain exactly mGE, mDM, and mCNA.")
    feature_names = []
    for modality in FUSION_MODALITY_ORDER:
        latent_dim = _validate_positive_integer(
            f"{modality} latent dimension", latent_dimensions[modality]
        )
        index_width = max(3, len(str(latent_dim - 1)))
        feature_names.extend(
            f"{modality}_z{index:0{index_width}d}" for index in range(latent_dim)
        )
    if len(feature_names) != len(set(feature_names)):
        raise AssertionError("Latent fusion feature names must be unique.")
    return feature_names


def build_fold_latent_slices(latent_dimensions: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    if set(latent_dimensions) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Latent dimensions must contain exactly mGE, mDM, and mCNA.")
    slices = {}
    start = 0
    for modality in FUSION_MODALITY_ORDER:
        width = _validate_positive_integer(f"{modality} latent dimension", latent_dimensions[modality])
        end = start + width
        slices[modality] = (start, end)
        start = end
    _validate_fold_latent_slices(slices, latent_dimensions, start)
    return slices


def _validate_fold_latent_slices(
    modality_slices: Dict[str, Tuple[int, int]],
    latent_dimensions: Dict[str, int],
    fused_width: int,
) -> None:
    if set(modality_slices) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Latent fusion slices must contain exactly mGE, mDM, and mCNA.")
    expected_start = 0
    for modality in FUSION_MODALITY_ORDER:
        start, end = modality_slices[modality]
        if start != expected_start or end <= start:
            raise AssertionError("Latent fusion slices must be contiguous positive half-open intervals.")
        if end - start != latent_dimensions[modality]:
            raise AssertionError(f"{modality} latent fusion slice width is invalid.")
        expected_start = end
    if expected_start != fused_width:
        raise AssertionError("Latent fusion slices do not cover the complete fused width.")


def _slice_recovery_check(source: np.ndarray, recovered: np.ndarray) -> Dict[str, Any]:
    expected = source.astype(np.float32, copy=False)
    if source.dtype == np.float32:
        passed = bool(np.array_equal(recovered, expected))
        comparison = "exact_float32"
    else:
        passed = bool(np.allclose(recovered, expected, rtol=1e-6, atol=1e-6))
        comparison = "allclose_after_float32_cast"
    if not passed:
        raise AssertionError("Fused latent slice does not recover its source modality matrix.")
    return {
        "passed": True,
        "comparison": comparison,
        "rtol": 0.0 if comparison == "exact_float32" else 1e-6,
        "atol": 0.0 if comparison == "exact_float32" else 1e-6,
    }


def fuse_fold_latents(
    materialized_fold: Dict[str, Any], autoencoder_result: Dict[str, Any]
) -> Dict[str, Any]:
    if autoencoder_result.get("repeat_id") != materialized_fold.get("repeat_id") or autoencoder_result.get(
        "fold_id"
    ) != materialized_fold.get("fold_id"):
        raise ValueError("Autoencoder result does not belong to the requested materialized fold.")
    modalities = autoencoder_result.get("modalities")
    if not isinstance(modalities, dict) or set(modalities) != set(FUSION_MODALITY_ORDER):
        raise ValueError("Autoencoder result must contain exactly mGE, mDM, and mCNA latents.")

    train_sample_ids = list(materialized_fold["train_sample_ids"])
    test_sample_ids = list(materialized_fold["test_sample_ids"])
    source_latents = {}
    latent_dimensions = {}
    source_shapes = {}
    for modality in FUSION_MODALITY_ORDER:
        fitted = modalities[modality]
        metadata = fitted.get("metadata") if isinstance(fitted, dict) else None
        if not isinstance(metadata, dict) or not isinstance(metadata.get("architecture"), dict):
            raise ValueError(f"{modality} Phase 4 latent metadata is missing.")
        latent_dim = metadata["architecture"].get("latent_dim")
        train_matrix = _validate_fold_latent_matrix(
            modality,
            fitted.get("outer_train_latent"),
            metadata.get("outer_train_sample_ids"),
            train_sample_ids,
            latent_dim,
            "outer-train",
        )
        test_matrix = _validate_fold_latent_matrix(
            modality,
            fitted.get("outer_test_latent"),
            metadata.get("outer_test_sample_ids"),
            test_sample_ids,
            latent_dim,
            "outer-test",
        )
        source_latents[modality] = {"train": train_matrix, "test": test_matrix}
        latent_dimensions[modality] = int(latent_dim)
        source_shapes[modality] = {
            "outer_train": list(train_matrix.shape),
            "outer_test": list(test_matrix.shape),
        }

    modality_slices = build_fold_latent_slices(latent_dimensions)
    feature_names = build_fold_latent_feature_names(latent_dimensions)
    expected_width = sum(latent_dimensions[modality] for modality in FUSION_MODALITY_ORDER)
    fused_outer_train = np.concatenate(
        [source_latents[modality]["train"] for modality in FUSION_MODALITY_ORDER], axis=1
    ).astype(np.float32)
    fused_outer_test = np.concatenate(
        [source_latents[modality]["test"] for modality in FUSION_MODALITY_ORDER], axis=1
    ).astype(np.float32)
    if fused_outer_train.shape != (len(train_sample_ids), expected_width):
        raise AssertionError("Fused outer-training latent shape is invalid.")
    if fused_outer_test.shape != (len(test_sample_ids), expected_width):
        raise AssertionError("Fused outer-test latent shape is invalid.")
    if fused_outer_train.dtype != np.float32 or fused_outer_test.dtype != np.float32:
        raise AssertionError("Fused latent matrices must use float32 dtype.")
    if not np.isfinite(fused_outer_train).all() or not np.isfinite(fused_outer_test).all():
        raise ValueError("Fused latent matrices contain non-finite values after float32 conversion.")
    _validate_fold_latent_slices(modality_slices, latent_dimensions, expected_width)

    slice_recovery = {}
    for modality in FUSION_MODALITY_ORDER:
        start, end = modality_slices[modality]
        slice_recovery[modality] = {
            "outer_train": _slice_recovery_check(
                source_latents[modality]["train"], fused_outer_train[:, start:end]
            ),
            "outer_test": _slice_recovery_check(
                source_latents[modality]["test"], fused_outer_test[:, start:end]
            ),
        }
    target_leakage_guard = {
        "passed": True,
        "excluded_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "latent_inputs_are_numeric_arrays": True,
        "labels_returned_separately": True,
    }
    if any(name in FORBIDDEN_FEATURE_COLUMNS for name in feature_names):
        raise AssertionError("Latent fusion feature names include a forbidden target or identifier column.")

    return {
        "repeat_id": materialized_fold["repeat_id"],
        "fold_id": materialized_fold["fold_id"],
        "fusion_order": list(FUSION_MODALITY_ORDER),
        "fused_outer_train": fused_outer_train,
        "fused_outer_test": fused_outer_test,
        "y_train": np.array(materialized_fold["y_train"], dtype=int, copy=True),
        "y_test": np.array(materialized_fold["y_test"], dtype=int, copy=True),
        "train_sample_ids": train_sample_ids,
        "test_sample_ids": test_sample_ids,
        "train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(train_sample_ids),
        "test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(test_sample_ids),
        "latent_feature_names": feature_names,
        "latent_feature_name_sha256": _feature_name_sha256(feature_names),
        "modality_slices": modality_slices,
        "latent_dimensions": latent_dimensions,
        "source_shapes": source_shapes,
        "fused_shapes": {
            "outer_train": list(fused_outer_train.shape),
            "outer_test": list(fused_outer_test.shape),
        },
        "dtype": str(fused_outer_train.dtype),
        "checks": {
            "finite_outputs": {
                "outer_train": True,
                "outer_test": True,
            },
            "target_leakage_guard": target_leakage_guard,
            "slice_round_trip_recovery": {"passed": True, "modalities": slice_recovery},
            "float32_cast": {
                "passed": True,
                "shape_preserved": True,
                "sample_id_order_preserved": True,
                "finite_after_cast": True,
            },
        },
    }


def build_fold_latent_fusion_audit(
    manifest_validation: Dict[str, Any],
    materialized_fold: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "action": "audited_latent_fusion",
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": fusion_result["repeat_id"],
            "fold_id": fusion_result["fold_id"],
            "outer_train_sample_count": len(fusion_result["train_sample_ids"]),
            "outer_test_sample_count": len(fusion_result["test_sample_ids"]),
            "train_class_counts": materialized_fold["train_class_counts"],
            "test_class_counts": materialized_fold["test_class_counts"],
            "train_sample_ids_canonical_json_sha256": fusion_result[
                "train_sample_ids_canonical_json_sha256"
            ],
            "test_sample_ids_canonical_json_sha256": fusion_result[
                "test_sample_ids_canonical_json_sha256"
            ],
        },
        "fusion": {
            "order": fusion_result["fusion_order"],
            "latent_dimensions": fusion_result["latent_dimensions"],
            "modality_slices": {
                modality: list(fusion_result["modality_slices"][modality])
                for modality in FUSION_MODALITY_ORDER
            },
            "modality_latent_shapes": {
                modality: fusion_result["source_shapes"][modality]
                for modality in FUSION_MODALITY_ORDER
            },
            "fused_shapes": fusion_result["fused_shapes"],
            "dtype": fusion_result["dtype"],
            "latent_feature_name_sha256": fusion_result["latent_feature_name_sha256"],
            "finite_outputs": fusion_result["checks"]["finite_outputs"],
            "target_leakage_guard": fusion_result["checks"]["target_leakage_guard"],
            "slice_round_trip_recovery": fusion_result["checks"][
                "slice_round_trip_recovery"
            ],
        },
    }


# =========================================
# SECTION 2D: Fold-scoped mandatory minority CTGAN utilities
# =========================================


CTGAN_REQUIRED_CONSTRUCTOR_ARGUMENTS = {"metadata", "epochs", "batch_size", "pac", "verbose"}
MINORITY_CTGAN_STRATEGY = "minority_only_ctgan"
CTGAN_FORBIDDEN_FEATURE_NAME_TOKENS = {
    "class",
    "sample_id",
    "target",
    "label",
    "is_synthetic",
    "synthetic",
    "real",
}
CTGAN_EXECUTION_BACKEND = "isolated_cpu_subprocess_v1"

# This worker intentionally imports no pipeline, TensorFlow, or Keras modules.
_PHASE10C_CTGAN_WORKER_SOURCE = r'''
import hashlib
import inspect
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
import sdv
import ctgan
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

def canonical(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

def array_hash(array):
    digest = hashlib.sha256()
    digest.update(canonical({"dtype": str(array.dtype), "shape": list(array.shape)}))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()

def main(request_path, input_path, output_path, response_path):
    request = json.loads(open(request_path, encoding="utf-8").read())
    with np.load(input_path) as input_data:
        features = input_data["features"]
        labels = input_data["labels"] if "labels" in input_data.files else None
    if array_hash(features) != request["input_features_sha256"]:
        raise ValueError("CTGAN worker input feature hash differs from request.")
    if labels is not None and array_hash(labels) != request["input_labels_sha256"]:
        raise ValueError("CTGAN worker input label hash differs from request.")
    if request["strategy"] not in {"minority_only_ctgan", "conditional_all_training_ctgan"}:
        raise ValueError("CTGAN worker strategy is unsupported.")
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
        raise ValueError("CTGAN worker features must be finite numeric matrix.")
    feature_names = request["feature_names"]
    if len(feature_names) != features.shape[1] or len(feature_names) != len(set(feature_names)):
        raise ValueError("CTGAN worker feature schema is invalid.")
    if request["feature_name_sha256"] != hashlib.sha256(canonical(feature_names)).hexdigest():
        raise ValueError("CTGAN worker feature-name hash differs from request.")
    config = request["ctgan_config"]
    required = {"metadata", "epochs", "batch_size", "pac", "verbose"}
    constructor_parameters = list(inspect.signature(CTGANSynthesizer).parameters)
    if required.difference(constructor_parameters):
        raise RuntimeError("Mandatory CTGAN constructor API is incompatible; no augmentation fallback exists.")
    if int(config["batch_size"]) % int(config["pac"]) != 0:
        raise ValueError("CTGAN worker batch size must be divisible by pac.")
    seed = int(request["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if request["strategy"] == "minority_only_ctgan":
        if labels is not None:
            raise ValueError("Minority-only CTGAN worker must not receive labels.")
        training = pd.DataFrame(features, columns=feature_names)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=training)
        if callable(getattr(metadata, "validate", None)):
            metadata.validate()
        metadata_schema = {"controlled_outcome_column": None, "latent_columns_only": True}
    else:
        if labels is None or labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("Conditional CTGAN worker labels must align to real training features.")
        training = pd.DataFrame(features, columns=feature_names)
        training[request["outcome_column"]] = labels.astype(int)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=training)
        metadata.update_column(column_name=request["outcome_column"], sdtype="categorical")
        metadata.validate()
        metadata.validate_data(data=training)
        metadata_schema = {"training_columns": training.columns.tolist(), "outcome_column": request["outcome_column"], "outcome_sdtype": metadata.columns[request["outcome_column"]]["sdtype"], "latent_sdtypes": {name: metadata.columns[name]["sdtype"] for name in feature_names}}
        if metadata_schema["outcome_sdtype"] != "categorical" or any(value != "numerical" for value in metadata_schema["latent_sdtypes"].values()):
            raise ValueError("Conditional CTGAN metadata schema is invalid.")
    kwargs = {"metadata": metadata, **config}
    if "enable_gpu" in constructor_parameters:
        kwargs["enable_gpu"] = False
    synthesizer = CTGANSynthesizer(**kwargs)
    synthesizer.fit(training)
    requested_rows = int(request["requested_synthetic_rows"])
    condition = None
    if request["strategy"] == "minority_only_ctgan":
        samples = synthesizer.sample(requested_rows)
        if samples.columns.tolist() != feature_names:
            raise ValueError("Minority-only CTGAN worker output columns are invalid.")
        synthetic = samples.to_numpy(dtype=np.float32, copy=True)
        synthetic_labels = None
    else:
        from sdv.sampling import Condition
        condition = Condition({request["outcome_column"]: int(request["minority_label"])}, num_rows=requested_rows)
        samples = synthesizer.sample_from_conditions([condition])
        expected_columns = feature_names + [request["outcome_column"]]
        if samples.columns.tolist() != expected_columns:
            raise ValueError("Conditional CTGAN worker output columns are invalid.")
        synthetic_labels = samples[request["outcome_column"]].to_numpy(dtype=int, copy=True)
        if not np.array_equal(synthetic_labels, np.full(requested_rows, int(request["minority_label"]), dtype=int)):
            raise ValueError("Conditional CTGAN worker output labels are invalid.")
        synthetic = samples.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)
    if synthetic.shape != (requested_rows, len(feature_names)) or not np.isfinite(synthetic).all():
        raise ValueError("CTGAN worker synthetic output is invalid.")
    np.savez_compressed(output_path, synthetic=synthetic, **({"synthetic_labels": synthetic_labels} if synthetic_labels is not None else {}))
    losses = {"available": False}
    get_loss_values = getattr(synthesizer, "get_loss_values", None)
    if callable(get_loss_values):
        values = get_loss_values()
        if isinstance(values, pd.DataFrame):
            summary = {"available": True, "row_count": int(len(values)), "columns": values.columns.tolist()}
            if len(values): summary["final_row"] = {str(key): float(value) for key, value in values.iloc[-1].items()}
            losses = {"summary": summary}
        else:
            losses = {"summary": {"available": True, "type": type(values).__name__}}
    response = {"schema_version": "phase10c-ctgan-subprocess-v1", "strategy": request["strategy"], "feature_names": feature_names, "feature_name_sha256": request["feature_name_sha256"], "requested_synthetic_rows": requested_rows, "returned_synthetic_rows": int(len(synthetic)), "input_features_sha256": array_hash(features), "input_labels_sha256": None if labels is None else array_hash(labels), "synthetic_sha256": array_hash(synthetic), "synthetic_labels_sha256": None if synthetic_labels is None else array_hash(synthetic_labels), "synthetic_dtype": str(synthetic.dtype), "ctgan_api": {"sdv_version": str(sdv.__version__), "ctgan_version": str(getattr(ctgan, "__version__", "unknown")), "constructor_parameter_names": constructor_parameters, "ctgan_class_module": CTGANSynthesizer.__module__, "ctgan_class_path": inspect.getfile(CTGANSynthesizer)}, "constructor_configuration": {"metadata_supplied": True, **{key: value for key, value in kwargs.items() if key != "metadata"}}, "seed_evidence": {"python_random_seed": seed, "numpy_seed": seed, "pytorch_seed": seed, "pytorch_version": str(torch.__version__), "ctgan_constructor_seed_control": "unsupported_in_sdv_1_37_3_public_constructor", "exact_ctgan_reproducibility_guaranteed": False}, "loss_values": losses, "metadata_schema": metadata_schema, "condition": None if condition is None else {"column": request["outcome_column"], "value": int(request["minority_label"]), "num_rows": requested_rows}, "execution_evidence": {"ctgan_execution_backend": "isolated_cpu_subprocess_v1", "tensorflow_present_in_ctgan_worker": False, "ctgan_gpu_enabled": False}}
    open(response_path, "w", encoding="utf-8").write(json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__":
    main(*sys.argv[1:])
'''


def _phase10c_run_ctgan_cpu_subprocess(strategy: str, features: np.ndarray, labels: Optional[np.ndarray], feature_names: List[str], feature_name_sha256: str, ctgan_config: Dict[str, Any], requested_synthetic_rows: int, minority_label: int, global_seed: int) -> Dict[str, Any]:
    """Run the mandatory CTGAN fit/sample operation outside the TensorFlow process."""
    if strategy not in {MINORITY_CTGAN_STRATEGY, CONDITIONAL_CTGAN_STRATEGY}:
        raise ValueError("Phase 10C CTGAN subprocess strategy is unsupported.")
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.dtype == object or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("Phase 10C CTGAN subprocess features must be a finite numeric matrix.")
    names = _validate_ctgan_feature_schema(feature_names, feature_name_sha256, matrix.shape[1])
    config = _validate_minority_ctgan_config(ctgan_config, list(CTGAN_REQUIRED_CONSTRUCTOR_ARGUMENTS))
    rows = _validate_positive_integer("requested CTGAN synthetic rows", requested_synthetic_rows)
    seed = _validate_nonnegative_integer("CTGAN subprocess seed", global_seed)
    label_array = None if labels is None else np.asarray(labels)
    if strategy == MINORITY_CTGAN_STRATEGY:
        if label_array is not None:
            raise ValueError("Minority-only CTGAN subprocess must receive minority features without labels.")
    else:
        if label_array is None or label_array.ndim != 1 or len(label_array) != len(matrix) or not np.issubdtype(label_array.dtype, np.number) or not np.isfinite(label_array).all():
            raise ValueError("Conditional CTGAN subprocess labels must be finite numeric values aligned to real training rows.")
        label_array = label_array.astype(int, copy=True)
    matrix = np.ascontiguousarray(matrix.astype(np.float32, copy=True))
    input_feature_hash = _array_sha256(matrix)
    input_label_hash = None if label_array is None else _array_sha256(label_array)
    request = {"schema_version": "phase10c-ctgan-subprocess-v1", "strategy": strategy, "feature_names": names, "feature_name_sha256": feature_name_sha256, "ctgan_config": config, "requested_synthetic_rows": rows, "minority_label": int(minority_label), "seed": seed, "outcome_column": CONDITIONAL_OUTCOME_COLUMN, "input_features_sha256": input_feature_hash, "input_labels_sha256": input_label_hash}
    with tempfile.TemporaryDirectory(prefix="phase10c-ctgan-") as temporary:
        directory = Path(temporary)
        request_path, input_path = directory / "request.json", directory / "input.npz"
        output_path, response_path = directory / "synthetic.npz", directory / "response.json"
        _phase10c_atomic_write_json(request_path, request)
        payload = {"features": matrix}
        if label_array is not None:
            payload["labels"] = label_array
        np.savez_compressed(input_path, **payload)
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONFAULTHANDLER": "1"})
        try:
            completed = subprocess.run([sys.executable, "-c", _PHASE10C_CTGAN_WORKER_SOURCE, str(request_path), str(input_path), str(output_path), str(response_path)], cwd=str(directory), env=environment, capture_output=True, text=True, check=False)
        except OSError as error:
            raise RuntimeError("Mandatory CTGAN isolated CPU subprocess could not be started; no augmentation fallback exists.") from error
        if completed.returncode != 0:
            raise RuntimeError(f"Mandatory CTGAN isolated CPU subprocess failed with return code {completed.returncode}; no augmentation fallback exists. stderr: {completed.stderr[-2000:]}")
        if not output_path.is_file() or not response_path.is_file():
            raise RuntimeError("Mandatory CTGAN isolated CPU subprocess did not produce its required output files; no augmentation fallback exists.")
        try:
            response = _phase10c_read_json(response_path, "Phase 10C CTGAN subprocess response")
            with np.load(output_path) as output:
                synthetic = output["synthetic"]
                synthetic_labels = output["synthetic_labels"] if "synthetic_labels" in output.files else None
        except (FileNotFoundError, KeyError, OSError, ValueError) as error:
            raise RuntimeError("Mandatory CTGAN isolated CPU subprocess output is malformed; no augmentation fallback exists.") from error
    required_response = {"schema_version": "phase10c-ctgan-subprocess-v1", "strategy": strategy, "feature_names": names, "feature_name_sha256": feature_name_sha256, "requested_synthetic_rows": rows, "returned_synthetic_rows": rows, "input_features_sha256": input_feature_hash, "input_labels_sha256": input_label_hash}
    if any(response.get(key) != value for key, value in required_response.items()):
        raise RuntimeError("Mandatory CTGAN isolated CPU subprocess response contract differs from the request; no augmentation fallback exists.")
    constructor_configuration = response.get("constructor_configuration")
    if not isinstance(constructor_configuration, dict) or constructor_configuration.get("metadata_supplied") is not True or any(constructor_configuration.get(key) != value for key, value in config.items()) or constructor_configuration.get("enable_gpu", False) is not False:
        raise RuntimeError("Mandatory CTGAN isolated CPU subprocess configuration differs from the request; no augmentation fallback exists.")
    execution = response.get("execution_evidence")
    if execution != {"ctgan_execution_backend": CTGAN_EXECUTION_BACKEND, "tensorflow_present_in_ctgan_worker": False, "ctgan_gpu_enabled": False}:
        raise RuntimeError("Mandatory CTGAN isolated CPU subprocess execution evidence is invalid; no augmentation fallback exists.")
    if not isinstance(synthetic, np.ndarray) or synthetic.ndim != 2 or synthetic.shape != (rows, len(names)) or synthetic.dtype == object or not np.issubdtype(synthetic.dtype, np.number) or not np.isfinite(synthetic).all():
        raise RuntimeError("Mandatory CTGAN isolated CPU subprocess synthetic output is invalid; no augmentation fallback exists.")
    synthetic = np.ascontiguousarray(synthetic.astype(np.float32, copy=False))
    if response.get("synthetic_dtype") != str(synthetic.dtype) or response.get("synthetic_sha256") != _array_sha256(synthetic):
        raise RuntimeError("Mandatory CTGAN isolated CPU subprocess synthetic hash or dtype is invalid; no augmentation fallback exists.")
    if strategy == MINORITY_CTGAN_STRATEGY:
        if synthetic_labels is not None:
            raise RuntimeError("Minority-only CTGAN isolated CPU subprocess output must not contain labels; no augmentation fallback exists.")
    else:
        if synthetic_labels is None or synthetic_labels.ndim != 1 or len(synthetic_labels) != rows or not np.issubdtype(synthetic_labels.dtype, np.number) or not np.array_equal(synthetic_labels.astype(int), np.full(rows, int(minority_label), dtype=int)):
            raise RuntimeError("Conditional CTGAN isolated CPU subprocess output labels are invalid; no augmentation fallback exists.")
        if response.get("synthetic_labels_sha256") != _array_sha256(synthetic_labels):
            raise RuntimeError("Conditional CTGAN isolated CPU subprocess output label hash is invalid; no augmentation fallback exists.")
    return {"synthetic_matrix": synthetic, "constructor_configuration": response["constructor_configuration"], "seed_evidence": response["seed_evidence"], "loss_values": response["loss_values"], "ctgan_api": response["ctgan_api"], "metadata_schema": response.get("metadata_schema"), "condition": response.get("condition"), "condition_check": None if strategy == MINORITY_CTGAN_STRATEGY else {"passed": True, "outcome_column": CONDITIONAL_OUTCOME_COLUMN, "outcome_value": int(minority_label), "outcome_values_all_match_minority": True}, "execution_evidence": execution}


def load_sdv_ctgan_api() -> Dict[str, Any]:
    try:
        sdv_module = importlib.import_module("sdv")
        metadata_module = importlib.import_module("sdv.metadata")
        single_table_module = importlib.import_module("sdv.single_table")
        ctgan_module = importlib.import_module("ctgan")
        metadata_class = getattr(metadata_module, "SingleTableMetadata")
        synthesizer_class = getattr(single_table_module, "CTGANSynthesizer")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Mandatory CTGAN dependency is unavailable or incompatible; no augmentation fallback exists."
        ) from error
    try:
        constructor_parameters = list(inspect.signature(synthesizer_class).parameters)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Mandatory CTGAN constructor API cannot be inspected; no augmentation fallback exists."
        ) from error
    missing_required = sorted(CTGAN_REQUIRED_CONSTRUCTOR_ARGUMENTS.difference(constructor_parameters))
    if missing_required:
        raise RuntimeError(
            "Installed CTGANSynthesizer lacks required constructor arguments "
            f"{missing_required}; no augmentation fallback exists."
        )
    return {
        "sdv_version": str(sdv_module.__version__),
        "ctgan_version": str(getattr(ctgan_module, "__version__", "unknown")),
        "metadata_class": metadata_class,
        "synthesizer_class": synthesizer_class,
        "constructor_parameter_names": constructor_parameters,
        "ctgan_class_module": synthesizer_class.__module__,
        "ctgan_class_path": inspect.getfile(synthesizer_class),
        "metadata_class_module": metadata_class.__module__,
        "metadata_class_path": inspect.getfile(metadata_class),
    }


def _validate_minority_ctgan_config(
    config: Dict[str, Any], constructor_parameter_names: List[str]
) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("CTGAN configuration must be an object.")
    allowed_config_keys = {"epochs", "batch_size", "pac", "verbose"}
    unsupported_config_keys = sorted(set(config).difference(allowed_config_keys))
    if unsupported_config_keys:
        raise ValueError(
            f"Unsupported CTGAN configuration arguments: {unsupported_config_keys}"
        )
    epochs = _validate_positive_integer("ctgan_epochs", config.get("epochs"))
    batch_size = _validate_positive_integer("ctgan_batch_size", config.get("batch_size"))
    pac = _validate_positive_integer("ctgan_pac", config.get("pac"))
    verbose = config.get("verbose")
    if not isinstance(verbose, bool):
        raise ValueError("ctgan_verbose must be boolean.")
    if batch_size % 2 != 0:
        raise ValueError("ctgan_batch_size must be even.")
    if batch_size % pac != 0:
        raise ValueError("ctgan_batch_size must be divisible by ctgan_pac.")
    unsupported = CTGAN_REQUIRED_CONSTRUCTOR_ARGUMENTS.difference(constructor_parameter_names)
    if unsupported:
        raise ValueError(
            f"Installed CTGANSynthesizer does not support required arguments: {sorted(unsupported)}"
        )
    return {"epochs": epochs, "batch_size": batch_size, "pac": pac, "verbose": verbose}


def _validate_ctgan_feature_schema(
    latent_feature_names: Any, latent_feature_name_sha256: Any, expected_width: int
) -> List[str]:
    if not isinstance(latent_feature_names, list) or len(latent_feature_names) != expected_width:
        raise ValueError("CTGAN latent feature names must be an ordered list matching fused width.")
    if len(latent_feature_names) != len(set(latent_feature_names)):
        raise ValueError("CTGAN latent feature names must be unique.")
    if _feature_name_sha256(latent_feature_names) != latent_feature_name_sha256:
        raise ValueError("CTGAN latent feature-name SHA256 does not match the supplied schema.")
    for feature_name in latent_feature_names:
        if not isinstance(feature_name, str) or not feature_name:
            raise ValueError("CTGAN latent feature names must be non-empty strings.")
        normalized = feature_name.lower()
        if normalized in CTGAN_FORBIDDEN_FEATURE_NAME_TOKENS or any(
            token in normalized for token in CTGAN_FORBIDDEN_FEATURE_NAME_TOKENS
        ):
            raise ValueError(
                f"CTGAN latent feature name is forbidden because it represents a target or identifier: {feature_name}"
            )
    return list(latent_feature_names)


def extract_minority_ctgan_training_input(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
) -> Dict[str, Any]:
    if not isinstance(fused_outer_train, np.ndarray) or fused_outer_train.ndim != 2:
        raise ValueError("CTGAN fused outer-training input must be a two-dimensional NumPy array.")
    if fused_outer_train.dtype == object or not np.issubdtype(fused_outer_train.dtype, np.number):
        raise ValueError("CTGAN fused outer-training input must have a numeric non-object dtype.")
    if not np.isfinite(fused_outer_train).all():
        raise ValueError("CTGAN fused outer-training input must contain only finite values.")
    if not isinstance(train_sample_ids, list) or len(train_sample_ids) != fused_outer_train.shape[0]:
        raise ValueError("CTGAN outer-training SAMPLE_ID metadata must match fused row count.")
    if len(train_sample_ids) != len(set(train_sample_ids)):
        raise ValueError("CTGAN outer-training SAMPLE_ID metadata must be unique.")
    feature_names = _validate_ctgan_feature_schema(
        latent_feature_names, latent_feature_name_sha256, fused_outer_train.shape[1]
    )
    labels = np.asarray(y_train)
    if labels.ndim != 1 or len(labels) != fused_outer_train.shape[0]:
        raise ValueError("CTGAN y_train must be a one-dimensional array matching fused row count.")
    if labels.dtype == object or not np.issubdtype(labels.dtype, np.number):
        raise ValueError("CTGAN y_train must have a numeric dtype.")
    if not np.isfinite(labels).all():
        raise ValueError("CTGAN y_train must contain only finite values.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Minority-only CTGAN requires exactly two observed training classes.")
    if counts[0] == counts[1]:
        raise ValueError("Minority-only CTGAN requires strict class imbalance.")
    minority_index = int(np.argmin(counts))
    majority_index = int(np.argmax(counts))
    minority_label = int(classes[minority_index])
    majority_label = int(classes[majority_index])
    minority_count = int(counts[minority_index])
    majority_count = int(counts[majority_index])
    needed_synthetic_count = majority_count - minority_count
    minority_df = pd.DataFrame(
        fused_outer_train[labels == classes[minority_index]].copy(), columns=feature_names
    )
    if minority_df.columns.tolist() != feature_names or minority_df.shape != (
        minority_count,
        fused_outer_train.shape[1],
    ):
        raise AssertionError("CTGAN minority training DataFrame schema is invalid.")
    if not all(pd.api.types.is_numeric_dtype(minority_df[column]) for column in minority_df.columns):
        raise ValueError("CTGAN minority training DataFrame must contain only numerical latent features.")
    return {
        "minority_df": minority_df,
        "labels": np.array(labels, dtype=int, copy=True),
        "minority_label": minority_label,
        "majority_label": majority_label,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "needed_synthetic_count": needed_synthetic_count,
        "train_sample_ids": list(train_sample_ids),
        "feature_names": feature_names,
        "feature_name_sha256": latent_feature_name_sha256,
    }


def _set_ctgan_global_seeds(seed: int) -> Dict[str, Any]:
    normalized_seed = _validate_nonnegative_integer("ctgan global seed", seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    return {
        "python_random_seed": normalized_seed,
        "numpy_seed": normalized_seed,
        "pytorch_seed": None,
        "pytorch_version": None,
        "ctgan_constructor_seed_control": "unsupported_in_sdv_1_37_3_public_constructor",
        "exact_ctgan_reproducibility_guaranteed": False,
    }


def _validate_ctgan_synthetic_output(
    synthetic_df: Any, feature_names: List[str], needed_synthetic_count: int
) -> np.ndarray:
    if not isinstance(synthetic_df, pd.DataFrame):
        raise ValueError("CTGAN sample output must be a pandas DataFrame.")
    if synthetic_df.shape[0] != needed_synthetic_count:
        raise ValueError("CTGAN sample row count does not equal the required synthetic count.")
    if synthetic_df.columns.tolist() != feature_names:
        raise ValueError("CTGAN sample columns do not exactly match the latent feature schema and order.")
    if not all(pd.api.types.is_numeric_dtype(synthetic_df[column]) for column in synthetic_df.columns):
        raise ValueError("CTGAN sample output must contain only numeric latent features.")
    synthetic_matrix = synthetic_df.to_numpy(dtype=np.float32, copy=True)
    if synthetic_matrix.shape != (needed_synthetic_count, len(feature_names)):
        raise AssertionError("CTGAN synthetic matrix shape changed during float32 conversion.")
    if not np.isfinite(synthetic_matrix).all():
        raise ValueError("CTGAN sample output contains non-finite values.")
    return synthetic_matrix


def _summarize_ctgan_loss_values(synthesizer: Any) -> Dict[str, Any]:
    get_loss_values = getattr(synthesizer, "get_loss_values", None)
    if not callable(get_loss_values):
        return {"available": False}
    try:
        loss_values = get_loss_values()
    except Exception as error:
        raise RuntimeError("Mandatory CTGAN loss retrieval failed; no augmentation fallback exists.") from error
    if isinstance(loss_values, pd.DataFrame):
        summary = {
            "available": True,
            "row_count": int(len(loss_values)),
            "columns": loss_values.columns.tolist(),
        }
        if len(loss_values):
            summary["final_row"] = {
                str(column): float(value)
                for column, value in loss_values.iloc[-1].items()
            }
        return {"summary": summary, "values": loss_values.copy()}
    return {"summary": {"available": True, "type": type(loss_values).__name__}, "values": loss_values}


def fit_and_sample_minority_ctgan(
    minority_df: pd.DataFrame,
    needed_synthetic_count: int,
    ctgan_config: Dict[str, Any],
    api: Dict[str, Any],
    global_seed: int,
) -> Dict[str, Any]:
    if not isinstance(minority_df, pd.DataFrame):
        raise ValueError("Minority-only CTGAN input must be a DataFrame.")
    names = minority_df.columns.tolist()
    return _phase10c_run_ctgan_cpu_subprocess(MINORITY_CTGAN_STRATEGY, minority_df.to_numpy(dtype=np.float32, copy=True), None, names, _feature_name_sha256(names), ctgan_config, needed_synthetic_count, 0, global_seed)


def build_minority_ctgan_augmentation(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
    repeat_id: int,
    fold_id: int,
    ctgan_config: Dict[str, Any],
    global_seed: int,
    api: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    real_train_before = np.array(fused_outer_train, copy=True)
    labels_before = np.array(y_train, copy=True)
    extracted = extract_minority_ctgan_training_input(
        fused_outer_train,
        y_train,
        train_sample_ids,
        latent_feature_names,
        latent_feature_name_sha256,
    )
    fitted = fit_and_sample_minority_ctgan(
        extracted["minority_df"],
        extracted["needed_synthetic_count"],
        ctgan_config,
        {} if api is None else api,
        global_seed,
    )
    if not np.array_equal(fused_outer_train, real_train_before) or not np.array_equal(y_train, labels_before):
        raise AssertionError("Mandatory CTGAN altered real outer-training features or labels.")
    real_outer_train = np.asarray(fused_outer_train, dtype=np.float32).copy()
    synthetic_minority = fitted["synthetic_matrix"]
    augmented_outer_train = np.vstack([real_outer_train, synthetic_minority]).astype(np.float32)
    y_augmented = np.concatenate(
        [
            extracted["labels"],
            np.full(extracted["needed_synthetic_count"], extracted["minority_label"], dtype=int),
        ]
    )
    is_synthetic = np.concatenate(
        [
            np.zeros(len(real_outer_train), dtype=bool),
            np.ones(extracted["needed_synthetic_count"], dtype=bool),
        ]
    )
    synthetic_record_ids = [
        f"SYNTHETIC:R{repeat_id:03d}:F{fold_id:03d}:CLASS{extracted['minority_label']}:{index:06d}"
        for index in range(extracted["needed_synthetic_count"])
    ]
    if len(synthetic_record_ids) != len(set(synthetic_record_ids)) or set(synthetic_record_ids).intersection(
        extracted["train_sample_ids"]
    ):
        raise AssertionError("Synthetic record identifiers must be unique and distinct from real SAMPLE_ID values.")
    augmented_record_ids = list(extracted["train_sample_ids"]) + synthetic_record_ids
    if augmented_outer_train.shape != (len(y_augmented), len(extracted["feature_names"])):
        raise AssertionError("Augmented CTGAN matrix shape is invalid.")
    if not np.isfinite(augmented_outer_train).all():
        raise ValueError("Augmented CTGAN matrix contains non-finite values.")
    original_class_counts = _class_counts(extracted["labels"])
    augmented_class_counts = _class_counts(y_augmented)
    if augmented_class_counts[str(extracted["minority_label"])] != extracted["majority_count"]:
        raise AssertionError("CTGAN augmentation did not balance the minority class to the majority count.")
    return {
        "strategy": MINORITY_CTGAN_STRATEGY,
        "repeat_id": int(repeat_id),
        "fold_id": int(fold_id),
        "real_outer_train": real_outer_train,
        "synthetic_minority": synthetic_minority,
        "augmented_outer_train": augmented_outer_train,
        "y_augmented": y_augmented,
        "is_synthetic": is_synthetic,
        "real_sample_ids": list(extracted["train_sample_ids"]),
        "synthetic_record_ids": synthetic_record_ids,
        "augmented_record_ids": augmented_record_ids,
        "minority_label": extracted["minority_label"],
        "majority_label": extracted["majority_label"],
        "minority_count": extracted["minority_count"],
        "majority_count": extracted["majority_count"],
        "needed_synthetic_count": extracted["needed_synthetic_count"],
        "generated_synthetic_count": int(len(synthetic_minority)),
        "original_class_counts": original_class_counts,
        "augmented_class_counts": augmented_class_counts,
        "feature_names": list(extracted["feature_names"]),
        "feature_name_sha256": extracted["feature_name_sha256"],
        "ctgan_api": fitted["ctgan_api"],
        "ctgan_configuration": fitted["constructor_configuration"],
        "seed_evidence": fitted["seed_evidence"],
        "loss_values": fitted["loss_values"],
        "ctgan_execution": fitted["execution_evidence"],
        "checks": {
            "synthetic_schema": {"passed": True},
            "synthetic_finite": {"passed": True},
            "real_rows_first": {"passed": True},
            "synthetic_rows_second": {"passed": True},
            "synthetic_record_ids": {"passed": True},
            "fallback_exists": False,
            "outer_test_supplied_to_ctgan": False,
            "real_outer_training_unchanged": {"passed": True},
        },
    }


def build_minority_ctgan_audit(
    manifest_validation: Dict[str, Any],
    materialized_fold: Dict[str, Any],
    fusion_result: Dict[str, Any],
    augmentation_result: Dict[str, Any],
    outer_test_unchanged: bool,
) -> Dict[str, Any]:
    loss_values = augmentation_result["loss_values"]
    loss_summary = loss_values.get("summary", {"available": False})
    return {
        "action": "audited_minority_ctgan",
        "strategy": augmentation_result["strategy"],
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": augmentation_result["repeat_id"],
            "fold_id": augmentation_result["fold_id"],
            "outer_train_sample_count": len(augmentation_result["real_sample_ids"]),
            "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                augmentation_result["real_sample_ids"]
            ),
        },
        "ctgan": {
            **augmentation_result["ctgan_api"],
            "execution": augmentation_result["ctgan_execution"],
            "constructor_configuration": augmentation_result["ctgan_configuration"],
            "seed_control": augmentation_result["seed_evidence"],
            "fit_scope": "minority_outer_training_latents_only",
            "minority_label": augmentation_result["minority_label"],
            "majority_label": augmentation_result["majority_label"],
            "minority_count": augmentation_result["minority_count"],
            "majority_count": augmentation_result["majority_count"],
            "minority_fit_matrix_shape": [
                augmentation_result["minority_count"],
                len(augmentation_result["feature_names"]),
            ],
            "latent_feature_name_sha256": augmentation_result["feature_name_sha256"],
            "needed_synthetic_count": augmentation_result["needed_synthetic_count"],
            "generated_synthetic_count": augmentation_result["generated_synthetic_count"],
            "synthetic_shape": list(augmentation_result["synthetic_minority"].shape),
            "synthetic_dtype": str(augmentation_result["synthetic_minority"].dtype),
            "schema_check": augmentation_result["checks"]["synthetic_schema"],
            "finite_check": augmentation_result["checks"]["synthetic_finite"],
            "fallback_exists": False,
            "outer_test_supplied_to_ctgan": False,
            "outer_test_features_and_labels_unchanged": bool(outer_test_unchanged),
            "loss_summary": loss_summary,
        },
        "augmentation": {
            "original_class_counts": augmentation_result["original_class_counts"],
            "augmented_class_counts": augmentation_result["augmented_class_counts"],
            "real_rows_first": augmentation_result["checks"]["real_rows_first"],
            "synthetic_rows_second": augmentation_result["checks"]["synthetic_rows_second"],
            "synthetic_record_ids": augmentation_result["checks"]["synthetic_record_ids"],
            "real_outer_training_unchanged": augmentation_result["checks"][
                "real_outer_training_unchanged"
            ],
        },
    }


# =========================================
# SECTION 2E: Fold-scoped mandatory conditional CTGAN utilities
# =========================================


CONDITIONAL_CTGAN_STRATEGY = "conditional_all_training_ctgan"
CONDITIONAL_OUTCOME_COLUMN = "OUTCOME_CLASS"


def load_sdv_conditional_ctgan_api() -> Dict[str, Any]:
    api = load_sdv_ctgan_api()
    try:
        sampling_module = importlib.import_module("sdv.sampling")
        condition_class = getattr(sampling_module, "Condition")
        sample_from_conditions = getattr(api["synthesizer_class"], "sample_from_conditions")
        update_column = getattr(api["metadata_class"], "update_column")
        validate_metadata = getattr(api["metadata_class"], "validate")
        validate_data = getattr(api["metadata_class"], "validate_data")
    except AttributeError as error:
        raise RuntimeError(
            "Mandatory conditional CTGAN API is unavailable; no conditional sampling fallback exists."
        ) from error
    if not callable(sample_from_conditions) or not callable(update_column) or not callable(validate_metadata) or not callable(validate_data):
        raise RuntimeError(
            "Mandatory conditional CTGAN API is incompatible; no conditional sampling fallback exists."
        )
    try:
        condition_signature = str(inspect.signature(condition_class))
        sampling_signature = str(inspect.signature(sample_from_conditions))
        update_signature = str(inspect.signature(update_column))
        validate_signature = str(inspect.signature(validate_metadata))
        validate_data_signature = str(inspect.signature(validate_data))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Mandatory conditional CTGAN API cannot be inspected; no conditional sampling fallback exists."
        ) from error
    return {
        **api,
        "condition_class": condition_class,
        "condition_module": condition_class.__module__,
        "condition_signature": condition_signature,
        "sample_from_conditions_signature": sampling_signature,
        "metadata_update_column_signature": update_signature,
        "metadata_validate_signature": validate_signature,
        "metadata_validate_data_signature": validate_data_signature,
    }


def extract_conditional_ctgan_training_input(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
) -> Dict[str, Any]:
    if not isinstance(fused_outer_train, np.ndarray) or fused_outer_train.ndim != 2:
        raise ValueError("Conditional CTGAN fused outer-training input must be a two-dimensional NumPy array.")
    if fused_outer_train.dtype == object or not np.issubdtype(fused_outer_train.dtype, np.number):
        raise ValueError("Conditional CTGAN fused outer-training input must have a numeric non-object dtype.")
    if not np.isfinite(fused_outer_train).all():
        raise ValueError("Conditional CTGAN fused outer-training input must contain only finite values.")
    if not isinstance(train_sample_ids, list) or len(train_sample_ids) != fused_outer_train.shape[0]:
        raise ValueError("Conditional CTGAN outer-training SAMPLE_ID metadata must match fused row count.")
    if len(train_sample_ids) != len(set(train_sample_ids)):
        raise ValueError("Conditional CTGAN outer-training SAMPLE_ID metadata must be unique.")
    feature_names = _validate_ctgan_feature_schema(
        latent_feature_names, latent_feature_name_sha256, fused_outer_train.shape[1]
    )
    if CONDITIONAL_OUTCOME_COLUMN in feature_names:
        raise ValueError("Conditional CTGAN outcome column must not be a latent feature.")
    labels = np.asarray(y_train)
    if labels.ndim != 1 or len(labels) != fused_outer_train.shape[0]:
        raise ValueError("Conditional CTGAN y_train must be one-dimensional and match fused row count.")
    if labels.dtype == object or not np.issubdtype(labels.dtype, np.number):
        raise ValueError("Conditional CTGAN y_train must have a numeric dtype.")
    if not np.isfinite(labels).all():
        raise ValueError("Conditional CTGAN y_train must contain only finite values.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Conditional CTGAN requires exactly two observed training classes.")
    if counts[0] == counts[1]:
        raise ValueError("Conditional CTGAN requires strict class imbalance.")
    minority_index = int(np.argmin(counts))
    majority_index = int(np.argmax(counts))
    minority_label = int(classes[minority_index])
    majority_label = int(classes[majority_index])
    minority_count = int(counts[minority_index])
    majority_count = int(counts[majority_index])
    needed_synthetic_count = majority_count - minority_count
    conditional_training_df = pd.DataFrame(fused_outer_train.copy(), columns=feature_names)
    conditional_training_df[CONDITIONAL_OUTCOME_COLUMN] = labels.astype(int).copy()
    expected_columns = feature_names + [CONDITIONAL_OUTCOME_COLUMN]
    if conditional_training_df.columns.tolist() != expected_columns:
        raise AssertionError("Conditional CTGAN training DataFrame column order is invalid.")
    if not all(pd.api.types.is_numeric_dtype(conditional_training_df[column]) for column in feature_names):
        raise ValueError("Conditional CTGAN latent training columns must be numerical.")
    return {
        "conditional_training_df": conditional_training_df,
        "labels": np.array(labels, dtype=int, copy=True),
        "minority_label": minority_label,
        "majority_label": majority_label,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "needed_synthetic_count": needed_synthetic_count,
        "train_sample_ids": list(train_sample_ids),
        "feature_names": feature_names,
        "feature_name_sha256": latent_feature_name_sha256,
    }


def build_conditional_ctgan_metadata(
    conditional_training_df: pd.DataFrame, feature_names: List[str], api: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    metadata_class = api.get("metadata_class") if isinstance(api, dict) else None
    if metadata_class is None:
        raise ValueError("Conditional CTGAN API descriptor is missing metadata support.")
    expected_columns = feature_names + [CONDITIONAL_OUTCOME_COLUMN]
    if conditional_training_df.columns.tolist() != expected_columns:
        raise ValueError("Conditional CTGAN training DataFrame schema is invalid before metadata construction.")
    try:
        metadata = metadata_class()
        metadata.detect_from_dataframe(data=conditional_training_df)
        metadata.update_column(column_name=CONDITIONAL_OUTCOME_COLUMN, sdtype="categorical")
        metadata.validate()
        metadata.validate_data(data=conditional_training_df)
        metadata_columns = metadata.columns
        outcome_sdtype = metadata_columns[CONDITIONAL_OUTCOME_COLUMN]["sdtype"]
        latent_sdtypes = {feature: metadata_columns[feature]["sdtype"] for feature in feature_names}
    except Exception as error:
        raise RuntimeError(
            "Mandatory conditional CTGAN metadata construction failed; no conditional sampling fallback exists."
        ) from error
    if outcome_sdtype != "categorical":
        raise RuntimeError("Conditional CTGAN outcome metadata must be categorical.")
    if set(metadata_columns) != set(expected_columns) or any(
        sdtype != "numerical" for sdtype in latent_sdtypes.values()
    ):
        raise RuntimeError("Conditional CTGAN metadata schema contains unexpected or non-numerical latent columns.")
    return metadata, {
        "training_columns": expected_columns,
        "outcome_column": CONDITIONAL_OUTCOME_COLUMN,
        "outcome_sdtype": outcome_sdtype,
        "latent_sdtypes": latent_sdtypes,
    }


def _validate_conditional_ctgan_output(
    conditional_samples: Any,
    feature_names: List[str],
    minority_label: int,
    needed_synthetic_count: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not isinstance(conditional_samples, pd.DataFrame):
        raise ValueError("Conditional CTGAN sample output must be a pandas DataFrame.")
    expected_columns = feature_names + [CONDITIONAL_OUTCOME_COLUMN]
    if conditional_samples.shape[0] != needed_synthetic_count:
        raise ValueError("Conditional CTGAN sample row count does not equal the required synthetic count.")
    if conditional_samples.columns.tolist() != expected_columns:
        raise ValueError("Conditional CTGAN sample columns do not exactly match the controlled training schema.")
    outcome_values = conditional_samples[CONDITIONAL_OUTCOME_COLUMN]
    if not pd.api.types.is_numeric_dtype(outcome_values):
        raise ValueError("Conditional CTGAN sampled outcome column must be numeric.")
    if not np.isfinite(outcome_values.to_numpy(dtype=float)).all() or not (
        outcome_values.to_numpy() == minority_label
    ).all():
        raise ValueError("Conditional CTGAN sampled outcome values do not all equal the minority label.")
    latent_samples = conditional_samples.loc[:, feature_names]
    if not all(pd.api.types.is_numeric_dtype(latent_samples[column]) for column in feature_names):
        raise ValueError("Conditional CTGAN sampled latent columns must be numerical.")
    synthetic_matrix = latent_samples.to_numpy(dtype=np.float32, copy=True)
    if synthetic_matrix.shape != (needed_synthetic_count, len(feature_names)):
        raise AssertionError("Conditional CTGAN synthetic shape changed after removing the outcome column.")
    if not np.isfinite(synthetic_matrix).all():
        raise ValueError("Conditional CTGAN sampled latent values contain non-finite values.")
    return synthetic_matrix, {
        "passed": True,
        "outcome_column": CONDITIONAL_OUTCOME_COLUMN,
        "outcome_value": minority_label,
        "outcome_values_all_match_minority": True,
    }


def fit_and_sample_conditional_ctgan(
    conditional_training_df: pd.DataFrame,
    feature_names: List[str],
    minority_label: int,
    needed_synthetic_count: int,
    ctgan_config: Dict[str, Any],
    api: Dict[str, Any],
    global_seed: int,
) -> Dict[str, Any]:
    expected_columns = list(feature_names) + [CONDITIONAL_OUTCOME_COLUMN]
    if not isinstance(conditional_training_df, pd.DataFrame) or conditional_training_df.columns.tolist() != expected_columns:
        raise ValueError("Conditional CTGAN input DataFrame schema is invalid.")
    return _phase10c_run_ctgan_cpu_subprocess(CONDITIONAL_CTGAN_STRATEGY, conditional_training_df.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True), conditional_training_df[CONDITIONAL_OUTCOME_COLUMN].to_numpy(dtype=int, copy=True), list(feature_names), _feature_name_sha256(feature_names), ctgan_config, needed_synthetic_count, minority_label, global_seed)


def build_conditional_ctgan_augmentation(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
    repeat_id: int,
    fold_id: int,
    ctgan_config: Dict[str, Any],
    global_seed: int,
    api: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    real_train_before = np.array(fused_outer_train, copy=True)
    labels_before = np.array(y_train, copy=True)
    extracted = extract_conditional_ctgan_training_input(
        fused_outer_train,
        y_train,
        train_sample_ids,
        latent_feature_names,
        latent_feature_name_sha256,
    )
    fitted = fit_and_sample_conditional_ctgan(
        extracted["conditional_training_df"],
        extracted["feature_names"],
        extracted["minority_label"],
        extracted["needed_synthetic_count"],
        ctgan_config,
        {} if api is None else api,
        global_seed,
    )
    if not np.array_equal(fused_outer_train, real_train_before) or not np.array_equal(y_train, labels_before):
        raise AssertionError("Mandatory conditional CTGAN altered real outer-training features or labels.")
    real_outer_train = np.asarray(fused_outer_train, dtype=np.float32).copy()
    synthetic_minority = fitted["synthetic_matrix"]
    augmented_outer_train = np.vstack([real_outer_train, synthetic_minority]).astype(np.float32)
    y_augmented = np.concatenate(
        [
            extracted["labels"],
            np.full(extracted["needed_synthetic_count"], extracted["minority_label"], dtype=int),
        ]
    )
    is_synthetic = np.concatenate(
        [
            np.zeros(len(real_outer_train), dtype=bool),
            np.ones(extracted["needed_synthetic_count"], dtype=bool),
        ]
    )
    synthetic_record_ids = [
        f"SYNTHETIC:CONDITIONAL:R{repeat_id:03d}:F{fold_id:03d}:CLASS{extracted['minority_label']}:{index:06d}"
        for index in range(extracted["needed_synthetic_count"])
    ]
    if len(synthetic_record_ids) != len(set(synthetic_record_ids)) or set(synthetic_record_ids).intersection(
        extracted["train_sample_ids"]
    ) or any(record_id.startswith("SYNTHETIC:R") for record_id in synthetic_record_ids):
        raise AssertionError("Conditional synthetic record identifiers are invalid or collide with real/minority-only IDs.")
    augmented_record_ids = list(extracted["train_sample_ids"]) + synthetic_record_ids
    if augmented_outer_train.shape != (len(y_augmented), len(extracted["feature_names"])):
        raise AssertionError("Conditional CTGAN augmented matrix shape is invalid.")
    if not np.isfinite(augmented_outer_train).all():
        raise ValueError("Conditional CTGAN augmented matrix contains non-finite values.")
    original_class_counts = _class_counts(extracted["labels"])
    augmented_class_counts = _class_counts(y_augmented)
    if augmented_class_counts[str(extracted["minority_label"])] != extracted["majority_count"]:
        raise AssertionError("Conditional CTGAN augmentation did not balance the minority class to the majority count.")
    return {
        "strategy": CONDITIONAL_CTGAN_STRATEGY,
        "repeat_id": int(repeat_id),
        "fold_id": int(fold_id),
        "real_outer_train": real_outer_train,
        "synthetic_minority": synthetic_minority,
        "augmented_outer_train": augmented_outer_train,
        "y_augmented": y_augmented,
        "is_synthetic": is_synthetic,
        "real_sample_ids": list(extracted["train_sample_ids"]),
        "synthetic_record_ids": synthetic_record_ids,
        "augmented_record_ids": augmented_record_ids,
        "minority_label": extracted["minority_label"],
        "majority_label": extracted["majority_label"],
        "minority_count": extracted["minority_count"],
        "majority_count": extracted["majority_count"],
        "needed_synthetic_count": extracted["needed_synthetic_count"],
        "generated_synthetic_count": int(len(synthetic_minority)),
        "original_class_counts": original_class_counts,
        "augmented_class_counts": augmented_class_counts,
        "feature_names": list(extracted["feature_names"]),
        "feature_name_sha256": extracted["feature_name_sha256"],
        "training_table_shape": list(extracted["conditional_training_df"].shape),
        "fit_row_count": int(len(extracted["conditional_training_df"])),
        "fit_class_counts": original_class_counts,
        "metadata_schema": fitted["metadata_schema"],
        "condition": fitted["condition"],
        "ctgan_api": fitted["ctgan_api"],
        "ctgan_configuration": fitted["constructor_configuration"],
        "seed_evidence": fitted["seed_evidence"],
        "loss_values": fitted["loss_values"],
        "ctgan_execution": fitted["execution_evidence"],
        "checks": {
            "conditional_target_values": fitted["condition_check"],
            "synthetic_schema": {"passed": True},
            "synthetic_finite": {"passed": True},
            "real_rows_first": {"passed": True},
            "synthetic_rows_second": {"passed": True},
            "synthetic_record_ids": {"passed": True},
            "fallback_exists": False,
            "outer_test_supplied_to_ctgan": False,
            "real_outer_training_unchanged": {"passed": True},
        },
    }


def build_ctgan_strategy_contract(
    strategy: str, fusion_result: Dict[str, Any], augmentation_result: Dict[str, Any]
) -> Dict[str, Any]:
    if strategy not in {MINORITY_CTGAN_STRATEGY, CONDITIONAL_CTGAN_STRATEGY}:
        raise ValueError(f"Unsupported CTGAN strategy: {strategy}")
    if augmentation_result.get("strategy", strategy) != strategy:
        raise ValueError("CTGAN augmentation strategy marker does not match its comparison contract.")
    if strategy == MINORITY_CTGAN_STRATEGY:
        fit_row_count = augmentation_result["minority_count"]
        fit_class_counts = {str(augmentation_result["minority_label"]): augmentation_result["minority_count"]}
        training_table_shape = [fit_row_count, len(augmentation_result["feature_names"])]
        metadata_schema = {"controlled_outcome_column": None, "latent_columns_only": True}
        sampling_method = "sample"
    else:
        fit_row_count = augmentation_result["fit_row_count"]
        fit_class_counts = augmentation_result["fit_class_counts"]
        training_table_shape = augmentation_result["training_table_shape"]
        metadata_schema = augmentation_result["metadata_schema"]
        sampling_method = "sample_from_conditions"
    configuration = augmentation_result["ctgan_configuration"]
    return {
        "strategy": strategy,
        "repeat_id": augmentation_result["repeat_id"],
        "fold_id": augmentation_result["fold_id"],
        "outer_train_sample_ids_canonical_json_sha256": fusion_result[
            "train_sample_ids_canonical_json_sha256"
        ],
        "latent_feature_name_sha256": fusion_result["latent_feature_name_sha256"],
        "real_fused_training_shape": list(fusion_result["fused_outer_train"].shape),
        "real_fused_training_dtype": str(fusion_result["fused_outer_train"].dtype),
        "required_synthetic_count": augmentation_result["needed_synthetic_count"],
        "ctgan_smoke_configuration": {
            key: configuration[key] for key in ["epochs", "batch_size", "pac", "verbose"]
        },
        "fit_row_count": fit_row_count,
        "fit_class_counts": fit_class_counts,
        "training_table_shape": training_table_shape,
        "metadata_schema": metadata_schema,
        "sampling_method": sampling_method,
        "requested_synthetic_rows": augmentation_result["needed_synthetic_count"],
        "generated_synthetic_rows": augmentation_result["generated_synthetic_count"],
        "synthetic_latent_shape": list(augmentation_result["synthetic_minority"].shape),
        "augmented_class_counts": augmentation_result["augmented_class_counts"],
        "sdv_version": augmentation_result["ctgan_api"].get("sdv_version"),
        "ctgan_version": augmentation_result["ctgan_api"].get("ctgan_version"),
        "seed_control_limitation": augmentation_result["seed_evidence"],
        "test_isolation": {
            "outer_test_supplied_to_ctgan": False,
            "real_outer_training_unchanged": augmentation_result["checks"][
                "real_outer_training_unchanged"
            ],
        },
        "no_fallback": not augmentation_result["checks"]["fallback_exists"],
    }


def build_conditional_ctgan_audit(
    manifest_validation: Dict[str, Any],
    materialized_fold: Dict[str, Any],
    fusion_result: Dict[str, Any],
    augmentation_result: Dict[str, Any],
    outer_test_unchanged: bool,
) -> Dict[str, Any]:
    strategy_contract = build_ctgan_strategy_contract(
        CONDITIONAL_CTGAN_STRATEGY, fusion_result, augmentation_result
    )
    return {
        "action": "audited_conditional_ctgan",
        "strategy": CONDITIONAL_CTGAN_STRATEGY,
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": augmentation_result["repeat_id"],
            "fold_id": augmentation_result["fold_id"],
            "outer_train_sample_count": len(augmentation_result["real_sample_ids"]),
            "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(
                augmentation_result["real_sample_ids"]
            ),
        },
        "ctgan": {
            **augmentation_result["ctgan_api"],
            "execution": augmentation_result["ctgan_execution"],
            "constructor_configuration": augmentation_result["ctgan_configuration"],
            "seed_control": augmentation_result["seed_evidence"],
            "fit_scope": "all_outer_training_latents_with_controlled_outcome_class",
            "fit_row_count": augmentation_result["fit_row_count"],
            "fit_class_counts": augmentation_result["fit_class_counts"],
            "training_table_shape": augmentation_result["training_table_shape"],
            "training_table_columns": augmentation_result["metadata_schema"]["training_columns"],
            "metadata_schema": augmentation_result["metadata_schema"],
            "condition": augmentation_result["condition"],
            "conditional_sampling_method": "sample_from_conditions",
            "minority_label": augmentation_result["minority_label"],
            "majority_label": augmentation_result["majority_label"],
            "minority_count": augmentation_result["minority_count"],
            "majority_count": augmentation_result["majority_count"],
            "latent_feature_name_sha256": augmentation_result["feature_name_sha256"],
            "needed_synthetic_count": augmentation_result["needed_synthetic_count"],
            "generated_synthetic_count": augmentation_result["generated_synthetic_count"],
            "returned_target_value_validation": augmentation_result["checks"][
                "conditional_target_values"
            ],
            "synthetic_latent_shape": list(augmentation_result["synthetic_minority"].shape),
            "synthetic_latent_dtype": str(augmentation_result["synthetic_minority"].dtype),
            "schema_check": augmentation_result["checks"]["synthetic_schema"],
            "finite_check": augmentation_result["checks"]["synthetic_finite"],
            "fallback_exists": False,
            "outer_test_supplied_to_ctgan": False,
            "outer_test_features_and_labels_unchanged": bool(outer_test_unchanged),
            "loss_summary": augmentation_result["loss_values"].get("summary", {"available": False}),
        },
        "augmentation": {
            "original_class_counts": augmentation_result["original_class_counts"],
            "augmented_class_counts": augmentation_result["augmented_class_counts"],
            "real_rows_first": augmentation_result["checks"]["real_rows_first"],
            "synthetic_rows_second": augmentation_result["checks"]["synthetic_rows_second"],
            "synthetic_record_ids": augmentation_result["checks"]["synthetic_record_ids"],
            "real_outer_training_unchanged": augmentation_result["checks"][
                "real_outer_training_unchanged"
            ],
        },
        "comparison_contract": strategy_contract,
    }


# =========================================
# SECTION 2F: Fold-scoped CTGAN quality diagnostics
# =========================================


CTGAN_QUALITY_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _quality_numeric_summary(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Quality summary requires one or more finite numeric values.")
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "quantiles": {
            f"{quantile:.2f}": float(np.quantile(array, quantile))
            for quantile in CTGAN_QUALITY_QUANTILES
        },
    }


def _validate_ctgan_quality_matrix(name: str, matrix: Any, expected_width: int) -> np.ndarray:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional NumPy matrix.")
    if matrix.dtype != np.float32:
        raise ValueError(f"{name} must use float32 dtype for exact-copy diagnostics.")
    if matrix.shape[1] != expected_width:
        raise ValueError(f"{name} width does not match the latent feature schema.")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two rows for Phase 7A diagnostics.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _extract_quality_comparison_context(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
    minority_only_result: Dict[str, Any],
    conditional_result: Dict[str, Any],
) -> Dict[str, Any]:
    expected_width = len(latent_feature_names)
    real_train = _validate_ctgan_quality_matrix(
        "Real fused outer-training matrix", fused_outer_train, expected_width
    )
    if not isinstance(train_sample_ids, list) or len(train_sample_ids) != len(real_train):
        raise ValueError("CTGAN quality train SAMPLE_ID metadata does not match real training rows.")
    if _feature_name_sha256(latent_feature_names) != latent_feature_name_sha256:
        raise ValueError("CTGAN quality latent feature-name SHA256 does not match supplied feature names.")
    labels = np.asarray(y_train)
    if labels.ndim != 1 or len(labels) != len(real_train):
        raise ValueError("CTGAN quality y_train must match real training rows.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        raise ValueError("CTGAN quality requires exactly two imbalanced training classes.")
    minority_label = int(classes[np.argmin(counts)])
    majority_count = int(np.max(counts))
    minority_count = int(np.min(counts))
    needed_synthetic_count = majority_count - minority_count
    real_minority = real_train[labels == minority_label].copy()
    if len(real_minority) != minority_count:
        raise AssertionError("CTGAN quality real-minority extraction is invalid.")

    expected_train_hash = _sample_id_list_sha256(train_sample_ids)
    expected_configuration = None
    validated_results = {}
    for strategy, result in [
        (MINORITY_CTGAN_STRATEGY, minority_only_result),
        (CONDITIONAL_CTGAN_STRATEGY, conditional_result),
    ]:
        if not isinstance(result, dict) or result.get("strategy") != strategy:
            raise ValueError(f"CTGAN quality result does not identify strategy {strategy}.")
        if result.get("repeat_id") != minority_only_result.get("repeat_id") or result.get(
            "fold_id"
        ) != minority_only_result.get("fold_id"):
            raise ValueError("CTGAN quality strategy repeat/fold identity differs.")
        if result.get("feature_names") != latent_feature_names or result.get(
            "feature_name_sha256"
        ) != latent_feature_name_sha256:
            raise ValueError("CTGAN quality strategy latent feature schema differs.")
        if result.get("minority_label") != minority_label:
            raise ValueError("CTGAN quality strategy minority label differs from real training labels.")
        if result.get("needed_synthetic_count") != needed_synthetic_count or result.get(
            "generated_synthetic_count"
        ) != needed_synthetic_count:
            raise ValueError("CTGAN quality strategy synthetic row count differs from the required count.")
        if result.get("real_sample_ids") != train_sample_ids or _sample_id_list_sha256(
            result.get("real_sample_ids", [])
        ) != expected_train_hash:
            raise ValueError("CTGAN quality strategy outer-training SAMPLE_ID identity differs.")
        strategy_real_train = _validate_ctgan_quality_matrix(
            f"{strategy} real outer-training matrix", result.get("real_outer_train"), expected_width
        )
        if not np.array_equal(strategy_real_train, real_train):
            raise ValueError("CTGAN quality strategy real outer-training matrix differs.")
        synthetic = _validate_ctgan_quality_matrix(
            f"{strategy} synthetic minority matrix", result.get("synthetic_minority"), expected_width
        )
        if len(synthetic) != needed_synthetic_count:
            raise ValueError("CTGAN quality synthetic matrix row count differs from required count.")
        configuration = result.get("ctgan_configuration")
        if not isinstance(configuration, dict):
            raise ValueError("CTGAN quality strategy configuration is missing.")
        comparable_configuration = {
            key: configuration.get(key) for key in ["epochs", "batch_size", "pac", "verbose"]
        }
        if any(value is None for value in comparable_configuration.values()):
            raise ValueError("CTGAN quality strategy configuration is incomplete.")
        if expected_configuration is None:
            expected_configuration = comparable_configuration
        elif comparable_configuration != expected_configuration:
            raise ValueError("CTGAN quality strategy configuration differs.")
        validated_results[strategy] = {"result": result, "synthetic": synthetic}
    if minority_only_result.get("repeat_id") != conditional_result.get("repeat_id") or minority_only_result.get(
        "fold_id"
    ) != conditional_result.get("fold_id"):
        raise ValueError("CTGAN quality strategies do not share the same repeat/fold.")
    return {
        "real_train": real_train,
        "real_minority": real_minority,
        "minority_label": minority_label,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "needed_synthetic_count": needed_synthetic_count,
        "train_sample_ids_hash": expected_train_hash,
        "feature_names": list(latent_feature_names),
        "feature_name_sha256": latent_feature_name_sha256,
        "configuration": expected_configuration,
        "repeat_id": minority_only_result["repeat_id"],
        "fold_id": minority_only_result["fold_id"],
        "strategies": validated_results,
    }


def _exact_row_membership_count(rows: np.ndarray, reference_rows: np.ndarray) -> int:
    reference = {row.tobytes() for row in np.ascontiguousarray(reference_rows)}
    return int(sum(row.tobytes() in reference for row in np.ascontiguousarray(rows)))


def _compute_duplicate_diversity_metrics(real_minority: np.ndarray, synthetic: np.ndarray) -> Dict[str, Any]:
    unique_synthetic_count = int(len(np.unique(synthetic, axis=0)))
    return {
        "exact_duplicate_synthetic_row_count": int(len(synthetic) - unique_synthetic_count),
        "synthetic_rows_exactly_matching_real_minority_count": _exact_row_membership_count(
            synthetic, real_minority
        ),
        "unique_synthetic_row_count": unique_synthetic_count,
        "unique_synthetic_row_ratio": float(unique_synthetic_count / len(synthetic)),
    }


def _compute_distribution_metrics(
    real_minority: np.ndarray, synthetic: np.ndarray, feature_names: List[str]
) -> Dict[str, Any]:
    per_feature = {}
    mean_abs_differences = []
    std_abs_differences = []
    wasserstein_distances = []
    ks_statistics = []
    lower_violations = []
    upper_violations = []
    for index, feature_name in enumerate(feature_names):
        real_values = real_minority[:, index]
        synthetic_values = synthetic[:, index]
        real_mean = float(np.mean(real_values))
        synthetic_mean = float(np.mean(synthetic_values))
        real_std = float(np.std(real_values, ddof=1))
        synthetic_std = float(np.std(synthetic_values, ddof=1))
        mean_difference = synthetic_mean - real_mean
        std_difference = synthetic_std - real_std
        lower_count = int(np.sum(synthetic_values < np.min(real_values)))
        upper_count = int(np.sum(synthetic_values > np.max(real_values)))
        wasserstein = float(wasserstein_distance(real_values, synthetic_values))
        ks_statistic = float(ks_2samp(real_values, synthetic_values).statistic)
        per_feature[feature_name] = {
            "real_mean": real_mean,
            "synthetic_mean": synthetic_mean,
            "mean_difference": mean_difference,
            "abs_mean_difference": abs(mean_difference),
            "real_std_ddof1": real_std,
            "synthetic_std_ddof1": synthetic_std,
            "std_difference": std_difference,
            "abs_std_difference": abs(std_difference),
            "wasserstein_distance": wasserstein,
            "ks_statistic": ks_statistic,
            "real_min": float(np.min(real_values)),
            "real_max": float(np.max(real_values)),
            "synthetic_min": float(np.min(synthetic_values)),
            "synthetic_max": float(np.max(synthetic_values)),
            "lower_range_violation_count": lower_count,
            "upper_range_violation_count": upper_count,
        }
        mean_abs_differences.append(abs(mean_difference))
        std_abs_differences.append(abs(std_difference))
        wasserstein_distances.append(wasserstein)
        ks_statistics.append(ks_statistic)
        lower_violations.append(lower_count)
        upper_violations.append(upper_count)
    return {
        "per_feature": per_feature,
        "aggregates": {
            "abs_mean_difference": _quality_numeric_summary(np.asarray(mean_abs_differences)),
            "abs_std_difference_ddof1": _quality_numeric_summary(np.asarray(std_abs_differences)),
            "wasserstein_distance": _quality_numeric_summary(np.asarray(wasserstein_distances)),
            "ks_statistic": _quality_numeric_summary(np.asarray(ks_statistics)),
        },
        "range_violations": {
            "per_feature": {
                feature_name: {
                    "lower": lower_violations[index],
                    "upper": upper_violations[index],
                    "total": lower_violations[index] + upper_violations[index],
                }
                for index, feature_name in enumerate(feature_names)
            },
            "total_lower": int(sum(lower_violations)),
            "total_upper": int(sum(upper_violations)),
            "total": int(sum(lower_violations) + sum(upper_violations)),
        },
    }


def _compute_correlation_preservation(
    real_minority: np.ndarray, synthetic: np.ndarray, feature_names: List[str]
) -> Dict[str, Any]:
    real_scales = np.std(real_minority, axis=0, ddof=1)
    synthetic_scales = np.std(synthetic, axis=0, ddof=1)
    real_zero_features = [
        feature_names[index]
        for index, scale in enumerate(real_scales)
        if not np.isfinite(scale) or np.isclose(scale, 0.0)
    ]
    synthetic_zero_features = [
        feature_names[index]
        for index, scale in enumerate(synthetic_scales)
        if not np.isfinite(scale) or np.isclose(scale, 0.0)
    ]
    excluded_features = sorted(set(real_zero_features).union(synthetic_zero_features))
    valid_indices = [index for index, feature_name in enumerate(feature_names) if feature_name not in excluded_features]
    if len(valid_indices) < 2:
        return {
            "valid_pair_count": 0,
            "excluded_zero_variance_features": excluded_features,
            "excluded_pair_count": int(len(feature_names) * (len(feature_names) - 1) // 2),
            "mean_abs_difference": None,
            "median_abs_difference": None,
            "max_abs_difference": None,
        }
    real_correlation = np.corrcoef(real_minority[:, valid_indices], rowvar=False)
    synthetic_correlation = np.corrcoef(synthetic[:, valid_indices], rowvar=False)
    upper_indices = np.triu_indices(len(valid_indices), k=1)
    differences = np.abs(real_correlation[upper_indices] - synthetic_correlation[upper_indices])
    valid_differences = differences[np.isfinite(differences)]
    total_pairs = len(feature_names) * (len(feature_names) - 1) // 2
    return {
        "valid_pair_count": int(len(valid_differences)),
        "excluded_zero_variance_features": excluded_features,
        "excluded_pair_count": int(total_pairs - len(valid_differences)),
        "mean_abs_difference": float(np.mean(valid_differences)) if len(valid_differences) else None,
        "median_abs_difference": float(np.median(valid_differences)) if len(valid_differences) else None,
        "max_abs_difference": float(np.max(valid_differences)) if len(valid_differences) else None,
    }


def _nearest_neighbor_distributions(real_minority: np.ndarray, synthetic: np.ndarray) -> Dict[str, Any]:
    synthetic_to_real = np.min(pairwise_distances(synthetic, real_minority, metric="euclidean"), axis=1)
    real_to_synthetic = np.min(pairwise_distances(real_minority, synthetic, metric="euclidean"), axis=1)
    real_to_real = pairwise_distances(real_minority, real_minority, metric="euclidean")
    np.fill_diagonal(real_to_real, np.inf)
    real_internal = np.min(real_to_real, axis=1)
    return {
        "synthetic_to_real": _quality_numeric_summary(synthetic_to_real),
        "real_to_synthetic": _quality_numeric_summary(real_to_synthetic),
        "real_internal_self_excluded": _quality_numeric_summary(real_internal),
    }


def _compute_nearest_neighbor_metrics(
    real_minority: np.ndarray, synthetic: np.ndarray, feature_names: List[str]
) -> Dict[str, Any]:
    real_means = np.mean(real_minority, axis=0)
    real_scales = np.std(real_minority, axis=0, ddof=1)
    zero_scale_indices = np.flatnonzero(~np.isfinite(real_scales) | np.isclose(real_scales, 0.0))
    safe_scales = real_scales.copy()
    safe_scales[zero_scale_indices] = 1.0
    standardized_real = (real_minority - real_means) / safe_scales
    standardized_synthetic = (synthetic - real_means) / safe_scales
    return {
        "raw_euclidean": {
            "metric": "euclidean",
            **_nearest_neighbor_distributions(real_minority, synthetic),
        },
        "standardized_euclidean": {
            "metric": "euclidean",
            "standardization_reference": "real_minority_only",
            "scale_ddof": 1,
            "zero_scale_feature_names": [feature_names[index] for index in zero_scale_indices],
            **_nearest_neighbor_distributions(standardized_real, standardized_synthetic),
        },
    }


def evaluate_ctgan_synthetic_quality(
    strategy_result: Dict[str, Any],
    real_minority: np.ndarray,
    feature_names: List[str],
    feature_name_sha256: str,
) -> Dict[str, Any]:
    synthetic = _validate_ctgan_quality_matrix(
        f"{strategy_result.get('strategy', 'unknown')} synthetic matrix",
        strategy_result.get("synthetic_minority"),
        len(feature_names),
    )
    if strategy_result.get("feature_names") != feature_names or strategy_result.get(
        "feature_name_sha256"
    ) != feature_name_sha256:
        raise ValueError("CTGAN quality strategy schema differs from the comparison schema.")
    distribution_metrics = _compute_distribution_metrics(real_minority, synthetic, feature_names)
    return {
        "strategy": strategy_result["strategy"],
        "repeat_id": strategy_result["repeat_id"],
        "fold_id": strategy_result["fold_id"],
        "reference_real_minority_shape": list(real_minority.shape),
        "synthetic_shape": list(synthetic.shape),
        "latent_feature_name_sha256": feature_name_sha256,
        "ctgan_configuration": {
            key: strategy_result["ctgan_configuration"][key]
            for key in ["epochs", "batch_size", "pac", "verbose"]
        },
        "duplicate_memorization_evidence": _compute_duplicate_diversity_metrics(
            real_minority, synthetic
        ),
        "distribution_metrics": distribution_metrics,
        "correlation_preservation": _compute_correlation_preservation(
            real_minority, synthetic, feature_names
        ),
        "nearest_neighbor": _compute_nearest_neighbor_metrics(
            real_minority, synthetic, feature_names
        ),
        "range_checks": distribution_metrics["range_violations"],
        "test_isolation": {
            "outer_test_accepted_by_quality_function": False,
            "real_inputs_unchanged": True,
        },
    }


def compare_ctgan_synthetic_quality(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
    minority_only_result: Dict[str, Any],
    conditional_result: Dict[str, Any],
) -> Dict[str, Any]:
    real_train_before = np.array(fused_outer_train, copy=True)
    labels_before = np.array(y_train, copy=True)
    context = _extract_quality_comparison_context(
        fused_outer_train,
        y_train,
        train_sample_ids,
        latent_feature_names,
        latent_feature_name_sha256,
        minority_only_result,
        conditional_result,
    )
    quality_results = {
        strategy: evaluate_ctgan_synthetic_quality(
            context["strategies"][strategy]["result"],
            context["real_minority"],
            context["feature_names"],
            context["feature_name_sha256"],
        )
        for strategy in [MINORITY_CTGAN_STRATEGY, CONDITIONAL_CTGAN_STRATEGY]
    }
    if not np.array_equal(fused_outer_train, real_train_before) or not np.array_equal(y_train, labels_before):
        raise AssertionError("CTGAN quality diagnostics altered real training inputs.")
    return {
        "repeat_id": context["repeat_id"],
        "fold_id": context["fold_id"],
        "real_minority_shape": list(context["real_minority"].shape),
        "minority_label": context["minority_label"],
        "minority_count": context["minority_count"],
        "majority_count": context["majority_count"],
        "needed_synthetic_count": context["needed_synthetic_count"],
        "train_sample_ids_canonical_json_sha256": context["train_sample_ids_hash"],
        "latent_feature_name_sha256": context["feature_name_sha256"],
        "fair_comparison_identity": {
            "passed": True,
            "real_training_matrix_identical": True,
            "outer_training_sample_id_hash_identical": True,
            "latent_feature_schema_identical": True,
            "minority_label_identical": True,
            "needed_generated_count_identical": True,
            "ctgan_configuration": context["configuration"],
        },
        "quality_results": quality_results,
        "interpretation_guard": {
            "classifier_metrics_computed": False,
            "quality_score_computed": False,
            "strategy_ranked": False,
            "winner_declared": False,
        },
    }


def build_ctgan_quality_audit(
    manifest_validation: Dict[str, Any], quality_result: Dict[str, Any], outer_test_unchanged: bool
) -> Dict[str, Any]:
    return {
        "action": "audited_ctgan_quality",
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": quality_result["repeat_id"],
            "fold_id": quality_result["fold_id"],
            "real_minority_shape": quality_result["real_minority_shape"],
            "minority_label": quality_result["minority_label"],
            "train_sample_ids_canonical_json_sha256": quality_result[
                "train_sample_ids_canonical_json_sha256"
            ],
        },
        "comparison_identity": quality_result["fair_comparison_identity"],
        "strategies": quality_result["quality_results"],
        "test_isolation": {
            "outer_test_features_and_labels_unchanged": bool(outer_test_unchanged),
            "outer_test_accepted_by_quality_functions": False,
        },
        "interpretation_guard": quality_result["interpretation_guard"],
    }


# =========================================
# SECTION 2G: Fold-scoped Logistic Regression smoke evaluation
# =========================================


LOGISTIC_SMOKE_CONFIGURATION = {
    "solver": "liblinear",
    "penalty": "l2",
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": None,
}
LOGISTIC_SMOKE_THRESHOLD = 0.5
LOGISTIC_SMOKE_COMPARISON_SCOPE = "augmentation_only_unweighted_logistic_smoke"
LOGISTIC_FORBIDDEN_FEATURE_NAMES = {
    "sample_id",
    "class",
    "outcome_class",
    "target",
    "label",
    "y",
    "provenance",
    "is_synthetic",
    "record_id",
    "synthetic_record_id",
}


def _array_sha256(array: np.ndarray) -> str:
    if not isinstance(array, np.ndarray):
        raise ValueError("Array hashing requires a NumPy array.")
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)}))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_logistic_feature_matrix(name: str, matrix: Any, expected_width: int) -> np.ndarray:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional NumPy matrix.")
    if matrix.dtype != np.float32:
        raise ValueError(f"{name} must use float32 dtype.")
    if matrix.shape[0] < 1 or matrix.shape[1] != expected_width:
        raise ValueError(f"{name} shape does not match the Phase 5 latent feature schema.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _validate_logistic_binary_labels(
    name: str, labels: Any, expected_rows: int, require_both_classes: bool
) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or len(array) != expected_rows:
        raise ValueError(f"{name} must be a one-dimensional vector aligned to its feature matrix.")
    if not np.issubdtype(array.dtype, np.integer) or not set(array.tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must contain only binary integer labels 0 and 1.")
    if require_both_classes and set(array.tolist()) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes 0 and 1 for classifier fitting.")
    return array


def _validate_logistic_feature_schema(feature_names: Any, feature_name_sha256: Any) -> List[str]:
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Logistic smoke evaluation requires a non-empty latent feature-name list.")
    if any(not isinstance(name, str) for name in feature_names):
        raise ValueError("Logistic smoke feature names must be strings.")
    if _feature_name_sha256(feature_names) != feature_name_sha256:
        raise ValueError("Logistic smoke latent feature-name SHA256 does not match the supplied schema.")
    forbidden = [name for name in feature_names if name.casefold() in LOGISTIC_FORBIDDEN_FEATURE_NAMES]
    if forbidden:
        raise ValueError(f"Logistic smoke model features contain forbidden columns: {forbidden}.")
    return list(feature_names)


def _validate_logistic_record_ids(name: str, record_ids: Any, expected_rows: int) -> List[str]:
    if not isinstance(record_ids, list) or len(record_ids) != expected_rows:
        raise ValueError(f"{name} must be an ordered list aligned to its feature matrix.")
    if any(not isinstance(record_id, str) for record_id in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError(f"{name} must contain unique string identifiers.")
    return list(record_ids)


def _build_real_only_logistic_variant(fusion_result: Dict[str, Any]) -> Dict[str, Any]:
    feature_names = _validate_logistic_feature_schema(
        fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256")
    )
    features = _validate_logistic_feature_matrix(
        "real_only outer-training matrix", fusion_result.get("fused_outer_train"), len(feature_names)
    )
    labels = _validate_logistic_binary_labels("real_only outer-training labels", fusion_result.get("y_train"), len(features), True)
    record_ids = _validate_logistic_record_ids(
        "real_only outer-training SAMPLE_ID values", fusion_result.get("train_sample_ids"), len(features)
    )
    if fusion_result.get("train_sample_ids_canonical_json_sha256") != _sample_id_list_sha256(record_ids):
        raise ValueError("real_only outer-training SAMPLE_ID hash is invalid.")
    return {
        "variant_name": "real_only",
        "provenance": "phase5_fused_outer_train",
        "repeat_id": fusion_result.get("repeat_id"),
        "fold_id": fusion_result.get("fold_id"),
        "features": features,
        "labels": labels,
        "record_ids": record_ids,
        "is_synthetic": np.zeros(len(features), dtype=bool),
        "synthetic_row_count": 0,
        "training_shape": list(features.shape),
        "training_class_counts": _class_counts(labels),
        "feature_names": feature_names,
        "feature_name_sha256": fusion_result["latent_feature_name_sha256"],
    }


def _build_augmented_logistic_variant(
    strategy_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
    expected_strategy: str,
    outer_test_features: np.ndarray,
    test_sample_ids: List[str],
) -> Dict[str, Any]:
    feature_names = _validate_logistic_feature_schema(
        fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256")
    )
    real_features = _validate_logistic_feature_matrix(
        "Phase 5 real outer-training matrix", fusion_result.get("fused_outer_train"), len(feature_names)
    )
    real_labels = _validate_logistic_binary_labels("Phase 5 real outer-training labels", fusion_result.get("y_train"), len(real_features), True)
    real_ids = _validate_logistic_record_ids("Phase 5 outer-training SAMPLE_ID values", fusion_result.get("train_sample_ids"), len(real_features))
    if not isinstance(strategy_result, dict) or strategy_result.get("strategy") != expected_strategy:
        raise ValueError(f"Logistic smoke variant does not identify strategy {expected_strategy}.")
    if strategy_result.get("repeat_id") != fusion_result.get("repeat_id") or strategy_result.get("fold_id") != fusion_result.get("fold_id"):
        raise ValueError("Logistic smoke augmentation repeat/fold differs from Phase 5.")
    if strategy_result.get("feature_names") != feature_names or strategy_result.get("feature_name_sha256") != fusion_result.get("latent_feature_name_sha256"):
        raise ValueError("Logistic smoke augmentation latent feature schema differs from Phase 5.")
    strategy_real = _validate_logistic_feature_matrix(
        f"{expected_strategy} real outer-training matrix", strategy_result.get("real_outer_train"), len(feature_names)
    )
    if not np.array_equal(strategy_real, real_features) or strategy_result.get("real_sample_ids") != real_ids:
        raise ValueError("Logistic smoke augmentation real-training provenance differs from Phase 5.")
    synthetic = _validate_logistic_feature_matrix(
        f"{expected_strategy} synthetic matrix", strategy_result.get("synthetic_minority"), len(feature_names)
    )
    generated_count = strategy_result.get("generated_synthetic_count")
    needed_count = strategy_result.get("needed_synthetic_count")
    if generated_count != len(synthetic) or needed_count != len(synthetic):
        raise ValueError("Logistic smoke augmentation synthetic row-count provenance is invalid.")
    features = _validate_logistic_feature_matrix(
        f"{expected_strategy} augmented outer-training matrix", strategy_result.get("augmented_outer_train"), len(feature_names)
    )
    labels = _validate_logistic_binary_labels(
        f"{expected_strategy} augmented outer-training labels", strategy_result.get("y_augmented"), len(features), True
    )
    record_ids = _validate_logistic_record_ids(
        f"{expected_strategy} augmented record identifiers", strategy_result.get("augmented_record_ids"), len(features)
    )
    marker = np.asarray(strategy_result.get("is_synthetic"))
    if marker.dtype != bool or marker.ndim != 1 or len(marker) != len(features):
        raise ValueError("Logistic smoke augmentation synthetic-row marker is invalid.")
    real_count = len(real_features)
    if (
        features.shape[0] != real_count + len(synthetic)
        or not np.array_equal(features[:real_count], real_features)
        or not np.array_equal(features[real_count:], synthetic)
        or not np.array_equal(labels[:real_count], real_labels)
        or not np.array_equal(marker[:real_count], np.zeros(real_count, dtype=bool))
        or not np.array_equal(marker[real_count:], np.ones(len(synthetic), dtype=bool))
        or record_ids[:real_count] != real_ids
    ):
        raise ValueError("Logistic smoke augmentation rows, labels, identifiers, or provenance are invalid.")
    synthetic_ids = _validate_logistic_record_ids(
        f"{expected_strategy} synthetic record identifiers", strategy_result.get("synthetic_record_ids"), len(synthetic)
    )
    if record_ids[real_count:] != synthetic_ids or set(synthetic_ids).intersection(test_sample_ids):
        raise ValueError("Synthetic records may not appear in outer-test evaluation.")
    if _exact_row_membership_count(synthetic, outer_test_features):
        raise ValueError("Synthetic rows may not appear in the outer-test matrix.")
    return {
        "variant_name": expected_strategy,
        "provenance": f"phase6_{expected_strategy}_augmented_outer_train",
        "repeat_id": fusion_result["repeat_id"],
        "fold_id": fusion_result["fold_id"],
        "features": features,
        "labels": labels,
        "record_ids": record_ids,
        "is_synthetic": marker.copy(),
        "synthetic_row_count": int(np.sum(marker)),
        "training_shape": list(features.shape),
        "training_class_counts": _class_counts(labels),
        "feature_names": feature_names,
        "feature_name_sha256": fusion_result["latent_feature_name_sha256"],
    }


def build_fold_logistic_training_variants(
    fusion_result: Dict[str, Any], minority_only_result: Dict[str, Any], conditional_result: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    feature_names = _validate_logistic_feature_schema(
        fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256")
    )
    outer_test_features = _validate_logistic_feature_matrix(
        "Phase 5 outer-test matrix", fusion_result.get("fused_outer_test"), len(feature_names)
    )
    test_sample_ids = _validate_logistic_record_ids(
        "Phase 5 outer-test SAMPLE_ID values", fusion_result.get("test_sample_ids"), len(outer_test_features)
    )
    if fusion_result.get("test_sample_ids_canonical_json_sha256") != _sample_id_list_sha256(test_sample_ids):
        raise ValueError("Phase 5 outer-test SAMPLE_ID hash is invalid.")
    return {
        "real_only": _build_real_only_logistic_variant(fusion_result),
        MINORITY_CTGAN_STRATEGY: _build_augmented_logistic_variant(
            minority_only_result, fusion_result, MINORITY_CTGAN_STRATEGY, outer_test_features, test_sample_ids
        ),
        CONDITIONAL_CTGAN_STRATEGY: _build_augmented_logistic_variant(
            conditional_result, fusion_result, CONDITIONAL_CTGAN_STRATEGY, outer_test_features, test_sample_ids
        ),
    }


def _validate_logistic_estimator_classes(estimator: Any) -> np.ndarray:
    classes = np.asarray(getattr(estimator, "classes_", None))
    if classes.ndim != 1 or len(classes) != 2 or not np.array_equal(np.sort(classes), np.array([0, 1])):
        raise ValueError("Logistic smoke estimator.classes_ must contain exactly binary classes 0 and 1.")
    return classes


def fit_fold_logistic_smoke_classifier(
    training_variant: Dict[str, Any],
    classifier_config: Dict[str, Any] = LOGISTIC_SMOKE_CONFIGURATION,
    classifier_factory: Any = LogisticRegression,
) -> Dict[str, Any]:
    if not isinstance(training_variant, dict):
        raise ValueError("Logistic smoke training variant must be a dictionary.")
    expected_config = dict(LOGISTIC_SMOKE_CONFIGURATION)
    if dict(classifier_config) != expected_config:
        raise ValueError("Logistic smoke classifier configuration must equal the fixed approved configuration.")
    features = _validate_logistic_feature_matrix(
        f"{training_variant.get('variant_name', 'unknown')} training matrix",
        training_variant.get("features"),
        len(training_variant.get("feature_names", [])),
    )
    labels = _validate_logistic_binary_labels(
        f"{training_variant.get('variant_name', 'unknown')} training labels", training_variant.get("labels"), len(features), True
    )
    estimator = classifier_factory(**expected_config)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(features, labels)
    effective_params = estimator.get_params(deep=False)
    effective_config = {key: effective_params.get(key) for key in expected_config}
    if effective_config != expected_config:
        raise ValueError("Logistic smoke estimator effective configuration differs from the fixed configuration.")
    classes = _validate_logistic_estimator_classes(estimator)
    warning_messages = [str(item.message) for item in caught_warnings if issubclass(item.category, ConvergenceWarning)]
    n_iter_raw = getattr(estimator, "n_iter_", None)
    n_iter = None if n_iter_raw is None else [int(value) for value in np.asarray(n_iter_raw).reshape(-1)]
    reached_max_iter = bool(n_iter is not None and any(value >= expected_config["max_iter"] for value in n_iter))
    convergence_passed = bool(n_iter is not None and not warning_messages and not reached_max_iter)
    return {
        "variant_name": training_variant["variant_name"],
        "estimator": estimator,
        "classes": classes,
        "effective_classifier_configuration": effective_config,
        "convergence": {
            "n_iter": n_iter,
            "n_iter_available": n_iter is not None,
            "configured_max_iter": expected_config["max_iter"],
            "convergence_warning_messages": warning_messages,
            "reached_configured_max_iter": reached_max_iter,
            "converged": convergence_passed,
        },
    }


def evaluate_fold_logistic_smoke_classifier(
    fitted_result: Dict[str, Any],
    outer_test_features: np.ndarray,
    y_test: np.ndarray,
    test_sample_ids: List[str],
    threshold: float = LOGISTIC_SMOKE_THRESHOLD,
) -> Dict[str, Any]:
    if threshold != LOGISTIC_SMOKE_THRESHOLD:
        raise ValueError("Phase 8A smoke evaluation requires the fixed threshold 0.5.")
    estimator = fitted_result.get("estimator")
    classes = _validate_logistic_estimator_classes(estimator)
    expected_width = int(getattr(estimator, "n_features_in_", outer_test_features.shape[1]))
    features = _validate_logistic_feature_matrix("Outer-test matrix", outer_test_features, expected_width)
    labels = _validate_logistic_binary_labels("Outer-test labels", y_test, len(features), False)
    record_ids = _validate_logistic_record_ids("Outer-test SAMPLE_ID values", test_sample_ids, len(features))
    features_before = features.copy()
    labels_before = labels.copy()
    probabilities = np.asarray(estimator.predict_proba(features))
    if probabilities.shape != (len(features), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("Logistic smoke predict_proba output must be finite and match the binary test shape.")
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError("Logistic smoke estimator must expose exactly one probability column for binary label 1.")
    probability_high_tmb = probabilities[:, int(positive_indices[0])]
    predicted_labels = (probability_high_tmb >= LOGISTIC_SMOKE_THRESHOLD).astype(int)
    matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    metric_failures = []
    metric_warnings = []
    specificity_denominator = tn + fp
    sensitivity_denominator = tp + fn
    specificity = None if specificity_denominator == 0 else float(tn / specificity_denominator)
    sensitivity = None if sensitivity_denominator == 0 else float(tp / sensitivity_denominator)
    if specificity is None:
        metric_failures.append("specificity is undefined because the outer test has no Low-TMB rows.")
    if sensitivity is None:
        metric_failures.append("sensitivity is undefined because the outer test has no High-TMB rows.")
    if tp + fp == 0:
        metric_warnings.append("precision used zero_division=0 because no High-TMB predictions were made.")
    if sensitivity_denominator == 0:
        metric_warnings.append("recall and F1 used zero_division=0 because no High-TMB rows were present.")
    has_both_test_classes = set(labels.tolist()) == {0, 1}
    if not has_both_test_classes:
        metric_failures.extend(
            [
                "balanced_accuracy is undefined because the outer test does not contain both classes.",
                "auroc is undefined because the outer test does not contain both classes.",
                "auprc is undefined because the outer test does not contain both classes.",
            ]
        )
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_labels)) if has_both_test_classes else None,
        "precision_high_tmb": float(precision_score(labels, predicted_labels, pos_label=1, zero_division=0)),
        "recall_high_tmb": float(recall_score(labels, predicted_labels, pos_label=1, zero_division=0)),
        "sensitivity_high_tmb": sensitivity,
        "specificity_low_tmb": specificity,
        "f1_high_tmb": float(f1_score(labels, predicted_labels, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probability_high_tmb)) if has_both_test_classes else None,
        "auprc": float(average_precision_score(labels, probability_high_tmb)) if has_both_test_classes else None,
    }
    records = [
        {
            "SAMPLE_ID": record_id,
            "true_binary_label": int(true_label),
            "predicted_binary_label": int(predicted_label),
            "probability_high_tmb": float(probability),
        }
        for record_id, true_label, predicted_label, probability in zip(
            record_ids, labels, predicted_labels, probability_high_tmb
        )
    ]
    if not np.array_equal(features, features_before) or not np.array_equal(labels, labels_before):
        raise AssertionError("Logistic smoke evaluation altered outer-test inputs.")
    return {
        "threshold": LOGISTIC_SMOKE_THRESHOLD,
        "positive_class": {"binary_label": 1, "semantic_label": "High-TMB", "probability_column_index": int(positive_indices[0])},
        "metrics": metrics,
        "metric_failures": metric_failures,
        "metric_warnings": metric_warnings,
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "predicted_class_counts": _class_counts(predicted_labels),
        "probability_summary": {
            "min": float(np.min(probability_high_tmb)),
            "max": float(np.max(probability_high_tmb)),
            "mean": float(np.mean(probability_high_tmb)),
        },
        "prediction_records": records,
        "prediction_record_sha256": hashlib.sha256(_canonical_json_bytes(records)).hexdigest(),
        "test_inputs_unchanged": True,
    }


def evaluate_fold_logistic_variants(
    fusion_result: Dict[str, Any],
    training_variants: Dict[str, Dict[str, Any]],
    classifier_config: Dict[str, Any] = LOGISTIC_SMOKE_CONFIGURATION,
    classifier_factory: Any = LogisticRegression,
) -> Dict[str, Any]:
    feature_names = _validate_logistic_feature_schema(
        fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256")
    )
    outer_test_features = _validate_logistic_feature_matrix(
        "Phase 5 outer-test matrix", fusion_result.get("fused_outer_test"), len(feature_names)
    )
    y_test = _validate_logistic_binary_labels("Phase 5 outer-test labels", fusion_result.get("y_test"), len(outer_test_features), False)
    test_sample_ids = _validate_logistic_record_ids(
        "Phase 5 outer-test SAMPLE_ID values", fusion_result.get("test_sample_ids"), len(outer_test_features)
    )
    if fusion_result.get("test_sample_ids_canonical_json_sha256") != _sample_id_list_sha256(test_sample_ids):
        raise ValueError("Phase 5 outer-test SAMPLE_ID hash is invalid.")
    expected_variants = ["real_only", MINORITY_CTGAN_STRATEGY, CONDITIONAL_CTGAN_STRATEGY]
    if not isinstance(training_variants, dict) or list(training_variants) != expected_variants:
        raise ValueError("Logistic smoke evaluation requires exactly the three approved training variants in canonical order.")
    outer_test_before = outer_test_features.copy()
    y_test_before = y_test.copy()
    test_ids_before = list(test_sample_ids)
    test_identity = {
        "outer_test_matrix_sha256": _array_sha256(outer_test_features),
        "outer_test_label_sha256": _array_sha256(y_test),
        "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(test_sample_ids),
        "outer_test_shape": list(outer_test_features.shape),
        "outer_test_class_counts": _class_counts(y_test),
        "latent_feature_name_sha256": fusion_result["latent_feature_name_sha256"],
    }
    result_variants = {}
    estimator_ids = set()
    for variant_name in expected_variants:
        training_variant = training_variants[variant_name]
        if (
            training_variant.get("repeat_id") != fusion_result.get("repeat_id")
            or training_variant.get("fold_id") != fusion_result.get("fold_id")
            or training_variant.get("feature_names") != feature_names
            or training_variant.get("feature_name_sha256") != fusion_result.get("latent_feature_name_sha256")
        ):
            raise ValueError("Logistic smoke training variant identity differs from the selected Phase 5 fold.")
        fitted_result = fit_fold_logistic_smoke_classifier(training_variant, classifier_config, classifier_factory)
        if id(fitted_result["estimator"]) in estimator_ids:
            raise ValueError("Each Logistic smoke variant must use a fresh independent estimator instance.")
        estimator_ids.add(id(fitted_result["estimator"]))
        evaluation_result = evaluate_fold_logistic_smoke_classifier(
            fitted_result, outer_test_features, y_test, test_sample_ids
        )
        result_variants[variant_name] = {
            "variant_name": variant_name,
            "provenance": training_variant["provenance"],
            "training_shape": training_variant["training_shape"],
            "training_class_counts": training_variant["training_class_counts"],
            "synthetic_row_count": training_variant["synthetic_row_count"],
            "effective_classifier_configuration": fitted_result["effective_classifier_configuration"],
            "convergence": fitted_result["convergence"],
            **evaluation_result,
        }
    configurations = [result_variants[name]["effective_classifier_configuration"] for name in expected_variants]
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise AssertionError("Logistic smoke variants did not use identical effective classifier configurations.")
    if (
        not np.array_equal(outer_test_features, outer_test_before)
        or not np.array_equal(y_test, y_test_before)
        or test_sample_ids != test_ids_before
    ):
        raise AssertionError("Logistic smoke evaluation altered shared outer-test inputs.")
    return {
        "repeat_id": fusion_result["repeat_id"],
        "fold_id": fusion_result["fold_id"],
        "comparison_scope": LOGISTIC_SMOKE_COMPARISON_SCOPE,
        "outer_test_identity": test_identity,
        "classifier_configuration": dict(LOGISTIC_SMOKE_CONFIGURATION),
        "fair_comparison_identity": {
            "same_repeat_and_fold": True,
            "same_phase5_feature_schema_and_hash": True,
            "same_unchanged_outer_test_matrix_labels_and_sample_ids": True,
            "effective_classifier_configurations_identical": True,
            "fresh_independent_estimators": True,
        },
        "variants": result_variants,
        "fit_test_isolation": {
            "fit_function_accepts_outer_test_inputs": False,
            "shared_outer_test_matrix_unchanged": True,
            "shared_outer_test_labels_unchanged": True,
            "shared_outer_test_sample_id_order_unchanged": True,
            "all_variants_used_same_outer_test_identity": True,
            "synthetic_records_in_outer_test": False,
        },
        "interpretation_guard": {
            "probabilities_calibrated": False,
            "hyperparameters_tuned": False,
            "threshold_optimized": False,
            "research_result": False,
            "aggregate_score_computed": False,
            "variants_ranked": False,
            "winner_declared": False,
            "final_classifier_selected": False,
        },
    }


def build_logistic_classifier_audit(
    manifest_validation: Dict[str, Any], evaluation_result: Dict[str, Any], outer_test_unchanged: bool
) -> Dict[str, Any]:
    audit_variants = {}
    for variant_name, result in evaluation_result["variants"].items():
        audit_variants[variant_name] = {
            key: value
            for key, value in result.items()
            if key != "prediction_records"
        }
        audit_variants[variant_name]["prediction_records"] = {
            "count": len(result["prediction_records"]),
            "sha256": result["prediction_record_sha256"],
            "ordered_preview": result["prediction_records"][:5],
        }
    return {
        "action": "audited_logistic_classifier_smoke",
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": evaluation_result["repeat_id"],
            "fold_id": evaluation_result["fold_id"],
            **evaluation_result["outer_test_identity"],
        },
        "comparison_scope": evaluation_result["comparison_scope"],
        "classifier_configuration": evaluation_result["classifier_configuration"],
        "fair_comparison_identity": evaluation_result["fair_comparison_identity"],
        "variants": audit_variants,
        "fit_test_isolation": {
            **evaluation_result["fit_test_isolation"],
            "outer_test_features_and_labels_unchanged_after_cli": bool(outer_test_unchanged),
        },
        "interpretation_guard": evaluation_result["interpretation_guard"],
    }


# =========================================
# SECTION 2H: Fold-scoped imbalance baselines
# =========================================


SMOTE_LATENT_STRATEGY = "smote_latent"
SMOTE_LATENT_CONFIGURATION = {
    "sampling_strategy": "auto",
    "random_state": 42,
    "k_neighbors": 5,
}
LOGISTIC_BALANCED_CONFIGURATION = {
    **LOGISTIC_SMOKE_CONFIGURATION,
    "class_weight": "balanced",
}
IMBALANCE_BASELINE_VARIANT_ORDER = [
    "real_only_unweighted",
    "real_only_class_weight_balanced",
    SMOTE_LATENT_STRATEGY,
    MINORITY_CTGAN_STRATEGY,
    CONDITIONAL_CTGAN_STRATEGY,
]
IMBALANCE_BASELINE_CLASSIFIER_CONFIGURATIONS = {
    "real_only_unweighted": dict(LOGISTIC_SMOKE_CONFIGURATION),
    "real_only_class_weight_balanced": dict(LOGISTIC_BALANCED_CONFIGURATION),
    SMOTE_LATENT_STRATEGY: dict(LOGISTIC_SMOKE_CONFIGURATION),
    MINORITY_CTGAN_STRATEGY: dict(LOGISTIC_SMOKE_CONFIGURATION),
    CONDITIONAL_CTGAN_STRATEGY: dict(LOGISTIC_SMOKE_CONFIGURATION),
}
IMBALANCE_BASELINE_COMPARISON_SCOPES = {
    "augmentation_only_unweighted_logistic": [
        "real_only_unweighted",
        SMOTE_LATENT_STRATEGY,
        MINORITY_CTGAN_STRATEGY,
        CONDITIONAL_CTGAN_STRATEGY,
    ],
    "imbalance_management_logistic": list(IMBALANCE_BASELINE_VARIANT_ORDER),
}


def load_imblearn_smote_api() -> Dict[str, Any]:
    try:
        imblearn_module = importlib.import_module("imblearn")
        sklearn_compat_module = importlib.import_module("sklearn_compat")
        over_sampling_module = importlib.import_module("imblearn.over_sampling")
        smote_class = getattr(over_sampling_module, "SMOTE")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Mandatory SMOTE dependency is unavailable or incompatible; no resampling fallback exists."
        ) from error
    constructor_parameters = list(inspect.signature(smote_class).parameters)
    required_parameters = {"sampling_strategy", "random_state", "k_neighbors"}
    if not required_parameters.issubset(constructor_parameters) or not callable(
        getattr(smote_class, "fit_resample", None)
    ):
        raise RuntimeError(
            "Mandatory SMOTE API is incompatible; no resampling fallback exists."
        )
    return {
        "imblearn_version": str(imblearn_module.__version__),
        "sklearn_compat_version": str(getattr(sklearn_compat_module, "__version__", "unknown")),
        "smote_class": smote_class,
        "constructor_parameter_names": constructor_parameters,
        "fit_resample_parameter_names": list(inspect.signature(smote_class.fit_resample).parameters),
        "smote_class_module": smote_class.__module__,
        "smote_class_path": inspect.getfile(smote_class),
    }


def _validate_smote_latent_configuration(smote_config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(smote_config, dict) or dict(smote_config) != SMOTE_LATENT_CONFIGURATION:
        raise ValueError("SMOTE latent configuration must equal the fixed approved configuration.")
    return dict(SMOTE_LATENT_CONFIGURATION)


def extract_smote_latent_training_input(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
    smote_config: Dict[str, Any] = SMOTE_LATENT_CONFIGURATION,
) -> Dict[str, Any]:
    configuration = _validate_smote_latent_configuration(smote_config)
    feature_names = _validate_logistic_feature_schema(latent_feature_names, latent_feature_name_sha256)
    features = _validate_logistic_feature_matrix(
        "SMOTE Phase 5 outer-training matrix", fused_outer_train, len(feature_names)
    )
    labels = _validate_logistic_binary_labels("SMOTE Phase 5 outer-training labels", y_train, len(features), False)
    record_ids = _validate_logistic_record_ids("SMOTE Phase 5 outer-training SAMPLE_ID values", train_sample_ids, len(features))
    if _sample_id_list_sha256(record_ids) != _sample_id_list_sha256(train_sample_ids):
        raise ValueError("SMOTE Phase 5 outer-training SAMPLE_ID metadata is invalid.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        raise ValueError("SMOTE latent training requires exactly two classes with strict class imbalance.")
    minority_index = int(np.argmin(counts))
    minority_label = int(classes[minority_index])
    minority_count = int(counts[minority_index])
    majority_label = int(classes[int(np.argmax(counts))])
    majority_count = int(np.max(counts))
    if minority_count <= configuration["k_neighbors"]:
        raise ValueError("SMOTE k_neighbors must be smaller than the minority training count.")
    return {
        "features": features,
        "labels": labels,
        "train_sample_ids": record_ids,
        "feature_names": feature_names,
        "feature_name_sha256": latent_feature_name_sha256,
        "minority_label": minority_label,
        "majority_label": majority_label,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "needed_synthetic_count": majority_count - minority_count,
        "smote_configuration": configuration,
    }


def fit_and_resample_smote_latent(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    smote_config: Dict[str, Any] = SMOTE_LATENT_CONFIGURATION,
    api: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    configuration = _validate_smote_latent_configuration(smote_config)
    if not isinstance(fused_outer_train, np.ndarray) or fused_outer_train.ndim != 2:
        raise ValueError("SMOTE outer-training features must be a two-dimensional NumPy matrix.")
    features = _validate_logistic_feature_matrix(
        "SMOTE outer-training features", fused_outer_train, fused_outer_train.shape[1]
    )
    labels = _validate_logistic_binary_labels("SMOTE outer-training labels", y_train, len(features), False)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        raise ValueError("SMOTE latent training requires exactly two classes with strict class imbalance.")
    minority_label = int(classes[int(np.argmin(counts))])
    minority_count = int(np.min(counts))
    majority_count = int(np.max(counts))
    if minority_count <= configuration["k_neighbors"]:
        raise ValueError("SMOTE k_neighbors must be smaller than the minority training count.")
    features_before = features.copy()
    labels_before = labels.copy()
    resolved_api = load_imblearn_smote_api() if api is None else api
    if not isinstance(resolved_api, dict) or not callable(resolved_api.get("smote_class")):
        raise ValueError("SMOTE API descriptor is invalid.")
    constructor_parameters = set(resolved_api.get("constructor_parameter_names", []))
    if not set(configuration).issubset(constructor_parameters):
        raise ValueError("SMOTE API descriptor does not support the fixed configuration.")
    try:
        resampler = resolved_api["smote_class"](**configuration)
    except Exception as error:
        raise RuntimeError("Mandatory SMOTE construction failed; no resampling fallback exists.") from error
    try:
        resampled_features, resampled_labels = resampler.fit_resample(features, labels)
    except Exception as error:
        raise RuntimeError("Mandatory SMOTE resampling failed; no resampling fallback exists.") from error
    if not np.array_equal(features, features_before) or not np.array_equal(labels, labels_before):
        raise AssertionError("SMOTE altered original outer-training inputs.")
    if not isinstance(resampled_features, np.ndarray) or resampled_features.dtype != np.float32:
        raise RuntimeError("Mandatory SMOTE resampled output validation failed; no resampling fallback exists.")
    if resampled_features.ndim != 2 or resampled_features.shape[1] != features.shape[1]:
        raise RuntimeError("Mandatory SMOTE resampled output validation failed; no resampling fallback exists.")
    if not np.isfinite(resampled_features).all() or len(resampled_features) < len(features):
        raise RuntimeError("Mandatory SMOTE resampled output validation failed; no resampling fallback exists.")
    resampled_labels = np.asarray(resampled_labels)
    if (
        resampled_labels.ndim != 1
        or len(resampled_labels) != len(resampled_features)
        or not np.issubdtype(resampled_labels.dtype, np.integer)
        or not set(resampled_labels.tolist()).issubset({0, 1})
        or not np.array_equal(resampled_features[: len(features)], features)
        or not np.array_equal(resampled_labels[: len(labels)], labels)
    ):
        raise RuntimeError("Mandatory SMOTE resampled output validation failed; no resampling fallback exists.")
    generated_count = int(len(resampled_features) - len(features))
    generated_labels = resampled_labels[len(labels) :]
    resampled_counts = _class_counts(resampled_labels)
    if (
        generated_count != majority_count - minority_count
        or not len(generated_labels)
        or not np.all(generated_labels == minority_label)
        or resampled_counts.get(str(minority_label)) != majority_count
        or resampled_counts.get(str(int(classes[int(np.argmax(counts))]))) != majority_count
    ):
        raise RuntimeError("Mandatory SMOTE resampled output validation failed; no resampling fallback exists.")
    return {
        "resampled_features": resampled_features,
        "resampled_labels": resampled_labels,
        "resampler": resampler,
        "generated_count": generated_count,
        "minority_label": minority_label,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "smote_configuration": configuration,
        "api": resolved_api,
        "prefix_validation": {
            "output_at_least_original_rows": True,
            "real_feature_prefix_exact": True,
            "real_label_prefix_exact": True,
            "generated_rows_are_appended": True,
        },
    }


def build_smote_latent_augmentation(
    fused_outer_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_ids: List[str],
    latent_feature_names: List[str],
    latent_feature_name_sha256: str,
    repeat_id: int,
    fold_id: int,
    smote_config: Dict[str, Any] = SMOTE_LATENT_CONFIGURATION,
    api: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    features_before = np.array(fused_outer_train, copy=True)
    labels_before = np.array(y_train, copy=True)
    extracted = extract_smote_latent_training_input(
        fused_outer_train,
        y_train,
        train_sample_ids,
        latent_feature_names,
        latent_feature_name_sha256,
        smote_config,
    )
    resampled = fit_and_resample_smote_latent(
        extracted["features"], extracted["labels"], extracted["smote_configuration"], api
    )
    if not np.array_equal(fused_outer_train, features_before) or not np.array_equal(y_train, labels_before):
        raise AssertionError("SMOTE altered original outer-training features or labels.")
    generated = resampled["resampled_features"][len(extracted["features"]) :]
    synthetic_ids = [
        f"SYNTHETIC:SMOTE:R{repeat_id:03d}:F{fold_id:03d}:CLASS{extracted['minority_label']}:{index:06d}"
        for index in range(resampled["generated_count"])
    ]
    if len(synthetic_ids) != len(set(synthetic_ids)) or set(synthetic_ids).intersection(extracted["train_sample_ids"]):
        raise AssertionError("SMOTE synthetic record identifiers must be unique and distinct from real SAMPLE_ID values.")
    augmented_ids = list(extracted["train_sample_ids"]) + synthetic_ids
    is_synthetic = np.concatenate(
        [np.zeros(len(extracted["features"]), dtype=bool), np.ones(len(generated), dtype=bool)]
    )
    return {
        "strategy": SMOTE_LATENT_STRATEGY,
        "repeat_id": int(repeat_id),
        "fold_id": int(fold_id),
        "real_outer_train": extracted["features"].copy(),
        "synthetic_minority": generated.copy(),
        "augmented_outer_train": resampled["resampled_features"].copy(),
        "y_augmented": resampled["resampled_labels"].copy(),
        "is_synthetic": is_synthetic,
        "real_sample_ids": list(extracted["train_sample_ids"]),
        "synthetic_record_ids": synthetic_ids,
        "augmented_record_ids": augmented_ids,
        "minority_label": extracted["minority_label"],
        "majority_label": extracted["majority_label"],
        "minority_count": extracted["minority_count"],
        "majority_count": extracted["majority_count"],
        "needed_synthetic_count": extracted["needed_synthetic_count"],
        "generated_synthetic_count": resampled["generated_count"],
        "original_class_counts": _class_counts(extracted["labels"]),
        "augmented_class_counts": _class_counts(resampled["resampled_labels"]),
        "feature_names": list(extracted["feature_names"]),
        "feature_name_sha256": extracted["feature_name_sha256"],
        "smote_configuration": dict(extracted["smote_configuration"]),
        "smote_api": {
            key: resampled["api"].get(key)
            for key in [
                "imblearn_version",
                "sklearn_compat_version",
                "constructor_parameter_names",
                "fit_resample_parameter_names",
                "smote_class_module",
                "smote_class_path",
            ]
            if key in resampled["api"]
        },
        "prefix_validation": resampled["prefix_validation"],
        "reproducibility_metadata": {
            "random_state": extracted["smote_configuration"]["random_state"],
            "configured_k_neighbors": extracted["smote_configuration"]["k_neighbors"],
            "sampling_strategy": extracted["smote_configuration"]["sampling_strategy"],
        },
        "checks": {
            "synthetic_schema": {"passed": True},
            "synthetic_finite": {"passed": True},
            "real_rows_first": {"passed": True},
            "synthetic_rows_second": {"passed": True},
            "synthetic_record_ids": {"passed": True},
            "fallback_exists": False,
            "outer_test_supplied_to_smote": False,
            "real_outer_training_unchanged": {"passed": True},
        },
    }


def _build_phase8b_real_only_variant(
    fusion_result: Dict[str, Any], variant_name: str, imbalance_method: str
) -> Dict[str, Any]:
    base = _build_real_only_logistic_variant(fusion_result)
    return {
        **base,
        "variant_name": variant_name,
        "provenance": "phase5_fused_outer_train",
        "features": base["features"].copy(),
        "labels": base["labels"].copy(),
        "record_ids": list(base["record_ids"]),
        "is_synthetic": base["is_synthetic"].copy(),
        "imbalance_method": imbalance_method,
        "generates_synthetic_rows": False,
    }


def _build_phase8b_augmented_variant(
    augmentation_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
    expected_strategy: str,
    provenance: str,
) -> Dict[str, Any]:
    feature_names = _validate_logistic_feature_schema(
        fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256")
    )
    real_features = _validate_logistic_feature_matrix(
        "Phase 5 real outer-training matrix", fusion_result.get("fused_outer_train"), len(feature_names)
    )
    real_labels = _validate_logistic_binary_labels(
        "Phase 5 real outer-training labels", fusion_result.get("y_train"), len(real_features), True
    )
    real_ids = _validate_logistic_record_ids(
        "Phase 5 outer-training SAMPLE_ID values", fusion_result.get("train_sample_ids"), len(real_features)
    )
    if not isinstance(augmentation_result, dict) or augmentation_result.get("strategy") != expected_strategy:
        raise ValueError(f"Phase 8B augmentation does not identify strategy {expected_strategy}.")
    if augmentation_result.get("repeat_id") != fusion_result.get("repeat_id") or augmentation_result.get("fold_id") != fusion_result.get("fold_id"):
        raise ValueError("Phase 8B augmentation repeat/fold differs from Phase 5.")
    if augmentation_result.get("feature_names") != feature_names or augmentation_result.get("feature_name_sha256") != fusion_result.get("latent_feature_name_sha256"):
        raise ValueError("Phase 8B augmentation latent feature schema differs from Phase 5.")
    strategy_real = _validate_logistic_feature_matrix(
        f"{expected_strategy} real outer-training matrix", augmentation_result.get("real_outer_train"), len(feature_names)
    )
    if not np.array_equal(strategy_real, real_features) or augmentation_result.get("real_sample_ids") != real_ids:
        raise ValueError("Phase 8B augmentation real-training provenance differs from Phase 5.")
    synthetic = _validate_logistic_feature_matrix(
        f"{expected_strategy} synthetic matrix", augmentation_result.get("synthetic_minority"), len(feature_names)
    )
    features = _validate_logistic_feature_matrix(
        f"{expected_strategy} augmented outer-training matrix", augmentation_result.get("augmented_outer_train"), len(feature_names)
    )
    labels = _validate_logistic_binary_labels(
        f"{expected_strategy} augmented outer-training labels", augmentation_result.get("y_augmented"), len(features), True
    )
    record_ids = _validate_logistic_record_ids(
        f"{expected_strategy} augmented record identifiers", augmentation_result.get("augmented_record_ids"), len(features)
    )
    marker = np.asarray(augmentation_result.get("is_synthetic"))
    if marker.dtype != bool or marker.ndim != 1 or len(marker) != len(features):
        raise ValueError("Phase 8B augmentation synthetic-row marker is invalid.")
    real_count = len(real_features)
    synthetic_ids = _validate_logistic_record_ids(
        f"{expected_strategy} synthetic record identifiers", augmentation_result.get("synthetic_record_ids"), len(synthetic)
    )
    if (
        augmentation_result.get("needed_synthetic_count") != len(synthetic)
        or augmentation_result.get("generated_synthetic_count") != len(synthetic)
        or len(features) != real_count + len(synthetic)
        or not np.array_equal(features[:real_count], real_features)
        or not np.array_equal(features[real_count:], synthetic)
        or not np.array_equal(labels[:real_count], real_labels)
        or not np.array_equal(marker[:real_count], np.zeros(real_count, dtype=bool))
        or not np.array_equal(marker[real_count:], np.ones(len(synthetic), dtype=bool))
        or record_ids[:real_count] != real_ids
        or record_ids[real_count:] != synthetic_ids
        or set(synthetic_ids).intersection(real_ids)
    ):
        raise ValueError("Phase 8B augmentation rows, labels, identifiers, or provenance are invalid.")
    smote_metadata = {}
    if expected_strategy == SMOTE_LATENT_STRATEGY:
        smote_metadata = {
            "smote_configuration": dict(augmentation_result.get("smote_configuration", {})),
            "smote_api": dict(augmentation_result.get("smote_api", {})),
            "prefix_validation": dict(augmentation_result.get("prefix_validation", {})),
            "reproducibility_metadata": dict(augmentation_result.get("reproducibility_metadata", {})),
        }
    return {
        "variant_name": expected_strategy,
        "provenance": provenance,
        "repeat_id": fusion_result["repeat_id"],
        "fold_id": fusion_result["fold_id"],
        "features": features.copy(),
        "labels": labels.copy(),
        "record_ids": list(record_ids),
        "is_synthetic": marker.copy(),
        "synthetic_row_count": int(np.sum(marker)),
        "training_shape": list(features.shape),
        "training_class_counts": _class_counts(labels),
        "feature_names": feature_names,
        "feature_name_sha256": fusion_result["latent_feature_name_sha256"],
        "imbalance_method": "synthetic_augmentation",
        "generates_synthetic_rows": True,
        **smote_metadata,
    }


def build_fold_imbalance_baseline_variants(
    fusion_result: Dict[str, Any],
    minority_only_result: Dict[str, Any],
    conditional_result: Dict[str, Any],
    smote_result: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    variants = {
        "real_only_unweighted": _build_phase8b_real_only_variant(
            fusion_result, "real_only_unweighted", "none"
        ),
        "real_only_class_weight_balanced": _build_phase8b_real_only_variant(
            fusion_result, "real_only_class_weight_balanced", "cost_sensitive_class_weight"
        ),
        SMOTE_LATENT_STRATEGY: _build_phase8b_augmented_variant(
            smote_result,
            fusion_result,
            SMOTE_LATENT_STRATEGY,
            "phase8b_smote_latent_augmented_outer_train",
        ),
        MINORITY_CTGAN_STRATEGY: _build_phase8b_augmented_variant(
            minority_only_result,
            fusion_result,
            MINORITY_CTGAN_STRATEGY,
            "phase6_minority_only_ctgan_augmented_outer_train",
        ),
        CONDITIONAL_CTGAN_STRATEGY: _build_phase8b_augmented_variant(
            conditional_result,
            fusion_result,
            CONDITIONAL_CTGAN_STRATEGY,
            "phase6_conditional_all_training_ctgan_augmented_outer_train",
        ),
    }
    synthetic_ids = []
    for variant_name in [SMOTE_LATENT_STRATEGY, MINORITY_CTGAN_STRATEGY, CONDITIONAL_CTGAN_STRATEGY]:
        variant = variants[variant_name]
        synthetic_ids.extend(
            record_id for record_id, is_synthetic in zip(variant["record_ids"], variant["is_synthetic"]) if is_synthetic
        )
    if len(synthetic_ids) != len(set(synthetic_ids)):
        raise ValueError("Phase 8B synthetic record identifiers collide across augmentation strategies.")
    for variant_name, variant in variants.items():
        variant["classifier_configuration"] = dict(IMBALANCE_BASELINE_CLASSIFIER_CONFIGURATIONS[variant_name])
        variant["comparison_scope_membership"] = [
            scope for scope, members in IMBALANCE_BASELINE_COMPARISON_SCOPES.items() if variant_name in members
        ]
    return variants


def _validate_imbalance_baseline_classifier_configuration(
    variant_name: str, classifier_config: Dict[str, Any]
) -> Dict[str, Any]:
    expected = IMBALANCE_BASELINE_CLASSIFIER_CONFIGURATIONS.get(variant_name)
    if expected is None or not isinstance(classifier_config, dict) or dict(classifier_config) != expected:
        raise ValueError("Imbalance baseline classifier configuration is not approved for this variant.")
    return dict(expected)


def fit_fold_logistic_imbalance_baseline(
    training_variant: Dict[str, Any],
    classifier_config: Dict[str, Any],
    classifier_factory: Any = LogisticRegression,
) -> Dict[str, Any]:
    if not isinstance(training_variant, dict):
        raise ValueError("Imbalance baseline training variant must be a dictionary.")
    variant_name = training_variant.get("variant_name")
    expected_config = _validate_imbalance_baseline_classifier_configuration(variant_name, classifier_config)
    features = _validate_logistic_feature_matrix(
        f"{variant_name} training matrix", training_variant.get("features"), len(training_variant.get("feature_names", []))
    )
    labels = _validate_logistic_binary_labels(f"{variant_name} training labels", training_variant.get("labels"), len(features), True)
    estimator = classifier_factory(**expected_config)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(features, labels)
    effective_params = estimator.get_params(deep=False)
    effective_config = {key: effective_params.get(key) for key in expected_config}
    if effective_config != expected_config:
        raise ValueError("Imbalance baseline estimator effective configuration differs from the approved configuration.")
    classes = _validate_logistic_estimator_classes(estimator)
    warning_messages = [str(item.message) for item in caught_warnings if issubclass(item.category, ConvergenceWarning)]
    n_iter_raw = getattr(estimator, "n_iter_", None)
    n_iter = None if n_iter_raw is None else [int(value) for value in np.asarray(n_iter_raw).reshape(-1)]
    reached_max_iter = bool(n_iter is not None and any(value >= expected_config["max_iter"] for value in n_iter))
    return {
        "variant_name": variant_name,
        "estimator": estimator,
        "classes": classes,
        "effective_classifier_configuration": effective_config,
        "convergence": {
            "n_iter": n_iter,
            "n_iter_available": n_iter is not None,
            "configured_max_iter": expected_config["max_iter"],
            "convergence_warning_messages": warning_messages,
            "reached_configured_max_iter": reached_max_iter,
            "converged": bool(n_iter is not None and not warning_messages and not reached_max_iter),
        },
    }


def evaluate_fold_imbalance_baselines(
    fusion_result: Dict[str, Any],
    training_variants: Dict[str, Dict[str, Any]],
    classifier_factory: Any = LogisticRegression,
) -> Dict[str, Any]:
    feature_names = _validate_logistic_feature_schema(
        fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256")
    )
    outer_test_features = _validate_logistic_feature_matrix(
        "Phase 5 outer-test matrix", fusion_result.get("fused_outer_test"), len(feature_names)
    )
    y_test = _validate_logistic_binary_labels("Phase 5 outer-test labels", fusion_result.get("y_test"), len(outer_test_features), False)
    test_sample_ids = _validate_logistic_record_ids(
        "Phase 5 outer-test SAMPLE_ID values", fusion_result.get("test_sample_ids"), len(outer_test_features)
    )
    if list(training_variants) != IMBALANCE_BASELINE_VARIANT_ORDER:
        raise ValueError("Imbalance baseline evaluation requires exactly five approved variants in canonical order.")
    outer_test_before = outer_test_features.copy()
    y_test_before = y_test.copy()
    test_ids_before = list(test_sample_ids)
    test_identity = {
        "outer_test_matrix_sha256": _array_sha256(outer_test_features),
        "outer_test_label_sha256": _array_sha256(y_test),
        "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(test_sample_ids),
        "outer_test_shape": list(outer_test_features.shape),
        "outer_test_class_counts": _class_counts(y_test),
        "latent_feature_name_sha256": fusion_result["latent_feature_name_sha256"],
    }
    results = {}
    estimators = []
    for variant_name in IMBALANCE_BASELINE_VARIANT_ORDER:
        variant = training_variants[variant_name]
        if (
            variant.get("repeat_id") != fusion_result.get("repeat_id")
            or variant.get("fold_id") != fusion_result.get("fold_id")
            or variant.get("feature_names") != feature_names
            or variant.get("feature_name_sha256") != fusion_result.get("latent_feature_name_sha256")
        ):
            raise ValueError("Imbalance baseline training variant identity differs from the selected Phase 5 fold.")
        if set(variant.get("record_ids", [])).intersection(test_sample_ids):
            raise ValueError("Training or synthetic record identifiers may not appear in outer-test evaluation.")
        fitted = fit_fold_logistic_imbalance_baseline(
            variant, variant["classifier_configuration"], classifier_factory
        )
        if any(fitted["estimator"] is prior_estimator for prior_estimator in estimators):
            raise ValueError("Each imbalance baseline variant must use a fresh independent estimator instance.")
        estimators.append(fitted["estimator"])
        evaluated = evaluate_fold_logistic_smoke_classifier(
            fitted, outer_test_features, y_test, test_sample_ids
        )
        results[variant_name] = {
            "variant_name": variant_name,
            "provenance": variant["provenance"],
            "imbalance_method": variant["imbalance_method"],
            "generates_synthetic_rows": variant["generates_synthetic_rows"],
            "training_shape": variant["training_shape"],
            "training_class_counts": variant["training_class_counts"],
            "synthetic_row_count": variant["synthetic_row_count"],
            "comparison_scope_membership": variant["comparison_scope_membership"],
            "effective_classifier_configuration": fitted["effective_classifier_configuration"],
            "convergence": fitted["convergence"],
            **evaluated,
        }
        if variant_name == SMOTE_LATENT_STRATEGY:
            results[variant_name].update(
                {
                    "smote_configuration": variant["smote_configuration"],
                    "smote_api": variant["smote_api"],
                    "prefix_validation": variant["prefix_validation"],
                    "reproducibility_metadata": variant["reproducibility_metadata"],
                }
            )
    unweighted_names = IMBALANCE_BASELINE_COMPARISON_SCOPES["augmentation_only_unweighted_logistic"]
    if any(results[name]["effective_classifier_configuration"] != LOGISTIC_SMOKE_CONFIGURATION for name in unweighted_names):
        raise AssertionError("All unweighted augmentation variants must use the same approved classifier configuration.")
    balanced_config = results["real_only_class_weight_balanced"]["effective_classifier_configuration"]
    if (
        balanced_config.get("class_weight") != "balanced"
        or {key: value for key, value in balanced_config.items() if key != "class_weight"}
        != {key: value for key, value in LOGISTIC_SMOKE_CONFIGURATION.items() if key != "class_weight"}
    ):
        raise AssertionError("The class-weighted baseline must differ only by class_weight.")
    if (
        not np.array_equal(outer_test_features, outer_test_before)
        or not np.array_equal(y_test, y_test_before)
        or test_sample_ids != test_ids_before
    ):
        raise AssertionError("Imbalance baseline evaluation altered shared outer-test inputs.")
    return {
        "repeat_id": fusion_result["repeat_id"],
        "fold_id": fusion_result["fold_id"],
        "comparison_scopes": {key: list(value) for key, value in IMBALANCE_BASELINE_COMPARISON_SCOPES.items()},
        "outer_test_identity": test_identity,
        "classifier_configurations": {
            name: dict(results[name]["effective_classifier_configuration"])
            for name in IMBALANCE_BASELINE_VARIANT_ORDER
        },
        "fair_comparison_identity": {
            "same_repeat_and_fold": True,
            "same_phase5_feature_schema_and_hash": True,
            "same_unchanged_outer_test_matrix_labels_and_sample_ids": True,
            "shared_fixed_threshold": LOGISTIC_SMOKE_THRESHOLD,
            "unweighted_augmentation_configurations_identical": True,
            "balanced_configuration_differs_only_by_class_weight": True,
            "fresh_independent_estimators": True,
        },
        "variants": results,
        "fit_test_isolation": {
            "smote_functions_accept_outer_test_inputs": False,
            "training_variant_construction_accepts_outer_test_inputs": False,
            "shared_outer_test_matrix_unchanged": True,
            "shared_outer_test_labels_unchanged": True,
            "shared_outer_test_sample_id_order_unchanged": True,
            "all_variants_used_same_outer_test_identity": True,
            "training_and_synthetic_record_ids_disjoint_from_outer_test": True,
        },
        "interpretation_guard": {
            "probabilities_calibrated": False,
            "hyperparameters_tuned": False,
            "threshold_optimized": False,
            "research_result": False,
            "aggregate_score_computed": False,
            "variants_ranked": False,
            "winner_declared": False,
            "final_method_selected": False,
        },
    }


def build_imbalance_baselines_audit(
    manifest_validation: Dict[str, Any], evaluation_result: Dict[str, Any], outer_test_unchanged: bool
) -> Dict[str, Any]:
    audit_variants = {}
    for variant_name, result in evaluation_result["variants"].items():
        audit_variants[variant_name] = {
            key: value for key, value in result.items() if key != "prediction_records"
        }
        audit_variants[variant_name]["prediction_records"] = {
            "count": len(result["prediction_records"]),
            "sha256": result["prediction_record_sha256"],
            "ordered_preview": result["prediction_records"][:5],
        }
    return {
        "action": "audited_imbalance_baselines",
        "manifest_validation": manifest_validation,
        "selected_fold": {
            "repeat_id": evaluation_result["repeat_id"],
            "fold_id": evaluation_result["fold_id"],
            **evaluation_result["outer_test_identity"],
        },
        "comparison_scopes": evaluation_result["comparison_scopes"],
        "classifier_configurations": evaluation_result["classifier_configurations"],
        "fair_comparison_identity": evaluation_result["fair_comparison_identity"],
        "variants": audit_variants,
        "fit_test_isolation": {
            **evaluation_result["fit_test_isolation"],
            "outer_test_features_and_labels_unchanged_after_cli": bool(outer_test_unchanged),
        },
        "interpretation_guard": evaluation_result["interpretation_guard"],
    }


# =========================================
# SECTION 2I: Real-only classifier-family smoke registry
# =========================================


PHASE9A_REGISTRY = {
    "logistic_linear": {"factory": LogisticRegression, "score_source": "predict_proba", "configuration": dict(LOGISTIC_SMOKE_CONFIGURATION)},
    "rbf_svc_decision": {"factory": SVC, "score_source": "decision_function", "configuration": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": False, "class_weight": None, "random_state": 42}},
    "random_forest_bagged": {"factory": RandomForestClassifier, "score_source": "predict_proba", "configuration": {"n_estimators": 200, "criterion": "gini", "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": None, "n_jobs": 1, "random_state": 42}},
    "hist_gradient_boosting": {"factory": HistGradientBoostingClassifier, "score_source": "predict_proba", "configuration": {"loss": "log_loss", "learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 1.0, "early_stopping": False, "random_state": 42, "class_weight": None}},
}
PHASE9A_CLASSIFIER_ORDER = list(PHASE9A_REGISTRY)


def build_phase9a_real_only_training_variant(fusion_result: Dict[str, Any]) -> Dict[str, Any]:
    base = _build_real_only_logistic_variant(fusion_result)
    return {**base, "variant_name": "real_only_unweighted", "provenance": "phase5_fused_outer_train", "features": base["features"].copy(), "labels": base["labels"].copy(), "record_ids": list(base["record_ids"])}


def fit_phase9a_registered_classifier(training_variant: Dict[str, Any], classifier_name: str, classifier_spec: Dict[str, Any], classifier_factory: Any = None) -> Dict[str, Any]:
    if classifier_name not in PHASE9A_REGISTRY or classifier_spec.get("score_source") != PHASE9A_REGISTRY[classifier_name]["score_source"] or classifier_spec.get("configuration") != PHASE9A_REGISTRY[classifier_name]["configuration"]:
        raise ValueError("Phase 9A classifier specification is not approved.")
    features = _validate_logistic_feature_matrix("Phase 9A real-only training matrix", training_variant.get("features"), len(training_variant.get("feature_names", [])))
    labels = _validate_logistic_binary_labels("Phase 9A real-only training labels", training_variant.get("labels"), len(features), True)
    factory = classifier_spec["factory"] if classifier_factory is None else classifier_factory
    estimator = factory(**classifier_spec["configuration"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(features, labels)
    classes = _validate_logistic_estimator_classes(estimator)
    effective = {key: estimator.get_params(deep=False).get(key) for key in classifier_spec["configuration"]}
    if effective != classifier_spec["configuration"]:
        raise ValueError("Phase 9A effective classifier configuration differs from the approved configuration.")
    warning_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    if classifier_name == "random_forest_bagged":
        convergence = {"convergence_status": "not_applicable_non_iterative_ensemble", "fitted_estimator_count": len(getattr(estimator, "estimators_", [])), "convergence_warning_messages": warning_messages}
    elif classifier_name == "rbf_svc_decision":
        fit_status = getattr(estimator, "fit_status_", None)
        n_iter = getattr(estimator, "n_iter_", None)
        convergence = {"convergence_status": "fit_failed" if fit_status not in (None, 0) or warning_messages else "fit_completed", "fit_status": None if fit_status is None else int(fit_status), "n_iter": None if n_iter is None else [int(value) for value in np.asarray(n_iter).reshape(-1)], "convergence_warning_messages": warning_messages}
    elif classifier_name == "hist_gradient_boosting":
        n_iter_values = np.asarray(getattr(estimator, "n_iter_", [])).reshape(-1)
        if len(n_iter_values) != 1:
            raise ValueError("Phase 9A HistGradientBoostingClassifier must expose one n_iter_ value.")
        n_iter = int(n_iter_values[0])
        completed_budget = n_iter == classifier_spec["configuration"]["max_iter"]
        convergence = {"convergence_status": "fixed_iteration_budget_completed" if completed_budget and not warning_messages else "fit_failed", "n_iter": n_iter, "max_iter": classifier_spec["configuration"]["max_iter"], "early_stopping": False, "convergence_warning_messages": warning_messages}
    else:
        n_iter = [int(value) for value in np.asarray(getattr(estimator, "n_iter_", [])).reshape(-1)]
        reached = any(value >= classifier_spec["configuration"]["max_iter"] for value in n_iter)
        convergence = {"convergence_status": "convergence_failed" if warning_messages or reached else "converged", "n_iter": n_iter, "max_iter": classifier_spec["configuration"]["max_iter"], "reached_max_iter": reached, "convergence_warning_messages": warning_messages}
    return {"classifier_name": classifier_name, "estimator": estimator, "classes": classes, "effective_configuration": effective, "score_source": classifier_spec["score_source"], "convergence": convergence}


def _phase9a_score(fitted: Dict[str, Any], features: np.ndarray) -> Dict[str, Any]:
    estimator, classes, source = fitted["estimator"], fitted["classes"], fitted["score_source"]
    if source == "predict_proba":
        values = np.asarray(estimator.predict_proba(features))
        if values.shape != (len(features), 2) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-7):
            raise ValueError("Phase 9A predict_proba output is invalid.")
        index = int(np.flatnonzero(classes == 1)[0])
        return {"scores": values[:, index], "score_type": "uncalibrated_probability", "score_source": "predict_proba", "positive_class_column_index": index, "threshold_type": "probability", "threshold_value": 0.5}
    if source == "decision_function":
        values = np.asarray(estimator.decision_function(features))
        if values.ndim != 1 or len(values) != len(features) or not np.isfinite(values).all():
            raise ValueError("Phase 9A decision_function output is invalid.")
        oriented = values if classes[1] == 1 else -values
        return {"scores": oriented, "score_type": "decision_function", "score_source": "decision_function", "positive_class_column_index": None, "threshold_type": "decision_boundary", "threshold_value": 0.0}
    raise ValueError("Phase 9A classifier exposes no approved continuous score source.")


def evaluate_phase9a_classifier_registry(fusion_result: Dict[str, Any], training_variant: Dict[str, Any], classifier_factories: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    names = _validate_logistic_feature_schema(fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256"))
    train_features = _validate_logistic_feature_matrix("Phase 5 outer-training matrix", training_variant.get("features"), len(names))
    train_labels = _validate_logistic_binary_labels("Phase 5 outer-training labels", training_variant.get("labels"), len(train_features), True)
    train_ids = _validate_logistic_record_ids("Phase 5 outer-training SAMPLE_ID values", training_variant.get("record_ids"), len(train_features))
    test_features = _validate_logistic_feature_matrix("Phase 5 outer-test matrix", fusion_result.get("fused_outer_test"), len(names))
    test_labels = _validate_logistic_binary_labels("Phase 5 outer-test labels", fusion_result.get("y_test"), len(test_features), False)
    test_ids = _validate_logistic_record_ids("Phase 5 outer-test SAMPLE_ID values", fusion_result.get("test_sample_ids"), len(test_features))
    before_train_features, before_train_labels, before_train_ids = train_features.copy(), train_labels.copy(), list(train_ids)
    before_features, before_labels, before_ids = test_features.copy(), test_labels.copy(), list(test_ids)
    results, estimators = {}, []
    for name in PHASE9A_CLASSIFIER_ORDER:
        fitted = fit_phase9a_registered_classifier(training_variant, name, PHASE9A_REGISTRY[name], None if classifier_factories is None else classifier_factories[name])
        if any(fitted["estimator"] is item for item in estimators): raise ValueError("Phase 9A classifiers must use fresh estimator instances.")
        estimators.append(fitted["estimator"])
        adapted = _phase9a_score(fitted, test_features)
        predicted = (adapted["scores"] >= adapted["threshold_value"]).astype(int)
        matrix = confusion_matrix(test_labels, predicted, labels=[0, 1]); tn, fp, fn, tp = [int(item) for item in matrix.ravel()]
        both = set(test_labels.tolist()) == {0, 1}
        records = [{"SAMPLE_ID": identifier, "true_binary_label": int(actual), "predicted_binary_label": int(prediction), "high_tmb_continuous_score": float(score), "score_type": adapted["score_type"]} for identifier, actual, prediction, score in zip(test_ids, test_labels, predicted, adapted["scores"])]
        results[name] = {"effective_configuration": fitted["effective_configuration"], "convergence": fitted["convergence"], **adapted, "score_shape": list(adapted["scores"].shape), "positive_class": {"binary_label": 1, "class_ordering": fitted["classes"].tolist(), "probabilities_calibrated": False}, "metrics": {"accuracy": float(accuracy_score(test_labels, predicted)), "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted)) if both else None, "precision_high_tmb": float(precision_score(test_labels, predicted, zero_division=0)), "recall_high_tmb": float(recall_score(test_labels, predicted, zero_division=0)), "sensitivity_high_tmb": float(tp / (tp + fn)) if tp + fn else None, "specificity_low_tmb": float(tn / (tn + fp)) if tn + fp else None, "f1_high_tmb": float(f1_score(test_labels, predicted, zero_division=0)), "auroc": float(roc_auc_score(test_labels, adapted["scores"])) if both else None, "auprc": float(average_precision_score(test_labels, adapted["scores"])) if both else None}, "confusion_matrix": matrix.tolist(), "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "predicted_class_counts": _class_counts(predicted), "probability_summary": _quality_numeric_summary(adapted["scores"]), "prediction_records": records, "prediction_record_sha256": hashlib.sha256(_canonical_json_bytes(records)).hexdigest()}
    if not np.array_equal(train_features, before_train_features) or not np.array_equal(train_labels, before_train_labels) or train_ids != before_train_ids: raise AssertionError("Phase 9A evaluation altered shared outer-training inputs.")
    if not np.array_equal(test_features, before_features) or not np.array_equal(test_labels, before_labels) or test_ids != before_ids: raise AssertionError("Phase 9A evaluation altered outer-test inputs.")
    logistic_variant = _build_real_only_logistic_variant(fusion_result); old_fit = fit_fold_logistic_smoke_classifier(logistic_variant); old_eval = evaluate_fold_logistic_smoke_classifier(old_fit, test_features, test_labels, test_ids); current = results["logistic_linear"]
    phase8a_scores = np.asarray([record["probability_high_tmb"] for record in old_eval["prediction_records"]])
    phase9a_predictions = (current["scores"] >= LOGISTIC_SMOKE_THRESHOLD).astype(int)
    phase8a_predictions = np.asarray([record["predicted_binary_label"] for record in old_eval["prediction_records"]])
    phase9a_compatible_records = [{"SAMPLE_ID": identifier, "true_binary_label": int(actual), "predicted_binary_label": int(prediction), "probability_high_tmb": float(score)} for identifier, actual, prediction, score in zip(test_ids, test_labels, phase9a_predictions, current["scores"])]
    phase9a_compatible_hash = hashlib.sha256(_canonical_json_bytes(phase9a_compatible_records)).hexdigest()
    metric_keys = ["accuracy", "balanced_accuracy", "precision_high_tmb", "recall_high_tmb", "sensitivity_high_tmb", "specificity_low_tmb", "f1_high_tmb", "auroc", "auprc"]
    parity = {"same_configuration": current["effective_configuration"] == LOGISTIC_SMOKE_CONFIGURATION, "maximum_absolute_probability_difference": float(np.max(np.abs(current["scores"] - phase8a_scores))), "same_predicted_labels": bool(np.array_equal(phase9a_predictions, phase8a_predictions)), "same_confusion_matrix": current["confusion_matrix"] == old_eval["confusion_matrix"], "same_metrics": all(np.isclose(current["metrics"][key], old_eval["metrics"][key], rtol=0, atol=1e-12) for key in metric_keys), "phase9a_phase8a_compatible_prediction_record_sha256": phase9a_compatible_hash, "phase8a_prediction_record_hash": old_eval["prediction_record_sha256"]}
    parity["same_prediction_record_hash"] = parity["phase9a_phase8a_compatible_prediction_record_sha256"] == parity["phase8a_prediction_record_hash"]
    parity["passed"] = all(value for key, value in parity.items() if key not in {"maximum_absolute_probability_difference", "phase9a_phase8a_compatible_prediction_record_sha256", "phase8a_prediction_record_hash"}) and parity["maximum_absolute_probability_difference"] <= 1e-12
    if not parity["passed"]: raise AssertionError("Phase 9A logistic_linear parity with Phase 8A real_only is broken.")
    train_identity = {"outer_train_matrix_sha256": _array_sha256(train_features), "outer_train_label_sha256": _array_sha256(train_labels), "outer_train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(train_ids), "outer_train_shape": list(train_features.shape), "outer_train_class_counts": _class_counts(train_labels)}
    identity = {"outer_test_matrix_sha256": _array_sha256(test_features), "outer_test_label_sha256": _array_sha256(test_labels), "outer_test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(test_ids), "outer_test_shape": list(test_features.shape), "outer_test_class_counts": _class_counts(test_labels), "latent_feature_name_sha256": fusion_result["latent_feature_name_sha256"]}
    return {"comparison_scope": "real_only_unweighted_classifier_family_smoke", "outer_training_identity": train_identity, "outer_test_identity": identity, "classifiers": results, "logistic_phase8a_parity": parity, "interpretation_guard": {"model_specific_post_fusion_scaling": False, "probabilities_calibrated": False, "hyperparameters_tuned": False, "threshold_optimized": False, "scientific_classifier_comparison": False, "research_result": False, "aggregate_score_computed": False, "classifiers_ranked": False, "winner_declared": False, "final_classifier_selected": False}}


def build_phase9a_classifier_registry_audit(manifest_validation: Dict[str, Any], fusion_result: Dict[str, Any], result: Dict[str, Any], outer_test_unchanged: bool) -> Dict[str, Any]:
    classifiers = {}
    for name, value in result["classifiers"].items():
        classifiers[name] = {key: item for key, item in value.items() if key not in {"prediction_records", "scores"}}
        classifiers[name]["prediction_records"] = {"count": len(value["prediction_records"]), "sha256": value["prediction_record_sha256"], "ordered_preview": value["prediction_records"][:5]}
    return {"action": "audited_classifier_registry_smoke", "manifest_validation": manifest_validation, "selected_fold": {"repeat_id": fusion_result["repeat_id"], "fold_id": fusion_result["fold_id"], **result["outer_training_identity"], **result["outer_test_identity"]}, "comparison_scope": result["comparison_scope"], "registry": {name: {"configuration": spec["configuration"], "score_source": spec["score_source"]} for name, spec in PHASE9A_REGISTRY.items()}, "classifiers": classifiers, "logistic_phase8a_parity": result["logistic_phase8a_parity"], "fit_test_isolation": {"fit_functions_accept_outer_test_inputs": False, "outer_test_features_and_labels_unchanged_after_cli": bool(outer_test_unchanged), "shared_outer_training_inputs_unchanged": True, "all_classifiers_used_same_real_only_training_identity": True, "all_classifiers_used_same_outer_test_identity": True}, "interpretation_guard": result["interpretation_guard"]}


# =========================================
# SECTION 2J: Real-only model-specific pipeline smoke registry
# =========================================


PHASE9B_STANDARD_SCALER_CONFIGURATION = {"copy": True, "with_mean": True, "with_std": True}
PHASE9B_PIPELINE_REGISTRY = {
    "logistic_linear": {**PHASE9A_REGISTRY["logistic_linear"], "post_fusion_scaling_enabled": True},
    "rbf_svc_decision": {**PHASE9A_REGISTRY["rbf_svc_decision"], "post_fusion_scaling_enabled": True},
    "random_forest_bagged": {**PHASE9A_REGISTRY["random_forest_bagged"], "post_fusion_scaling_enabled": False},
    "hist_gradient_boosting": {**PHASE9A_REGISTRY["hist_gradient_boosting"], "post_fusion_scaling_enabled": False},
}
PHASE9B_CLASSIFIER_ORDER = list(PHASE9B_PIPELINE_REGISTRY)


def build_phase9b_real_only_training_variant(fusion_result: Dict[str, Any]) -> Dict[str, Any]:
    base = _build_real_only_logistic_variant(fusion_result)
    return {**base, "variant_name": "real_only_unweighted", "provenance": "phase5_fused_outer_train", "features": base["features"].copy(), "labels": base["labels"].copy(), "record_ids": list(base["record_ids"])}


def _phase9b_convergence(classifier_name: str, classifier: Any, configuration: Dict[str, Any], warning_messages: List[str]) -> Dict[str, Any]:
    if classifier_name == "random_forest_bagged":
        return {"convergence_status": "not_applicable_non_iterative_ensemble", "fitted_estimator_count": len(getattr(classifier, "estimators_", [])), "convergence_warning_messages": warning_messages}
    if classifier_name == "rbf_svc_decision":
        fit_status, n_iter = getattr(classifier, "fit_status_", None), getattr(classifier, "n_iter_", None)
        return {"convergence_status": "fit_failed" if fit_status not in (None, 0) or warning_messages else "fit_completed", "fit_status": None if fit_status is None else int(fit_status), "n_iter": None if n_iter is None else [int(value) for value in np.asarray(n_iter).reshape(-1)], "convergence_warning_messages": warning_messages}
    if classifier_name == "hist_gradient_boosting":
        values = np.asarray(getattr(classifier, "n_iter_", [])).reshape(-1)
        if len(values) != 1:
            raise ValueError("Phase 9B HistGradientBoostingClassifier must expose one n_iter_ value.")
        n_iter = int(values[0])
        return {"convergence_status": "fixed_iteration_budget_completed" if n_iter == configuration["max_iter"] and not warning_messages else "fit_failed", "n_iter": n_iter, "max_iter": configuration["max_iter"], "early_stopping": False, "convergence_warning_messages": warning_messages}
    n_iter = [int(value) for value in np.asarray(getattr(classifier, "n_iter_", [])).reshape(-1)]
    reached = any(value >= configuration["max_iter"] for value in n_iter)
    return {"convergence_status": "convergence_failed" if warning_messages or reached else "converged", "n_iter": n_iter, "max_iter": configuration["max_iter"], "reached_max_iter": reached, "convergence_warning_messages": warning_messages}


def _phase9b_scaler_evidence(scaler: Any, train_features: np.ndarray, test_features: np.ndarray, feature_names: List[str], train_id_hash: str) -> Dict[str, Any]:
    transformed_train, transformed_test = np.asarray(scaler.transform(train_features)), np.asarray(scaler.transform(test_features))
    mean, variance, scale = np.asarray(scaler.mean_), np.asarray(scaler.var_), np.asarray(scaler.scale_)
    if mean.shape != (len(feature_names),) or variance.shape != mean.shape or scale.shape != mean.shape:
        raise ValueError("Phase 9B scaler statistics do not match the Phase 5 feature schema.")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or not np.isfinite(scale).all() or not np.isfinite(transformed_train).all() or not np.isfinite(transformed_test).all():
        raise ValueError("Phase 9B scaler produced non-finite statistics or transformed features.")
    zero_mask = variance == 0
    if np.any(scale[zero_mask] != 1.0):
        raise ValueError("Phase 9B zero-variance features must use a finite unit scale.")
    seen = np.asarray(getattr(scaler, "n_samples_seen_", [])).reshape(-1)
    if len(seen) < 1 or not np.all(seen == len(train_features)):
        raise ValueError("Phase 9B scaler n_samples_seen_ must equal the outer-training row count.")
    return {"scaler_configuration": {key: scaler.get_params(deep=False).get(key) for key in PHASE9B_STANDARD_SCALER_CONFIGURATION}, "scaler_n_features_in": int(getattr(scaler, "n_features_in_", -1)), "scaler_n_samples_seen": int(seen[0]), "scaler_fit_row_count": len(train_features), "scaler_training_sample_ids_canonical_json_sha256": train_id_hash, "mean": mean.tolist(), "var": variance.tolist(), "scale": scale.tolist(), "zero_variance_feature_names": [name for name, is_zero in zip(feature_names, zero_mask) if is_zero], "zero_variance_features_have_unit_scale": True, "transformed_train_shape": list(transformed_train.shape), "transformed_test_shape": list(transformed_test.shape), "transformed_train_dtype": str(transformed_train.dtype), "transformed_test_dtype": str(transformed_test.dtype), "transformed_train_matrix_sha256": _array_sha256(transformed_train), "transformed_test_matrix_sha256": _array_sha256(transformed_test), "transformed_train_finite": True, "transformed_test_finite": True, "transformed_test_sample_id_order_preserved": True}


def fit_phase9b_registered_pipeline(training_variant: Dict[str, Any], classifier_name: str, classifier_spec: Dict[str, Any], classifier_factory: Any = None, scaler_factory: Any = StandardScaler) -> Dict[str, Any]:
    if classifier_name not in PHASE9B_PIPELINE_REGISTRY or classifier_spec.get("score_source") != PHASE9B_PIPELINE_REGISTRY[classifier_name]["score_source"] or classifier_spec.get("configuration") != PHASE9B_PIPELINE_REGISTRY[classifier_name]["configuration"] or classifier_spec.get("post_fusion_scaling_enabled") != PHASE9B_PIPELINE_REGISTRY[classifier_name]["post_fusion_scaling_enabled"]:
        raise ValueError("Phase 9B classifier pipeline specification is not approved.")
    feature_names = training_variant.get("feature_names", [])
    features = _validate_logistic_feature_matrix("Phase 9B real-only training matrix", training_variant.get("features"), len(feature_names))
    labels = _validate_logistic_binary_labels("Phase 9B real-only training labels", training_variant.get("labels"), len(features), True)
    factory = classifier_spec["factory"] if classifier_factory is None else classifier_factory
    classifier = factory(**classifier_spec["configuration"])
    if classifier_spec["post_fusion_scaling_enabled"]:
        scaler = scaler_factory(**PHASE9B_STANDARD_SCALER_CONFIGURATION)
        estimator = Pipeline([("standard_scaler", scaler), ("classifier", classifier)])
    else:
        scaler, estimator = None, classifier
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(features, labels)
    terminal_classifier = estimator.named_steps["classifier"] if isinstance(estimator, Pipeline) else estimator
    classes = _validate_logistic_estimator_classes(terminal_classifier)
    effective = {key: terminal_classifier.get_params(deep=False).get(key) for key in classifier_spec["configuration"]}
    if effective != classifier_spec["configuration"]:
        raise ValueError("Phase 9B effective classifier configuration differs from the approved configuration.")
    warning_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    return {"classifier_name": classifier_name, "estimator": estimator, "classifier": terminal_classifier, "scaler": scaler, "classes": classes, "effective_configuration": effective, "score_source": classifier_spec["score_source"], "post_fusion_scaling_enabled": classifier_spec["post_fusion_scaling_enabled"], "pipeline_steps": list(estimator.named_steps) if isinstance(estimator, Pipeline) else ["classifier"], "convergence": _phase9b_convergence(classifier_name, terminal_classifier, classifier_spec["configuration"], warning_messages)}


def evaluate_phase9b_classifier_pipelines(fusion_result: Dict[str, Any], training_variant: Dict[str, Any], classifier_factories: Optional[Dict[str, Any]] = None, scaler_factory: Any = StandardScaler) -> Dict[str, Any]:
    names = _validate_logistic_feature_schema(fusion_result.get("latent_feature_names"), fusion_result.get("latent_feature_name_sha256"))
    train_features = _validate_logistic_feature_matrix("Phase 5 outer-training matrix", training_variant.get("features"), len(names)); train_labels = _validate_logistic_binary_labels("Phase 5 outer-training labels", training_variant.get("labels"), len(train_features), True); train_ids = _validate_logistic_record_ids("Phase 5 outer-training SAMPLE_ID values", training_variant.get("record_ids"), len(train_features))
    test_features = _validate_logistic_feature_matrix("Phase 5 outer-test matrix", fusion_result.get("fused_outer_test"), len(names)); test_labels = _validate_logistic_binary_labels("Phase 5 outer-test labels", fusion_result.get("y_test"), len(test_features), False); test_ids = _validate_logistic_record_ids("Phase 5 outer-test SAMPLE_ID values", fusion_result.get("test_sample_ids"), len(test_features))
    before_train, before_train_labels, before_test, before_test_labels = train_features.copy(), train_labels.copy(), test_features.copy(), test_labels.copy()
    raw_identity = {"repeat_id": fusion_result["repeat_id"], "fold_id": fusion_result["fold_id"], "raw_train_shape": list(train_features.shape), "raw_test_shape": list(test_features.shape), "raw_train_dtype": str(train_features.dtype), "raw_test_dtype": str(test_features.dtype), "raw_train_matrix_sha256": _array_sha256(train_features), "raw_test_matrix_sha256": _array_sha256(test_features), "raw_train_label_sha256": _array_sha256(train_labels), "raw_test_label_sha256": _array_sha256(test_labels), "train_sample_ids_canonical_json_sha256": _sample_id_list_sha256(train_ids), "test_sample_ids_canonical_json_sha256": _sample_id_list_sha256(test_ids), "latent_feature_name_sha256": fusion_result["latent_feature_name_sha256"], "ordered_feature_names": list(names)}
    results, estimators = {}, []
    for name in PHASE9B_CLASSIFIER_ORDER:
        fitted = fit_phase9b_registered_pipeline(training_variant, name, PHASE9B_PIPELINE_REGISTRY[name], None if classifier_factories is None else classifier_factories[name], scaler_factory)
        if any(fitted["estimator"] is item for item in estimators):
            raise ValueError("Phase 9B classifiers must use fresh estimator instances.")
        estimators.append(fitted["estimator"])
        adapted = _phase9a_score({"estimator": fitted["estimator"], "classes": fitted["classes"], "score_source": fitted["score_source"]}, test_features)
        predicted = (adapted["scores"] >= adapted["threshold_value"]).astype(int); matrix = confusion_matrix(test_labels, predicted, labels=[0, 1]); tn, fp, fn, tp = [int(value) for value in matrix.ravel()]; both = set(test_labels.tolist()) == {0, 1}
        records = [{"SAMPLE_ID": identifier, "true_binary_label": int(actual), "predicted_binary_label": int(prediction), "high_tmb_continuous_score": float(score), "score_type": adapted["score_type"]} for identifier, actual, prediction, score in zip(test_ids, test_labels, predicted, adapted["scores"])]
        if fitted["scaler"] is not None:
            transformation = _phase9b_scaler_evidence(fitted["scaler"], train_features, test_features, names, raw_identity["train_sample_ids_canonical_json_sha256"])
            transformation.update({"post_fusion_scaling_enabled": True, "classifier_input_space": "standardized_post_fusion", "scaler_instantiated": True})
        else:
            transformation = {"post_fusion_scaling_enabled": False, "classifier_input_space": "raw_fused_latent", "scaler_instantiated": False, "classifier_training_input_matches_raw_phase5": True, "evaluation_input_matches_raw_phase5": True, "raw_row_and_feature_order_preserved": True, "transformed_train_shape": list(train_features.shape), "transformed_test_shape": list(test_features.shape), "transformed_train_dtype": str(train_features.dtype), "transformed_test_dtype": str(test_features.dtype), "transformed_train_matrix_sha256": _array_sha256(train_features), "transformed_test_matrix_sha256": _array_sha256(test_features), "transformed_train_finite": True, "transformed_test_finite": True}
        results[name] = {"raw_phase5_identity": raw_identity, "pipeline_steps": fitted["pipeline_steps"], "effective_configuration": fitted["effective_configuration"], "convergence": fitted["convergence"], "transformation": transformation, **adapted, "score_shape": list(adapted["scores"].shape), "positive_class": {"binary_label": 1, "class_ordering": fitted["classes"].tolist(), "probabilities_calibrated": False}, "metrics": {"accuracy": float(accuracy_score(test_labels, predicted)), "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted)) if both else None, "precision_high_tmb": float(precision_score(test_labels, predicted, zero_division=0)), "recall_high_tmb": float(recall_score(test_labels, predicted, zero_division=0)), "sensitivity_high_tmb": float(tp / (tp + fn)) if tp + fn else None, "specificity_low_tmb": float(tn / (tn + fp)) if tn + fp else None, "f1_high_tmb": float(f1_score(test_labels, predicted, zero_division=0)), "auroc": float(roc_auc_score(test_labels, adapted["scores"])) if both else None, "auprc": float(average_precision_score(test_labels, adapted["scores"])) if both else None}, "confusion_matrix": matrix.tolist(), "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "predicted_class_counts": _class_counts(predicted), "score_summary": _quality_numeric_summary(adapted["scores"]), "prediction_records": records, "prediction_record_sha256": hashlib.sha256(_canonical_json_bytes(records)).hexdigest()}
    if not np.array_equal(train_features, before_train) or not np.array_equal(train_labels, before_train_labels) or not np.array_equal(test_features, before_test) or not np.array_equal(test_labels, before_test_labels):
        raise AssertionError("Phase 9B pipeline evaluation altered raw Phase 5 inputs.")
    return {"comparison_scope": "real_only_unweighted_model_specific_pipeline_smoke", "raw_phase5_identity": raw_identity, "classifiers": results, "fit_test_isolation": {"fit_functions_accept_outer_test_inputs": False, "shared_raw_train_and_test_unchanged": True, "all_classifiers_used_same_raw_phase5_identity": True}, "interpretation_guard": {"model_specific_pipeline_policy_enabled": True, "model_specific_post_fusion_scaling": True, "probabilities_calibrated": False, "hyperparameters_tuned": False, "threshold_optimized": False, "scientific_classifier_comparison": False, "research_result": False, "aggregate_score_computed": False, "classifiers_ranked": False, "winner_declared": False, "final_classifier_selected": False}}


def build_phase9b_classifier_pipelines_audit(manifest_validation: Dict[str, Any], result: Dict[str, Any], outer_test_unchanged: bool) -> Dict[str, Any]:
    classifiers = {}
    for name, value in result["classifiers"].items():
        classifiers[name] = {key: item for key, item in value.items() if key not in {"prediction_records", "scores", "raw_phase5_identity"}}
        classifiers[name]["prediction_records"] = {"count": len(value["prediction_records"]), "sha256": value["prediction_record_sha256"], "ordered_preview": value["prediction_records"][:5]}
    return {"action": "audited_classifier_pipelines_smoke", "manifest_validation": manifest_validation, "selected_fold": result["raw_phase5_identity"], "comparison_scope": result["comparison_scope"], "registry": {name: {"configuration": spec["configuration"], "score_source": spec["score_source"], "post_fusion_scaling_enabled": spec["post_fusion_scaling_enabled"]} for name, spec in PHASE9B_PIPELINE_REGISTRY.items()}, "classifiers": classifiers, "fit_test_isolation": {**result["fit_test_isolation"], "outer_test_features_and_labels_unchanged_after_cli": bool(outer_test_unchanged)}, "interpretation_guard": result["interpretation_guard"]}


# =========================================
# SECTION 2K: Prespecified nested search-space manifest
# =========================================


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _nested_dependency_versions() -> Dict[str, str]:
    return {name: _package_version(package) for name, package in {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit_learn": "scikit-learn", "tensorflow": "tensorflow", "keras": "keras", "sdv": "sdv", "ctgan": "ctgan", "imbalanced_learn": "imbalanced-learn", "torch": "torch"}.items()}


def _nested_feature_contract(data: Dict[str, Any]) -> Dict[str, Any]:
    modality_keys = {"mGE": "rna", "mDM": "dna", "mCNA": "cna"}
    modalities = {}
    for modality, feature_key in modality_keys.items():
        names = list(data["feature_columns"][feature_key])
        modalities[modality] = {"input_feature_count": len(names), "ordered_feature_names": names, "feature_name_sha256": _feature_name_sha256(names)}
    return {"modalities": modalities, "labels_are_not_features": True}


def _nested_bottleneck_candidates(feature_contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []
    for candidate_id, ratio in [("ratio_025", 0.25), ("ratio_050", 0.50), ("ratio_075", 0.75)]:
        dimensions = {modality: int(np.ceil(feature_contract["modalities"][modality]["input_feature_count"] * ratio)) for modality in ["mGE", "mDM", "mCNA"]}
        if any(value < 2 or value >= feature_contract["modalities"][modality]["input_feature_count"] for modality, value in dimensions.items()):
            raise ValueError("Nested search-space bottleneck candidate is not compressive for every modality.")
        candidates.append({"candidate_id": candidate_id, "compression_ratio": ratio, "latent_dimensions": dimensions, "fused_latent_width": sum(dimensions.values()), "proportional_tuple_only": True})
    return candidates


def _nested_classifier_registry() -> Dict[str, Any]:
    definitions = {
        "logistic_linear": [("logistic_c_0_1", {"C": 0.1}), ("logistic_c_1_baseline", {"C": 1.0}), ("logistic_c_10", {"C": 10.0})],
        "rbf_svc_decision": [("svc_c_0_25", {"C": 0.25}), ("svc_c_1_baseline", {"C": 1.0}), ("svc_c_4", {"C": 4.0})],
        "random_forest_bagged": [("rf_leaf_1", {"min_samples_leaf": 1}), ("rf_leaf_2_baseline", {"min_samples_leaf": 2}), ("rf_leaf_4", {"min_samples_leaf": 4})],
        "hist_gradient_boosting": [("hgb_l2_0_1", {"l2_regularization": 0.1}), ("hgb_l2_1_baseline", {"l2_regularization": 1.0}), ("hgb_l2_10", {"l2_regularization": 10.0})],
    }
    registry = {}
    for family in PHASE9B_CLASSIFIER_ORDER:
        baseline = dict(PHASE9B_PIPELINE_REGISTRY[family]["configuration"])
        constructor_parameters = sorted(inspect.signature(PHASE9B_PIPELINE_REGISTRY[family]["factory"]).parameters)
        configurations = []
        for configuration_id, overrides in definitions[family]:
            configuration = {**baseline, **overrides}
            configurations.append({"configuration_id": configuration_id, "parameters": configuration, "is_phase9b_canonical_baseline": configuration == baseline})
        registry[family] = {"factory_name": PHASE9B_PIPELINE_REGISTRY[family]["factory"].__name__, "constructor_parameter_names": constructor_parameters, "class_weight_supported": "class_weight" in constructor_parameters, "post_fusion_scaling_enabled": PHASE9B_PIPELINE_REGISTRY[family]["post_fusion_scaling_enabled"], "classifier_input_space": "standardized_post_fusion" if PHASE9B_PIPELINE_REGISTRY[family]["post_fusion_scaling_enabled"] else "raw_fused_latent", "score_source": PHASE9B_PIPELINE_REGISTRY[family]["score_source"], "configurations": configurations}
    return registry


def _nested_search_workload(classifier_registry: Dict[str, Any], representation_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    classifier_configuration_count = sum(len(value["configurations"]) for value in classifier_registry.values())
    balanced_supported = sum(len(value["configurations"]) for value in classifier_registry.values() if value["class_weight_supported"])
    stage_a_classifier_fits = len(representation_candidates) * INNER_N_SPLITS * len(PHASE9B_CLASSIFIER_ORDER)
    stage_b_valid_per_inner = classifier_configuration_count * 4 + balanced_supported
    per_outer = {"stage_a_ae_fits": len(representation_candidates) * INNER_N_SPLITS * 3, "stage_a_classifier_fits": stage_a_classifier_fits, "stage_b_ctgan_fits": INNER_N_SPLITS * 2, "stage_b_smote_runs": INNER_N_SPLITS, "stage_b_classifier_fits": stage_b_valid_per_inner * INNER_N_SPLITS, "final_refit_ae_fits": 3, "final_refit_classifier_fits": 1, "final_refit_selected_augmentation_runs": 1, "final_refit_ctgan_fits_range": [0, 1], "final_refit_smote_runs_range": [0, 1]}
    totals = {key: value * (OUTER_N_SPLITS * OUTER_N_REPEATS) for key, value in per_outer.items() if isinstance(value, int)}
    totals["final_refit_ctgan_fits_range"] = [0, OUTER_N_SPLITS * OUTER_N_REPEATS]
    totals["final_refit_smote_runs_range"] = [0, OUTER_N_SPLITS * OUTER_N_REPEATS]
    return {"stage_b_classifier_configuration_count": classifier_configuration_count, "stage_b_valid_candidate_count_per_inner_fold": stage_b_valid_per_inner, "stage_b_valid_candidate_count_per_outer_fold": stage_b_valid_per_inner * INNER_N_SPLITS, "per_outer_fold": per_outer, "all_outer_folds": totals}


def build_nested_search_space_manifest(data: Dict[str, Any], outer_manifest: Dict[str, Any], outer_manifest_sha256: str, inner_manifest: Dict[str, Any], inner_manifest_sha256: str) -> Dict[str, Any]:
    fingerprint = build_outer_data_fingerprint(data)
    features = _nested_feature_contract(data)
    representations = _nested_bottleneck_candidates(features)
    classifiers = _nested_classifier_registry()
    workload = _nested_search_workload(classifiers, representations)
    return {"schema_version": NESTED_SEARCH_SPACE_SCHEMA_VERSION, "bindings": {"data_fingerprint": fingerprint, "feature_contract": features, "outer_manifest_schema_version": OUTER_MANIFEST_SCHEMA_VERSION, "outer_manifest_sha256": outer_manifest_sha256, "inner_manifest_schema_version": INNER_MANIFEST_SCHEMA_VERSION, "inner_manifest_sha256": inner_manifest_sha256, "dependency_versions": _nested_dependency_versions()}, "original_method_reference": {"observed_original_code_behavior": "Dense(128) hidden layer followed by input-width reconstruction and full autoencoder predict output", "original_hidden_width_anchor": 128, "observed_output_is_not_exported_latent_width": True, "paper_intended_reduced_latent_ae_ctgan_method": "non_selectable_reference_pending_provenance", "corrected_named_bottleneck_method": "selectable_primary_method"}, "ae_contract": {"architecture": {"hidden_dims": [128], "bottleneck": {"named": True, "activation": "linear", "exported_width_source": "candidate latent_dim"}, "decoder": {"hidden_dims": [128], "activation": "relu"}, "reconstruction_activation": "linear"}, "training": {"optimizer": "adam", "loss": "mse", "max_epochs": 50, "batch_size": 32, "early_stopping_monitor": "val_loss", "patience": 10, "restore_best_weights": True, "shuffle": False, "deterministic_per_modality_seeds": True}, "inner_selection_early_stopping": {"validation_fraction": AE_EARLY_STOPPING_VALIDATION_FRACTION, "splitter": "StratifiedShuffleSplit", "random_state_base": AE_EARLY_STOPPING_RANDOM_STATE_BASE, "seed_derivation": "AE_EARLY_STOPPING_RANDOM_STATE_BASE + repeat_id * 1000 + fold_id * 10 + inner_fold_id", "labels_used_only_for": "stratification", "shared_across_modalities": True, "inner_validation_excluded_from_ae_fitting": True}, "final_outer_refit_early_stopping": {"validation_fraction": AE_EARLY_STOPPING_VALIDATION_FRACTION, "splitter": "StratifiedShuffleSplit", "random_state_base": AE_OUTER_REFIT_EARLY_STOPPING_RANDOM_STATE_BASE, "seed_derivation": "AE_OUTER_REFIT_EARLY_STOPPING_RANDOM_STATE_BASE + repeat_id * 100 + fold_id", "labels_used_only_for": "stratification", "outer_test_excluded_from_ae_fitting": True}}, "representation_candidates": representations, "classifier_registry": classifiers, "class_weight_support_matrix": {family: {"supported": value["class_weight_supported"], "supported_balanced_override": {"class_weight": "balanced"} if value["class_weight_supported"] else None, "unsupported_behavior": "record_invalid_never_emulate"} for family, value in classifiers.items()}, "imbalance_strategies": [{"strategy": "real_only_unweighted", "augmentation": "none", "class_weight_override": None}, {"strategy": "real_only_class_weight_balanced", "augmentation": "none", "class_weight_override": "balanced", "unsupported_combination_policy": "record_invalid_never_emulate"}, {"strategy": "smote_latent", "configuration": dict(SMOTE_LATENT_CONFIGURATION), "sequence": ["fit_augmentation_standard_scaler_on_real_inner_training_raw_fused_latents_only", "transform_real_inner_training_only", "run_smote_in_standardized_space", "identify_synthetic_suffix", "inverse_transform_synthetic_suffix_only", "return_real_raw_rows_then_synthetic_raw_rows"], "inner_validation_and_outer_test_excluded": True}, {"strategy": "minority_only_ctgan", "configuration": {"epochs": 300, "pac": 10, "verbose": False}}, {"strategy": "conditional_all_training_ctgan", "configuration": {"epochs": 300, "pac": 10, "verbose": False}}], "ctgan_contract": {"epochs": 300, "pac": 10, "verbose": False, "batch_size_rule": "floor(min(500, fit_row_count) / pac) * pac", "batch_size_requirements": ["batch_size >= pac", "batch_size % pac == 0"], "no_fallback": True, "no_bootstrap": True, "no_gaussian_noise_replacement": True, "no_random_oversampling": True, "constructor_level_exact_seed_claim": False, "raw_fused_latent_input_only": True}, "stage_a": {"name": "model_family_robust_representation_selection", "representation_candidates": [item["candidate_id"] for item in representations], "inner_fold_count": INNER_N_SPLITS, "classifier_configuration_ids": [value["configurations"][1]["configuration_id"] for value in classifiers.values()], "imbalance_strategy": "real_only_unweighted", "fold_level_metrics": ["mean_auprc_across_four_classifier_families", "mean_balanced_accuracy_across_four_classifier_families", "mean_sensitivity_high_tmb_across_four_classifier_families"], "aggregation": "aggregate_only_three_fold_level_summaries", "primary_metric": "auprc_high_tmb_class_1", "tie_break_order": ["highest_mean_auprc", "highest_mean_balanced_accuracy", "highest_mean_sensitivity_high_tmb", "lowest_auprc_standard_deviation_across_three_fold_summaries", "smaller_fused_latent_width", "canonical_candidate_id"]}, "stage_b": {"selected_stage_a_representation_only": True, "classifier_configuration_count": workload["stage_b_classifier_configuration_count"], "imbalance_strategy_count": 5, "inner_fold_count": INNER_N_SPLITS, "primary_metric": "auprc_high_tmb_class_1", "secondary_metrics": ["auroc", "balanced_accuracy", "sensitivity_high_tmb", "specificity_low_tmb", "precision_high_tmb", "f1_high_tmb", "accuracy"], "thresholds": {"predict_proba": 0.5, "decision_function": 0.0}, "probabilities_calibrated": False, "threshold_optimized": False, "undocumented_aggregate_score": False, "tie_break_order": ["highest_mean_auprc", "highest_mean_balanced_accuracy", "highest_mean_sensitivity_high_tmb", "lowest_auprc_standard_deviation", "canonical_classifier_configuration_id", "canonical_imbalance_strategy_id"], "candidate_failure_policy": "exclude_candidate_if_any_required_inner_fold_fails_retain_failure_evidence_no_fallback"}, "final_refit_contract": {"discard_all_inner_fitted_objects": True, "refit_preprocessing_on_complete_outer_training_only": True, "derive_outer_training_only_ae_early_stopping_split": True, "refit_three_selected_aes": True, "transform_complete_outer_training_and_untouched_outer_test": True, "rebuild_only_selected_imbalance_strategy_on_complete_outer_training": True, "fit_one_fresh_selected_classifier_pipeline": True, "evaluate_outer_test_exactly_once": True, "outer_test_influences": {"selection": False, "ae_epoch_choice": False, "augmentation": False, "classifier_fitting": False, "thresholds": False, "calibration": False}}, "runtime_device_policy": {"local_smoke_cpu_allowed": True, "final_colab_one_consistent_available_gpu_allowed": True, "device_is_runtime_evidence_not_candidate": True, "runtime_evidence": {"selected_device": "record_at_execution", "hardware_descriptor": "record_at_execution", "cuda_version": "record_at_execution", "pytorch_version": _nested_dependency_versions()["torch"], "sdv_version": _nested_dependency_versions()["sdv"], "ctgan_version": _nested_dependency_versions()["ctgan"]}, "ctgan_bit_identical_reproducibility_claim": False}, "workload_and_cache_contract": {"workload": workload, "representation_cache_key_fields": ["outer_identity", "inner_fold_identity", "inner_train_id_hash", "ae_fit_id_hash", "ae_early_stopping_id_hash", "preprocessing_config_hash", "representation_candidate_id", "ae_config_hash", "modality_feature_hashes", "dependency_runtime_contract"], "augmentation_cache_key_fields": ["representation_hash", "inner_train_id_hash", "inner_train_label_hash", "latent_schema_hash", "imbalance_strategy", "augmentation_configuration", "ctgan_runtime_device_evidence"], "classifier_cache_key_fields": ["training_variant_hash", "classifier_configuration_id", "scaling_policy_id", "score_contract_version"], "resumable_checkpoint_boundaries": ["preprocessing_and_representation", "smote_or_ctgan_generation", "classifier_inner_fold_result", "failure_ledger_row"], "failure_ledger_schema": ["stage", "candidate_id", "outer_identity", "inner_fold_id", "component", "exception_category", "message", "input_identity"]}}


_build_nested_search_space_manifest_base = build_nested_search_space_manifest


def build_nested_search_space_manifest(data: Dict[str, Any], outer_manifest: Dict[str, Any], outer_manifest_sha256: str, inner_manifest: Dict[str, Any], inner_manifest_sha256: str) -> Dict[str, Any]:
    manifest = _build_nested_search_space_manifest_base(data, outer_manifest, outer_manifest_sha256, inner_manifest, inner_manifest_sha256)
    candidate_ids = [candidate["candidate_id"] for candidate in manifest["representation_candidates"]]
    manifest["representation_method"] = {
        "id": "corrected_named_bottleneck_method",
        "candidate_origin": "author_prespecified_compression_ratios_for_new_study",
        "original_paper_dimensions_claimed": False,
        "original_code_dimensions_recovered": False,
        "selected_only_by_inner_validation": True,
        "outer_test_used_for_selection": False,
        "candidate_ids": candidate_ids,
    }
    manifest["original_method_reference"] = {
        "id": "observed_original_code_behavior",
        "selectable": False,
        "hidden_width": 128,
        "exported_representation_behavior": "autoencoder_reconstruction_output_at_input_width",
        "note": "128 is not treated as a verified AE bottleneck dimension",
    }
    return manifest


def _validate_representation_methodology_contract(manifest: Dict[str, Any], data: Dict[str, Any]) -> None:
    method = manifest.get("representation_method")
    if not isinstance(method, dict) or method.get("candidate_origin") != "author_prespecified_compression_ratios_for_new_study":
        raise ValueError("Nested representation candidate provenance is missing or invalid.")
    if method.get("id") != "corrected_named_bottleneck_method" or method.get("original_paper_dimensions_claimed") is not False or method.get("original_code_dimensions_recovered") is not False:
        raise ValueError("Nested representation methodology must not claim dimensions recovered from the original paper or code.")
    if method.get("selected_only_by_inner_validation") is not True or method.get("outer_test_used_for_selection") is not False:
        raise ValueError("Nested representation candidates must be selected only by inner validation without outer-test selection.")
    candidates = manifest.get("representation_candidates")
    if not isinstance(candidates, list):
        raise ValueError("Nested representation candidates are missing.")
    if method.get("candidate_ids") != [candidate.get("candidate_id") for candidate in candidates]:
        raise ValueError("Nested representation methodology candidate IDs do not match the candidate registry.")
    original = manifest.get("original_method_reference")
    if not isinstance(original, dict) or original.get("id") != "observed_original_code_behavior" or original.get("selectable") is not False or original.get("hidden_width") != 128 or original.get("exported_representation_behavior") != "autoencoder_reconstruction_output_at_input_width":
        raise ValueError("Nested original-method reference is missing or selectable.")
    expected_candidates = _nested_bottleneck_candidates(_nested_feature_contract(data))
    if candidates != expected_candidates:
        raise ValueError("Nested representation candidates are not derived from frozen Phase 1 modality feature counts.")
    for candidate in candidates:
        if candidate.get("original_paper_dimension") is True or "original" in str(candidate.get("origin", "")).casefold():
            raise ValueError("Nested representation candidates may not be labeled as original-paper dimensions.")
        for modality, latent_dim in candidate["latent_dimensions"].items():
            input_count = len(data["feature_columns"][FOLD_MODALITIES[modality]["feature_key"]])
            if not isinstance(latent_dim, int) or latent_dim < 2 or latent_dim >= input_count:
                raise ValueError("Nested representation candidate latent dimensions must be compressive and at least two.")


def validate_nested_search_space_manifest(manifest: Dict[str, Any], data: Dict[str, Any], outer_manifest: Dict[str, Any], outer_manifest_sha256: str, inner_manifest: Dict[str, Any], inner_manifest_sha256: str) -> Dict[str, Any]:
    fingerprint = build_outer_data_fingerprint(data)
    validate_outer_fold_manifest(outer_manifest, data["sample_ids"], data["y_binary"], fingerprint)
    validate_inner_fold_manifest(inner_manifest, outer_manifest, outer_manifest_sha256, data["sample_ids"], data["y_binary"], fingerprint)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != NESTED_SEARCH_SPACE_SCHEMA_VERSION:
        raise ValueError("Nested search-space manifest schema version does not match this protocol.")
    _validate_representation_methodology_contract(manifest, data)
    expected = build_nested_search_space_manifest(data, outer_manifest, outer_manifest_sha256, inner_manifest, inner_manifest_sha256)
    if manifest != expected:
        raise ValueError("Nested search-space manifest contents or bindings do not match the deterministic protocol rebuild.")
    workload = manifest["workload_and_cache_contract"]["workload"]
    recomputed = _nested_search_workload(manifest["classifier_registry"], manifest["representation_candidates"])
    if workload != recomputed:
        raise ValueError("Nested search-space manifest workload does not recompute from its registries.")
    return {"passed": True, "representation_candidate_count": len(manifest["representation_candidates"]), "classifier_configuration_count": sum(len(value["configurations"]) for value in manifest["classifier_registry"].values()), "stage_a_evaluation_count_per_outer_fold": workload["per_outer_fold"]["stage_a_classifier_fits"], "stage_b_evaluation_count_per_outer_fold": workload["per_outer_fold"]["stage_b_classifier_fits"], "deterministic_rebuild_matches": True}


def write_nested_search_space_manifest(manifest: Dict[str, Any], path: Path) -> Dict[str, Any]:
    if not path.parent.exists():
        raise FileNotFoundError(f"Manifest parent directory does not exist: {path.parent}")
    payload = _canonical_json_bytes(manifest)
    path.write_bytes(payload)
    return {"manifest_path": str(path), "manifest_sha256": hashlib.sha256(payload).hexdigest(), "manifest_size_bytes": len(payload)}


def load_nested_search_space_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Nested search-space manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Nested search-space manifest is not valid JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Nested search-space manifest must be a JSON object.")
    return manifest


# =========================================
# SECTION 2L: Phase 10C nested-selection integration smoke
# =========================================


PHASE10C_SCHEMA_VERSION = "nested-selection-integration-smoke-v1"
PHASE10C_GUARDS = {
    "research_eligible": False,
    "scientific_results_export_allowed": False,
    "smoke_runtime_overrides_applied": True,
    "hyperparameters_tuned_for_research": False,
    "probabilities_calibrated": False,
    "threshold_optimized": False,
    "final_method_selected": False,
    "paper_result": False,
}
PHASE10C_LOW_MARGIN_PERCENTILE = 0.10
PHASE10C_OOD_PERCENTILE = 0.95
PHASE10C_RELIABILITY_SCHEMA_VERSION = "phase10c-reliability-v1"
PHASE10C_FULL_GUARDS = {
    "research_eligible": True,
    "scientific_results_export_allowed": True,
    "smoke_runtime_overrides_applied": False,
    "hyperparameters_tuned_for_research": True,
    "probabilities_calibrated": False,
    "threshold_optimized": False,
    "final_method_selected": False,
    "paper_result": False,
}
PHASE10C_ACTIVE_GUARDS: ContextVar[Dict[str, Any]] = ContextVar("phase10c_active_guards", default=PHASE10C_GUARDS)


def _phase10c_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _phase10c_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_phase10c_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _phase10c_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _phase10c_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(_phase10c_jsonable(payload))).hexdigest()


def _phase10c_atomic_write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    encoded = _canonical_json_bytes(_phase10c_jsonable(payload))
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _phase10c_create_immutable_json(path: Path, payload: Dict[str, Any]) -> str:
    """Create an evidence file exactly once; a conflicting file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(_phase10c_jsonable(payload))
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise FileExistsError(f"Immutable Phase 10C evidence already exists: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # A partial immutable state is evidence.  It is intentionally retained.
        raise
    return hashlib.sha256(encoded).hexdigest()


def _phase10c_read_json(path: Path, context: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object: {path}")
    return value


def _phase10c_read_json_record_stream(path: Path, expected_record_count: Optional[int] = None, unique_key: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Read concatenated JSON objects, supporting legacy pretty records and compact JSONL."""
    if not path.is_file():
        raise FileNotFoundError(f"Phase 10C JSON record stream does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    decoder, offset, records = json.JSONDecoder(), 0, []
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        try:
            value, offset = decoder.raw_decode(text, offset)
        except json.JSONDecodeError as error:
            raise ValueError(f"Phase 10C JSON record stream is malformed: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Phase 10C JSON record stream values must be objects: {path}")
        records.append(value)
    if not records:
        raise ValueError(f"Phase 10C JSON record stream is empty: {path}")
    if expected_record_count is not None and len(records) != expected_record_count:
        raise ValueError(f"Phase 10C JSON record stream count differs from expectation: {path}")
    if unique_key is not None:
        keys = [unique_key(record) for record in records]
        if any(key is None for key in keys) or len(keys) != len(set(keys)):
            raise ValueError(f"Phase 10C JSON record stream contains duplicate or unexpected records: {path}")
    return records


def _phase10c_write_compact_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("Phase 10C compact JSONL requires one or more object records.")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_phase10c_jsonable(record), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _phase10c_publish_directory(target: Path, writer: Any) -> None:
    """Publish a complete directory as one rename; never replace a prior publication."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Phase 10C publication already exists: {target}")
    temporary = target.parent / ("." + target.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        writer(temporary)
        os.rename(temporary, target)
    except Exception:
        # Retain partial evidence.  It can never be mistaken for a publication.
        raise


def _phase10c_runtime_evidence() -> Dict[str, Any]:
    devices = []
    try:
        devices = [str(item) for item in tf.config.list_physical_devices()]
    except Exception:
        devices = ["unavailable"]
    return {"tensorflow_devices": devices, "dependency_versions": _nested_dependency_versions()}


def build_phase10c_execution_profile_contract(
    outer_sha256: str, inner_sha256: str, search_sha256: str, repeat_id: int, fold_id: int
) -> Dict[str, Any]:
    return {
        "schema_version": PHASE10C_SCHEMA_VERSION,
        "execution_profile": "integration_smoke",
        "selected_outer_fold": {"repeat_id": int(repeat_id), "fold_id": int(fold_id)},
        "bindings": {
            "outer_manifest_sha256": outer_sha256,
            "inner_manifest_sha256": inner_sha256,
            "search_space_manifest_sha256": search_sha256,
        },
        "scientific_settings": {"ae_max_epochs": 50, "ae_patience": 10, "ctgan_epochs": 300},
        "active_smoke_settings": {"ae_max_epochs_override": 2, "ae_patience_override": 1, "ctgan_epochs_override": 1},
        "active_training_settings": {"ae_max_epochs": 2, "ae_patience": 1, "ctgan_epochs": 1},
        **PHASE10C_GUARDS,
    }


def build_phase10c_full_execution_profile_contract(
    search: Dict[str, Any], outer_sha256: str, inner_sha256: str, search_sha256: str
) -> Dict[str, Any]:
    training = search["ae_contract"]["training"]
    ctgan = search["ctgan_contract"]
    if training["max_epochs"] != 50 or training["patience"] != 10 or ctgan["epochs"] != 300 or ctgan["pac"] != 10:
        raise ValueError("Frozen nested search-space scientific settings do not match the required full-run protocol.")
    return {
        "schema_version": PHASE10C_SCHEMA_VERSION,
        "execution_profile": "full_nested_scientific_run",
        "outer_fold_count": OUTER_N_SPLITS * OUTER_N_REPEATS,
        "bindings": {"outer_manifest_sha256": outer_sha256, "inner_manifest_sha256": inner_sha256, "search_space_manifest_sha256": search_sha256},
        "scientific_settings": {"ae_max_epochs": training["max_epochs"], "ae_patience": training["patience"], "ctgan_epochs": ctgan["epochs"], "ctgan_pac": ctgan["pac"], "ctgan_batch_size_rule": ctgan["batch_size_rule"]},
        "active_training_settings": {"ae_max_epochs": training["max_epochs"], "ae_patience": training["patience"], "ctgan_epochs": ctgan["epochs"]},
        **PHASE10C_FULL_GUARDS,
    }


def _phase10c_lock_binding(profile_hash: str, outer_sha256: str, inner_sha256: str, search_sha256: str, repeat_id: int, fold_id: int) -> Dict[str, Any]:
    return {"outer_manifest_sha256": outer_sha256, "inner_manifest_sha256": inner_sha256, "search_space_manifest_sha256": search_sha256, "repeat_id": int(repeat_id), "fold_id": int(fold_id), "execution_profile_hash": profile_hash}


def _phase10c_pid_is_active(owner: Dict[str, Any]) -> bool:
    if owner.get("hostname") != socket.gethostname():
        raise RuntimeError("Phase 10C lock belongs to a different host and cannot be proven stale.")
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("Phase 10C lock owner PID is invalid and cannot be recovered safely.")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _phase10c_recover_stale_lock(lock_path: Path, expected_binding: Dict[str, Any]) -> None:
    guard = lock_path.with_name(".phase10c_lock_recovery_guard")
    try:
        descriptor = os.open(str(guard), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError("Phase 10C stale-lock recovery is already in progress.") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes({"owner": {"hostname": socket.gethostname(), "pid": os.getpid()}}))
            handle.flush(); os.fsync(handle.fileno())
        lock = _phase10c_read_json(lock_path, "Phase 10C active lock")
        if lock.get("binding") != expected_binding:
            raise RuntimeError("Phase 10C output directory has a stale lock for a different run binding.")
        if _phase10c_pid_is_active(lock.get("owner", {})):
            raise RuntimeError("Phase 10C lock is actively owned and cannot be removed.")
        recovered = lock_path.parent / "lock_lifecycle" / "recovered" / (str(lock.get("lock_id", "unknown")) + ".json")
        recovered.parent.mkdir(parents=True, exist_ok=True)
        if recovered.exists():
            raise RuntimeError("Phase 10C stale-lock recovery evidence already exists.")
        os.rename(lock_path, recovered)
    finally:
        if guard.exists():
            try:
                guard.unlink()
            except OSError:
                pass


def acquire_phase10c_run_lock(output_dir: Path, binding: Dict[str, Any]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".phase10c_run_lock.json"
    lock_id = uuid.uuid4().hex
    payload = {"schema_version": "phase10c-run-lock-v1", "lock_id": lock_id, "binding": binding, "owner": {"hostname": socket.gethostname(), "pid": os.getpid(), "process_identity": f"{socket.gethostname()}:{os.getpid()}"}, "lifecycle": {"acquired_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, **PHASE10C_ACTIVE_GUARDS.get()}
    try:
        _phase10c_create_immutable_json(lock_path, payload)
    except FileExistsError:
        existing = _phase10c_read_json(lock_path, "Phase 10C active lock")
        try:
            active = _phase10c_pid_is_active(existing.get("owner", {}))
        except RuntimeError:
            raise RuntimeError(f"Phase 10C output directory is locked: {existing.get('binding')}")
        if active:
            raise RuntimeError(f"Phase 10C output directory is locked: {existing.get('binding')}")
        _phase10c_recover_stale_lock(lock_path, binding)
        _phase10c_create_immutable_json(lock_path, payload)
    _phase10c_atomic_write_json(output_dir / "lock_lifecycle" / "acquired" / f"{lock_id}.json", payload)
    return {"path": lock_path, "lock_id": lock_id, "binding": binding}


def release_phase10c_run_lock(lock: Dict[str, Any], outcome: str) -> None:
    path = lock["path"]
    if not path.exists():
        return
    payload = _phase10c_read_json(path, "Phase 10C run lock")
    if payload.get("lock_id") != lock["lock_id"]:
        raise RuntimeError("Refusing to release a Phase 10C lock owned by another process.")
    _phase10c_atomic_write_json(path.parent / "lock_lifecycle" / "released" / f"{lock['lock_id']}.json", {"lock_id": lock["lock_id"], "binding": lock["binding"], "outcome": outcome, "released_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **PHASE10C_ACTIVE_GUARDS.get()})
    path.unlink()


def _phase10c_record_failure(output_dir: Path, event: Dict[str, Any]) -> None:
    identity = {key: event.get(key) for key in ["stage", "candidate_id", "inner_fold_id", "component", "exception_category", "input_hashes"]}
    event = {**event, "event_id": _phase10c_hash(identity), **PHASE10C_ACTIVE_GUARDS.get()}
    event_path = output_dir / "failure_events" / (event["event_id"] + ".json")
    if not event_path.exists():
        _phase10c_create_immutable_json(event_path, event)
        with (output_dir / "failure_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json_bytes(event).decode("utf-8") + "\n")
            handle.flush(); os.fsync(handle.fileno())


def _phase10c_inner_records(inner_manifest: Dict[str, Any], repeat_id: int, fold_id: int) -> List[Dict[str, Any]]:
    matches = [record for record in inner_manifest["outer_folds"] if record.get("repeat_id") == repeat_id and record.get("fold_id") == fold_id]
    if len(matches) != 1:
        raise ValueError("Selected Phase 10C outer fold must have exactly one inner-fold record.")
    records = sorted(matches[0].get("inner_folds", []), key=lambda item: item.get("inner_fold_id"))
    if [item.get("inner_fold_id") for item in records] != list(range(INNER_N_SPLITS)):
        raise ValueError("Selected Phase 10C fold must contain exactly the canonical three inner folds.")
    return records


def _phase10c_materialize_partition(data: Dict[str, Any], train_ids: List[str], validation_ids: List[str]) -> Dict[str, Any]:
    labels = pd.Series(data["y_binary"], index=pd.Index(data["sample_ids"], name="SAMPLE_ID"))
    if set(train_ids).intersection(validation_ids):
        raise ValueError("Phase 10C partition train/validation IDs overlap.")
    modalities = {}
    for modality, keys in FOLD_MODALITIES.items():
        matrix, features = data[keys["matrix_key"]], list(data["feature_columns"][keys["feature_key"]])
        train_df, validation_df = matrix.loc[train_ids, features].copy(), matrix.loc[validation_ids, features].copy()
        if train_df.index.tolist() != train_ids or validation_df.index.tolist() != validation_ids:
            raise AssertionError("Phase 10C partition did not preserve manifest SAMPLE_ID order.")
        modalities[modality] = {"train_df": train_df, "validation_df": validation_df, "feature_names": features, "feature_name_sha256": _feature_name_sha256(features)}
    _assert_target_leakage_guards({name: value["feature_names"] for name, value in modalities.items()}, {f"{name}_train": value["train_df"] for name, value in modalities.items()})
    return {"train_ids": list(train_ids), "validation_ids": list(validation_ids), "y_train": labels.loc[train_ids].to_numpy(dtype=int), "y_validation": labels.loc[validation_ids].to_numpy(dtype=int), "modalities": modalities}


def _phase10c_preprocess_partition(partition: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for modality, value in partition["modalities"].items():
        if value["train_df"].isna().all().any():
            raise ValueError(f"{modality} Phase 10C inner training has entirely missing features.")
        preprocessor = build_preprocessor()
        train = preprocessor.fit_transform(value["train_df"]).astype(np.float32)
        validation = preprocessor.transform(value["validation_df"]).astype(np.float32)
        if not np.isfinite(train).all() or not np.isfinite(validation).all():
            raise ValueError("Phase 10C preprocessing produced non-finite values.")
        result[modality] = {"train": train, "validation": validation, "preprocessor": preprocessor, "feature_names": value["feature_names"], "feature_name_sha256": value["feature_name_sha256"], "fit_ids": list(partition["train_ids"])}
    return result


def _phase10c_ae_split(train_ids: List[str], labels: np.ndarray, seed: int) -> Dict[str, Any]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=AE_EARLY_STOPPING_VALIDATION_FRACTION, random_state=seed)
    fit_index, stop_index = next(splitter.split(np.zeros(len(labels)), labels))
    ordered_fit = np.random.default_rng(seed).permutation(fit_index)
    fit_index = np.asarray(ordered_fit, dtype=int)
    return {"fit_indices": fit_index, "stop_indices": np.asarray(stop_index, dtype=int), "fit_ids": [train_ids[index] for index in fit_index], "stop_ids": [train_ids[index] for index in stop_index], "seed": seed}


def _phase10c_fit_final_refit_modality(
    matrices: Dict[str, Any],
    split: Dict[str, Any],
    modality: str,
    architecture: Dict[str, Any],
    training: Dict[str, Any],
    model_seed: int,
    factories: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select an epoch on outer-train only, then refit fresh weights on all outer-train rows."""
    resolved = {} if factories is None else factories
    epoch_selector = resolved.get("epoch_selection_fit_factory", fit_fold_modality_autoencoder)
    temporary = epoch_selector(
        matrices["train"][split["fit_indices"]],
        matrices["train"][split["stop_indices"]],
        modality,
        architecture,
        training,
        model_seed,
    )
    temporary_metadata = temporary.get("metadata") if isinstance(temporary, dict) else None
    history = temporary_metadata.get("validation_loss_history") if isinstance(temporary_metadata, dict) else None
    if not isinstance(history, list) or not history:
        raise ValueError(f"{modality} final-refit epoch selection has no validation-loss history.")
    try:
        validation_losses = np.asarray(history, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{modality} final-refit validation-loss history is malformed.") from error
    if validation_losses.ndim != 1 or not np.isfinite(validation_losses).all():
        raise ValueError(f"{modality} final-refit validation-loss history must be finite.")
    selected_epoch_count = int(np.argmin(validation_losses) + 1)
    if not 1 <= selected_epoch_count <= training["epochs"]:
        raise ValueError(f"{modality} final-refit selected epoch is outside the configured budget.")

    seed_setter = resolved.get("seed_setter", _set_fold_autoencoder_seed)
    model_builder = resolved.get("final_model_builder", build_fold_autoencoder)
    seed_setter(model_seed)
    final_autoencoder, final_encoder = model_builder(
        modality,
        matrices["train"].shape[1],
        architecture["hidden_dims"],
        architecture["latent_dim"],
    )
    if final_autoencoder is temporary.get("autoencoder") or final_encoder is temporary.get("encoder"):
        raise AssertionError("Phase 10C final refit must use fresh Autoencoder and encoder instances.")
    del temporary
    final_autoencoder.fit(
        matrices["train"],
        matrices["train"],
        epochs=selected_epoch_count,
        batch_size=min(training["batch_size"], len(matrices["train"])),
        shuffle=False,
        verbose=0,
    )
    outer_train_latent = np.asarray(final_encoder.predict(matrices["train"], verbose=0), dtype=np.float32)
    outer_test_latent = np.asarray(final_encoder.predict(matrices["validation"], verbose=0), dtype=np.float32)
    if not np.isfinite(outer_train_latent).all() or not np.isfinite(outer_test_latent).all():
        raise ValueError(f"{modality} final-refit encoder produced non-finite latents.")
    expected_shape = (len(matrices["train"]), architecture["latent_dim"])
    if outer_train_latent.shape != expected_shape or outer_test_latent.shape != (len(matrices["validation"]), architecture["latent_dim"]):
        raise AssertionError(f"{modality} final-refit latent shape is invalid.")
    evidence = {
        **temporary_metadata,
        "epoch_selection": {
            "fit_sample_count": len(split["fit_ids"]),
            "fit_sample_ids_sha256": _sample_id_list_sha256(split["fit_ids"]),
            "early_stopping_sample_count": len(split["stop_ids"]),
            "early_stopping_sample_ids_sha256": _sample_id_list_sha256(split["stop_ids"]),
            "selected_epoch_count": selected_epoch_count,
            "best_validation_loss": float(validation_losses[selected_epoch_count - 1]),
            "temporary_model_used_only_for_epoch_selection": True,
        },
        "full_outer_training_refit": {
            "sample_count": len(matrices["train"]),
            "sample_ids_sha256": _sample_id_list_sha256(matrices["fit_ids"]),
            "matches_complete_outer_training_ids": True,
            "epochs": selected_epoch_count,
            "validation_data_used": False,
            "early_stopping_callback_used": False,
            "outer_test_supplied_to_fit": False,
            "fresh_model_distinct_from_epoch_selection_model": True,
        },
    }
    return {"outer_train_latent": outer_train_latent, "outer_test_latent": outer_test_latent, "metadata": evidence}


def _phase10c_build_representation(partition: Dict[str, Any], candidate: Dict[str, Any], repeat_id: int, fold_id: int, inner_fold_id: int, profile: Dict[str, Any], final_refit: bool = False, final_refit_factories: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    preprocessed = _phase10c_preprocess_partition(partition)
    split_seed = 6200 + repeat_id * 100 + fold_id if final_refit else 5200 + repeat_id * 1000 + fold_id * 10 + inner_fold_id
    split = _phase10c_ae_split(partition["train_ids"], partition["y_train"], split_seed)
    latents_train, latents_validation, evidence = {}, {}, {}
    training = {"epochs": profile["active_training_settings"]["ae_max_epochs"], "batch_size": 32, "patience": profile["active_training_settings"]["ae_patience"], "seed": (8000 + repeat_id * 100 + fold_id) if final_refit else (7000 + repeat_id * 1000 + fold_id * 10 + inner_fold_id)}
    for modality_index, modality in enumerate(FUSION_MODALITY_ORDER):
        matrices = preprocessed[modality]
        architecture = {"hidden_dims": [128], "latent_dim": candidate["latent_dimensions"][modality]}
        model_seed = training["seed"] + modality_index
        if final_refit:
            matrices = {**matrices, "fit_ids": list(partition["train_ids"])}
            fitted = _phase10c_fit_final_refit_modality(matrices, split, modality, architecture, training, model_seed, final_refit_factories)
            latents_train[modality] = fitted["outer_train_latent"]
            latents_validation[modality] = fitted["outer_test_latent"]
            evidence[modality] = fitted["metadata"]
        else:
            fitted = fit_fold_modality_autoencoder(matrices["train"][split["fit_indices"]], matrices["train"][split["stop_indices"]], modality, architecture, training, model_seed)
            latents_train[modality] = np.asarray(fitted["encoder"].predict(matrices["train"], verbose=0), dtype=np.float32)
            latents_validation[modality] = np.asarray(fitted["encoder"].predict(matrices["validation"], verbose=0), dtype=np.float32)
            evidence[modality] = fitted["metadata"]
    train = np.concatenate([latents_train[item] for item in FUSION_MODALITY_ORDER], axis=1).astype(np.float32)
    validation = np.concatenate([latents_validation[item] for item in FUSION_MODALITY_ORDER], axis=1).astype(np.float32)
    names = build_fold_latent_feature_names(candidate["latent_dimensions"])
    return {"train": train, "validation": validation, "y_train": partition["y_train"].copy(), "y_validation": partition["y_validation"].copy(), "train_ids": list(partition["train_ids"]), "validation_ids": list(partition["validation_ids"]), "feature_names": names, "feature_name_sha256": _feature_name_sha256(names), "ae_split": {"fit_ids": split["fit_ids"], "stop_ids": split["stop_ids"], "seed": split["seed"]}, "preprocessing_evidence": {name: {"fit_ids_sha256": _sample_id_list_sha256(value["fit_ids"]), "feature_name_sha256": value["feature_name_sha256"]} for name, value in preprocessed.items()}, "ae_evidence": evidence}


def _phase10c_representation_key(partition: Dict[str, Any], candidate: Dict[str, Any], profile_hash: str, repeat_id: int, fold_id: int, inner_fold_id: int) -> str:
    return _phase10c_hash({"outer_identity": [repeat_id, fold_id], "inner_fold_id": inner_fold_id, "train_ids": _sample_id_list_sha256(partition["train_ids"]), "validation_ids": _sample_id_list_sha256(partition["validation_ids"]), "candidate_id": candidate["candidate_id"], "candidate": candidate, "profile_hash": profile_hash, "dependency_runtime_contract": _nested_dependency_versions()})


def _phase10c_write_representation_cache(path: Path, key: str, value: Dict[str, Any]) -> None:
    def writer(directory: Path) -> None:
        np.savez_compressed(directory / "matrices.npz", train=value["train"], validation=value["validation"], y_train=value["y_train"], y_validation=value["y_validation"])
        metadata = {key: item for key, item in value.items() if key not in {"train", "validation", "y_train", "y_validation"}}
        _phase10c_atomic_write_json(directory / "metadata.json", {"cache_key": key, "payload_hashes": {"train": _array_sha256(value["train"]), "validation": _array_sha256(value["validation"]), "labels": _array_sha256(value["y_train"])}, "metadata": metadata, "completion": True, **PHASE10C_ACTIVE_GUARDS.get()})
    _phase10c_publish_directory(path, writer)


def _phase10c_load_representation_cache(path: Path, key: str) -> Optional[Dict[str, Any]]:
    try:
        metadata = _phase10c_read_json(path / "metadata.json", "Phase 10C representation cache")
        if metadata.get("cache_key") != key or not metadata.get("completion"):
            return None
        with np.load(path / "matrices.npz") as matrices:
            value = {"train": matrices["train"].astype(np.float32), "validation": matrices["validation"].astype(np.float32), "y_train": matrices["y_train"].astype(int), "y_validation": matrices["y_validation"].astype(int), **metadata["metadata"]}
        hashes = metadata["payload_hashes"]
        if _array_sha256(value["train"]) != hashes["train"] or _array_sha256(value["validation"]) != hashes["validation"] or _array_sha256(value["y_train"]) != hashes["labels"]:
            return None
        return value
    except (FileNotFoundError, ValueError, KeyError, OSError):
        return None


def _phase10c_fit_classifier(training: Dict[str, Any], family: str, configuration: Dict[str, Any], class_weight_override: Any = None) -> Dict[str, Any]:
    if family not in PHASE9B_PIPELINE_REGISTRY:
        raise ValueError("Phase 10C classifier family is not approved.")
    config = dict(configuration)
    if class_weight_override is not None:
        config["class_weight"] = class_weight_override
    spec = {**PHASE9B_PIPELINE_REGISTRY[family], "configuration": config}
    # The Phase 9B public helper intentionally accepts only its frozen baseline config.
    features = _validate_logistic_feature_matrix("Phase 10C classifier training", training["features"], len(training["feature_names"]))
    labels = _validate_logistic_binary_labels("Phase 10C classifier labels", training["labels"], len(features), True)
    classifier = spec["factory"](**config)
    estimator = Pipeline([("standard_scaler", StandardScaler(**PHASE9B_STANDARD_SCALER_CONFIGURATION)), ("classifier", classifier)]) if spec["post_fusion_scaling_enabled"] else classifier
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning); estimator.fit(features, labels)
    terminal = estimator.named_steps["classifier"] if isinstance(estimator, Pipeline) else estimator
    classes = _validate_logistic_estimator_classes(terminal)
    effective = {name: terminal.get_params(deep=False).get(name) for name in config}
    if effective != config:
        raise ValueError("Phase 10C effective classifier configuration differs from manifest.")
    return {"estimator": estimator, "classes": classes, "score_source": spec["score_source"], "effective_configuration": effective, "post_fusion_scaling_enabled": spec["post_fusion_scaling_enabled"], "convergence": _phase9b_convergence(family, terminal, config, [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)])}


def _phase10c_evaluate_classifier(fitted: Dict[str, Any], features: np.ndarray, labels: np.ndarray, ids: List[str]) -> Dict[str, Any]:
    adapted = _phase9a_score(fitted, features)
    scores = np.asarray(adapted["scores"])
    predicted = (scores >= adapted["threshold_value"]).astype(int)
    both = set(labels.tolist()) == {0, 1}
    matrix = confusion_matrix(labels, predicted, labels=[0, 1]); tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    records = [{"SAMPLE_ID": sample_id, "true_binary_label": int(label), "predicted_binary_label": int(prediction), "high_tmb_continuous_score": float(score), "score_type": adapted["score_type"]} for sample_id, label, prediction, score in zip(ids, labels, predicted, scores)]
    return {"metrics": {"accuracy": float(accuracy_score(labels, predicted)), "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)) if both else None, "sensitivity_high_tmb": float(tp / (tp + fn)) if tp + fn else None, "specificity_low_tmb": float(tn / (tn + fp)) if tn + fp else None, "precision_high_tmb": float(precision_score(labels, predicted, zero_division=0)), "f1_high_tmb": float(f1_score(labels, predicted, zero_division=0)), "auroc": float(roc_auc_score(labels, scores)) if both else None, "auprc": float(average_precision_score(labels, scores)) if both else None}, "scores": scores, "predicted": predicted, "records": records, "prediction_record_sha256": _phase10c_hash(records), "score_type": adapted["score_type"], "threshold": adapted["threshold_value"], "confusion_matrix": matrix.tolist()}


def _phase10c_extract_terminal_classifier_and_scaler(estimator: Any) -> Tuple[Any, Optional[Any]]:
    if isinstance(estimator, Pipeline):
        if list(estimator.named_steps) != ["standard_scaler", "classifier"]:
            raise ValueError("Phase 10C explanation pipeline steps are not approved.")
        return estimator.named_steps["classifier"], estimator.named_steps["standard_scaler"]
    return estimator, None


def _phase10c_feature_modalities(feature_names: List[str], modality_slices: Dict[str, Tuple[int, int]]) -> List[str]:
    _validate_fold_latent_slices(modality_slices, {modality: end - start for modality, (start, end) in modality_slices.items()}, len(feature_names))
    modalities = []
    for modality in FUSION_MODALITY_ORDER:
        start, end = modality_slices[modality]
        modalities.extend([modality] * (end - start))
    if len(modalities) != len(feature_names):
        raise ValueError("Phase 10C explanation modality slices do not cover fused features.")
    return modalities


def _phase10c_aggregate_feature_values(values: np.ndarray, modalities: List[str]) -> Dict[str, Dict[str, float]]:
    return {
        modality: {
            "signed_total": float(np.sum(values[np.asarray([item == modality for item in modalities])])),
            "absolute_total": float(np.sum(np.abs(values[np.asarray([item == modality for item in modalities])]))),
        }
        for modality in FUSION_MODALITY_ORDER
    }


def _phase10c_build_global_training_reference_importance(
    fitted_estimator: Any,
    outer_train_features: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    modality_slices: Dict[str, Tuple[int, int]],
    permutation_importance_factory: Any = permutation_importance,
) -> Dict[str, Any]:
    result = permutation_importance_factory(
        fitted_estimator,
        outer_train_features,
        y_train,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    means, stds = np.asarray(result.importances_mean, dtype=float), np.asarray(result.importances_std, dtype=float)
    if means.shape != (len(feature_names),) or stds.shape != means.shape or not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Phase 10C training-reference permutation importance is invalid.")
    modalities = _phase10c_feature_modalities(feature_names, modality_slices)
    order = sorted(range(len(feature_names)), key=lambda index: (-abs(float(means[index])), feature_names[index]))
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    features = [{"feature_name": feature_names[index], "modality": modalities[index], "importance_mean": float(means[index]), "importance_std": float(stds[index]), "absolute_importance": float(abs(means[index])), "rank": ranks[index]} for index in range(len(feature_names))]
    return {"method": "permutation_importance", "scoring": "average_precision", "n_repeats": 5, "random_state": 42, "n_jobs": 1, "reference_scope": "complete_real_outer_training_only", "outer_test_labels_used": False, "outer_test_prediction_calls": 0, "performance_claim": False, "features": features, "modality_aggregates": _phase10c_aggregate_feature_values(means, modalities)}


def _phase10c_build_logistic_latent_contributions(
    terminal_classifier: Any,
    scaler: Any,
    outer_train_features: np.ndarray,
    outer_test_features: np.ndarray,
    outer_test_ids: List[str],
    feature_names: List[str],
    modality_slices: Dict[str, Tuple[int, int]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if scaler is None or not all(hasattr(terminal_classifier, attribute) for attribute in ["classes_", "coef_", "intercept_"]):
        raise ValueError("Phase 10C logistic local explanation requires the fitted Phase 9B scaler and LogisticRegression.")
    classes = _validate_logistic_estimator_classes(terminal_classifier)
    coefficient = np.asarray(terminal_classifier.coef_, dtype=float)
    intercept = np.asarray(terminal_classifier.intercept_, dtype=float)
    if coefficient.shape != (1, len(feature_names)) or intercept.shape != (1,):
        raise ValueError("Phase 10C logistic coefficient shape is invalid.")
    orientation = 1.0 if int(classes[1]) == 1 else -1.0
    oriented_coefficient, oriented_intercept = orientation * coefficient[0], float(orientation * intercept[0])
    background_raw = np.median(outer_train_features, axis=0, keepdims=True)
    background = np.asarray(scaler.transform(background_raw), dtype=float)
    transformed_patients = np.asarray(scaler.transform(outer_test_features), dtype=float)
    if background.shape != (1, len(feature_names)) or transformed_patients.shape != outer_test_features.shape or not np.isfinite(background).all() or not np.isfinite(transformed_patients).all():
        raise ValueError("Phase 10C logistic explanation scaler output is invalid.")
    background_decision = oriented_intercept + float(np.dot(oriented_coefficient, background[0]))
    modalities = _phase10c_feature_modalities(feature_names, modality_slices)
    patients = []
    for sample_id, raw_values, transformed_values in zip(outer_test_ids, outer_test_features, transformed_patients):
        contributions = oriented_coefficient * (transformed_values - background[0])
        reconstructed = background_decision + float(np.sum(contributions))
        algebraic = oriented_intercept + float(np.dot(oriented_coefficient, transformed_values))
        if not np.isclose(reconstructed, algebraic, rtol=0.0, atol=1e-10):
            raise AssertionError("Phase 10C logistic contribution reconstruction is not exact.")
        positive = sorted((index for index in range(len(feature_names)) if contributions[index] > 0), key=lambda index: (-float(contributions[index]), feature_names[index]))[:top_k]
        negative = sorted((index for index in range(len(feature_names)) if contributions[index] < 0), key=lambda index: (float(contributions[index]), feature_names[index]))[:top_k]
        format_item = lambda index: {"feature_name": feature_names[index], "modality": modalities[index], "contribution": float(contributions[index])}
        patients.append({"SAMPLE_ID": sample_id, "fused_latent_values": [float(value) for value in raw_values], "local_explanation_available": True, "local_explanation_method": "native_logistic_coefficient_times_standardized_delta_from_training_median", "contribution_scale": "class_1_logit_decision_scale", "probability_contribution": False, "background_decision_value": background_decision, "reconstructed_class_1_decision_value": reconstructed, "top_positive_latent_contributions": [format_item(index) for index in positive], "top_negative_latent_contributions": [format_item(index) for index in negative], "modality_contribution_totals": _phase10c_aggregate_feature_values(contributions, modalities)})
    return patients


def _phase10c_build_selected_model_explanation_core(
    classifier_family: str,
    classifier_configuration_id: str,
    fitted_estimator: Any,
    outer_train_features: np.ndarray,
    y_train: np.ndarray,
    outer_train_ids: List[str],
    outer_test_features: np.ndarray,
    outer_test_ids: List[str],
    feature_names: List[str],
    feature_name_sha256: str,
    modality_slices: Dict[str, Tuple[int, int]],
    permutation_importance_factory: Any = permutation_importance,
) -> Dict[str, Any]:
    if classifier_family not in PHASE9B_PIPELINE_REGISTRY or len(outer_train_features) != len(y_train) or len(outer_train_ids) != len(outer_train_features) or len(outer_test_ids) != len(outer_test_features):
        raise ValueError("Phase 10C selected-model explanation inputs are invalid.")
    if _feature_name_sha256(feature_names) != feature_name_sha256 or not np.isfinite(outer_train_features).all() or not np.isfinite(outer_test_features).all():
        raise ValueError("Phase 10C selected-model explanation latent schema is invalid.")
    terminal, scaler = _phase10c_extract_terminal_classifier_and_scaler(fitted_estimator)
    global_importance = _phase10c_build_global_training_reference_importance(fitted_estimator, outer_train_features, y_train, feature_names, modality_slices, permutation_importance_factory)
    limitations, native_global_importance = [], {}
    if classifier_family == "logistic_linear":
        patients = _phase10c_build_logistic_latent_contributions(terminal, scaler, outer_train_features, outer_test_features, outer_test_ids, feature_names, modality_slices)
    else:
        messages = {"rbf_svc_decision": "RBF SVC has no native direct per-latent-feature contribution representation.", "random_forest_bagged": "Random forest native feature importance is global only; no native local contribution is claimed.", "hist_gradient_boosting": "HistGradientBoosting exposes no supported native direct per-latent-feature local explanation."}
        limitations.append(messages[classifier_family])
        patients = [{"SAMPLE_ID": sample_id, "fused_latent_values": [float(value) for value in values], "local_explanation_available": False, "local_explanation_method": None, "local_explanation_limitation": messages[classifier_family]} for sample_id, values in zip(outer_test_ids, outer_test_features)]
        if classifier_family == "random_forest_bagged" and hasattr(terminal, "feature_importances_"):
            values = np.asarray(terminal.feature_importances_, dtype=float)
            if values.shape != (len(feature_names),) or not np.isfinite(values).all():
                raise ValueError("Phase 10C random-forest native importance is invalid.")
            native_global_importance = {"method": "feature_importances_", "scope": "global_only", "features": [{"feature_name": name, "importance": float(value)} for name, value in zip(feature_names, values)]}
    core = {"method": {"id": "selected_model_fused_latent_explanation", "classifier_family": classifier_family, "classifier_configuration_id": classifier_configuration_id, "direct_gene_attribution_from_final_model": False, "limitations": limitations}, "background": {"source": "complete_real_outer_training_fused_latents", "matrix_sha256": _array_sha256(outer_train_features), "sample_ids_sha256": _sample_id_list_sha256(outer_train_ids), "label_usage": "training_labels_for_global_permutation_importance_only"}, "fused_latent_schema": {"feature_names": list(feature_names), "feature_name_sha256": feature_name_sha256, "modality_slices": {modality: list(modality_slices[modality]) for modality in FUSION_MODALITY_ORDER}}, "global_training_reference_importance": global_importance, "native_global_importance": native_global_importance, "patients": patients, "secondary_surrogate_biological_feature_explanation": {"available": False, "label": "secondary surrogate biological-feature explanation", "generated_in_this_repair": False}, "guards": {"outer_test_labels_used": False, "additional_outer_test_prediction_calls": 0, "selection_influenced": False, "metrics_influenced": False, "reliability_influenced": False}, **PHASE10C_ACTIVE_GUARDS.get()}
    _phase10c_validate_selected_model_explanations(core, require_predictions=False)
    return core


def _phase10c_attach_published_prediction_records(explanation_core: Dict[str, Any], prediction_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = copy.deepcopy(explanation_core)
    patients = result.get("patients")
    if not isinstance(patients, list) or [item.get("SAMPLE_ID") for item in patients] != [item.get("SAMPLE_ID") for item in prediction_records]:
        raise ValueError("Phase 10C explanation patients do not match final prediction SAMPLE_ID order.")
    for patient, prediction in zip(patients, prediction_records):
        patient.update({"predicted_class": int(prediction["predicted_binary_label"]), "score_type": prediction["score_type"], "continuous_prediction_score": float(prediction["high_tmb_continuous_score"])})
    result["prediction_record_sha256"] = _phase10c_prediction_record_identity(prediction_records)
    _phase10c_validate_selected_model_explanations(result, require_predictions=True)
    return result


def _phase10c_validate_selected_model_explanations(explanation: Dict[str, Any], require_predictions: bool) -> None:
    if not isinstance(explanation, dict) or explanation.get("method", {}).get("id") != "selected_model_fused_latent_explanation" or explanation.get("method", {}).get("direct_gene_attribution_from_final_model") is not False:
        raise ValueError("Phase 10C selected-model explanation method contract is invalid.")
    if explanation.get("background", {}).get("source") != "complete_real_outer_training_fused_latents" or explanation.get("background", {}).get("label_usage") != "training_labels_for_global_permutation_importance_only":
        raise ValueError("Phase 10C explanation background contract is invalid.")
    guards = explanation.get("guards", {})
    if guards.get("outer_test_labels_used") is not False or guards.get("additional_outer_test_prediction_calls") != 0 or guards.get("selection_influenced") is not False or guards.get("metrics_influenced") is not False or guards.get("reliability_influenced") is not False:
        raise ValueError("Phase 10C explanation leakage guards are invalid.")
    surrogate = explanation.get("secondary_surrogate_biological_feature_explanation", {})
    if surrogate != {"available": False, "label": "secondary surrogate biological-feature explanation", "generated_in_this_repair": False}:
        raise ValueError("Phase 10C explanation surrogate boundary is invalid.")
    forbidden = {"true_binary_label", "y_test", "outer_test_labels"}
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("Phase 10C explanation contains forbidden outer-test labels.")
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(explanation)
    for patient in explanation.get("patients", []):
        if require_predictions and {"predicted_class", "score_type", "continuous_prediction_score"}.difference(patient):
            raise ValueError("Phase 10C published explanation lacks final prediction metadata.")


def _phase10c_extract_selected_inner_validation_score_reference(selected_candidate: Dict[str, Any], outer_train_ids: List[str]) -> Dict[str, Any]:
    inner_results = selected_candidate.get("inner_results")
    if not isinstance(inner_results, list) or len(inner_results) != INNER_N_SPLITS:
        raise ValueError("Phase 10C selected candidate lacks exactly three inner-validation score records.")
    ordered = sorted(inner_results, key=lambda item: item.get("inner_fold_id"))
    if [item.get("inner_fold_id") for item in ordered] != list(range(INNER_N_SPLITS)):
        raise ValueError("Phase 10C selected candidate inner-fold score identities are incomplete or duplicated.")
    score_types, boundaries, records = set(), set(), []
    for result in ordered:
        values = result.get("inner_validation_scores")
        if not isinstance(values, list) or not values:
            raise ValueError("Phase 10C selected candidate inner-validation scores are missing.")
        score_types.add(result.get("score_type")); boundaries.add(result.get("threshold"))
        records.extend(values)
    if len(score_types) != 1 or len(boundaries) != 1 or None in score_types or None in boundaries:
        raise ValueError("Phase 10C selected candidate inner-validation score contracts differ across folds.")
    ids = [item.get("SAMPLE_ID") for item in records]
    scores = np.asarray([item.get("continuous_score") for item in records], dtype=float)
    if any(not isinstance(identifier, str) for identifier in ids) or len(ids) != len(set(ids)) or set(ids) != set(outer_train_ids) or len(ids) != len(outer_train_ids) or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Phase 10C selected candidate inner-validation score coverage is invalid.")
    return {"records": records, "sample_ids": ids, "scores": scores, "score_type": next(iter(score_types)), "decision_boundary": float(next(iter(boundaries)))}


def _phase10c_build_margin_reference(score_reference: Dict[str, Any]) -> Dict[str, Any]:
    score_type, boundary = score_reference["score_type"], score_reference["decision_boundary"]
    expected_boundary = 0.5 if score_type == "uncalibrated_probability" else 0.0 if score_type == "decision_function" else None
    if expected_boundary is None or boundary != expected_boundary:
        raise ValueError("Phase 10C reliability score type or decision boundary is unsupported.")
    margins = np.abs(score_reference["scores"] - boundary)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all():
        raise ValueError("Phase 10C reliability margins are invalid.")
    return {"reference_count": int(len(margins)), "reference_sample_ids_sha256": _sample_id_list_sha256(score_reference["sample_ids"]), "score_type": score_type, "decision_boundary": boundary, "margin_min": float(np.min(margins)), "margin_max": float(np.max(margins)), "margin_mean": float(np.mean(margins)), "margin_median": float(np.median(margins)), "low_margin_percentile": PHASE10C_LOW_MARGIN_PERCENTILE, "low_margin_threshold": float(np.quantile(margins, PHASE10C_LOW_MARGIN_PERCENTILE)), "reference_margins": [float(value) for value in margins]}


def _phase10c_build_latent_ood_reference(outer_train_features: np.ndarray, outer_train_ids: List[str], feature_names: List[str]) -> Dict[str, Any]:
    if not isinstance(outer_train_features, np.ndarray) or outer_train_features.ndim != 2 or len(outer_train_features) < 2 or len(outer_train_ids) != len(outer_train_features) or not np.isfinite(outer_train_features).all():
        raise ValueError("Phase 10C reliability outer-training latent reference is invalid.")
    scaler = StandardScaler(**PHASE9B_STANDARD_SCALER_CONFIGURATION)
    standardized = np.asarray(scaler.fit_transform(outer_train_features), dtype=float)
    if standardized.shape != outer_train_features.shape or not np.isfinite(standardized).all():
        raise ValueError("Phase 10C reliability diagnostic scaler output is invalid.")
    distances = pairwise_distances(standardized, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    if not np.isfinite(nearest).all():
        raise ValueError("Phase 10C reliability self-excluded nearest-neighbor reference is invalid.")
    scale = np.asarray(scaler.scale_, dtype=float)
    zero_names = [name for name, variance in zip(feature_names, np.asarray(scaler.var_, dtype=float)) if variance == 0]
    return {"reference_matrix_sha256": _array_sha256(outer_train_features), "training_sample_ids_sha256": _sample_id_list_sha256(outer_train_ids), "scaler_mean": [float(value) for value in scaler.mean_], "scaler_scale": [float(value) for value in scale], "scaler_mean_sha256": _array_sha256(np.asarray(scaler.mean_, dtype=float)), "scaler_scale_sha256": _array_sha256(scale), "zero_scale_handling": {"standard_scaler_unit_scale_for_zero_variance": True, "zero_variance_feature_names": zero_names}, "distance_metric": "euclidean", "self_excluded": True, "ood_percentile": PHASE10C_OOD_PERCENTILE, "ood_distance_threshold": float(np.quantile(nearest, PHASE10C_OOD_PERCENTILE)), "reference_nearest_distances": [float(value) for value in nearest], "standardized_reference_latents": standardized.tolist(), "finite_reference_checks": True}


def _phase10c_build_selected_model_reliability_core(selected_candidate: Dict[str, Any], outer_train_features: np.ndarray, outer_train_ids: List[str], feature_names: List[str], feature_name_sha256: str) -> Dict[str, Any]:
    if _feature_name_sha256(feature_names) != feature_name_sha256:
        raise ValueError("Phase 10C reliability feature schema hash is invalid.")
    score_reference = _phase10c_extract_selected_inner_validation_score_reference(selected_candidate, outer_train_ids)
    margin_reference = _phase10c_build_margin_reference(score_reference)
    ood_reference = _phase10c_build_latent_ood_reference(outer_train_features, outer_train_ids, feature_names)
    core = {"schema_version": PHASE10C_RELIABILITY_SCHEMA_VERSION, "method": {"id": "training_reference_margin_and_latent_ood", "classifier_family": selected_candidate["family"], "classifier_configuration_id": selected_candidate["configuration_id"], "probabilities_calibrated": False, "statistical_coverage_guarantee": False, "clinical_validation_claim": False}, "training_reference": {"margin_reference": margin_reference, "latent_ood_reference": ood_reference}, "guards": {"outer_test_labels_used": False, "additional_outer_test_prediction_calls": 0, "thresholds_tuned_on_outer_test": False, "classifier_refitted": False, "selection_influenced": False, "metrics_influenced": False, "explanation_influenced": False}, **PHASE10C_ACTIVE_GUARDS.get()}
    _phase10c_validate_selected_model_reliability(core, require_patients=False)
    return core


def _phase10c_prediction_record_identity(prediction_records: List[Dict[str, Any]]) -> str:
    return _phase10c_hash([{key: record[key] for key in ["SAMPLE_ID", "predicted_binary_label", "score_type", "high_tmb_continuous_score"]} for record in prediction_records])


def _phase10c_attach_published_reliability_records(reliability_core: Dict[str, Any], prediction_records: List[Dict[str, Any]], outer_test_features: np.ndarray, outer_test_ids: List[str]) -> Dict[str, Any]:
    _phase10c_validate_selected_model_reliability(reliability_core, require_patients=False)
    if [record.get("SAMPLE_ID") for record in prediction_records] != outer_test_ids or len(outer_test_features) != len(outer_test_ids):
        raise ValueError("Phase 10C reliability prediction IDs do not match outer-test SAMPLE_ID order.")
    margin_reference, ood_reference = reliability_core["training_reference"]["margin_reference"], reliability_core["training_reference"]["latent_ood_reference"]
    score_type, boundary = margin_reference["score_type"], margin_reference["decision_boundary"]
    means, scales = np.asarray(ood_reference["scaler_mean"], dtype=float), np.asarray(ood_reference["scaler_scale"], dtype=float)
    reference = np.asarray(ood_reference["standardized_reference_latents"], dtype=float)
    standardized_test = (np.asarray(outer_test_features, dtype=float) - means) / scales
    if standardized_test.shape[1] != reference.shape[1] or not np.isfinite(standardized_test).all() or not np.isfinite(reference).all():
        raise ValueError("Phase 10C reliability diagnostic outer-test transformation is invalid.")
    nearest = np.min(pairwise_distances(standardized_test, reference, metric="euclidean"), axis=1)
    margins = np.asarray(margin_reference["reference_margins"], dtype=float)
    patients = []
    for record, distance in zip(prediction_records, nearest):
        if record.get("score_type") != score_type:
            raise ValueError("Phase 10C reliability published score type differs from training reference.")
        score = float(record["high_tmb_continuous_score"])
        margin = abs(score - boundary)
        percentile = float(np.mean(margins <= margin))
        low_margin, latent_ood = percentile < PHASE10C_LOW_MARGIN_PERCENTILE, bool(distance > ood_reference["ood_distance_threshold"])
        flags = (["low_relative_margin"] if low_margin else []) + (["latent_out_of_distribution"] if latent_ood else [])
        patients.append({"SAMPLE_ID": record["SAMPLE_ID"], "predicted_binary_label": int(record["predicted_binary_label"]), "score_type": score_type, "continuous_score": score, "score_margin": float(margin), "relative_margin_percentile": percentile, "nearest_training_distance": float(distance), "low_margin": low_margin, "latent_ood": latent_ood, "reliability_status": "inconclusive" if flags else "supported_with_caution", "reliability_flags": flags})
    result = {"schema_version": PHASE10C_RELIABILITY_SCHEMA_VERSION, "method": reliability_core["method"], "training_reference": reliability_core["training_reference"], "patients": patients, "prediction_record_sha256": _phase10c_prediction_record_identity(prediction_records), "summary": {"patient_count": len(patients), "supported_with_caution_count": sum(item["reliability_status"] == "supported_with_caution" for item in patients), "inconclusive_count": sum(item["reliability_status"] == "inconclusive" for item in patients)}, "guards": reliability_core["guards"], **PHASE10C_ACTIVE_GUARDS.get()}
    _phase10c_validate_selected_model_reliability(result, require_patients=True)
    return result


def _phase10c_validate_selected_model_reliability(reliability: Dict[str, Any], require_patients: bool) -> None:
    if not isinstance(reliability, dict) or reliability.get("schema_version") != PHASE10C_RELIABILITY_SCHEMA_VERSION or reliability.get("method", {}).get("id") != "training_reference_margin_and_latent_ood":
        raise ValueError("Phase 10C reliability schema is invalid.")
    guards = reliability.get("guards", {})
    required_guards = {"outer_test_labels_used": False, "additional_outer_test_prediction_calls": 0, "thresholds_tuned_on_outer_test": False, "classifier_refitted": False, "selection_influenced": False, "metrics_influenced": False, "explanation_influenced": False}
    if any(guards.get(key) != value for key, value in required_guards.items()):
        raise ValueError("Phase 10C reliability leakage guards are invalid.")
    margin = reliability.get("training_reference", {}).get("margin_reference", {})
    ood = reliability.get("training_reference", {}).get("latent_ood_reference", {})
    if margin.get("low_margin_percentile") != PHASE10C_LOW_MARGIN_PERCENTILE or ood.get("ood_percentile") != PHASE10C_OOD_PERCENTILE or margin.get("score_type") not in {"uncalibrated_probability", "decision_function"} or not np.isfinite(float(margin.get("low_margin_threshold", np.nan))) or not np.isfinite(float(ood.get("ood_distance_threshold", np.nan))):
        raise ValueError("Phase 10C reliability training reference is invalid.")
    forbidden = {"true_binary_label", "y_test", "outer_test_labels"}
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden.intersection(value): raise ValueError("Phase 10C reliability contains forbidden outer-test labels.")
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(reliability)
    if require_patients:
        patients = reliability.get("patients")
        if not isinstance(patients, list) or reliability.get("summary", {}).get("patient_count") != len(patients) or any({"SAMPLE_ID", "predicted_binary_label", "score_type", "continuous_score", "reliability_status"}.difference(item) for item in patients):
            raise ValueError("Phase 10C published reliability patients are invalid.")


def build_phase10c_standardized_smote_augmentation(features: np.ndarray, labels: np.ndarray, ids: List[str], feature_names: List[str], repeat_id: int, fold_id: int, inner_fold_id: int, api: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The approved Phase 10C adapter standardizes neighbor geometry only."""
    scaler = StandardScaler(**PHASE9B_STANDARD_SCALER_CONFIGURATION)
    raw = np.asarray(features, dtype=np.float32)
    standardized = scaler.fit_transform(raw).astype(np.float32)
    sampled = fit_and_resample_smote_latent(standardized, labels, SMOTE_LATENT_CONFIGURATION, api)
    generated_standardized = sampled["resampled_features"][len(raw):]
    generated = scaler.inverse_transform(generated_standardized).astype(np.float32)
    augmented = np.vstack([raw, generated]).astype(np.float32)
    if not np.array_equal(augmented[:len(raw)], raw):
        raise AssertionError("Phase 10C standardized SMOTE changed real raw rows.")
    synthetic_ids = [f"SYNTHETIC:SMOTE:R{repeat_id:03d}:F{fold_id:03d}:I{inner_fold_id:03d}:CLASS{sampled['minority_label']}:{index:06d}" for index in range(len(generated))]
    return {"strategy": SMOTE_LATENT_STRATEGY, "features": augmented, "labels": sampled["resampled_labels"].astype(int), "ids": list(ids) + synthetic_ids, "feature_names": list(feature_names), "synthetic_count": int(len(generated)), "evidence": {"augmentation_scaler_fit_ids_sha256": _sample_id_list_sha256(ids), "standardized_train_sha256": _array_sha256(standardized), "synthetic_standardized_sha256": _array_sha256(generated_standardized), "synthetic_raw_sha256": _array_sha256(generated), "sequence": ["fit_augmentation_standard_scaler_on_real_inner_training_raw_fused_latents_only", "transform_real_inner_training_only", "run_smote_in_standardized_space", "identify_synthetic_suffix", "inverse_transform_synthetic_suffix_only", "return_real_raw_rows_then_synthetic_raw_rows"], "artifact_disposition": "newly_generated"}}


def _phase10c_ctgan_config(search: Dict[str, Any], row_count: int, profile: Dict[str, Any]) -> Dict[str, Any]:
    pac = int(search["ctgan_contract"]["pac"])
    batch = (min(500, row_count) // pac) * pac
    if batch < pac:
        raise ValueError("Phase 10C CTGAN dynamic batch size is smaller than pac.")
    epochs = int(profile["active_training_settings"]["ctgan_epochs"])
    if profile["execution_profile"] == "full_nested_scientific_run" and epochs != int(search["ctgan_contract"]["epochs"]):
        raise ValueError("Full nested scientific run CTGAN epochs differ from the frozen search-space contract.")
    return {"epochs": epochs, "batch_size": batch, "pac": pac, "verbose": bool(search["ctgan_contract"]["verbose"])}


def _phase10c_build_augmentation(representation: Dict[str, Any], strategy: Dict[str, Any], search: Dict[str, Any], repeat_id: int, fold_id: int, inner_fold_id: int, profile: Dict[str, Any]) -> Dict[str, Any]:
    name, features, labels, ids, names = strategy["strategy"], representation["train"], representation["y_train"], representation["train_ids"], representation["feature_names"]
    if name in {"real_only_unweighted", "real_only_class_weight_balanced"}:
        return {"strategy": name, "features": features.copy(), "labels": labels.copy(), "ids": list(ids), "feature_names": list(names), "synthetic_count": 0, "evidence": {"artifact_disposition": "reused", "augmentation": "none"}}
    if name == SMOTE_LATENT_STRATEGY:
        return build_phase10c_standardized_smote_augmentation(features, labels, ids, names, repeat_id, fold_id, inner_fold_id)
    ctgan_fit_rows = int(np.min(np.unique(labels, return_counts=True)[1])) if name == MINORITY_CTGAN_STRATEGY else len(features)
    config = _phase10c_ctgan_config(search, ctgan_fit_rows, profile)
    if name == MINORITY_CTGAN_STRATEGY:
        result = build_minority_ctgan_augmentation(features, labels, ids, names, representation["feature_name_sha256"], repeat_id, fold_id, config, 9000 + repeat_id * 1000 + fold_id * 10 + inner_fold_id)
    elif name == CONDITIONAL_CTGAN_STRATEGY:
        result = build_conditional_ctgan_augmentation(features, labels, ids, names, representation["feature_name_sha256"], repeat_id, fold_id, config, 10000 + repeat_id * 1000 + fold_id * 10 + inner_fold_id)
    else:
        raise ValueError("Phase 10C imbalance strategy is not approved.")
    return {"strategy": name, "features": result["augmented_outer_train"].astype(np.float32), "labels": result["y_augmented"].astype(int), "ids": list(result["augmented_record_ids"]), "feature_names": list(names), "synthetic_count": int(result["generated_synthetic_count"]), "evidence": {"artifact_disposition": "newly_generated", "ctgan_configuration": result["ctgan_configuration"], "ctgan_api": result["ctgan_api"], "ctgan_execution": result["ctgan_execution"], "payload_sha256": _array_sha256(result["augmented_outer_train"]), "ctgan_regeneration_byte_identical_claim": False}}


def _phase10c_strategy_cache_key(representation: Dict[str, Any], strategy: Dict[str, Any], profile_hash: str) -> str:
    backend = CTGAN_EXECUTION_BACKEND if strategy["strategy"] in {MINORITY_CTGAN_STRATEGY, CONDITIONAL_CTGAN_STRATEGY} else None
    return _phase10c_hash({"representation_hash": _array_sha256(representation["train"]), "inner_train_id_hash": _sample_id_list_sha256(representation["train_ids"]), "inner_train_label_hash": _array_sha256(representation["y_train"]), "latent_schema_hash": representation["feature_name_sha256"], "imbalance_strategy": strategy, "ctgan_execution_backend": backend, "profile_hash": profile_hash})


def _phase10c_write_augmentation_cache(path: Path, key: str, value: Dict[str, Any]) -> None:
    def writer(directory: Path) -> None:
        np.savez_compressed(directory / "training.npz", features=value["features"], labels=value["labels"])
        _phase10c_atomic_write_json(directory / "metadata.json", {"cache_key": key, "ids": value["ids"], "feature_names": value["feature_names"], "strategy": value["strategy"], "synthetic_count": value["synthetic_count"], "evidence": value["evidence"], "payload_hashes": {"features": _array_sha256(value["features"]), "labels": _array_sha256(value["labels"])}, "completion": True, **PHASE10C_ACTIVE_GUARDS.get()})
    _phase10c_publish_directory(path, writer)


def _phase10c_load_augmentation_cache(path: Path, key: str) -> Optional[Dict[str, Any]]:
    try:
        metadata = _phase10c_read_json(path / "metadata.json", "Phase 10C augmentation cache")
        if metadata.get("cache_key") != key or not metadata.get("completion"):
            return None
        with np.load(path / "training.npz") as data:
            features, labels = data["features"].astype(np.float32), data["labels"].astype(int)
        if _array_sha256(features) != metadata["payload_hashes"]["features"] or _array_sha256(labels) != metadata["payload_hashes"]["labels"]:
            return None
        if metadata.get("strategy") in {MINORITY_CTGAN_STRATEGY, CONDITIONAL_CTGAN_STRATEGY} and metadata.get("evidence", {}).get("ctgan_execution", {}).get("ctgan_execution_backend") != CTGAN_EXECUTION_BACKEND:
            return None
        return {"features": features, "labels": labels, "ids": metadata["ids"], "feature_names": metadata["feature_names"], "strategy": metadata["strategy"], "synthetic_count": metadata["synthetic_count"], "evidence": {**metadata["evidence"], "artifact_disposition": "reused"}}
    except (FileNotFoundError, ValueError, KeyError, OSError):
        return None


def _phase10c_cache_classifier_result(path: Path, key: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if path.is_file():
        try:
            current = _phase10c_read_json(path, "Phase 10C classifier cache")
            if current.get("cache_key") == key and current.get("completion"):
                return current["payload"]
        except (ValueError, FileNotFoundError):
            pass
    _phase10c_atomic_write_json(path, {"cache_key": key, "completion": True, "payload": payload, **PHASE10C_ACTIVE_GUARDS.get()})
    return None


def _phase10c_select(candidates: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    valid = [item for item in candidates if not item.get("failed")]
    if not valid:
        raise RuntimeError(f"Phase 10C {stage} has no valid candidate.")
    def key(item: Dict[str, Any]) -> Tuple[Any, ...]:
        metrics = item["aggregate"]
        if stage == "stage_a":
            return (-metrics["mean_auprc"], -metrics["mean_balanced_accuracy"], -metrics["mean_sensitivity_high_tmb"], metrics["std_auprc"], item["fused_latent_width"], item["candidate_id"])
        return (-metrics["mean_auprc"], -metrics["mean_balanced_accuracy"], -metrics["mean_sensitivity_high_tmb"], metrics["std_auprc"], item["configuration_id"], item["strategy"])
    return sorted(valid, key=key)[0]


def _phase10c_event_paths(output_dir: Path, attempt_id: str) -> Dict[str, Path]:
    base = output_dir / "final_refit"
    return {"not_started": base / "evaluation_state" / f"00_not_started--{attempt_id}.json", "prepared": base / "evaluation_state" / f"10_prepared--{attempt_id}.json", "started": base / "evaluation_state" / f"20_evaluation_started--{attempt_id}.json", "completed": base / "evaluation_state" / f"30_completed--{attempt_id}.json", "start_event": base / "evaluation_events" / f"evaluation-start--{attempt_id}.json", "completed_event": base / "evaluation_events" / f"evaluation-completed--{attempt_id}.json", "prepared_bundle": base / "prepared" / attempt_id, "published": base / "published" / attempt_id}


def _phase10c_attempt_id(binding: Dict[str, Any], representation_id: str, candidate_id: str) -> str:
    return _phase10c_hash({"schema_version": "phase10c-final-evaluation-attempt-v1", "binding": binding, "selected_representation_id": representation_id, "selected_stage_b_candidate_id": candidate_id})


def _phase10c_state_payload(name: str, attempt_id: str, binding: Dict[str, Any], representation_id: str, candidate_id: str, prepared_manifest: Dict[str, Any], predecessor_hash: Optional[str] = None, publication_hash: Optional[str] = None) -> Dict[str, Any]:
    return {"schema_version": "phase10c-final-evaluation-state-v1", "state": name, "attempt_id": attempt_id, "binding": binding, "selected_representation_id": representation_id, "selected_stage_b_candidate_id": candidate_id, "prepared_bundle_hash": _phase10c_hash(prepared_manifest), "final_refit_preprocessing_evidence_sha256": prepared_manifest["preprocessing_evidence_sha256"], "final_ae_evidence_sha256": prepared_manifest["ae_evidence_sha256"], "selected_augmentation_evidence_sha256": prepared_manifest["augmentation_evidence_sha256"], "final_classifier_configuration": prepared_manifest["final_classifier_configuration"], "fitted_classifier_sha256": prepared_manifest["fitted_classifier_sha256"], "outer_test_matrix_sha256": prepared_manifest["outer_test_matrix_sha256"], "outer_test_label_sha256": prepared_manifest["outer_test_label_sha256"], "outer_test_sample_ids_sha256": prepared_manifest["outer_test_sample_ids_sha256"], "predecessor_state_sha256": predecessor_hash, "publication_manifest_sha256": publication_hash, **PHASE10C_ACTIVE_GUARDS.get()}


def _phase10c_validate_event_history(paths: Dict[str, Path], attempt_id: str, binding: Dict[str, Any], require_completed: bool) -> Dict[str, Any]:
    event_dir = paths["start_event"].parent
    start_events = []
    completed_events = []
    if event_dir.exists():
        for item in event_dir.glob("*.json"):
            event = _phase10c_read_json(item, "Phase 10C evaluation event")
            if event.get("attempt_id") == attempt_id:
                (start_events if event.get("event_type") == "evaluation_started" else completed_events if event.get("event_type") == "evaluation_completed" else []).append(event)
    if len(start_events) != 1:
        raise RuntimeError("ambiguous_outer_test_evaluation_state: expected exactly one immutable evaluation-start event.")
    if start_events[0].get("event_id") != _phase10c_hash({key: value for key, value in start_events[0].items() if key != "event_id"}):
        raise RuntimeError("ambiguous_outer_test_evaluation_state: evaluation-start event identity is invalid.")
    if start_events[0].get("binding") != binding:
        raise RuntimeError("ambiguous_outer_test_evaluation_state: evaluation-start binding differs.")
    if not paths["started"].is_file() or _phase10c_hash(_phase10c_read_json(paths["started"], "Phase 10C evaluation-started state")) != start_events[0].get("evaluation_started_state_sha256"):
        raise RuntimeError("ambiguous_outer_test_evaluation_state: evaluation-start state hash differs.")
    if require_completed:
        hash_keys = ["outer_test_matrix_sha256", "outer_test_label_sha256", "outer_test_sample_ids_sha256"]
        if len(completed_events) != 1 or completed_events[0].get("start_event_id") != start_events[0].get("event_id") or completed_events[0].get("binding") != binding or any(completed_events[0].get(key) != start_events[0].get(key) for key in hash_keys):
            raise RuntimeError("ambiguous_outer_test_evaluation_state: completion event is missing or conflicts.")
        if completed_events[0].get("event_id") != _phase10c_hash({key: value for key, value in completed_events[0].items() if key != "event_id"}) or not paths["completed"].is_file() or _phase10c_hash(_phase10c_read_json(paths["completed"], "Phase 10C completed evaluation state")) != completed_events[0].get("completed_state_sha256"):
            raise RuntimeError("ambiguous_outer_test_evaluation_state: completion evidence identity differs.")
    elif completed_events:
        raise RuntimeError("ambiguous_outer_test_evaluation_state: completion event exists without a completed state.")
    return {"start": start_events[0], "completed": completed_events[0] if completed_events else None, "outer_test_evaluation_count": len(start_events)}


def _phase10c_validate_final_resume(paths: Dict[str, Path], attempt_id: str, binding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if paths["started"].exists() and not paths["completed"].exists():
        # Any start-state presence is ambiguous, including an incomplete JSON state.
        raise RuntimeError("ambiguous_outer_test_evaluation_state: evaluation_started exists without a fully validated completion.")
    if paths["completed"].exists():
        try:
            state = _phase10c_read_json(paths["completed"], "Phase 10C completed evaluation state")
            if state.get("attempt_id") != attempt_id or state.get("binding") != binding or not paths["published"].is_dir():
                raise ValueError("completed state binding or publication is invalid")
            events = _phase10c_validate_event_history(paths, attempt_id, binding, True)
            publication = _phase10c_read_json(paths["published"] / "publication_manifest.json", "Phase 10C final publication")
            if publication.get("attempt_id") != attempt_id or publication.get("binding") != binding or publication.get("outer_test_evaluation_count") != 1:
                raise ValueError("publication contract is invalid")
            result = _phase10c_read_json(paths["published"] / "final_outer_test_result.json", "Phase 10C final result")
            explanation = _phase10c_read_json(paths["published"] / "selected_model_explanations.json", "Phase 10C published explanations")
            reliability = _phase10c_read_json(paths["published"] / "selected_model_reliability.json", "Phase 10C published reliability")
            if _sha256_file(paths["published"] / "ordered_predictions.jsonl") != publication.get("ordered_predictions_sha256") or _phase10c_hash(result) != publication.get("result_sha256") or _phase10c_hash(explanation) != publication.get("selected_model_explanations_sha256") or _phase10c_hash(reliability) != publication.get("selected_model_reliability_sha256"):
                raise ValueError("publication hashes are invalid")
            _phase10c_validate_selected_model_explanations(explanation, require_predictions=True)
            _phase10c_validate_selected_model_reliability(reliability, require_patients=True)
            if explanation.get("prediction_record_sha256") != reliability.get("prediction_record_sha256"):
                raise ValueError("published explanation and reliability prediction identities differ")
            return {"result": result, "explanations": explanation, "reliability": reliability, "events": events, "publication": publication, "resumed_completed": True}
        except (FileNotFoundError, ValueError, KeyError, OSError) as error:
            raise RuntimeError("ambiguous_outer_test_evaluation_state: completed evidence is invalid.") from error
    return None


def _phase10c_write_prepared_bundle(paths: Dict[str, Path], prepared: Dict[str, Any], run_guards: Dict[str, Any] = PHASE10C_GUARDS) -> Dict[str, Any]:
    if paths["prepared_bundle"].is_dir():
        manifest = _phase10c_read_json(paths["prepared_bundle"] / "prepared_bundle_manifest.json", "Phase 10C prepared bundle")
        return manifest
    def writer(directory: Path) -> None:
        joblib.dump(prepared["fitted_classifier"], directory / "fitted_classifier.joblib")
        np.save(directory / "outer_test_features.npy", prepared["outer_test_features"])
        np.save(directory / "outer_test_labels.npy", prepared["outer_test_labels"])
        _phase10c_atomic_write_json(directory / "outer_test_sample_ids.json", {"sample_ids": prepared["outer_test_ids"]})
        manifest = {"schema_version": "phase10c-prepared-final-refit-v1", "attempt_id": prepared["attempt_id"], "binding": prepared["binding"], "preprocessing_evidence_sha256": _phase10c_hash(prepared["preprocessing_evidence"]), "ae_evidence_sha256": _phase10c_hash(prepared["ae_evidence"]), "augmentation_evidence_sha256": _phase10c_hash(prepared["augmentation_evidence"]), "explanation_core_sha256": _phase10c_hash(prepared["explanation_core"]), "reliability_core_sha256": _phase10c_hash(prepared["reliability_core"]), "final_classifier_configuration": prepared["classifier_configuration"], "fitted_classifier_sha256": _sha256_file(directory / "fitted_classifier.joblib"), "outer_test_matrix_sha256": _array_sha256(prepared["outer_test_features"]), "outer_test_label_sha256": _array_sha256(prepared["outer_test_labels"]), "outer_test_sample_ids_sha256": _sample_id_list_sha256(prepared["outer_test_ids"]), "outer_test_shape": list(prepared["outer_test_features"].shape), **run_guards}
        _phase10c_atomic_write_json(directory / "preprocessing_evidence.json", prepared["preprocessing_evidence"])
        _phase10c_atomic_write_json(directory / "ae_evidence.json", prepared["ae_evidence"])
        _phase10c_atomic_write_json(directory / "augmentation_evidence.json", prepared["augmentation_evidence"])
        _phase10c_atomic_write_json(directory / "classifier_configuration.json", prepared["classifier_configuration"])
        _phase10c_atomic_write_json(directory / "explanation_core.json", prepared["explanation_core"])
        _phase10c_atomic_write_json(directory / "reliability_core.json", prepared["reliability_core"])
        _phase10c_atomic_write_json(directory / "prepared_bundle_manifest.json", manifest)
    _phase10c_publish_directory(paths["prepared_bundle"], writer)
    return _phase10c_read_json(paths["prepared_bundle"] / "prepared_bundle_manifest.json", "Phase 10C prepared bundle")


def _phase10c_load_prepared_bundle(paths: Dict[str, Path], attempt_id: str, binding: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest = _phase10c_read_json(paths["prepared_bundle"] / "prepared_bundle_manifest.json", "Phase 10C prepared bundle")
    if manifest.get("attempt_id") != attempt_id or manifest.get("binding") != binding:
        raise RuntimeError("Phase 10C prepared final-refit bundle binding is invalid.")
    classifier_path = paths["prepared_bundle"] / "fitted_classifier.joblib"
    features = np.load(paths["prepared_bundle"] / "outer_test_features.npy").astype(np.float32)
    labels = np.load(paths["prepared_bundle"] / "outer_test_labels.npy").astype(int)
    ids = _phase10c_read_json(paths["prepared_bundle"] / "outer_test_sample_ids.json", "Phase 10C outer test IDs")["sample_ids"]
    explanation = _phase10c_read_json(paths["prepared_bundle"] / "explanation_core.json", "Phase 10C explanation core")
    reliability = _phase10c_read_json(paths["prepared_bundle"] / "reliability_core.json", "Phase 10C reliability core")
    if _sha256_file(classifier_path) != manifest.get("fitted_classifier_sha256") or _array_sha256(features) != manifest.get("outer_test_matrix_sha256") or _array_sha256(labels) != manifest.get("outer_test_label_sha256") or _sample_id_list_sha256(ids) != manifest.get("outer_test_sample_ids_sha256") or _phase10c_hash(explanation) != manifest.get("explanation_core_sha256") or _phase10c_hash(reliability) != manifest.get("reliability_core_sha256"):
        raise RuntimeError("Phase 10C prepared final-refit bundle payload hashes are invalid.")
    _phase10c_validate_selected_model_explanations(explanation, require_predictions=False)
    _phase10c_validate_selected_model_reliability(reliability, require_patients=False)
    return manifest, {"estimator": joblib.load(classifier_path), "features": features, "labels": labels, "ids": ids, "explanation_core": explanation, "reliability_core": reliability}


def _phase10c_run_final_evaluation(paths: Dict[str, Path], attempt_id: str, binding: Dict[str, Any], representation_id: str, candidate_id: str, prepared_manifest: Dict[str, Any], run_guards: Dict[str, Any] = PHASE10C_GUARDS, action: str = "nested_selection_integration_smoke_final_outer_test") -> Dict[str, Any]:
    existing = _phase10c_validate_final_resume(paths, attempt_id, binding)
    if existing is not None:
        return existing
    manifest, prepared = _phase10c_load_prepared_bundle(paths, attempt_id, binding)
    state_paths = [("not_started", None), ("prepared", "not_started"), ("started", "prepared")]
    prior_hash = None
    for name, predecessor in state_paths:
        path = paths[name]
        if path.exists():
            payload = _phase10c_read_json(path, f"Phase 10C {name} state")
            expected_state = "evaluation_started" if name == "started" else name
            if payload.get("state") != expected_state or payload.get("attempt_id") != attempt_id or payload.get("binding") != binding:
                raise RuntimeError("ambiguous_outer_test_evaluation_state: final evaluation state contract is invalid.")
            prior_hash = _phase10c_hash(payload)
            continue
        payload = _phase10c_state_payload("evaluation_started" if name == "started" else name, attempt_id, binding, representation_id, candidate_id, manifest, prior_hash)
        prior_hash = _phase10c_create_immutable_json(path, payload)
    start_payload = {"schema_version": "phase10c-final-evaluation-event-v1", "event_type": "evaluation_started", "attempt_id": attempt_id, "binding": binding, "evaluation_started_state_sha256": prior_hash, "outer_test_matrix_sha256": manifest["outer_test_matrix_sha256"], "outer_test_label_sha256": manifest["outer_test_label_sha256"], "outer_test_sample_ids_sha256": manifest["outer_test_sample_ids_sha256"], **run_guards}
    start_payload["event_id"] = _phase10c_hash({key: value for key, value in start_payload.items() if key != "event_id"})
    _phase10c_create_immutable_json(paths["start_event"], start_payload)
    # This is the only permitted outer-test scoring call for the attempt.
    fitted = {"estimator": prepared["estimator"], "classes": _validate_logistic_estimator_classes(prepared["estimator"].named_steps["classifier"] if isinstance(prepared["estimator"], Pipeline) else prepared["estimator"]), "score_source": manifest["final_classifier_configuration"]["score_source"]}
    evaluation = _phase10c_evaluate_classifier(fitted, prepared["features"], prepared["labels"], prepared["ids"])
    explanations = _phase10c_attach_published_prediction_records(prepared["explanation_core"], evaluation["records"])
    reliability = _phase10c_attach_published_reliability_records(prepared["reliability_core"], evaluation["records"], prepared["features"], prepared["ids"])
    if explanations.get("prediction_record_sha256") != reliability.get("prediction_record_sha256"):
        raise AssertionError("Phase 10C explanation and reliability prediction identities differ.")
    result = {"action": action, "attempt_id": attempt_id, "binding": binding, "selected_representation_id": representation_id, "selected_stage_b_candidate_id": candidate_id, "outer_test_evaluation_count": 1, "metrics": evaluation["metrics"], "score_type": evaluation["score_type"], "threshold": evaluation["threshold"], "confusion_matrix": evaluation["confusion_matrix"], "outer_test_identity": {"matrix_sha256": manifest["outer_test_matrix_sha256"], "label_sha256": manifest["outer_test_label_sha256"], "sample_ids_sha256": manifest["outer_test_sample_ids_sha256"]}, **run_guards}
    def writer(directory: Path) -> None:
        prediction_path = directory / "ordered_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in evaluation["records"]:
                handle.write(_canonical_json_bytes(record).decode("utf-8") + "\n")
            handle.flush(); os.fsync(handle.fileno())
        if [item["SAMPLE_ID"] for item in evaluation["records"]] != prepared["ids"] or len(evaluation["records"]) != len(prepared["ids"]) or not np.isfinite(evaluation["scores"]).all():
            raise ValueError("Phase 10C final prediction contract is invalid.")
        _phase10c_atomic_write_json(directory / "final_outer_test_result.json", result)
        _phase10c_atomic_write_json(directory / "selected_model_explanations.json", explanations)
        _phase10c_atomic_write_json(directory / "selected_model_reliability.json", reliability)
        _phase10c_atomic_write_json(directory / "publication_manifest.json", {"attempt_id": attempt_id, "binding": binding, "ordered_predictions_sha256": _sha256_file(prediction_path), "result_sha256": _phase10c_hash(result), "selected_model_explanations_sha256": _phase10c_hash(explanations), "selected_model_reliability_sha256": _phase10c_hash(reliability), "outer_test_evaluation_count": 1, **run_guards})
    _phase10c_publish_directory(paths["published"], writer)
    publication = _phase10c_read_json(paths["published"] / "publication_manifest.json", "Phase 10C publication manifest")
    completed_state = _phase10c_state_payload("completed", attempt_id, binding, representation_id, candidate_id, manifest, prior_hash, _phase10c_hash(publication))
    completed_hash = _phase10c_create_immutable_json(paths["completed"], completed_state)
    completed_event = {"schema_version": "phase10c-final-evaluation-event-v1", "event_type": "evaluation_completed", "attempt_id": attempt_id, "binding": binding, "start_event_id": start_payload["event_id"], "completed_state_sha256": completed_hash, "ordered_predictions_sha256": publication["ordered_predictions_sha256"], "result_sha256": publication["result_sha256"], "selected_model_explanations_sha256": publication["selected_model_explanations_sha256"], "selected_model_reliability_sha256": publication["selected_model_reliability_sha256"], "outer_test_matrix_sha256": manifest["outer_test_matrix_sha256"], "outer_test_label_sha256": manifest["outer_test_label_sha256"], "outer_test_sample_ids_sha256": manifest["outer_test_sample_ids_sha256"], **run_guards}
    completed_event["event_id"] = _phase10c_hash({key: value for key, value in completed_event.items() if key != "event_id"})
    _phase10c_create_immutable_json(paths["completed_event"], completed_event)
    return _phase10c_validate_final_resume(paths, attempt_id, binding) or {}


def _run_phase10c_nested_selection_fold(data: Dict[str, Any], outer_manifest: Dict[str, Any], inner_manifest: Dict[str, Any], search: Dict[str, Any], output_dir: Path, repeat_id: int, fold_id: int, outer_sha256: str, inner_sha256: str, search_sha256: str, profile_contract: Dict[str, Any], run_guards: Dict[str, Any], action: str) -> Dict[str, Any]:
    profile_hash = _phase10c_hash(profile_contract)
    binding = _phase10c_lock_binding(profile_hash, outer_sha256, inner_sha256, search_sha256, repeat_id, fold_id)
    guard_token = PHASE10C_ACTIVE_GUARDS.set(run_guards)
    lock = acquire_phase10c_run_lock(output_dir, binding)
    outcome = "failed"
    try:
        profile = {**profile_contract, "execution_profile_hash": profile_hash, "runtime_device_evidence": _phase10c_runtime_evidence()}
        existing_profile = output_dir / "execution_profile.json"
        if existing_profile.exists() and _phase10c_read_json(existing_profile, "Phase 10C execution profile").get("execution_profile_hash") != profile_hash:
            raise ValueError("Phase 10C output directory execution profile does not match requested run.")
        if not existing_profile.exists():
            _phase10c_atomic_write_json(existing_profile, profile)
        _phase10c_atomic_write_json(output_dir / "run_contract.json", {"schema_version": PHASE10C_SCHEMA_VERSION, "binding": binding, "manifest_validation": {"outer": True, "inner": True, "search": True}, **run_guards})
        selected_a_path, selected_b_path = output_dir / "stage_a" / "selected_representation.json", output_dir / "stage_b" / "selected_candidate.json"
        if selected_a_path.is_file() and selected_b_path.is_file():
            previous_a = _phase10c_read_json(selected_a_path, "Phase 10C selected representation")["selected"]
            previous_b = _phase10c_read_json(selected_b_path, "Phase 10C selected candidate")["selected"]
            prior_attempt = _phase10c_attempt_id(binding, previous_a["candidate_id"], previous_b["candidate_id"])
            prior_final = _phase10c_validate_final_resume(_phase10c_event_paths(output_dir, prior_attempt), prior_attempt, binding)
            if prior_final is not None:
                summary = {"action": action, "binding": binding, "selected_representation": previous_a, "selected_stage_b_candidate": previous_b, "final": prior_final, "resumed_completed": True, **run_guards}
                _phase10c_atomic_write_json(output_dir / "run_summary.json", summary)
                outcome = "completed"; return summary
        outer_fold = materialize_outer_fold(data, outer_manifest, repeat_id, fold_id)
        inner_records = _phase10c_inner_records(inner_manifest, repeat_id, fold_id)
        outer_train_set = set(outer_fold["train_sample_ids"])
        if any(not set(item["inner_train_sample_ids"]).issubset(outer_train_set) or not set(item["inner_validation_sample_ids"]).issubset(outer_train_set) for item in inner_records):
            raise ValueError("Phase 10C inner partition contains a non-outer-training ID.")
        representations: Dict[Tuple[str, int], Dict[str, Any]] = {}
        stage_a_rows, stage_a_candidates = [], []
        for candidate in search["representation_candidates"]:
            fold_rows, failed = [], False
            for inner in inner_records:
                inner_id = int(inner["inner_fold_id"]); partition = _phase10c_materialize_partition(data, list(inner["inner_train_sample_ids"]), list(inner["inner_validation_sample_ids"]))
                cache_key = _phase10c_representation_key(partition, candidate, profile_hash, repeat_id, fold_id, inner_id); cache_path = output_dir / "cache" / "representations" / cache_key
                representation = _phase10c_load_representation_cache(cache_path, cache_key)
                if representation is None:
                    try:
                        representation = _phase10c_build_representation(partition, candidate, repeat_id, fold_id, inner_id, profile)
                        _phase10c_write_representation_cache(cache_path, cache_key, representation)
                    except Exception as error:
                        failed = True; _phase10c_record_failure(output_dir, {"stage": "stage_a", "candidate_id": candidate["candidate_id"], "outer_identity": {"repeat_id": repeat_id, "fold_id": fold_id}, "inner_fold_id": inner_id, "component": "representation", "exception_category": type(error).__name__, "message": str(error), "input_hashes": {"cache_key": cache_key}}); continue
                representations[(candidate["candidate_id"], inner_id)] = representation
                metrics = []
                for family in PHASE9B_CLASSIFIER_ORDER:
                    config = next(item["parameters"] for item in search["classifier_registry"][family]["configurations"] if item["is_phase9b_canonical_baseline"])
                    try:
                        fitted = _phase10c_fit_classifier({"features": representation["train"], "labels": representation["y_train"], "feature_names": representation["feature_names"]}, family, config)
                        evaluated = _phase10c_evaluate_classifier(fitted, representation["validation"], representation["y_validation"], representation["validation_ids"])
                        metrics.append(evaluated["metrics"]); stage_a_rows.append({"candidate_id": candidate["candidate_id"], "inner_fold_id": inner_id, "family": family, "metrics": evaluated["metrics"], **run_guards})
                    except Exception as error:
                        failed = True; _phase10c_record_failure(output_dir, {"stage": "stage_a", "candidate_id": candidate["candidate_id"], "outer_identity": {"repeat_id": repeat_id, "fold_id": fold_id}, "inner_fold_id": inner_id, "component": "classifier", "exception_category": type(error).__name__, "message": str(error), "input_hashes": {"cache_key": cache_key, "family": family}})
                if len(metrics) == len(PHASE9B_CLASSIFIER_ORDER):
                    fold_rows.append({"auprc": float(np.mean([item["auprc"] for item in metrics])), "balanced_accuracy": float(np.mean([item["balanced_accuracy"] for item in metrics])), "sensitivity_high_tmb": float(np.mean([item["sensitivity_high_tmb"] for item in metrics]))})
                else:
                    failed = True
            aggregate = {"mean_auprc": float(np.mean([item["auprc"] for item in fold_rows])) if len(fold_rows) == 3 else None, "mean_balanced_accuracy": float(np.mean([item["balanced_accuracy"] for item in fold_rows])) if len(fold_rows) == 3 else None, "mean_sensitivity_high_tmb": float(np.mean([item["sensitivity_high_tmb"] for item in fold_rows])) if len(fold_rows) == 3 else None, "std_auprc": float(np.std([item["auprc"] for item in fold_rows])) if len(fold_rows) == 3 else None}
            stage_a_candidates.append({"candidate_id": candidate["candidate_id"], "fused_latent_width": candidate["fused_latent_width"], "fold_summaries": fold_rows, "aggregate": aggregate, "failed": failed or len(fold_rows) != 3})
        _phase10c_atomic_write_json(output_dir / "stage_a" / "representation_summary.json", {"candidates": stage_a_candidates, **run_guards})
        with (output_dir / "stage_a" / "fold_results.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in stage_a_rows: handle.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
        selected_a = _phase10c_select(stage_a_candidates, "stage_a")
        _phase10c_atomic_write_json(output_dir / "stage_a" / "selected_representation.json", {"selected": selected_a, **run_guards})
        stage_b_rows, candidates_b = [], []
        strategies = search["imbalance_strategies"]
        for family, family_spec in search["classifier_registry"].items():
            for configuration in family_spec["configurations"]:
                for strategy in strategies:
                    if strategy["strategy"] == "real_only_class_weight_balanced" and not search["class_weight_support_matrix"][family]["supported"]:
                        continue
                    results, failed = [], False
                    candidate_id = f"{family}:{configuration['configuration_id']}:{strategy['strategy']}"
                    for inner in inner_records:
                        inner_id = int(inner["inner_fold_id"]); representation = representations[(selected_a["candidate_id"], inner_id)]
                        augmentation_key = _phase10c_strategy_cache_key(representation, strategy, profile_hash); augmentation_path = output_dir / "cache" / "augmentations" / augmentation_key
                        augmentation = _phase10c_load_augmentation_cache(augmentation_path, augmentation_key)
                        if augmentation is None:
                            try:
                                augmentation = _phase10c_build_augmentation(representation, strategy, search, repeat_id, fold_id, inner_id, profile); _phase10c_write_augmentation_cache(augmentation_path, augmentation_key, augmentation)
                            except Exception as error:
                                failed = True; _phase10c_record_failure(output_dir, {"stage": "stage_b", "candidate_id": candidate_id, "outer_identity": {"repeat_id": repeat_id, "fold_id": fold_id}, "inner_fold_id": inner_id, "component": "augmentation", "exception_category": type(error).__name__, "message": str(error), "input_hashes": {"cache_key": augmentation_key}}); continue
                        training = {"features": augmentation["features"], "labels": augmentation["labels"], "feature_names": augmentation["feature_names"]}
                        classifier_key = _phase10c_hash({"training_variant_hash": _array_sha256(training["features"]), "classifier_configuration_id": configuration["configuration_id"], "family": family, "strategy": strategy["strategy"], "score_contract_version": "phase9a"})
                        result_path = output_dir / "cache" / "classifiers" / (classifier_key + ".json")
                        cached = _phase10c_cache_classifier_result(result_path, classifier_key, {}) if result_path.exists() else None
                        try:
                            if cached is None:
                                fitted = _phase10c_fit_classifier(training, family, configuration["parameters"], strategy.get("class_weight_override"))
                                evaluated = _phase10c_evaluate_classifier(fitted, representation["validation"], representation["y_validation"], representation["validation_ids"])
                                cached = {"metrics": evaluated["metrics"], "effective_configuration": fitted["effective_configuration"], "score_type": evaluated["score_type"], "threshold": evaluated["threshold"], "inner_validation_scores": [{"SAMPLE_ID": record["SAMPLE_ID"], "score_type": record["score_type"], "continuous_score": record["high_tmb_continuous_score"]} for record in evaluated["records"]], "augmentation_evidence": augmentation["evidence"]}
                                _phase10c_atomic_write_json(result_path, {"cache_key": classifier_key, "completion": True, "payload": cached, **run_guards})
                            results.append({"inner_fold_id": inner_id, **cached}); stage_b_rows.append({"candidate_id": candidate_id, "inner_fold_id": inner_id, "metrics": cached["metrics"], "augmentation_evidence": cached.get("augmentation_evidence", {}), **run_guards})
                        except Exception as error:
                            failed = True; _phase10c_record_failure(output_dir, {"stage": "stage_b", "candidate_id": candidate_id, "outer_identity": {"repeat_id": repeat_id, "fold_id": fold_id}, "inner_fold_id": inner_id, "component": "classifier", "exception_category": type(error).__name__, "message": str(error), "input_hashes": {"cache_key": classifier_key}})
                    aggregate = {"mean_auprc": float(np.mean([item["metrics"]["auprc"] for item in results])) if len(results) == 3 else None, "mean_balanced_accuracy": float(np.mean([item["metrics"]["balanced_accuracy"] for item in results])) if len(results) == 3 else None, "mean_sensitivity_high_tmb": float(np.mean([item["metrics"]["sensitivity_high_tmb"] for item in results])) if len(results) == 3 else None, "std_auprc": float(np.std([item["metrics"]["auprc"] for item in results])) if len(results) == 3 else None}
                    candidates_b.append({"candidate_id": candidate_id, "family": family, "configuration_id": configuration["configuration_id"], "configuration": configuration["parameters"], "strategy": strategy["strategy"], "class_weight_override": strategy.get("class_weight_override"), "aggregate": aggregate, "inner_results": results, "failed": failed or len(results) != 3})
        (output_dir / "stage_b").mkdir(parents=True, exist_ok=True)
        with (output_dir / "stage_b" / "candidate_results.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in stage_b_rows: handle.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
        selected_b = _phase10c_select(candidates_b, "stage_b")
        _phase10c_atomic_write_json(output_dir / "stage_b" / "selected_candidate.json", {"selected": selected_b, "candidate_count": len(candidates_b), **run_guards})
        # Fresh final refit uses no inner-fitted object.  Outer test is transformed but never scored here.
        partition = _phase10c_materialize_partition(data, outer_fold["train_sample_ids"], outer_fold["test_sample_ids"])
        candidate = next(item for item in search["representation_candidates"] if item["candidate_id"] == selected_a["candidate_id"])
        final_representation = _phase10c_build_representation(partition, candidate, repeat_id, fold_id, 0, profile, final_refit=True)
        final_strategy = next(item for item in strategies if item["strategy"] == selected_b["strategy"])
        final_augmentation = _phase10c_build_augmentation(final_representation, final_strategy, search, repeat_id, fold_id, 99, profile)
        final_fitted = _phase10c_fit_classifier({"features": final_augmentation["features"], "labels": final_augmentation["labels"], "feature_names": final_augmentation["feature_names"]}, selected_b["family"], selected_b["configuration"], selected_b["class_weight_override"])
        modality_slices = build_fold_latent_slices(candidate["latent_dimensions"])
        explanation_core = _phase10c_build_selected_model_explanation_core(
            selected_b["family"],
            selected_b["configuration_id"],
            final_fitted["estimator"],
            final_representation["train"],
            final_representation["y_train"],
            final_representation["train_ids"],
            final_representation["validation"],
            final_representation["validation_ids"],
            final_representation["feature_names"],
            final_representation["feature_name_sha256"],
            modality_slices,
        )
        reliability_core = _phase10c_build_selected_model_reliability_core(
            selected_b,
            final_representation["train"],
            final_representation["train_ids"],
            final_representation["feature_names"],
            final_representation["feature_name_sha256"],
        )
        attempt_id = _phase10c_attempt_id(binding, selected_a["candidate_id"], selected_b["candidate_id"]); paths = _phase10c_event_paths(output_dir, attempt_id)
        prepared = {"attempt_id": attempt_id, "binding": binding, "fitted_classifier": final_fitted["estimator"], "outer_test_features": final_representation["validation"], "outer_test_labels": final_representation["y_validation"], "outer_test_ids": final_representation["validation_ids"], "preprocessing_evidence": final_representation["preprocessing_evidence"], "ae_evidence": final_representation["ae_evidence"], "augmentation_evidence": final_augmentation["evidence"], "explanation_core": explanation_core, "reliability_core": reliability_core, "classifier_configuration": {"family": selected_b["family"], "configuration_id": selected_b["configuration_id"], "parameters": final_fitted["effective_configuration"], "score_source": final_fitted["score_source"]}}
        prepared_manifest = _phase10c_write_prepared_bundle(paths, prepared, run_guards)
        final = _phase10c_run_final_evaluation(paths, attempt_id, binding, selected_a["candidate_id"], selected_b["candidate_id"], prepared_manifest, run_guards, f"{action}_final_outer_test")
        summary = {"action": action, "binding": binding, "selected_representation": selected_a, "selected_stage_b_candidate": selected_b, "final": final, **run_guards}
        _phase10c_atomic_write_json(output_dir / "run_summary.json", summary)
        outcome = "completed"; return summary
    finally:
        release_phase10c_run_lock(lock, outcome)
        PHASE10C_ACTIVE_GUARDS.reset(guard_token)


def run_nested_selection_smoke(data: Dict[str, Any], outer_manifest: Dict[str, Any], inner_manifest: Dict[str, Any], search: Dict[str, Any], output_dir: Path, repeat_id: int, fold_id: int, outer_sha256: str, inner_sha256: str, search_sha256: str) -> Dict[str, Any]:
    profile_contract = build_phase10c_execution_profile_contract(outer_sha256, inner_sha256, search_sha256, repeat_id, fold_id)
    return _run_phase10c_nested_selection_fold(data, outer_manifest, inner_manifest, search, output_dir, repeat_id, fold_id, outer_sha256, inner_sha256, search_sha256, profile_contract, PHASE10C_GUARDS, "nested_selection_integration_smoke")


def _phase10c_resume_completed_fold(output_dir: Path, profile_hash: str, outer_sha256: str, inner_sha256: str, search_sha256: str, repeat_id: int, fold_id: int) -> Optional[Dict[str, Any]]:
    selected_a_path, selected_b_path = output_dir / "stage_a" / "selected_representation.json", output_dir / "stage_b" / "selected_candidate.json"
    if not selected_a_path.is_file() or not selected_b_path.is_file():
        return None
    selected_a = _phase10c_read_json(selected_a_path, "Phase 10C selected representation")["selected"]
    selected_b = _phase10c_read_json(selected_b_path, "Phase 10C selected candidate")["selected"]
    binding = _phase10c_lock_binding(profile_hash, outer_sha256, inner_sha256, search_sha256, repeat_id, fold_id)
    attempt_id = _phase10c_attempt_id(binding, selected_a["candidate_id"], selected_b["candidate_id"])
    final = _phase10c_validate_final_resume(_phase10c_event_paths(output_dir, attempt_id), attempt_id, binding)
    if final is None:
        return None
    return {"action": "full_nested_scientific_run", "binding": binding, "selected_representation": selected_a, "selected_stage_b_candidate": selected_b, "final": final, "resumed_completed": True, **PHASE10C_FULL_GUARDS}


def _phase10c_write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
        handle.flush(); os.fsync(handle.fileno())


def _phase10c_metric_summary_rows(rows: List[Dict[str, Any]], metric_names: List[str]) -> List[Dict[str, Any]]:
    summary = []
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows if row.get(metric) is not None], dtype=float)
        summary.append({"metric": metric, "count": int(len(values)), "mean": float(np.mean(values)) if len(values) else None, "sample_standard_deviation_ddof_1": float(np.std(values, ddof=1)) if len(values) > 1 else None, "minimum": float(np.min(values)) if len(values) else None, "maximum": float(np.max(values)) if len(values) else None})
    return summary


def _phase10c_aggregate_inner_rows(fold_directories: List[Tuple[int, int, Path]], relative_path: str, key_fields: List[str], stream_unique_fields: List[str], scope: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for repeat_id, fold_id, directory in fold_directories:
        stream = _phase10c_read_json_record_stream(directory / relative_path, unique_key=lambda item: tuple(item.get(key) for key in stream_unique_fields))
        for item in stream:
            key = tuple(item.get(field) for field in key_fields)
            if any(value is None for value in key) or not isinstance(item.get("metrics"), dict):
                raise ValueError(f"Phase 10C {scope} record is invalid.")
            grouped.setdefault(key, []).append(item["metrics"])
    metric_names = ["accuracy", "balanced_accuracy", "precision_high_tmb", "sensitivity_high_tmb", "specificity_low_tmb", "f1_high_tmb", "auroc", "auprc"]
    rows = []
    for key, metric_rows in sorted(grouped.items()):
        row = {field: value for field, value in zip(key_fields, key)}
        row["interpretation"] = "inner-validation model-selection evidence; not final outer-test performance"
        row["valid_inner_evaluation_count"] = len(metric_rows)
        for metric in metric_names:
            values = np.asarray([item[metric] for item in metric_rows if item.get(metric) is not None], dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values)) if len(values) else None
            row[f"sample_standard_deviation_ddof_1_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return rows


def _phase10c_build_full_aggregate(data: Dict[str, Any], outer_manifest: Dict[str, Any], output_dir: Path, root_binding: Dict[str, Any], fold_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = ["accuracy", "balanced_accuracy", "precision_high_tmb", "sensitivity_high_tmb", "specificity_low_tmb", "f1_high_tmb", "auroc", "auprc"]
    fold_rows, prediction_rows, reliability_totals, fold_directories = [], [], {"supported_with_caution_count": 0, "inconclusive_count": 0, "low_margin_count": 0, "latent_ood_count": 0, "joint_low_margin_and_ood_count": 0}, []
    for summary in sorted(fold_summaries, key=lambda item: (item["binding"]["repeat_id"], item["binding"]["fold_id"])):
        repeat_id, fold_id = summary["binding"]["repeat_id"], summary["binding"]["fold_id"]
        fold_record = next(item for item in outer_manifest["folds"] if item["repeat_id"] == repeat_id and item["fold_id"] == fold_id)
        fold_dir = output_dir / "folds" / f"repeat_{repeat_id}_fold_{fold_id}"
        publication_dir = fold_dir / "final_refit" / "published" / summary["final"]["result"]["attempt_id"]
        predictions = _phase10c_read_json_record_stream(publication_dir / "ordered_predictions.jsonl", expected_record_count=len(fold_record["test_sample_ids"]), unique_key=lambda item: item.get("SAMPLE_ID"))
        if [item.get("SAMPLE_ID") for item in predictions] != fold_record["test_sample_ids"]:
            raise ValueError(f"Full nested scientific run fold {(repeat_id, fold_id)} prediction IDs do not match its frozen outer test partition.")
        result, selected_a, selected_b = summary["final"]["result"], summary["selected_representation"], summary["selected_stage_b_candidate"]
        matrix = result.get("confusion_matrix")
        if not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(item, list) or len(item) != 2 for item in matrix):
            raise ValueError("Completed fold confusion matrix is invalid.")
        metrics = result.get("metrics", {})
        fold_rows.append({"repeat_id": repeat_id, "fold_id": fold_id, "selected_representation": selected_a["candidate_id"], "selected_classifier_family": selected_b["family"], "selected_classifier_configuration": selected_b["configuration_id"], "selected_imbalance_strategy": selected_b["strategy"], "selected_candidate_id": selected_b["candidate_id"], **{metric: metrics.get(metric) for metric in metric_names}, "tn": matrix[0][0], "fp": matrix[0][1], "fn": matrix[1][0], "tp": matrix[1][1]})
        prediction_rows.extend([{**item, "repeat_id": repeat_id, "fold_id": fold_id} for item in predictions])
        reliability = summary["final"]["reliability"]
        patients = reliability.get("patients", [])
        reliability_totals["supported_with_caution_count"] += sum(item.get("reliability_status") == "supported_with_caution" for item in patients)
        reliability_totals["inconclusive_count"] += sum(item.get("reliability_status") == "inconclusive" for item in patients)
        reliability_totals["low_margin_count"] += sum(bool(item.get("low_margin")) for item in patients)
        reliability_totals["latent_ood_count"] += sum(bool(item.get("latent_ood")) for item in patients)
        reliability_totals["joint_low_margin_and_ood_count"] += sum(bool(item.get("low_margin")) and bool(item.get("latent_ood")) for item in patients)
        fold_directories.append((repeat_id, fold_id, fold_dir))
    expected_records = len(data["sample_ids"]) * OUTER_N_REPEATS
    by_patient = {sample_id: [] for sample_id in data["sample_ids"]}
    for row in prediction_rows:
        if row.get("SAMPLE_ID") not in by_patient or not isinstance(row.get("repeat_id"), int) or not isinstance(row.get("fold_id"), int):
            raise ValueError("Aggregate prediction record provenance is invalid.")
        by_patient[row["SAMPLE_ID"]].append(row)
    if len(prediction_rows) != expected_records or len(by_patient) != len(data["sample_ids"]) or any(len(rows) != OUTER_N_REPEATS or {row["repeat_id"] for row in rows} != set(range(OUTER_N_REPEATS)) for rows in by_patient.values()):
        raise ValueError("Full nested scientific aggregate prediction coverage is invalid.")
    for repeat_id in range(OUTER_N_REPEATS):
        repeat_ids = [row["SAMPLE_ID"] for row in prediction_rows if row["repeat_id"] == repeat_id]
        if len(repeat_ids) != len(data["sample_ids"]) or set(repeat_ids) != set(data["sample_ids"]) or len(set(repeat_ids)) != len(repeat_ids):
            raise ValueError("Full nested scientific aggregate repeat coverage is invalid.")
    inner_candidate_rows = _phase10c_aggregate_inner_rows(fold_directories, "stage_b/candidate_results.jsonl", ["candidate_id"], ["candidate_id", "inner_fold_id"], "Stage B")
    representation_rows = _phase10c_aggregate_inner_rows(fold_directories, "stage_a/fold_results.jsonl", ["candidate_id"], ["candidate_id", "inner_fold_id", "family"], "Stage A")
    frequency_rows = []
    for column, label in [("selected_representation", "representation_candidate"), ("selected_classifier_family", "classifier_family"), ("selected_classifier_configuration", "classifier_configuration"), ("selected_imbalance_strategy", "imbalance_strategy"), ("selected_candidate_id", "complete_selected_candidate_id")]:
        values = [row[column] for row in fold_rows]
        for value in sorted(set(values)):
            count = values.count(value)
            frequency_rows.append({"selection_dimension": label, "selection_value": value, "count": count, "percentage": 100.0 * count / len(fold_rows)})
    confusion = {name: int(sum(row[name] for row in fold_rows)) for name in ["tn", "fp", "fn", "tp"]}
    aggregate_dir = output_dir / "aggregate"
    required_files = ["per_fold_results.csv", "metric_summary.csv", "selected_pipeline_per_fold.csv", "selection_frequencies.csv", "inner_candidate_comparison.csv", "representation_comparison.csv", "aggregate_confusion_matrix.csv", "reliability_summary.csv", "aggregate_outer_test_predictions.csv", "aggregate_outer_test_predictions.jsonl", "aggregate_manifest.json", "research_results_manifest.json"]
    def writer(directory: Path) -> None:
        _phase10c_write_csv(directory / "per_fold_results.csv", list(fold_rows[0]), fold_rows)
        _phase10c_write_csv(directory / "metric_summary.csv", ["metric", "count", "mean", "sample_standard_deviation_ddof_1", "minimum", "maximum"], _phase10c_metric_summary_rows(fold_rows, metric_names))
        _phase10c_write_csv(directory / "selected_pipeline_per_fold.csv", ["repeat_id", "fold_id", "selected_representation", "selected_classifier_family", "selected_classifier_configuration", "selected_imbalance_strategy", "selected_candidate_id"], [{key: row[key] for key in ["repeat_id", "fold_id", "selected_representation", "selected_classifier_family", "selected_classifier_configuration", "selected_imbalance_strategy", "selected_candidate_id"]} for row in fold_rows])
        _phase10c_write_csv(directory / "selection_frequencies.csv", ["selection_dimension", "selection_value", "count", "percentage"], frequency_rows)
        comparison_fields = list(inner_candidate_rows[0])
        _phase10c_write_csv(directory / "inner_candidate_comparison.csv", comparison_fields, inner_candidate_rows)
        _phase10c_write_csv(directory / "representation_comparison.csv", list(representation_rows[0]), representation_rows)
        _phase10c_write_csv(directory / "aggregate_confusion_matrix.csv", ["tn", "fp", "fn", "tp"], [confusion])
        _phase10c_write_csv(directory / "reliability_summary.csv", ["prediction_record_count", *reliability_totals], [{"prediction_record_count": len(prediction_rows), **reliability_totals}])
        _phase10c_write_csv(directory / "aggregate_outer_test_predictions.csv", list(prediction_rows[0]), prediction_rows)
        _phase10c_write_compact_jsonl(directory / "aggregate_outer_test_predictions.jsonl", prediction_rows)
        manifest = {"schema_version": "phase10c-full-nested-aggregate-v2", "binding": root_binding, "outer_fold_count": len(fold_rows), "prediction_record_count": len(prediction_rows), "aggregate_outer_test_predictions_sha256": _sha256_file(directory / "aggregate_outer_test_predictions.jsonl"), "candidate_comparison_interpretation": "inner-validation model-selection evidence; not final outer-test performance", **PHASE10C_FULL_GUARDS}
        _phase10c_atomic_write_json(directory / "aggregate_manifest.json", manifest)
        research = {"schema_version": "phase10c-research-results-v1", "binding": root_binding, "research_eligible": True, "smoke_runtime_overrides_applied": False, "outer_fold_count_expected": OUTER_N_SPLITS * OUTER_N_REPEATS, "outer_fold_count_completed": len(fold_rows), "outer_test_evaluation_count": len(fold_rows), "outer_test_evaluation_count_per_fold": 1, "additional_outer_test_scoring_calls_during_aggregation": 0, "prediction_record_count": len(prediction_rows), "candidate_comparison_scope": "inner_validation_only", "candidate_comparison_interpretation": "inner-validation model-selection evidence; not final outer-test performance", "probabilities_calibrated": False, "threshold_optimized": False, "direct_gene_attribution_from_final_model": False, "aggregate_confusion_matrix": confusion, "aggregate_manifest_sha256": _phase10c_hash(manifest), **PHASE10C_FULL_GUARDS}
        _phase10c_atomic_write_json(directory / "research_results_manifest.json", research)
    if not aggregate_dir.exists():
        _phase10c_publish_directory(aggregate_dir, writer)
    else:
        missing = [name for name in required_files if not (aggregate_dir / name).is_file()]
        if missing:
            temporary = aggregate_dir.parent / (".aggregate-repair-" + uuid.uuid4().hex)
            _phase10c_publish_directory(temporary, writer)
            for name in missing:
                os.replace(temporary / name, aggregate_dir / name)
            shutil.rmtree(temporary)
    aggregate_manifest = _phase10c_read_json(aggregate_dir / "aggregate_manifest.json", "Phase 10C full aggregate")
    research_manifest = _phase10c_read_json(aggregate_dir / "research_results_manifest.json", "Phase 10C research results")
    predictions_path = aggregate_dir / "aggregate_outer_test_predictions.jsonl"
    if aggregate_manifest.get("binding") != root_binding or aggregate_manifest.get("outer_fold_count") != len(fold_rows) or aggregate_manifest.get("prediction_record_count") != len(prediction_rows) or _sha256_file(predictions_path) != aggregate_manifest.get("aggregate_outer_test_predictions_sha256") or research_manifest.get("additional_outer_test_scoring_calls_during_aggregation") != 0:
        raise ValueError("Phase 10C full aggregate contract is invalid.")
    return aggregate_manifest


def run_nested_selection_full(data: Dict[str, Any], outer_manifest: Dict[str, Any], inner_manifest: Dict[str, Any], search: Dict[str, Any], output_dir: Path, outer_sha256: str, inner_sha256: str, search_sha256: str) -> Dict[str, Any]:
    """Run every frozen outer fold and atomically publish the complete scientific aggregate."""
    profile_contract = build_phase10c_full_execution_profile_contract(search, outer_sha256, inner_sha256, search_sha256)
    profile_hash = _phase10c_hash(profile_contract)
    root_binding = {"execution_profile_hash": profile_hash, "outer_manifest_sha256": outer_sha256, "inner_manifest_sha256": inner_sha256, "search_space_manifest_sha256": search_sha256, "outer_fold_count": OUTER_N_SPLITS * OUTER_N_REPEATS}
    guard_token = PHASE10C_ACTIVE_GUARDS.set(PHASE10C_FULL_GUARDS)
    root_lock = acquire_phase10c_run_lock(output_dir, root_binding)
    outcome = "failed"
    try:
        profile = {**profile_contract, "execution_profile_hash": profile_hash, "runtime_device_evidence": _phase10c_runtime_evidence()}
        profile_path = output_dir / "execution_profile.json"
        if profile_path.exists() and _phase10c_read_json(profile_path, "Phase 10C full execution profile").get("execution_profile_hash") != profile_hash:
            raise ValueError("Phase 10C full output directory execution profile does not match requested run.")
        if not profile_path.exists():
            _phase10c_atomic_write_json(profile_path, profile)
        _phase10c_atomic_write_json(output_dir / "run_contract.json", {"schema_version": PHASE10C_SCHEMA_VERSION, "binding": root_binding, "manifest_validation": {"outer": True, "inner": True, "search": True}, **PHASE10C_FULL_GUARDS})
        fold_summaries = []
        for outer_record in sorted(outer_manifest["folds"], key=lambda item: (item["repeat_id"], item["fold_id"])):
            repeat_id, fold_id = int(outer_record["repeat_id"]), int(outer_record["fold_id"])
            fold_dir = output_dir / "folds" / f"repeat_{repeat_id}_fold_{fold_id}"
            resumed = _phase10c_resume_completed_fold(fold_dir, profile_hash, outer_sha256, inner_sha256, search_sha256, repeat_id, fold_id)
            fold_summaries.append(resumed if resumed is not None else _run_phase10c_nested_selection_fold(data, outer_manifest, inner_manifest, search, fold_dir, repeat_id, fold_id, outer_sha256, inner_sha256, search_sha256, profile_contract, PHASE10C_FULL_GUARDS, "full_nested_scientific_run"))
        expected_pairs = {(repeat_id, fold_id) for repeat_id in range(OUTER_N_REPEATS) for fold_id in range(OUTER_N_SPLITS)}
        actual_pairs = {(summary["binding"]["repeat_id"], summary["binding"]["fold_id"]) for summary in fold_summaries}
        if actual_pairs != expected_pairs or len(fold_summaries) != len(expected_pairs):
            raise ValueError("Full nested scientific run did not produce exactly one summary for each outer fold.")
        aggregate_manifest = _phase10c_build_full_aggregate(data, outer_manifest, output_dir, root_binding, fold_summaries)
        summary = {"action": "full_nested_scientific_run", "binding": root_binding, "fold_count": len(fold_summaries), "prediction_record_count": len(data["sample_ids"]) * OUTER_N_REPEATS, "aggregate_manifest": aggregate_manifest, **PHASE10C_FULL_GUARDS}
        _phase10c_atomic_write_json(output_dir / "run_summary.json", summary)
        outcome = "completed"
        return summary
    finally:
        release_phase10c_run_lock(root_lock, outcome)
        PHASE10C_ACTIVE_GUARDS.reset(guard_token)


# =========================================
# SECTION 3: Encoder / latent feature extraction
# =========================================
# What: train per-modality autoencoders and extract latent features
# Why: preserve original latent-space modeling intent from notebook
# Input: preprocessed modality matrices
# Output: trained encoder models and concatenated latent representation

"""
This section is the feature learning part of the pipeline.

Instead of feeding the raw RNA, DNA, and CNA features directly into the final classifier, the script first compresses each modality into a smaller learned representation called a latent vector.

That is why this section is called latent feature extraction.

Big idea of Section 3

For each omics modality:

RNA gets its own autoencoder
DNA gets its own autoencoder
CNA gets its own autoencoder

Each autoencoder learns how to compress the input into a smaller hidden representation.

Then the script keeps only the encoder part and uses it to produce latent features.

Finally, it concatenates the three latent vectors into one combined feature vector.

So the flow is:

preprocessed RNA / DNA / CNA
→ encode each one separately
→ get smaller latent vectors
→ join them together
→ send to logistic regression
"""


def build_autoencoder(input_dim: int, latent_dim: int) -> Tuple[Model, Model]:
    inputs = Input(shape=(input_dim,))
    x = Dense(max(32, input_dim), activation="relu")(inputs)
    latent = Dense(latent_dim, activation="linear", name="latent")(x)
    x = Dense(max(32, input_dim), activation="relu")(latent)
    outputs = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inputs=inputs, outputs=outputs)
    encoder = Model(inputs=inputs, outputs=latent)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


"""
This function trains an autoencoder and returns only the encoder.
"""


def fit_encoder(X_train: np.ndarray, X_val: np.ndarray, latent_dim: int) -> Model:
    autoencoder, encoder = build_autoencoder(
        input_dim=X_train.shape[1], latent_dim=latent_dim
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    return encoder


"""
This function takes the trained encoders and uses them to transform full datasets into latent features.
"""


def transform_latent(
    encoder_rna: Model,
    encoder_dna: Model,
    encoder_cna: Model,
    X_rna: np.ndarray,
    X_dna: np.ndarray,
    X_cna: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    z_rna = encoder_rna.predict(X_rna, verbose=0)
    z_dna = encoder_dna.predict(X_dna, verbose=0)
    z_cna = encoder_cna.predict(X_cna, verbose=0)
    latent = np.concatenate([z_rna, z_dna, z_cna], axis=1)

    start = 0
    modality_slices = {}
    for name, z in [("mGE", z_rna), ("mDM", z_dna), ("CNA", z_cna)]:
        modality_slices[name] = (start, start + z.shape[1])
        start += z.shape[1]
    return latent, modality_slices


# =========================================
# SECTION 4: Augmentation (train-only, non-leaky)
# =========================================
# What: balance only the training fold in latent space
# Why: prevent leakage from synthetic data into held-out evaluation
# Input: latent train matrix + train labels
# Output: augmented latent train matrix and labels

"""
This section fixes class imbalance.

That means if one class has fewer samples than the other, the script creates extra training samples for the minority class so the classifier does not become biased toward the majority class.

Very important: it does this only on the training set, not on the test set.

That is why the comment says train-only, non-leaky.
"""


def augment_minority_class(
    X_latent_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    classes, counts = np.unique(y_train, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    minority_count = counts[np.argmin(counts)]
    majority_count = counts[np.argmax(counts)]
    needed = max(0, majority_count - minority_count)

    if needed == 0:
        return (
            X_latent_train,
            y_train,
            {
                "augmentation_source": "none",
                "minority_class": int(minority_class),
                "added_samples": 0,
            },
        )

    minority_data = X_latent_train[y_train == minority_class]

    synthetic_samples: np.ndarray
    augmentation_source = "bootstrap_fallback"

    try:
        import importlib

        sdv_metadata_module = importlib.import_module("sdv.metadata")
        sdv_single_table_module = importlib.import_module("sdv.single_table")
        SingleTableMetadata = getattr(sdv_metadata_module, "SingleTableMetadata")
        CTGANSynthesizer = getattr(sdv_single_table_module, "CTGANSynthesizer")

        latent_df = pd.DataFrame(minority_data)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=latent_df)
        ctgan = CTGANSynthesizer(
            metadata=metadata, epochs=100, batch_size=min(500, len(latent_df))
        )
        ctgan.fit(latent_df)
        synthetic_df = ctgan.sample(needed)
        synthetic_samples = np.asarray(synthetic_df, dtype=np.float32)
        augmentation_source = "ctgan"
    except Exception:
        # If SDV/CTGAN is unavailable, we keep augmentation deterministic and simple.
        synthetic_samples = np.asarray(
            resample(
                minority_data,
                replace=True,
                n_samples=needed,
                random_state=random_state,
            ),
            dtype=np.float32,
        )
        noise = np.random.normal(0, 0.01, size=synthetic_samples.shape).astype(
            np.float32
        )
        synthetic_samples = synthetic_samples + noise

    X_aug = np.vstack([X_latent_train, synthetic_samples])
    y_aug = np.concatenate(
        [y_train, np.full(shape=(needed,), fill_value=minority_class)]
    )

    perm = np.random.permutation(len(X_aug))
    X_aug = X_aug[perm]
    y_aug = y_aug[perm]

    return (
        X_aug,
        y_aug,
        {
            "augmentation_source": augmentation_source,
            "minority_class": int(minority_class),
            "majority_class": int(majority_class),
            "added_samples": int(needed),
        },
    )


# =========================================
# SECTION 5: Prediction model and helper metrics
# =========================================
# What: logistic classifier + optional disagreement ensemble
# Why: interpretable latent contributions and robust reliability checks
# Input: augmented latent train data
# Output: fitted classifier and ensemble members

"""
🔹 Big idea of Section 5

This section introduces:

1. Main model

👉 Logistic Regression (your classifier)

2. Ensemble models

👉 Multiple versions of the model to check disagreement

3. Helper functions

👉 For measuring:

distance (OOD detection)
data completeness
"""


def train_disagreement_ensemble(
    X_train: np.ndarray, y_train: np.ndarray, n_models: int = 5
) -> List[LogisticRegression]:
    ensemble = []
    n = len(X_train)
    for i in range(n_models):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42 + i
        )
        clf.fit(X_train[idx], y_train[idx])
        ensemble.append(clf)
    return ensemble


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    diff = x - mean
    return float(np.sqrt(np.maximum(0.0, diff @ inv_cov @ diff.T)))


def compute_data_completeness(
    row_rna: pd.DataFrame,
    row_dna: pd.DataFrame,
    row_cna: pd.DataFrame,
) -> float:
    total = row_rna.shape[1] + row_dna.shape[1] + row_cna.shape[1]
    non_missing = (
        row_rna.notna().sum(axis=1).iloc[0]
        + row_dna.notna().sum(axis=1).iloc[0]
        + row_cna.notna().sum(axis=1).iloc[0]
    )
    return float(non_missing / total)


# =========================================
# SECTION 6A: Gene-level explainer (Option 1)
# =========================================
# What: train a secondary interpretable model in preprocessed feature space
# Why: provide top contributing genes/features while keeping latent model untouched


def build_gene_feature_names(feature_columns: Dict[str, List[str]]) -> List[str]:
    names = []
    names.extend([f"mGE::{c}" for c in feature_columns["rna"]])
    names.extend([f"mDM::{c}" for c in feature_columns["dna"]])
    names.extend([f"CNA::{c}" for c in feature_columns["cna"]])
    return names


def stack_preprocessed_modalities(
    X_rna: np.ndarray,
    X_dna: np.ndarray,
    X_cna: np.ndarray,
) -> np.ndarray:
    return np.concatenate([X_rna, X_dna, X_cna], axis=1)


def train_gene_explainer(
    X_gene_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    explainer = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.5,
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
    )
    explainer.fit(X_gene_train, y_train)
    return explainer


def explain_top_genes_from_explainer(
    gene_vector: np.ndarray,
    prob_high_tmb: float,
    gene_explainer: LogisticRegression,
    gene_feature_names: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    coef = gene_explainer.coef_.reshape(-1)
    contributions = gene_vector.reshape(-1) * coef

    top_indices = np.argsort(np.abs(contributions))[-top_k:][::-1]
    top_genes = []
    for idx in top_indices:
        contribution = float(contributions[idx])
        direction = "supports_high_tmb" if contribution >= 0 else "supports_low_tmb"
        top_genes.append(
            {
                "feature_name": gene_feature_names[int(idx)],
                "contribution": round(contribution, 6),
                "direction": direction,
            }
        )

    predicted_class = "High-TMB" if prob_high_tmb >= 0.5 else "Low-TMB"
    confidence = float(max(prob_high_tmb, 1.0 - prob_high_tmb))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "top_contributing_genes": top_genes,
        "gene_explanation_summary": (
            "Top contributors are produced by a secondary L1-logistic explainer trained "
            "in preprocessed feature space. Primary prediction still comes from the latent-space model."
        ),
    }


# =========================================
# SECTION 6: Explainability layer
# =========================================
# What: honest latent-space explanation and modality contribution ranking
# Why: avoid misleading raw-gene explanations when model consumes latent features
# Input: latent vector, classifier, modality slices
# Output: structured explanation dictionary


"""
This section explains why the model made a prediction.

But it does it in a careful way.

It does not pretend to explain raw genes directly, because the final classifier is not using raw genes.
It is using latent features from the autoencoders.

That is why this section is actually very good scientifically: it avoids fake or misleading explanations.

So instead of saying:

❌ “Gene X caused this prediction” (misleading)

It correctly says:

✅ “Latent components and modalities contributed”

"""


def explain_latent_prediction(
    latent_vector: np.ndarray,
    prob_high_tmb: float,
    classifier: LogisticRegression,
    modality_slices: Dict[str, Tuple[int, int]],
    top_k: int = 5,
) -> Dict[str, Any]:
    # ==================================================
    # ADDED BLOCK -- EXPLAINABILITY MODULE
    # Purpose: produce honest explanation outputs for latent-space prediction
    # Where to place: after classifier training and before final inference report assembly
    # ==================================================
    # Important scientific note:
    # The classifier consumes latent components, not raw genes directly.
    # Therefore, reporting "Top 5 Genes" here would be misleading unless a validated
    # latent-to-gene attribution pipeline is implemented. We expose top latent signals
    # and modality-level contributions instead.

    coef = classifier.coef_.reshape(-1)
    contributions = latent_vector.reshape(-1) * coef

    top_indices = np.argsort(np.abs(contributions))[-top_k:][::-1]
    top_signals = []
    for idx in top_indices:
        contribution = float(contributions[idx])
        direction = "supports_high_tmb" if contribution >= 0 else "supports_low_tmb"
        top_signals.append(
            {
                "latent_component": f"z_{int(idx)}",
                "contribution": round(contribution, 6),
                "direction": direction,
            }
        )

    modality_scores = {}
    for modality, (start, end) in modality_slices.items():
        modality_scores[modality] = float(np.sum(np.abs(contributions[start:end])))

    dominant_omics = [
        k
        for k, _ in sorted(modality_scores.items(), key=lambda kv: kv[1], reverse=True)
    ]

    predicted_class = "High-TMB" if prob_high_tmb >= 0.5 else "Low-TMB"
    confidence = float(max(prob_high_tmb, 1.0 - prob_high_tmb))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "dominant_omics": dominant_omics,
        "top_contributing_signals": top_signals,
        "modality_contribution_scores": modality_scores,
        "explanation_summary": (
            "Prediction rationale is provided in latent/component space, "
            "with modality-level contribution ranking. Raw gene ranking is intentionally "
            "disabled to avoid misleading attribution in this latent-space pipeline."
        ),
    }


# =========================================
# SECTION 7: Reliability / uncertainty / data quality layer
# =========================================
# What: confidence, inconclusive zone, data completeness, OOD risk, disagreement
# Why: model must identify weak-evidence cases and recommend confirmation
# Input: probability + latent vector + training distribution references
# Output: structured reliability dictionary


"""
This section is one of the most important parts of the whole script.

Why?
Because a model should not only say:

“Here is my prediction”

It should also say:

“How trustworthy is this prediction?”
“Is this sample unusual?”
“Is the input incomplete?”
“Are models disagreeing?”
“Should this be treated with caution?”

That is exactly what Section 7 does.
"""


def reliability_assessment(
    prob_high_tmb: float,
    data_completeness: float,
    latent_vector: np.ndarray,
    train_mean: np.ndarray,
    train_inv_cov: np.ndarray,
    ood_threshold: float,
    disagreement_models: List[LogisticRegression],
    inconclusive_prob_low: float = 0.40,
    inconclusive_prob_high: float = 0.60,
    disagreement_std_threshold: float = 0.10,
) -> Dict[str, Any]:
    # ==================================================
    # ADDED BLOCK -- RELIABILITY / UNCERTAINTY MODULE
    # Purpose: add safety framing and detect weak-confidence / weak-quality cases
    # Where to place: after explanation generation and before clinical output assembly
    # ==================================================

    # CHANGED:
    # Inconclusive zone is probability-based around 0.4-0.6 as requested.
    # This directly captures borderline model belief near the decision boundary.
    confidence = float(max(prob_high_tmb, 1.0 - prob_high_tmb))
    is_inconclusive = inconclusive_prob_low <= prob_high_tmb <= inconclusive_prob_high

    if data_completeness < 0.80:
        data_quality_warning = "high_missingness_detected"
    elif data_completeness < 0.95:
        data_quality_warning = "moderate_missingness_detected"
    else:
        data_quality_warning = "none"

    sample_dist = mahalanobis_distance(
        latent_vector.reshape(-1), train_mean, train_inv_cov
    )
    if sample_dist >= ood_threshold:
        ood_risk = "high"
    elif sample_dist >= 0.85 * ood_threshold:
        ood_risk = "moderate"
    else:
        ood_risk = "low"

    # Optional disagreement logic: practical here using bootstrap logistic ensemble.
    ensemble_probs = [
        m.predict_proba(latent_vector.reshape(1, -1))[0, 1] for m in disagreement_models
    ]
    disagreement_std = float(np.std(ensemble_probs))
    model_disagreement_flag = disagreement_std >= disagreement_std_threshold

    caution_reasons = []
    if is_inconclusive:
        caution_reasons.append("borderline_confidence")
    if data_quality_warning != "none":
        caution_reasons.append(data_quality_warning)
    if ood_risk in {"moderate", "high"}:
        caution_reasons.append(f"ood_risk_{ood_risk}")
    if model_disagreement_flag:
        caution_reasons.append("model_disagreement")

    if len(caution_reasons) == 0:
        reliability_status = "Reliable"
        reliability_summary = "Prediction confidence and sample quality appear acceptable for decision-support use."
        recommendation = "Use as decision-support only; correlate with full clinical context and standard testing."
    else:
        reliability_status = "Caution"
        reliability_summary = (
            "Prediction carries caution flags: " + ", ".join(caution_reasons) + "."
        )
        recommendation = (
            "Treat as inconclusive/at-risk decision-support output; perform confirmatory review "
            "with orthogonal clinical or molecular evidence."
        )

    return {
        "confidence": round(confidence, 6),
        "is_inconclusive": bool(is_inconclusive),
        "data_completeness": round(data_completeness, 6),
        "data_quality_warning": data_quality_warning,
        "out_of_distribution_risk": ood_risk,
        "mahalanobis_distance": round(sample_dist, 6),
        "ood_threshold": round(float(ood_threshold), 6),
        "model_disagreement_std": round(disagreement_std, 6),
        "model_disagreement_flag": bool(model_disagreement_flag),
        "reliability_status": reliability_status,
        "warnings": caution_reasons,
        "reliability_summary": reliability_summary,
        "recommendation_for_confirmation": recommendation,
    }


def build_clinical_output(
    prob_high_tmb: float,
    explanation: Dict[str, Any],
    reliability: Dict[str, Any],
    gene_explanation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # =========================================
    # SECTION 8: Clinical decision-support report
    # =========================================
    # What: convert model outputs to medically cautious report format
    # Why: frontend and clinical users need structured, careful interpretation
    # Input: probability + explainability output + reliability output
    # Output: JSON-serializable report dictionary

    # ==================================================
    # ADDED BLOCK -- CLINICAL DECISION-SUPPORT OUTPUT
    # Purpose: build a structured inference-ready report object for later project integration
    # Where to place: final stage of inference after explainability + reliability are computed
    # ==================================================

    """
        This section takes the outputs from the earlier sections and turns them into a final report object that is ready for:

    frontend display
    JSON export
    API response
    safer clinical-style interpretation

    So before this section, the script has separate pieces:

    prediction probability
    explainability info
    reliability info

    Section 8 combines them into one final structured report.
    """

    prediction = "High-TMB" if prob_high_tmb >= 0.5 else "Low-TMB"
    predicted_tmb_category = (
        "Inconclusive" if reliability["is_inconclusive"] else prediction
    )

    if prediction == "High-TMB":
        clinical_interpretation = (
            "This profile may indicate higher tumor mutational burden signal in latent multi-omics space. "
            "This output is decision-support only and should be clinically validated."
        )
    else:
        clinical_interpretation = (
            "This profile may indicate lower tumor mutational burden signal in latent multi-omics space. "
            "This output is decision-support only and should be clinically validated."
        )

    if gene_explanation is None:
        top_genes = []
        gene_explanation_summary = (
            "No gene-level explainer output available for this run."
        )
    else:
        top_genes = gene_explanation["top_contributing_genes"]
        gene_explanation_summary = gene_explanation["gene_explanation_summary"]

    return {
        "prediction": prediction,
        "predicted_tmb_category": predicted_tmb_category,
        "probability_high_tmb": round(float(prob_high_tmb), 6),
        "confidence": reliability["confidence"],
        "inconclusive": reliability["is_inconclusive"],
        "data_completeness": reliability["data_completeness"],
        "data_quality_warning": reliability["data_quality_warning"],
        "out_of_distribution_risk": reliability["out_of_distribution_risk"],
        "main_contributing_omics": explanation["dominant_omics"],
        "top_model_signals": explanation["top_contributing_signals"],
        "top_contributing_genes": top_genes,
        "gene_explanation_summary": gene_explanation_summary,
        "dominant_omics": explanation["dominant_omics"],
        "top_contributing_signals": explanation["top_contributing_signals"],
        "explanation_summary": explanation["explanation_summary"],
        "reliability_status": reliability["reliability_status"],
        "warnings": reliability["warnings"],
        "clinical_interpretation": clinical_interpretation,
        "reliability_summary": reliability["reliability_summary"],
        "recommendation": reliability["recommendation_for_confirmation"],
        "disclaimer": "Decision-support only. Requires clinical confirmation.",
    }


# =========================================
# SECTION 9: Evaluation + inference helper
# =========================================
# What: reusable single-sample inference function from loaded artifacts
# Why: clean integration path for your main project runtime
# Input: loaded bundle + one-row DataFrames for each omics modality
# Output: one structured clinical report dictionary

"""
This section is very important for deployment/integration.

Earlier sections were mostly about:

training
explanation
reliability
report building

Section 9 answers a different question:

After I save the trained artifacts, how do I use them later on one new sample?

That is exactly what this section does.
"""


def build_report_from_loaded_bundle(
    loaded_bundle: Dict[str, Any],
    raw_rna_row: pd.DataFrame,
    raw_dna_row: pd.DataFrame,
    raw_cna_row: pd.DataFrame,
) -> Dict[str, Any]:
    pre_rna = loaded_bundle["preprocessors"]["rna"]
    pre_dna = loaded_bundle["preprocessors"]["dna"]
    pre_cna = loaded_bundle["preprocessors"]["cna"]

    feature_cols = loaded_bundle["feature_columns"]
    raw_rna_row = pd.DataFrame(raw_rna_row.loc[:, feature_cols["rna"]])
    raw_dna_row = pd.DataFrame(raw_dna_row.loc[:, feature_cols["dna"]])
    raw_cna_row = pd.DataFrame(raw_cna_row.loc[:, feature_cols["cna"]])

    x_rna = pre_rna.transform(raw_rna_row)
    x_dna = pre_dna.transform(raw_dna_row)
    x_cna = pre_cna.transform(raw_cna_row)
    x_gene = stack_preprocessed_modalities(x_rna, x_dna, x_cna)

    encoder_rna = loaded_bundle["encoders"]["rna"]
    encoder_dna = loaded_bundle["encoders"]["dna"]
    encoder_cna = loaded_bundle["encoders"]["cna"]

    latent, _ = transform_latent(
        encoder_rna, encoder_dna, encoder_cna, x_rna, x_dna, x_cna
    )
    latent_vector = latent[0]

    classifier = loaded_bundle["classifier"]
    prob_high = float(classifier.predict_proba(latent)[0, 1])

    explanation = explain_latent_prediction(
        latent_vector=latent_vector,
        prob_high_tmb=prob_high,
        classifier=classifier,
        modality_slices=loaded_bundle["modality_slices"],
        top_k=loaded_bundle["explainability_config"]["top_k_signals"],
    )

    ood_stats = loaded_bundle["ood_stats"]
    rel_cfg = loaded_bundle["reliability_config"]
    reliability = reliability_assessment(
        prob_high_tmb=prob_high,
        data_completeness=compute_data_completeness(
            raw_rna_row, raw_dna_row, raw_cna_row
        ),
        latent_vector=latent_vector,
        train_mean=ood_stats["train_mean"],
        train_inv_cov=ood_stats["train_inv_cov"],
        ood_threshold=ood_stats["ood_threshold"],
        disagreement_models=loaded_bundle["disagreement_models"],
        inconclusive_prob_low=rel_cfg["inconclusive_prob_low"],
        inconclusive_prob_high=rel_cfg["inconclusive_prob_high"],
        disagreement_std_threshold=rel_cfg["disagreement_std_threshold"],
    )

    gene_explanation = None
    if (
        "gene_explainer_model" in loaded_bundle
        and "gene_feature_names" in loaded_bundle
    ):
        gene_explanation = explain_top_genes_from_explainer(
            gene_vector=x_gene[0],
            prob_high_tmb=prob_high,
            gene_explainer=loaded_bundle["gene_explainer_model"],
            gene_feature_names=loaded_bundle["gene_feature_names"],
            top_k=loaded_bundle.get("explainability_config", {}).get(
                "top_k_signals", 5
            ),
        )

    return build_clinical_output(
        prob_high_tmb=prob_high,
        explanation=explanation,
        reliability=reliability,
        gene_explanation=gene_explanation,
    )


# =========================================
# SECTION 10: SAVING ARTIFACTS
# =========================================
# What: robust export strategy for mixed sklearn + Keras objects
# Why: raw pickling Keras models is fragile across environments


def save_export_bundle(
    export_dir: Path,
    encoder_rna: Model,
    encoder_dna: Model,
    encoder_cna: Model,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    # ==================================================
    # ADDED BLOCK -- EXPORT / PKL PREPARATION
    # Purpose: prepare bundled inference artifact for use in another project
    # Where to place: after model training/validation and before integration handoff
    # ==================================================
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras encoders separately for robust TF loading in other projects.
    # Keeping them outside the PKL avoids brittle TensorFlow pickling behavior.
    encoder_rna_path = export_dir / "encoder_rna.keras"
    encoder_dna_path = export_dir / "encoder_dna.keras"
    encoder_cna_path = export_dir / "encoder_cna.keras"
    encoder_rna.save(encoder_rna_path)
    encoder_dna.save(encoder_dna_path)
    encoder_cna.save(encoder_cna_path)

    bundle_path = export_dir / "oncoassist_inference_bundle.pkl"
    bundle_payload = {
        **payload,
        "encoder_paths": {
            "rna": encoder_rna_path.name,
            "dna": encoder_dna_path.name,
            "cna": encoder_cna_path.name,
        },
    }
    # What is saved in PKL:
    # - fitted preprocessors
    # - fitted classifier + disagreement ensemble
    # - label mapping
    # - latent modality layout and OOD statistics
    # - reliability / explainability configuration
    # - metadata required for future inference assembly
    joblib.dump(bundle_payload, bundle_path)

    return {
        "bundle_pkl": str(bundle_path),
        "encoder_rna": str(encoder_rna_path),
        "encoder_dna": str(encoder_dna_path),
        "encoder_cna": str(encoder_cna_path),
    }


# =========================================
# SECTION 11: LOADING FOR INFERENCE
# =========================================
# What: load PKL metadata plus Keras encoder files
# Why: reconstruct full inference pipeline in deployment


def load_export_bundle(export_dir: Path) -> Dict[str, Any]:
    bundle_path = export_dir / "oncoassist_inference_bundle.pkl"
    payload = joblib.load(bundle_path)
    payload["encoders"] = {
        "rna": tf.keras.models.load_model(export_dir / payload["encoder_paths"]["rna"]),
        "dna": tf.keras.models.load_model(export_dir / payload["encoder_paths"]["dna"]),
        "cna": tf.keras.models.load_model(export_dir / payload["encoder_paths"]["cna"]),
    }
    return payload


# =========================================
# SECTION 12: End-to-end training and run
# =========================================
# What: train, evaluate, explain, assess reliability, report, save, reload
# Why: one-command reproducible model development flow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or audit the released BLCA multi-omics training data."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate the training-data contract and print its JSON summary without training.",
    )
    mode_group.add_argument(
        "--generate-outer-fold-manifest",
        type=Path,
        metavar="PATH",
        help="Generate and validate a deterministic outer-fold manifest without training.",
    )
    mode_group.add_argument(
        "--audit-outer-fold-manifest",
        type=Path,
        metavar="PATH",
        help="Validate an existing deterministic outer-fold manifest without training.",
    )
    mode_group.add_argument(
        "--generate-inner-fold-manifest",
        type=Path,
        nargs=2,
        metavar=("OUTER_PATH", "INNER_PATH"),
        help="Generate deterministic nested inner folds without preprocessing or training.",
    )
    mode_group.add_argument(
        "--audit-inner-fold-manifest",
        type=Path,
        nargs=2,
        metavar=("OUTER_PATH", "INNER_PATH"),
        help="Validate deterministic nested inner folds without preprocessing or training.",
    )
    mode_group.add_argument(
        "--generate-nested-search-space-manifest",
        type=Path,
        nargs=3,
        metavar=("OUTER_PATH", "INNER_PATH", "OUTPUT_PATH"),
        help="Generate a prespecified nested search-space manifest without training.",
    )
    mode_group.add_argument(
        "--audit-nested-search-space-manifest",
        type=Path,
        nargs=3,
        metavar=("OUTER_PATH", "INNER_PATH", "MANIFEST_PATH"),
        help="Validate a prespecified nested search-space manifest without training.",
    )
    mode_group.add_argument(
        "--run-nested-selection-smoke",
        type=Path,
        nargs=4,
        metavar=("OUTER_PATH", "INNER_PATH", "SEARCH_SPACE_PATH", "OUTPUT_DIR"),
        help="Run one locked, resumable nested-selection integration smoke for one outer fold.",
    )
    mode_group.add_argument(
        "--run-nested-selection-full",
        type=Path,
        nargs=4,
        metavar=("OUTER_PATH", "INNER_PATH", "SEARCH_SPACE_PATH", "OUTPUT_DIR"),
        help="Run the complete locked 15-fold scientific nested-selection protocol.",
    )
    mode_group.add_argument(
        "--run-legacy-80-20",
        action="store_true",
        help="Run the legacy non-nested 80/20 workflow; not research-eligible.",
    )
    mode_group.add_argument(
        "--audit-preprocessing",
        type=Path,
        metavar="PATH",
        help="Materialize one outer fold and audit train-only preprocessing without training.",
    )
    mode_group.add_argument(
        "--audit-autoencoder",
        type=Path,
        metavar="PATH",
        help="Train per-modality fold autoencoders and audit named-bottleneck latents only.",
    )
    mode_group.add_argument(
        "--audit-latent-fusion",
        type=Path,
        metavar="PATH",
        help="Train smoke autoencoders, fuse their latents, and audit fusion only.",
    )
    mode_group.add_argument(
        "--audit-minority-ctgan",
        type=Path,
        metavar="PATH",
        help="Train mandatory minority-only CTGAN in fused latent space and audit it only.",
    )
    mode_group.add_argument(
        "--audit-conditional-ctgan",
        type=Path,
        metavar="PATH",
        help="Train mandatory conditional CTGAN in fused latent space and audit it only.",
    )
    mode_group.add_argument(
        "--audit-ctgan-quality",
        type=Path,
        metavar="PATH",
        help="Run both CTGAN smoke strategies and audit train-only synthetic quality only.",
    )
    mode_group.add_argument(
        "--audit-logistic-classifier",
        type=Path,
        metavar="PATH",
        help="Run one-fold fixed Logistic Regression smoke evaluation for all approved training variants.",
    )
    mode_group.add_argument(
        "--audit-imbalance-baselines",
        type=Path,
        metavar="PATH",
        help="Run one-fold Logistic Regression imbalance-baseline evaluation only.",
    )
    mode_group.add_argument("--audit-classifier-registry", type=Path, metavar="PATH", help="Run one-fold real-only classifier-family smoke evaluation.")
    mode_group.add_argument("--audit-classifier-pipelines", type=Path, metavar="PATH", help="Run one-fold real-only model-specific classifier-pipeline smoke evaluation.")
    parser.add_argument("--repeat-id", type=int, help="Outer-fold repeat identifier.")
    parser.add_argument("--fold-id", type=int, help="Outer-fold fold identifier.")
    parser.add_argument("--mge-latent-dim", type=int, help="Smoke-audit mGE bottleneck width.")
    parser.add_argument("--mdm-latent-dim", type=int, help="Smoke-audit mDM bottleneck width.")
    parser.add_argument("--mcna-latent-dim", type=int, help="Smoke-audit mCNA bottleneck width.")
    parser.add_argument("--ae-epochs", type=int, help="Smoke-audit autoencoder epoch limit.")
    parser.add_argument(
        "--ae-hidden-dims",
        help="Comma-separated smoke-audit hidden widths; copied into each modality configuration.",
    )
    parser.add_argument("--ae-batch-size", type=int, help="Smoke-audit autoencoder batch size.")
    parser.add_argument("--ae-patience", type=int, help="Smoke-audit early-stopping patience.")
    parser.add_argument(
        "--inner-validation-fraction",
        type=float,
        help="Outer-training fraction reserved for the deterministic inner validation split.",
    )
    parser.add_argument("--inner-split-seed", type=int, help="Deterministic inner split/order seed.")
    parser.add_argument("--ae-seed", type=int, help="Base seed for independent modality autoencoders.")
    parser.add_argument("--ctgan-epochs", type=int, help="Smoke-audit CTGAN epoch count.")
    parser.add_argument("--ctgan-batch-size", type=int, help="Smoke-audit CTGAN batch size.")
    parser.add_argument("--ctgan-pac", type=int, help="Smoke-audit CTGAN pac value.")
    parser.add_argument(
        "--ctgan-verbose", action="store_true", help="Enable CTGAN smoke-audit training output."
    )
    args = parser.parse_args()

    execution_modes = [
        args.audit_only,
        args.generate_outer_fold_manifest,
        args.audit_outer_fold_manifest,
        args.generate_inner_fold_manifest,
        args.audit_inner_fold_manifest,
        args.generate_nested_search_space_manifest,
        args.audit_nested_search_space_manifest,
        args.run_nested_selection_smoke,
        args.run_nested_selection_full,
        args.run_legacy_80_20,
        args.audit_preprocessing,
        args.audit_autoencoder,
        args.audit_latent_fusion,
        args.audit_minority_ctgan,
        args.audit_conditional_ctgan,
        args.audit_ctgan_quality,
        args.audit_logistic_classifier,
        args.audit_imbalance_baselines,
        args.audit_classifier_registry,
        args.audit_classifier_pipelines,
    ]
    if not any(execution_modes):
        parser.error(
            "An explicit execution mode is required. The legacy 80/20 workflow is available only with --run-legacy-80-20."
        )

    base_dir = Path(__file__).resolve().parent
    rna_path = base_dir / "mGE.csv"
    dna_path = base_dir / "mDM.csv"
    cna_path = base_dir / "mCNA.csv"

    data = load_and_align_multiomics(
        rna_path=rna_path, dna_path=dna_path, cna_path=cna_path
    )
    if args.audit_only:
        print(json.dumps(data["audit_summary"], indent=2))
        return
    if args.generate_outer_fold_manifest:
        data_fingerprint = build_outer_data_fingerprint(data)
        manifest = build_outer_fold_manifest(
            data["sample_ids"], data["y_binary"], data_fingerprint
        )
        validation = validate_outer_fold_manifest(
            manifest, data["sample_ids"], data["y_binary"], data_fingerprint
        )
        output = write_outer_fold_manifest(manifest, args.generate_outer_fold_manifest)
        print(
            json.dumps(
                {"action": "generated", **output, "validation": validation}, indent=2
            )
        )
        return
    if args.audit_outer_fold_manifest:
        data_fingerprint = build_outer_data_fingerprint(data)
        manifest_path = args.audit_outer_fold_manifest
        manifest = load_outer_fold_manifest(manifest_path)
        validation = validate_outer_fold_manifest(
            manifest, data["sample_ids"], data["y_binary"], data_fingerprint
        )
        payload = manifest_path.read_bytes()
        print(
            json.dumps(
                {
                    "action": "audited",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                    "manifest_size_bytes": len(payload),
                    "validation": validation,
                },
                indent=2,
            )
        )
        return
    if args.generate_inner_fold_manifest:
        outer_path, inner_path = args.generate_inner_fold_manifest
        data_fingerprint = build_outer_data_fingerprint(data)
        outer_manifest = load_outer_fold_manifest(outer_path)
        outer_validation = validate_outer_fold_manifest(
            outer_manifest, data["sample_ids"], data["y_binary"], data_fingerprint
        )
        outer_payload = outer_path.read_bytes()
        inner_manifest = build_inner_fold_manifest(
            outer_manifest,
            hashlib.sha256(outer_payload).hexdigest(),
            data["sample_ids"],
            data["y_binary"],
            data_fingerprint,
        )
        validation = validate_inner_fold_manifest(
            inner_manifest,
            outer_manifest,
            hashlib.sha256(outer_payload).hexdigest(),
            data["sample_ids"],
            data["y_binary"],
            data_fingerprint,
        )
        output = write_inner_fold_manifest(inner_manifest, inner_path)
        print(json.dumps({"action": "generated_inner_folds", "outer_manifest_validation": outer_validation, **output, "validation": validation}, indent=2))
        return
    if args.audit_inner_fold_manifest:
        outer_path, inner_path = args.audit_inner_fold_manifest
        data_fingerprint = build_outer_data_fingerprint(data)
        outer_manifest = load_outer_fold_manifest(outer_path)
        outer_validation = validate_outer_fold_manifest(
            outer_manifest, data["sample_ids"], data["y_binary"], data_fingerprint
        )
        outer_payload = outer_path.read_bytes()
        inner_manifest = load_inner_fold_manifest(inner_path)
        validation = validate_inner_fold_manifest(
            inner_manifest,
            outer_manifest,
            hashlib.sha256(outer_payload).hexdigest(),
            data["sample_ids"],
            data["y_binary"],
            data_fingerprint,
        )
        inner_payload = inner_path.read_bytes()
        print(json.dumps({"action": "audited_inner_folds", "outer_manifest_validation": outer_validation, "inner_manifest_path": str(inner_path), "inner_manifest_sha256": hashlib.sha256(inner_payload).hexdigest(), "inner_manifest_size_bytes": len(inner_payload), "validation": validation}, indent=2))
        return
    if args.generate_nested_search_space_manifest:
        outer_path, inner_path, output_path = args.generate_nested_search_space_manifest
        fingerprint = build_outer_data_fingerprint(data)
        outer_manifest, inner_manifest = load_outer_fold_manifest(outer_path), load_inner_fold_manifest(inner_path)
        outer_payload, inner_payload = outer_path.read_bytes(), inner_path.read_bytes()
        outer_sha256, inner_sha256 = hashlib.sha256(outer_payload).hexdigest(), hashlib.sha256(inner_payload).hexdigest()
        outer_validation = validate_outer_fold_manifest(outer_manifest, data["sample_ids"], data["y_binary"], fingerprint)
        inner_validation = validate_inner_fold_manifest(inner_manifest, outer_manifest, outer_sha256, data["sample_ids"], data["y_binary"], fingerprint)
        manifest = build_nested_search_space_manifest(data, outer_manifest, outer_sha256, inner_manifest, inner_sha256)
        validation = validate_nested_search_space_manifest(manifest, data, outer_manifest, outer_sha256, inner_manifest, inner_sha256)
        output = write_nested_search_space_manifest(manifest, output_path)
        print(json.dumps({"action": "generated_nested_search_space", "outer_manifest_validation": outer_validation, "inner_manifest_validation": inner_validation, **output, "validation": validation}, indent=2))
        return
    if args.audit_nested_search_space_manifest:
        outer_path, inner_path, manifest_path = args.audit_nested_search_space_manifest
        fingerprint = build_outer_data_fingerprint(data)
        outer_manifest, inner_manifest, manifest = load_outer_fold_manifest(outer_path), load_inner_fold_manifest(inner_path), load_nested_search_space_manifest(manifest_path)
        outer_payload, inner_payload, manifest_payload = outer_path.read_bytes(), inner_path.read_bytes(), manifest_path.read_bytes()
        outer_sha256, inner_sha256 = hashlib.sha256(outer_payload).hexdigest(), hashlib.sha256(inner_payload).hexdigest()
        outer_validation = validate_outer_fold_manifest(outer_manifest, data["sample_ids"], data["y_binary"], fingerprint)
        inner_validation = validate_inner_fold_manifest(inner_manifest, outer_manifest, outer_sha256, data["sample_ids"], data["y_binary"], fingerprint)
        validation = validate_nested_search_space_manifest(manifest, data, outer_manifest, outer_sha256, inner_manifest, inner_sha256)
        print(json.dumps({"action": "audited_nested_search_space", "outer_manifest_validation": outer_validation, "inner_manifest_validation": inner_validation, "manifest_path": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(), "manifest_size_bytes": len(manifest_payload), "validation": validation}, indent=2))
        return
    if args.run_nested_selection_smoke:
        if args.repeat_id is None or args.fold_id is None:
            parser.error("--run-nested-selection-smoke requires --repeat-id and --fold-id.")
        outer_path, inner_path, search_path, output_dir = args.run_nested_selection_smoke
        fingerprint = build_outer_data_fingerprint(data)
        outer_manifest = load_outer_fold_manifest(outer_path)
        inner_manifest = load_inner_fold_manifest(inner_path)
        search_manifest = load_nested_search_space_manifest(search_path)
        outer_sha256, inner_sha256, search_sha256 = _sha256_file(outer_path), _sha256_file(inner_path), _sha256_file(search_path)
        validate_outer_fold_manifest(outer_manifest, data["sample_ids"], data["y_binary"], fingerprint)
        validate_inner_fold_manifest(inner_manifest, outer_manifest, outer_sha256, data["sample_ids"], data["y_binary"], fingerprint)
        validate_nested_search_space_manifest(search_manifest, data, outer_manifest, outer_sha256, inner_manifest, inner_sha256)
        result = run_nested_selection_smoke(data, outer_manifest, inner_manifest, search_manifest, output_dir, args.repeat_id, args.fold_id, outer_sha256, inner_sha256, search_sha256)
        print(json.dumps(_phase10c_jsonable(result), indent=2))
        return
    if args.run_nested_selection_full:
        outer_path, inner_path, search_path, output_dir = args.run_nested_selection_full
        fingerprint = build_outer_data_fingerprint(data)
        outer_manifest = load_outer_fold_manifest(outer_path)
        inner_manifest = load_inner_fold_manifest(inner_path)
        search_manifest = load_nested_search_space_manifest(search_path)
        outer_sha256, inner_sha256, search_sha256 = _sha256_file(outer_path), _sha256_file(inner_path), _sha256_file(search_path)
        validate_outer_fold_manifest(outer_manifest, data["sample_ids"], data["y_binary"], fingerprint)
        validate_inner_fold_manifest(inner_manifest, outer_manifest, outer_sha256, data["sample_ids"], data["y_binary"], fingerprint)
        validate_nested_search_space_manifest(search_manifest, data, outer_manifest, outer_sha256, inner_manifest, inner_sha256)
        result = run_nested_selection_full(data, outer_manifest, inner_manifest, search_manifest, output_dir, outer_sha256, inner_sha256, search_sha256)
        print(json.dumps(_phase10c_jsonable(result), indent=2))
        return
    if args.audit_preprocessing:
        if args.repeat_id is None or args.fold_id is None:
            parser.error("--audit-preprocessing requires --repeat-id and --fold-id.")
        manifest, manifest_validation = load_validated_outer_fold_manifest(
            data, args.audit_preprocessing
        )
        materialized_fold = materialize_outer_fold(
            data, manifest, args.repeat_id, args.fold_id
        )
        transformed_modalities = fit_fold_preprocessors(materialized_fold)
        preprocessing_audit = build_fold_preprocessing_audit(
            manifest_validation, materialized_fold, transformed_modalities
        )
        print(json.dumps(preprocessing_audit, indent=2))
        return
    if (
        args.audit_autoencoder
        or args.audit_latent_fusion
        or args.audit_minority_ctgan
        or args.audit_conditional_ctgan
        or args.audit_ctgan_quality
        or args.audit_logistic_classifier
        or args.audit_imbalance_baselines
        or args.audit_classifier_registry
        or args.audit_classifier_pipelines
    ):
        audit_manifest_path = (
            args.audit_autoencoder
            if args.audit_autoencoder is not None
            else args.audit_latent_fusion
            if args.audit_latent_fusion is not None
            else args.audit_minority_ctgan
            if args.audit_minority_ctgan is not None
            else args.audit_conditional_ctgan
            if args.audit_conditional_ctgan is not None
            else args.audit_ctgan_quality
            if args.audit_ctgan_quality is not None
            else args.audit_logistic_classifier
            if args.audit_logistic_classifier is not None
            else args.audit_imbalance_baselines
            if args.audit_imbalance_baselines is not None
            else args.audit_classifier_registry
            if args.audit_classifier_registry is not None
            else args.audit_classifier_pipelines
        )
        audit_mode = (
            "--audit-autoencoder"
            if args.audit_autoencoder
            else "--audit-latent-fusion"
            if args.audit_latent_fusion
            else "--audit-minority-ctgan"
            if args.audit_minority_ctgan
            else "--audit-conditional-ctgan"
            if args.audit_conditional_ctgan
            else "--audit-ctgan-quality"
            if args.audit_ctgan_quality
            else "--audit-logistic-classifier"
            if args.audit_logistic_classifier
            else "--audit-imbalance-baselines"
            if args.audit_imbalance_baselines
            else "--audit-classifier-registry"
            if args.audit_classifier_registry
            else "--audit-classifier-pipelines"
        )
        required_arguments = {
            "--repeat-id": args.repeat_id,
            "--fold-id": args.fold_id,
            "--mge-latent-dim": args.mge_latent_dim,
            "--mdm-latent-dim": args.mdm_latent_dim,
            "--mcna-latent-dim": args.mcna_latent_dim,
            "--ae-epochs": args.ae_epochs,
            "--ae-hidden-dims": args.ae_hidden_dims,
            "--ae-batch-size": args.ae_batch_size,
            "--ae-patience": args.ae_patience,
            "--inner-validation-fraction": args.inner_validation_fraction,
            "--inner-split-seed": args.inner_split_seed,
            "--ae-seed": args.ae_seed,
        }
        if (
            args.audit_minority_ctgan
            or args.audit_conditional_ctgan
            or args.audit_ctgan_quality
            or args.audit_logistic_classifier
            or args.audit_imbalance_baselines
        ):
            required_arguments.update(
                {
                    "--ctgan-epochs": args.ctgan_epochs,
                    "--ctgan-batch-size": args.ctgan_batch_size,
                    "--ctgan-pac": args.ctgan_pac,
                }
            )
        missing_arguments = [name for name, value in required_arguments.items() if value is None]
        if missing_arguments:
            parser.error(
                audit_mode + " requires " + ", ".join(missing_arguments) + "."
            )
        try:
            hidden_dims = _parse_hidden_dims(args.ae_hidden_dims)
        except ValueError as error:
            parser.error(str(error))
        manifest, manifest_validation = load_validated_outer_fold_manifest(
            data, audit_manifest_path
        )
        materialized_fold = materialize_outer_fold(
            data, manifest, args.repeat_id, args.fold_id
        )
        transformed_modalities = fit_fold_preprocessors(materialized_fold)
        modality_configs = {
            "mGE": {"latent_dim": args.mge_latent_dim, "hidden_dims": list(hidden_dims)},
            "mDM": {"latent_dim": args.mdm_latent_dim, "hidden_dims": list(hidden_dims)},
            "mCNA": {"latent_dim": args.mcna_latent_dim, "hidden_dims": list(hidden_dims)},
        }
        try:
            autoencoder_result = fit_fold_autoencoders(
                materialized_fold,
                transformed_modalities,
                modality_configs,
                {
                    "epochs": args.ae_epochs,
                    "batch_size": args.ae_batch_size,
                    "patience": args.ae_patience,
                    "seed": args.ae_seed,
                },
                args.inner_validation_fraction,
                args.inner_split_seed,
            )
        except ValueError as error:
            parser.error(str(error))
        if args.audit_autoencoder:
            audit = build_fold_autoencoder_audit(
                manifest_validation, materialized_fold, autoencoder_result
            )
        elif args.audit_latent_fusion:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            audit = build_fold_latent_fusion_audit(
                manifest_validation, materialized_fold, fusion_result
            )
        elif args.audit_minority_ctgan:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            outer_test_features_before = fusion_result["fused_outer_test"].copy()
            outer_test_labels_before = materialized_fold["y_test"].copy()
            try:
                augmentation_result = build_minority_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    {
                        "epochs": args.ctgan_epochs,
                        "batch_size": args.ctgan_batch_size,
                        "pac": args.ctgan_pac,
                        "verbose": args.ctgan_verbose,
                    },
                    args.ae_seed,
                )
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
            outer_test_unchanged = bool(
                np.array_equal(fusion_result["fused_outer_test"], outer_test_features_before)
                and np.array_equal(materialized_fold["y_test"], outer_test_labels_before)
            )
            audit = build_minority_ctgan_audit(
                manifest_validation,
                materialized_fold,
                fusion_result,
                augmentation_result,
                outer_test_unchanged,
            )
        elif args.audit_conditional_ctgan:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            outer_test_features_before = fusion_result["fused_outer_test"].copy()
            outer_test_labels_before = materialized_fold["y_test"].copy()
            try:
                augmentation_result = build_conditional_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    {
                        "epochs": args.ctgan_epochs,
                        "batch_size": args.ctgan_batch_size,
                        "pac": args.ctgan_pac,
                        "verbose": args.ctgan_verbose,
                    },
                    args.ae_seed,
                )
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
            outer_test_unchanged = bool(
                np.array_equal(fusion_result["fused_outer_test"], outer_test_features_before)
                and np.array_equal(materialized_fold["y_test"], outer_test_labels_before)
            )
            audit = build_conditional_ctgan_audit(
                manifest_validation,
                materialized_fold,
                fusion_result,
                augmentation_result,
                outer_test_unchanged,
            )
        elif args.audit_ctgan_quality:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            outer_test_features_before = fusion_result["fused_outer_test"].copy()
            outer_test_labels_before = materialized_fold["y_test"].copy()
            ctgan_config = {
                "epochs": args.ctgan_epochs,
                "batch_size": args.ctgan_batch_size,
                "pac": args.ctgan_pac,
                "verbose": args.ctgan_verbose,
            }
            try:
                minority_result = build_minority_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    ctgan_config,
                    args.ae_seed,
                )
                conditional_result = build_conditional_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    ctgan_config,
                    args.ae_seed,
                )
                quality_result = compare_ctgan_synthetic_quality(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    minority_result,
                    conditional_result,
                )
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
            outer_test_unchanged = bool(
                np.array_equal(fusion_result["fused_outer_test"], outer_test_features_before)
                and np.array_equal(materialized_fold["y_test"], outer_test_labels_before)
            )
            audit = build_ctgan_quality_audit(
                manifest_validation, quality_result, outer_test_unchanged
            )
        elif args.audit_logistic_classifier:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            outer_test_features_before = fusion_result["fused_outer_test"].copy()
            outer_test_labels_before = fusion_result["y_test"].copy()
            outer_test_ids_before = list(fusion_result["test_sample_ids"])
            ctgan_config = {
                "epochs": args.ctgan_epochs,
                "batch_size": args.ctgan_batch_size,
                "pac": args.ctgan_pac,
                "verbose": args.ctgan_verbose,
            }
            try:
                minority_result = build_minority_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    ctgan_config,
                    args.ae_seed,
                )
                conditional_result = build_conditional_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    ctgan_config,
                    args.ae_seed,
                )
                training_variants = build_fold_logistic_training_variants(
                    fusion_result, minority_result, conditional_result
                )
                logistic_evaluation = evaluate_fold_logistic_variants(
                    fusion_result, training_variants
                )
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
            outer_test_unchanged = bool(
                np.array_equal(fusion_result["fused_outer_test"], outer_test_features_before)
                and np.array_equal(fusion_result["y_test"], outer_test_labels_before)
                and fusion_result["test_sample_ids"] == outer_test_ids_before
            )
            audit = build_logistic_classifier_audit(
                manifest_validation, logistic_evaluation, outer_test_unchanged
            )
        elif args.audit_imbalance_baselines:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            outer_test_features_before = fusion_result["fused_outer_test"].copy()
            outer_test_labels_before = fusion_result["y_test"].copy()
            outer_test_ids_before = list(fusion_result["test_sample_ids"])
            ctgan_config = {
                "epochs": args.ctgan_epochs,
                "batch_size": args.ctgan_batch_size,
                "pac": args.ctgan_pac,
                "verbose": args.ctgan_verbose,
            }
            try:
                minority_result = build_minority_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    ctgan_config,
                    args.ae_seed,
                )
                conditional_result = build_conditional_ctgan_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                    ctgan_config,
                    args.ae_seed,
                )
                smote_result = build_smote_latent_augmentation(
                    fusion_result["fused_outer_train"],
                    fusion_result["y_train"],
                    fusion_result["train_sample_ids"],
                    fusion_result["latent_feature_names"],
                    fusion_result["latent_feature_name_sha256"],
                    fusion_result["repeat_id"],
                    fusion_result["fold_id"],
                )
                training_variants = build_fold_imbalance_baseline_variants(
                    fusion_result, minority_result, conditional_result, smote_result
                )
                imbalance_evaluation = evaluate_fold_imbalance_baselines(
                    fusion_result, training_variants
                )
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
            outer_test_unchanged = bool(
                np.array_equal(fusion_result["fused_outer_test"], outer_test_features_before)
                and np.array_equal(fusion_result["y_test"], outer_test_labels_before)
                and fusion_result["test_sample_ids"] == outer_test_ids_before
            )
            audit = build_imbalance_baselines_audit(
                manifest_validation, imbalance_evaluation, outer_test_unchanged
            )
        elif args.audit_classifier_registry:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            test_before = fusion_result["fused_outer_test"].copy()
            labels_before = fusion_result["y_test"].copy()
            ids_before = list(fusion_result["test_sample_ids"])
            try:
                phase9a_variant = build_phase9a_real_only_training_variant(fusion_result)
                phase9a_result = evaluate_phase9a_classifier_registry(fusion_result, phase9a_variant)
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
            unchanged = bool(np.array_equal(fusion_result["fused_outer_test"], test_before) and np.array_equal(fusion_result["y_test"], labels_before) and fusion_result["test_sample_ids"] == ids_before)
            audit = build_phase9a_classifier_registry_audit(manifest_validation, fusion_result, phase9a_result, unchanged)
        else:
            fusion_result = fuse_fold_latents(materialized_fold, autoencoder_result)
            test_before = fusion_result["fused_outer_test"].copy()
            labels_before = fusion_result["y_test"].copy()
            ids_before = list(fusion_result["test_sample_ids"])
            try:
                phase9b_variant = build_phase9b_real_only_training_variant(fusion_result)
                phase9b_result = evaluate_phase9b_classifier_pipelines(fusion_result, phase9b_variant)
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
            unchanged = bool(np.array_equal(fusion_result["fused_outer_test"], test_before) and np.array_equal(fusion_result["y_test"], labels_before) and fusion_result["test_sample_ids"] == ids_before)
            audit = build_phase9b_classifier_pipelines_audit(manifest_validation, phase9b_result, unchanged)
        print(json.dumps(audit, indent=2))
        return

    if not args.run_legacy_80_20:
        raise RuntimeError("Unreachable legacy 80/20 workflow without --run-legacy-80-20.")
    print("LEGACY 80/20 WORKFLOW — NON-NESTED AND NOT RESEARCH-ELIGIBLE")

    X_rna = data["X_rna"]
    X_dna = data["X_dna"]
    X_cna = data["X_cna"]
    y = data["y_binary"]

    split_idx_train, split_idx_test = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_rna_train_df, X_rna_test_df = (
        X_rna.iloc[split_idx_train],
        X_rna.iloc[split_idx_test],
    )
    X_dna_train_df, X_dna_test_df = (
        X_dna.iloc[split_idx_train],
        X_dna.iloc[split_idx_test],
    )
    X_cna_train_df, X_cna_test_df = (
        X_cna.iloc[split_idx_train],
        X_cna.iloc[split_idx_test],
    )
    y_train, y_test = y[split_idx_train], y[split_idx_test]

    pre_rna = build_preprocessor()
    pre_dna = build_preprocessor()
    pre_cna = build_preprocessor()

    X_rna_train = pre_rna.fit_transform(X_rna_train_df)
    X_dna_train = pre_dna.fit_transform(X_dna_train_df)
    X_cna_train = pre_cna.fit_transform(X_cna_train_df)

    X_rna_test = pre_rna.transform(X_rna_test_df)
    X_dna_test = pre_dna.transform(X_dna_test_df)
    X_cna_test = pre_cna.transform(X_cna_test_df)

    X_gene_train = stack_preprocessed_modalities(X_rna_train, X_dna_train, X_cna_train)
    X_gene_test = stack_preprocessed_modalities(X_rna_test, X_dna_test, X_cna_test)
    gene_feature_names = build_gene_feature_names(data["feature_columns"])
    gene_explainer_model = train_gene_explainer(X_gene_train, y_train)

    latent_rna_dim = min(16, X_rna_train.shape[1])
    latent_dna_dim = min(16, X_dna_train.shape[1])
    latent_cna_dim = min(8, X_cna_train.shape[1])

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=42, stratify=y_train
    )

    encoder_rna = fit_encoder(
        X_rna_train[train_idx], X_rna_train[val_idx], latent_rna_dim
    )
    encoder_dna = fit_encoder(
        X_dna_train[train_idx], X_dna_train[val_idx], latent_dna_dim
    )
    encoder_cna = fit_encoder(
        X_cna_train[train_idx], X_cna_train[val_idx], latent_cna_dim
    )

    X_latent_train, modality_slices = transform_latent(
        encoder_rna, encoder_dna, encoder_cna, X_rna_train, X_dna_train, X_cna_train
    )
    X_latent_test, _ = transform_latent(
        encoder_rna, encoder_dna, encoder_cna, X_rna_test, X_dna_test, X_cna_test
    )

    X_train_aug, y_train_aug, aug_info = augment_minority_class(X_latent_train, y_train)

    classifier = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    )
    classifier.fit(X_train_aug, y_train_aug)

    disagreement_models = train_disagreement_ensemble(
        X_train_aug, y_train_aug, n_models=5
    )

    y_test_prob = classifier.predict_proba(X_latent_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_test_prob)),
        "classification_report": classification_report(
            y_test, y_test_pred, output_dict=True
        ),
    }

    # OOD statistics from training latent distribution
    cov = np.cov(X_latent_train.T)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    train_mean = X_latent_train.mean(axis=0)
    train_inv_cov = np.linalg.pinv(cov)
    train_distances = np.array(
        [mahalanobis_distance(x, train_mean, train_inv_cov) for x in X_latent_train]
    )
    ood_threshold = float(np.quantile(train_distances, 0.95))

    # Use first held-out sample as demonstration for block-by-block outputs
    demo_idx = 0
    demo_prob = float(y_test_prob[demo_idx])
    demo_latent = X_latent_test[demo_idx]
    demo_gene_vector = X_gene_test[demo_idx]

    demo_completeness = compute_data_completeness(
        X_rna_test_df.iloc[[demo_idx]],
        X_dna_test_df.iloc[[demo_idx]],
        X_cna_test_df.iloc[[demo_idx]],
    )

    explanation = explain_latent_prediction(
        latent_vector=demo_latent,
        prob_high_tmb=demo_prob,
        classifier=classifier,
        modality_slices=modality_slices,
        top_k=5,
    )

    gene_explanation = explain_top_genes_from_explainer(
        gene_vector=demo_gene_vector,
        prob_high_tmb=demo_prob,
        gene_explainer=gene_explainer_model,
        gene_feature_names=gene_feature_names,
        top_k=5,
    )

    reliability = reliability_assessment(
        prob_high_tmb=demo_prob,
        data_completeness=demo_completeness,
        latent_vector=demo_latent,
        train_mean=train_mean,
        train_inv_cov=train_inv_cov,
        ood_threshold=ood_threshold,
        disagreement_models=disagreement_models,
    )

    clinical_output = build_clinical_output(
        prob_high_tmb=demo_prob,
        explanation=explanation,
        reliability=reliability,
        gene_explanation=gene_explanation,
    )

    export_dir = base_dir / "export_artifacts"
    export_paths = save_export_bundle(
        export_dir=export_dir,
        encoder_rna=encoder_rna,
        encoder_dna=encoder_dna,
        encoder_cna=encoder_cna,
        payload={
            "preprocessors": {"rna": pre_rna, "dna": pre_dna, "cna": pre_cna},
            "classifier": classifier,
            "gene_explainer_model": gene_explainer_model,
            "gene_feature_names": gene_feature_names,
            "disagreement_models": disagreement_models,
            "feature_columns": data["feature_columns"],
            "label_mapping": data["label_mapping"],
            "modality_slices": modality_slices,
            "reliability_config": {
                "inconclusive_prob_low": 0.40,
                "inconclusive_prob_high": 0.60,
                "disagreement_std_threshold": 0.10,
                "ood_quantile": 0.95,
            },
            "explainability_config": {
                "top_k_signals": 5,
                "note": "Latent-space and modality-level explanation only; no raw top-gene attribution.",
                "gene_explainer_note": "Top contributing genes/features are from secondary L1-logistic explainer in preprocessed feature space.",
            },
            "ood_stats": {
                "train_mean": train_mean,
                "train_inv_cov": train_inv_cov,
                "ood_threshold": ood_threshold,
            },
            "augmentation_info": aug_info,
            "training_metrics": metrics,
            "demo_output": clinical_output,
        },
    )

    # Verify artifact loading
    loaded_bundle = load_export_bundle(export_dir)
    loaded_keys = sorted([k for k in loaded_bundle.keys() if k != "encoders"])

    inference_report = build_report_from_loaded_bundle(
        loaded_bundle=loaded_bundle,
        raw_rna_row=X_rna_test_df.iloc[[demo_idx]],
        raw_dna_row=X_dna_test_df.iloc[[demo_idx]],
        raw_cna_row=X_cna_test_df.iloc[[demo_idx]],
    )

    print("=== TRAINING / EVAL SUMMARY ===")
    print(json.dumps({"metrics": metrics, "augmentation_info": aug_info}, indent=2))

    print("\n=== AFTER EXPLAINABILITY BLOCK ===")
    print(json.dumps(explanation, indent=2))

    print("\n=== AFTER TOP-5 GENE EXPLAINER BLOCK ===")
    print(json.dumps(gene_explanation, indent=2))

    print("\n=== AFTER RELIABILITY BLOCK ===")
    print(json.dumps(reliability, indent=2))

    print("\n=== AFTER CLINICAL OUTPUT BLOCK ===")
    print(json.dumps(clinical_output, indent=2))

    print("\n=== AFTER EXPORT BLOCK ===")
    print(
        json.dumps(
            {
                "export_paths": export_paths,
                "loaded_bundle_keys": loaded_keys,
                "inference_report_from_loaded_bundle": inference_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
