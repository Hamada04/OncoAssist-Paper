"""Validated BLCA data-contract logic extracted from reference/current_working_source.py.

Scientific behavior must be preserved. This module does not reconstruct or assert
the unresolved numeric TMB cutoff or feature-selection provenance.
"""

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_TRAINING_COLUMNS = {"SAMPLE_ID", "CLASS"}
FORBIDDEN_FEATURE_COLUMNS = {"SAMPLE_ID", "CLASS"}
PROJECT_LABEL_MAPPING = {
    "raw_to_binary": {"1": 0, "2": 1},
    "binary_to_tmb": {0: "Low-TMB", 1: "High-TMB"},
    "raw_semantics": {"1": "Low-TMB", "2": "High-TMB"},
    "mapping_basis": (
        "Project-configured mapping based on the released BLCA dataset; "
        "no numeric clinical TMB cutoff is asserted."
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_and_validate_training_table(
    modality: str, path: Path
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
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
    feature_columns: dict[str, list[str]], feature_matrices: dict[str, pd.DataFrame]
) -> dict[str, Any]:
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
    rna_path: Path, dna_path: Path, cna_path: Path
) -> dict[str, Any]:
    """Load and validate the three released BLCA training tables."""
    table_specs = {
        "mGE": (rna_path, "rna"),
        "mDM": (dna_path, "dna"),
        "CNA": (cna_path, "cna"),
    }
    tables: dict[str, pd.DataFrame] = {}
    feature_columns: dict[str, list[str]] = {}
    audit_files: dict[str, dict[str, Any]] = {}
    sample_id_sets: dict[str, set[str]] = {}

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
