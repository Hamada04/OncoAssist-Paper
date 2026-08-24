import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from oncoassist_research import data


class DataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_csv(
        self, name: str, rows: list[list[object]], headers: list[str] | None = None
    ) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers or ["SAMPLE_ID", "CLASS", "FEATURE"])
            writer.writerows(rows)
        return path

    def write_valid_modalities(
        self,
        rna_rows: list[list[object]] | None = None,
        dna_rows: list[list[object]] | None = None,
        cna_rows: list[list[object]] | None = None,
    ) -> tuple[Path, Path, Path]:
        return (
            self.write_csv(
                "mGE.csv",
                rna_rows or [["S-2", 2, 2.5], ["S-1", 1, 1.5]],
            ),
            self.write_csv(
                "mDM.csv",
                dna_rows or [["S-1", 1, 10.5], ["S-2", 2, 20.5]],
            ),
            self.write_csv(
                "mCNA.csv",
                cna_rows or [["S-2", 2, 200.5], ["S-1", 1, 100.5]],
            ),
        )

    def load_valid_data(self) -> dict[str, object]:
        return data.load_and_align_multiomics(*self.write_valid_modalities())

    def test_valid_three_modality_loading_and_alignment(self) -> None:
        loaded = self.load_valid_data()
        self.assertEqual(loaded["sample_ids"], ["S-1", "S-2"])
        self.assertEqual(loaded["audit_summary"]["sample_alignment"]["passed"], True)
        self.assertEqual(loaded["audit_summary"]["sample_alignment"]["aligned_sample_id_count"], 2)

    def test_class_one_maps_to_binary_zero(self) -> None:
        rna, dna, cna = self.write_valid_modalities(
            rna_rows=[["S-1", 1, 1.5]],
            dna_rows=[["S-1", 1, 10.5]],
            cna_rows=[["S-1", 1, 100.5]],
        )
        loaded = data.load_and_align_multiomics(rna, dna, cna)
        self.assertEqual(loaded["y_binary"].tolist(), [0])

    def test_class_two_maps_to_binary_one(self) -> None:
        rna, dna, cna = self.write_valid_modalities(
            rna_rows=[["S-1", 2, 1.5]],
            dna_rows=[["S-1", 2, 10.5]],
            cna_rows=[["S-1", 2, 100.5]],
        )
        loaded = data.load_and_align_multiomics(rna, dna, cna)
        self.assertEqual(loaded["y_binary"].tolist(), [1])

    def test_sample_id_is_excluded_from_feature_columns(self) -> None:
        loaded = self.load_valid_data()
        for columns in loaded["feature_columns"].values():
            self.assertNotIn("SAMPLE_ID", columns)

    def test_class_is_excluded_from_feature_columns(self) -> None:
        loaded = self.load_valid_data()
        for columns in loaded["feature_columns"].values():
            self.assertNotIn("CLASS", columns)

    def test_different_modality_row_orders_are_aligned(self) -> None:
        loaded = self.load_valid_data()
        self.assertEqual(loaded["X_rna"].index.tolist(), ["S-1", "S-2"])
        self.assertEqual(loaded["X_dna"].index.tolist(), ["S-1", "S-2"])
        self.assertEqual(loaded["X_cna"].index.tolist(), ["S-1", "S-2"])

    def test_duplicate_sample_id_raises_error(self) -> None:
        rna, dna, cna = self.write_valid_modalities(
            rna_rows=[["S-1", 1, 1.5], ["S-1", 1, 2.5]]
        )
        with self.assertRaisesRegex(ValueError, "duplicate SAMPLE_ID"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_blank_sample_id_raises_error(self) -> None:
        rna, dna, cna = self.write_valid_modalities(rna_rows=[[" ", 1, 1.5]])
        with self.assertRaisesRegex(ValueError, "null or blank SAMPLE_ID"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_missing_sample_id_column_raises_error(self) -> None:
        _, dna, cna = self.write_valid_modalities()
        rna = self.write_csv("mGE.csv", [[1, 1.5]], ["CLASS", "FEATURE"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_missing_class_column_raises_error(self) -> None:
        _, dna, cna = self.write_valid_modalities()
        rna = self.write_csv("mGE.csv", [["S-1", 1.5]], ["SAMPLE_ID", "FEATURE"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_invalid_class_values_raise_error(self) -> None:
        rna, dna, cna = self.write_valid_modalities(rna_rows=[["S-1", 3, 1.5]])
        with self.assertRaisesRegex(ValueError, "released BLCA labels 1 and 2"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_non_numeric_biological_feature_raises_error(self) -> None:
        rna, dna, cna = self.write_valid_modalities(rna_rows=[["S-1", 1, "not-numeric"]])
        with self.assertRaisesRegex(ValueError, "contains non-numeric values"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_infinite_biological_feature_raises_error(self) -> None:
        rna, dna, cna = self.write_valid_modalities(rna_rows=[["S-1", 1, "inf"]])
        with self.assertRaisesRegex(ValueError, "infinite values"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_missing_biological_feature_values_are_allowed(self) -> None:
        rna, dna, cna = self.write_valid_modalities(
            rna_rows=[["S-1", 1, ""]],
            dna_rows=[["S-1", 1, 10.5]],
            cna_rows=[["S-1", 1, 100.5]],
        )
        loaded = data.load_and_align_multiomics(rna, dna, cna)
        self.assertTrue(np.isnan(loaded["X_rna"].iloc[0, 0]))
        self.assertEqual(loaded["audit_summary"]["files"]["mGE"]["missing_value_count"], 1)

    def test_mismatched_class_labels_raise_error(self) -> None:
        rna, dna, cna = self.write_valid_modalities(
            rna_rows=[["S-1", 1, 1.5]],
            dna_rows=[["S-1", 2, 10.5]],
            cna_rows=[["S-1", 1, 100.5]],
        )
        with self.assertRaisesRegex(ValueError, "CLASS must agree"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_duplicate_csv_header_names_are_rejected(self) -> None:
        _, dna, cna = self.write_valid_modalities()
        rna = self.write_csv(
            "mGE.csv",
            [["S-1", 1, 1.5, 2.5]],
            ["SAMPLE_ID", "CLASS", "FEATURE", "FEATURE"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            data.load_and_align_multiomics(rna, dna, cna)

    def test_target_leakage_guard_rejects_forbidden_columns(self) -> None:
        for forbidden_column in ("CLASS", "SAMPLE_ID"):
            with self.subTest(location="feature list", column=forbidden_column):
                with self.assertRaises(AssertionError):
                    data._assert_target_leakage_guards({"rna": [forbidden_column]}, {})
            with self.subTest(location="feature matrix", column=forbidden_column):
                with self.assertRaises(AssertionError):
                    data._assert_target_leakage_guards(
                        {}, {"rna": pd.DataFrame({forbidden_column: [1.0]})}
                    )

    def test_sha256_audit_metadata_is_deterministic_for_unchanged_file(self) -> None:
        paths = self.write_valid_modalities()
        first = data.load_and_align_multiomics(*paths)
        second = data.load_and_align_multiomics(*paths)
        self.assertEqual(
            first["audit_summary"]["files"]["mGE"]["sha256"],
            second["audit_summary"]["files"]["mGE"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
