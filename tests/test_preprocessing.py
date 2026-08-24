import inspect
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from oncoassist_research import preprocessing
from oncoassist_research.artifacts import payload_sha256


class PreprocessingTests(unittest.TestCase):
    def training_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature_a": [1.0, 2.0, 3.0], "feature_b": [10.0, np.nan, 30.0]},
            index=["TRAIN-1", "TRAIN-2", "TRAIN-3"],
        )

    def fit(self) -> preprocessing.FittedPreprocessor:
        frame = self.training_frame()
        return preprocessing.fit_preprocessor(
            frame, frame.index.tolist(), ["feature_a", "feature_b"]
        )

    def test_training_only_imputation_scaling_and_extreme_heldout_leakage(self) -> None:
        fitted = self.fit()
        imputer = fitted._pipeline.named_steps["imputer"]
        scaler = fitted._pipeline.named_steps["scaler"]
        np.testing.assert_allclose(imputer.statistics_, [2.0, 20.0])
        np.testing.assert_allclose(scaler.mean_, [2.0, 20.0])
        np.testing.assert_allclose(scaler.var_, [2.0 / 3.0, 200.0 / 3.0])
        statistics_before = imputer.statistics_.copy()
        mean_before, variance_before, scale_before = scaler.mean_.copy(), scaler.var_.copy(), scaler.scale_.copy()
        for values in ([1_000_000.0, -1_000_000.0], [-1_000_000.0, 1_000_000.0]):
            heldout = pd.DataFrame([values], columns=fitted.feature_names, index=["HELDOUT"])
            preprocessing.transform_with_preprocessor(fitted, heldout, ["HELDOUT"], fitted.feature_names)
        np.testing.assert_array_equal(imputer.statistics_, statistics_before)
        np.testing.assert_array_equal(scaler.mean_, mean_before)
        np.testing.assert_array_equal(scaler.var_, variance_before)
        np.testing.assert_array_equal(scaler.scale_, scale_before)

    def test_heldout_missing_values_use_training_medians_and_outputs_are_finite(self) -> None:
        fitted = self.fit()
        heldout = pd.DataFrame(
            {"feature_a": [np.nan], "feature_b": [np.nan]}, index=["HELDOUT"]
        )
        result = preprocessing.transform_with_preprocessor(
            fitted, heldout, ["HELDOUT"], ["feature_a", "feature_b"]
        )
        np.testing.assert_allclose(result.matrix, [[0.0, 0.0]])
        self.assertEqual(result.matrix.dtype, np.float32)
        self.assertTrue(np.isfinite(result.matrix).all())

    def test_feature_and_patient_order_shape_are_preserved(self) -> None:
        fitted = self.fit()
        heldout = pd.DataFrame(
            {"feature_a": [4.0, 5.0], "feature_b": [40.0, 50.0]},
            index=["HELDOUT-2", "HELDOUT-1"],
        )
        result = preprocessing.transform_with_preprocessor(
            fitted, heldout, ["HELDOUT-2", "HELDOUT-1"], fitted.feature_names
        )
        self.assertEqual(result.sample_ids, ("HELDOUT-2", "HELDOUT-1"))
        self.assertEqual(result.feature_names, fitted.feature_names)
        self.assertEqual(result.matrix.shape, (2, 2))

    def test_schema_mismatch_rejections(self) -> None:
        fitted = self.fit()
        cases = [
            pd.DataFrame({"feature_b": [1.0], "feature_a": [2.0]}, index=["H"]),
            pd.DataFrame({"feature_a": [1.0]}, index=["H"]),
            pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0], "extra": [3.0]}, index=["H"]),
        ]
        for frame in cases:
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaises(ValueError):
                    preprocessing.transform_with_preprocessor(fitted, frame, ["H"], fitted.feature_names)

    def test_target_columns_are_rejected_for_fit_and_transform(self) -> None:
        for forbidden in ("CLASS", "SAMPLE_ID"):
            frame = pd.DataFrame({"feature_a": [1.0], forbidden: [1.0]}, index=["ID"])
            with self.subTest(operation="fit", forbidden=forbidden):
                with self.assertRaises(ValueError):
                    preprocessing.fit_preprocessor(frame, ["ID"], frame.columns.tolist())
            with self.subTest(operation="transform", forbidden=forbidden):
                with self.assertRaises(ValueError):
                    preprocessing.transform_with_preprocessor(self.fit(), frame, ["ID"], frame.columns.tolist())

    def test_fit_input_validation(self) -> None:
        frame = self.training_frame()
        cases = [
            ("not-dataframe", ["A"], ["feature_a"]),
            (frame.iloc[0:0], [], ["feature_a", "feature_b"]),
            (frame, ["TRAIN-1"], ["feature_a", "feature_b"]),
            (frame, ["TRAIN-1", "TRAIN-1", "TRAIN-3"], ["feature_a", "feature_b"]),
            (frame, ["TRAIN-1", " ", "TRAIN-3"], ["feature_a", "feature_b"]),
            (frame, ["TRAIN-2", "TRAIN-1", "TRAIN-3"], ["feature_a", "feature_b"]),
            (frame, frame.index.tolist(), []),
            (frame, frame.index.tolist(), ["feature_a", "feature_a"]),
        ]
        for data, ids, names in cases:
            with self.subTest(ids=ids, names=names):
                with self.assertRaises((TypeError, ValueError)):
                    preprocessing.fit_preprocessor(data, ids, names)
        all_missing = pd.DataFrame({"feature_a": [np.nan, np.nan]}, index=["A", "B"])
        with self.assertRaises(ValueError):
            preprocessing.fit_preprocessor(all_missing, ["A", "B"], ["feature_a"])

    def test_non_numeric_and_infinite_inputs_are_rejected(self) -> None:
        for value in ("bad", np.inf, -np.inf):
            frame = pd.DataFrame({"feature_a": [1.0, value]}, index=["A", "B"])
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    preprocessing.fit_preprocessor(frame, ["A", "B"], ["feature_a"])
        fitted = self.fit()
        invalid = pd.DataFrame({"feature_a": [np.inf], "feature_b": [1.0]}, index=["H"])
        with self.assertRaises(ValueError):
            preprocessing.transform_with_preprocessor(fitted, invalid, ["H"], fitted.feature_names)

    def test_transform_id_validation_and_fitted_type_are_enforced(self) -> None:
        fitted = self.fit()
        heldout = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]}, index=["A", "B"])
        for ids in (["A"], ["A", "A"], ["A", " "], ["B", "A"]):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    preprocessing.transform_with_preprocessor(fitted, heldout, ids, fitted.feature_names)
        with self.assertRaises(TypeError):
            preprocessing.transform_with_preprocessor(Pipeline([]), heldout, ["A", "B"], fitted.feature_names)

    def test_metadata_and_deterministic_independent_fits(self) -> None:
        first, second = self.fit(), self.fit()
        self.assertEqual(first.fit_sample_ids, ("TRAIN-1", "TRAIN-2", "TRAIN-3"))
        self.assertEqual(first.fit_sample_ids_sha256, payload_sha256(list(first.fit_sample_ids)))
        self.assertEqual(first.metadata["fit_sample_ids_sha256"], first.fit_sample_ids_sha256)
        self.assertEqual(first.metadata["feature_names"], ["feature_a", "feature_b"])
        transformed_first = preprocessing.transform_with_preprocessor(first, self.training_frame(), first.fit_sample_ids, first.feature_names)
        transformed_second = preprocessing.transform_with_preprocessor(second, self.training_frame(), second.fit_sample_ids, second.feature_names)
        np.testing.assert_array_equal(transformed_first.matrix, transformed_second.matrix)
        other = pd.DataFrame({"feature_a": [100.0, 200.0], "feature_b": [1.0, 3.0]}, index=["OTHER-1", "OTHER-2"])
        other_fitted = preprocessing.fit_preprocessor(other, other.index.tolist(), other.columns.tolist())
        self.assertNotEqual(first.metadata["scaler_means"], other_fitted.metadata["scaler_means"])

    def test_public_fit_accepts_no_labels_and_heldout_ids_are_not_fit_evidence(self) -> None:
        self.assertEqual(list(inspect.signature(preprocessing.fit_preprocessor).parameters), ["training_df", "training_sample_ids", "feature_names"])
        fitted = self.fit()
        heldout = pd.DataFrame({"feature_a": [4.0], "feature_b": [40.0]}, index=["HELDOUT"])
        preprocessing.transform_with_preprocessor(fitted, heldout, ["HELDOUT"], fitted.feature_names)
        self.assertNotIn("HELDOUT", fitted.fit_sample_ids)
        self.assertNotIn("label", " ".join(fitted.metadata).lower())

    def test_ae_fit_stop_refit_compatibility(self) -> None:
        complete = pd.DataFrame(
            {"feature_a": [1.0, 2.0, 3.0, 4.0], "feature_b": [10.0, 20.0, 30.0, 40.0]},
            index=["INNER-1", "INNER-2", "INNER-3", "INNER-4"],
        )
        temporary = preprocessing.fit_preprocessor(complete.iloc[:2], ["INNER-1", "INNER-2"], complete.columns)
        stop = preprocessing.transform_with_preprocessor(temporary, complete.iloc[2:], ["INNER-3", "INNER-4"], complete.columns)
        self.assertEqual(stop.sample_ids, ("INNER-3", "INNER-4"))
        del temporary
        full = preprocessing.fit_preprocessor(complete, complete.index.tolist(), complete.columns)
        self.assertEqual(full.fit_sample_ids, tuple(complete.index))
        self.assertNotEqual(full.metadata["scaler_means"], {"feature_a": 1.5, "feature_b": 15.0})


if __name__ == "__main__":
    unittest.main()
