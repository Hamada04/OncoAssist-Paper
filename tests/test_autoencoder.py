import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

from oncoassist_research import autoencoder


class _History:
    def __init__(self, loss, validation_loss):
        self.history = {"loss": loss, "val_loss": validation_loss}


class AutoencoderTests(unittest.TestCase):
    def raw_data(self):
        ids = [f"ID-{index}" for index in range(10)]
        labels = np.asarray([0, 1] * 5, dtype=int)
        values = np.arange(40, dtype=float).reshape(10, 4) + 1.0
        return pd.DataFrame(values, index=ids, columns=["f1", "f2", "f3", "f4"]), ids, labels

    def architecture(self, modality="mGE"):
        return autoencoder.AutoencoderArchitecture(modality, 4, 128, 2)

    def config(self, epochs=3):
        return autoencoder.AutoencoderTrainingConfig(epochs, 4, 1)

    def test_architecture_contract_and_independent_models(self):
        architecture = self.architecture()
        auto, encoder = autoencoder.build_autoencoder(architecture)
        self.assertIs(encoder.output, auto.get_layer("mGE_bottleneck").output)
        self.assertEqual(encoder.output_shape[-1], 2)
        self.assertEqual(auto.output_shape[-1], 4)
        self.assertEqual(auto.get_layer("mGE_encoder_hidden").activation.__name__, "relu")
        self.assertEqual(auto.get_layer("mGE_bottleneck").activation.__name__, "linear")
        self.assertEqual(auto.get_layer("mGE_reconstruction").activation.__name__, "linear")
        self.assertEqual(auto.loss, "mse")
        other_auto, other_encoder = autoencoder.build_autoencoder(self.architecture("mDM"))
        self.assertIsNot(auto, other_auto)
        self.assertIsNot(encoder, other_encoder)

    def test_architecture_and_ratio_validation(self):
        for values in [("", 4, 128, 2), ("mGE", 2, 128, 2), ("mGE", 4, 2, 2), ("mGE", 4, 128, 1), ("mGE", True, 128, 2)]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    autoencoder.AutoencoderArchitecture(*values)
        self.assertEqual([autoencoder.latent_dim_from_ratio(23, ratio) for ratio in (.25, .5, .75)], [6, 12, 18])
        self.assertEqual([autoencoder.latent_dim_from_ratio(42, ratio) for ratio in (.25, .5, .75)], [11, 21, 32])
        self.assertEqual([autoencoder.latent_dim_from_ratio(14, ratio) for ratio in (.25, .5, .75)], [4, 7, 11])
        with self.assertRaises(ValueError):
            autoencoder.latent_dim_from_ratio(4, .25)

    def test_split_contract_and_determinism(self):
        _, ids, labels = self.raw_data()
        first = autoencoder.build_autoencoder_split(ids, labels, .2, 77)
        second = autoencoder.build_autoencoder_split(ids, labels, .2, 77)
        self.assertEqual(first, second)
        self.assertFalse(set(first.fit_sample_ids).intersection(first.stop_sample_ids))
        self.assertEqual(set(first.fit_sample_ids).union(first.stop_sample_ids), set(ids))
        self.assertEqual(first.evidence["labels_used_only_for"], "stratification")
        with self.assertRaises(ValueError):
            autoencoder.build_autoencoder_split(ids, labels, 1.0, 77)

    def test_temporary_selection_uses_ae_fit_only_and_minimum_validation_epoch(self):
        frame, ids, labels = self.raw_data()
        split = autoencoder.build_autoencoder_split(ids, labels, .2, 31)
        frame.loc[list(split.stop_sample_ids), "f1"] = 1_000_000.0
        captured = {}
        def fake_fit(model, fit_matrix, stop_matrix, config):
            captured["fit_shape"] = fit_matrix.shape
            captured["stop_shape"] = stop_matrix.shape
            return _History([3.0, 2.0, 2.0], [5.0, 1.0, 3.0])
        with patch("oncoassist_research.autoencoder._fit_temporary_model", side_effect=fake_fit):
            result = autoencoder.select_epoch_for_modality(frame, ids, frame.columns, split, self.architecture(), self.config(), 101)
        self.assertEqual(result.selected_epoch_count, 2)
        self.assertNotEqual(result.selected_epoch_count, result.evidence["epochs_ran"])
        self.assertEqual(result.evidence["temporary_preprocessing_fit_sample_ids_sha256"], split.evidence["fit_sample_ids_sha256"])
        self.assertNotEqual(result.evidence["temporary_preprocessing_metadata"]["scaler_means"]["f1"], 1_000_000.0)
        self.assertEqual(captured["fit_shape"][0], len(split.fit_sample_ids))
        self.assertEqual(captured["stop_shape"][0], len(split.stop_sample_ids))
        self.assertTrue(result.evidence["temporary_model_used_only_for_epoch_selection"])
        self.assertNotIn("labels", result.evidence)

    def test_fresh_refit_uses_new_complete_preprocessing_and_no_validation_arguments(self):
        frame, ids, labels = self.raw_data()
        split = autoencoder.build_autoencoder_split(ids, labels, .2, 18)
        heldout = pd.DataFrame(np.full((2, 4), 99.0), index=["HELDOUT-1", "HELDOUT-2"], columns=frame.columns)
        built = []
        original_builder = autoencoder.build_autoencoder
        def recording_builder(architecture):
            models = original_builder(architecture)
            built.append(models)
            return models
        with patch("oncoassist_research.autoencoder.build_autoencoder", side_effect=recording_builder), patch("oncoassist_research.autoencoder._fit_temporary_model", return_value=_History([2.0, 1.0], [4.0, 1.0])):
            selection = autoencoder.select_epoch_for_modality(frame, ids, frame.columns, split, self.architecture(), self.config(), 101)
            captured = {}
            def fake_final(model, training_matrix, selected_epochs, config):
                captured["shape"] = training_matrix.shape
                captured["epochs"] = selected_epochs
                return None
            with patch("oncoassist_research.autoencoder._fit_final_model", side_effect=fake_final):
                final = autoencoder.refit_selected_epoch_modality(frame, ids, heldout, heldout.index.tolist(), frame.columns, self.architecture(), self.config(), selection.selected_epoch_count, 202)
        self.assertEqual(final.preprocessor.fit_sample_ids, tuple(ids))
        self.assertEqual(captured["epochs"], selection.selected_epoch_count)
        self.assertEqual(captured["shape"][0], len(ids))
        self.assertIsNot(built[0][0], final.autoencoder)
        self.assertIsNot(built[0][1], final.encoder)
        self.assertFalse(final.evidence["validation_data_used"])
        self.assertFalse(final.evidence["early_stopping_used"])
        self.assertFalse(final.evidence["heldout_supplied_to_fit"])
        self.assertEqual(final.training_latents.shape, (10, 2))
        self.assertEqual(final.heldout_latents.shape, (2, 2))

    def test_tiny_real_tensorflow_selection_and_refit(self):
        frame, ids, labels = self.raw_data()
        split = autoencoder.build_autoencoder_split(ids, labels, .2, 9)
        selection = autoencoder.select_epoch_for_modality(frame, ids, frame.columns, split, self.architecture(), self.config(2), 303)
        heldout = pd.DataFrame(np.full((2, 4), 20.0), index=["H-1", "H-2"], columns=frame.columns)
        final = autoencoder.refit_selected_epoch_modality(frame, ids, heldout, heldout.index.tolist(), frame.columns, self.architecture(), self.config(2), selection.selected_epoch_count, 404)
        self.assertTrue(np.isfinite(final.training_latents).all())
        self.assertTrue(np.isfinite(final.heldout_latents).all())
        self.assertEqual(final.training_latents.dtype, np.float32)
        self.assertEqual(final.evidence["epochs_trained"], selection.selected_epoch_count)

    def test_seed_evidence_is_conservative(self):
        evidence = autoencoder.configure_deterministic_seed(123)
        self.assertTrue(evidence["deterministic_settings_requested"])
        self.assertIn("deterministic_operations_enabled", evidence)
        self.assertNotIn("bit_identical", " ".join(evidence).lower())


if __name__ == "__main__":
    unittest.main()
