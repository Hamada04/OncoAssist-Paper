import unittest
from unittest.mock import patch
import numpy as np
from subprocess import CompletedProcess

from oncoassist_research.artifacts import payload_sha256
from oncoassist_research import ctgan


class CTGANTests(unittest.TestCase):
    def inputs(self):
        features = np.arange(30, dtype=np.float32).reshape(10,3)
        labels = np.array([0]*6 + [1]*4, dtype=int)
        ids = [f"R{index}" for index in range(10)]
        names = ("mGE_z000", "mDM_z000", "mCNA_z000")
        return features, labels, ids, names, payload_sha256(list(names))

    def test_extraction_and_batch_contracts(self):
        features, labels, ids, names, name_hash = self.inputs()
        minority = ctgan.extract_minority_training_input(features, labels, ids, names, name_hash)
        self.assertEqual(minority.minority_label, 1); self.assertEqual(minority.majority_label, 0); self.assertEqual(minority.needed_synthetic_count, 2); self.assertEqual(minority.minority_features.shape, (4,3))
        self.assertEqual(ctgan.derive_ctgan_batch_size(4, 2), 4)
        with self.assertRaises(ValueError): ctgan.derive_ctgan_batch_size(4, 3)
        with self.assertRaises(ValueError): ctgan.extract_minority_training_input(features, np.zeros(10), ids, names, name_hash)
        with self.assertRaises(ValueError): ctgan.extract_minority_training_input(features, np.array([0,1]*5), ids, names, name_hash)
        with self.assertRaises(ValueError): ctgan.extract_minority_training_input(features, labels, ids, ("CLASS","b","c"), name_hash)

    def test_augmentation_real_first_balanced_and_no_fallback(self):
        features, labels, ids, names, name_hash = self.inputs(); synthetic = np.full((2,3), 9, dtype=np.float32)
        response = {"execution_evidence": {"ctgan_execution_backend": ctgan.CTGAN_EXECUTION_BACKEND, "tensorflow_present_in_worker": False, "ctgan_gpu_enabled": False}, "seed_evidence": {"exact_regeneration_guaranteed": False}}
        with patch("oncoassist_research.ctgan.fit_and_sample_minority_ctgan", return_value=(synthetic, response)):
            augmented = ctgan.augment_with_minority_ctgan(features, labels, ids, names, name_hash, ctgan.CTGANConfig(1,2,False), 7, "TEST")
        np.testing.assert_array_equal(augmented.features[:10], features); np.testing.assert_array_equal(augmented.labels[:10], labels)
        self.assertEqual(augmented.labels.tolist().count(0), 6); self.assertEqual(augmented.labels.tolist().count(1), 6)
        self.assertTrue(augmented.is_synthetic[-2:].all()); self.assertFalse(augmented.is_synthetic[:10].any()); self.assertTrue(augmented.evidence["real_rows_first"]); self.assertFalse(augmented.evidence["fallback_exists"])

    def test_tiny_isolated_worker_smoke(self):
        features, labels, ids, names, name_hash = self.inputs()
        result = ctgan.augment_with_minority_ctgan(features, labels, ids, names, name_hash, ctgan.CTGANConfig(1,2,False), 19, "SMOKE")
        self.assertEqual(result.features.shape, (12,3)); self.assertTrue(np.isfinite(result.features).all())
        self.assertEqual(result.evidence["worker"]["execution_evidence"]["tensorflow_present_in_worker"], False)
        self.assertEqual(result.evidence["worker"]["execution_evidence"]["ctgan_gpu_enabled"], False)

    def test_no_fit_worker_probe_checks_only_compatibility(self):
        response = {
            "schema_version": ctgan.WORKER_SCHEMA_VERSION,
            "preflight_only": True,
            "ctgan_execution_backend": ctgan.CTGAN_EXECUTION_BACKEND,
            "cuda_visible_devices": "",
            "ctgan_gpu_enabled": False,
            "constructor_parameters": ["metadata", "epochs", "batch_size", "pac", "verbose"],
            "versions": {"sdv": "x", "ctgan": "x", "torch": "x", "numpy": "x", "pandas": "x"},
        }
        with patch("oncoassist_research.ctgan.subprocess.run", return_value=CompletedProcess([], 0, "", "")), patch(
            "oncoassist_research.ctgan.read_json_object", return_value=response
        ), patch("oncoassist_research.ctgan.fit_and_sample_minority_ctgan", side_effect=AssertionError):
            result = ctgan.probe_isolated_ctgan_worker(ctgan.CTGANConfig(300, 10, False))
        self.assertTrue(result["preflight_only"])
