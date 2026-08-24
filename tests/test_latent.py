import unittest
import numpy as np

from oncoassist_research.autoencoder import FittedEncoder
from oncoassist_research.artifacts import payload_sha256
from oncoassist_research.latent import build_latent_feature_names, build_latent_slices, fuse_fitted_encoders


class LatentTests(unittest.TestCase):
    def encoder(self, modality, width, train, heldout):
        return FittedEncoder(None, object(), object(), train, heldout, {"architecture": {"latent_dim": width}, "complete_training_sample_ids_sha256": payload_sha256(["T1", "T2"]), "heldout_sample_ids_sha256": payload_sha256(["H1"])})

    def encoders(self):
        return {
            "mGE": self.encoder("mGE", 2, np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[5, 6]], dtype=np.float32)),
            "mDM": self.encoder("mDM", 3, np.ones((2, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)),
            "mCNA": self.encoder("mCNA", 1, np.full((2, 1), 7, dtype=np.float32), np.full((1, 1), 8, dtype=np.float32)),
        }

    def test_fusion_order_names_slices_and_recovery(self):
        sources = self.encoders(); original = sources["mGE"].training_latents.copy()
        result = fuse_fitted_encoders(sources, ["T1", "T2"], ["H1"])
        self.assertEqual(result.feature_names, ("mGE_z000", "mGE_z001", "mDM_z000", "mDM_z001", "mDM_z002", "mCNA_z000"))
        self.assertEqual(result.modality_slices, {"mGE": (0,2), "mDM": (2,5), "mCNA": (5,6)})
        self.assertEqual(result.training.dtype, np.float32); self.assertFalse(result.training.flags.writeable)
        np.testing.assert_array_equal(result.training[:, :2], original); np.testing.assert_array_equal(sources["mGE"].training_latents, original)
        self.assertEqual(build_latent_feature_names({"mGE":2,"mDM":3,"mCNA":1}), result.feature_names)
        self.assertEqual(build_latent_slices({"mGE":2,"mDM":3,"mCNA":1}), result.modality_slices)

    def test_fusion_rejects_invalid_modalities_and_matrices(self):
        sources = self.encoders()
        with self.assertRaises(ValueError): fuse_fitted_encoders({"mGE": sources["mGE"]}, ["T1","T2"], ["H1"])
        bad = self.encoders(); bad["mDM"].training_latents[0,0] = np.nan
        with self.assertRaises(ValueError): fuse_fitted_encoders(bad, ["T1","T2"], ["H1"])
        bad = self.encoders(); bad["mCNA"].training_latents = np.array([[object()], [object()]], dtype=object)
        with self.assertRaises(ValueError): fuse_fitted_encoders(bad, ["T1","T2"], ["H1"])
        bad = self.encoders(); bad["mGE"].evidence["architecture"]["latent_dim"] = 3
        with self.assertRaises(ValueError): fuse_fitted_encoders(bad, ["T1","T2"], ["H1"])
