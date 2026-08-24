import unittest
import numpy as np

from oncoassist_research.classifiers import LogisticRegressionConfig, fit_logistic_classifier, score_logistic_classifier
from oncoassist_research.ctgan import AugmentedTrainingSet


class ClassifierTests(unittest.TestCase):
    def training(self):
        return AugmentedTrainingSet(np.array([[0.,0.],[1.,1.],[9.,9.],[10.,10.]], dtype=np.float32), np.array([0,0,1,1]), ("R1","R2","S1","S2"), np.array([False,False,True,True]), ("mGE_z000","mDM_z000"), {})

    def test_primary_configuration_scaling_and_scores(self):
        training = self.training(); original = training.features.copy()
        fitted = fit_logistic_classifier(training, LogisticRegressionConfig(.1), 77)
        self.assertEqual(fitted.model.get_params()["solver"], "liblinear"); self.assertEqual(fitted.model.get_params()["penalty"], "l2"); self.assertEqual(fitted.model.get_params()["max_iter"], 1000); self.assertIsNone(fitted.model.get_params()["class_weight"]); self.assertEqual(fitted.model.get_params()["random_state"], 77)
        self.assertEqual(fitted.evidence["scaler_fit_row_count"], 4); self.assertTrue(fitted.evidence["scaler_fit_includes_synthetic_rows"])
        scores = score_logistic_classifier(fitted, np.array([[.5,.5],[9.5,9.5]], dtype=np.float32), ("V1","V2"), training.feature_names)
        self.assertTrue(np.isfinite(scores.decision_scores).all()); self.assertTrue(np.all((scores.probabilities >= 0) & (scores.probabilities <= 1))); self.assertFalse(scores.decision_scores.flags.writeable); np.testing.assert_array_equal(training.features, original)

    def test_classifier_rejects_invalid_contracts(self):
        with self.assertRaises(ValueError): LogisticRegressionConfig(True)
        with self.assertRaises(ValueError): LogisticRegressionConfig(0)
        training = self.training(); fitted = fit_logistic_classifier(training, LogisticRegressionConfig(1), 1)
        with self.assertRaises(ValueError): score_logistic_classifier(fitted, np.array([[1.,2.]], dtype=np.float32), ("V",), ("mDM_z000","mGE_z000"))
