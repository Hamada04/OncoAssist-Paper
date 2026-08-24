import unittest
import numpy as np

from oncoassist_research.metrics import compute_binary_metrics, compute_ranking_metrics


class MetricsTests(unittest.TestCase):
    def test_ranking_and_threshold_metrics(self):
        labels = [0, 0, 1, 1]; scores = [-2.0, -1.0, 1.0, 2.0]
        ranking = compute_ranking_metrics(labels, scores); diagnostic = compute_binary_metrics(labels, scores, 0.0, "decision_score")
        self.assertEqual((ranking.auprc, ranking.auroc), (1.0, 1.0))
        self.assertEqual((diagnostic.tn, diagnostic.fp, diagnostic.fn, diagnostic.tp), (2, 0, 0, 2))
        self.assertEqual(diagnostic.label, "diagnostic_uncalibrated_threshold_metrics")

    def test_invalid_metric_inputs_rejected(self):
        for labels, scores in (([0, 0], [1, 2]), ([0, 1], [np.nan, 1]), ([0, 2], [0, 1]), ([0], [0, 1])):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError): compute_ranking_metrics(labels, scores)
        with self.assertRaises(ValueError): compute_binary_metrics([0,1], [0,1], 0.5, "unknown")
