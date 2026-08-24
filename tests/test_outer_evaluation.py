import unittest
from dataclasses import replace

from tests import test_search
from tests.test_final_refit import FinalRefitTests
from oncoassist_research import final_refit, folds
from oncoassist_research.artifacts import payload_sha256
from oncoassist_research.final_refit import score_outer_test
from oncoassist_research.metrics import compute_binary_metrics, compute_ranking_metrics
from oncoassist_research.outer_evaluation import evaluate_outer_test_predictions
from oncoassist_research.protocol import create_primary_v1_run_provenance, ordered_patient_ids_sha256
from oncoassist_research.calibration import cross_fit_sigmoid_calibration
from oncoassist_research.thresholds import select_operational_threshold


class OuterEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = FinalRefitTests("runTest")
        fixture.setUp()
        bundle, _ = fixture._build_bundle()
        frozen = score_outer_test(bundle, fixture.test_modalities, fixture.test_ids, search_result=fixture.official, run_provenance=fixture.provenance, aligned_data=fixture.data)
        labels = [fixture.data["y_binary"][fixture.data["sample_ids"].index(sample_id)] for sample_id in fixture.test_ids]
        cls.authority = fixture, bundle, frozen, labels

    def authoritative(self):
        return self.authority

    def evaluate(self, fixture, bundle, frozen, labels):
        return evaluate_outer_test_predictions(frozen, labels, fixture.test_ids, run_provenance=fixture.provenance, aligned_data=fixture.data, search_result=fixture.official, frozen_final_model=bundle)

    def reidentified_predictions(self, frozen, sample_ids):
        predictions = tuple(replace(prediction, sample_id=sample_id) for prediction, sample_id in zip(frozen.predictions, sample_ids))
        return replace(
            frozen,
            predictions=predictions,
            patient_ids_hash=ordered_patient_ids_sha256(sample_ids),
            prediction_hash=payload_sha256([final_refit._prediction_payload(prediction) for prediction in predictions]),
        )

    def test_metrics_use_only_validated_frozen_sources(self):
        fixture, bundle, frozen, labels = self.authoritative()
        result = self.evaluate(fixture, bundle, frozen, labels)
        self.assertTrue(result.evidence["outer_test_labels_used_only_for_evaluation"])
        self.assertFalse(result.evidence["post_test_adaptation_performed"])
        self.assertEqual(result.evidence["run_provenance_identity_sha256"], fixture.provenance.identity_sha256)
        self.assertEqual(result.evidence["prediction_hash"], frozen.prediction_hash)
        raw_scores = [prediction.raw_decision_score for prediction in frozen.predictions]
        predicted_labels = [prediction.predicted_label for prediction in frozen.predictions]
        self.assertEqual(result.primary_ranking_metrics, compute_ranking_metrics(labels, raw_scores))
        self.assertEqual(result.operational_metrics.balanced_accuracy, compute_binary_metrics(labels, predicted_labels, .5, "decision_score").balanced_accuracy)

    def test_rejects_noncanonical_label_vectors_before_metrics(self):
        fixture, bundle, frozen, labels = self.authoritative()
        swapped = list(labels)
        zero, one = swapped.index(0), swapped.index(1)
        swapped[zero], swapped[one] = swapped[one], swapped[zero]
        fingerprint = folds.build_outer_data_fingerprint(fixture.data)
        manifest = folds.build_outer_fold_manifest(fixture.data["sample_ids"], fixture.data["y_binary"], fingerprint, fixture.context.fold_protocol)
        other_ids = next(tuple(record["test_sample_ids"]) for record in manifest["folds"] if tuple(record["test_sample_ids"]) != fixture.test_ids and len(record["test_sample_ids"]) == len(fixture.test_ids))
        label_by_id = dict(zip(fixture.data["sample_ids"], fixture.data["y_binary"].tolist()))
        other_fold_labels = [label_by_id[sample_id] for sample_id in other_ids]
        if other_fold_labels == labels:
            other_fold_labels = list(reversed(other_fold_labels))
        cases = {
            "flipped": [1 - labels[0], *labels[1:]],
            "patient_label_mapping": swapped,
            "other_fold": other_fold_labels,
            "missing": labels[:-1],
            "extra": [*labels, 0],
            "nonbinary": [2, *labels[1:]],
        }
        from unittest.mock import patch
        for name, submitted in cases.items():
            with self.subTest(labels=name), patch("oncoassist_research.outer_evaluation.compute_ranking_metrics", side_effect=AssertionError):
                with self.assertRaises(ValueError):
                    self.evaluate(fixture, bundle, frozen, submitted)

    def test_rejects_rehashed_alternate_outer_prediction_cohorts_before_metrics(self):
        fixture, bundle, frozen, labels = self.authoritative()
        fingerprint = folds.build_outer_data_fingerprint(fixture.data)
        manifest = folds.build_outer_fold_manifest(fixture.data["sample_ids"], fixture.data["y_binary"], fingerprint, fixture.context.fold_protocol)
        other_ids = next(tuple(record["test_sample_ids"]) for record in manifest["folds"] if tuple(record["test_sample_ids"]) != fixture.test_ids and len(record["test_sample_ids"]) == len(fixture.test_ids))
        cases = {
            "missing": self.reidentified_predictions(frozen, fixture.test_ids[:-1]),
            "foreign": self.reidentified_predictions(frozen, ("FOREIGN", *fixture.test_ids[1:])),
            "training": self.reidentified_predictions(frozen, (fixture.context.outer_training_ids[0], *fixture.test_ids[1:])),
            "other_fold": self.reidentified_predictions(frozen, other_ids),
        }
        from unittest.mock import patch
        for name, frozen_case in cases.items():
            with self.subTest(cohort=name), patch("oncoassist_research.outer_evaluation.compute_ranking_metrics", side_effect=AssertionError):
                with self.assertRaises(ValueError):
                    self.evaluate(fixture, bundle, frozen_case, labels)

    def test_rejects_run_fold_bundle_and_prediction_substitution_before_metrics(self):
        fixture, bundle, frozen, labels = self.authoritative()
        run_b = create_primary_v1_run_provenance(run_id="run-b", root_seed=fixture.provenance.root_seed + 1, protocol=fixture.context.protocol, aligned_data=fixture.data)
        cases = (
            (frozen, labels, fixture.test_ids, run_b, fixture.official, bundle),
            (replace(frozen, evidence={**frozen.evidence, "run_provenance_identity_sha256": "0" * 64}), labels, fixture.test_ids, fixture.provenance, fixture.official, bundle),
            (replace(frozen, evidence={**frozen.evidence, "aligned_data_content_identity_sha256": "0" * 64}), labels, fixture.test_ids, fixture.provenance, fixture.official, bundle),
            (replace(frozen, evidence={**frozen.evidence, "protocol_sha256": "0" * 64}), labels, fixture.test_ids, fixture.provenance, fixture.official, bundle),
            (replace(frozen, evidence={**frozen.evidence, "search_selection_identity_sha256": "0" * 64}), labels, fixture.test_ids, fixture.provenance, fixture.official, bundle),
            (replace(frozen, evidence={**frozen.evidence, "candidate_identity_sha256": "0" * 64}), labels, fixture.test_ids, fixture.provenance, fixture.official, bundle),
            (replace(frozen, evidence={**frozen.evidence, "frozen_model_state_sha256": "0" * 64}), labels, fixture.test_ids, fixture.provenance, fixture.official, bundle),
            (replace(frozen, predictions=(replace(frozen.predictions[0], raw_decision_score=frozen.predictions[0].raw_decision_score + .1), *frozen.predictions[1:])), labels, fixture.test_ids, fixture.provenance, fixture.official, bundle),
            (frozen, labels, tuple(reversed(fixture.test_ids)), fixture.provenance, fixture.official, bundle),
            (frozen, labels, fixture.test_ids, fixture.provenance, replace(fixture.official, selected_search=replace(fixture.official.selected_search, search_selection_identity_sha256="1" * 64)), bundle),
            (frozen, labels, fixture.test_ids, fixture.provenance, fixture.official, replace(bundle, final_model_identity_sha256="2" * 64)),
        )
        for frozen_case, labels_case, ids_case, provenance_case, search_case, bundle_case in cases:
            with self.subTest(case=provenance_case.run_id):
                with self.assertRaises(ValueError):
                    evaluate_outer_test_predictions(frozen_case, labels_case, ids_case, run_provenance=provenance_case, aligned_data=fixture.data, search_result=search_case, frozen_final_model=bundle_case)

    def test_run_a_chain_is_rejected_under_run_b_before_downstream_work(self):
        fixture, bundle, frozen, labels = self.authoritative()
        run_b = create_primary_v1_run_provenance(run_id="run-b", root_seed=fixture.provenance.root_seed + 1, protocol=fixture.context.protocol, aligned_data=fixture.data)
        with self.assertRaises(ValueError): cross_fit_sigmoid_calibration(fixture.official, run_provenance=run_b, aligned_data=fixture.data)
        with self.assertRaises(ValueError): select_operational_threshold(fixture.calibration, search_result=fixture.official, run_provenance=run_b, aligned_data=fixture.data)
        with self.assertRaises(ValueError): evaluate_outer_test_predictions(frozen, labels, fixture.test_ids, run_provenance=run_b, aligned_data=fixture.data, search_result=fixture.official, frozen_final_model=bundle)

    def test_evaluation_does_not_fit_or_score(self):
        fixture, bundle, frozen, labels = self.authoritative()
        from unittest.mock import patch
        with patch("oncoassist_research.final_refit.transform_with_preprocessor", side_effect=AssertionError), patch("oncoassist_research.final_refit.score_logistic_classifier", side_effect=AssertionError), patch("oncoassist_research.final_refit.fit_logistic_classifier", side_effect=AssertionError), patch("oncoassist_research.final_refit.score_outer_test", side_effect=AssertionError), patch("oncoassist_research.calibration._fit", side_effect=AssertionError), patch("oncoassist_research.thresholds.select_operational_threshold", side_effect=AssertionError):
            self.evaluate(fixture, bundle, frozen, labels)


if __name__ == "__main__": unittest.main()
