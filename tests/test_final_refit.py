import copy
import inspect
import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tests import test_search
from oncoassist_research import autoencoder, classifiers, final_refit, latent
from oncoassist_research.artifacts import payload_sha256
from oncoassist_research.calibration import (
    cross_fit_sigmoid_calibration,
    fit_final_sigmoid_calibrator,
)
from oncoassist_research.classifiers import LogisticScores
from oncoassist_research.ctgan import AugmentedTrainingSet
from oncoassist_research.final_refit import (
    FinalRefitSeedBook,
    fit_final_primary_model,
    score_outer_test,
    validate_final_primary_model_bundle,
    validate_frozen_outer_predictions,
)
from oncoassist_research.protocol import ordered_patient_ids_sha256
from oncoassist_research.search import (
    _finalize_primary_search_execution,
    _run_primary_inner_search_with_builder,
)
from oncoassist_research.thresholds import select_operational_threshold


class _FakeAutoencoder:
    def predict(self, matrix, verbose=0):
        return np.asarray(matrix, dtype=np.float32)


class _FakeEncoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def get_weights(self):
        return [np.arange(self.latent_dim, dtype=np.float32).reshape(1, -1)]

    def predict(self, matrix, verbose=0):
        rows = len(matrix)
        return np.tile(np.arange(self.latent_dim, dtype=np.float32), (rows, 1))


class FinalRefitTests(unittest.TestCase):
    def setUp(self):
        self.official = test_search.result()
        self.context = self.official.context
        self.candidate = self.official.selected_search.selected_candidate
        _, data, _, self.provenance = test_search.context_and_binding()
        self.data = data
        self.calibration = cross_fit_sigmoid_calibration(self.official, run_provenance=self.provenance, aligned_data=data)
        self.final_calibrator = fit_final_sigmoid_calibrator(
            self.official.selected_search.selected_oof_predictions,
            selected_search=self.official.selected_search,
            context=self.context,
        )
        self.threshold = select_operational_threshold(
            self.calibration,
            search_result=self.official, run_provenance=self.provenance, aligned_data=data,
        )
        self.training_modalities = {
            "mGE": data["X_rna"].loc[list(self.context.outer_training_ids)].copy(),
            "mDM": data["X_dna"].loc[list(self.context.outer_training_ids)].copy(),
            "mCNA": data["X_cna"].loc[list(self.context.outer_training_ids)].copy(),
        }
        self.contracts = {
            modality: tuple(frame.columns)
            for modality, frame in self.training_modalities.items()
        }
        self.test_ids = self.context.outer_testing_ids
        self.test_modalities = {
            "mGE": data["X_rna"].loc[list(self.test_ids)].copy(),
            "mDM": data["X_dna"].loc[list(self.test_ids)].copy(),
            "mCNA": data["X_cna"].loc[list(self.test_ids)].copy(),
        }
        self.seed_book = self.context.seed_manifest.final_refit_seed_book()

    def _fit_kwargs(self, **overrides):
        values = {
            "search_result": self.official,
            "cross_fitted_calibration": self.calibration,
            "protocol": self.context.protocol,
            "synthetic_namespace": "final-refit-test",
            "run_provenance": self.provenance,
            "aligned_data": self.data,
        }
        values.update(overrides)
        return values

    def _fit_args(self):
        return (
            self.training_modalities,
            self.context.outer_training_ids,
            self.context.outer_training_labels,
            self.test_modalities,
            self.test_ids,
            self.contracts,
            self.context.protocol.make_autoencoder_training_config(),
            self.context.protocol.make_ctgan_config(),
            self.seed_book,
            self.final_calibrator,
            self.threshold,
        )

    def _assert_no_final_fit_calls(self, call):
        preprocessing = Mock()
        split = Mock()
        select = Mock()
        refit = Mock()
        fusion = Mock()
        augment = Mock()
        classifier = Mock()
        with patch("oncoassist_research.autoencoder.fit_preprocessor", preprocessing), patch(
            "oncoassist_research.final_refit.build_autoencoder_split", split
        ), patch(
            "oncoassist_research.final_refit.select_epoch_for_modality", select
        ), patch("oncoassist_research.final_refit.refit_selected_epoch_modality", refit), patch(
            "oncoassist_research.final_refit.fuse_fitted_encoders", fusion
        ), patch(
            "oncoassist_research.final_refit.augment_with_minority_ctgan", augment
        ), patch("oncoassist_research.final_refit.fit_logistic_classifier", classifier):
            with self.assertRaises((TypeError, ValueError)):
                call()
        self.assertEqual(preprocessing.call_count, 0)
        self.assertEqual(split.call_count, 0)
        self.assertEqual(select.call_count, 0)
        self.assertEqual(refit.call_count, 0)
        self.assertEqual(fusion.call_count, 0)
        self.assertEqual(augment.call_count, 0)
        self.assertEqual(classifier.call_count, 0)

    def _assert_scoring_rejected_before_transform(self, bundle):
        with patch(
            "oncoassist_research.final_refit.transform_with_preprocessor",
            side_effect=AssertionError,
        ), patch(
            "oncoassist_research.final_refit.score_logistic_classifier",
            side_effect=AssertionError,
        ):
            with self.assertRaises(ValueError):
                score_outer_test(bundle, self.test_modalities, self.test_ids, search_result=self.official, run_provenance=self.provenance, aligned_data=self.data)

    def _build_bundle(self):
        calls = {
            "split": [],
            "temporary_preprocessing_ids": [],
            "temporary_ae_rows": [],
            "final_preprocessing_ids": [],
            "final_ae": [],
            "selection": [],
            "refit": [],
            "fusion_order": [],
            "ctgan": [],
            "classifier": [],
        }
        split_original = final_refit.build_autoencoder_split
        selection_original = final_refit.select_epoch_for_modality
        refit_original = final_refit.refit_selected_epoch_modality
        fusion_original = latent.fuse_fitted_encoders
        classifier_original = classifiers.fit_logistic_classifier
        preprocessor_original = autoencoder.fit_preprocessor

        def split_spy(ids, labels, fraction, seed):
            result = split_original(ids, labels, fraction, seed)
            calls["split"].append(result)
            return result

        def preprocessing_spy(frame, ids, names):
            ids = tuple(ids)
            if len(ids) < len(self.context.outer_training_ids):
                calls["temporary_preprocessing_ids"].append(ids)
            else:
                calls["final_preprocessing_ids"].append(ids)
            return preprocessor_original(frame, ids, names)

        def temporary_fit(auto, fit_matrix, stop_matrix, config):
            calls["temporary_ae_rows"].append((len(fit_matrix), len(stop_matrix)))
            return type("History", (), {"history": {"loss": [1.0, 0.5], "val_loss": [1.0, 0.25]}})()

        def final_fit(auto, training_matrix, selected_epoch_count, config):
            calls["final_ae"].append((len(training_matrix), selected_epoch_count, config))
            return object()

        def selection_spy(*args, **kwargs):
            calls["selection"].append((args[4].modality, tuple(args[1]), tuple(args[3].fit_sample_ids)))
            return selection_original(*args, **kwargs)

        def refit_spy(*args, **kwargs):
            calls["refit"].append((args[5].modality, tuple(args[1]), tuple(args[3])))
            return refit_original(*args, **kwargs)

        def fusion_spy(encoders, training_ids, heldout_ids):
            calls["fusion_order"].append(tuple(encoders))
            return fusion_original(encoders, training_ids, heldout_ids)

        def ctgan_spy(features, labels, ids, names, names_hash, config, seed, namespace):
            calls["ctgan"].append((np.array(features, copy=True), tuple(ids), tuple(names), config))
            synthetic = np.asarray(features[:1], dtype=np.float32)
            return AugmentedTrainingSet(
                np.vstack([features, synthetic]).astype(np.float32),
                np.concatenate([np.asarray(labels, dtype=int), np.asarray([0], dtype=int)]),
                tuple(ids) + ("SYNTHETIC:final-refit-test:MINORITY:0:000000",),
                np.asarray([False] * len(ids) + [True]),
                tuple(names),
                {
                    "strategy": "minority_only_ctgan",
                    "training_ids_sha256": ordered_patient_ids_sha256(ids),
                    "heldout_supplied_to_ctgan": False,
                },
            )

        def classifier_spy(training, config, seed):
            calls["classifier"].append((config.C, tuple(training.record_ids), np.array(training.features, copy=True)))
            return classifier_original(training, config, seed)

        with ExitStack() as stack:
            stack.enter_context(patch("oncoassist_research.final_refit.build_autoencoder_split", split_spy))
            stack.enter_context(patch("oncoassist_research.final_refit.select_epoch_for_modality", selection_spy))
            stack.enter_context(patch("oncoassist_research.final_refit.refit_selected_epoch_modality", refit_spy))
            stack.enter_context(patch("oncoassist_research.final_refit.fuse_fitted_encoders", fusion_spy))
            stack.enter_context(patch("oncoassist_research.final_refit.augment_with_minority_ctgan", ctgan_spy))
            stack.enter_context(patch("oncoassist_research.final_refit.fit_logistic_classifier", classifier_spy))
            stack.enter_context(patch("oncoassist_research.autoencoder.fit_preprocessor", preprocessing_spy))
            stack.enter_context(patch("oncoassist_research.autoencoder.configure_deterministic_seed", lambda seed: {"model_seed": seed, "deterministic_settings_requested": True, "deterministic_operations_enabled": True}))
            stack.enter_context(patch("oncoassist_research.autoencoder.build_autoencoder", lambda architecture: (_FakeAutoencoder(), _FakeEncoder(architecture.latent_dim))))
            stack.enter_context(patch("oncoassist_research.autoencoder._fit_temporary_model", temporary_fit))
            stack.enter_context(patch("oncoassist_research.autoencoder._fit_final_model", final_fit))
            bundle = fit_final_primary_model(*self._fit_args(), **self._fit_kwargs())
        return bundle, calls

    def _official_b_result(self):
        context, data, binding, provenance = test_search.context_and_binding()

        def scorer(config, features, ids, names):
            raw = np.asarray([-1.0 if index % 2 == 0 else 1.0 for index in range(len(ids))])
            if config.C != 1.0:
                raw = -raw
            return LogisticScores(tuple(ids), raw, 1.0 / (1.0 + np.exp(-raw)), {})

        execution = _run_primary_inner_search_with_builder(
            context,
            binding,
            lambda augmented, config, seed: config,
            scorer,
        )
        result = _finalize_primary_search_execution(context, execution)
        self.assertEqual(result.selected_search.selected_candidate.logistic_c, 1.0)
        return result

    def test_public_authority_signature_and_candidate_substitution_are_closed(self):
        parameters = inspect.signature(fit_final_primary_model).parameters
        self.assertIn("search_result", parameters)
        self.assertNotIn("selected_candidate", parameters)
        self.assertNotIn("selected_search", parameters)
        self.assertNotIn("context", parameters)
        with self.assertRaises(TypeError):
            inspect.signature(fit_final_primary_model).bind(*self._fit_args(), **self._fit_kwargs(selected_candidate=self.candidate))
        without_authority = self._fit_kwargs()
        without_authority.pop("search_result")
        with self.assertRaises(TypeError):
            fit_final_primary_model(*self._fit_args(), **without_authority)

    def test_valid_official_chain_preserves_selected_configuration_and_scientific_boundaries(self):
        bundle, calls = self._build_bundle()
        self.assertEqual(bundle.selected_candidate, self.candidate)
        self.assertEqual(bundle.candidate_identity_sha256, self.candidate.candidate_identity_sha256)
        self.assertEqual(bundle.fused_latent_dimensions, self.candidate.latent_dimensions)
        self.assertEqual(bundle.search_selection_identity_sha256, self.official.selected_search.search_selection_identity_sha256)
        self.assertEqual(bundle.final_calibrator.final_calibrator_identity_sha256, self.final_calibrator.final_calibrator_identity_sha256)
        self.assertEqual(len(calls["split"]), 1)
        self.assertEqual([item[0] for item in calls["selection"]], ["mGE", "mDM", "mCNA"])
        self.assertEqual([item[0] for item in calls["refit"]], ["mGE", "mDM", "mCNA"])
        self.assertEqual(calls["fusion_order"], [("mGE", "mDM", "mCNA")])
        self.assertEqual(len(calls["ctgan"]), 1)
        self.assertEqual(len(calls["classifier"]), 1)
        self.assertEqual(calls["classifier"][0][0], self.candidate.logistic_c)
        self.assertEqual(calls["ctgan"][0][1], self.context.outer_training_ids)
        self.assertTrue(all(set(ids).issubset(set(self.context.outer_training_ids)) for ids in calls["temporary_preprocessing_ids"]))
        self.assertEqual(calls["final_preprocessing_ids"], [self.context.outer_training_ids] * 3)
        self.assertTrue(all(set(training_ids) == set(self.context.outer_training_ids) and set(heldout_ids) == set(self.test_ids) for _, training_ids, heldout_ids in calls["refit"]))
        self.assertTrue(all(test_id not in ids for ids in calls["temporary_preprocessing_ids"] + calls["final_preprocessing_ids"] for test_id in self.test_ids))
        self.assertTrue(all(rows[0] < len(self.context.outer_training_ids) for rows in calls["temporary_ae_rows"]))
        self.assertEqual(calls["final_ae"], [(len(self.context.outer_training_ids), 2, self.context.protocol.make_autoencoder_training_config())] * 3)
        self.assertTrue(all(test_id not in record_ids for _, record_ids, _ in calls["classifier"] for test_id in self.test_ids))

    def test_invalid_official_result_is_rejected_before_every_fit(self):
        invalid = replace(
            self.official,
            selected_search=replace(
                self.official.selected_search,
                selected_oof_predictions=self.official.selected_search.selected_oof_predictions[:-1],
            ),
        )
        self._assert_no_final_fit_calls(
            lambda: fit_final_primary_model(*self._fit_args(), **self._fit_kwargs(search_result=invalid))
        )

    def test_outer_training_test_overlap_is_rejected_before_every_fit(self):
        overlap_ids = (self.context.outer_training_ids[0], *self.test_ids[1:])
        overlap_modalities = {
            modality: frame.set_axis(overlap_ids, axis="index", copy=True)
            for modality, frame in self.test_modalities.items()
        }
        args = list(self._fit_args())
        args[3] = overlap_modalities
        args[4] = overlap_ids
        self._assert_no_final_fit_calls(
            lambda: fit_final_primary_model(*args, **self._fit_kwargs())
        )

    def test_noncanonical_training_inputs_are_rejected_before_every_fit(self):
        def copied_args():
            args = list(self._fit_args())
            args[0] = {modality: frame.copy(deep=True) for modality, frame in self.training_modalities.items()}
            args[1] = tuple(args[1])
            args[2] = np.asarray(args[2], dtype=int).copy()
            args[5] = {modality: tuple(names) for modality, names in self.contracts.items()}
            return args

        def changed_value(modality):
            def mutate(args):
                args[0][modality].iat[0, 0] += 1.0
            return mutate

        def reordered_labels(args):
            zero, one = np.flatnonzero(args[2] == 0)[0], np.flatnonzero(args[2] == 1)[0]
            args[2][zero], args[2][one] = args[2][one], args[2][zero]

        def reordered_rows(args):
            frame = args[0]["mGE"]
            args[0]["mGE"] = frame.iloc[[1, 0, *range(2, len(frame))]].copy()

        def reordered_columns(args):
            columns = list(args[5]["mDM"])
            columns[0], columns[1] = columns[1], columns[0]
            args[0]["mDM"] = args[0]["mDM"].loc[:, columns].copy()
            args[5]["mDM"] = tuple(columns)

        def renamed_feature(args):
            old = args[5]["mCNA"][0]
            new = "forged-cna-feature"
            args[0]["mCNA"] = args[0]["mCNA"].rename(columns={old: new})
            args[5]["mCNA"] = (new, *args[5]["mCNA"][1:])

        def same_shape_substitution(args):
            frame = args[0]["mGE"]
            args[0]["mGE"] = pd.DataFrame(np.full(frame.shape, 99.0), index=frame.index, columns=frame.columns)

        def missing_patient(args):
            args[1] = args[1][:-1]
            args[2] = args[2][:-1]
            args[0] = {modality: frame.iloc[:-1].copy() for modality, frame in args[0].items()}

        def outer_test_patient(args):
            args[1] = (self.test_ids[0], *args[1][1:])

        def extra_patient(args):
            args[1] = (*args[1], self.test_ids[0])
            args[2] = np.append(args[2], 1)

        def missing_feature(args):
            args[0]["mGE"] = args[0]["mGE"].iloc[:, 1:].copy()
            args[5]["mGE"] = args[5]["mGE"][1:]

        def extra_feature(args):
            args[0]["mGE"] = args[0]["mGE"].assign(forged_feature=0.0)
            args[5]["mGE"] = (*args[5]["mGE"], "forged_feature")

        def cross_modality(args):
            args[0]["mGE"] = args[0]["mDM"].copy()

        def nan_mismatch(args):
            args[0]["mGE"].iat[0, 0] = np.nan

        def patient_row_values(args):
            frame = args[0]["mCNA"]
            replacement = frame.iloc[[1, 0, *range(2, len(frame))]].copy()
            replacement.index = frame.index
            args[0]["mCNA"] = replacement

        cases = {
            "changed_training_label": lambda args: args[2].__setitem__(0, 1 - args[2][0]),
            "reordered_patient_labels": reordered_labels,
            "short_training_labels": lambda args: args.__setitem__(2, args[2][:-1]),
            "mGE_value": changed_value("mGE"),
            "mDM_value": changed_value("mDM"),
            "mCNA_value": changed_value("mCNA"),
            "same_shape_matrix": same_shape_substitution,
            "row_order": reordered_rows,
            "column_order": reordered_columns,
            "feature_name": renamed_feature,
            "missing_patient": missing_patient,
            "outer_test_patient": outer_test_patient,
            "extra_patient": extra_patient,
            "missing_feature": missing_feature,
            "extra_feature": extra_feature,
            "cross_modality": cross_modality,
            "nan_finite_mismatch": nan_mismatch,
            "patient_row_values": patient_row_values,
        }
        for name, mutate in cases.items():
            with self.subTest(mutation=name):
                args = copied_args()
                mutate(args)
                self._assert_no_final_fit_calls(lambda: fit_final_primary_model(*args, **self._fit_kwargs()))

    def test_live_imputer_state_mutation_is_rejected_before_outer_transform(self):
        bundle, _ = self._build_bundle()
        imputer = bundle.modality_refits["mGE"].preprocessor._pipeline.named_steps["imputer"]
        imputer.statistics_[0] += 1.0
        self._assert_scoring_rejected_before_transform(bundle)

    def test_live_scaler_state_mutation_is_rejected_before_outer_transform(self):
        bundle, _ = self._build_bundle()
        scaler = bundle.modality_refits["mDM"].preprocessor._pipeline.named_steps["scaler"]
        scaler.mean_[0] += 1.0
        scaler.var_[0] += 1.0
        scaler.scale_[0] += 1.0
        self._assert_scoring_rejected_before_transform(bundle)

    def test_genuine_candidate_b_downstream_chain_is_rejected_before_fit(self):
        official_b = self._official_b_result()
        calibration_b = cross_fit_sigmoid_calibration(official_b, run_provenance=self.provenance, aligned_data=self.data)
        final_b = fit_final_sigmoid_calibrator(
            official_b.selected_search.selected_oof_predictions,
            selected_search=official_b.selected_search,
            context=official_b.context,
        )
        threshold_b = select_operational_threshold(calibration_b, search_result=official_b, run_provenance=self.provenance, aligned_data=self.data)
        self._assert_no_final_fit_calls(
            lambda: fit_final_primary_model(
                *self._fit_args()[:9],
                final_b,
                threshold_b,
                **self._fit_kwargs(cross_fitted_calibration=calibration_b),
            )
        )

    def test_tampered_a_chain_is_rejected_before_fit(self):
        cases = {
            "calibration": self._fit_kwargs(cross_fitted_calibration=replace(self.calibration, cross_fitted_calibration_sha256="0" * 64)),
            "final_sigmoid": None,
            "threshold": None,
        }
        for name, kwargs in cases.items():
            with self.subTest(component=name):
                args = self._fit_args()
                if name == "final_sigmoid":
                    args = (*args[:9], replace(self.final_calibrator, final_calibrator_identity_sha256="1" * 64), args[10])
                    kwargs = self._fit_kwargs()
                elif name == "threshold":
                    args = (*args[:10], replace(self.threshold, threshold=0.99))
                    kwargs = self._fit_kwargs()
                self._assert_no_final_fit_calls(lambda: fit_final_primary_model(*args, **kwargs))

    def test_bundle_identity_tampering_and_declared_a_model_b_forgery_are_rejected(self):
        bundle, _ = self._build_bundle()
        candidate_b = next(candidate for candidate in self.official.selected_search.all_candidate_summaries if candidate.candidate.logistic_c != self.candidate.logistic_c).candidate
        classifier_b = copy.deepcopy(bundle.classifier)
        classifier_b.model.C = candidate_b.logistic_c
        classifier_b.evidence = {**classifier_b.evidence, "C": candidate_b.logistic_c}
        forged_evidence = dict(bundle.evidence)
        forged_bundle = replace(bundle, classifier=classifier_b, evidence=forged_evidence)
        forged_evidence["frozen_model_state_sha256"] = final_refit._frozen_model_state_hash(
            forged_bundle.modality_refits,
            forged_bundle.classifier,
            forged_bundle.final_calibrator,
            forged_bundle.threshold,
        )
        cases = {
            "candidate": replace(bundle, candidate_id="forged-candidate"),
            "classifier_c": replace(bundle, classifier=classifier_b),
            "fused_dimensions": replace(bundle, fused_latent_dimensions={"mGE": 3, "mDM": 2, "mCNA": 2}),
            "model_state": replace(bundle, evidence={**bundle.evidence, "frozen_model_state_sha256": "0" * 64}),
            "final_identity": replace(bundle, final_model_identity_sha256="1" * 64),
            "declared_a_model_b": forged_bundle,
        }
        for name, tampered in cases.items():
            with self.subTest(component=name):
                with self.assertRaises(ValueError):
                    validate_final_primary_model_bundle(tampered, search_result=self.official, run_provenance=self.provenance, aligned_data=self.data)

    def test_score_outer_test_is_label_free_fit_free_and_fails_before_transform_for_invalid_bundle(self):
        bundle, _ = self._build_bundle()
        parameters = inspect.signature(score_outer_test).parameters
        self.assertFalse({"label", "labels", "y", "outer_test_labels"}.intersection(parameters))
        transform_original = final_refit.transform_with_preprocessor
        score_original = final_refit.score_logistic_classifier
        transform = Mock(side_effect=transform_original)
        score = Mock(side_effect=score_original)
        with patch("oncoassist_research.final_refit.transform_with_preprocessor", transform), patch(
            "oncoassist_research.final_refit.score_logistic_classifier", score
        ), patch("oncoassist_research.final_refit.build_autoencoder_split", side_effect=AssertionError), patch(
            "oncoassist_research.final_refit.select_epoch_for_modality", side_effect=AssertionError
        ), patch("oncoassist_research.final_refit.refit_selected_epoch_modality", side_effect=AssertionError), patch(
            "oncoassist_research.final_refit.augment_with_minority_ctgan", side_effect=AssertionError
        ), patch("oncoassist_research.final_refit.fit_logistic_classifier", side_effect=AssertionError):
            frozen = score_outer_test(bundle, self.test_modalities, self.test_ids, search_result=self.official, run_provenance=self.provenance, aligned_data=self.data)
        self.assertEqual(transform.call_count, 3)
        self.assertEqual(score.call_count, 1)
        validate_frozen_outer_predictions(frozen)
        invalid = replace(bundle, final_model_identity_sha256="0" * 64)
        with patch("oncoassist_research.final_refit.transform_with_preprocessor", side_effect=AssertionError), patch(
            "oncoassist_research.final_refit.score_logistic_classifier", side_effect=AssertionError
        ):
            with self.assertRaises(ValueError):
                score_outer_test(invalid, self.test_modalities, self.test_ids, search_result=self.official, run_provenance=self.provenance, aligned_data=self.data)

    def test_frozen_prediction_integrity_rejects_record_and_hash_tampering(self):
        bundle, _ = self._build_bundle()
        frozen = score_outer_test(bundle, self.test_modalities, self.test_ids, search_result=self.official, run_provenance=self.provenance, aligned_data=self.data)
        predictions = frozen.predictions
        cases = {
            "raw_score": replace(frozen, predictions=(replace(predictions[0], raw_decision_score=predictions[0].raw_decision_score + 1.0), *predictions[1:])),
            "uncalibrated_probability": replace(frozen, predictions=(replace(predictions[0], uncalibrated_probability=0.0 if predictions[0].uncalibrated_probability != 0.0 else 1.0), *predictions[1:])),
            "calibrated_probability": replace(frozen, predictions=(replace(predictions[0], calibrated_probability=0.0), *predictions[1:])),
            "threshold": replace(frozen, predictions=(replace(predictions[0], threshold=0.0), *predictions[1:])),
            "predicted_label": replace(frozen, predictions=(replace(predictions[0], predicted_label=1 - predictions[0].predicted_label), *predictions[1:])),
            "patient_order": replace(frozen, predictions=tuple(reversed(predictions))),
            "missing_patient": replace(frozen, predictions=predictions[1:]),
            "extra_patient": replace(frozen, predictions=(*predictions, predictions[0])),
            "duplicate_patient": replace(frozen, predictions=(predictions[0], replace(predictions[1], sample_id=predictions[0].sample_id), *predictions[2:])),
            "foreign_patient": replace(frozen, predictions=(replace(predictions[0], sample_id="FOREIGN-PATIENT"), *predictions[1:])),
            "candidate_id": replace(frozen, predictions=(replace(predictions[0], candidate_id="forged-candidate"), *predictions[1:])),
            "candidate_identity": replace(frozen, predictions=(replace(predictions[0], candidate_identity_sha256="0" * 64), *predictions[1:])),
            "model_identity": replace(frozen, predictions=(replace(predictions[0], final_model_identity_sha256="1" * 64), *predictions[1:])),
            "stale_hash": replace(frozen, prediction_hash="2" * 64),
        }
        for name, tampered in cases.items():
            with self.subTest(component=name):
                with self.assertRaises(ValueError):
                    validate_frozen_outer_predictions(tampered)


if __name__ == "__main__":
    unittest.main()
