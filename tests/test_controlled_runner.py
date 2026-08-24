import tempfile
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oncoassist_research import artifacts, controlled_runner as runner
from oncoassist_research.final_refit import FrozenOuterPredictions, OuterPrediction
from oncoassist_research.metrics import BinaryMetrics, RankingMetrics
from oncoassist_research.outer_evaluation import OuterEvaluation
from oncoassist_research.search import build_primary_candidates


class ControlledRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = runner.ControlledRunnerConfig(
            Path("rna.csv"), Path("dna.csv"), Path("cna.csv"), self.root / "outputs", "synthetic-controlled", 19, "cpu"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def data(self, majority=25, minority=20, feature_count=8):
        ids = [f"P{index:03d}" for index in range(majority + minority)]
        labels = np.asarray([0] * majority + [1] * minority, dtype=int)
        feature_columns = {
            "rna": tuple(f"rna_{index}" for index in range(feature_count)),
            "dna": tuple(f"dna_{index}" for index in range(feature_count)),
            "cna": tuple(f"cna_{index}" for index in range(feature_count)),
        }
        result = {
            "sample_ids": ids,
            "y_binary": labels,
            "feature_columns": feature_columns,
            "label_mapping": {"raw_to_binary": {"1": 0, "2": 1}},
            "audit_summary": {
                "files": {
                    "mGE": {"sha256": "a" * 64, "provenance": "synthetic_test_fixture"},
                    "mDM": {"sha256": "b" * 64, "provenance": "synthetic_test_fixture"},
                    "CNA": {"sha256": "c" * 64, "provenance": "synthetic_test_fixture"},
                }
            },
        }
        for offset, (matrix_key, feature_key) in enumerate((("X_rna", "rna"), ("X_dna", "dna"), ("X_cna", "cna"))):
            result[matrix_key] = pd.DataFrame(
                np.arange(len(ids) * feature_count, dtype=float).reshape(len(ids), feature_count) + offset,
                index=ids,
                columns=feature_columns[feature_key],
            )
        return result

    def environment(self):
        compatibility = {
            "schema_version": "controlled-primary-v1-environment-v1",
            "python": "test",
            "package_versions": {"numpy": "test"},
            "ae_device_policy": "cpu",
        }
        return {
            **compatibility,
            "environment_compatibility_identity_sha256": artifacts.payload_sha256(compatibility),
            "visible_tensorflow_devices": [],
            "gpu_inventory_informational": [],
        }

    def worker_probe(self):
        return {
            "schema_version": "research-minority-ctgan-worker-v1",
            "preflight_only": True,
            "ctgan_execution_backend": "isolated_cpu_subprocess_v1",
            "cuda_visible_devices": "",
            "ctgan_gpu_enabled": False,
            "constructor_parameters": ["metadata", "epochs", "batch_size", "pac", "verbose"],
            "versions": {"test": "test"},
        }

    def prepare(self, data=None, config=None):
        with patch("oncoassist_research.controlled_runner.load_and_align_multiomics", return_value=data or self.data()), patch(
            "oncoassist_research.controlled_runner.probe_isolated_ctgan_worker", return_value=self.worker_probe()
        ), patch("oncoassist_research.controlled_runner._runtime_environment", return_value=self.environment()), patch(
            "oncoassist_research.controlled_runner._filesystem_capability_probe",
            return_value={"create_once_files": True, "atomic_replace": True, "same_parent_directory_publication": True, "locking": True},
        ):
            return runner.prepare_study(config or self.config)

    def scoring_receipt(self, prepared, repeat_id=0, fold_id=0):
        binding = prepared.binding.payload
        return runner.ScoringPublicationReceipt(
            study_identity_sha256=prepared.binding.study_identity_sha256,
            repeat_id=repeat_id,
            fold_id=fold_id,
            run_provenance_identity_sha256=binding["run_provenance_identity_sha256"],
            immutable_reference_sha256=binding["immutable_reference_sha256"],
            protocol_sha256=binding["protocol_identity_sha256"],
            seed_manifest_identity_sha256=binding["seed_manifest_identity_sha256"],
            outer_manifest_identity_sha256=binding["outer_manifest_identity_sha256"],
            inner_manifest_identity_sha256=binding["inner_manifest_identities_sha256"][f"{repeat_id}:{fold_id}"],
            fold_authority_identity_sha256=binding["coordinate_fold_authority_sha256"][f"{repeat_id}:{fold_id}"],
            search_selection_identity_sha256="1" * 64,
            selected_candidate_identity_sha256="2" * 64,
            selected_oof_predictions_sha256="3" * 64,
            cross_fitted_calibration_sha256="4" * 64,
            final_sigmoid_identity_sha256="5" * 64,
            threshold_identity_sha256="6" * 64,
            final_model_identity_sha256="7" * 64,
            frozen_model_state_sha256="8" * 64,
            aligned_data_content_identity_sha256=binding["aligned_data_content_identity_sha256"],
            ordered_outer_test_ids_sha256=runner.ordered_patient_ids_sha256(self.outer_test_ids(prepared, repeat_id, fold_id)),
            frozen_prediction_hash="9" * 64,
        )

    def outer_test_ids(self, prepared, repeat_id=0, fold_id=0):
        return next(record["test_sample_ids"] for record in prepared.outer_manifest["folds"] if (record["repeat_id"], record["fold_id"]) == (repeat_id, fold_id))

    def frozen_fixture(self, prepared, receipt, repeat_id=0, fold_id=0):
        ids = self.outer_test_ids(prepared, repeat_id, fold_id)
        predictions = tuple(
            OuterPrediction(sample_id, float(index), 0.25, 0.75, 0.5, 1, "candidate", receipt.selected_candidate_identity_sha256, receipt.final_model_identity_sha256)
            for index, sample_id in enumerate(ids)
        )
        prediction_hash = artifacts.payload_sha256([
            {"sample_id": item.sample_id, "raw_decision_score": item.raw_decision_score, "uncalibrated_probability": item.uncalibrated_probability, "calibrated_probability": item.calibrated_probability, "threshold": item.threshold, "predicted_label": item.predicted_label, "candidate_id": item.candidate_id, "candidate_identity_sha256": item.candidate_identity_sha256, "final_model_identity_sha256": item.final_model_identity_sha256}
            for item in predictions
        ])
        frozen = FrozenOuterPredictions(
            predictions,
            runner.ordered_patient_ids_sha256(ids),
            prediction_hash,
            {
                "prediction_state_frozen": True,
                "outer_labels_seen": False,
                "run_provenance_identity_sha256": receipt.run_provenance_identity_sha256,
                "aligned_data_content_identity_sha256": receipt.aligned_data_content_identity_sha256,
                "protocol_sha256": receipt.protocol_sha256,
                "search_selection_identity_sha256": receipt.search_selection_identity_sha256,
                "candidate_id": "candidate",
                "candidate_identity_sha256": receipt.selected_candidate_identity_sha256,
                "final_model_identity_sha256": receipt.final_model_identity_sha256,
                "frozen_model_state_sha256": receipt.frozen_model_state_sha256,
                "outer_manifest_identity_sha256": receipt.outer_manifest_identity_sha256,
                "inner_manifest_identity_sha256": receipt.inner_manifest_identity_sha256,
                "fold_authority_identity_sha256": receipt.fold_authority_identity_sha256,
            },
            "candidate",
            receipt.selected_candidate_identity_sha256,
            receipt.final_model_identity_sha256,
            receipt.search_selection_identity_sha256,
        )
        return frozen

    def publish_scoring_fixture(self, prepared, repeat_id=0, fold_id=0):
        target = prepared.study_directory / "outer_folds" / f"repeat-{repeat_id:02d}" / f"fold-{fold_id:02d}" / "scoring"
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = self.scoring_receipt(prepared, repeat_id, fold_id)
        frozen = self.frozen_fixture(prepared, receipt, repeat_id, fold_id)
        receipt = runner.ScoringPublicationReceipt(**{**receipt.__dict__, "frozen_prediction_hash": frozen.prediction_hash})
        documents = {
            "publication.json": receipt.as_json(),
            "frozen_predictions.json": runner._frozen_predictions_json(frozen),
            "selected_search_summary.json": {"selected_candidate": {"fixture": "candidate"}, "all_candidate_summaries": [], "search_selection_identity_sha256": receipt.search_selection_identity_sha256, "selected_candidate_identity_sha256": receipt.selected_candidate_identity_sha256, "selected_oof_predictions_sha256": receipt.selected_oof_predictions_sha256},
            "calibration_threshold_summary.json": {"cross_fitted_calibration_sha256": receipt.cross_fitted_calibration_sha256, "final_sigmoid_identity_sha256": receipt.final_sigmoid_identity_sha256, "threshold_identity_sha256": receipt.threshold_identity_sha256},
            "final_refit_summary.json": {"candidate_identity_sha256": receipt.selected_candidate_identity_sha256, "final_model_identity_sha256": receipt.final_model_identity_sha256, "frozen_model_state_sha256": receipt.frozen_model_state_sha256},
        }
        artifacts.publish_directory(target, lambda temporary: [artifacts.create_immutable_json(temporary / name, payload) for name, payload in documents.items()])
        return receipt

    def publish_evaluation_fixture(self, prepared, receipt, repeat_id=0, fold_id=0):
        target = prepared.study_directory / "outer_folds" / f"repeat-{repeat_id:02d}" / f"fold-{fold_id:02d}" / "evaluation"
        evaluation = {"fixture": "evaluation"}
        content = {
            "schema_version": runner.EVALUATION_PUBLICATION_SCHEMA_VERSION,
            "study_identity_sha256": prepared.binding.study_identity_sha256,
            "coordinate": {"repeat_id": repeat_id, "fold_id": fold_id},
            "scoring_publication_identity_sha256": receipt.scoring_publication_identity_sha256,
            "evaluation_payload_sha256": artifacts.payload_sha256(evaluation),
        }
        publication = {**content, "evaluation_publication_identity_sha256": artifacts.payload_sha256(content)}
        artifacts.publish_directory(target, lambda temporary: [artifacts.create_immutable_json(temporary / "publication.json", publication), artifacts.create_immutable_json(temporary / "evaluation.json", evaluation)])

    def execution_fixtures(self, prepared, repeat_id=0, fold_id=0):
        outer = next(record for record in prepared.outer_manifest["folds"] if (record["repeat_id"], record["fold_id"]) == (repeat_id, fold_id))
        inner = next(record for record in prepared.inner_manifest["outer_folds"] if (record["repeat_id"], record["fold_id"]) == (repeat_id, fold_id))
        candidate = build_primary_candidates({"mGE": 8, "mDM": 8, "mCNA": 8}, prepared.protocol)[0]
        selected = SimpleNamespace(
            selected_oof_predictions=(),
            selected_candidate=candidate,
            selected_candidate_identity_sha256=candidate.candidate_identity_sha256,
            selected_oof_predictions_sha256="a" * 64,
            search_selection_identity_sha256="b" * 64,
            all_candidate_summaries=(SimpleNamespace(candidate_id=candidate.candidate_id, candidate_identity_sha256=candidate.candidate_identity_sha256, mean_inner_auprc=.5, mean_inner_auroc=.6, inner_auprc_sd=.1, identity_sha256="c" * 64),),
        )
        context = SimpleNamespace(
            repeat_id=repeat_id,
            fold_id=fold_id,
            outer_training_ids=tuple(outer["train_sample_ids"]),
            outer_testing_ids=tuple(outer["test_sample_ids"]),
            inner_folds=tuple({"inner_fold_id": item["inner_fold_id"], "inner_train_sample_ids": tuple(item["inner_train_sample_ids"]), "inner_validation_sample_ids": tuple(item["inner_validation_sample_ids"])} for item in inner["inner_folds"]),
            outer_manifest_identity_sha256=prepared.binding.payload["outer_manifest_identity_sha256"],
            inner_manifest_identity_sha256=prepared.binding.payload["inner_manifest_identities_sha256"][f"{repeat_id}:{fold_id}"],
            fold_authority_identity_sha256=prepared.binding.payload["coordinate_fold_authority_sha256"][f"{repeat_id}:{fold_id}"],
            run_provenance_identity_sha256=prepared.provenance.identity_sha256,
            seed_manifest=prepared.seed_manifest,
        )
        search = SimpleNamespace(context=context, selected_search=selected)
        calibration = SimpleNamespace(cross_fitted_calibration_sha256="d" * 64)
        calibrator = SimpleNamespace(final_calibrator_identity_sha256="e" * 64)
        threshold = SimpleNamespace(threshold_identity_sha256="f" * 64, threshold=.5, metrics=BinaryMetrics(.5, .5, .5, .5, .5, .5, .0, 1, 1, 1, 1, .5, "probability"))
        bundle = SimpleNamespace(candidate_id=candidate.candidate_id, candidate_identity_sha256=candidate.candidate_identity_sha256, final_model_identity_sha256="1" * 64, evidence={"frozen_model_state_sha256": "2" * 64})
        receipt = runner.ScoringPublicationReceipt(
            prepared.binding.study_identity_sha256, repeat_id, fold_id, prepared.provenance.identity_sha256,
            prepared.binding.payload["immutable_reference_sha256"], prepared.protocol.identity_sha256,
            prepared.seed_manifest.identity_sha256, context.outer_manifest_identity_sha256, context.inner_manifest_identity_sha256,
            context.fold_authority_identity_sha256, selected.search_selection_identity_sha256,
            candidate.candidate_identity_sha256, selected.selected_oof_predictions_sha256,
            calibration.cross_fitted_calibration_sha256, calibrator.final_calibrator_identity_sha256,
            threshold.threshold_identity_sha256, bundle.final_model_identity_sha256, bundle.evidence["frozen_model_state_sha256"],
            prepared.provenance.aligned_data_content_identity_sha256, runner.ordered_patient_ids_sha256(context.outer_testing_ids), "0" * 64,
        )
        frozen = self.frozen_fixture(prepared, receipt, repeat_id, fold_id)
        receipt = runner.ScoringPublicationReceipt(**{**receipt.__dict__, "frozen_prediction_hash": frozen.prediction_hash})
        evaluation = OuterEvaluation(
            RankingMetrics(.7, .8),
            BinaryMetrics(.6, .7, .5, .6, .5, .6, .1, 2, 1, 1, 2, .5, "frozen_predicted_label"),
            .2,
            .3,
            {"secondary_calibrated_probability_ranking_diagnostic": {"auprc": .6, "auroc": .7, "not_primary": True}},
            {"fixture": True},
        )
        return search, calibration, calibrator, threshold, bundle, frozen, evaluation

    def test_study_identity_is_deterministic_and_coordinate_plan_is_complete(self):
        first, second = self.prepare(), self.prepare()
        self.assertEqual(first.binding.study_identity_sha256, second.binding.study_identity_sha256)
        self.assertEqual(runner.primary_coordinate_plan(first.protocol), tuple((repeat_id, fold_id) for repeat_id in range(5) for fold_id in range(5)))
        self.assertEqual(len(first.binding.payload["coordinates"]), 25)

    def test_wrong_reference_sha_fails_before_data_loading(self):
        reference = self.root / "reference.py"
        reference.write_text("wrong", encoding="utf-8")
        config = runner.ControlledRunnerConfig(Path("r"), Path("d"), Path("c"), self.root / "outputs", "run", 1, "cpu", reference)
        with patch("oncoassist_research.controlled_runner.load_and_align_multiomics", side_effect=AssertionError):
            with self.assertRaises(ValueError):
                runner.prepare_study(config)

    def test_ctgan_infeasibility_fails_before_worker_probe_or_scientific_execution(self):
        with patch("oncoassist_research.controlled_runner.load_and_align_multiomics", return_value=self.data(25, 5)), patch(
            "oncoassist_research.controlled_runner.probe_isolated_ctgan_worker", side_effect=AssertionError
        ), patch("oncoassist_research.search.run_primary_inner_search", side_effect=AssertionError):
            with self.assertRaises(ValueError):
                runner.prepare_study(self.config)

    def test_empty_study_reconstructs_pending_and_runtime_cache_is_not_authority(self):
        prepared = self.prepare()
        runner.initialize_study(prepared)
        self.assertEqual(len(list((prepared.study_directory / "outer_folds").glob("repeat-*/fold-*"))), 25)
        state = runner.reconstruct_runtime_state(prepared.study_directory, prepared.binding)
        self.assertEqual(state["derived_state"], "PREFLIGHTED")
        self.assertEqual({item["state"] for item in state["coordinates"]}, {"PENDING"})
        working = runner.write_runtime_state(prepared.study_directory, prepared.binding, "RUNNING", (0, 0))
        self.assertEqual(working["coordinates"][0]["state"], "EXECUTING")
        artifacts.atomic_write_json(prepared.study_directory / "runtime_state.json", {"derived_state": "COMPLETE"})
        self.assertEqual(runner.reconstruct_runtime_state(prepared.study_directory, prepared.binding)["complete_coordinate_count"], 0)

    def test_resume_classifications_and_corrupt_evidence_fail_closed(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        receipt = self.publish_scoring_fixture(prepared)
        self.assertEqual(runner.classify_coordinate(prepared.study_directory, prepared.binding, 0, 0)["resume_classification"], "EVALUATION_ONLY_RESUME")
        self.publish_evaluation_fixture(prepared, receipt)
        self.assertEqual(runner.classify_coordinate(prepared.study_directory, prepared.binding, 0, 0)["resume_classification"], "COMPLETE")
        target = prepared.study_directory / "outer_folds" / "repeat-00" / "fold-01" / "scoring"
        target.mkdir(parents=True)
        artifacts.create_immutable_json(target / "publication.json", {"bad": True})
        with self.assertRaises(ValueError):
            runner.classify_coordinate(prepared.study_directory, prepared.binding, 0, 1)

    def test_duplicate_publication_and_binding_or_manifest_mismatch_fail(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        self.publish_scoring_fixture(prepared)
        with self.assertRaises(FileExistsError):
            self.publish_scoring_fixture(prepared)
        changed_data = self.data(feature_count=9)
        changed = self.prepare(changed_data)
        with self.assertRaises(ValueError):
            runner.validate_study_directory(prepared.study_directory, changed.binding)
        artifacts.atomic_write_json(prepared.study_directory / "fold_manifests" / "outer.json", {"bad": True})
        with self.assertRaises(ValueError):
            runner.reconstruct_runtime_state(prepared.study_directory, prepared.binding)

    def test_every_scientific_or_environment_binding_mismatch_prevents_resume(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        for name in (
            "protocol_identity_sha256",
            "run_provenance_identity_sha256",
            "aligned_data_content_identity_sha256",
            "seed_manifest_identity_sha256",
            "fold_protocol_identity_sha256",
            "outer_manifest_identity_sha256",
            "environment_compatibility_identity_sha256",
        ):
            with self.subTest(binding=name):
                payload = dict(prepared.binding.payload)
                payload[name] = "0" * 64
                with self.assertRaises(ValueError):
                    runner.validate_study_directory(prepared.study_directory, runner.StudyBinding(payload))
        payload = dict(prepared.binding.payload)
        payload["feature_schema_sha256"] = {"mGE": "0" * 64, "mDM": "0" * 64, "mCNA": "0" * 64}
        with self.assertRaises(ValueError):
            runner.validate_study_directory(prepared.study_directory, runner.StudyBinding(payload))

    def test_explicit_lock_recovery_never_breaks_remote_lock_automatically(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        binding = runner._study_lock_binding(prepared.binding)
        lock_path = prepared.study_directory / ".run_lock.json"
        artifacts.atomic_write_json(lock_path, {"schema_version": artifacts.RUN_LOCK_SCHEMA_VERSION, "lock_id": "remote-lock", "binding": binding, "owner": {"hostname": "remote", "pid": 1, "process_identity": "remote:1"}, "lifecycle": {}})
        with self.assertRaises(RuntimeError):
            runner.acquire_study_lock(prepared.study_directory, prepared.binding)
        with self.assertRaises(ValueError):
            runner.recover_abandoned_study_lock(prepared.study_directory, prepared.binding, "wrong-lock")
        runner.recover_abandoned_study_lock(prepared.study_directory, prepared.binding, "remote-lock")
        lock = runner.acquire_study_lock(prepared.study_directory, prepared.binding)
        artifacts.release_run_lock(lock, outcome="test")

    def test_execute_coordinate_uses_public_chain_then_publishes_and_evaluates(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        search, calibration, calibrator, threshold, bundle, frozen, evaluation = self.execution_fixtures(prepared)
        order = []
        def searched(**kwargs):
            order.append("search")
            self.assertNotIn("outer_test_labels", kwargs)
            return search
        def fitted(*args, **kwargs):
            order.append("final_refit")
            self.assertEqual(kwargs["search_result"].selected_search.selected_candidate.logistic_c, search.selected_search.selected_candidate.logistic_c)
            self.assertEqual(tuple(args[1]), search.context.outer_training_ids)
            self.assertEqual(tuple(args[4]), search.context.outer_testing_ids)
            return bundle
        def scored(*args, **kwargs):
            order.append("score")
            self.assertNotIn("labels", kwargs)
            return frozen
        def evaluated(*args, **kwargs):
            order.append("evaluate")
            self.assertTrue((prepared.study_directory / "outer_folds" / "repeat-00" / "fold-00" / "scoring").is_dir())
            self.assertIn("scoring_publication_receipt", kwargs)
            return evaluation
        with patch("oncoassist_research.controlled_runner.run_primary_inner_search", side_effect=searched), patch(
            "oncoassist_research.controlled_runner._validate_search_coordinate"
        ), patch("oncoassist_research.controlled_runner.cross_fit_sigmoid_calibration", side_effect=lambda *args, **kwargs: (order.append("calibration"), calibration)[1]), patch(
            "oncoassist_research.controlled_runner.select_operational_threshold", side_effect=lambda *args, **kwargs: (order.append("threshold"), threshold)[1]
        ), patch("oncoassist_research.controlled_runner.fit_final_sigmoid_calibrator", side_effect=lambda *args, **kwargs: (order.append("final_sigmoid"), calibrator)[1]), patch(
            "oncoassist_research.controlled_runner.fit_final_primary_model", side_effect=fitted
        ), patch("oncoassist_research.controlled_runner.validate_final_primary_model_bundle"), patch(
            "oncoassist_research.controlled_runner.score_outer_test", side_effect=scored
        ), patch("oncoassist_research.controlled_runner.evaluate_outer_test_predictions", side_effect=evaluated):
            result = runner.execute_coordinate(prepared, (0, 0))
        self.assertEqual(result.resulting_classification, "COMPLETE")
        self.assertEqual(order, ["search", "calibration", "threshold", "final_sigmoid", "final_refit", "score", "evaluate"])
        publication = artifacts.read_json_object(prepared.study_directory / "outer_folds" / "repeat-00" / "fold-00" / "scoring" / "publication.json")
        self.assertEqual(publication["frozen_prediction_hash"], frozen.prediction_hash)
        aggregation = artifacts.read_json_object(prepared.study_directory / "outer_folds" / "repeat-00" / "fold-00" / "evaluation" / "evaluation.json")["aggregation_input"]
        for name in ("raw_score_auprc", "raw_score_auroc", "frozen_threshold_metrics", "brier_score", "log_loss", "final_model_identity_sha256", "frozen_prediction_hash"):
            self.assertIn(name, aggregation)

    def test_evaluation_only_resume_does_no_scientific_work(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        receipt = self.publish_scoring_fixture(prepared)
        evaluation = OuterEvaluation(RankingMetrics(.7, .8), BinaryMetrics(.6, .7, .5, .6, .5, .6, .1, 2, 1, 1, 2, .5, "frozen_predicted_label"), .2, .3, {}, {})
        with patch("oncoassist_research.controlled_runner.run_primary_inner_search", side_effect=AssertionError), patch(
            "oncoassist_research.controlled_runner.fit_final_primary_model", side_effect=AssertionError
        ), patch("oncoassist_research.controlled_runner.score_outer_test", side_effect=AssertionError), patch(
            "oncoassist_research.controlled_runner.evaluate_outer_test_predictions", return_value=evaluation
        ) as evaluate:
            result = runner.execute_coordinate(prepared, (0, 0))
        self.assertEqual(result.resume_classification, "EVALUATION_ONLY_RESUME")
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(runner.classify_coordinate(prepared.study_directory, prepared.binding, 0, 0)["resume_classification"], "COMPLETE")

    def test_published_receipt_evaluation_requires_valid_frozen_predictions_without_models(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        receipt = self.publish_scoring_fixture(prepared)
        loaded_receipt, frozen = runner._load_scoring_publication(prepared.study_directory, prepared.binding, 0, 0)
        ids = self.outer_test_ids(prepared)
        labels = runner._canonical_labels(prepared, ids)
        from oncoassist_research.outer_evaluation import evaluate_outer_test_predictions
        result = evaluate_outer_test_predictions(
            frozen, labels, ids, run_provenance=prepared.provenance, aligned_data=prepared.aligned_data,
            scoring_publication_receipt=loaded_receipt.as_json(),
        )
        self.assertEqual(result.evidence["prediction_hash"], receipt.frozen_prediction_hash)
        corrupt = dict(runner._frozen_predictions_json(frozen)); corrupt["prediction_hash"] = "0" * 64
        artifacts.atomic_write_json(prepared.study_directory / "outer_folds" / "repeat-00" / "fold-00" / "scoring" / "frozen_predictions.json", corrupt)
        with self.assertRaises(ValueError):
            runner.classify_coordinate(prepared.study_directory, prepared.binding, 0, 0)

    def test_evaluation_publication_failure_preserves_scoring_for_retry(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        self.publish_scoring_fixture(prepared)
        evaluation = OuterEvaluation(RankingMetrics(.7, .8), BinaryMetrics(.6, .7, .5, .6, .5, .6, .1, 2, 1, 1, 2, .5, "frozen_predicted_label"), .2, .3, {}, {})
        with patch("oncoassist_research.controlled_runner.evaluate_outer_test_predictions", return_value=evaluation), patch(
            "oncoassist_research.controlled_runner._publish_evaluation", side_effect=RuntimeError("publication failed")
        ):
            with self.assertRaises(RuntimeError):
                runner.execute_coordinate(prepared, (0, 0))
        self.assertTrue((prepared.study_directory / "outer_folds" / "repeat-00" / "fold-00" / "scoring").is_dir())
        self.assertEqual(runner.classify_coordinate(prepared.study_directory, prepared.binding, 0, 0)["resume_classification"], "EVALUATION_ONLY_RESUME")

    def test_execute_rejects_invalid_coordinate_and_run_study_is_sequential(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        self.assertEqual(tuple(inspect.signature(runner.execute_coordinate).parameters), ("prepared", "coordinate"))
        with self.assertRaises(ValueError):
            runner.execute_coordinate(prepared, (5, 0))
        observed = []
        def execute(study, coordinate):
            observed.append(coordinate)
            return runner.CoordinateExecutionResult(*coordinate, "COMPLETE", "COMPLETE", "a" * 64, "b" * 64)
        with patch("oncoassist_research.controlled_runner.execute_coordinate", side_effect=execute):
            results = runner.run_study(prepared)
        self.assertEqual(observed, list(runner.primary_coordinate_plan(prepared.protocol)))
        self.assertEqual(len(results), 25)

    def test_interrupted_coordinate_records_sanitized_failure_and_remains_recomputable(self):
        prepared = self.prepare(); runner.initialize_study(prepared)
        with patch("oncoassist_research.controlled_runner.run_primary_inner_search", side_effect=RuntimeError("inner search interrupted")):
            with self.assertRaises(RuntimeError):
                runner.execute_coordinate(prepared, (0, 0))
        failure_files = list((prepared.study_directory / "failures").glob("*.json"))
        self.assertEqual(len(failure_files), 1)
        failure = artifacts.read_json_object(failure_files[0])
        self.assertEqual(failure["stage"], "coordinate_execution")
        self.assertFalse({"raw_biological_matrices", "patient_labels", "model_weights", "synthetic_feature_rows"}.intersection(failure))
        self.assertEqual(runner.classify_coordinate(prepared.study_directory, prepared.binding, 0, 0)["resume_classification"], "FULL_COORDINATE_RECOMPUTE")
        receipt = self.scoring_receipt(prepared)
        self.assertNotIn("model", receipt.as_json())
        self.assertNotIn("weights", receipt.as_json())


if __name__ == "__main__":
    unittest.main()
