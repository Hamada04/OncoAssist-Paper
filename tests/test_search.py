import inspect
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from oncoassist_research import folds
from oncoassist_research.classifiers import LogisticScores
from oncoassist_research.ctgan import AugmentedTrainingSet
from oncoassist_research.protocol import ModalityAdapter, PrimaryProtocolV1, PrimarySeedManifest, create_primary_v1_run_provenance, patient_set_sha256
from oncoassist_research.search import (
    PrimarySearchResult,
    PrimarySharedBuilderBinding,
    RawPrimarySearchExecution,
    _build_primary_search_context,
    _derive_primary_fold_authority,
    _finalize_primary_search_execution,
    _run_primary_inner_search_with_builder,
    build_primary_candidates,
    run_primary_inner_search,
    validate_primary_search_result,
)


def context_and_binding():
    ids = tuple(f"P{index:02d}" for index in range(12))
    labels = np.asarray([0, 1] * 6)
    names = {"rna": tuple(f"r{index}" for index in range(8)), "dna": tuple(f"d{index}" for index in range(8)), "cna": tuple(f"c{index}" for index in range(8))}
    data = {
        "feature_columns": names,
        "sample_ids": list(ids),
        "y_binary": labels,
        "label_mapping": {"raw_to_binary": {"1": 0, "2": 1}},
        "audit_summary": {"files": {"mGE": {"sha256": "a" * 64, "provenance": "synthetic_test_fixture"}, "mDM": {"sha256": "b" * 64, "provenance": "synthetic_test_fixture"}, "CNA": {"sha256": "c" * 64, "provenance": "synthetic_test_fixture"}}},
    }
    for offset, (key, name) in enumerate((("X_rna", "rna"), ("X_dna", "dna"), ("X_cna", "cna"))):
        data[key] = pd.DataFrame(np.arange(96.0).reshape(12, 8) + offset, index=ids, columns=names[name])
    protocol = PrimaryProtocolV1()
    adapter = ModalityAdapter.from_aligned_data(data)
    candidates = build_primary_candidates({"mGE": 8, "mDM": 8, "mCNA": 8})
    provenance = create_primary_v1_run_provenance(run_id="synthetic-search", root_seed=7, protocol=protocol, aligned_data=data)
    manifest = PrimarySeedManifest.generate_primary(provenance, candidates)
    fold_protocol = protocol.make_fold_protocol(manifest, provenance)
    authority = _derive_primary_fold_authority(protocol, provenance, data, 0, 0)
    context = _build_primary_search_context(protocol, manifest, fold_protocol, adapter, data, authority, provenance)

    def build(ratio, dimensions, inner, book):
        from oncoassist_research.search import SharedInnerRepresentation
        validation_ids = tuple(inner["inner_validation_sample_ids"])
        validation_labels = np.asarray([labels[ids.index(value)] for value in validation_ids])
        augmented = AugmentedTrainingSet(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), np.array([0, 1]), ("R", "S"), np.array([False, True]), ("z0", "z1"), {})
        return SharedInnerRepresentation(augmented, np.zeros((len(validation_ids), 2), dtype=np.float32), validation_ids, augmented.feature_names, validation_labels, {})

    binding = PrimarySharedBuilderBinding("primary-shared-builder-v1", "primary-inner-v1", protocol.protocol_id, protocol.identity_sha256, patient_set_sha256(context.outer_training_ids), adapter.identity_sha256, fold_protocol, protocol.fold_protocol_identity_sha256(fold_protocol, manifest, provenance), manifest.identity_sha256, provenance.identity_sha256, build)
    return context, data, binding, provenance


def result():
    context, data, binding, provenance = context_and_binding()

    def score(config, features, ids, names):
        raw = np.asarray([1.0 if int(sample_id[1:]) % 2 else -1.0 for sample_id in ids])
        return LogisticScores(tuple(ids), raw, 1.0 / (1.0 + np.exp(-raw)), {})

    return _finalize_primary_search_execution(context, _run_primary_inner_search_with_builder(context, binding, lambda augmented, config, seed: config, score))


class SearchTests(unittest.TestCase):
    def test_public_signature_has_only_repeat_fold_authority(self):
        names = inspect.signature(run_primary_inner_search).parameters
        self.assertIn("repeat_id", names)
        self.assertIn("fold_id", names)
        for name in ("fold_protocol", "outer_training_ids", "outer_training_labels", "inner_folds", "outer_fold_identity", "context", "shared_builder"):
            self.assertNotIn(name, names)

    def test_context_is_derived_from_exact_manifest_records(self):
        context, data, _, provenance = context_and_binding()
        protocol, seed_manifest = context.protocol, context.seed_manifest
        fold_protocol = protocol.make_fold_protocol(seed_manifest, provenance)
        fingerprint = folds.build_outer_data_fingerprint(data)
        outer = folds.build_outer_fold_manifest(data["sample_ids"], data["y_binary"], fingerprint, fold_protocol)
        outer_identity = folds.manifest_identity_sha256(outer)
        inner = folds.build_inner_fold_manifest(outer, outer_identity, data["sample_ids"], data["y_binary"], fingerprint, fold_protocol)
        inner_identity = folds.manifest_identity_sha256(inner)
        outer_record = next(record for record in outer["folds"] if (record["repeat_id"], record["fold_id"]) == (0, 0))
        inner_record = next(record for record in inner["outer_folds"] if (record["repeat_id"], record["fold_id"]) == (0, 0))
        self.assertEqual(context.outer_training_ids, tuple(outer_record["train_sample_ids"]))
        self.assertEqual(context.outer_testing_ids, tuple(outer_record["test_sample_ids"]))
        self.assertEqual(context.inner_folds, tuple({"inner_fold_id": record["inner_fold_id"], "inner_train_sample_ids": tuple(record["inner_train_sample_ids"]), "inner_validation_sample_ids": tuple(record["inner_validation_sample_ids"])} for record in inner_record["inner_folds"]))
        self.assertEqual(context.outer_manifest_identity_sha256, outer_identity)
        self.assertEqual(context.inner_manifest_identity_sha256, inner_identity)

    def test_public_search_derives_manifest_fold_and_retains_authority(self):
        context, data, binding, provenance = context_and_binding()
        raw = _run_primary_inner_search_with_builder(context, binding, lambda augmented, config, seed: config, lambda config, features, ids, names: LogisticScores(tuple(ids), np.asarray([1.0 if int(sample_id[1:]) % 2 else -1.0 for sample_id in ids]), np.asarray([0.75 if int(sample_id[1:]) % 2 else 0.25 for sample_id in ids]), {}))
        with patch("oncoassist_research.primary_inner.make_primary_inner_builder", return_value=binding), patch("oncoassist_research.search._run_primary_inner_search_with_builder", return_value=raw):
            official = run_primary_inner_search(run_provenance=provenance, aligned_data=data, repeat_id=0, fold_id=0, ae_training_config=context.protocol.make_autoencoder_training_config(), ctgan_config=context.protocol.make_ctgan_config(), ae_validation_fraction=.2, synthetic_namespace_prefix="X")
        self.assertIsInstance(official, PrimarySearchResult)
        self.assertEqual(official.context.fold_authority_identity_sha256, context.fold_authority_identity_sha256)
        self.assertEqual(official.selected_search.outer_manifest_identity_sha256, context.outer_manifest_identity_sha256)
        self.assertEqual(official.selected_search.inner_manifest_identity_sha256, context.inner_manifest_identity_sha256)
        validate_primary_search_result(official, run_provenance=provenance, aligned_data=data)

    def test_out_of_range_repeat_fold_is_rejected_before_builder_execution(self):
        context, data, _, provenance = context_and_binding()
        builder = Mock()
        with patch("oncoassist_research.primary_inner.make_primary_inner_builder", builder):
            with self.assertRaises(ValueError):
                run_primary_inner_search(run_provenance=provenance, aligned_data=data, repeat_id=5, fold_id=0, ae_training_config=context.protocol.make_autoencoder_training_config(), ctgan_config=context.protocol.make_ctgan_config(), ae_validation_fraction=.2, synthetic_namespace_prefix="X")
        self.assertEqual(builder.call_count, 0)

    def test_official_result_validator_rejects_tampered_content_and_fold_authority(self):
        official = result(); context, data, _, provenance = context_and_binding()
        validate_primary_search_result(official, run_provenance=provenance, aligned_data=data)
        with self.assertRaises(ValueError):
            validate_primary_search_result(replace(official, context=replace(official.context, fold_authority_identity_sha256="x" * 64)), run_provenance=provenance, aligned_data=data)
        altered = replace(official.selected_search, selected_oof_predictions=official.selected_search.selected_oof_predictions[:-1])
        with self.assertRaises(ValueError):
            validate_primary_search_result(replace(official, selected_search=altered), run_provenance=provenance, aligned_data=data)
        with self.assertRaises(ValueError):
            validate_primary_search_result(replace(official, selected_search=replace(official.selected_search, outer_manifest_identity_sha256="y" * 64)), run_provenance=provenance, aligned_data=data)

    def test_selected_oof_adversarial_records_rejected(self):
        official = result(); context, data, _, provenance = context_and_binding()
        artifact = official.selected_search
        records = list(artifact.selected_oof_predictions)
        candidate_b = artifact.all_candidate_summaries[1].candidate
        cases = [
            tuple([replace(records[0], inner_fold_id=(records[0].inner_fold_id + 1) % 3), *records[1:]]),
            tuple([replace(records[0], true_label=1 - records[0].true_label), *records[1:]]),
            tuple(records[:-1]),
            tuple([replace(records[0], sample_id="FOREIGN"), *records[1:]]),
            tuple([records[0], *records[1:-1], replace(records[-1], sample_id=records[0].sample_id)]),
            tuple([replace(records[0], sample_id="SYNTHETIC:attack"), *records[1:]]),
            tuple([replace(records[0], candidate_id=candidate_b.candidate_id, candidate_identity_sha256=candidate_b.candidate_identity_sha256), *records[1:]]),
            tuple([replace(records[0], decision_score=records[0].decision_score + 0.25), *records[1:]]),
        ]
        for forged in cases:
            with self.subTest(forged=forged[0]):
                with self.assertRaises(ValueError):
                    validate_primary_search_result(replace(official, selected_search=replace(artifact, selected_oof_predictions=forged)), run_provenance=provenance, aligned_data=data)

    def test_run_identity_and_canonical_reconstruction_are_authoritative(self):
        official = result(); context, data, _, provenance = context_and_binding()
        renamed_run = create_primary_v1_run_provenance(run_id="different-label", root_seed=provenance.root_seed, protocol=context.protocol, aligned_data=data)
        changed_root = create_primary_v1_run_provenance(run_id=provenance.run_id, root_seed=provenance.root_seed + 1, protocol=context.protocol, aligned_data=data)
        self.assertEqual(context.seed_manifest.seeds, PrimarySeedManifest.generate_primary(renamed_run, build_primary_candidates({"mGE": 8, "mDM": 8, "mCNA": 8})).seeds)
        self.assertNotEqual(context.seed_manifest.seeds, PrimarySeedManifest.generate_primary(changed_root, build_primary_candidates({"mGE": 8, "mDM": 8, "mCNA": 8})).seeds)
        for forged in (replace(official, context=replace(official.context, repeat_id=1)), replace(official, context=replace(official.context, outer_testing_ids=tuple(reversed(official.context.outer_testing_ids))))):
            with self.assertRaises(ValueError): validate_primary_search_result(forged, run_provenance=provenance, aligned_data=data)
        with self.assertRaises(ValueError): validate_primary_search_result(official, run_provenance=changed_root, aligned_data=data)


if __name__ == "__main__":
    unittest.main()
