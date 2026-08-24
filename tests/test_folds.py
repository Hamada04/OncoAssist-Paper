import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from oncoassist_research import folds


class FoldManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.sample_ids = [f"P-{index:03d}" for index in range(30)]
        self.labels = np.asarray([0, 1] * 15, dtype=int)
        self.protocol = folds.FoldProtocol(5, 3, 42, 3, 4200)
        self.fingerprint = {
            "csv_sha256": {"mGE": "a" * 64, "mDM": "b" * 64, "CNA": "c" * 64},
            "sample_count": len(self.sample_ids),
            "label_mapping": {"raw_to_binary": {"1": 0, "2": 1}},
            "ordered_sample_ids_canonical_json_sha256": folds._sample_id_list_sha256(self.sample_ids),
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def outer_manifest(self, protocol: folds.FoldProtocol | None = None) -> dict:
        return folds.build_outer_fold_manifest(
            self.sample_ids, self.labels, self.fingerprint, protocol or self.protocol
        )

    def outer_sha(self, manifest: dict) -> str:
        return hashlib.sha256(folds._canonical_json_bytes(manifest)).hexdigest()

    def inner_manifest(self, outer: dict | None = None) -> tuple[dict, dict, str]:
        outer = outer or self.outer_manifest()
        outer_sha = self.outer_sha(outer)
        return (
            folds.build_inner_fold_manifest(
                outer, outer_sha, self.sample_ids, self.labels, self.fingerprint, self.protocol
            ),
            outer,
            outer_sha,
        )

    def test_fold_protocol_accepts_valid_values(self) -> None:
        protocol = folds.FoldProtocol(5, 3, 42, 3, 4200)
        self.assertEqual(protocol.as_dict()["outer_n_splits"], 5)
        self.assertTrue(protocol.__dataclass_params__.frozen)

    def test_fold_protocol_rejects_invalid_values(self) -> None:
        invalid_values = [
            (1, 3, 42, 3, 4200),
            (5, 0, 42, 3, 4200),
            (5, 3, 42, 1, 4200),
            (5, 3, "42", 3, 4200),
            (True, 3, 42, 3, 4200),
            (5, 3, False, 3, 4200),
        ]
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    folds.FoldProtocol(*values)

    def test_historical_protocol_has_15_records_and_sklearn_parity(self) -> None:
        manifest = self.outer_manifest()
        self.assertEqual(len(manifest["folds"]), 15)
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        expected = []
        for train_indices, test_indices in splitter.split(np.zeros(len(self.labels)), self.labels):
            expected.append((
                sorted(self.sample_ids[index] for index in train_indices),
                sorted(self.sample_ids[index] for index in test_indices),
            ))
        actual = [(record["train_sample_ids"], record["test_sample_ids"]) for record in manifest["folds"]]
        self.assertEqual(actual, expected)

    def test_five_by_five_protocol_has_25_records(self) -> None:
        protocol = folds.FoldProtocol(5, 5, 42, 3, 4200)
        self.assertEqual(len(self.outer_manifest(protocol)["folds"]), 25)

    def test_outer_folds_partition_and_cover_each_repeat(self) -> None:
        manifest = self.outer_manifest()
        for repeat_id in range(self.protocol.outer_n_repeats):
            repeated_tests = []
            for record in [r for r in manifest["folds"] if r["repeat_id"] == repeat_id]:
                train_ids, test_ids = set(record["train_sample_ids"]), set(record["test_sample_ids"])
                self.assertFalse(train_ids.intersection(test_ids))
                self.assertEqual(train_ids.union(test_ids), set(self.sample_ids))
                self.assertEqual(set(record["train_partition"]["class_counts"]), {"0", "1"})
                self.assertTrue(all(record["train_partition"]["class_counts"].values()))
                self.assertTrue(all(record["test_partition"]["class_counts"].values()))
                repeated_tests.extend(record["test_sample_ids"])
            self.assertEqual(sorted(repeated_tests), self.sample_ids)

    def test_outer_manifest_is_deterministic_and_random_state_matters(self) -> None:
        first, second = self.outer_manifest(), self.outer_manifest()
        self.assertEqual(first, second)
        self.assertEqual(folds.manifest_identity_sha256(first), folds.manifest_identity_sha256(second))
        changed = self.outer_manifest(folds.FoldProtocol(5, 3, 43, 3, 4200))
        self.assertNotEqual(first["folds"], changed["folds"])
        self.assertNotEqual(folds.manifest_identity_sha256(first), folds.manifest_identity_sha256(changed))

    def test_outer_validation_rejects_mutations_and_protocol_change(self) -> None:
        manifest = self.outer_manifest()
        mutations = []
        wrong_fingerprint = copy.deepcopy(manifest)
        mutations.append((wrong_fingerprint, {**self.fingerprint, "sample_count": 999}, self.protocol))
        duplicate = copy.deepcopy(manifest)
        duplicate["folds"][1]["repeat_id"] = duplicate["folds"][0]["repeat_id"]
        duplicate["folds"][1]["fold_id"] = duplicate["folds"][0]["fold_id"]
        mutations.append((duplicate, self.fingerprint, self.protocol))
        missing = copy.deepcopy(manifest)
        missing["folds"].pop()
        mutations.append((missing, self.fingerprint, self.protocol))
        overlap = copy.deepcopy(manifest)
        overlap["folds"][0]["train_sample_ids"].append(overlap["folds"][0]["test_sample_ids"][0])
        overlap["folds"][0]["train_sample_ids"].sort()
        mutations.append((overlap, self.fingerprint, self.protocol))
        absent = copy.deepcopy(manifest)
        absent["folds"][0]["test_sample_ids"].append("P-999")
        absent["folds"][0]["test_sample_ids"].sort()
        mutations.append((absent, self.fingerprint, self.protocol))
        for mutated, fingerprint, protocol in mutations:
            with self.subTest(mutated=mutated):
                with self.assertRaises(ValueError):
                    folds.validate_outer_fold_manifest(mutated, self.sample_ids, self.labels, fingerprint, protocol)
        with self.assertRaises(ValueError):
            folds.validate_outer_fold_manifest(
                manifest, self.sample_ids, self.labels, self.fingerprint,
                folds.FoldProtocol(5, 5, 42, 3, 4200),
            )

    def test_outer_manifest_write_load_and_deterministic_sha(self) -> None:
        manifest = self.outer_manifest()
        first = folds.write_outer_fold_manifest(manifest, self.root / "outer-one.json")
        second = folds.write_outer_fold_manifest(manifest, self.root / "outer-two.json")
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(folds.load_outer_fold_manifest(self.root / "outer-one.json"), manifest)

    def test_inner_seed_is_deterministic_and_varies_by_outer_identity(self) -> None:
        self.assertEqual(folds.derive_inner_fold_seed(0, 0, self.protocol), 4200)
        self.assertEqual(folds.derive_inner_fold_seed(0, 0, self.protocol), 4200)
        self.assertNotEqual(
            folds.derive_inner_fold_seed(0, 0, self.protocol),
            folds.derive_inner_fold_seed(1, 0, self.protocol),
        )

    def test_inner_folds_are_valid_and_match_sklearn_parity(self) -> None:
        inner, outer, outer_sha = self.inner_manifest()
        result = folds.validate_inner_fold_manifest(
            inner, outer, outer_sha, self.sample_ids, self.labels, self.fingerprint, self.protocol
        )
        self.assertEqual(result["inner_record_count"], 45)
        labels_by_id = dict(zip(self.sample_ids, self.labels.tolist()))
        for outer_record in inner["outer_folds"]:
            outer_source = next(
                source for source in outer["folds"]
                if (source["repeat_id"], source["fold_id"])
                == (outer_record["repeat_id"], outer_record["fold_id"])
            )
            outer_train, outer_test = set(outer_source["train_sample_ids"]), set(outer_source["test_sample_ids"])
            validation_ids = []
            for record in outer_record["inner_folds"]:
                train_ids, valid_ids = set(record["inner_train_sample_ids"]), set(record["inner_validation_sample_ids"])
                self.assertFalse(train_ids.intersection(valid_ids))
                self.assertEqual(train_ids.union(valid_ids), outer_train)
                self.assertFalse(train_ids.intersection(outer_test))
                self.assertFalse(valid_ids.intersection(outer_test))
                self.assertTrue(all(record["inner_train_partition"]["class_counts"].values()))
                self.assertTrue(all(record["inner_validation_partition"]["class_counts"].values()))
                validation_ids.extend(record["inner_validation_sample_ids"])
            self.assertEqual(sorted(validation_ids), sorted(outer_train))
        first_outer = outer["folds"][0]
        first_inner = inner["outer_folds"][0]["inner_folds"]
        outer_train_ids = first_outer["train_sample_ids"]
        outer_labels = np.asarray([labels_by_id[sample_id] for sample_id in outer_train_ids])
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=4200)
        expected = [
            (sorted(outer_train_ids[index] for index in train), sorted(outer_train_ids[index] for index in valid))
            for train, valid in splitter.split(np.zeros(len(outer_labels)), outer_labels)
        ]
        actual = [(record["inner_train_sample_ids"], record["inner_validation_sample_ids"]) for record in first_inner]
        self.assertEqual(actual, expected)

    def test_inner_manifest_is_deterministic(self) -> None:
        first, _, _ = self.inner_manifest()
        second, _, _ = self.inner_manifest()
        self.assertEqual(first, second)

    def test_inner_validation_rejects_required_mutations(self) -> None:
        manifest, outer, outer_sha = self.inner_manifest()
        with self.assertRaises(ValueError):
            folds.validate_inner_fold_manifest(manifest, outer, "wrong", self.sample_ids, self.labels, self.fingerprint, self.protocol)
        mutations = []
        leakage = copy.deepcopy(manifest)
        leakage["outer_folds"][0]["inner_folds"][0]["inner_train_sample_ids"].append(outer["folds"][0]["test_sample_ids"][0])
        leakage["outer_folds"][0]["inner_folds"][0]["inner_train_sample_ids"].sort()
        mutations.append(leakage)
        duplicate = copy.deepcopy(manifest)
        duplicate["outer_folds"][0]["inner_folds"][1]["inner_fold_id"] = 0
        mutations.append(duplicate)
        missing = copy.deepcopy(manifest)
        missing["outer_folds"][0]["inner_folds"].pop()
        mutations.append(missing)
        seed = copy.deepcopy(manifest)
        seed["outer_folds"][0]["inner_folds"][0]["inner_seed"] += 1
        mutations.append(seed)
        hash_mismatch = copy.deepcopy(manifest)
        hash_mismatch["outer_folds"][0]["inner_folds"][0]["inner_train_sample_ids_canonical_json_sha256"] = "0" * 64
        mutations.append(hash_mismatch)
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                with self.assertRaises(ValueError):
                    folds.validate_inner_fold_manifest(mutated, outer, outer_sha, self.sample_ids, self.labels, self.fingerprint, self.protocol)
        with self.assertRaises(ValueError):
            folds.validate_inner_fold_manifest(
                manifest, outer, outer_sha, self.sample_ids, self.labels, self.fingerprint,
                folds.FoldProtocol(5, 5, 42, 3, 4200),
            )

    def test_inner_manifest_write_load_round_trip(self) -> None:
        manifest, _, _ = self.inner_manifest()
        report = folds.write_inner_fold_manifest(manifest, self.root / "inner.json")
        self.assertEqual(report["manifest_sha256"], hashlib.sha256((self.root / "inner.json").read_bytes()).hexdigest())
        self.assertEqual(folds.load_inner_fold_manifest(self.root / "inner.json"), manifest)

    def test_unsorted_input_sample_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lexicographically ordered"):
            folds.build_outer_fold_manifest(
                list(reversed(self.sample_ids)), self.labels, self.fingerprint, self.protocol
            )

    def test_data_fingerprint_uses_step_one_audit_metadata(self) -> None:
        data = {
            "sample_ids": self.sample_ids,
            "label_mapping": {"raw_to_binary": {"1": 0, "2": 1}},
            "audit_summary": {
                "files": {
                    "mGE": {"sha256": "a" * 64},
                    "mDM": {"sha256": "b" * 64},
                    "CNA": {"sha256": "c" * 64},
                }
            },
        }
        self.assertEqual(folds.build_outer_data_fingerprint(data), self.fingerprint)


if __name__ == "__main__":
    unittest.main()
