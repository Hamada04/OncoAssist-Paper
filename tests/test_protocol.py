import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from oncoassist_research.artifacts import payload_sha256
from oncoassist_research.folds import FoldProtocol
from oncoassist_research.protocol import ModalityAdapter, PrimaryProtocolV1, PrimarySeedManifest, aligned_matrix_content_sha256, create_primary_v1_run_provenance, ordered_patient_ids_sha256, patient_set_sha256, validate_primary_v1_run_provenance
from oncoassist_research.search import build_primary_candidates


class ProtocolTests(unittest.TestCase):
    def canonical_data(self):
        ids = tuple(f"P{index:02d}" for index in range(12))
        names = {"rna": tuple(f"r{index}" for index in range(8)), "dna": tuple(f"d{index}" for index in range(8)), "cna": tuple(f"c{index}" for index in range(8))}
        data = {"sample_ids": list(ids), "y_binary": np.asarray([0, 1] * 6), "feature_columns": names, "label_mapping": {"raw_to_binary": {"1": 0, "2": 1}}, "audit_summary": {"files": {"mGE": {"sha256": "a" * 64, "provenance": "synthetic_test_fixture"}, "mDM": {"sha256": "b" * 64, "provenance": "synthetic_test_fixture"}, "CNA": {"sha256": "c" * 64, "provenance": "synthetic_test_fixture"}}}}
        for offset, (key, name) in enumerate((("X_rna", "rna"), ("X_dna", "dna"), ("X_cna", "cna"))):
            data[key] = pd.DataFrame(np.arange(96.0).reshape(12, 8) + offset, index=ids, columns=names[name])
        return data

    def copy_data(self, data):
        return {
            **data,
            "feature_columns": {name: tuple(columns) for name, columns in data["feature_columns"].items()},
            "audit_summary": {"files": {name: dict(audit) for name, audit in data["audit_summary"]["files"].items()}},
            "X_rna": data["X_rna"].copy(deep=True),
            "X_dna": data["X_dna"].copy(deep=True),
            "X_cna": data["X_cna"].copy(deep=True),
        }

    def test_frozen_protocol_and_fold_enforcement(self):
        protocol = PrimaryProtocolV1()
        self.assertEqual((protocol.outer_n_splits, protocol.outer_n_repeats, protocol.inner_n_splits), (5, 5, 3))
        self.assertEqual(protocol.modalities, ("mGE", "mDM", "mCNA")); self.assertEqual(protocol.ae_hidden_width, 128)
        self.assertEqual(protocol.ae_ratios, (.25, .5, .75)); self.assertEqual(protocol.logistic_cs, (.1, 1., 10.))
        self.assertEqual((protocol.ctgan_epochs, protocol.ctgan_pac, protocol.ctgan_verbose), (300, 10, False))
        self.assertEqual(protocol.feature_provenance_status, "UNKNOWN")
        protocol.validate_fold_protocol(FoldProtocol(5, 5, 1, 3, 2))
        for values in ((4, 5, 3), (5, 4, 3), (5, 5, 2)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError): protocol.validate_fold_protocol(FoldProtocol(values[0], values[1], 1, values[2], 2))
        self.assertEqual(protocol.identity_sha256, PrimaryProtocolV1().identity_sha256)

    def test_adapter_candidates_and_seed_manifest_are_deterministic(self):
        features = {"rna": ("r1", "r2", "r3", "r4"), "dna": ("d1", "d2", "d3", "d4"), "cna": ("c1", "c2", "c3", "c4")}
        import pandas as pd
        data = {"feature_columns": features, "X_rna": pd.DataFrame(columns=features["rna"]), "X_dna": pd.DataFrame(columns=features["dna"]), "X_cna": pd.DataFrame(columns=features["cna"])}
        adapter = ModalityAdapter.from_aligned_data(data)
        self.assertEqual(adapter.bindings["mGE"].matrix_key, "X_rna"); self.assertEqual(adapter.bindings["mCNA"].feature_key, "cna")
        candidates = build_primary_candidates({"mGE": 8, "mDM": 8, "mCNA": 8})
        self.assertEqual(len(candidates), 9); self.assertEqual(len(candidates[0].candidate_identity_sha256), 64)
        keys = ("final:ae_split", "final:ctgan", "final:lr", "final:ae_select:mGE", "final:ae_select:mDM", "final:ae_select:mCNA", "final:ae_refit:mGE", "final:ae_refit:mDM", "final:ae_refit:mCNA")
        first = PrimarySeedManifest.generate(9, {"protocol_hash": PrimaryProtocolV1().identity_sha256, "fold": [0, 0]}, keys)
        second = PrimarySeedManifest.generate(9, {"protocol_hash": PrimaryProtocolV1().identity_sha256, "fold": [0, 0]}, keys)
        self.assertEqual(first.identity_sha256, second.identity_sha256); self.assertEqual(first.final_refit_seed_book().ctgan_seed, first.require("final:ctgan"))
        with self.assertRaises(ValueError): PrimarySeedManifest(True, {}, {"x": 1})
        self.assertEqual(patient_set_sha256(["B", "A"]), patient_set_sha256(["A", "B"]))
        self.assertNotEqual(ordered_patient_ids_sha256(["B", "A"]), ordered_patient_ids_sha256(["A", "B"]))

    def test_primary_seed_manifest_binds_deterministic_fold_protocol(self):
        protocol = PrimaryProtocolV1()
        import numpy as np
        import pandas as pd
        ids = tuple(f"P{index:02d}" for index in range(12)); labels = np.asarray([0, 1] * 6)
        names = {"rna": tuple(f"r{index}" for index in range(8)), "dna": tuple(f"d{index}" for index in range(8)), "cna": tuple(f"c{index}" for index in range(8))}
        data = {"sample_ids": list(ids), "y_binary": labels, "feature_columns": names, "label_mapping": {"raw_to_binary": {"1": 0, "2": 1}}, "audit_summary": {"files": {"mGE": {"sha256": "a" * 64, "provenance": "synthetic_test_fixture"}, "mDM": {"sha256": "b" * 64, "provenance": "synthetic_test_fixture"}, "CNA": {"sha256": "c" * 64, "provenance": "synthetic_test_fixture"}}}}
        for offset, (key, name) in enumerate((("X_rna", "rna"), ("X_dna", "dna"), ("X_cna", "cna"))): data[key] = pd.DataFrame(np.arange(96.0).reshape(12, 8) + offset, index=ids, columns=names[name])
        candidates = build_primary_candidates({"mGE": 8, "mDM": 8, "mCNA": 8})
        first_run = create_primary_v1_run_provenance(run_id="run-a", root_seed=19, protocol=protocol, aligned_data=data)
        same_run = create_primary_v1_run_provenance(run_id="run-b", root_seed=19, protocol=protocol, aligned_data=data)
        changed_run = create_primary_v1_run_provenance(run_id="run-a", root_seed=20, protocol=protocol, aligned_data=data)
        first = PrimarySeedManifest.generate_primary(first_run, candidates); second = PrimarySeedManifest.generate_primary(first_run, candidates); same_seed_run = PrimarySeedManifest.generate_primary(same_run, candidates); changed = PrimarySeedManifest.generate_primary(changed_run, candidates)
        self.assertEqual(first, second); self.assertEqual(first.seeds, same_seed_run.seeds); self.assertEqual(first.identity_sha256, same_seed_run.identity_sha256)
        self.assertEqual(protocol.make_fold_protocol(first, first_run), protocol.make_fold_protocol(second, first_run)); self.assertNotEqual(first.seeds, changed.seeds)
        protocol.validate_primary_seed_manifest(first, first_run); validate_primary_v1_run_provenance(first_run, protocol=protocol, aligned_data=data)
        with self.assertRaises(ValueError): protocol.validate_primary_seed_manifest(PrimarySeedManifest(first.root_seed, first.binding, {**first.seeds, "extra": 1}), first_run)

    def test_canonical_aligned_matrix_content_is_provenance_bound(self):
        protocol, data = PrimaryProtocolV1(), self.canonical_data()
        provenance = create_primary_v1_run_provenance(run_id="run-a", root_seed=19, protocol=protocol, aligned_data=data)
        identical = create_primary_v1_run_provenance(run_id="run-a", root_seed=19, protocol=protocol, aligned_data=self.copy_data(data))
        renamed = create_primary_v1_run_provenance(run_id="run-b", root_seed=19, protocol=protocol, aligned_data=data)
        validate_primary_v1_run_provenance(provenance, protocol=protocol, aligned_data=data)
        self.assertEqual(provenance.modality_content_sha256, identical.modality_content_sha256)
        self.assertEqual(provenance.aligned_data_content_identity_sha256, identical.aligned_data_content_identity_sha256)
        self.assertEqual(provenance.modality_content_sha256, renamed.modality_content_sha256)
        self.assertEqual(provenance.aligned_data_content_identity_sha256, renamed.aligned_data_content_identity_sha256)
        self.assertNotEqual(provenance.identity_sha256, renamed.identity_sha256)
        self.assertEqual(
            provenance.modality_content_sha256["mGE"],
            aligned_matrix_content_sha256("mGE", data["X_rna"], data["sample_ids"], data["feature_columns"]["rna"]),
        )

        def reorder_rows(mutated):
            mutated["X_rna"] = mutated["X_rna"].iloc[[1, 0, *range(2, len(mutated["X_rna"]))]].copy()

        def reorder_columns(mutated):
            columns = list(mutated["feature_columns"]["dna"])
            columns[0], columns[1] = columns[1], columns[0]
            mutated["X_dna"] = mutated["X_dna"].loc[:, columns].copy()
            mutated["feature_columns"]["dna"] = tuple(columns)

        def rename_feature(mutated):
            old = mutated["feature_columns"]["cna"][0]
            new = "renamed-cna-feature"
            mutated["X_cna"] = mutated["X_cna"].rename(columns={old: new})
            mutated["feature_columns"]["cna"] = (new, *mutated["feature_columns"]["cna"][1:])

        cases = {
            "mGE_value": lambda mutated: mutated["X_rna"].iat.__setitem__((0, 0), mutated["X_rna"].iat[0, 0] + 1.0),
            "mDM_value": lambda mutated: mutated["X_dna"].iat.__setitem__((0, 0), mutated["X_dna"].iat[0, 0] + 1.0),
            "mCNA_value": lambda mutated: mutated["X_cna"].iat.__setitem__((0, 0), mutated["X_cna"].iat[0, 0] + 1.0),
            "row_order": reorder_rows,
            "column_order": reorder_columns,
            "feature_name": rename_feature,
            "same_shape_substitution": lambda mutated: mutated.__setitem__("X_rna", pd.DataFrame(np.full(mutated["X_rna"].shape, 99.0), index=mutated["X_rna"].index, columns=mutated["X_rna"].columns)),
            "old_matrix_hash_after_mutation": lambda mutated: mutated["X_rna"].iat.__setitem__((1, 1), mutated["X_rna"].iat[1, 1] + 1.0),
        }
        for name, mutate in cases.items():
            with self.subTest(mutation=name):
                mutated = self.copy_data(data)
                mutate(mutated)
                with self.assertRaises(ValueError):
                    validate_primary_v1_run_provenance(provenance, protocol=protocol, aligned_data=mutated)
        copied_modality_hash = replace(provenance, modality_content_sha256={**provenance.modality_content_sha256, "mDM": provenance.modality_content_sha256["mGE"]})
        with self.assertRaises(ValueError):
            validate_primary_v1_run_provenance(copied_modality_hash, protocol=protocol, aligned_data=data)
        stale_combined_hash = replace(provenance, aligned_data_content_identity_sha256="0" * 64)
        with self.assertRaises(ValueError):
            validate_primary_v1_run_provenance(stale_combined_hash, protocol=protocol, aligned_data=data)


if __name__ == "__main__":
    unittest.main()
