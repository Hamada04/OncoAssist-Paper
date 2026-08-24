import unittest
from tests import test_search
from oncoassist_research.primary_inner import make_primary_inner_builder
from oncoassist_research.ctgan import CTGANConfig
class PrimaryInnerTests(unittest.TestCase):
 def test_factory_returns_bound_builder_and_rejects_non_v1_config(self):
  context,data,binding,provenance=test_search.context_and_binding()
  builder=make_primary_inner_builder(data,context.modality_adapter,context.outer_training_ids,context.outer_training_labels,context.protocol.make_autoencoder_training_config(),context.protocol.make_ctgan_config(),protocol=context.protocol,seed_manifest=context.seed_manifest,run_provenance=provenance,fold_protocol=context.fold_protocol,ae_validation_fraction=.2,synthetic_namespace_prefix="X",protocol_hash=context.protocol.identity_sha256,outer_fold_identity={})
  self.assertEqual(builder.seed_manifest_identity_sha256,context.seed_manifest.identity_sha256)
  with self.assertRaises(ValueError): make_primary_inner_builder(data,context.modality_adapter,context.outer_training_ids,context.outer_training_labels,context.protocol.make_autoencoder_training_config(),CTGANConfig(1,2,False),protocol=context.protocol,seed_manifest=context.seed_manifest,run_provenance=provenance,fold_protocol=context.fold_protocol,ae_validation_fraction=.2,synthetic_namespace_prefix="X",protocol_hash=context.protocol.identity_sha256,outer_fold_identity={})
if __name__=="__main__": unittest.main()
