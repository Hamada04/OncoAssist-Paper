import unittest
from dataclasses import replace
from tests import test_search
from oncoassist_research.calibration import cross_fit_sigmoid_calibration
from oncoassist_research.thresholds import select_operational_threshold,validate_operational_threshold_result,threshold_candidates
class ThresholdTests(unittest.TestCase):
 def test_threshold_content_chain(self):
  official=test_search.result(); _,data,_,provenance=test_search.context_and_binding(); calibration=cross_fit_sigmoid_calibration(official,run_provenance=provenance,aligned_data=data); result=select_operational_threshold(calibration,search_result=official,run_provenance=provenance,aligned_data=data); validate_operational_threshold_result(result,calibration=calibration,search_result=official,run_provenance=provenance,aligned_data=data)
  with self.assertRaises(ValueError): validate_operational_threshold_result(replace(result,threshold=.99),calibration=calibration,search_result=official,run_provenance=provenance,aligned_data=data)
  with self.assertRaises(ValueError): select_operational_threshold(replace(calibration,cross_fitted_calibration_sha256="x"*64),search_result=official,run_provenance=provenance,aligned_data=data)
 def test_candidates(self): self.assertEqual(threshold_candidates([.2,.2,1,0]),(0.,.2,1.))
if __name__=="__main__": unittest.main()
