import inspect
import unittest
from dataclasses import replace
from tests import test_search
from oncoassist_research.calibration import cross_fit_sigmoid_calibration,validate_cross_fitted_calibration_result
from oncoassist_research.thresholds import select_operational_threshold

class CalibrationTests(unittest.TestCase):
 def test_official_result_authority_and_content_validation(self):
  official=test_search.result(); _,data,_,provenance=test_search.context_and_binding(); calibration=cross_fit_sigmoid_calibration(official,run_provenance=provenance,aligned_data=data); validate_cross_fitted_calibration_result(calibration,search_result=official,run_provenance=provenance,aligned_data=data)
  self.assertIn("run_provenance",inspect.signature(cross_fit_sigmoid_calibration).parameters)
  predictions=list(calibration.predictions); predictions[0]=replace(predictions[0],cross_fitted_probability=min(.99,predictions[0].cross_fitted_probability+.1))
  with self.assertRaises(ValueError): validate_cross_fitted_calibration_result(replace(calibration,predictions=tuple(predictions)),search_result=official,run_provenance=provenance,aligned_data=data)
  evidence=dict(calibration.fold_evidence); evidence[0]=dict(evidence[0]); evidence[0]["heldout_inner_fold_id"]=1
  with self.assertRaises(ValueError): validate_cross_fitted_calibration_result(replace(calibration,fold_evidence=evidence),search_result=official,run_provenance=provenance,aligned_data=data)
  with self.assertRaises(ValueError): select_operational_threshold(replace(calibration,predictions=tuple(predictions)),search_result=official,run_provenance=provenance,aligned_data=data)
 def test_wrong_fold_and_label_rejected_before_calibration(self):
  official=test_search.result(); _,data,_,provenance=test_search.context_and_binding(); records=list(official.selected_search.selected_oof_predictions)
  records[0]=replace(records[0],inner_fold_id=(records[0].inner_fold_id+1)%3)
  altered=replace(official.selected_search,selected_oof_predictions=tuple(records))
  with self.assertRaises(ValueError): cross_fit_sigmoid_calibration(replace(official,selected_search=altered),run_provenance=provenance,aligned_data=data)
  records=list(official.selected_search.selected_oof_predictions); records[0]=replace(records[0],true_label=1-records[0].true_label)
  with self.assertRaises(ValueError): cross_fit_sigmoid_calibration(replace(official,selected_search=replace(official.selected_search,selected_oof_predictions=tuple(records))),run_provenance=provenance,aligned_data=data)
 def test_forged_wrong_fold_calibration_cannot_bypass_official_result(self):
  official=test_search.result(); _,data,_,provenance=test_search.context_and_binding(); records=list(official.selected_search.selected_oof_predictions); records[0]=replace(records[0],inner_fold_id=(records[0].inner_fold_id+1)%3)
  forged=replace(official,selected_search=replace(official.selected_search,selected_oof_predictions=tuple(records)))
  with self.assertRaises(ValueError): cross_fit_sigmoid_calibration(forged,run_provenance=provenance,aligned_data=data)
if __name__=="__main__": unittest.main()
