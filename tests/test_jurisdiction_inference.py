import pytest
from mellea_lrc.jurisdiction_inference import evaluate_court_inference
from mellea_lrc.jurisdiction_inference.types import CourtInferenceStatus

def test_court_inference_infers_resolved():
    lead = evaluate_court_inference("scotus")
    assert lead.status == CourtInferenceStatus.RESOLVED
    assert lead.extracted_court == "scotus"
    assert lead.courts_db_classification is not None
    assert lead.courts_db_classification.system == "federal"
    assert lead.courts_db_classification.type == "appellate"

def test_court_inference_missing():
    lead = evaluate_court_inference(None)
    assert lead.status == CourtInferenceStatus.MISSING_COURT
    assert lead.extracted_court is None
    assert lead.courts_db_classification is None

def test_court_inference_unrecognized():
    lead = evaluate_court_inference("fake_court")
    assert lead.status == CourtInferenceStatus.UNRECOGNIZED
    assert lead.extracted_court == "fake_court"
    assert lead.courts_db_classification is None
