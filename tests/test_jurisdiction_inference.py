import pytest
from mellea_lrc.jurisdiction_inference import evaluate_court_lead
from mellea_lrc.jurisdiction_inference.types import CourtLeadStatus

def test_court_lead_infers_resolved():
    lead = evaluate_court_lead("scotus")
    assert lead.status == CourtLeadStatus.RESOLVED
    assert lead.extracted_court == "scotus"
    assert lead.cl_court_taxonomy is not None
    assert lead.cl_court_taxonomy.system == "federal"
    assert lead.cl_court_taxonomy.type == "appellate"

def test_court_lead_missing():
    lead = evaluate_court_lead(None)
    assert lead.status == CourtLeadStatus.MISSING_COURT
    assert lead.extracted_court is None
    assert lead.cl_court_taxonomy is None

def test_court_lead_unrecognized():
    lead = evaluate_court_lead("fake_court")
    assert lead.status == CourtLeadStatus.UNRECOGNIZED
    assert lead.extracted_court == "fake_court"
    assert lead.cl_court_taxonomy is None
