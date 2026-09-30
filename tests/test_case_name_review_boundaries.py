"""Reviewer responses cannot treat an unquoted absence as a quoted case name."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations.body_evidence import BodyFilingFields
from mellea_lrc.model.citations.docket_lookup import DocketLookupCaseNameAssessment
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterCaseNameAssessment


@pytest.mark.parametrize(
    "assessment_type",
    (ReporterCaseNameAssessment, DocketLookupCaseNameAssessment),
)
@pytest.mark.parametrize(
    ("propose_replacement", "quote"),
    ((False, None), (True, "Acme v. Reed")),
)
def test_reviewer_assessment_rejects_not_stated_name(
    assessment_type: type[ReporterCaseNameAssessment | DocketLookupCaseNameAssessment],
    propose_replacement: bool,
    quote: str | None,
) -> None:
    with pytest.raises(ValidationError, match="must describe a quoted name"):
        assessment_type.model_validate(
            {
                "propose_replacement": propose_replacement,
                "quote": quote,
                "normalized": {"kind": "not_stated"},
                "result": MatchResult.UNAVAILABLE,
                "reason": "No case name is printed.",
            }
        )


@pytest.mark.parametrize(
    "assessment_type",
    (ReporterCaseNameAssessment, DocketLookupCaseNameAssessment),
)
def test_reviewer_assessment_keeps_null_for_no_name(
    assessment_type: type[ReporterCaseNameAssessment | DocketLookupCaseNameAssessment],
) -> None:
    response = assessment_type.model_validate(
        {
            "propose_replacement": False,
            "quote": None,
            "normalized": None,
            "result": MatchResult.UNAVAILABLE,
            "reason": "No case name is printed.",
        }
    )
    assert response.normalized is None


def test_body_filing_rejects_not_stated_name_with_quote() -> None:
    with pytest.raises(ValidationError, match="must describe a quoted name"):
        BodyFilingFields.model_validate(
            {
                "locator": "550 U.S. 544",
                "case_name": "Acme v. Reed",
                "normalized_case_name": {"kind": "not_stated"},
                "court": None,
                "date": None,
            }
        )


def test_body_filing_keeps_null_for_no_name() -> None:
    response = BodyFilingFields.model_validate(
        {
            "locator": "550 U.S. 544",
            "case_name": None,
            "normalized_case_name": None,
            "court": None,
            "date": None,
        }
    )
    assert response.normalized_case_name is None
