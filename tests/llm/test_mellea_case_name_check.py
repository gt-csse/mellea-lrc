"""Live evaluations for Mellea semantic case-name comparison."""

from __future__ import annotations

import asyncio

import pytest
from dotenv import load_dotenv

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import ExtractedCitation
from mellea_lrc.validation import (
    CitationValidation,
    ExactCaseNameCheckNode,
    FieldCheckOutcome,
    MelleaCaseNameCheckOutcome,
    ValidationNodeStatus,
)
from mellea_lrc.validation.field_checks import run_mellea_case_name_check

load_dotenv(".env")


@pytest.mark.llm_evaluation
@pytest.mark.parametrize(
    ("extracted", "retrieved", "expected_outcome"),
    [
        (
            "Gambale v. Deutsche Bank National Trust Company",
            "Gambale v. Deutsche Bank Natl. Trust Co.",
            MelleaCaseNameCheckOutcome.MATCH,
        ),
        (
            "Methodist Hospital of Sacramento v. Shalala",
            "Methodist Hosp. of Sacramento v. Shalala",
            MelleaCaseNameCheckOutcome.MATCH,
        ),
        (
            "Brown v. Board of Education",
            "Plessy v. Ferguson",
            MelleaCaseNameCheckOutcome.MISMATCH,
        ),
    ],
)
def test_mellea_case_name_check(
    extracted: str,
    retrieved: str,
    expected_outcome: MelleaCaseNameCheckOutcome,
) -> None:
    """Classify normal legal abbreviation as match and distinct cases as mismatch."""
    citation = ExtractedCitation(
        citation_id="live-semantic-case-name",
        span=Span(0, 1),
        locator_span=Span(0, 1),
        matched_text="x",
        citation=FullCaseCitation(),
    )
    exact_node = ExactCaseNameCheckNode(
        node_id="live-semantic-case-name:exact_case_name_check",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=FieldCheckOutcome.MISMATCH,
        extracted_case_name=extracted,
        retrieved_case_name=retrieved,
        depends_on=(),
    )

    node = asyncio.run(
        run_mellea_case_name_check(
            CitationValidation(citation=citation).append(exact_node),
            case_name_evidence=exact_node,
        )
    )

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.outcome is expected_outcome
