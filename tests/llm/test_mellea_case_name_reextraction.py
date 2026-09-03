"""Live evaluations for locally grounded Mellea case-name re-extraction."""

from __future__ import annotations

import asyncio

import pytest
from dotenv import load_dotenv

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerOpinionCluster
from mellea_lrc.extraction import ExtractedCitation
from mellea_lrc.validation import (
    CitationValidation,
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    MelleaCaseNameCheckNode,
    MelleaCaseNameCheckOutcome,
    ValidationNodeStatus,
)
from mellea_lrc.validation.field_checks.mellea_case_name_reextraction import (
    run_mellea_case_name_reextraction,
)

load_dotenv(".env")


@pytest.mark.llm_evaluation
@pytest.mark.parametrize(
    ("text", "locator", "expected_plaintiff", "expected_defendant"),
    [
        (
            "The court considered Gambale v. Deutsche Bank Natl. Trust Co., "
            "377 F. App'x 247, 249 (2d Cir. 2010), before deciding the issue.",
            "377 F. App'x 247",
            "Gambale",
            "Deutsche Bank Natl. Trust Co.",
        ),
        (
            "As explained in Rivero v. Bd. of Regents of Univ. of N.M., "
            "950 F.3d 754, 758 (10th Cir. 2020), the standard applies.",
            "950 F.3d 754",
            "Rivero",
            "Bd. of Regents of Univ. of N.M.",
        ),
        (
            "See Deutsche Bank Natl. Trust Co. v. White, 110 A.D.3d 759, "
            "760 (2d Dep't 2013), for the controlling analysis.",
            "110 A.D.3d 759",
            "Deutsche Bank Natl. Trust Co.",
            "White",
        ),
    ],
)
def test_mellea_case_name_reextraction(
    text: str,
    locator: str,
    expected_plaintiff: str,
    expected_defendant: str,
) -> None:
    """Extract parties verbatim from the citation-local document text."""
    start = text.index(locator)
    citation = ExtractedCitation(
        citation_id="live-case-name",
        full_span=Span(0, len(text)),
        locator_span=Span(start, start + len(locator)),
        matched_text=locator,
        citation=FullCaseCitation(),
    )
    lookup = ExactLocatorLookupNode(
        node_id="live-case-name:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.FOUND,
        locator=locator,
        cluster=CourtListenerOpinionCluster(case_name="not used by re-extraction"),
        candidate_count=1,
    )
    semantic_case_name_check = MelleaCaseNameCheckNode(
        node_id="live-case-name:mellea_case_name_check",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaCaseNameCheckOutcome.MISMATCH,
        extracted_case_name="not used",
        retrieved_case_name="not used",
        depends_on=(lookup.node_id,),
    )

    node = asyncio.run(
        run_mellea_case_name_reextraction(
            CitationValidation(citation=citation).append(lookup).append(semantic_case_name_check),
            trigger=semantic_case_name_check,
            locator_lookup=lookup,
            document_text=text,
        )
    )

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.plaintiff == expected_plaintiff
    assert node.defendant == expected_defendant
