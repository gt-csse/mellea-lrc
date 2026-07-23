"""Live evaluations for Mellea case-name search-term preparation."""

from __future__ import annotations

import asyncio
from collections.abc import Collection

import pytest
from dotenv import load_dotenv

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import ExtractedCitation
from mellea_lrc.validation import (
    CitationValidation,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    ValidationNodeStatus,
)
from mellea_lrc.validation.case_search import run_mellea_case_name_query_preparation

load_dotenv(".env")


@pytest.mark.llm_evaluation
@pytest.mark.parametrize(
    ("plaintiff", "defendant", "accepted_terms"),
    [
        (
            "County of Los Angeles",
            "Los Angeles County",
            {
                ("Los Angeles", "Los Angeles"),
                ("Los Angeles County", "Los Angeles County"),
            },
        ),
        (
            "Federal Deposit Insurance Corporation",
            "Brennan",
            {("Federal Deposit Insurance Corporation", "Brennan")},
        ),
    ],
)
def test_mellea_case_name_query_preparation(
    plaintiff: str,
    defendant: str,
    accepted_terms: Collection[tuple[str, str]],
) -> None:
    """Prepare concise, faithful search terms from two case parties."""
    citation = ExtractedCitation(
        citation_id="live-case-name-query",
        span=Span(0, 1),
        locator_span=Span(0, 1),
        matched_text="x",
        citation=FullCaseCitation(court="scotus"),
    )
    reextraction = MelleaCaseNameReextractionNode(
        node_id="live-case-name-query:mellea_case_name_reextraction",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaCaseNameReextractionOutcome.COMPLETE,
        plaintiff=plaintiff,
        defendant=defendant,
        depends_on=(),
    )
    validation = CitationValidation(citation=citation).append(reextraction)

    node = asyncio.run(run_mellea_case_name_query_preparation(validation, reextraction=reextraction))

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert (node.query_plaintiff, node.query_defendant) in accepted_terms
