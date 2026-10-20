"""GovInfo review grounds decisions in saved package and filing evidence."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookupCaseNameAssessment,
    DocketLookupFieldAssessment,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.fields.base import NotStatedField, current_reading
from mellea_lrc.model.citations.fields.case_name import CaseName
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.providers.govinfo import GovInfoSearchPage
from mellea_lrc.validation.retrieve_govinfo_docket_roots import retrieve_govinfo_docket_roots
from mellea_lrc.validation.review_govinfo_docket_roots_llm import review_govinfo_docket_roots_llm
from mellea_lrc.validation.review_govinfo_docket_roots_llm.reviewer import GovInfoDocketReviewContext


def _input(*, results: list[dict[str, object]]) -> Document:
    source = "Acme v. Reed, Case No. 2:31-cv-45821 (D. Mass. 2031)."
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = document.complete_substage(
        "validate_roots_citation_identity.lookup_docket_roots.retrieve_courtlistener_dockets"
    ).complete_substage(
        "validate_roots_citation_identity.lookup_docket_roots.review_courtlistener_dockets_llm"
    )

    class Client:
        def search(self, _query: str, *, offset_mark: str = "*", page_size: int = 100) -> GovInfoSearchPage:
            assert offset_mark == "*" and page_size == 100
            raw = {"count": len(results), "results": results}
            return GovInfoSearchPage(
                raw_json=raw, results=tuple(results), count=len(results), next_offset_mark=None
            )

    return retrieve_govinfo_docket_roots(document, client=Client())


def _field(result: MatchResult, *, quote: str | None = None) -> DocketLookupFieldAssessment:
    return DocketLookupFieldAssessment(
        action="replace" if quote is not None else "keep",
        quote=quote,
        result=result,
        reason="Compared the filing and the saved package.",
    )


def _decision(index: int | None, *, docket_quote: str | None = None) -> DocketLookupReviewDecision:
    result = MatchResult.MATCH if index is not None else MatchResult.UNAVAILABLE
    return DocketLookupReviewDecision(
        selected_candidate_index=index,
        docket_number=_field(result, quote=docket_quote),
        case_name=DocketLookupCaseNameAssessment(
            action="keep",
            quote=None,
            normalized=CaseName.from_quote("Acme v. Reed") if index is not None else None,
            result=result,
            reason="Compared the filing's parties with the saved package title.",
        ),
        court=_field(result),
        date=_field(result),
        reason="The saved package identifies the cited case." if index is not None else "No case selected.",
    )


def test_review_records_field_decisions_and_replays_from_document() -> None:
    before = _input(
        results=[
            {
                "packageId": "USCOURTS-mad-2_31-cv-45821",
                "title": "Acme v. Reed",
                "governmentAuthor": ["United States District Court for the District of Massachusetts"],
                "dateIssued": "2035-12-20",
            }
        ]
    )
    seen: list[GovInfoDocketReviewContext] = []

    async def reviewer(context: GovInfoDocketReviewContext) -> DocketLookupReviewDecision:
        seen.append(context)
        return _decision(0)

    after = asyncio.run(review_govinfo_docket_roots_llm(before, reviewer=reviewer))
    review = after.roots[0].govinfo_docket_review
    assert review is not None and review.decision is not None
    assert review.decision.selected_candidate_index == 0
    assert review.decision.date.result is MatchResult.MATCH
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments == ()
    assert (
        after.roots[0].next_substage
        == "validate_roots_citation_identity.lookup_docket_roots.judge_identity_from_fields"
    )
    assert after.roots[0].routes[-1].node_id == review.node_id
    assert seen[0].candidates[0]["filing_year_digits"] == "31"
    assert review.node_id == after.roots[0].nodes[-1].id
    assert "dateIssued" not in str(seen[0].candidates)
    assert (
        after.get_substage("validate_roots_citation_identity.lookup_docket_roots.retrieve_govinfo_dockets")
        == before
    )
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_review_without_shortlisted_package_is_deterministic() -> None:
    before = _input(results=[])

    async def unused(_context: GovInfoDocketReviewContext) -> DocketLookupReviewDecision:
        raise AssertionError("Review must not call a model without a candidate")

    after = asyncio.run(review_govinfo_docket_roots_llm(before, reviewer=unused))
    review = after.roots[0].govinfo_docket_review
    assert review is not None and review.decision is not None
    assert review.decision.selected_candidate_index is None
    assert review.decision.docket_number.result is MatchResult.UNAVAILABLE
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments
    assert after.roots[0].routes == before.roots[0].routes


def test_ungrounded_correction_is_saved_as_failure() -> None:
    before = _input(results=[{"packageId": "USCOURTS-mad-2_31-cv-45821", "title": "Acme v. Reed"}])

    async def reviewer(_context: GovInfoDocketReviewContext) -> DocketLookupReviewDecision:
        return _decision(0, docket_quote="a number absent from the filing")

    after = asyncio.run(review_govinfo_docket_roots_llm(before, reviewer=reviewer))
    review = after.roots[0].govinfo_docket_review
    assert review is not None and review.decision is None
    assert review.failure_reason is not None and "allowed filing window" in review.failure_reason
    assert after.roots[0].locator == before.roots[0].locator
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments
    assert after.roots[0].routes == before.roots[0].routes


@pytest.mark.parametrize("field", ("case_name", "court", "date"))
def test_govinfo_review_clears_fields_without_erasing_the_previous_reading(field: str) -> None:
    before = _input(
        results=[
            {
                "packageId": "USCOURTS-mad-2_31-cv-45821",
                "title": "Acme v. Reed",
                "governmentAuthor": ["United States District Court for the District of Massachusetts"],
            }
        ]
    )
    prior = before.roots[0]
    payload = _decision(0).model_dump(mode="json")
    payload[field].update(action="not_stated", quote=None, result="unavailable")
    if field == "case_name":
        payload[field]["normalized"] = None
    decision = DocketLookupReviewDecision.model_validate(payload)

    async def reviewer(context: GovInfoDocketReviewContext) -> DocketLookupReviewDecision:
        assert context.choice_error(decision) is None
        return decision

    after = asyncio.run(review_govinfo_docket_roots_llm(before, reviewer=reviewer))
    root = after.roots[0]
    readings = getattr(root, field)
    assert readings[:-1] == getattr(prior, field)
    assert isinstance(readings[-1], NotStatedField)
    assert readings[-1].node_id == root.govinfo_docket_review.node_id
    assert current_reading(readings) is None
    assert root.govinfo_docket_review.decision is not None
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_substage(before.substage_runs[-1]) == before
    assert GovInfoDocketReviewContext.from_document(after, root).readings[field] is None
