"""The docket review stage selects from saved search evidence once per root."""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest

from mellea_lrc.courtlistener.models import CourtListenerSearchResult
from mellea_lrc.model import Document, FullDocketCitation, Span
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupCandidate,
    DocketLookupFailure,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.validation.docket_root_lookup_review import (
    STAGE,
    DocketLookupReviewContext,
    DocketLookupReviewOutcome,
    IvrDocketLookupReviewer,
    docket_root_lookup_review,
)

SOURCE = "Smith v. Jones, No. 05-4206 (2d Cir. 2007)."


def _decision(
    selected: int | None,
    *,
    docket_number: str = "match",
    case_name: str = "match",
    court: str = "match",
    date: str = "match",
) -> DocketLookupReviewDecision:
    if selected is None:
        docket_number = case_name = court = date = "undetermined"
    return DocketLookupReviewDecision.model_validate(
        {
            "selected_candidate_index": selected,
            "docket_number": {"result": docket_number, "reason": "Docket numbers were compared."},
            "case_name": {"result": case_name, "reason": "Parties were compared."},
            "court": {"result": court, "reason": "Tribunals were compared."},
            "date": {"result": date, "reason": "Opinion dates were compared."},
            "reason": "The supplied records and citation context were considered together.",
        }
    )


def _document(
    *, shortlist: tuple[int, ...] = (0, 1), partial: bool = False, verbose: bool = False
) -> Document:
    start = SOURCE.index("No. ")
    number_start = SOURCE.index("05-4206")
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        stage="test_sites",
        source=SOURCE,
        span=Span(start, number_start + len("05-4206")),
        number_span=Span(number_start, number_start + len("05-4206")),
    )
    document = Document.from_source(SOURCE).add_citation(root).complete("test_sites")
    document = document.replace_citation(root.record("10_roots").with_root(root.id)).complete("10_roots")
    root = document.roots[0]
    assert isinstance(root, FullDocketCitation)
    lookup_node = root.record("docket_root_lookup")
    attempts = (
        DocketLookupAttempt(
            source_type="d",
            query="docketNumber:(05-4206)",
            pages=(
                {
                    "count": 1,
                    "next": "https://courtlistener.example/search/?cursor=more" if partial else None,
                    "previous": None,
                    "results": [
                        {
                            "docket_id": 41,
                            "docketNumber": "05-4206",
                            "caseName": "Smith v. Jones",
                            "court_id": "ca2",
                            "dateFiled": "2005-01-01",
                            "unmodeled": {"saved": True},
                            "parties": (
                                [{"name": f"Party {index} " + "x" * 500} for index in range(12)]
                                if verbose
                                else []
                            ),
                            "snippet": "y" * 10000 if verbose else None,
                        }
                    ],
                },
            ),
        ),
        DocketLookupAttempt(
            source_type="o",
            query="docketNumber:(05-4206)",
            failure=(
                DocketLookupFailure(
                    failure_type="http_error", message="Rate limited", upstream_status_code=429
                )
                if partial
                else None
            ),
            pages=(
                {
                    "count": 1,
                    "next": None,
                    "previous": None,
                    "results": [
                        {
                            "cluster_id": 51,
                            "docket_id": 41,
                            "docketNumber": "05-4206",
                            "caseNameFull": "Smith v. Jones",
                            "court_id": "ca2",
                            "dateFiled": "2007-09-03",
                        }
                    ],
                },
            ),
        ),
    )
    candidates = (
        DocketLookupCandidate(
            source_type="d",
            record_id="41",
            attempt_index=0,
            page_index=0,
            result_index=0,
            docket_number="05-4206",
            docket_similarity=100,
        ),
        DocketLookupCandidate(
            source_type="o",
            record_id="51",
            attempt_index=1,
            page_index=0,
            result_index=0,
            docket_number="05-4206",
            docket_similarity=100,
        ),
    )
    lookup = DocketLookup(
        node_id=lookup_node.nodes[-1].id,
        attempts=attempts,
        candidates=candidates,
        shortlisted_candidate_indices=shortlist,
        failure=(
            DocketLookupFailure(failure_type="partial_search", message="Search stopped before all pages")
            if partial
            else None
        ),
    )
    return document.replace_citation(lookup_node.with_docket_lookup(lookup)).complete("docket_root_lookup")


class FakeReviewer:
    def __init__(self, decision: DocketLookupReviewDecision | DocketLookupReviewOutcome) -> None:
        self.decision = decision
        self.contexts: list[DocketLookupReviewContext] = []

    async def __call__(
        self, context: DocketLookupReviewContext
    ) -> DocketLookupReviewDecision | DocketLookupReviewOutcome:
        self.contexts.append(context)
        return self.decision


def _run(*, success: bool = False) -> IvrRun:
    return IvrRun.model_validate(
        {
            "success": success,
            "selected_attempt": 0,
            "attempts": [
                {
                    "output": "{}",
                    "requirements": [
                        {
                            "description": "Return the required schema",
                            "passed": success,
                            "reason": None if success else "Incomplete review JSON",
                            "score": None,
                        }
                    ],
                    "request": [{"role": "user", "content": "review this docket"}],
                    "response": {"finish_reason": "stop"},
                }
            ],
            "backend": "FakeBackend",
            "model": "test-model",
            "model_options": {},
            "instruction": "Review one docket",
            "prefix": "CourtListener records",
            "grounding_context": {},
            "user_variables": {"candidates": "saved"},
            "output_schema": None,
        }
    )


def test_review_selects_opinion_from_mixed_shortlist_and_saves_independent_fields() -> None:
    before = _document()
    reviewer = FakeReviewer(_decision(1, docket_number="mismatch", case_name="mismatch", court="mismatch"))

    after = asyncio.run(docket_root_lookup_review(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    context = reviewer.contexts[0]
    assert context.shortlisted_candidate_indices == (0, 1)
    assert [item["candidate_index"] for item in context.candidates] == [0, 1]
    assert [item["source_type"] for item in context.candidates] == ["d", "o"]
    assert "unmodeled" not in context.candidates[0]["record_summary"]
    assert context.candidates[0]["record_summary"]["dateFiled_meaning"] == "case_docket_initiation"
    assert context.candidates[1]["record_summary"]["dateFiled_meaning"] == "opinion_record_filing"
    assert context.candidates[0]["court_name_context"] == {
        "raw_court_id": "ca2",
        "full_name": "Court of Appeals for the Second Circuit",
    }
    root = after.roots[0]
    assert root.docket_lookup_review.decision.selected_candidate_index == 1
    assert root.docket_lookup_review.decision.docket_number.result is MatchResult.MISMATCH
    assert root.docket_lookup_review.decision.case_name.result is MatchResult.MISMATCH
    assert root.docket_lookup_review.decision.court.result is MatchResult.MISMATCH
    assert root.docket_lookup_review.decision.date.result is MatchResult.MATCH
    assert root.identity_judgments == before.roots[0].identity_judgments
    assert root.docket_lookup == before.roots[0].docket_lookup
    assert root.docket_lookup.attempts[0].pages[0]["results"][0]["unmodeled"] == {"saved": True}
    assert root.nodes[-1].stage == STAGE
    assert after.get_stage("docket_root_lookup") == before
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_supported_search_aliases_reach_the_review_context() -> None:
    raw = {
        "docket_number": "05-4206",
        "case_name": "Smith v. Jones",
        "case_name_full": "Smith v. Jones",
        "courtId": "ca2",
        "date_filed": "2007-09-03",
    }
    parsed = CourtListenerSearchResult.model_validate(raw)
    module = importlib.import_module("mellea_lrc.validation.docket_root_lookup_review")

    summary = module._record_summary(raw, parsed, "o")

    assert summary["docketNumber"] == "05-4206"
    assert summary["caseName"] == summary["caseNameFull"] == "Smith v. Jones"
    assert summary["court_id"] == "ca2"
    assert summary["dateFiled"] == "2007-09-03"
    assert DocketLookupReviewContext._court_name_context(parsed)["full_name"] == (
        "Court of Appeals for the Second Circuit"
    )


def test_single_candidate_still_gets_one_model_call() -> None:
    before = _document(shortlist=(0,))
    reviewer = FakeReviewer(_decision(0, date="undetermined"))

    after = asyncio.run(docket_root_lookup_review(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    assert [item["candidate_index"] for item in reviewer.contexts[0].candidates] == [0]
    assert after.roots[0].docket_lookup_review.decision.selected_candidate_index == 0


def test_partial_search_is_visible_to_review_and_preserved_in_lookup() -> None:
    before = _document(partial=True)
    reviewer = FakeReviewer(_decision(None))

    after = asyncio.run(docket_root_lookup_review(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    status = reviewer.contexts[0].search_status
    assert status["incomplete"] is True
    assert status["attempts"][0]["next_page_available"] is True
    assert status["attempts"][1]["failure"]["upstream_status_code"] == 429
    assert status["lookup_failure"]["failure_type"] == "partial_search"
    assert after.roots[0].docket_lookup == before.roots[0].docket_lookup
    assert after.roots[0].docket_lookup_review.decision.selected_candidate_index is None


def test_model_context_truncates_large_search_hit_without_losing_saved_raw() -> None:
    before = _document(shortlist=(0,), verbose=True)
    root = before.roots[0]
    assert isinstance(root, FullDocketCitation)

    context = DocketLookupReviewContext.from_document(before, root)

    assert len(json.dumps(context.candidates)) < 2000
    summary = context.candidates[0]["record_summary"]
    assert len(summary["party_names"]) == 4
    assert len(summary["snippets"][0]) <= 241
    assert len(root.docket_lookup.attempts[0].pages[0]["results"][0]["snippet"]) == 10000


def test_empty_shortlist_persists_explicit_no_selection_without_model_call() -> None:
    before = _document(shortlist=())
    reviewer = FakeReviewer(_decision(0))

    after = asyncio.run(docket_root_lookup_review(before, reviewer=reviewer))

    assert reviewer.contexts == []
    review = after.roots[0].docket_lookup_review
    assert review.decision.selected_candidate_index is None
    assert review.ivr is None
    assert review.failure_reason is None
    assert all(
        getattr(review.decision, field).result is MatchResult.UNDETERMINED
        for field in ("docket_number", "case_name", "court", "date")
    )
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_empty_partial_search_records_its_limitation_without_model_call() -> None:
    before = _document(shortlist=(), partial=True)
    reviewer = FakeReviewer(_decision(0))

    after = asyncio.run(docket_root_lookup_review(before, reviewer=reviewer))

    assert reviewer.contexts == []
    review = after.roots[0].docket_lookup_review
    assert review.decision.selected_candidate_index is None
    assert "stopped before all results" in review.decision.reason


@pytest.mark.parametrize("decision", [_decision(99), _decision(0, date="mismatch")])
def test_invalid_choice_or_docket_filing_date_judgment_becomes_review_failure(
    decision: DocketLookupReviewDecision,
) -> None:
    before = _document()
    run = _run(success=True)
    reviewer = FakeReviewer(DocketLookupReviewOutcome(decision=decision, run=run))

    after = asyncio.run(docket_root_lookup_review(before, reviewer=reviewer))

    review = after.roots[0].docket_lookup_review
    assert review.decision is None
    assert review.failure_reason
    assert review.ivr == run
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_failed_ivr_keeps_complete_run_and_reason() -> None:
    before = _document(shortlist=(1,))
    run = _run()
    reviewer = FakeReviewer(
        DocketLookupReviewOutcome(decision=None, run=run, failure_reason="Incomplete review JSON")
    )

    after = asyncio.run(docket_root_lookup_review(before, reviewer=reviewer))

    review = after.roots[0].docket_lookup_review
    assert review.decision is None
    assert review.failure_reason == "Incomplete review JSON"
    assert review.ivr == run
    assert review.ivr.attempts[0].request == [{"role": "user", "content": "review this docket"}]
    assert review.ivr.attempts[0].response == {"finish_reason": "stop"}
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_stage_order_and_replay_checks() -> None:
    with pytest.raises(ValueError, match="Complete docket root lookup"):
        asyncio.run(docket_root_lookup_review(Document.from_source(SOURCE)))
    before = _document(shortlist=())
    after = asyncio.run(docket_root_lookup_review(before))
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(docket_root_lookup_review(after))


def test_ivr_prompt_distinguishes_docket_and_opinion_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _document()
    root = before.roots[0]
    assert isinstance(root, FullDocketCitation)
    context = DocketLookupReviewContext.from_document(before, root)
    captured: list[object] = []

    async def fake_ivr(_session: object, spec: object, **_kwargs: object) -> IvrRun:
        captured.append(spec)
        return _run()

    module = importlib.import_module("mellea_lrc.validation.docket_root_lookup_review")
    monkeypatch.setattr(module, "run_instruct_ivr", fake_ivr)

    outcome = asyncio.run(IvrDocketLookupReviewer(session=object(), model_options={})(context))

    assert outcome.decision is None
    assert outcome.run == _run()
    assert len(captured) == 1
    spec = captured[0]
    assert spec.output_format is DocketLookupReviewDecision
    assert "type=d" in spec.prefix
    assert "type=o" in spec.prefix
    assert "do not compare" in spec.prefix
    assert "may be wrong" in spec.prefix
    assert '"candidate_index":0' in spec.user_variables["candidates"]
    assert json.loads(spec.user_variables["search_status"])["incomplete"] is False
