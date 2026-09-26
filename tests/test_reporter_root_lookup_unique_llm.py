"""One unique reporter candidate is reread and judged in one model review."""

from __future__ import annotations

import asyncio

import pytest
from mellea_lrc.validation._support.reporter_unique_llm import (
    ReporterUniqueReviewDecision,
    ReporterUniqueReviewOutcome,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm import (
    STAGE,
    reporter_root_lookup_unique_llm,
)

from mellea_lrc.api import Document, grow_roots, reporter_root_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import Span
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.ivr import IvrRun

SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."


class FakeLookupClient:
    def __init__(self, *, case_name_full: str | None = None, court_id: str = "scotus") -> None:
        self.response = CourtListenerCitationLookup.model_validate(
            {
                "citation": "550 U.S. 544",
                "status": 200,
                "clusters": [
                    {
                        "id": 1,
                        "caseName": "Bell Atlantic Corporation v. Twombly",
                        "caseNameFull": case_name_full,
                        "court_id": court_id,
                        "dateFiled": "2007-05-21",
                        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                    }
                ],
            }
        )

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("550", "U.S.", "544")
        return self.response

    def get_docket(self, docket_id: str) -> None:
        pytest.fail(f"The unique review should use the saved lookup, not fetch docket {docket_id}")


class FakeReviewer:
    def __init__(self, outcome: ReporterUniqueReviewDecision | ReporterUniqueReviewOutcome) -> None:
        self.outcome = outcome
        self.contexts: list[object] = []

    async def __call__(self, context: object) -> ReporterUniqueReviewDecision | ReporterUniqueReviewOutcome:
        self.contexts.append(context)
        return self.outcome


def _decision(
    *,
    case_name_quote: str | None = "Bell Atl. Corp. v. Twombly",
    case_name_result: str = "match",
    court_result: str = "match",
    date_result: str = "match",
) -> ReporterUniqueReviewDecision:
    return ReporterUniqueReviewDecision.model_validate(
        {
            "case_name": {
                "quote": case_name_quote,
                "result": case_name_result,
                "reason": "The written parties identify the retrieved case.",
            },
            "court": {
                "quote": None,
                "result": court_result,
                "reason": "The reporter supplies the stated court.",
            },
            "date": {
                "quote": "2007",
                "result": date_result,
                "reason": "The stated year agrees with the opinion date.",
            },
            "reason": "The citation and the saved opinion record describe the same case.",
        }
    )


def _review_input(*, incorrect_case_name: bool = False) -> Document:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    if incorrect_case_name:
        root = roots.roots[0]
        start = SOURCE.index("Corp. v.")
        end = SOURCE.index(", 550")
        misread = root.record("test_incorrect_reading").with_case_name(SOURCE, Span(start, end))
        roots = roots.replace_citation(misread).complete("test_incorrect_reading")
    result = reporter_root_lookup(roots, client=FakeLookupClient())
    assert result.roots[0].identity_judgments[-1].next_stage == STAGE
    return result


def _trace() -> IvrRun:
    return IvrRun.model_validate(
        {
            "success": False,
            "selected_attempt": 0,
            "attempts": [
                {
                    "output": '{"case_name":',
                    "requirements": [
                        {
                            "description": "Return the required schema",
                            "passed": False,
                            "reason": "Incomplete JSON",
                            "score": None,
                        }
                    ],
                }
            ],
            "backend": "FakeBackend",
            "model": "test-model",
            "model_options": {},
            "instruction": "Review this citation",
            "prefix": None,
            "grounding_context": {},
            "user_variables": {},
            "output_schema": None,
        }
    )


def test_nullable_correction_quotes_are_required_by_the_provider_schema() -> None:
    schema = ReporterUniqueReviewDecision.model_json_schema()
    assessment = schema["$defs"]["ReporterUniqueFieldAssessment"]
    assert set(assessment["required"]) == {"quote", "result", "reason"}


def test_combined_review_corrects_grounded_name_and_judges_latest_readings() -> None:
    before = _review_input(incorrect_case_name=True)
    previous = before.roots[0]
    reviewer = FakeReviewer(_decision())

    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    assert after.stage_runs[-1] == STAGE
    root = after.roots[0]
    assert len(root.nodes) == len(previous.nodes) + 1
    assert len(root.case_name) == len(previous.case_name) + 1
    assert root.case_name[-1].quote == "Bell Atl. Corp. v. Twombly"
    assert root.case_name[-1].span == Span(0, SOURCE.index(", 550"))
    assert root.case_name[-1].node_id == root.nodes[-1].id
    for judgments, readings in (
        (root.case_name_judgments, root.case_name),
        (root.court_judgments, root.court),
        (root.date_judgments, root.date),
    ):
        judgment = judgments[-1]
        assert judgment.node_id == root.nodes[-1].id
        assert judgment.candidate_index == 0
        assert judgment.reading_index == len(readings) - 1
        assert judgment.result is MatchResult.MATCH
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert root.identity_judgments[-1].next_stage is None
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_stage("reporter_root_lookup") == before
    assert restored.get_stage(STAGE) == after


def test_unique_review_processes_only_citations_routed_to_its_stage() -> None:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    already_admitted = reporter_root_lookup(
        roots,
        client=FakeLookupClient(case_name_full="Bell Atlantic Corporation v. Twombly"),
    )
    assert already_admitted.roots[0].identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    reviewer = FakeReviewer(_decision())

    after = asyncio.run(reporter_root_lookup_unique_llm(already_admitted, reviewer=reviewer))

    assert reviewer.contexts == []
    assert after.roots == already_admitted.roots
    assert after.stage_runs[-1] == STAGE


def test_model_field_mismatch_produces_wrong_identity() -> None:
    before = _review_input()
    reviewer = FakeReviewer(_decision(case_name_result="mismatch"))

    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=reviewer))

    root = after.roots[0]
    assert len(reviewer.contexts) == 1
    assert root.case_name_judgments[-1].result is MatchResult.MISMATCH
    assert root.court_judgments[-1].result is MatchResult.MATCH
    assert root.date_judgments[-1].result is MatchResult.MATCH
    assert root.identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert root.identity_judgments[-1].next_stage is None


def test_failed_review_preserves_ivr_trace_and_routes_to_search() -> None:
    before = _review_input()
    run = _trace()
    reviewer = FakeReviewer(
        ReporterUniqueReviewOutcome(decision=None, run=run, failure_reason="Incomplete JSON")
    )

    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    root = after.roots[0]
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_stage == "reporter_root_search"
    assert "Incomplete JSON" in root.model_dump_json()
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_ungrounded_model_correction_is_not_written_and_routes_to_search() -> None:
    before = _review_input()
    earlier = before.roots[0]
    reviewer = FakeReviewer(_decision(case_name_quote="Name absent from the source v. Twombly"))

    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=reviewer))

    root = after.roots[0]
    assert len(reviewer.contexts) == 1
    assert root.case_name == earlier.case_name
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_stage == "reporter_root_search"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_review_records_an_undetermined_field_without_a_source_reading() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544."
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    before = reporter_root_lookup(roots, client=FakeLookupClient())
    assert before.roots[0].identity_judgments[-1].next_stage == STAGE
    decision = _decision(date_result="undetermined").model_copy(
        update={"date": _decision(date_result="undetermined").date.model_copy(update={"quote": None})}
    )

    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=FakeReviewer(decision)))

    root = after.roots[0]
    assert not root.date
    assert root.date_judgments[-1].reading_index is None
    assert root.date_judgments[-1].result is MatchResult.UNDETERMINED
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert Document.model_validate_json(after.model_dump_json()) == after
