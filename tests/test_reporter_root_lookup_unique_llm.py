"""One unique reporter candidate is reread and judged in one model review."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from mellea_lrc.api import (
    Document,
    grow_roots,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_lookup_docket_retrieval,
    reporter_root_lookup_unique_rule_judgment,
)
from mellea_lrc.providers.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import Span
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterUniqueFieldAssessment
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment import (
    STAGE,
    reporter_root_lookup_unique_llm_judgment,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment.reviewer import (
    ReporterUniqueReviewDecision,
    ReporterUniqueReviewOutcome,
)

SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
NORMALIZED_NAME = {"kind": "adversarial", "plaintiff": "Bell Atl. Corp.", "defendant": "Twombly"}
REVIEW_STAGE = "13.1_reporter_root_lookup_unique_rule_judgment"
DOCKET_STAGE = "12.2_reporter_root_lookup_docket_retrieval"


class FakeLookupClient:
    def __init__(
        self,
        *,
        case_name: str | None = "Bell Atlantic Corporation v. Twombly",
        case_name_full: str | None = None,
        court_id: str = "scotus",
        date_filed: str | None = "2007-05-21",
    ) -> None:
        self.lookup_calls = 0
        self.response = CourtListenerCitationLookup.model_validate(
            {
                "citation": "550 U.S. 544",
                "status": 200,
                "clusters": [
                    {
                        "id": 1,
                        "caseName": case_name,
                        "caseNameFull": case_name_full,
                        "court_id": court_id,
                        "dateFiled": date_filed,
                        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                    }
                ],
            }
        )

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("550", "U.S.", "544")
        self.lookup_calls += 1
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
    court_quote: str | None = None,
    date_quote: str | None = "2007",
    date_result: str = "match",
    case_name_normalized: dict[str, str] | None = NORMALIZED_NAME,
) -> ReporterUniqueReviewDecision:
    return ReporterUniqueReviewDecision.model_validate(
        {
            "case_name": {
                "propose_replacement": case_name_quote is not None,
                "quote": case_name_quote,
                "normalized": case_name_normalized,
                "result": case_name_result,
                "reason": "The written parties identify the retrieved case.",
            },
            "court": {
                "propose_replacement": court_quote is not None,
                "quote": court_quote,
                "result": court_result,
                "reason": "The reporter supplies the stated court.",
            },
            "date": {
                "propose_replacement": date_quote is not None,
                "quote": date_quote,
                "result": date_result,
                "reason": "The stated year agrees with the opinion date.",
            },
            "reason": "The citation and the saved opinion record describe the same case.",
        }
    )


def _review_input(
    *, incorrect_case_name: bool = False, source: str = SOURCE, client: FakeLookupClient | None = None
) -> Document:
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    if incorrect_case_name:
        root = roots.roots[0]
        start = source.index("Corp. v.")
        end = source.index(", 550")
        misread = root.record("test_incorrect_reading").with_case_name(source, Span(start, end))
        roots = roots.replace_citation(misread).complete("test_incorrect_reading")
    service = client or FakeLookupClient()
    retrieved = reporter_root_lookup_cluster_retrieval(roots, client=service)
    assert retrieved.roots[0].case_name_judgments == ()
    assert retrieved.roots[0].identity_judgments == ()
    calls = service.lookup_calls
    dockets = reporter_root_lookup_docket_retrieval(retrieved, client=service)
    result = reporter_root_lookup_unique_rule_judgment(dockets)
    assert service.lookup_calls == calls == 1
    assert result.get_stage("12.1_reporter_root_lookup_cluster_retrieval") == retrieved
    assert result.get_stage("12.2_reporter_root_lookup_docket_retrieval") == dockets
    assert result.roots[0].identity_judgments == ()
    assert result.roots[0].next_stage == STAGE
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


def test_replacement_intent_and_nullable_quotes_are_required_by_the_provider_schema() -> None:
    schema = ReporterUniqueReviewDecision.model_json_schema()
    assert set(schema["$defs"]["ReporterCaseNameAssessment"]["required"]) == {
        "propose_replacement",
        "quote",
        "normalized",
        "result",
        "reason",
    }
    assert set(schema["$defs"]["ReporterUniqueFieldAssessment"]["required"]) == {
        "propose_replacement",
        "quote",
        "result",
        "reason",
    }


@pytest.mark.parametrize(
    ("propose_replacement", "quote", "valid"),
    [
        (False, None, True),
        (True, "Bell Atl. Corp. v. Twombly", True),
        (True, None, False),
        (True, "", False),
        (False, "Bell Atl. Corp. v. Twombly", False),
    ],
)
def test_replacement_intent_agrees_with_quote(
    propose_replacement: bool, quote: str | None, valid: bool
) -> None:
    assessment = {
        "propose_replacement": propose_replacement,
        "quote": quote,
        "result": "match",
        "reason": "The filing and opinion agree.",
    }
    if valid:
        parsed = ReporterUniqueFieldAssessment.model_validate(assessment)
        assert parsed.propose_replacement is propose_replacement
        assert parsed.quote == quote
    else:
        with pytest.raises(ValidationError):
            ReporterUniqueFieldAssessment.model_validate(assessment)


@pytest.mark.parametrize(
    ("propose_replacement", "quote"),
    [(True, None), (False, "Bell Atl. Corp. v. Twombly")],
)
def test_generated_review_json_rejects_inconsistent_replacement_intent(
    propose_replacement: bool, quote: str | None
) -> None:
    generated = json.loads(_decision().model_dump_json())
    generated["case_name"]["propose_replacement"] = propose_replacement
    generated["case_name"]["quote"] = quote

    with pytest.raises(ValidationError):
        ReporterUniqueReviewDecision.model_validate_json(json.dumps(generated))


def test_combined_review_corrects_grounded_name_and_judges_latest_readings() -> None:
    client = FakeLookupClient()
    before = _review_input(incorrect_case_name=True, client=client)
    previous = before.roots[0]
    reviewer = FakeReviewer(_decision())

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=reviewer))

    assert client.lookup_calls == 1
    assert len(reviewer.contexts) == 1
    assert reviewer.contexts[0].inferred_court_note is not None
    assert "U.S." in reviewer.contexts[0].inferred_court_note
    assert "Supreme Court" in reviewer.contexts[0].inferred_court_note
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
    assert root.next_stage is None
    assert [route.value for route in root.routes] == [DOCKET_STAGE, REVIEW_STAGE, STAGE, None]
    assert root.routes[-1].node_id == root.nodes[-1].id
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_stage(REVIEW_STAGE) == before
    assert restored.get_stage(STAGE) == after


def test_no_replacement_proposal_keeps_existing_readings() -> None:
    before = _review_input()
    previous = before.roots[0]
    reviewer = FakeReviewer(_decision(case_name_quote=None, date_quote=None))

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=reviewer))

    root = after.roots[0]
    assert len(reviewer.contexts) == 1
    assert root.case_name == previous.case_name
    assert root.court == previous.court
    assert root.date == previous.date
    assert root.reporter_unique_review.decision.case_name.propose_replacement is False
    assert root.reporter_unique_review.decision.date.propose_replacement is False
    assert root.case_name_judgments[-1].reading_index == len(previous.case_name) - 1
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY


def test_model_normalizes_a_grounded_name_across_page_layout_noise() -> None:
    source = "Bell Atl. Corp.\nPage 14\nv. Twombly, 550 U.S. 544 (2007)."
    quote = source[: source.index(", 550")]
    before = _review_input(source=source)
    previous = before.roots[0]

    after = asyncio.run(
        reporter_root_lookup_unique_llm_judgment(
            before, reviewer=FakeReviewer(_decision(case_name_quote=quote))
        )
    )

    root = after.roots[0]
    assert len(root.case_name) == len(previous.case_name) + 1
    reading = root.case_name[-1]
    assert reading.quote == quote
    assert reading.span == Span(0, len(quote))
    assert reading.normalized_by == "model"
    assert reading.get_normalized() == CaseName.model_validate(NORMALIZED_NAME)
    assert reading.get_normalized().as_citation() == "Bell Atl. Corp. v. Twombly"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_model_can_change_only_the_normalization_of_an_existing_grounded_name() -> None:
    before = _review_input()
    previous = before.roots[0]
    expanded = {**NORMALIZED_NAME, "plaintiff": "Bell Atlantic Corp."}

    after = asyncio.run(
        reporter_root_lookup_unique_llm_judgment(
            before,
            reviewer=FakeReviewer(_decision(case_name_quote=None, case_name_normalized=expanded)),
        )
    )

    root = after.roots[0]
    assert len(root.case_name) == len(previous.case_name) + 1
    assert root.case_name[-1].quote == previous.case_name[-1].quote
    assert root.case_name[-1].span == previous.case_name[-1].span
    assert root.case_name[-1].normalized_by == "model"
    assert root.case_name[-1].get_normalized() == CaseName(
        kind=CaseNameKind.ADVERSARIAL, plaintiff="Bell Atlantic Corp.", defendant="Twombly"
    )
    assert root.case_name_judgments[-1].reading_index == len(root.case_name) - 1
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_grounded_court_and_date_replacements_are_normalized_by_rules() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007). Later 2d Cir. 2008."
    before = _review_input(source=source, incorrect_case_name=True)
    previous = before.roots[0]
    decision = _decision(
        court_quote="2d Cir.",
        date_quote="2008",
    )

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=FakeReviewer(decision)))

    root = after.roots[0]
    assert root.reporter_unique_review.decision == decision
    for field, expected_quote, normalized_attribute, normalized_value in (
        ("court", "2d Cir.", "id", "ca2"),
        ("date", "2008", "year", 2008),
    ):
        old_readings = getattr(previous, field)
        readings = getattr(root, field)
        judgment = getattr(root, f"{field}_judgments")[-1]
        assert readings[:-1] == old_readings
        assert len(readings) == len(old_readings) + 1
        assert readings[-1].quote == expected_quote
        assert readings[-1].span == Span(
            source.index(expected_quote), source.index(expected_quote) + len(expected_quote)
        )
        assert readings[-1].normalizable
        assert getattr(readings[-1].get_normalized(), normalized_attribute) == normalized_value
        assert readings[-1].node_id == root.nodes[-1].id
        assert judgment.reading_index == len(readings) - 1
        assert judgment.result is MatchResult.MATCH
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_grounded_name_without_model_normalization_is_rejected() -> None:
    before = _review_input()
    previous = before.roots[0]

    after = asyncio.run(
        reporter_root_lookup_unique_llm_judgment(
            before,
            reviewer=FakeReviewer(_decision(case_name_quote=None, case_name_normalized=None)),
        )
    )

    root = after.roots[0]
    assert root.case_name == previous.case_name
    assert root.reporter_unique_review.decision is None
    assert "normalized name" in root.reporter_unique_review.failure_reason
    assert root.identity_judgments == previous.identity_judgments == ()
    assert root.next_stage == "reporter_root_search"


def test_unique_review_accepts_a_grounded_partial_without_supplying_missing_party() -> None:
    source = "Bell Atl. Corp., 550 U.S. 544 (2007)."
    before = _review_input(source=source, client=FakeLookupClient(case_name="Other v. Party"))
    fragment = "Bell Atl. Corp."
    partial = {"kind": "partial", "partial": fragment}
    decision = _decision(
        case_name_quote=fragment,
        case_name_normalized=partial,
        case_name_result="mismatch",
    )

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=FakeReviewer(decision)))
    root = after.roots[0]
    assert root.reporter_unique_review.decision == decision
    assert root.case_name[-1].quote == fragment
    assert root.case_name[-1].get_normalized() == CaseName(kind=CaseNameKind.PARTIAL, partial=fragment)
    assert root.case_name_judgments[-1].result is MatchResult.MISMATCH
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_model_normalization_without_a_grounded_name_is_rejected() -> None:
    before = _review_input(source="550 U.S. 544 (2007).")
    assert not before.roots[0].case_name

    after = asyncio.run(
        reporter_root_lookup_unique_llm_judgment(
            before,
            reviewer=FakeReviewer(_decision(case_name_quote=None, case_name_result="unavailable")),
        )
    )

    root = after.roots[0]
    assert not root.case_name
    assert root.reporter_unique_review.decision is None
    assert "grounded reading" in root.reporter_unique_review.failure_reason
    assert root.identity_judgments == ()
    assert root.next_stage == "reporter_root_search"


def test_unique_review_processes_only_citations_routed_to_its_stage() -> None:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    retrieved = reporter_root_lookup_cluster_retrieval(
        roots,
        client=FakeLookupClient(case_name_full="Bell Atlantic Corporation v. Twombly"),
    )
    dockets = reporter_root_lookup_docket_retrieval(retrieved)
    already_admitted = reporter_root_lookup_unique_rule_judgment(dockets)
    assert already_admitted.roots[0].identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    reviewer = FakeReviewer(_decision())

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(already_admitted, reviewer=reviewer))

    assert reviewer.contexts == []
    assert after.roots == already_admitted.roots
    assert after.stage_runs[-1] == STAGE


def test_model_field_mismatch_produces_wrong_identity() -> None:
    before = _review_input()
    reviewer = FakeReviewer(_decision(case_name_result="mismatch"))

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=reviewer))

    root = after.roots[0]
    assert len(reviewer.contexts) == 1
    assert root.case_name_judgments[-1].result is MatchResult.MISMATCH
    assert root.court_judgments[-1].result is MatchResult.MATCH
    assert root.date_judgments[-1].result is MatchResult.MATCH
    assert root.identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert root.next_stage is None
    assert [route.value for route in root.routes] == [DOCKET_STAGE, REVIEW_STAGE, STAGE, None]


def test_failed_review_preserves_ivr_trace_and_routes_to_search() -> None:
    before = _review_input()
    run = _trace()
    reviewer = FakeReviewer(
        ReporterUniqueReviewOutcome(decision=None, run=run, failure_reason="Incomplete JSON")
    )

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    root = after.roots[0]
    assert root.identity_judgments == before.roots[0].identity_judgments == ()
    assert root.next_stage == "reporter_root_search"
    assert [route.value for route in root.routes] == [
        DOCKET_STAGE,
        REVIEW_STAGE,
        STAGE,
        "reporter_root_search",
    ]
    assert "Incomplete JSON" in root.model_dump_json()
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_ungrounded_model_correction_is_not_written_and_routes_to_search() -> None:
    before = _review_input()
    earlier = before.roots[0]
    reviewer = FakeReviewer(_decision(case_name_quote="Name absent from the source v. Twombly"))

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=reviewer))

    root = after.roots[0]
    assert len(reviewer.contexts) == 1
    assert root.case_name == earlier.case_name
    assert root.identity_judgments == earlier.identity_judgments == ()
    assert root.next_stage == "reporter_root_search"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_review_records_unavailable_field_without_a_source_reading() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544."
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    retrieved = reporter_root_lookup_cluster_retrieval(roots, client=FakeLookupClient())
    dockets = reporter_root_lookup_docket_retrieval(retrieved)
    before = reporter_root_lookup_unique_rule_judgment(dockets)
    assert before.roots[0].next_stage == STAGE
    decision = _decision(date_quote=None, date_result="unavailable")

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=FakeReviewer(decision)))

    root = after.roots[0]
    assert not root.date
    assert root.date_judgments[-1].reading_index is None
    assert root.date_judgments[-1].result is MatchResult.UNAVAILABLE
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert Document.model_validate_json(after.model_dump_json()) == after


@pytest.mark.parametrize(
    ("source", "decision", "expected_reason"),
    [
        (SOURCE, _decision(case_name_result="unavailable"), "both have evidence"),
        (
            "Bell Atl. Corp. v. Twombly, 550 U.S. 544.",
            _decision(date_quote=None, date_result="match"),
            "has no filing reading",
        ),
    ],
)
def test_review_rejects_field_state_inconsistent_with_context(
    source: str, decision: ReporterUniqueReviewDecision, expected_reason: str
) -> None:
    before = _review_input(source=source)

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=FakeReviewer(decision)))

    root = after.roots[0]
    assert root.reporter_unique_review.decision is None
    assert expected_reason in root.reporter_unique_review.failure_reason
    assert root.identity_judgments == ()
    assert root.next_stage == "reporter_root_search"


def test_review_rejects_match_when_candidate_has_no_name_evidence() -> None:
    before = _review_input(client=FakeLookupClient(case_name=None))

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=FakeReviewer(_decision())))

    root = after.roots[0]
    assert root.reporter_unique_review.decision is None
    assert "case_name has no usable selected-record evidence" in root.reporter_unique_review.failure_reason
    assert root.identity_judgments == ()
    assert root.next_stage == "reporter_root_search"


def test_review_accepts_unavailable_when_candidate_lacks_date_evidence() -> None:
    before = _review_input(client=FakeLookupClient(date_filed=None))
    decision = _decision(date_quote=None, date_result="unavailable")

    after = asyncio.run(reporter_root_lookup_unique_llm_judgment(before, reviewer=FakeReviewer(decision)))

    root = after.roots[0]
    assert root.reporter_unique_review.decision == decision
    assert root.date_judgments[-1].reading_index == len(root.date) - 1
    assert root.date_judgments[-1].result is MatchResult.UNAVAILABLE
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
