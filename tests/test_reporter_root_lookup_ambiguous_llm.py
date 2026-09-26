"""Model review of ambiguous reporter lookups uses every saved candidate."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from mellea_lrc.api import Document, grow_roots, reporter_root_lookup, reporter_root_lookup_ambiguous
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import Span
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactAmbiguityOutcome
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.validation._support.reporter_ambiguous_llm import (
    ReporterAmbiguousReviewDecision,
    ReporterAmbiguousReviewOutcome,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm import (
    STAGE,
    reporter_root_lookup_ambiguous_llm,
)

SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
NORMALIZED_NAME = {"kind": "adversarial", "plaintiff": "Bell Atl. Corp.", "defendant": "Twombly"}


def _cluster(identifier: int, name: str, *, full_name_available: bool = True) -> dict[str, object]:
    return {
        "id": identifier,
        "caseName": name,
        "caseNameFull": name if full_name_available else None,
        "court_id": "scotus",
        "dateFiled": "2007-05-21",
        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
    }


class FakeLookupClient:
    def __init__(self, names: tuple[str, ...], *, no_full_names: tuple[int, ...] = ()) -> None:
        self.response = CourtListenerCitationLookup.model_validate(
            {
                "citation": "550 U.S. 544",
                "status": 300,
                "clusters": [
                    _cluster(index + 1, name, full_name_available=index not in no_full_names)
                    for index, name in enumerate(names)
                ],
            }
        )
        self.lookup_calls = 0

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("550", "U.S.", "544")
        self.lookup_calls += 1
        return self.response

    def get_docket(self, docket_id: str) -> None:
        pytest.fail(f"The saved lookup needs no docket fetch: {docket_id}")


class FakeReviewer:
    def __init__(self, result: ReporterAmbiguousReviewDecision | ReporterAmbiguousReviewOutcome) -> None:
        self.result = result
        self.contexts: list[object] = []

    async def __call__(
        self, context: object
    ) -> ReporterAmbiguousReviewDecision | ReporterAmbiguousReviewOutcome:
        self.contexts.append(context)
        return self.result


def _decision(
    selected_candidate_index: int | None,
    *,
    case_name_quote: str | None = None,
    case_name_result: str | None = None,
) -> ReporterAmbiguousReviewDecision:
    result = case_name_result or ("match" if selected_candidate_index is not None else "undetermined")
    other_result = "match" if selected_candidate_index is not None else "undetermined"
    return ReporterAmbiguousReviewDecision.model_validate(
        {
            "selected_candidate_index": selected_candidate_index,
            "case_name": {
                "propose_replacement": case_name_quote is not None,
                "quote": case_name_quote,
                "normalized": NORMALIZED_NAME,
                "result": result,
                "reason": "The source and candidate parties were compared.",
            },
            "court": {
                "propose_replacement": False,
                "quote": None,
                "result": other_result,
                "reason": "The reporter and candidate court were compared.",
            },
            "date": {
                "propose_replacement": False,
                "quote": None,
                "result": other_result,
                "reason": "The stated year and candidate date were compared.",
            },
            "reason": "The saved candidates were considered together.",
        }
    )


def _review_input(
    names: tuple[str, ...], *, incorrect_case_name: bool = False, no_full_names: tuple[int, ...] = ()
) -> tuple[Document, FakeLookupClient]:
    client = FakeLookupClient(names, no_full_names=no_full_names)
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    if incorrect_case_name:
        root = roots.roots[0]
        start = SOURCE.index("Corp. v.")
        end = SOURCE.index(", 550")
        misread = root.record("test_incorrect_reading").with_case_name(SOURCE, Span(start, end))
        roots = roots.replace_citation(misread).complete("test_incorrect_reading")
    lookup = reporter_root_lookup(roots, client=client)
    before = reporter_root_lookup_ambiguous(lookup, client=client)
    resolution = before.roots[0].reporter_exact_ambiguity_resolution
    assert resolution is not None
    assert resolution.outcome is ReporterExactAmbiguityOutcome.NO_UNIQUE_RULE_MATCH
    assert before.roots[0].identity_judgments[-1].next_stage == STAGE
    return before, client


def test_model_choice_is_required_and_no_choice_cannot_compare_a_candidate() -> None:
    assert "selected_candidate_index" in ReporterAmbiguousReviewDecision.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        _decision(None, case_name_result="match")


@pytest.mark.parametrize(
    ("names", "no_full_names", "passing", "selected"),
    [
        (("Jones v. Smith", "Bell Atlantic Corporation v. Twombly"), (1,), (), 1),
        (
            (
                "Bell Atlantic Corporation v. Twombly",
                "Bell Atlantic Corporation v. Twombly",
                "Jones v. Smith",
            ),
            (),
            (0, 1),
            0,
        ),
    ],
)
def test_model_can_select_one_of_all_saved_candidates_after_zero_or_multiple_rule_matches(
    names: tuple[str, ...], no_full_names: tuple[int, ...], passing: tuple[int, ...], selected: int
) -> None:
    before, client = _review_input(names, no_full_names=no_full_names)
    reviewer = FakeReviewer(_decision(selected))

    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=reviewer))

    assert client.lookup_calls == 1
    assert len(reviewer.contexts) == 1
    context = reviewer.contexts[0]
    assert tuple(cluster.id for cluster in context.candidates) == tuple(
        str(index) for index in range(1, len(names) + 1)
    )
    assert context.passing_candidate_indices == passing
    assert len(context.rule_results) == len(names)
    assert before.roots[0].reporter_exact_ambiguity_resolution.passing_candidate_indices == passing
    assert after.get_stage("reporter_root_lookup_ambiguous") == before
    root = after.roots[0]
    previous = before.roots[0]
    assert len(root.nodes) == len(previous.nodes) + 1
    assert root.nodes[-1].stage == STAGE
    assert root.reporter_exact_lookup == previous.reporter_exact_lookup
    assert root.reporter_exact_ambiguity_resolution == previous.reporter_exact_ambiguity_resolution
    assert root.reporter_ambiguous_review.decision.selected_candidate_index == selected
    for judgments, earlier_judgments, readings in (
        (root.case_name_judgments, previous.case_name_judgments, root.case_name),
        (root.court_judgments, previous.court_judgments, root.court),
        (root.date_judgments, previous.date_judgments, root.date),
    ):
        assert judgments[:-1] == earlier_judgments
        judgment = judgments[-1]
        assert judgment.node_id == root.nodes[-1].id
        assert judgment.candidate_index == selected
        assert judgment.reading_index == len(readings) - 1
        assert judgment.result is MatchResult.MATCH
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert root.identity_judgments[-1].next_stage is None
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_stage("reporter_root_lookup_ambiguous") == before


def test_no_model_selection_remains_deferred_without_new_candidate_judgments() -> None:
    before, _ = _review_input(("Bell Atlantic Corporation v. Twombly",) * 2)
    previous = before.roots[0]
    reviewer = FakeReviewer(_decision(None))

    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    root = after.roots[0]
    assert root.reporter_ambiguous_review.decision.selected_candidate_index is None
    assert root.reporter_exact_lookup == previous.reporter_exact_lookup
    assert root.case_name_judgments == previous.case_name_judgments
    assert root.court_judgments == previous.court_judgments
    assert root.date_judgments == previous.date_judgments
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_stage == "reporter_root_search"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_no_selection_can_save_a_grounded_replacement_for_search() -> None:
    before, _ = _review_input(
        ("Bell Atlantic Corporation v. Twombly", "Jones v. Smith"),
        incorrect_case_name=True,
        no_full_names=(0,),
    )
    previous = before.roots[0]
    reviewer = FakeReviewer(_decision(None, case_name_quote="Bell Atl. Corp. v. Twombly"))

    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=reviewer))

    root = after.roots[0]
    assert root.case_name[-1].span == Span(0, SOURCE.index(", 550"))
    assert len(root.case_name) == len(previous.case_name) + 1
    assert root.case_name_judgments == previous.case_name_judgments
    assert root.court_judgments == previous.court_judgments
    assert root.date_judgments == previous.date_judgments
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_stage == "reporter_root_search"


def test_selected_candidate_uses_grounded_corrected_reading() -> None:
    before, _ = _review_input(
        ("Bell Atlantic Corporation v. Twombly", "Jones v. Smith"),
        incorrect_case_name=True,
        no_full_names=(0,),
    )
    previous = before.roots[0]
    assert previous.reporter_exact_ambiguity_resolution.passing_candidate_indices == ()
    reviewer = FakeReviewer(_decision(0, case_name_quote="Bell Atl. Corp. v. Twombly"))

    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=reviewer))

    root = after.roots[0]
    assert len(root.case_name) == len(previous.case_name) + 1
    assert root.case_name[-1].quote == "Bell Atl. Corp. v. Twombly"
    assert root.case_name[-1].span == Span(0, SOURCE.index(", 550"))
    assert root.case_name[-1].node_id == root.nodes[-1].id
    assert root.case_name_judgments[-1].candidate_index == 0
    assert root.case_name_judgments[-1].reading_index == len(root.case_name) - 1
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY


@pytest.mark.parametrize(
    ("case_name_result", "verdict", "next_stage"),
    [
        ("mismatch", IdentityVerdict.WRONG_IDENTITY, None),
        ("undetermined", IdentityVerdict.DEFERRED, "reporter_root_search"),
    ],
)
def test_selected_candidate_identity_follows_field_assessments(
    case_name_result: str, verdict: IdentityVerdict, next_stage: str | None
) -> None:
    before, _ = _review_input(("Bell Atlantic Corporation v. Twombly",) * 2)

    after = asyncio.run(
        reporter_root_lookup_ambiguous_llm(
            before, reviewer=FakeReviewer(_decision(1, case_name_result=case_name_result))
        )
    )

    root = after.roots[0]
    assert root.case_name_judgments[-1].candidate_index == 1
    assert root.case_name_judgments[-1].result is MatchResult(case_name_result)
    assert root.identity_judgments[-1].verdict is verdict
    assert root.identity_judgments[-1].next_stage == next_stage


@pytest.mark.parametrize(
    "decision",
    [
        _decision(99),
        _decision(0, case_name_quote="A name absent from the filing v. Twombly"),
    ],
)
def test_invalid_selection_or_ungrounded_correction_defers_without_partial_updates(
    decision: ReporterAmbiguousReviewDecision,
) -> None:
    before, _ = _review_input(("Bell Atlantic Corporation v. Twombly",) * 2)
    previous = before.roots[0]

    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=FakeReviewer(decision)))

    root = after.roots[0]
    assert root.reporter_ambiguous_review.decision is None
    assert root.reporter_ambiguous_review.failure_reason
    assert root.case_name == previous.case_name
    assert root.court == previous.court
    assert root.date == previous.date
    assert root.case_name_judgments == previous.case_name_judgments
    assert root.court_judgments == previous.court_judgments
    assert root.date_judgments == previous.date_judgments
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_stage == "reporter_root_search"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_failed_model_review_keeps_trace_and_defers() -> None:
    before, _ = _review_input(("Bell Atlantic Corporation v. Twombly",) * 2)
    run = IvrRun.model_validate(
        {
            "success": False,
            "selected_attempt": 0,
            "attempts": [
                {
                    "output": '{"selected_candidate_index":',
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
            "instruction": "Review every candidate",
            "prefix": None,
            "grounding_context": {},
            "user_variables": {},
            "output_schema": None,
        }
    )
    reviewer = FakeReviewer(
        ReporterAmbiguousReviewOutcome(decision=None, run=run, failure_reason="Incomplete JSON")
    )

    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=reviewer))

    root = after.roots[0]
    assert root.reporter_ambiguous_review.decision is None
    assert root.reporter_ambiguous_review.ivr == run
    assert root.reporter_ambiguous_review.failure_reason == "Incomplete JSON"
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_stage == "reporter_root_search"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_stage_ignores_rule_selected_root_and_does_not_call_reviewer() -> None:
    client = FakeLookupClient(("Jones v. Smith", "Bell Atlantic Corporation v. Twombly"))
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    lookup = reporter_root_lookup(roots, client=client)
    before = reporter_root_lookup_ambiguous(lookup, client=client)
    assert before.roots[0].identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    reviewer = FakeReviewer(_decision(1))

    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=reviewer))

    assert reviewer.contexts == []
    assert after.roots == before.roots
    assert after.stage_runs[-1] == STAGE
    assert after.get_stage("reporter_root_lookup_ambiguous") == before
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(reporter_root_lookup_ambiguous_llm(after, reviewer=reviewer))
