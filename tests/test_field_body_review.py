"""Field body review proposes an intended case without validating the cited locator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.citations.field_body_evidence import (
    FieldBodySearch,
    IntendedCaseConfidence,
    IntendedCaseDecision,
)
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_search.common import make_body_evidences
from mellea_lrc.validation.intended_case_llm_selection import SUBSTAGE, intended_case_llm_selection
from mellea_lrc.validation.intended_case_llm_selection.reviewer import IntendedCaseContext

SOURCE = "Smith v. Jones, No. 05-4206 (2d Cir. 2007)."
BODY = "An independent court cites Smith v. Jones, No. 01-9999 (2d Cir. 2006) as authority."
CITATION = "Smith v. Jones, No. 01-9999 (2d Cir. 2006)"
FIELD_SUBSTAGES = (
    (
        BodySource.COURTLISTENER_OPINION,
        "validate_roots.intended_case_discovery.courtlistener_opinion_retrieval",
    ),
    (BodySource.COURTLISTENER_RECAP, "validate_roots.intended_case_discovery.courtlistener_recap_retrieval"),
)


def _ready(
    *, bodies: tuple[tuple[BodySource, str], ...] = ((BodySource.COURTLISTENER_OPINION, BODY),)
) -> Document:
    number_start = SOURCE.index("05-4206")
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        substage="sites",
        source=SOURCE,
        span=Span(SOURCE.index("No."), number_start + len("05-4206")),
        number_span=Span(number_start, number_start + len("05-4206")),
    )
    document = Document.from_source(SOURCE).add_citation(root).complete_substage("sites")
    root = root.record("fields").with_case_name(SOURCE, Span(0, len("Smith v. Jones")))
    document = document.replace_citation(root).complete_substage("fields")
    root = root.record("roots").with_root(root.id)
    document = document.replace_citation(root).complete_substage("roots")
    root = root.record("lookup").with_identity_judgment(IdentityVerdict.UNDETERMINED)
    document = document.replace_citation(root).complete_substage("lookup")
    root = root.record("validate_roots.locator_body_corroboration.llm_judgment").with_route(
        "validate_roots.intended_case_discovery.courtlistener_opinion_retrieval"
    )
    document = document.replace_citation(root).complete_substage(
        "validate_roots.locator_body_corroboration.llm_judgment"
    )
    body_by_source = dict(bodies)
    for source, substage in FIELD_SUBSTAGES:
        root = document.roots[0].record(substage)
        body = body_by_source.get(source)
        evidence = (
            make_body_evidences(
                body_id=f"{source.value}:1",
                parent_id=None,
                url="https://example.test/case",
                issued_on=date(2010, 1, 1),
                date_basis="opinion.date_filed",
                metadata={"id": 1},
                body_text=body,
                locator="Smith",
                source_text=SOURCE,
                anchor_kind="case_name",
            )
            if body is not None
            else ()
        )
        root = root.with_field_body_search(
            FieldBodySearch(
                node_id=root.nodes[-1].id,
                source=source,
                retrospective_date=None,
                query_name="Smith",
                evidence=evidence,
            )
        )
        document = document.replace_citation(root).complete_substage(substage)
    return document


def _decision(**changes: object) -> IntendedCaseDecision:
    values = {
        "source": BodySource.COURTLISTENER_OPINION,
        "evidence_index": 0,
        "citation_quote": CITATION,
        "case_name": "Smith v. Jones",
        "locator": "No. 01-9999",
        "court": "2d Cir.",
        "date": "2006",
        "confidence": IntendedCaseConfidence.LIKELY,
        "reason": "The independent citation matches the parties and court, but prints another docket number.",
    }
    return IntendedCaseDecision.model_validate({**values, **changes})


def _decline(reason: str) -> IntendedCaseDecision:
    return IntendedCaseDecision(
        source=None,
        evidence_index=None,
        citation_quote=None,
        case_name=None,
        locator=None,
        court=None,
        date=None,
        confidence=None,
        reason=reason,
    )


@dataclass
class FakeReviewer:
    decision: IntendedCaseDecision
    calls: int = 0

    async def __call__(self, context: IntendedCaseContext) -> IntendedCaseDecision:
        self.calls += 1
        assert context.current["locator"] == "05-4206"
        return self.decision


def test_review_saves_grounded_intended_case_without_changing_cited_identity() -> None:
    before = _ready()
    reviewer = FakeReviewer(_decision())

    after = asyncio.run(intended_case_llm_selection(before, reviewer=reviewer))

    assert reviewer.calls == 1
    assert after.substage_runs[-1] == SUBSTAGE
    assert after.get_substage(FIELD_SUBSTAGES[-1][1]) == before
    original, recorded = before.roots[0], after.roots[0]
    assert recorded.locator == original.locator
    assert recorded.identity_judgments == original.identity_judgments
    assert recorded.body_searches == original.body_searches
    assert recorded.next_substage == "intended_case_resolution"
    review = recorded.intended_case_reviews[0]
    assert review.node_id == recorded.nodes[-1].id
    assert review.decision == _decision()
    assert review.grounded_quote == CITATION
    assert review.quote_similarity == 100
    assert review.quote_span is not None
    excerpt = recorded.field_body_searches[0].evidence[0].excerpt
    assert excerpt[review.quote_span.start : review.quote_span.end] == CITATION
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_no_evidence_declines_without_calling_reviewer() -> None:
    before = _ready(bodies=())
    reviewer = FakeReviewer(_decision())

    after = asyncio.run(intended_case_llm_selection(before, reviewer=reviewer))

    assert reviewer.calls == 0
    review = after.roots[0].intended_case_reviews[0]
    assert review.decision is not None and review.decision.source is None
    assert review.grounded_quote is None
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments
    assert after.roots[0].next_substage == "open_web_search"


def test_ungrounded_candidate_quote_becomes_review_failure() -> None:
    before = _ready()
    reviewer = FakeReviewer(_decision(citation_quote="Smith v. Jones, No. 99-9999 (2d Cir. 2006)"))

    after = asyncio.run(intended_case_llm_selection(before, reviewer=reviewer))

    review = after.roots[0].intended_case_reviews[0]
    assert review.decision is None
    assert review.failure_reason is not None and "98% similarity" in review.failure_reason
    assert after.roots[0].next_substage == "intended_case_review_retry"
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments


def test_contradictory_candidates_can_be_declined_with_reason() -> None:
    other = "Another opinion cites Smith v. Jones, No. 88-8888 (5th Cir. 1999) for a different rule."
    before = _ready(
        bodies=((BodySource.COURTLISTENER_OPINION, BODY), (BodySource.COURTLISTENER_RECAP, other))
    )

    class DecliningReviewer:
        async def __call__(self, context: IntendedCaseContext) -> IntendedCaseDecision:
            assert len(context.evidence) == 2
            return _decline("Two same-name citations have conflicting locators, courts, and dates.")

    after = asyncio.run(intended_case_llm_selection(before, reviewer=DecliningReviewer()))

    review = after.roots[0].intended_case_reviews[0]
    assert review.decision is not None and review.decision.source is None
    assert "conflicting locators" in review.decision.reason
    assert after.roots[0].next_substage == "open_web_search"
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments
