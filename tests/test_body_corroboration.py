"""Third-party body citations use one review across independently saved providers."""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.body_evidence import (
    BodyCorroborationDecision,
    BodySearch,
    BodySource,
)
from mellea_lrc.model.citations.judgments import IdentityBasis, IdentityVerdict
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_corroboration import STAGE, body_corroboration_review
from mellea_lrc.validation.body_corroboration.reviewer import BodyCorroborationContext, _diverse_evidence
from mellea_lrc.validation.body_search.common import make_body_evidences

SOURCE = "Smith v. Jones, No. 05-4206 (2d Cir. 2007)."
BODY = "The court discussed Smith v. Jones, No. 05-4206 (2d Cir. 2007), in its analysis."
STAGES = (
    (BodySource.COURTLISTENER_OPINION, "20_courtlistener_opinion_body_search"),
    (BodySource.COURTLISTENER_RECAP, "21_courtlistener_recap_body_search"),
    (BodySource.GOVINFO_OPINION, "22_govinfo_opinion_body_search"),
)


def _document(*, include_evidence: bool = True, body: str = BODY) -> Document:
    number_start = SOURCE.index("05-4206")
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        stage="sites",
        source=SOURCE,
        span=Span(SOURCE.index("No."), number_start + len("05-4206")),
        number_span=Span(number_start, number_start + len("05-4206")),
    )
    document = Document.from_source(SOURCE).add_citation(root).complete("sites")
    root = root.record("fields")
    root = root.with_case_name(SOURCE, Span(0, len("Smith v. Jones")))
    root = root.with_court(SOURCE, Span(SOURCE.index("2d Cir."), SOURCE.index("2d Cir.") + 7))
    root = root.with_date(SOURCE, Span(SOURCE.index("2007"), SOURCE.index("2007") + 4))
    document = document.replace_citation(root).complete("fields")
    document = document.replace_citation(root.record("roots").with_root(root.id)).complete("roots")
    for source, stage in STAGES:
        root = document.roots[0].record(stage)
        evidence = (
            make_body_evidences(
                body_id=f"{source.value}:1",
                parent_id=None,
                url="https://example.test/source",
                issued_on=date(2006, 1, 1),
                date_basis="opinion.date_filed",
                metadata={"id": 1},
                body_text=body,
                locator="05-4206",
                source_text=SOURCE,
                case_name="Smith v. Jones",
            )
            if include_evidence and source is BodySource.GOVINFO_OPINION
            else ()
        )
        root = root.with_body_search(
            BodySearch(node_id=root.nodes[-1].id, source=source, retrospective_date=None, evidence=evidence)
        )
        document = document.replace_citation(root).complete(stage)
    return document


def _decision(
    *,
    source: BodySource = BodySource.GOVINFO_OPINION,
    court_result: str = "match",
    quote: str | None = None,
    third_party_locator: str = "05-4206",
) -> BodyCorroborationDecision:
    return BodyCorroborationDecision.model_validate(
        {
            "source": source.value,
            "evidence_index": 0,
            "citation_quote": quote or "Smith v. Jones, No. 05-4206 (2d Cir. 2007)",
            "filing": {
                "locator": "05-4206",
                "case_name": "Smith v. Jones",
                "normalized_case_name": {
                    "kind": "adversarial",
                    "plaintiff": "Smith",
                    "defendant": "Jones",
                    "subject": None,
                },
                "court": "2d Cir.",
                "date": "2007",
            },
            "third_party": {
                "locator": third_party_locator,
                "case_name": "Smith v. Jones",
                "court": "2d Cir.",
                "date": "2007",
            },
            "comparisons": {
                field: {
                    "result": court_result if field == "court" else "match",
                    "reason": "Compared both citations.",
                }
                for field in ("locator", "case_name", "court", "date")
            },
            "reason": "The third-party filing cites the same docket and named case.",
        }
    )


class FakeReviewer:
    def __init__(self, decision: BodyCorroborationDecision) -> None:
        self.decision = decision
        self.contexts: list[object] = []

    async def __call__(self, context: object) -> BodyCorroborationDecision:
        self.contexts.append(context)
        return self.decision


def test_one_judgment_can_select_govinfo_after_other_provider_searches() -> None:
    document = _document()
    reviewer = FakeReviewer(_decision())
    reviewed = asyncio.run(body_corroboration_review(document, reviewer=reviewer))
    root = reviewed.roots[0]
    assert len(reviewer.contexts) == 1
    assert len(root.body_searches) == 3
    assert root.body_reviews[-1].decision.source is BodySource.GOVINFO_OPINION
    assert root.body_reviews[-1].grounded_quote == "Smith v. Jones, No. 05-4206 (2d Cir. 2007)"
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert root.identity_judgments[-1].basis is IdentityBasis.THIRD_PARTY
    assert reviewed.get_stage(STAGES[-1][1]) == document
    assert Document.model_validate_json(reviewed.model_dump_json()) == reviewed


def test_independent_field_mismatch_flags_wrong_identity() -> None:
    reviewed = asyncio.run(
        body_corroboration_review(_document(), reviewer=FakeReviewer(_decision(court_result="mismatch")))
    )
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert reviewed.roots[0].body_reviews[-1].decision.comparisons.court.result.value == "mismatch"


def test_case_name_anchor_allows_model_to_judge_equivalent_docket_spelling() -> None:
    body = "A later brief cited Smith v. Jones, No. 05-CV-4206 (2d Cir. 2007)."
    document = _document(body=body)
    assert document.roots[0].body_searches[-1].evidence[0].anchor_kind == "case_name"
    decision = _decision(
        quote="Smith v. Jones, No. 05-CV-4206 (2d Cir. 2007)",
        third_party_locator="05-CV-4206",
    )
    reviewed = asyncio.run(body_corroboration_review(document, reviewer=FakeReviewer(decision)))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY


def test_ungrounded_quote_cannot_create_a_verdict() -> None:
    decision = _decision(quote="Smith v. Jones")
    reviewed = asyncio.run(body_corroboration_review(_document(), reviewer=FakeReviewer(decision)))
    root = reviewed.roots[0]
    assert root.body_reviews[-1].decision is None
    assert "anchor" in root.body_reviews[-1].failure_reason
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED


def test_unquoted_third_party_field_cannot_create_a_verdict() -> None:
    decision = _decision()
    fabricated = decision.model_copy(
        update={"third_party": decision.third_party.model_copy(update={"case_name": "Another v. Case"})}
    )
    reviewed = asyncio.run(body_corroboration_review(_document(), reviewer=FakeReviewer(fabricated)))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert "case_name" in reviewed.roots[0].body_reviews[-1].failure_reason


def test_docket_number_digit_must_match_the_grounded_citation() -> None:
    decision = _decision(third_party_locator="05-4208")
    reviewed = asyncio.run(body_corroboration_review(_document(), reviewer=FakeReviewer(decision)))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert "numeric parts" in reviewed.roots[0].body_reviews[-1].failure_reason


@pytest.mark.parametrize("field,replacement", [("court", "3d Cir."), ("date", "2008")])
def test_other_numeric_field_must_match_the_grounded_citation(field: str, replacement: str) -> None:
    decision = _decision()
    fabricated = decision.model_copy(
        update={"third_party": decision.third_party.model_copy(update={field: replacement})}
    )
    reviewed = asyncio.run(body_corroboration_review(_document(), reviewer=FakeReviewer(fabricated)))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert field in reviewed.roots[0].body_reviews[-1].failure_reason


def test_no_fetched_body_skips_model_and_preserves_a_deferred_result() -> None:
    document = _document(include_evidence=False)
    reviewed = asyncio.run(body_corroboration_review(document))
    root = reviewed.roots[0]
    assert root.body_reviews[-1].decision.source is None
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert json.loads(reviewed.model_dump_json())["stage_runs"][-1] == STAGE
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(body_corroboration_review(reviewed))


def test_review_completes_when_no_root_needs_body_evidence() -> None:
    document = Document.from_source("No citations here.").complete("10_roots")
    reviewed = asyncio.run(body_corroboration_review(document))
    assert reviewed.stage_runs[-1] == STAGE
    assert reviewed.citations == ()


def test_one_fetched_body_preserves_three_separate_occurrences() -> None:
    repeated = " // ".join((BODY, BODY, BODY, BODY))
    found = make_body_evidences(
        body_id="opinion:1",
        parent_id=None,
        url=None,
        issued_on=None,
        date_basis=None,
        metadata={},
        body_text=repeated,
        locator="05-4206",
        source_text=SOURCE,
    )
    assert len(found) == 3
    assert len({item.source_offset + item.anchor_span.start for item in found}) == 3
    assert all(item.anchor_kind == "locator" for item in found)


def test_source_filing_with_page_furniture_is_not_independent_evidence() -> None:
    found = make_body_evidences(
        body_id="filing-copy",
        parent_id=None,
        url=None,
        issued_on=None,
        date_basis=None,
        metadata={},
        body_text="Page 1\n" + SOURCE,
        locator="05-4206",
        source_text=SOURCE,
        case_name="Smith v. Jones",
    )
    assert found == ()


def test_saved_body_evidence_cannot_violate_retrospective_cutoff() -> None:
    evidence = _document().roots[0].body_searches[-1].evidence
    with pytest.raises(ValidationError, match="retrospective cutoff"):
        BodySearch(
            node_id="root:node:1",
            source=BodySource.GOVINFO_OPINION,
            retrospective_date=date(2005, 1, 1),
            evidence=evidence,
        )


def test_reporter_root_uses_same_body_review_without_changing_its_locator() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    locator = "550 U.S. 544"
    body = "A later opinion cited Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    root = FullReporterCitation.from_locator(
        citation_id="reporter:0",
        stage="sites",
        source=source,
        span=Span(source.index(locator), source.index(locator) + len(locator)),
    )
    document = Document.from_source(source).add_citation(root).complete("sites")
    root = root.record("fields")
    root = root.with_case_name(source, Span(0, len("Bell Atl. Corp. v. Twombly")))
    root = root.with_date(source, Span(source.index("2007"), source.index("2007") + 4))
    document = document.replace_citation(root).complete("fields")
    document = document.replace_citation(root.record("roots").with_root(root.id)).complete("roots")
    root = document.roots[0].record("22_govinfo_opinion_body_search")
    found = make_body_evidences(
        body_id="opinion:2",
        parent_id=None,
        url=None,
        issued_on=date(2008, 1, 1),
        date_basis="opinion.date_filed",
        metadata={},
        body_text=body,
        locator=locator,
        source_text=source,
        case_name="Bell Atl. Corp. v. Twombly",
    )
    root = root.with_body_search(
        BodySearch(
            node_id=root.nodes[-1].id,
            source=BodySource.GOVINFO_OPINION,
            retrospective_date=None,
            evidence=found,
        )
    )
    document = document.replace_citation(root).complete("22_govinfo_opinion_body_search")
    decision = BodyCorroborationDecision.model_validate(
        {
            "source": "govinfo_opinion",
            "evidence_index": 0,
            "citation_quote": "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)",
            "filing": {
                "locator": locator,
                "case_name": "Bell Atl. Corp. v. Twombly",
                "normalized_case_name": {
                    "kind": "adversarial",
                    "plaintiff": "Bell Atl. Corp.",
                    "defendant": "Twombly",
                    "subject": None,
                },
                "court": None,
                "date": "2007",
            },
            "third_party": {
                "locator": locator,
                "case_name": "Bell Atl. Corp. v. Twombly",
                "court": None,
                "date": "2007",
            },
            "comparisons": {
                field: {
                    "result": "unavailable" if field == "court" else "match",
                    "reason": "Compared the written citation fields.",
                }
                for field in ("locator", "case_name", "court", "date")
            },
            "reason": "The third-party opinion cites the same reported authority.",
        }
    )
    reviewed = asyncio.run(body_corroboration_review(document, reviewer=FakeReviewer(decision)))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert len(reviewed.roots[0].locator) == 1
    fabricated = decision.model_copy(
        update={"third_party": decision.third_party.model_copy(update={"locator": "550 U.S. 545"})}
    )
    rejected = asyncio.run(body_corroboration_review(document, reviewer=FakeReviewer(fabricated)))
    assert rejected.roots[0].identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert "numeric parts" in rejected.roots[0].body_reviews[-1].failure_reason


def test_review_shows_distinct_documents_before_repeated_citations() -> None:
    items = tuple(
        item
        for index in range(8)
        for item in make_body_evidences(
            body_id=f"opinion:{index}",
            parent_id=None,
            url=None,
            issued_on=None,
            date_basis=None,
            metadata={},
            body_text=" // ".join((BODY, BODY, BODY)),
            locator="05-4206",
            source_text=SOURCE,
        )
    )
    selected = _diverse_evidence(BodySource.COURTLISTENER_OPINION, items)
    assert len(selected) == 6
    assert len({item.body_id for _, _, item in selected}) == 6


def test_reviewer_context_serializes_anchor_offsets_for_the_model() -> None:
    document = _document()
    context = BodyCorroborationContext.from_document(document, document.roots[0])
    payload = json.loads(context.prompt_evidence())
    assert payload[0]["anchor_span"] == {"start": 40, "end": 47}
