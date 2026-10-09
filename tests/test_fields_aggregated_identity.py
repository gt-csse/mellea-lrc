"""Saved docket field assessments become durable identities without rereview."""

from __future__ import annotations

import pytest

from mellea_lrc.api import fields_aggregated_identity as public_aggregate
from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation, Span
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupCandidate,
    DocketLookupReview,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.govinfo_lookup import (
    GovInfoDocketLookup,
    GovInfoDocketReview,
    GovInfoLookupAttempt,
    GovInfoLookupCandidate,
)
from mellea_lrc.model.citations.judgments import IdentityBasis, IdentityVerdict
from mellea_lrc.validation.body_search.common import roots_for_body_search
from mellea_lrc.validation.fields_aggregated_identity import SUBSTAGE, fields_aggregated_identity

SOURCE = "Smith v. Jones, No. 05-4206 (2d Cir. 2007)."
CL_LOOKUP = "validate_roots.docket_lookup.courtlistener_retrieval"
CL_REVIEW = "validate_roots.docket_lookup.courtlistener_review"
GOV_LOOKUP = "validate_roots.docket_lookup.govinfo_retrieval"
GOV_REVIEW = "validate_roots.docket_lookup.govinfo_review"
BODY_SEARCH = "validate_roots.locator_body_corroboration.courtlistener_opinion_retrieval"
BODY_REVIEW = "validate_roots.locator_body_corroboration.llm_judgment"


def _decision(selected: int | None = 0, **results: str) -> DocketLookupReviewDecision:
    def field(name: str) -> dict[str, object]:
        return {
            "propose_replacement": False,
            "quote": None,
            "result": results.get(name, "match") if selected is not None else "unavailable",
            "reason": "Compared the filing with the saved selected record.",
        }

    return DocketLookupReviewDecision.model_validate(
        {
            "selected_candidate_index": selected,
            "docket_number": field("docket_number"),
            "case_name": {
                **field("case_name"),
                "normalized": {
                    "kind": "adversarial",
                    "plaintiff": "Smith",
                    "defendant": "Jones",
                    "subject": None,
                },
            },
            "court": field("court"),
            "date": field("date"),
            "reason": "Selected from saved lookup evidence." if selected is not None else "No selection.",
        }
    )


def _rooted() -> Document:
    number_start = SOURCE.index("05-4206")
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        substage="test_sites",
        source=SOURCE,
        span=Span(SOURCE.index("No."), number_start + len("05-4206")),
        number_span=Span(number_start, number_start + len("05-4206")),
    )
    document = Document.from_source(SOURCE).add_citation(root).complete_substage("test_sites")
    root = root.record("test_readings").with_case_name(SOURCE, Span(0, len("Smith v. Jones")))
    for field, quote in (("court", "2d Cir."), ("date", "2007")):
        span = Span(SOURCE.index(quote), SOURCE.index(quote) + len(quote))
        root = root.with_court(SOURCE, span) if field == "court" else root.with_date(SOURCE, span)
    document = document.replace_citation(root).complete_substage("test_readings")
    return document.replace_citation(
        root.record("grow_roots.root_formation.rule").with_root(root.id)
    ).complete_substage("grow_roots.root_formation.rule")


def _reviewed(
    *,
    cl: DocketLookupReviewDecision | str | None = None,
    gov: DocketLookupReviewDecision | str | None = None,
    pending: bool | None = None,
) -> Document:
    """Build native saved lookup/review nodes, following the existing docket fixtures."""
    cl = _decision() if cl is None else cl
    document = _rooted()
    root = document.roots[0].record(CL_LOOKUP)
    lookup = DocketLookup(
        node_id=root.nodes[-1].id,
        attempts=(
            DocketLookupAttempt(
                source_type="d",
                query="docketNumber:(05-4206)",
                pages=(
                    {
                        "results": [
                            {
                                "docket_id": 41,
                                "docketNumber": "05-4206",
                                "caseName": "Smith v. Jones",
                                "court_id": "ca2",
                                "dateFiled": "2005-01-01",
                            }
                        ]
                    },
                ),
            ),
        ),
        candidates=(
            DocketLookupCandidate(
                source_type="d",
                record_id="41",
                attempt_index=0,
                page_index=0,
                result_index=0,
                docket_number="05-4206",
                docket_similarity=100,
            ),
        ),
        shortlisted_candidate_indices=(0,),
    )
    document = document.replace_citation(root.with_docket_lookup(lookup)).complete_substage(CL_LOOKUP)
    root = document.roots[0].record(CL_REVIEW)
    review = DocketLookupReview(
        node_id=root.nodes[-1].id,
        decision=cl if isinstance(cl, DocketLookupReviewDecision) else None,
        failure_reason=cl if isinstance(cl, str) else None,
    )
    root = root.with_docket_lookup_review(review)
    if review.decision is not None and review.decision.selected_candidate_index is not None:
        root = root.with_route(SUBSTAGE)
    document = document.replace_citation(root).complete_substage(CL_REVIEW)

    if gov is None:
        document = document.complete_substage(GOV_LOOKUP).complete_substage(GOV_REVIEW)
    else:
        root = document.roots[0].record(GOV_LOOKUP)
        package = "USCOURTS-ca2-05-4206"
        lookup = GovInfoDocketLookup(
            node_id=root.nodes[-1].id,
            attempts=(
                GovInfoLookupAttempt(
                    query="collection:USCOURTS AND caseNumber:(05-4206)",
                    pages=({"results": [{"packageId": package, "title": "Smith v. Jones"}]},),
                ),
            ),
            candidates=(
                GovInfoLookupCandidate(
                    attempt_index=0,
                    page_index=0,
                    result_index=0,
                    package_id=package,
                    court_code="ca2",
                    docket_number="05-4206",
                    docket_similarity=100,
                ),
            ),
            shortlisted_candidate_indices=(0,),
        )
        document = document.replace_citation(root.with_govinfo_docket_lookup(lookup)).complete_substage(
            GOV_LOOKUP
        )
        root = document.roots[0].record(GOV_REVIEW)
        review = GovInfoDocketReview(
            node_id=root.nodes[-1].id,
            decision=gov if isinstance(gov, DocketLookupReviewDecision) else None,
            failure_reason=gov if isinstance(gov, str) else None,
        )
        root = root.with_govinfo_docket_review(review)
        if review.decision is not None and review.decision.selected_candidate_index is not None:
            root = root.with_route(SUBSTAGE)
        document = document.replace_citation(root).complete_substage(GOV_REVIEW)
    if pending is not None:
        root = document.roots[0].record("test_route").with_route(SUBSTAGE if pending else None)
        document = document.replace_citation(root).complete_substage("test_route")
    return document


def test_public_stage_records_identity_and_replays_native_checkpoint() -> None:
    assert SUBSTAGE == "validate_roots.docket_lookup.identity_aggregation"
    assert public_aggregate is fields_aggregated_identity
    before = _reviewed()
    assert roots_for_body_search(before) == ()

    after = public_aggregate(before)
    root = after.roots[0]
    (judgment,) = root.identity_judgments
    assert judgment.verdict is IdentityVerdict.CORRECT_IDENTITY
    assert judgment.basis is None
    assert judgment.node_id == root.nodes[-1].id
    assert root.nodes[-1].substage == SUBSTAGE
    assert root.routes[-1].node_id == judgment.node_id
    assert root.next_substage is None
    assert roots_for_body_search(after) == ()
    assert root.docket_lookup == before.roots[0].docket_lookup
    assert root.docket_lookup_review == before.roots[0].docket_lookup_review
    assert root.locator == before.roots[0].locator
    assert root.case_name == before.roots[0].case_name
    assert root.court == before.roots[0].court
    assert root.date == before.roots[0].date
    assert after.substage_runs == (*before.substage_runs, SUBSTAGE)
    assert after.get_substage(GOV_REVIEW) == before
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_substage(SUBSTAGE) == after
    assert restored.get_substage(GOV_REVIEW) == before


@pytest.mark.parametrize("field", ("docket_number", "case_name", "court", "date"))
def test_every_field_mismatch_is_a_wrong_identity(field: str) -> None:
    before = _reviewed(cl=_decision(**{field: "mismatch"}))
    after = fields_aggregated_identity(before)
    assert after.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert after.roots[0].next_substage is None
    assert roots_for_body_search(after) == ()


@pytest.mark.parametrize("field", ("docket_number", "court", "date"))
def test_mismatch_takes_precedence_over_unavailable_case_name(field: str) -> None:
    before = _reviewed(cl=_decision(case_name="unavailable", **{field: "mismatch"}))
    after = fields_aggregated_identity(before)
    assert after.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert after.roots[0].next_substage is None


@pytest.mark.parametrize(
    ("unavailable", "expected"),
    [
        (("docket_number",), IdentityVerdict.UNDETERMINED),
        (("case_name",), IdentityVerdict.UNDETERMINED),
        (("court",), IdentityVerdict.CORRECT_IDENTITY),
        (("date",), IdentityVerdict.CORRECT_IDENTITY),
        (("court", "date"), IdentityVerdict.CORRECT_IDENTITY),
        (("docket_number", "case_name", "court", "date"), IdentityVerdict.UNDETERMINED),
    ],
)
def test_identifying_fields_are_required_but_court_and_date_can_be_unavailable(
    unavailable: tuple[str, ...], expected: IdentityVerdict
) -> None:
    before = _reviewed(cl=_decision(**dict.fromkeys(unavailable, "unavailable")))
    after = fields_aggregated_identity(before)
    root = after.roots[0]
    assert root.identity_judgments[-1].verdict is expected
    assert root.identity_judgments[-1].basis is None
    if expected is IdentityVerdict.UNDETERMINED:
        assert root.next_substage == BODY_SEARCH
        assert roots_for_body_search(after) == after.roots
    else:
        assert root.next_substage is None
        assert roots_for_body_search(after) == ()


@pytest.mark.parametrize(
    ("cl", "gov", "expected"),
    [
        (_decision(case_name="mismatch"), _decision(), IdentityVerdict.CORRECT_IDENTITY),
        (_decision(), _decision(docket_number="mismatch"), IdentityVerdict.WRONG_IDENTITY),
    ],
)
def test_selected_govinfo_review_takes_precedence(
    cl: DocketLookupReviewDecision, gov: DocketLookupReviewDecision, expected: IdentityVerdict
) -> None:
    before = _reviewed(cl=cl, gov=gov)
    after = fields_aggregated_identity(before)
    assert after.roots[0].identity_judgments[-1].verdict is expected
    assert after.roots[0].docket_lookup_review == before.roots[0].docket_lookup_review
    assert after.roots[0].govinfo_docket_review == before.roots[0].govinfo_docket_review


@pytest.mark.parametrize("gov", (_decision(None), "GovInfo review failed"))
def test_unselected_or_failed_govinfo_falls_back_to_selected_courtlistener(
    gov: DocketLookupReviewDecision | str,
) -> None:
    before = _reviewed(cl=_decision(docket_number="mismatch"), gov=gov)
    after = fields_aggregated_identity(before)
    assert after.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY


@pytest.mark.parametrize("cl", (_decision(None), "CourtListener review failed"))
def test_failed_or_unselected_unqueued_review_is_untouched(cl: DocketLookupReviewDecision | str) -> None:
    before = _reviewed(cl=cl)
    after = fields_aggregated_identity(before)
    assert after.citations == before.citations
    assert after.roots[0].identity_judgments == ()
    assert after.substage_runs == (*before.substage_runs, SUBSTAGE)
    assert roots_for_body_search(after) == after.roots


@pytest.mark.parametrize(
    "route", (None, "validate_roots.intended_case_discovery.courtlistener_opinion_retrieval")
)
def test_selected_review_does_not_aggregate_without_its_pending_route(route: str | None) -> None:
    before = _reviewed(pending=False)
    if route is not None:
        root = before.roots[0].record("test_other_route").with_route(route)
        before = before.replace_citation(root).complete_substage("test_other_route")
    after = fields_aggregated_identity(before)
    assert after.citations == before.citations
    assert after.roots[0].identity_judgments == ()
    assert after.roots[0].next_substage == route


@pytest.mark.parametrize("cl", (_decision(None), "CourtListener review failed"))
def test_pending_route_requires_an_accepted_selected_review(cl: DocketLookupReviewDecision | str) -> None:
    before = _reviewed(cl=cl, pending=True)
    with pytest.raises(ValueError):
        fields_aggregated_identity(before)


def test_pending_route_with_an_existing_identity_raises() -> None:
    before = _reviewed()
    root = (
        before.roots[0].record("test_prior_identity").with_identity_judgment(IdentityVerdict.WRONG_IDENTITY)
    )
    before = before.replace_citation(root).complete_substage("test_prior_identity")
    with pytest.raises(ValueError):
        fields_aggregated_identity(before)


@pytest.mark.parametrize("verdict", tuple(IdentityVerdict))
def test_nonpending_later_body_identity_and_route_are_preserved(verdict: IdentityVerdict) -> None:
    before = _reviewed()
    route = (
        "validate_roots.intended_case_discovery.courtlistener_opinion_retrieval"
        if verdict
        in {
            IdentityVerdict.PARTIALLY_CORROBORATED,
            IdentityVerdict.UNDETERMINED,
        }
        else None
    )
    root = (
        before.roots[0]
        .record(BODY_REVIEW)
        .with_identity_judgment(verdict, basis=IdentityBasis.THIRD_PARTY)
        .with_route(route)
    )
    before = before.replace_citation(root).complete_substage(BODY_REVIEW)
    after = fields_aggregated_identity(before)
    assert after.citations == before.citations
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments
    assert after.roots[0].next_substage == route
    assert after.roots[0].routes == before.roots[0].routes
    assert after.get_substage(BODY_REVIEW) == before
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_queued_reporter_root_is_an_unsupported_type() -> None:
    source = "Smith v. Jones, 123 F.3d 456 (2d Cir. 2007)."
    quote = "123 F.3d 456"
    root = FullReporterCitation.from_locator(
        citation_id="reporter:0",
        substage="test_sites",
        source=source,
        span=Span(source.index(quote), source.index(quote) + len(quote)),
    )
    document = Document.from_source(source).add_citation(root).complete_substage("test_sites")
    document = document.replace_citation(
        root.record("grow_roots.root_formation.rule").with_root(root.id)
    ).complete_substage("grow_roots.root_formation.rule")
    root = document.roots[0].record(GOV_REVIEW).with_route(SUBSTAGE)
    before = document.replace_citation(root).complete_substage(GOV_REVIEW)
    with pytest.raises(ValueError):
        fields_aggregated_identity(before)


def test_completed_aggregation_stage_cannot_run_twice() -> None:
    after = fields_aggregated_identity(_reviewed())
    with pytest.raises(ValueError, match="already completed"):
        fields_aggregated_identity(after)
