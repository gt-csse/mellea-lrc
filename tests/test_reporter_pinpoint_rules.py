"""Offline evidence preparation and final judgments use occurrence histories."""

import pytest
from pydantic import ValidationError

from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind, PinCiteTarget
from mellea_lrc.model.citations.reporter_opinion import (
    OpinionRetrievalOutcome,
    ReporterRootOpinionRetrieval,
    ReporterRootOpinionSource,
    RetrievedReporterOpinion,
)
from mellea_lrc.model.citations.reporter_page_resolution import (
    OpinionPageReference,
    ReporterCitationOpinionDecision,
    ReporterCitationOpinionReview,
    ReporterCitationPageResolution,
    ReporterOpinionPageChoice,
    ReporterPageCandidates,
    ReporterPageResolutionOutcome,
)
from mellea_lrc.model.citations.reporter_pinpoint import (
    GroundedPassage,
    OpinionEvidenceQuote,
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointEvidenceOutcome,
    PinpointPageAssessment,
    PropositionDecision,
    ReporterCitationProposition,
    ReporterCitationSupportReview,
    ReporterOpinionEvidence,
    ReporterPinpointJudgment,
    ReporterPinpointVerdict,
    ReporterSupportDecision,
)
from mellea_lrc.providers.courtlistener.models import CourtListenerCluster
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import (
    SUBSTAGE as EVIDENCE_SUBSTAGE,
)
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import (
    prepare_reporter_citation_pinpoint_evidence,
)
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import (
    SOURCE_SUBSTAGE as FULL_SUBSTAGE,
)
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import (
    SUBSTAGE as JUDGMENT_SUBSTAGE,
)
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import (
    judge_reporter_citation_pinpoints,
)
from mellea_lrc.validation.reporter_root_opinion_page_index import index_reporter_root_opinion_pages

PAGE_SUBSTAGE = "validate_pincite.support_review.page_review"


def _page(first, last=None, *, kind=PinCiteKind.PAGE, footnote=None):
    return PinCiteTarget(first=first, last=first if last is None else last, kind=kind, footnote=footnote)


def _span(source, quote):
    start = source.index(quote)
    return Span(start, start + len(quote))


def _before_evidence(
    *,
    resolution="resolved",
    proposition="read",
    second_opinion="retrieved",
    unconfirmed=False,
    range_ready=False,
):
    pin = (
        "unknown"
        if resolution == "unnormalizable"
        else "556-557"
        if resolution == "unlocated" or range_ready
        else "556"
    )
    source = f"The rule permits relief. Alpha, 550 U.S. 544, {pin}."
    root = FullReporterCitation.from_locator(
        citation_id="root",
        substage="grow_roots.locator_discovery.full_reporter_locators",
        source=source,
        span=_span(source, "550 U.S. 544"),
    )
    document = (
        Document.from_source(source)
        .add_citation(root)
        .complete_substage("grow_roots.locator_discovery.full_reporter_locators")
    )
    root = root.record("grow_roots.root_formation.rule").with_root(root.id)
    if resolution != "no_pin":
        root = root.with_pin_cite(source, _span(source, pin))
    document = document.replace_citation(root).complete_substage("grow_roots.root_formation.rule")
    root = root.record("validate_pincite.opinion_preparation.retrieval")
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate({"id": 1, "sub_opinions": [20, 21]}),
        )
    )
    opinions = (
        RetrievedReporterOpinion(
            opinion_id="20",
            cluster_id="1",
            outcome=OpinionRetrievalOutcome.RETRIEVED,
            response={
                "id": 20,
                "cluster": 1,
                "html": (
                    '<page-number label="556" volume="550" edition="U.S.">556</page-number>'
                    "The rule permits relief. "
                    '<page-number label="557" volume="550" edition="U.S.">557</page-number>'
                    "Other grounds permit relief."
                ),
            },
        ),
        RetrievedReporterOpinion(
            opinion_id="21",
            cluster_id="1",
            outcome=OpinionRetrievalOutcome.RETRIEVED
            if second_opinion == "markup_only"
            else OpinionRetrievalOutcome(second_opinion),
            response=None
            if second_opinion == "not_found"
            else {"id": 21, "cluster": 1, "html": "<p></p>"}
            if second_opinion == "markup_only"
            else {
                "id": 21,
                "cluster": 1,
                "plain_text": "Second writing." if second_opinion == "retrieved" else "",
            },
        ),
    )
    root = root.with_reporter_root_opinion_retrieval(
        ReporterRootOpinionRetrieval(
            node_id=root.nodes[-1].id, cluster_id="1", sub_opinion_ids=("20", "21"), opinions=opinions
        )
    )
    document = document.replace_citation(root).complete_substage(
        "validate_pincite.opinion_preparation.retrieval"
    )
    document = index_reporter_root_opinion_pages(document)
    root = document.roots[0].record("validate_pincite.citation_preparation.page_resolution")
    requested = ()
    if resolution in {"resolved", "unlocated"}:
        requested = (
            ReporterPageCandidates(
                target_index=0,
                label=556,
                kind=PinCiteKind.PAGE,
                candidates=(
                    OpinionPageReference(opinion_id="20", page_index=0, pagination_confirmed=not unconfirmed),
                ),
            ),
        )
        if resolution == "unlocated":
            requested += (
                ReporterPageCandidates(target_index=0, label=557, kind=PinCiteKind.PAGE, candidates=()),
            )
        elif range_ready:
            requested += (
                ReporterPageCandidates(
                    target_index=0,
                    label=557,
                    kind=PinCiteKind.PAGE,
                    candidates=(
                        OpinionPageReference(opinion_id="20", page_index=1, pagination_confirmed=True),
                    ),
                ),
            )
    outcome = (
        ReporterPageResolutionOutcome.AMBIGUOUS if unconfirmed else ReporterPageResolutionOutcome(resolution)
    )
    root = root.with_reporter_page_resolution(
        ReporterCitationPageResolution(
            node_id=root.nodes[-1].id,
            root_id=root.id,
            locator_citation_id=root.id,
            locator_reading_index=0,
            pin_citation_id=None if resolution == "no_pin" else root.id,
            pin_reading_index=None if resolution == "no_pin" else 0,
            outcome=outcome,
            pages=requested,
            reason="Saved occurrence target.",
        )
    )
    document = document.replace_citation(root).complete_substage(
        "validate_pincite.citation_preparation.page_resolution"
    )
    if unconfirmed:
        root = document.roots[0].record("validate_pincite.citation_preparation.opinion_review")
        root = root.with_reporter_opinion_review(
            ReporterCitationOpinionReview(
                node_id=root.nodes[-1].id,
                resolution_index=0,
                decision=ReporterCitationOpinionDecision(
                    choices=(ReporterOpinionPageChoice(page_index=0, candidate_index=0),),
                    reason="Selected writing.",
                ),
            )
        )
        document = document.replace_citation(root)
    document = document.complete_substage("validate_pincite.citation_preparation.opinion_review")
    if proposition != "missing":
        root = document.roots[0].record("validate_pincite.citation_preparation.propositions")
        quote = "The rule permits relief."
        decision = (
            None
            if proposition == "failed"
            else PropositionDecision(
                quotes=() if proposition == "empty" else (quote,), reason="Read the attributed use."
            )
        )
        passages = (GroundedPassage(quote=quote, span=_span(source, quote)),) if proposition == "read" else ()
        root = root.with_reporter_proposition(
            ReporterCitationProposition(
                node_id=root.nodes[-1].id,
                resolution_index=0,
                decision=decision,
                passages=passages,
                failure_reason="Reader unavailable" if proposition == "failed" else None,
            )
        )
        document = document.replace_citation(root)
    return document.complete_substage("validate_pincite.citation_preparation.propositions")


def _append_review(
    document,
    *,
    substage,
    scope,
    result,
    quote=None,
    correct=None,
    pagination=True,
    found=None,
    failure=None,
    evidence_index=0,
    opinion_id="20",
):
    root = document.roots[0].record(substage)
    evidence_indices = ()
    quotes = ()
    if quote is not None:
        opinion = next(
            item for item in root.reporter_root_opinion_page_index.opinions if item.opinion_id == opinion_id
        )
        evidence_indices = (len(root.reporter_opinion_evidence),)
        root = root.with_reporter_opinion_evidence(
            ReporterOpinionEvidence(
                node_id=root.nodes[-1].id,
                root_id=root.id,
                opinion_id=opinion.opinion_id,
                quote=quote,
                span=_span(opinion.text, quote),
            )
        )
        quotes = (OpinionEvidenceQuote(opinion_id=opinion.opinion_id, quote=quote),)
    if found is None:
        label = {"The rule permits relief.": 556, "Other grounds permit relief.": 557}.get(quote)
        found = (_page(label),) if pagination and label is not None else ()
    root = root.with_reporter_support_review(
        ReporterCitationSupportReview(
            node_id=root.nodes[-1].id,
            evidence_index=evidence_index,
            scope=scope,
            decision=None
            if failure
            else ReporterSupportDecision(
                result=result,
                evidence=quotes,
                pagination_available=pagination,
                correct_page=correct,
                found_pages=found,
                reason="Saved support assessment.",
            ),
            opinion_evidence_indices=evidence_indices,
            failure_reason=failure,
        )
    )
    return document.replace_citation(root)


def _reviewed(
    *,
    scope=OpinionReviewScope.FULL_OPINION,
    result=OpinionSupportResult.SUPPORTED,
    quote="The rule permits relief.",
    correct=True,
    pagination=True,
    found=None,
    opinion_id="20",
    **fixture,
):
    document = prepare_reporter_citation_pinpoint_evidence(_before_evidence(**fixture))
    substage = PAGE_SUBSTAGE if scope is OpinionReviewScope.CITED_PAGES else FULL_SUBSTAGE
    if substage == FULL_SUBSTAGE:
        document = document.complete_substage(PAGE_SUBSTAGE)
    document = _append_review(
        document,
        substage=substage,
        scope=scope,
        result=result,
        quote=quote,
        correct=correct,
        pagination=pagination,
        found=found,
        opinion_id=opinion_id,
    )
    document = document.complete_substage(substage)
    return document.complete_substage(FULL_SUBSTAGE) if substage == PAGE_SUBSTAGE else document


@pytest.mark.parametrize(
    "fixture, outcome, available_pages",
    [
        ({}, PinpointEvidenceOutcome.READY, 1),
        ({"resolution": "no_pin", "proposition": "missing"}, PinpointEvidenceOutcome.NO_PINCITE, 0),
        ({"proposition": "empty"}, PinpointEvidenceOutcome.NO_PROPOSITION, 1),
        ({"proposition": "failed"}, PinpointEvidenceOutcome.READING_FAILED, 1),
        ({"proposition": "missing"}, PinpointEvidenceOutcome.READING_FAILED, 1),
        ({"resolution": "unnormalizable"}, PinpointEvidenceOutcome.UNNORMALIZABLE, 0),
        ({"resolution": "unlocated"}, PinpointEvidenceOutcome.MISSING_PAGES, 1),
    ],
)
def test_evidence_keeps_distinct_unavailable_states_and_available_selections(
    fixture, outcome, available_pages
):
    before = _before_evidence(**fixture)
    after = prepare_reporter_citation_pinpoint_evidence(before)
    evidence = after.roots[0].reporter_pinpoint_evidence[0]

    assert evidence.outcome is outcome
    assert evidence.resolution_index == 0 and evidence.root_id == "root"
    assert len(evidence.pages) == available_pages and evidence.reason
    assert after.roots[0].identity_judgments == ()
    saved = Document.model_validate_json(after.model_dump_json())
    assert (
        saved == after and saved.get_substage("validate_pincite.citation_preparation.propositions") == before
    )
    with pytest.raises(ValueError, match="already completed"):
        prepare_reporter_citation_pinpoint_evidence(saved)


@pytest.mark.parametrize(
    "missing_stage",
    [
        "validate_pincite.citation_preparation.opinion_review",
        "validate_pincite.citation_preparation.propositions",
    ],
)
def test_evidence_requires_both_selection_and_proposition_stages(missing_stage):
    document = Document.from_source("No citations.").complete_substage(
        "validate_pincite.citation_preparation.propositions"
        if missing_stage.startswith("42")
        else "validate_pincite.citation_preparation.opinion_review"
    )
    with pytest.raises(ValueError, match="Complete opinion selection"):
        prepare_reporter_citation_pinpoint_evidence(document)


@pytest.mark.parametrize(
    "scope, result, complete, expected",
    [
        (
            OpinionReviewScope.CITED_PAGES,
            OpinionSupportResult.NOT_FOUND,
            "retrieved",
            ReporterPinpointVerdict.UNDETERMINED,
        ),
        (
            OpinionReviewScope.CITED_PAGES,
            OpinionSupportResult.CONTRADICTED,
            "retrieved",
            ReporterPinpointVerdict.UNDETERMINED,
        ),
        (
            OpinionReviewScope.FULL_OPINION,
            OpinionSupportResult.CONTRADICTED,
            "not_found",
            ReporterPinpointVerdict.WRONG_PINCITE,
        ),
        (
            OpinionReviewScope.FULL_OPINION,
            OpinionSupportResult.NOT_FOUND,
            "retrieved",
            ReporterPinpointVerdict.WRONG_PINCITE,
        ),
        (
            OpinionReviewScope.FULL_OPINION,
            OpinionSupportResult.NOT_FOUND,
            "not_found",
            ReporterPinpointVerdict.UNDETERMINED,
        ),
        (
            OpinionReviewScope.FULL_OPINION,
            OpinionSupportResult.NOT_FOUND,
            "empty_text",
            ReporterPinpointVerdict.UNDETERMINED,
        ),
        (
            OpinionReviewScope.FULL_OPINION,
            OpinionSupportResult.NOT_FOUND,
            "markup_only",
            ReporterPinpointVerdict.UNDETERMINED,
        ),
        (
            OpinionReviewScope.FULL_OPINION,
            OpinionSupportResult.UNAVAILABLE,
            "retrieved",
            ReporterPinpointVerdict.UNDETERMINED,
        ),
    ],
)
def test_only_full_review_and_complete_text_can_prove_absence(scope, result, complete, expected):
    quote = "The rule permits relief." if result is OpinionSupportResult.CONTRADICTED else None
    before = _reviewed(scope=scope, result=result, quote=quote, correct=None, second_opinion=complete)
    after = judge_reporter_citation_pinpoints(before)
    judgment = after.roots[0].reporter_pinpoint_judgments[0]

    assert judgment.verdict is expected
    assert after.roots[0].identity_judgments == ()
    if result is OpinionSupportResult.NOT_FOUND and complete != "retrieved":
        assert "Missing or empty opinion text" in judgment.reason
    saved = Document.model_validate_json(after.model_dump_json())
    assert saved.get_substage(FULL_SUBSTAGE) == before
    with pytest.raises(ValueError, match="already completed"):
        judge_reporter_citation_pinpoints(saved)


@pytest.mark.parametrize(
    "quote, correct, pagination, found, unconfirmed",
    [
        ("The rule permits relief.", True, True, (_page(556),), False),
        ("Other grounds permit relief.", False, True, (_page(557),), False),
        ("Other grounds permit relief.", None, True, (_page(557),), False),
        ("Other grounds permit relief.", True, True, (_page(557),), False),
        ("The rule permits relief.", None, True, (_page(556),), True),
        ("The rule permits relief.", None, False, (), True),
    ],
)
def test_supported_attribution_preserves_explicit_page_assessment(
    quote, correct, pagination, found, unconfirmed
):
    document = _reviewed(
        quote=quote, correct=correct, pagination=pagination, found=found, unconfirmed=unconfirmed
    )
    judgment = judge_reporter_citation_pinpoints(document).roots[0].reporter_pinpoint_judgments[0]

    assert judgment.verdict is ReporterPinpointVerdict.CORRECT_PINCITE
    assert judgment.pagination_available is pagination
    assert judgment.correct_page is correct
    assert judgment.found_pages == found
    assert judgment.review_index == 0


def test_latest_successful_full_review_takes_precedence_over_page_and_failed_full_reviews():
    document = prepare_reporter_citation_pinpoint_evidence(_before_evidence())
    document = _append_review(
        document,
        substage=PAGE_SUBSTAGE,
        scope=OpinionReviewScope.CITED_PAGES,
        result=OpinionSupportResult.SUPPORTED,
        quote="The rule permits relief.",
        correct=True,
    )
    document = document.complete_substage(PAGE_SUBSTAGE)
    document = _append_review(
        document,
        substage=FULL_SUBSTAGE,
        scope=OpinionReviewScope.FULL_OPINION,
        result=OpinionSupportResult.NOT_FOUND,
    )
    document = _append_review(
        document,
        substage=FULL_SUBSTAGE,
        scope=OpinionReviewScope.FULL_OPINION,
        result=OpinionSupportResult.SUPPORTED,
        quote="Other grounds permit relief.",
        correct=False,
    )
    document = _append_review(
        document,
        substage=FULL_SUBSTAGE,
        scope=OpinionReviewScope.FULL_OPINION,
        result=OpinionSupportResult.UNAVAILABLE,
        failure="Transport failed",
    )
    document = document.complete_substage(FULL_SUBSTAGE)

    judgment = judge_reporter_citation_pinpoints(document).roots[0].reporter_pinpoint_judgments[0]

    assert judgment.review_index == 2
    assert judgment.verdict is ReporterPinpointVerdict.CORRECT_PINCITE
    assert judgment.pagination_available is True
    assert judgment.correct_page is False
    assert judgment.found_pages == (_page(557),)


def test_failed_full_review_preserves_successful_page_support():
    document = prepare_reporter_citation_pinpoint_evidence(_before_evidence())
    document = _append_review(
        document,
        substage=PAGE_SUBSTAGE,
        scope=OpinionReviewScope.CITED_PAGES,
        result=OpinionSupportResult.SUPPORTED,
        quote="The rule permits relief.",
        correct=True,
    )
    document = document.complete_substage(PAGE_SUBSTAGE)
    document = _append_review(
        document,
        substage=FULL_SUBSTAGE,
        scope=OpinionReviewScope.FULL_OPINION,
        result=OpinionSupportResult.UNAVAILABLE,
        failure="Transport failed",
    )
    judgment = (
        judge_reporter_citation_pinpoints(document.complete_substage(FULL_SUBSTAGE))
        .roots[0]
        .reporter_pinpoint_judgments[0]
    )

    assert judgment.review_index == 0 and judgment.verdict is ReporterPinpointVerdict.CORRECT_PINCITE


@pytest.mark.parametrize("proposition, count", [("empty", 0), ("missing", 1), ("failed", 1)])
def test_skip_no_proposition_but_keep_unavailable_judgments(proposition, count):
    document = prepare_reporter_citation_pinpoint_evidence(_before_evidence(proposition=proposition))
    document = document.complete_substage(PAGE_SUBSTAGE).complete_substage(FULL_SUBSTAGE)
    after = judge_reporter_citation_pinpoints(document)

    assert len(after.roots[0].reporter_pinpoint_judgments) == count
    if count:
        judgment = after.roots[0].reporter_pinpoint_judgments[0]
        assert judgment.review_index is None
        assert judgment.verdict is ReporterPinpointVerdict.UNDETERMINED
        assert "Evidence preparation" in judgment.reason


def test_judgment_requires_full_review_stage():
    with pytest.raises(ValueError, match="Complete full-opinion"):
        judge_reporter_citation_pinpoints(Document.from_source("No citations."))


def test_unpaginated_support_preserves_unknown_page_assessment():
    document = _reviewed(quote="Second writing.", opinion_id="21", correct=None, pagination=False)
    judgment = judge_reporter_citation_pinpoints(document).roots[0].reporter_pinpoint_judgments[0]

    assert judgment.verdict is ReporterPinpointVerdict.CORRECT_PINCITE
    assert judgment.pagination_available is False
    assert judgment.correct_page is None and judgment.found_pages == ()


def test_supported_footnote_mismatch_keeps_support_and_target_precision_separate():
    document = _reviewed(correct=False, found=(_page(556, footnote="3"),))
    judgment = judge_reporter_citation_pinpoints(document).roots[0].reporter_pinpoint_judgments[0]

    assert judgment.verdict is ReporterPinpointVerdict.CORRECT_PINCITE
    assert judgment.correct_page is False
    assert judgment.found_pages == (_page(556, footnote="3"),)


def test_grounded_quote_may_cross_contiguous_selected_pages():
    document = _reviewed(
        quote="The rule permits relief. Other grounds permit relief.",
        range_ready=True,
        found=(_page(556, 557),),
    )
    judgment = judge_reporter_citation_pinpoints(document).roots[0].reporter_pinpoint_judgments[0]

    assert judgment.verdict is ReporterPinpointVerdict.CORRECT_PINCITE
    assert judgment.correct_page is True
    assert judgment.found_pages == (_page(556, 557),)


def test_judgment_uses_latest_evidence_without_rewriting_earlier_preparation():
    document = prepare_reporter_citation_pinpoint_evidence(_before_evidence())
    original = document.roots[0].reporter_pinpoint_evidence[0]
    root = document.roots[0].record("test_new_preparation")
    root = root.with_reporter_pinpoint_evidence(original.model_copy(update={"node_id": root.nodes[-1].id}))
    document = document.replace_citation(root).complete_substage("test_new_preparation")
    document = _append_review(
        document,
        substage=PAGE_SUBSTAGE,
        scope=OpinionReviewScope.CITED_PAGES,
        result=OpinionSupportResult.SUPPORTED,
        quote="The rule permits relief.",
        correct=True,
        evidence_index=0,
    )
    document = document.complete_substage(PAGE_SUBSTAGE).complete_substage(FULL_SUBSTAGE)
    after = judge_reporter_citation_pinpoints(document)
    judgment = after.roots[0].reporter_pinpoint_judgments[0]

    assert len(after.roots[0].reporter_pinpoint_judgments) == 1
    assert after.roots[0].reporter_pinpoint_evidence[0] == original
    assert judgment.evidence_index == 1 and judgment.review_index is None
    assert judgment.verdict is ReporterPinpointVerdict.UNDETERMINED


def test_preparation_uses_latest_proposition_for_its_resolution():
    document = _before_evidence()
    root = document.roots[0].record("test_new_resolution")
    resolution = root.reporter_page_resolutions[0].model_copy(update={"node_id": root.nodes[-1].id})
    root = root.with_reporter_page_resolution(resolution)
    document = document.replace_citation(root).complete_substage("test_new_resolution")
    for substage, resolution_index, reason in (
        ("test_current_proposition", 1, "Current reader failed"),
        ("test_old_proposition", 0, "Old reader failed"),
    ):
        root = document.roots[0].record(substage)
        root = root.with_reporter_proposition(
            ReporterCitationProposition(
                node_id=root.nodes[-1].id,
                resolution_index=resolution_index,
                decision=None,
                failure_reason=reason,
            )
        )
        document = document.replace_citation(root).complete_substage(substage)

    evidence = prepare_reporter_citation_pinpoint_evidence(document).roots[0].reporter_pinpoint_evidence[0]

    assert evidence.resolution_index == 1 and evidence.proposition_index == 1
    assert evidence.outcome is PinpointEvidenceOutcome.READING_FAILED
    assert "Current reader failed" in evidence.reason


def test_loading_failed_support_review_with_accepted_evidence_rejects_malformed_document():
    import json

    payload = _reviewed().model_dump(mode="json")
    review = payload["citations"][0]["reporter_support_reviews"][0]
    assert review["opinion_evidence_indices"] == [0]
    review["decision"] = None
    review["failure_reason"] = "Review failed"

    with pytest.raises(
        ValidationError, match="Failed support review cannot retain accepted opinion evidence"
    ):
        Document.model_validate_json(json.dumps(payload))


def _assessment(model, **page_fields):
    if model is ReporterSupportDecision:
        return model(
            result=OpinionSupportResult.CONTRADICTED,
            evidence=(OpinionEvidenceQuote(opinion_id="20", quote="The rule permits relief."),),
            reason="The source at the written page contradicts the attributed rule.",
            **page_fields,
        )
    return model(
        node_id="test-node",
        evidence_index=0,
        review_index=None,
        verdict=ReporterPinpointVerdict.WRONG_PINCITE,
        reason="The source at the written page contradicts the attributed rule.",
        **page_fields,
    )


@pytest.mark.parametrize("model", [ReporterSupportDecision, ReporterPinpointJudgment])
def test_common_page_fields_are_required_flat_and_independent_of_content_verdict(model):
    assessment = _assessment(model, pagination_available=True, correct_page=True, found_pages=(_page(556),))
    assert isinstance(assessment, PinpointPageAssessment)
    payload = assessment.model_dump(mode="json")
    assert payload["pagination_available"] is True and payload["correct_page"] is True
    assert payload["found_pages"] == [{"first": 556, "last": 556, "kind": "page", "footnote": None}]
    assert (
        "page_assessment" not in payload and "location" not in payload and "target_supported" not in payload
    )
    assert model.model_validate_json(assessment.model_dump_json()) == assessment
    for field in ("pagination_available", "correct_page", "found_pages"):
        missing = dict(payload)
        del missing[field]
        with pytest.raises(ValidationError):
            model.model_validate(missing)
    for legacy in ("target_supported", "location"):
        with pytest.raises(ValidationError):
            model.model_validate({**payload, legacy: True})


@pytest.mark.parametrize("model", [ReporterSupportDecision, ReporterPinpointJudgment])
@pytest.mark.parametrize(
    "field,value",
    [
        ("pagination_available", 1),
        ("pagination_available", "false"),
        ("correct_page", 0),
        ("correct_page", "true"),
    ],
)
def test_page_assessment_booleans_do_not_coerce_strings_or_numbers(model, field, value):
    fields = {"pagination_available": True, "correct_page": None, "found_pages": ()}
    fields[field] = value
    with pytest.raises(ValidationError):
        _assessment(model, **fields)


@pytest.mark.parametrize("model", [ReporterSupportDecision, ReporterPinpointJudgment])
@pytest.mark.parametrize("correct,found", [(True, ()), (False, ()), (None, (_page(556),))])
def test_missing_pagination_cannot_assert_a_page_or_recovered_pages(model, correct, found):
    with pytest.raises(ValidationError):
        _assessment(model, pagination_available=False, correct_page=correct, found_pages=found)
    valid = _assessment(model, pagination_available=False, correct_page=None, found_pages=())
    assert valid.pagination_available is False and valid.correct_page is None and valid.found_pages == ()


@pytest.mark.parametrize("correct", [True, False, None])
def test_contradicted_use_can_preserve_each_explicit_page_assessment(correct):
    document = _reviewed(result=OpinionSupportResult.CONTRADICTED, correct=correct)
    judgment = judge_reporter_citation_pinpoints(document).roots[0].reporter_pinpoint_judgments[0]
    assert judgment.verdict is ReporterPinpointVerdict.WRONG_PINCITE
    assert judgment.pagination_available is True
    assert judgment.correct_page is correct
    assert judgment.found_pages == (_page(556),)
