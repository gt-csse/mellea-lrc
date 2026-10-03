"""Page and full-opinion reviews respect evidence scope and durable histories."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind, PinCiteTarget
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_opinion import ReporterRootOpinionSource
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionEvidenceQuote,
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointEvidenceOutcome,
    PropositionDecision,
    ReporterSupportDecision,
)
from mellea_lrc.model.citations.short_reporter import ShortReporterCitation
from mellea_lrc.providers.courtlistener.models import CourtListenerCluster
from mellea_lrc.validation.reporter_citation_full_opinion_review import (
    STAGE as FULL_STAGE,
)
from mellea_lrc.validation.reporter_citation_full_opinion_review import (
    review_reporter_citation_full_opinions,
)
from mellea_lrc.validation.reporter_citation_page_resolution import resolve_reporter_citation_pages
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import (
    STAGE as EVIDENCE_STAGE,
)
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import (
    prepare_reporter_citation_pinpoint_evidence,
)
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import (
    STAGE as PAGE_STAGE,
)
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import (
    review_reporter_citation_pinpoint_pages,
)
from mellea_lrc.validation.reporter_citation_propositions import read_reporter_citation_propositions
from mellea_lrc.validation.reporter_pinpoint_review import reviewer as service
from mellea_lrc.validation.reporter_root_opinion_page_index import index_reporter_root_opinion_pages
from mellea_lrc.validation.reporter_root_opinion_retrieval import reporter_root_opinion_retrieval
from tests.test_reporter_citation_propositions import _run

FIRST_PROPOSITION = "The pleading must provide fair notice."
SECOND_PROPOSITION = "A limited exception remains available."
PRELUDE = "Unpaginated introductory material."
FIRST_PAGE = "The lead opinion requires fair notice before relief may be granted."
SECOND_PAGE = "The lead opinion allows a limited exception after adequate safeguards."
DISSENT_PAGE = "The dissent rejects the notice requirement and would grant relief."


def _span(source, quote):
    start = source.index(quote)
    return Span(start, start + len(quote))


class Client:
    def __init__(self, opinions):
        self.opinions = opinions
        self.calls = []

    def get_opinion(self, opinion_id):
        self.calls.append(opinion_id)
        return self.opinions[opinion_id]


def _ready(
    *,
    include_leaf=False,
    root_pin="556",
    empty=False,
    missing_dissent=False,
    combined=False,
    unrenderable_dissent=False,
    unpaginated=False,
    inline_pagination=False,
    unknown_kind_pagination=False,
    leaf_locator="550 U.S. at 557",
    leaf_pin="557",
):
    source = f"{FIRST_PROPOSITION} Alpha, 550 U.S. 544, {root_pin} (2007)."
    if include_leaf:
        source += f" {SECOND_PROPOSITION} Alpha, {leaf_locator}."
    document = Document.from_source(source)
    root = FullReporterCitation.from_locator(
        citation_id="root",
        stage="1_full_reporter_locators",
        source=source,
        span=_span(source, "550 U.S. 544"),
    )
    document = document.add_citation(root).complete("1_full_reporter_locators")
    root = root.record("10_roots").with_root(root.id).with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    root = root.with_pin_cite(source, _span(source, root_pin))
    document = document.replace_citation(root).complete("10_roots")
    if include_leaf:
        leaf = (
            ShortReporterCitation.from_short_locator(
                citation_id="leaf",
                stage="test_leaf",
                source=source,
                span=_span(source, leaf_locator),
                pin_cite_span=_span(source, leaf_pin),
            )
            .record("test_leaf")
            .with_root(root.id)
        )
        document = document.add_citation(leaf).complete("test_leaf")
    root = document.roots[0].record("test_opinion_source")
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate({"id": 1, "sub_opinions": [20, 21]}),
        )
    )
    document = document.replace_citation(root).complete("test_opinion_source")
    marker = '<page-number label="{label}" volume="550" edition="U.S.">{label}</page-number>'
    if unknown_kind_pagination:
        marker = '<span class="star-pagination" volume="550" edition="U.S.">*{label}</span>'
    first_html = PRELUDE + marker.format(label="556") + FIRST_PAGE + marker.format(label="557") + SECOND_PAGE
    dissent_html = marker.format(label="570") + DISSENT_PAGE
    if unpaginated:
        first_html = PRELUDE + FIRST_PAGE + SECOND_PAGE
        dissent_html = DISSENT_PAGE
    if inline_pagination:
        first_html = PRELUDE + " *556 " + FIRST_PAGE + " *557 " + SECOND_PAGE
        dissent_html = "*570 " + DISSENT_PAGE
    if unrenderable_dissent:
        dissent_html = "<script>Unavailable rendered opinion</script>"
    client = Client(
        {
            "20": {
                "id": 20,
                "cluster": 1,
                "type": "010combined" if combined else "020lead",
                "author_str": "Justice A",
                "html_with_citations": "" if empty else first_html,
            },
            "21": None
            if missing_dissent
            else {
                "id": 21,
                "cluster": 1,
                "type": "040dissent",
                "author_str": "Justice B",
                "html_with_citations": "" if empty else dissent_html,
            },
        }
    )
    document = reporter_root_opinion_retrieval(document, client=client)
    document = index_reporter_root_opinion_pages(document)
    document = resolve_reporter_citation_pages(document).complete("42_reporter_citation_opinion_review")

    async def proposition(context):
        return PropositionDecision(
            quotes=(FIRST_PROPOSITION if context.citation_id == "root" else SECOND_PROPOSITION,),
            reason="The filing attributes this passage to this occurrence.",
        )

    document = asyncio.run(read_reporter_citation_propositions(document, reviewer=proposition))
    return prepare_reporter_citation_pinpoint_evidence(document)


def _page(first, last=None, *, kind=PinCiteKind.PAGE, footnote=None):
    return PinCiteTarget(first=first, last=first if last is None else last, kind=kind, footnote=footnote)


def _decision(
    result=OpinionSupportResult.SUPPORTED,
    *,
    quote=FIRST_PAGE,
    opinion_id="20",
    correct=True,
    pagination=True,
    found=None,
):
    if found is None:
        label = {FIRST_PAGE: 556, SECOND_PAGE: 557, DISSENT_PAGE: 570}.get(quote)
        found = (_page(label),) if pagination and label is not None else ()
    return ReporterSupportDecision(
        result=result,
        evidence=() if quote is None else (OpinionEvidenceQuote(opinion_id=opinion_id, quote=quote),),
        pagination_available=pagination,
        correct_page=correct,
        found_pages=found,
        reason="The supplied text establishes this assessment of the filing's attributed use.",
    )


class Reviewer:
    def __init__(self, responses):
        self.responses = responses
        self.contexts = []

    async def __call__(self, context):
        self.contexts.append(context)
        return self.responses[context.citation_id]


def test_page_prefix_exposes_only_selected_pages_and_writing_metadata():
    document = _ready(combined=True)
    context = service.ReporterPinpointReviewContext.from_document(
        document, document.roots[0], OpinionReviewScope.CITED_PAGES
    )
    bundle = json.loads(context.prefix.split("Shared saved opinion sources:\n", 1)[1])

    assert FIRST_PAGE in context.prefix
    assert SECOND_PAGE not in context.prefix
    assert DISSENT_PAGE not in context.prefix
    assert PRELUDE not in context.prefix
    assert [(item["opinion_id"], item["type"]) for item in bundle["opinions"]] == [("20", "010combined")]
    assert context.excerpts[0].offset > 0
    assert "combined opinion" in context.prefix


def test_full_opinion_prefix_is_identical_across_occurrences_and_dynamic_propositions():
    document = _ready(include_leaf=True)
    root, leaf = document.citations
    first = service.ReporterPinpointReviewContext.from_document(
        document, root, OpinionReviewScope.FULL_OPINION
    )
    second = service.ReporterPinpointReviewContext.from_document(
        document, leaf, OpinionReviewScope.FULL_OPINION
    )

    assert first.prefix == second.prefix
    assert FIRST_PAGE in first.prefix and SECOND_PAGE in first.prefix and DISSENT_PAGE in first.prefix
    assert first.proposition != second.proposition
    assert first.target != second.target
    assert FIRST_PROPOSITION not in first.prefix and SECOND_PROPOSITION not in first.prefix
    assert FIRST_PROPOSITION in first.proposition and SECOND_PROPOSITION in second.proposition


def test_full_opinion_prefix_is_byte_identical_for_parallel_reporter_occurrences():
    document = _ready(include_leaf=True, leaf_locator="127 S. Ct. at 1955", leaf_pin="1955")
    root, leaf = document.citations
    first = service.ReporterPinpointReviewContext.from_document(
        document, root, OpinionReviewScope.FULL_OPINION
    )
    second = service.ReporterPinpointReviewContext.from_document(
        document, leaf, OpinionReviewScope.FULL_OPINION
    )

    assert first.root_id == second.root_id
    assert first.prefix.encode("utf-8") == second.prefix.encode("utf-8")
    assert json.loads(first.target)["reporter"] == {"volume": 550, "edition": "U.S."}
    assert json.loads(second.target)["reporter"] == {"volume": 127, "edition": "S. Ct."}
    assert first.proposition != second.proposition
    assert any(excerpt.pages for excerpt in first.excerpts)
    assert all(not excerpt.pages for excerpt in second.excerpts)
    bundle = json.loads(first.prefix.split("Shared saved opinion sources:\n", 1)[1])
    assert bundle["opinions"][0]["page_markers"]
    assert "cited_reporter_pages" not in bundle["opinions"][0]


def test_ivr_fallback_preserves_each_attribution_and_shared_full_source_prefix(monkeypatch):
    before = _ready(include_leaf=True)
    captured = []

    async def fake_run(session, spec, *, strategy, model_options):
        captured.append(spec)
        if spec.user_variables["scope"] == OpinionReviewScope.CITED_PAGES.value:
            decision = _decision(OpinionSupportResult.UNAVAILABLE, quote=None, correct=None)
        else:
            quote = FIRST_PAGE if FIRST_PROPOSITION in spec.user_variables["proposition"] else SECOND_PAGE
            decision = _decision(quote=quote)
        return _run(
            decision.model_dump_json(),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={})
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=reviewer))
    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=reviewer))
    page_requests = [spec for spec in captured if spec.user_variables["scope"] == "cited_pages"]
    full_requests = [spec for spec in captured if spec.user_variables["scope"] == "full_opinion"]

    assert len(page_requests) == len(full_requests) == 2
    assert full_requests[0].prefix == full_requests[1].prefix
    assert page_requests[0].prefix != full_requests[0].prefix
    assert page_requests[0].prefix != page_requests[1].prefix
    assert full_requests[0].user_variables["proposition"] != full_requests[1].user_variables["proposition"]
    assert full_requests[0].user_variables["target"] != full_requests[1].user_variables["target"]
    instruction = captured[0].description
    assert all(spec.description == instruction for spec in captured)
    assert (
        "Assess every material assertion and qualification actually attributed to this citation"
        in instruction
    )
    assert "do not approve only the matching part" in instruction
    assert "Full-opinion fallback expands the source evidence" in instruction
    for page, full in zip(page_requests, full_requests, strict=True):
        for field in ("citation_quote", "citing_context", "proposition", "target"):
            assert page.user_variables[field] == full.user_variables[field]
        assert "Assess every material assertion" not in full.prefix
        assert FIRST_PROPOSITION not in full.prefix and SECOND_PROPOSITION not in full.prefix
    for original, reviewed in zip(before.citations, after.citations, strict=True):
        assert reviewed.reporter_propositions == original.reporter_propositions
        assert [review.evidence_index for review in reviewed.reporter_support_reviews] == [0, 0]
        assert reviewed.reporter_support_reviews[-1].decision.result is OpinionSupportResult.SUPPORTED


def test_supported_target_exits_after_page_review_without_initializing_full_reviewer(monkeypatch):
    before = _ready()
    page_reviewer = Reviewer({"root": _decision()})
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=page_reviewer))

    def fail(_profile):
        pytest.fail("Supported target must not initialize a full-opinion reviewer")

    monkeypatch.setattr(service.IvrReporterPinpointReviewer, "from_profile", fail)
    after = asyncio.run(review_reporter_citation_full_opinions(pages))

    assert len(page_reviewer.contexts) == 1
    assert len(after.roots[0].reporter_support_reviews) == 1
    assert after.roots[0].reporter_support_reviews[0].scope is OpinionReviewScope.CITED_PAGES
    assert FULL_STAGE in after.stage_runs


@pytest.mark.parametrize(
    "page_result",
    [
        _decision(OpinionSupportResult.UNAVAILABLE, quote=None, correct=None),
        _decision(OpinionSupportResult.CONTRADICTED, correct=True),
        _decision(correct=False),
        service.ReporterPinpointReviewOutcome(None, failure_reason="IVR repair exhausted"),
    ],
)
def test_uncertain_negative_wrong_target_and_failed_pages_call_full_opinion(page_result):
    before = _ready()
    pages = asyncio.run(
        review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({"root": page_result}))
    )
    full = Reviewer({"root": _decision(quote=SECOND_PAGE, correct=False)})

    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=full))

    assert len(full.contexts) == 1
    assert full.contexts[0].scope is OpinionReviewScope.FULL_OPINION
    assert after.roots[0].reporter_support_reviews[-1].scope is OpinionReviewScope.FULL_OPINION
    assert after.roots[0].reporter_opinion_evidence[-1].quote == SECOND_PAGE


def test_missing_page_routes_directly_to_full_opinion_and_reports_missing_writing():
    before = _ready(root_pin="599", missing_dissent=True)
    assert before.roots[0].reporter_pinpoint_evidence[-1].outcome is PinpointEvidenceOutcome.MISSING_PAGES
    page = Reviewer({})
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=page))
    full = Reviewer({"root": _decision(correct=False)})

    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=full))

    assert page.contexts == []
    assert full.contexts[0].source_complete is False
    assert full.contexts[0].scope is OpinionReviewScope.FULL_OPINION
    assert len(after.roots[0].reporter_support_reviews) == 1


def test_source_completeness_requires_usable_rendered_text_for_every_writing():
    before = _ready(unrenderable_dissent=True)
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )

    assert before.roots[0].reporter_root_opinion_retrieval.opinions[1].text_field is not None
    assert before.roots[0].reporter_root_opinion_page_index.opinions[1].text == ""
    assert context.source_complete is False
    assert {excerpt.opinion_id for excerpt in context.excerpts} == {"20"}


def test_empty_opinion_bundle_records_failure_without_a_model_call(monkeypatch):
    before = _ready(empty=True)
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({})))

    def fail(_profile):
        pytest.fail("Empty source must not initialize an LLM")

    monkeypatch.setattr(service.IvrReporterPinpointReviewer, "from_profile", fail)
    after = asyncio.run(review_reporter_citation_full_opinions(pages))
    review = after.roots[0].reporter_support_reviews[-1]

    assert review.decision is None and review.ivr is None
    assert review.opinion_evidence_indices == ()
    assert review.failure_reason == "The saved root opinion bundle contains no nonempty text."
    assert Document.model_validate_json(after.model_dump_json()).get_stage(PAGE_STAGE) == pages


@pytest.mark.parametrize(
    "bad",
    [
        _decision(opinion_id="999"),
        _decision(quote=SECOND_PAGE),
        _decision(quote=DISSENT_PAGE, opinion_id="21"),
    ],
)
def test_page_review_rejects_unknown_opinion_and_quotes_outside_allowed_page(bad):
    before = _ready()
    after = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({"root": bad})))

    review = after.roots[0].reporter_support_reviews[-1]
    assert review.decision is None
    assert "not grounded inside a supplied excerpt" in review.failure_reason
    assert review.opinion_evidence_indices == ()
    assert after.roots[0].reporter_opinion_evidence == ()


def test_page_absence_requires_ivr_repair_to_unavailable():
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.CITED_PAGES
    )
    bad = _decision(OpinionSupportResult.NOT_FOUND, quote=None, correct=None)
    result = service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad.model_dump_json())), context
    )

    assert not result.as_bool()
    assert (
        result.reason
        == "Selected pages cannot establish absence from the full opinion. Return unavailable for a page result that needs full-opinion review."
    )
    good = _decision(OpinionSupportResult.UNAVAILABLE, quote=None, correct=None)
    assert service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=good.model_dump_json())), context
    ).as_bool()
    full = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    assert service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad.model_dump_json())), full
    ).as_bool()


def test_evidence_spans_history_indexes_native_rewind_and_identity_are_preserved():
    before = _ready()
    pages = asyncio.run(
        review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({"root": _decision(correct=False)}))
    )
    after = asyncio.run(
        review_reporter_citation_full_opinions(
            pages, reviewer=Reviewer({"root": _decision(quote=SECOND_PAGE, correct=False)})
        )
    )
    root = after.roots[0]
    indexed = root.reporter_root_opinion_page_index.opinions[0].text

    assert [review.opinion_evidence_indices for review in root.reporter_support_reviews] == [(0,), (1,)]
    assert all(review.evidence_index == 0 for review in root.reporter_support_reviews)
    for passage, quote in zip(root.reporter_opinion_evidence, (FIRST_PAGE, SECOND_PAGE)):
        assert passage.root_id == root.id and passage.opinion_id == "20"
        assert passage.span == _span(indexed, quote)
        passage.validate_source(indexed)
    assert root.root_id == before.roots[0].root_id
    assert root.locator == before.roots[0].locator and root.pin_cite == before.roots[0].pin_cite
    assert root.identity_judgments == before.roots[0].identity_judgments
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored.get_stage(EVIDENCE_STAGE) == before
    assert restored.get_stage(PAGE_STAGE) == pages
    assert restored.get_stage(FULL_STAGE) == after


def test_prior_supported_page_review_does_not_skip_new_evidence_history():
    before = _ready()
    pages = asyncio.run(
        review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({"root": _decision()}))
    )
    root = pages.roots[0].record("test_refresh_evidence")
    refreshed = root.reporter_pinpoint_evidence[-1].model_copy(update={"node_id": root.nodes[-1].id})
    pages = pages.replace_citation(root.with_reporter_pinpoint_evidence(refreshed)).complete(
        "test_refresh_evidence"
    )
    full = Reviewer({"root": _decision(quote=SECOND_PAGE, correct=False)})

    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=full))

    assert len(full.contexts) == 1
    assert full.contexts[0].evidence_index == 1
    assert after.roots[0].reporter_support_reviews[-1].evidence_index == 1
    assert after.roots[0].reporter_support_reviews[-1].opinion_evidence_indices == (1,)


def test_shared_ivr_repairs_scope_and_grounding_and_saves_dynamic_context(monkeypatch):
    before = _ready()
    bad = _decision(quote=SECOND_PAGE)
    good = _decision()
    captured = {}

    async def fake_run(session, spec, *, strategy, model_options):
        captured["spec"] = spec
        validate = spec.requirements[0].validation_fn
        assert validate(SimpleNamespace(last_output=lambda: SimpleNamespace(value="{}"))).as_bool()
        rejected = validate(SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad.model_dump_json())))
        assert not rejected.as_bool() and "supplied excerpt" in rejected.reason
        assert validate(
            SimpleNamespace(last_output=lambda: SimpleNamespace(value=good.model_dump_json()))
        ).as_bool()
        initial = _run(
            bad.model_dump_json(),
            success=False,
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )
        final = _run(
            good.model_dump_json(),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )
        return final.model_copy(
            update={
                "attempts": (*initial.attempts, *final.attempts),
                "selected_attempt": 1,
                "output_schema": ReporterSupportDecision.model_json_schema(),
            }
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={"max_tokens": 6000})
    after = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=reviewer))
    review = after.roots[0].reporter_support_reviews[-1]

    assert review.decision == good
    assert review.ivr.attempts[0].output == bad.model_dump_json()
    assert review.ivr.attempts[1].output == good.model_dump_json()
    assert review.ivr.selected_attempt == 1
    assert captured["spec"].user_variables["scope"] == OpinionReviewScope.CITED_PAGES.value
    assert FIRST_PROPOSITION in captured["spec"].user_variables["proposition"]
    assert FIRST_PROPOSITION not in captured["spec"].prefix
    assert (
        Document.model_validate_json(after.model_dump_json()).roots[0].reporter_support_reviews[-1].ivr
        == review.ivr
    )


def test_failed_ivr_or_successful_but_ungrounded_output_retains_its_trace(monkeypatch):
    before = _ready()
    bad = _decision(quote=SECOND_PAGE)

    async def fake_run(session, spec, *, strategy, model_options):
        return _run(
            bad.model_dump_json(),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={})
    after = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=reviewer))
    review = after.roots[0].reporter_support_reviews[-1]

    assert review.decision is None
    assert "not grounded" in review.failure_reason
    assert review.ivr.output == bad.model_dump_json()
    assert review.opinion_evidence_indices == ()


def test_footnote_qualifier_remains_in_target_and_unconfirmed_footnote_routes_to_full():
    before = _ready(root_pin="556 n.2")
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.CITED_PAGES
    )
    target = json.loads(context.target)
    assert target["quote"] == "556 n.2"
    assert target["normalized"][0]["footnote"] == "2"
    pages = asyncio.run(
        review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({"root": _decision(correct=None)}))
    )
    full = Reviewer({"root": _decision(correct=None)})

    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=full))

    assert len(full.contexts) == 1
    decision = after.roots[0].reporter_support_reviews[-1].decision
    assert decision.pagination_available is True
    assert decision.correct_page is None
    assert decision.found_pages == (_page(556),)


@pytest.mark.parametrize(
    "corruption", ["unknown_opinion", "wrong_source_span", "wrong_declared_opinion", "missing_evidence_index"]
)
def test_native_loading_rejects_corrupted_opinion_source_and_evidence_pointers(corruption):
    before = _ready()
    pages = asyncio.run(
        review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({"root": _decision()}))
    )
    data = pages.model_dump(mode="json")
    root = data["citations"][0]
    evidence = root["reporter_opinion_evidence"][0]
    review = root["reporter_support_reviews"][0]
    if corruption == "unknown_opinion":
        evidence["opinion_id"] = "999"
    elif corruption == "wrong_source_span":
        evidence["span"]["start"] += 1
        evidence["span"]["end"] += 1
    elif corruption == "wrong_declared_opinion":
        review["decision"]["evidence"][0]["opinion_id"] = "21"
    else:
        review["opinion_evidence_indices"] = [99]

    with pytest.raises(ValueError):
        Document.model_validate(data)


def test_duplicate_and_missing_stage_guards_precede_model_calls():
    before = _ready()
    reviewer = Reviewer({"root": _decision()})
    with pytest.raises(ValueError, match="Prepare reporter pinpoint evidence"):
        asyncio.run(
            review_reporter_citation_pinpoint_pages(
                before.get_stage("43_reporter_citation_propositions"), reviewer=reviewer
            )
        )
    with pytest.raises(ValueError, match="Complete page review"):
        asyncio.run(review_reporter_citation_full_opinions(before, reviewer=reviewer))
    assert reviewer.contexts == []
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=reviewer))
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(review_reporter_citation_pinpoint_pages(pages, reviewer=reviewer))
    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=Reviewer({})))
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(review_reporter_citation_full_opinions(after, reviewer=Reviewer({})))


@pytest.mark.parametrize("found", [(_page(557),), (_page(599),), (_page(556, 557),)])
def test_indexed_quote_cannot_claim_a_different_or_unseen_found_page(found):
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    bad = _decision(found=found)
    validation = service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad.model_dump_json())), context
    )
    assert not validation.as_bool()
    assert "opinion_id=20" in validation.reason
    assert "source_span=" in validation.reason
    assert "550 U.S. page 556" in validation.reason
    assert "finding a passage elsewhere alone does not prove the target is wrong" in validation.reason


def test_declared_page_repair_retains_trace_and_accepts_grounded_alternative(monkeypatch):
    before = _ready(root_pin="599")
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({})))
    bad = _decision(quote=SECOND_PAGE, correct=False, found=(_page(556),))
    good = _decision(quote=SECOND_PAGE, correct=False, found=(_page(557),))

    async def fake_run(session, spec, *, strategy, model_options):
        validate = spec.requirements[0].validation_fn
        assert not validate(
            SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad.model_dump_json()))
        ).as_bool()
        assert validate(
            SimpleNamespace(last_output=lambda: SimpleNamespace(value=good.model_dump_json()))
        ).as_bool()
        initial = _run(bad.model_dump_json(), success=False)
        repaired = _run(good.model_dump_json())
        return repaired.model_copy(
            update={"attempts": (*initial.attempts, *repaired.attempts), "selected_attempt": 1}
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={})
    reviewed = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=reviewer))
    review = reviewed.roots[0].reporter_support_reviews[-1]
    assert review.decision == good
    assert review.ivr.attempts[0].output == bad.model_dump_json()
    assert review.ivr.attempts[1].output == good.model_dump_json()
    assert reviewed.roots[0].reporter_opinion_evidence[-1].quote == SECOND_PAGE


def test_unpaginated_full_opinion_can_settle_support_without_a_page_negative():
    before = _ready(unpaginated=True)
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({})))
    decision = _decision(pagination=False, correct=None)
    full = Reviewer({"root": decision})
    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=full))
    review = after.roots[0].reporter_support_reviews[-1]
    assert review.decision == decision
    assert decision.result is OpinionSupportResult.SUPPORTED
    assert decision.pagination_available is False
    assert decision.correct_page is None and decision.found_pages == ()
    assert after.roots[0].reporter_opinion_evidence[-1].quote == FIRST_PAGE
    assert full.contexts[0].source_complete is True


def test_correct_page_requires_a_relevant_quote_at_a_confirmed_written_target():
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    decision = _decision(quote=SECOND_PAGE, correct=True)
    result = service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=decision.model_dump_json())), context
    )
    assert not result.as_bool()
    assert "correct_page=true needs a quoted relevant passage at the written target" in result.reason
    assert "550 U.S. page 557" in result.reason


def test_unindexed_inline_page_markers_do_not_force_an_unknown_page_assessment():
    before = _ready(inline_pagination=True)
    assert all(not opinion.pages for opinion in before.roots[0].reporter_root_opinion_page_index.opinions)
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    decision = _decision()
    assert "*556" in context.prefix
    result = service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=decision.model_dump_json())), context
    )
    assert result.as_bool()
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({})))
    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=Reviewer({"root": decision})))
    retained = after.roots[0].reporter_support_reviews[-1].decision
    assert retained.pagination_available is True and retained.correct_page is True
    assert retained.found_pages == (_page(556),)


def test_unknown_kind_source_marker_remains_a_semantic_page_assessment():
    before = _ready(unknown_kind_pagination=True)
    indexed = before.roots[0].reporter_root_opinion_page_index.opinions[0]
    assert [(page.label, page.kind, page.volume, page.edition) for page in indexed.pages] == [
        ("556", None, 550, "U.S."),
        ("557", None, 550, "U.S."),
    ]
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    decision = _decision()
    validation = service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=decision.model_dump_json())), context
    )
    assert validation.as_bool()
    pages = asyncio.run(review_reporter_citation_pinpoint_pages(before, reviewer=Reviewer({})))
    after = asyncio.run(review_reporter_citation_full_opinions(pages, reviewer=Reviewer({"root": decision})))
    assert after.roots[0].reporter_support_reviews[-1].decision == decision
    passage = after.roots[0].reporter_opinion_evidence[-1]
    assert passage.quote == FIRST_PAGE and passage.span == _span(indexed.text, FIRST_PAGE)
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored.get_stage(EVIDENCE_STAGE) == before
    assert restored.get_stage(PAGE_STAGE) == pages
    assert restored.roots[0].reporter_support_reviews[-1].decision == decision
