"""Page and full-opinion reviews respect evidence scope and durable histories."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from mellea.stdlib.requirements import req
from pydantic import ValidationError

from mellea_lrc.llm.ivr import InstructIvrSpec, _requirements_for
from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind, PinCiteTarget
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_opinion import ReporterRootOpinionSource
from mellea_lrc.model.citations.reporter_pages import OpinionPage
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionEvidenceQuote,
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointEvidenceOutcome,
    PropositionDecision,
    ReporterPinpointVerdict,
    ReporterSupportDecision,
)
from mellea_lrc.model.citations.short_reporter import ShortReporterCitation
from mellea_lrc.providers.courtlistener.models import CourtListenerCluster
from mellea_lrc.validation.index_reporter_root_opinion_pages import index_reporter_root_opinion_pages
from mellea_lrc.validation.judge_reporter_citation_content import judge_reporter_citation_content
from mellea_lrc.validation.prepare_reporter_citation_evidence import (
    SUBSTAGE as EVIDENCE_SUBSTAGE,
)
from mellea_lrc.validation.prepare_reporter_citation_evidence import (
    prepare_reporter_citation_evidence,
)
from mellea_lrc.validation.read_reporter_citation_propositions import read_reporter_citation_propositions
from mellea_lrc.validation.reporter_pinpoint_review import reviewer as service
from mellea_lrc.validation.reporter_pinpoint_review.responses import (
    MAX_EVIDENCE_QUOTES,
    MAX_QUOTE_CHARACTERS,
    PaginatedSupportResponse,
    UnpaginatedSupportResponse,
)
from mellea_lrc.validation.resolve_reporter_citation_pages import resolve_reporter_citation_pages
from mellea_lrc.validation.retrieve_reporter_root_opinions import retrieve_reporter_root_opinions
from mellea_lrc.validation.review_reporter_citation_full_opinions_llm import (
    SUBSTAGE as FULL_SUBSTAGE,
)
from mellea_lrc.validation.review_reporter_citation_full_opinions_llm import (
    review_reporter_citation_full_opinions_llm,
)
from mellea_lrc.validation.review_reporter_citation_pages_llm import (
    SUBSTAGE as PAGE_SUBSTAGE,
)
from mellea_lrc.validation.review_reporter_citation_pages_llm import (
    review_reporter_citation_pages_llm,
)
from tests.test_read_reporter_citation_propositions import _run

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
    repeated_quote=False,
    first_page=FIRST_PAGE,
    intervening_page=None,
    leaf_locator="550 U.S. at 557",
    leaf_pin="557",
):
    source = f"{FIRST_PROPOSITION} Alpha, 550 U.S. 544, {root_pin} (2007)."
    if include_leaf:
        source += f" {SECOND_PROPOSITION} Alpha, {leaf_locator}."
    document = Document.from_source(source)
    root = FullReporterCitation.from_locator(
        citation_id="root",
        substage="grow_roots.discover_root_locators.find_full_reporter_locators",
        source=source,
        span=_span(source, "550 U.S. 544"),
    )
    document = document.add_citation(root).complete_substage(
        "grow_roots.discover_root_locators.find_full_reporter_locators"
    )
    root = (
        root.record("grow_roots.form_root_groups.form_roots")
        .with_root(root.id)
        .with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    )
    root = root.with_pin_cite(source, _span(source, root_pin))
    document = document.replace_citation(root).complete_substage("grow_roots.form_root_groups.form_roots")
    if include_leaf:
        leaf = (
            ShortReporterCitation.from_short_locator(
                citation_id="leaf",
                substage="test_leaf",
                source=source,
                span=_span(source, leaf_locator),
                pin_cite_span=_span(source, leaf_pin),
            )
            .record("test_leaf")
            .with_root(root.id)
        )
        document = document.add_citation(leaf).complete_substage("test_leaf")
    root = document.roots[0].record("test_opinion_source")
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate({"id": 1, "sub_opinions": [20, 21]}),
        )
    )
    document = document.replace_citation(root).complete_substage("test_opinion_source")
    marker = '<page-number label="{label}" volume="550" edition="U.S.">{label}</page-number>'
    if unknown_kind_pagination:
        marker = '<span class="star-pagination" volume="550" edition="U.S.">*{label}</span>'
    second_page = FIRST_PAGE if repeated_quote else SECOND_PAGE
    first_html = PRELUDE + marker.format(label="556") + first_page + marker.format(label="557") + second_page
    if intervening_page is not None:
        first_html = (
            PRELUDE
            + marker.format(label="556")
            + first_page
            + marker.format(label="557")
            + intervening_page
            + marker.format(label="558")
            + second_page
        )
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
    document = retrieve_reporter_root_opinions(document, client=client)
    document = index_reporter_root_opinion_pages(document)
    document = resolve_reporter_citation_pages(document).complete_substage(
        "validate_all_citation_content.prepare_citation_evidence.select_opinions_llm"
    )

    async def proposition(context):
        return PropositionDecision(
            quotes=(FIRST_PROPOSITION if context.citation_id == "root" else SECOND_PROPOSITION,),
            reason="The filing attributes this passage to this occurrence.",
        )

    document = asyncio.run(read_reporter_citation_propositions(document, reviewer=proposition))
    return prepare_reporter_citation_evidence(document)


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


def _wire(decision, *, paginated=True, footnote=None):
    evidence = []
    for quote in decision.evidence:
        value = {"opinion_id": quote.opinion_id, "quote": quote.quote}
        if paginated:
            value["footnote"] = footnote
        evidence.append(value)
    return json.dumps({"result": decision.result.value, "evidence": evidence, "reason": decision.reason})


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


def test_full_source_blocks_reconstruct_exact_opinions_and_keep_source_page_ids():
    document = _ready()
    root = document.roots[0]
    context = service.ReporterPinpointReviewContext.from_document(
        document, root, OpinionReviewScope.FULL_OPINION
    )
    bundle = json.loads(context.prefix.split("Shared saved opinion sources:\n", 1)[1])
    indexed = {opinion.opinion_id: opinion for opinion in root.reporter_root_opinion_page_index.opinions}

    for supplied in bundle["opinions"]:
        opinion = indexed[supplied["opinion_id"]]
        blocks = supplied["blocks"]
        assert "text" not in supplied and "page_markers" not in supplied
        assert "".join(block["text"] for block in blocks) == opinion.text
        offset = 0
        for block in blocks:
            assert block["source_offset"] == offset
            assert block["text"] == opinion.text[offset : offset + len(block["text"])]
            offset += len(block["text"])
            for reference in block["pages"]:
                page = opinion.pages[reference["page_index"]]
                assert reference["label"] == page.label
                assert reference["kind"] == page.kind
                assert reference["volume"] == page.volume
                assert reference["edition"] == page.edition
        assert offset == len(opinion.text)
    lead = next(supplied for supplied in bundle["opinions"] if supplied["opinion_id"] == "20")
    assert PRELUDE in lead["blocks"][0]["text"]
    assert lead["blocks"][0]["pages"] == []


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
    assert first.response_model.__name__ == "PaginatedSupportResponse"
    assert second.response_model.__name__ == "UnpaginatedSupportResponse"
    bundle = json.loads(first.prefix.split("Shared saved opinion sources:\n", 1)[1])
    assert any(block["pages"] for block in bundle["opinions"][0]["blocks"])
    assert "text" not in bundle["opinions"][0]
    assert "page_markers" not in bundle["opinions"][0]
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
            _wire(decision),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={}, max_attempts=3)
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=reviewer))
    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=reviewer))
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


def test_page_review_keeps_correct_location_separate_from_contradicted_use(monkeypatch):
    before = _ready()
    captured = []
    decision = _decision(OpinionSupportResult.CONTRADICTED, correct=True)

    async def fake_run(session, spec, *, strategy, model_options):
        captured.append(spec)
        return _run(
            _wire(decision),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={}, max_attempts=3)
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=reviewer))
    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=reviewer))

    assert [spec.user_variables["scope"] for spec in captured] == ["cited_pages", "full_opinion"]
    for spec in captured:
        assert "Assess the attributed use independently of page placement" in spec.prefix
        assert "program grounds each quote and compares its recovered location" in spec.prefix
        assert (
            "misrepresented, contradicted by its context, or irrelevant to the asserted proposition"
            in spec.prefix
        )
        assert "Those content problems do not by themselves make its location wrong" in spec.prefix
        assert (
            "Distinguish no relevant discussion or quoted passage anywhere in the complete intended opinion"
            in spec.prefix
        )
        assert "from an existing passage whose meaning the filing misrepresents" in spec.prefix
        assert "Do not generate page indices" in spec.prefix
        assert "program derives page placement from the grounded quotes" in spec.description
        assert "independently of result" in spec.description

    reviews = after.roots[0].reporter_support_reviews
    assert [review.scope for review in reviews] == [
        OpinionReviewScope.CITED_PAGES,
        OpinionReviewScope.FULL_OPINION,
    ]
    assert all(review.decision == decision for review in reviews)
    assert all(review.decision.correct_page is True for review in reviews)
    assert all(review.decision.found_pages == (_page(556),) for review in reviews)
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored.roots[0].reporter_support_reviews == reviews
    assert restored.get_substage(EVIDENCE_SUBSTAGE) == before


def test_supported_correct_target_exits_without_initializing_full_reviewer(monkeypatch):
    before = _ready()
    page_reviewer = Reviewer({"root": _decision()})
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=page_reviewer))

    def fail(_profile):
        pytest.fail("Correct target must not initialize a full-opinion reviewer")

    monkeypatch.setattr(service.IvrReporterPinpointReviewer, "from_profile", fail)
    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages))

    assert len(page_reviewer.contexts) == 1
    assert len(after.roots[0].reporter_support_reviews) == 1
    assert after.roots[0].reporter_support_reviews[0].scope is OpinionReviewScope.CITED_PAGES
    assert FULL_SUBSTAGE in after.substage_runs
    judged = judge_reporter_citation_content(after)
    judgment = judged.roots[0].reporter_pinpoint_judgments[-1]
    assert judgment.verdict is ReporterPinpointVerdict.CORRECT_PINCITE
    assert judgment.correct_page is True
    assert judgment.review_index == 0
    restored = Document.model_validate_json(judged.model_dump_json())
    assert restored.get_substage(PAGE_SUBSTAGE) == pages


@pytest.mark.parametrize(
    "page_result",
    [
        _decision(OpinionSupportResult.UNAVAILABLE, quote=None, correct=None),
        _decision(OpinionSupportResult.UNAVAILABLE, correct=True),
        _decision(OpinionSupportResult.CONTRADICTED, correct=True),
        _decision(OpinionSupportResult.CONTRADICTED, correct=False),
        _decision(OpinionSupportResult.CONTRADICTED, correct=None),
        _decision(correct=False),
        _decision(correct=None),
        service.ReporterPinpointReviewOutcome(None, failure_reason="IVR repair exhausted"),
    ],
)
def test_uncertain_negative_wrong_target_and_failed_pages_call_full_opinion(page_result):
    before = _ready()
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({"root": page_result})))
    full = Reviewer({"root": _decision(quote=SECOND_PAGE, correct=False)})

    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=full))

    assert len(full.contexts) == 1
    assert full.contexts[0].scope is OpinionReviewScope.FULL_OPINION
    assert after.roots[0].reporter_support_reviews[-1].scope is OpinionReviewScope.FULL_OPINION
    assert after.roots[0].reporter_opinion_evidence[-1].quote == SECOND_PAGE


def test_missing_page_routes_directly_to_full_opinion_and_reports_missing_writing():
    before = _ready(root_pin="599", missing_dissent=True)
    assert before.roots[0].reporter_pinpoint_evidence[-1].outcome is PinpointEvidenceOutcome.MISSING_PAGES
    page = Reviewer({})
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=page))
    full = Reviewer({"root": _decision(correct=False)})

    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=full))

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
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({})))

    def fail(_profile):
        pytest.fail("Empty source must not initialize an LLM")

    monkeypatch.setattr(service.IvrReporterPinpointReviewer, "from_profile", fail)
    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages))
    review = after.roots[0].reporter_support_reviews[-1]

    assert review.decision is None and review.ivr is None
    assert review.opinion_evidence_indices == ()
    assert review.failure_reason == "The saved root opinion bundle contains no nonempty text."
    assert Document.model_validate_json(after.model_dump_json()).get_substage(PAGE_SUBSTAGE) == pages


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
    after = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({"root": bad})))

    review = after.roots[0].reporter_support_reviews[-1]
    assert review.decision is None
    assert "not grounded in its supplied" in review.failure_reason
    assert review.opinion_evidence_indices == ()
    assert after.roots[0].reporter_opinion_evidence == ()


def test_page_absence_requires_ivr_repair_to_unavailable():
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.CITED_PAGES
    )
    bad = _decision(OpinionSupportResult.NOT_FOUND, quote=None, correct=None)
    result = service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=_wire(bad))), context
    )

    assert not result.as_bool()
    assert "Selected pages cannot establish absence from the full opinion" in result.reason
    good = _decision(OpinionSupportResult.UNAVAILABLE, quote=None, correct=None)
    assert service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=_wire(good))), context
    ).as_bool()
    full = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    assert service._validate_evidence(
        SimpleNamespace(last_output=lambda: SimpleNamespace(value=_wire(bad))), full
    ).as_bool()


def test_evidence_spans_history_indexes_native_rewind_and_identity_are_preserved():
    before = _ready()
    pages = asyncio.run(
        review_reporter_citation_pages_llm(before, reviewer=Reviewer({"root": _decision(correct=False)}))
    )
    after = asyncio.run(
        review_reporter_citation_full_opinions_llm(
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
    assert restored.get_substage(EVIDENCE_SUBSTAGE) == before
    assert restored.get_substage(PAGE_SUBSTAGE) == pages
    assert restored.get_substage(FULL_SUBSTAGE) == after


def test_prior_supported_page_review_does_not_skip_new_evidence_history():
    before = _ready()
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({"root": _decision()})))
    root = pages.roots[0].record("test_refresh_evidence")
    refreshed = root.reporter_pinpoint_evidence[-1].model_copy(update={"node_id": root.nodes[-1].id})
    pages = pages.replace_citation(root.with_reporter_pinpoint_evidence(refreshed)).complete_substage(
        "test_refresh_evidence"
    )
    full = Reviewer({"root": _decision(quote=SECOND_PAGE, correct=False)})

    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=full))

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
        rejected = validate(SimpleNamespace(last_output=lambda: SimpleNamespace(value=_wire(bad))))
        assert not rejected.as_bool() and "supplied" in rejected.reason
        assert validate(SimpleNamespace(last_output=lambda: SimpleNamespace(value=_wire(good)))).as_bool()
        initial = _run(
            _wire(bad),
            success=False,
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )
        final = _run(
            _wire(good),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )
        return final.model_copy(
            update={
                "attempts": (*initial.attempts, *final.attempts),
                "selected_attempt": 1,
                "output_schema": spec.output_format.model_json_schema(),
            }
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(
        session=object(), model_options={"max_tokens": 6000}, max_attempts=3
    )
    after = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=reviewer))
    review = after.roots[0].reporter_support_reviews[-1]

    assert review.decision == good
    assert review.ivr.attempts[0].output == _wire(bad)
    assert review.ivr.attempts[1].output == _wire(good)
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
            _wire(bad),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={}, max_attempts=3)
    after = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=reviewer))
    review = after.roots[0].reporter_support_reviews[-1]

    assert review.decision is None
    assert "not grounded" in review.failure_reason
    assert review.ivr.output == _wire(bad)
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
        review_reporter_citation_pages_llm(before, reviewer=Reviewer({"root": _decision(correct=None)}))
    )
    full = Reviewer({"root": _decision(correct=None)})

    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=full))

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
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({"root": _decision()})))
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
            review_reporter_citation_pages_llm(
                before.get_substage(
                    "validate_all_citation_content.prepare_citation_evidence.read_propositions"
                ),
                reviewer=reviewer,
            )
        )
    with pytest.raises(ValueError, match="Complete page review"):
        asyncio.run(review_reporter_citation_full_opinions_llm(before, reviewer=reviewer))
    assert reviewer.contexts == []
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=reviewer))
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(review_reporter_citation_pages_llm(pages, reviewer=reviewer))
    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=Reviewer({})))
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(review_reporter_citation_full_opinions_llm(after, reviewer=Reviewer({})))


@pytest.mark.parametrize("field", ["page_indices", "found_pages", "correct_page"])
def test_model_supplied_page_locations_are_rejected_by_the_schema(field):
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    raw = json.loads(_wire(_decision()))
    if field == "page_indices":
        raw["evidence"][0][field] = [99]
    else:
        raw[field] = True if field == "correct_page" else [556]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        context.parse_response(json.dumps(raw), "review")


def test_recovered_quote_pages_use_the_requested_reporter_namespace():
    document = _ready()
    root = document.roots[0]
    index = root.reporter_root_opinion_page_index
    opinion = index.opinions[0]
    parallel = OpinionPage(
        label="1955",
        kind=PinCiteKind.PAGE,
        volume=127,
        edition="S. Ct.",
        span=opinion.pages[0].span,
        citation_index="parallel",
    )
    opinion = opinion.model_copy(update={"pages": (opinion.pages[0], parallel, opinion.pages[1])})
    root = root.model_copy(
        update={
            "reporter_root_opinion_page_index": index.model_copy(
                update={"opinions": (opinion, *index.opinions[1:])}
            )
        }
    )
    document = document.model_copy(update={"citations": (root, *document.citations[1:])})
    context = service.ReporterPinpointReviewContext.from_document(
        document, root, OpinionReviewScope.FULL_OPINION
    )
    bundle = json.loads(context.prefix.split("Shared saved opinion sources:\n", 1)[1])
    lead = next(item for item in bundle["opinions"] if item["opinion_id"] == "20")
    shared_block = next(block for block in lead["blocks"] if FIRST_PAGE in block["text"])
    assert {page["edition"] for page in shared_block["pages"]} == {"U.S.", "S. Ct."}
    assert context.excerpts[0].page_indices == (0, 2)
    assert "".join(block["text"] for block in lead["blocks"]) == opinion.text
    decision, passages = context.parse_response(_wire(_decision()), "review")
    assert decision.found_pages == (_page(556),)
    assert passages[0].span == _span(opinion.text, FIRST_PAGE)


def test_quotes_in_unindexed_text_retain_grounding_without_inventing_a_page():
    document = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        document, document.roots[0], OpinionReviewScope.FULL_OPINION
    )
    response = _wire(_decision(quote=PRELUDE, correct=None, found=()))
    decision, passages = context.parse_response(response, "review")
    assert decision.pagination_available is True
    assert decision.correct_page is None and decision.found_pages == ()
    assert passages[0].quote == PRELUDE
    indexed, _ = context.parse_response(_wire(_decision()), "review")
    assert indexed.found_pages == (_page(556),)
    assert indexed.correct_page is True


def test_quote_repair_retains_trace_and_recovers_the_alternative_page(monkeypatch):
    before = _ready(root_pin="599")
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({})))
    bad = _decision(quote="This invented passage does not appear in any opinion.", correct=False, found=())
    good = _decision(quote=SECOND_PAGE, correct=False, found=(_page(557),))
    bad_output = _wire(bad)
    good_output = _wire(good)

    async def fake_run(session, spec, *, strategy, model_options):
        validate = spec.requirements[0].validation_fn
        assert not validate(SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad_output))).as_bool()
        assert validate(SimpleNamespace(last_output=lambda: SimpleNamespace(value=good_output))).as_bool()
        initial = _run(bad_output, success=False)
        repaired = _run(good_output)
        return repaired.model_copy(
            update={"attempts": (*initial.attempts, *repaired.attempts), "selected_attempt": 1}
        )

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={}, max_attempts=3)
    reviewed = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=reviewer))
    review = reviewed.roots[0].reporter_support_reviews[-1]
    assert review.decision == good
    assert review.ivr.attempts[0].output == bad_output
    assert review.ivr.attempts[1].output == good_output
    assert reviewed.roots[0].reporter_opinion_evidence[-1].quote == SECOND_PAGE


def test_unpaginated_full_opinion_can_settle_support_without_a_page_negative():
    before = _ready(unpaginated=True)
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({})))
    decision = _decision(pagination=False, correct=None)
    full = Reviewer({"root": decision})
    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=full))
    review = after.roots[0].reporter_support_reviews[-1]
    assert review.decision == decision
    assert decision.result is OpinionSupportResult.SUPPORTED
    assert decision.pagination_available is False
    assert decision.correct_page is None and decision.found_pages == ()
    assert after.roots[0].reporter_opinion_evidence[-1].quote == FIRST_PAGE
    assert full.contexts[0].source_complete is True


def test_unpaginated_ivr_requests_content_only_and_saves_canonical_page_fields(monkeypatch):
    before = _ready(unpaginated=True)
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({})))
    expected = _decision(pagination=False, correct=None)
    raw = _wire(expected, paginated=False)
    captured = {}

    async def fake_run(session, spec, *, strategy, model_options):
        captured["spec"] = spec
        assert set(spec.output_format.model_json_schema()["properties"]) == {"result", "evidence", "reason"}
        evidence_schema = spec.output_format.model_json_schema()["$defs"]["OpinionEvidenceQuote"]
        assert set(evidence_schema["properties"]) == {"opinion_id", "quote"}
        return _run(raw, prefix=spec.prefix, instruction=spec.description, variables=spec.user_variables)

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterPinpointReviewer(session=object(), model_options={}, max_attempts=3)
    after = asyncio.run(review_reporter_citation_full_opinions_llm(pages, reviewer=reviewer))
    review = after.roots[0].reporter_support_reviews[-1]
    assert review.decision == expected
    assert review.ivr.output == raw
    assert captured["spec"].output_format.__name__ == "UnpaginatedSupportResponse"
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    with pytest.raises(ValueError):
        context.parse_response(_wire(expected), "review")
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored.roots[0].reporter_support_reviews[-1] == review
    assert restored.get_substage(PAGE_SUBSTAGE) == pages


@pytest.mark.parametrize("result", [OpinionSupportResult.SUPPORTED, OpinionSupportResult.CONTRADICTED])
def test_grounded_quote_recovers_page_independently_of_content_result(result):
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    for quote, correct, page in [(FIRST_PAGE, True, 556), (SECOND_PAGE, False, 557)]:
        decision, passages = context.parse_response(_wire(_decision(result, quote=quote)), "review")
        assert decision.result is result
        assert decision.correct_page is correct
        assert decision.found_pages == (_page(page),)
        assert passages[0].quote == quote


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
@pytest.mark.parametrize("reported", [False, True])
def test_repeated_quote_grounds_at_written_or_reported_page(scope, reported):
    before = _ready(root_pin="557", repeated_quote=True)
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    decision = _decision(quote=FIRST_PAGE, correct=True, found=(_page(557),) if reported else ())
    passages = context.ground(decision, "review")
    opinion = before.roots[0].reporter_root_opinion_page_index.opinions[0]
    start = opinion.text.index(FIRST_PAGE, opinion.text.index(FIRST_PAGE) + 1)
    assert passages[0].span == Span(start, start + len(FIRST_PAGE))
    passages[0].validate_source(opinion.text)


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
def test_source_spans_recover_every_page_crossed_by_a_contiguous_quote(scope):
    before = _ready(root_pin="556-57")
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    quote = FIRST_PAGE + " " + SECOND_PAGE
    raw = _wire(_decision(quote=quote, found=(_page(556, 557),)))
    decision, passages = context.parse_response(raw, "review")
    opinion = before.roots[0].reporter_root_opinion_page_index.opinions[0]
    assert decision.found_pages == (_page(556), _page(557))
    assert passages[0].quote == quote
    assert passages[0].span == _span(opinion.text, quote)
    passages[0].validate_source(opinion.text)


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
def test_quote_cannot_ground_across_unsupplied_source_prose(scope):
    intervening = "The intervening page supplies a separate limitation that cannot be omitted."
    before = _ready(root_pin="556, 558", intervening_page=intervening)
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    quote = FIRST_PAGE + " " + intervening + " " + SECOND_PAGE
    if scope is OpinionReviewScope.CITED_PAGES:
        assert FIRST_PAGE in context.prefix and SECOND_PAGE in context.prefix
        assert intervening not in context.prefix
        assert len(context.excerpts) == 2
    else:
        assert intervening in context.prefix
    if scope is OpinionReviewScope.CITED_PAGES:
        with pytest.raises(ValueError, match="not grounded"):
            context.parse_response(_wire(_decision(quote=quote, correct=None)), "review")
    else:
        decision, passages = context.parse_response(_wire(_decision(quote=quote, correct=None)), "review")
        assert decision.found_pages == (_page(556), _page(557), _page(558))
        assert passages[0].quote == quote


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
def test_repeated_quote_within_written_range_uses_first_grounded_target_occurrence(scope):
    before = _ready(root_pin="556-57", repeated_quote=True)
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    decision, passages = context.parse_response(_wire(_decision()), "review")
    opinion = before.roots[0].reporter_root_opinion_page_index.opinions[0]
    start = opinion.text.index(FIRST_PAGE)
    assert decision.found_pages == (_page(556),)
    assert passages[0].span == Span(start, start + len(FIRST_PAGE))
    passages[0].validate_source(opinion.text)


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
def test_reported_page_disambiguates_repeated_quote_within_written_range(scope):
    before = _ready(root_pin="556-57", repeated_quote=True)
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    passages = context.ground(_decision(quote=FIRST_PAGE, correct=True, found=(_page(557),)), "review")
    opinion = before.roots[0].reporter_root_opinion_page_index.opinions[0]
    start = opinion.text.index(FIRST_PAGE, opinion.text.index(FIRST_PAGE) + 1)
    assert passages[0].span == Span(start, start + len(FIRST_PAGE))
    passages[0].validate_source(opinion.text)


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
@pytest.mark.parametrize("wire_response", [False, True])
@pytest.mark.parametrize(
    "earlier", [FIRST_PAGE.replace("granted.", "granted ."), FIRST_PAGE.replace("notice", "notlce")]
)
def test_earlier_tolerant_quote_is_not_hidden_by_later_exact_quote(scope, wire_response, earlier):
    before = _ready(root_pin="556-57", repeated_quote=True, first_page=earlier)
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    # Start within the first page to exercise clipping of absolute page spans
    # to an excerpt with a nonzero source offset.
    excerpt = context.excerpts[0]
    trim = excerpt.text.index(earlier)
    context = replace(
        context,
        excerpts=(
            replace(excerpt, text=excerpt.text[trim:], offset=excerpt.offset + trim),
            *context.excerpts[1:],
        ),
    )
    if wire_response:
        _, passages = context.parse_response(_wire(_decision()), "review")
    else:
        passages = context.ground(_decision(quote=FIRST_PAGE, correct=True, found=(_page(556),)), "review")
    opinion = before.roots[0].reporter_root_opinion_page_index.opinions[0]
    start = opinion.text.index(earlier)
    assert passages[0].quote == earlier
    assert passages[0].span == Span(start, start + len(earlier))
    passages[0].validate_source(opinion.text)


def test_full_review_retains_later_repeated_quote_and_replays_its_location():
    before = _ready(root_pin="557", repeated_quote=True)
    uncertain = _decision(result=OpinionSupportResult.UNAVAILABLE, quote=None, correct=None, found=())
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({"root": uncertain})))
    decision = _decision(quote=FIRST_PAGE, correct=True, found=(_page(557),))
    after = asyncio.run(
        review_reporter_citation_full_opinions_llm(pages, reviewer=Reviewer({"root": decision}))
    )
    root = after.roots[0]
    assert root.reporter_support_reviews[-1].decision == decision
    opinion = root.reporter_root_opinion_page_index.opinions[0]
    passage = root.reporter_opinion_evidence[-1]
    start = opinion.text.index(FIRST_PAGE, opinion.text.index(FIRST_PAGE) + 1)
    assert passage.span == Span(start, start + len(FIRST_PAGE))
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored.roots[0].reporter_opinion_evidence[-1] == passage
    assert restored.get_substage(PAGE_SUBSTAGE) == pages


@pytest.mark.parametrize("wire_response", [False, True])
def test_repeated_quote_keeps_whitespace_and_edit_distance_grounding(wire_response):
    before = _ready(root_pin="557", repeated_quote=True)
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    excerpt = context.excerpts[0]
    start = excerpt.text.index(FIRST_PAGE, excerpt.text.index(FIRST_PAGE) + 1)
    canonical = FIRST_PAGE.replace("notice", "notlce").replace(" ", "\n", 1)
    text = excerpt.text[:start] + canonical + excerpt.text[start + len(FIRST_PAGE) :]
    context = replace(context, excerpts=(replace(excerpt, text=text), *context.excerpts[1:]))
    if wire_response:
        _, passages = context.parse_response(_wire(_decision()), "review")
    else:
        passages = context.ground(_decision(quote=FIRST_PAGE, found=(_page(557),)), "review")
    assert passages[0].quote == canonical
    assert passages[0].span == Span(start, start + len(canonical))
    passages[0].validate_source(text)


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
@pytest.mark.parametrize("wire_response", [False, True])
def test_content_quotes_allow_minor_copy_errors_and_retain_exact_source(scope, wire_response):
    canonical = FIRST_PAGE.replace("notice", "notlce").replace("granted", "grantcd")
    before = _ready(first_page=canonical)
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    if wire_response:
        _, passages = context.parse_response(_wire(_decision()), "review")
    else:
        passages = context.ground(_decision(), "review")
    assert passages[0].quote == canonical
    source = before.roots[0].reporter_root_opinion_page_index.opinions[0].text
    passages[0].validate_source(source)


def test_content_quotes_still_reject_substantially_different_words():
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    altered = FIRST_PAGE.replace("fair notice", "no warning").replace("granted", "denied")
    with pytest.raises(ValueError, match="not grounded"):
        context.parse_response(_wire(_decision(quote=altered)), "review")


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
@pytest.mark.parametrize("paginated", [False, True])
def test_separate_sentences_ground_individually_with_fuzziness_and_whitespace(scope, paginated):
    third = "The parties received an opportunity to address the proposed exception."
    unrelated = "An unrelated procedural discussion separates these passages. " * 5
    before = _ready(root_pin="556-57", first_page=FIRST_PAGE + " " + unrelated + third)
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    if not paginated:
        context = replace(
            context,
            excerpts=tuple(replace(item, pages=(), page_indices=()) for item in context.excerpts),
        )
    quotes = (FIRST_PAGE, SECOND_PAGE, third)
    evidence = []
    for quote in quotes:
        # Each independent quote needs its own tolerance; a later exact match
        # cannot compensate for an earlier bad quote.
        proposed = quote.replace(" ", "  ").replace("opinion", "oplnion").replace("parties", "partles")
        item = {"opinion_id": "20", "quote": proposed}
        if paginated:
            item["footnote"] = None
        evidence.append(item)
    raw = {"result": "supported", "evidence": evidence, "reason": "Together these passages support the use."}

    decision, passages = context.parse_response(json.dumps(raw), "review")
    source = before.roots[0].reporter_root_opinion_page_index.opinions[0].text
    assert len(decision.evidence) == len(passages) == 3
    assert tuple(item.quote for item in passages) == quotes
    assert tuple(item.span for item in passages) == tuple(_span(source, quote) for quote in quotes)
    assert decision.found_pages == ((_page(556), _page(557)) if paginated else ())
    assert "no requirement that the items form a continuous chunk" in context.prefix
    for passage in passages:
        passage.validate_source(source)

    evidence[1]["quote"] = "This unrelated claim never appears in the source."
    with pytest.raises(ValueError, match="not grounded"):
        context.parse_response(json.dumps(raw), "review")


@pytest.mark.parametrize("paginated", [False, True])
@pytest.mark.parametrize("wire_response", [False, True])
@pytest.mark.parametrize("bad", ["four_quotes", "oversized_last_quote"])
def test_quote_limits_preflight_all_items_before_any_fuzzy_matching(
    monkeypatch, paginated, wire_response, bad
):
    before = _ready(unpaginated=not paginated)
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    quotes = [OpinionEvidenceQuote(opinion_id="20", quote=FIRST_PAGE)]
    if bad == "four_quotes":
        quotes *= MAX_EVIDENCE_QUOTES + 1
    else:
        quotes.append(OpinionEvidenceQuote(opinion_id="20", quote="x" * (MAX_QUOTE_CHARACTERS + 1)))
    decision = _decision(pagination=paginated, correct=True if paginated else None).model_copy(
        update={"evidence": tuple(quotes)}
    )

    def fail(*args, **kwargs):
        pytest.fail("Every evidence item must pass length checks before any grounding")

    monkeypatch.setattr(service.ReporterPinpointReviewContext, "_ground_quote", fail)
    with pytest.raises(
        ValueError, match=r"Too many evidence quotes|is too long|at most 1200 characters"
    ) as caught:
        if wire_response:
            context.parse_response(_wire(decision, paginated=paginated), "review")
        else:
            context.ground(decision, "review")
    if wire_response and bad == "oversized_last_quote":
        error = caught.value.errors(include_url=False)[0]
        assert error["type"] == "string_too_long"
        assert error["loc"] == ("evidence", 1, "quote")
    else:
        assert "one or two" in str(caught.value)
        assert "reason" in str(caught.value)


@pytest.mark.parametrize("paginated", [False, True])
def test_oversized_quote_uses_native_ivr_schema_repair_feedback(monkeypatch, paginated):
    before = _ready(unpaginated=not paginated)
    context = service.ReporterPinpointReviewContext.from_document(
        before,
        before.roots[0],
        OpinionReviewScope.FULL_OPINION,
    )
    bad = _decision(
        quote="x" * 58_203,
        pagination=paginated,
        correct=None,
        found=(),
    )
    output = SimpleNamespace(last_output=lambda: SimpleNamespace(value=_wire(bad, paginated=paginated)))

    def fail(*args, **kwargs):
        pytest.fail("Schema repair must reject a long quote before grounding")

    monkeypatch.setattr(service.ReporterPinpointReviewContext, "_ground_quote", fail)
    requirements = _requirements_for(
        InstructIvrSpec(
            description="Review supplied evidence.",
            output_format=context.response_model,
            requirements=(
                req("Ground evidence.", validation_fn=lambda ctx: service._validate_evidence(ctx, context)),
            ),
        )
    )
    rejection = requirements[0].validation_fn(output)
    assert not rejection.as_bool()
    assert "evidence.[0].quote" in rejection.reason
    assert "at most 1200 characters" in rejection.reason
    assert requirements[1].validation_fn(output).as_bool()


@pytest.mark.parametrize("response_model", [PaginatedSupportResponse, UnpaginatedSupportResponse])
def test_short_quote_with_legal_abbreviations_does_not_guess_sentence_boundaries(response_model):
    quote = "The standard in 123 F. Supp. 2d 456 applies under Fed. R. Civ. P. 12(b)(6)."
    response = {
        "result": "supported",
        "evidence": [{"opinion_id": "20", "quote": quote}],
        "reason": "Supported.",
    }
    if response_model is PaginatedSupportResponse:
        response["evidence"][0]["footnote"] = None
    assert response_model.model_validate(response).evidence[0].quote == quote
    assert response_model.model_json_schema()["properties"]["evidence"]["maxItems"] == 3


@pytest.mark.parametrize("response_model", [PaginatedSupportResponse, UnpaginatedSupportResponse])
def test_response_quote_length_boundary_is_in_json_schema(response_model):
    response = {
        "result": "supported",
        "evidence": [{"opinion_id": "20", "quote": "x" * MAX_QUOTE_CHARACTERS}],
        "reason": "Supported.",
    }
    definition = "OpinionEvidenceQuote"
    if response_model is PaginatedSupportResponse:
        response["evidence"][0]["footnote"] = None
        definition = "PageEvidenceQuote"
    assert len(response_model.model_validate(response).evidence[0].quote) == MAX_QUOTE_CHARACTERS
    schema = response_model.model_json_schema()
    assert schema["$defs"][definition]["properties"]["quote"]["maxLength"] == MAX_QUOTE_CHARACTERS
    assert schema["properties"]["evidence"]["items"] == {"$ref": f"#/$defs/{definition}"}
    assert schema["properties"]["evidence"]["maxItems"] == MAX_EVIDENCE_QUOTES

    response["evidence"][0]["quote"] += "x"
    with pytest.raises(ValidationError) as caught:
        response_model.model_validate(response)
    error = caught.value.errors(include_url=False)[0]
    assert error["type"] == "string_too_long"
    assert error["loc"] == ("evidence", 0, "quote")
    assert error["ctx"] == {"max_length": MAX_QUOTE_CHARACTERS}


def test_quote_size_policy_does_not_change_durable_decision_serialization():
    decision = _decision(quote="x" * 58_203, correct=None, found=())
    assert ReporterSupportDecision.model_validate_json(decision.model_dump_json()) == decision
    assert "maxLength" not in OpinionEvidenceQuote.model_json_schema()["properties"]["quote"]
    assert (
        "maxLength"
        not in ReporterSupportDecision.model_json_schema()["$defs"]["OpinionEvidenceQuote"]["properties"][
            "quote"
        ]
    )
    with pytest.raises(ValueError, match="at most 1200 characters"):
        PaginatedSupportResponse.model_validate_json(_wire(decision))


@pytest.mark.parametrize("wire_response", [False, True])
def test_later_repeated_quote_can_cross_page_boundaries(wire_response):
    before = _ready(root_pin="557", repeated_quote=True)
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    excerpt = context.excerpts[0]
    start = excerpt.text.index(FIRST_PAGE, excerpt.text.index(FIRST_PAGE) + 1)
    boundary = start + len(FIRST_PAGE) // 2
    pages = (
        excerpt.pages[0],
        excerpt.pages[1].model_copy(update={"span": Span(excerpt.pages[1].span.start, boundary)}),
        OpinionPage(
            label="558",
            kind=PinCiteKind.PAGE,
            volume=550,
            edition="U.S.",
            span=Span(boundary, len(excerpt.text)),
        ),
    )
    context = replace(
        context,
        excerpts=(replace(excerpt, pages=pages, page_indices=(0, 1, 2)), *context.excerpts[1:]),
    )
    if wire_response:
        decision, passages = context.parse_response(_wire(_decision()), "review")
        assert decision.found_pages == (_page(557), _page(558))
    else:
        passages = context.ground(_decision(quote=FIRST_PAGE, found=(_page(557, 558),)), "review")
    assert passages[0].span == Span(start, start + len(FIRST_PAGE))
    passages[0].validate_source(excerpt.text)


def test_repeated_quote_cannot_supply_a_page_absent_from_the_source():
    before = _ready(root_pin="557", repeated_quote=True)
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    decision = _decision(quote=FIRST_PAGE, correct=True, found=(_page(558),))
    with pytest.raises(ValueError, match="found_pages must locate quoted passages"):
        context.ground(decision, "review")


def test_unindexed_inline_page_markers_use_content_only_response():
    before = _ready(inline_pagination=True)
    assert all(not opinion.pages for opinion in before.roots[0].reporter_root_opinion_page_index.opinions)
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    expected = _decision(pagination=False, correct=None)
    assert "*556" in context.prefix
    assert context.response_model.__name__ == "UnpaginatedSupportResponse"
    decision, passages = context.parse_response(_wire(expected, paginated=False), "review")
    assert decision == expected
    assert passages[0].quote == FIRST_PAGE
    with pytest.raises(ValueError):
        context.parse_response(_wire(_decision()), "review")


def test_unknown_kind_source_marker_uses_content_only_response():
    before = _ready(unknown_kind_pagination=True)
    indexed = before.roots[0].reporter_root_opinion_page_index.opinions[0]
    assert [(page.label, page.kind, page.volume, page.edition) for page in indexed.pages] == [
        ("556", None, 550, "U.S."),
        ("557", None, 550, "U.S."),
    ]
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    expected = _decision(pagination=False, correct=None)
    assert context.response_model.__name__ == "UnpaginatedSupportResponse"
    decision, passages = context.parse_response(_wire(expected, paginated=False), "validation")
    assert decision == expected
    assert passages[0].span == _span(indexed.text, FIRST_PAGE)
    pages = asyncio.run(review_reporter_citation_pages_llm(before, reviewer=Reviewer({})))
    after = asyncio.run(
        review_reporter_citation_full_opinions_llm(pages, reviewer=Reviewer({"root": decision}))
    )
    assert after.roots[0].reporter_support_reviews[-1].decision == decision
    passage = after.roots[0].reporter_opinion_evidence[-1]
    assert passage.quote == FIRST_PAGE and passage.span == _span(indexed.text, FIRST_PAGE)
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored.get_substage(EVIDENCE_SUBSTAGE) == before
    assert restored.get_substage(PAGE_SUBSTAGE) == pages
    assert restored.roots[0].reporter_support_reviews[-1].decision == decision


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
@pytest.mark.parametrize("footnote,correct", [(None, None), ("2", True), ("3", False)])
def test_grounded_page_with_footnote_keeps_unknown_and_different_notes_explicit(scope, footnote, correct):
    before = _ready(root_pin="556 n.2")
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    decision, passages = context.parse_response(_wire(_decision(), footnote=footnote), "review")
    assert decision.correct_page is correct
    assert decision.found_pages == (_page(556, footnote=footnote),)
    assert passages[0].quote == FIRST_PAGE


@pytest.mark.parametrize("result", [OpinionSupportResult.NOT_FOUND, OpinionSupportResult.UNAVAILABLE])
def test_no_quote_does_not_generate_a_negative_page_judgment(result):
    before = _ready()
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    decision, passages = context.parse_response(_wire(_decision(result, quote=None)), "review")
    assert decision.result is result
    assert decision.pagination_available is True
    assert decision.correct_page is None and decision.found_pages == ()
    assert passages == ()


def test_unindexed_evidence_keeps_page_uncertain_without_discarding_content():
    before = _ready(root_pin="599")
    context = service.ReporterPinpointReviewContext.from_document(
        before, before.roots[0], OpinionReviewScope.FULL_OPINION
    )
    raw = json.loads(_wire(_decision(quote=FIRST_PAGE)))
    raw["evidence"].append({"opinion_id": "20", "quote": PRELUDE, "footnote": None})
    decision, passages = context.parse_response(json.dumps(raw), "review")
    assert decision.result is OpinionSupportResult.SUPPORTED
    assert decision.found_pages == (_page(556),)
    assert decision.correct_page is None
    assert tuple(passage.quote for passage in passages) == (FIRST_PAGE, PRELUDE)


@pytest.mark.parametrize("response_model", [PaginatedSupportResponse, UnpaginatedSupportResponse])
def test_review_schema_requests_quotes_without_a_generated_page_judgment(response_model):
    schema = response_model.model_json_schema()
    assert set(schema["properties"]) == {"result", "evidence", "reason"}
    definition = "PageEvidenceQuote" if response_model is PaginatedSupportResponse else "OpinionEvidenceQuote"
    fields = (
        {"opinion_id", "quote", "footnote"}
        if response_model is PaginatedSupportResponse
        else {"opinion_id", "quote"}
    )
    assert set(schema["$defs"][definition]["properties"]) == fields


@pytest.mark.parametrize("scope", [OpinionReviewScope.CITED_PAGES, OpinionReviewScope.FULL_OPINION])
def test_target_fuzzy_match_precedes_exact_duplicate_outside_written_range(scope):
    fuzzy = FIRST_PAGE.replace("notice", "notlce")
    before = _ready(
        root_pin="556-57",
        first_page="Unrelated procedural history.",
        intervening_page=fuzzy,
        repeated_quote=True,
    )
    context = service.ReporterPinpointReviewContext.from_document(before, before.roots[0], scope)
    decision, passages = context.parse_response(_wire(_decision()), "review")
    source = before.roots[0].reporter_root_opinion_page_index.opinions[0].text
    assert source.index(fuzzy) < source.index(FIRST_PAGE)
    assert passages[0].quote == fuzzy
    assert passages[0].span == _span(source, fuzzy)
    passages[0].validate_source(source)
    assert decision.correct_page is True
    assert decision.found_pages == (_page(557),)
