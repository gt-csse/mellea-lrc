"""TOA components tag their own occurrences without suppressing shared authorities."""

from __future__ import annotations

import asyncio
import json

import pytest

from mellea_lrc.api import grow_leaves, grow_roots, reporter_root_lookup_cluster_retrieval
from mellea_lrc.model import Document, Span, TableOfAuthoritiesComponent
from mellea_lrc.model.citations import (
    FullDocketCitation,
    FullReporterCitation,
    IdCitation,
    ReferenceCitation,
    ShortReporterCitation,
    SupraCitation,
)
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind, PinCiteTarget
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_page_resolution import (
    ReporterCitationOpinionDecision,
    ReporterOpinionPageChoice,
    ReporterPageResolutionOutcome,
)
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionEvidenceQuote,
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointEvidenceOutcome,
    PropositionDecision,
    ReporterPinpointVerdict,
    ReporterSupportDecision,
)
from mellea_lrc.model.citations.tags import CitationTagKind
from mellea_lrc.validation.reporter_citation_full_opinion_review import (
    review_reporter_citation_full_opinions,
)
from mellea_lrc.validation.reporter_citation_opinion_review import review_reporter_citation_opinions
from mellea_lrc.validation.reporter_citation_page_resolution import resolve_reporter_citation_pages
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import (
    prepare_reporter_citation_pinpoint_evidence,
)
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import judge_reporter_citation_pinpoints
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import (
    review_reporter_citation_pinpoint_pages,
)
from mellea_lrc.validation.reporter_citation_propositions import read_reporter_citation_propositions
from mellea_lrc.validation.reporter_root_opinion_page_index import index_reporter_root_opinion_pages
from mellea_lrc.validation.reporter_root_opinion_retrieval import reporter_root_opinion_retrieval
from tests.test_reporter_root_opinion_retrieval import Client, opinion

TOA = CitationTagKind.TABLE_OF_AUTHORITIES
QUOTES = {
    "full_reporter": "550 U.S. 544",
    "full_docket": "No. 21-123",
    "short_reporter": "550 U.S. at 556",
    "reference": "Alpha",
    "id": "Id. at 556",
    "supra": "Alpha, supra, at 556",
}


def _span(source: str, quote: str) -> Span:
    start = source.index(quote)
    return Span(start, start + len(quote))


def _document(source: str, components=()) -> Document:
    document = Document.from_source(source)
    return Document.model_validate({**document.model_dump(mode="python"), "index_spans": components})


def _citation(kind: str, source: str, *, substage: str = "test_sites"):
    span = _span(source, QUOTES[kind])
    if kind == "full_reporter":
        return FullReporterCitation.from_locator(
            citation_id=kind, substage=substage, source=source, span=span
        )
    if kind == "full_docket":
        return FullDocketCitation.from_locator(
            citation_id=kind,
            substage=substage,
            source=source,
            span=span,
            number_span=_span(source, "21-123"),
        )
    if kind == "short_reporter":
        return ShortReporterCitation.from_short_locator(
            citation_id=kind,
            substage=substage,
            source=source,
            span=span,
            pin_cite_span=_span(source, "556"),
        )
    if kind == "reference":
        return ReferenceCitation.from_source(source=source, span=span, substage=substage)
    if kind == "id":
        return IdCitation.from_source(
            source=source, span=span, substage=substage, pin_span=_span(source, "556")
        )
    return SupraCitation.from_source(source=source, span=span, substage=substage)


@pytest.mark.parametrize("kind", QUOTES)
@pytest.mark.parametrize("containment", ["whole", "partial", "abutting"])
def test_every_citation_kind_requires_its_whole_own_site_inside_toa(kind, containment):
    source = f"Header. {QUOTES[kind]} Body."
    citation = _citation(kind, source)
    site = citation.site_span
    if containment == "whole":
        component = TableOfAuthoritiesComponent(site.start, site.end)
    elif containment == "partial":
        component = TableOfAuthoritiesComponent(site.start, site.end - 1)
    else:
        component = TableOfAuthoritiesComponent(0, site.start)
    before = _document(source, (component,)).complete_substage("before_sites")
    after = before.add_citation(citation).complete_substage("test_sites")
    stored = after.citations[0]

    assert after.text == source
    assert stored.site_span == site
    assert stored.model_dump(exclude={"tags"}) == citation.model_dump(exclude={"tags"})
    assert stored.has_tag(TOA) is (containment == "whole")
    assert len(stored.tags) == (1 if containment == "whole" else 0)
    if stored.tags:
        assert stored.tags[0].node_id == stored.nodes[0].id
        assert stored.tags[0].component_index == 0
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_substage("before_sites") == before


@pytest.mark.parametrize("kind", QUOTES)
@pytest.mark.parametrize("component_owner", ["root", "leaf"])
def test_attached_occurrences_do_not_inherit_the_roots_toa_tag(kind, component_owner):
    source = f"490 U.S. 386. Body citation: {QUOTES[kind]}."
    root_span = _span(source, "490 U.S. 386")
    leaf = _citation(kind, source, substage="test_leaf")
    containing_span = root_span if component_owner == "root" else leaf.site_span
    document = _document(source, (TableOfAuthoritiesComponent(containing_span.start, containing_span.end),))
    root = FullReporterCitation.from_locator(
        citation_id="root", substage="test_root", source=source, span=root_span
    )
    document = document.add_citation(root)
    root = document.citations[0].record("test_root").with_root("root")
    document = document.replace_citation(root).complete_substage("test_root")
    document = document.add_citation(leaf.record("test_leaf").with_root("root")).complete_substage(
        "test_leaf"
    )

    assert document.roots[0].has_tag(TOA) is (component_owner == "root")
    assert document.leaves[0].has_tag(TOA) is (component_owner == "leaf")
    assert document.leaves[0].id == leaf.id


def _tagged_document():
    source = "First index. Body gap. Second index: 550 U.S. 544."
    locator_span = _span(source, QUOTES["full_reporter"])
    components = (
        TableOfAuthoritiesComponent(0, len("First index.")),
        TableOfAuthoritiesComponent(locator_span.start, locator_span.end),
    )
    before = _document(source, components).complete_substage("before_sites")
    created = before.add_citation(_citation("full_reporter", source)).complete_substage("test_sites")
    root = created.citations[0].record("test_roots").with_root("full_reporter")
    return before, created, created.replace_citation(root).complete_substage("test_roots")


def test_native_json_records_component_kind_creation_node_and_exact_checkpoint_history():
    before, created, after = _tagged_document()
    saved = json.loads(after.model_dump_json())

    assert saved["index_spans"] == [
        {"start": component.start, "end": component.end, "kind": "table_of_authorities"}
        for component in after.index_spans
    ]
    assert saved["citations"][0]["tags"] == [
        {
            "node_id": created.citations[0].nodes[0].id,
            "kind": "table_of_authorities",
            "component_index": 1,
        }
    ]
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_substage("test_sites") == created
    assert restored.get_substage("before_sites") == before
    assert restored.get_substage("test_sites").citations[0].tags == restored.citations[0].tags


@pytest.mark.parametrize("corruption", ["unknown_component", "later_node", "outside_component"])
def test_native_json_rejects_invalid_toa_tag_provenance(corruption):
    _, _, document = _tagged_document()
    saved = document.model_dump(mode="json")
    tag = saved["citations"][0]["tags"][0]
    if corruption == "unknown_component":
        tag["component_index"] = len(saved["index_spans"])
        error = "unknown TOA component"
    elif corruption == "later_node":
        tag["node_id"] = saved["citations"][0]["nodes"][-1]["id"]
        error = "creation node"
    else:
        tag["component_index"] = 0
        error = "inside the TOA component"

    with pytest.raises(ValueError, match=error):
        Document.model_validate(saved)


def test_loading_missing_tags_does_not_retrofit_occurrence_metadata():
    _, _, document = _tagged_document()
    saved = document.model_dump(mode="json")
    for component in saved["index_spans"]:
        del component["kind"]
    del saved["citations"][0]["tags"]

    restored = Document.model_validate(saved)

    assert restored.index_spans == document.index_spans
    assert restored.citations[0].tags == ()
    assert not restored.citations[0].has_tag(TOA)
    replayed = (
        _document(restored.text, restored.index_spans)
        .add_citation(restored.get_substage("test_sites").citations[0])
        .complete_substage("test_sites")
    )
    assert replayed.citations[0].tags == document.citations[0].tags


def test_later_decision_cannot_remove_creation_tags():
    _, _, document = _tagged_document()
    original = document.citations[0]
    recorded = original.record("test_later")
    stripped = type(recorded).model_validate({**recorded.model_dump(mode="python"), "tags": ()})

    with pytest.raises(ValueError, match="tags must be append-only"):
        document.replace_citation(stripped)
    assert document.citations[0] == original


@pytest.mark.parametrize("root_has_pin", [True, False])
def test_toa_root_remains_retrievable_and_only_body_leaf_gets_support_reviews(root_has_pin):
    root_pin = ", 556" if root_has_pin else ""
    index = f"Alpha v. Beta, 550 U.S. 544{root_pin} (2007) .... 12\n"
    proposition = "The pleading must provide adequate notice."
    source = index + f"ARGUMENT\n{proposition} Alpha, 550 U.S. at 556."
    document = asyncio.run(grow_roots(_document(source, (TableOfAuthoritiesComponent(0, len(index)),))))
    document = asyncio.run(grow_leaves(document, review_leaves=False))
    root, leaf = document.roots[0], document.short_reporters[0]
    assert root.has_tag(TOA)
    assert not leaf.has_tag(TOA)
    assert leaf.root_id[-1].value == root.id
    opinion_quote = "A pleading must provide adequate notice before relief is granted."
    marker = '<page-number label="556" volume="550" edition="U.S.">556</page-number>'
    client = Client(
        [{"id": 1, "sub_opinions": [20, 21]}],
        {
            "20": opinion("20", html_with_citations=marker + opinion_quote),
            "21": opinion(
                "21", type="040dissent", html_with_citations=marker + "The dissent would require less notice."
            ),
        },
    )
    document = reporter_root_lookup_cluster_retrieval(document, client=client)
    root = document.roots[0].record("test_identity").with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    document = document.replace_citation(root).complete_substage("test_identity")
    admitted = document
    document = reporter_root_opinion_retrieval(document, client=client)
    assert client.calls == ["20", "21"]
    assert document.roots[0].reporter_root_opinion_retrieval is not None
    document = index_reporter_root_opinion_pages(document)
    assert document.roots[0].reporter_root_opinion_page_index is not None
    document = resolve_reporter_citation_pages(document)
    assert (
        document.short_reporters[0].reporter_page_resolutions[-1].outcome
        is ReporterPageResolutionOutcome.AMBIGUOUS
    )
    resolved = document
    calls = []

    async def select_opinion(context):
        calls.append(("opinion", context.citation_id))
        assert context.citation_id == leaf.id
        return ReporterCitationOpinionDecision(
            choices=(ReporterOpinionPageChoice(page_index=0, candidate_index=0),),
            reason="The body citation refers to the lead opinion.",
        )

    async def read_proposition(context):
        calls.append(("proposition", context.citation_id))
        assert context.citation_id == leaf.id
        return PropositionDecision(quotes=(proposition,), reason="The body attributes this proposition.")

    async def review_pages(context):
        calls.append(("pages", context.citation_id))
        assert context.citation_id == leaf.id
        return ReporterSupportDecision(
            result=OpinionSupportResult.UNAVAILABLE,
            evidence=(),
            pagination_available=True,
            correct_page=None,
            found_pages=(),
            reason="Review the complete opinion before deciding support.",
        )

    async def review_full_opinion(context):
        calls.append(("full", context.citation_id))
        assert context.citation_id == leaf.id
        return ReporterSupportDecision(
            result=OpinionSupportResult.SUPPORTED,
            evidence=(OpinionEvidenceQuote(opinion_id="20", quote=opinion_quote),),
            pagination_available=True,
            correct_page=True,
            found_pages=(PinCiteTarget(first=556, last=556, kind=PinCiteKind.PAGE),),
            reason="The lead opinion supports the body proposition at its written pinpoint.",
        )

    document = asyncio.run(review_reporter_citation_opinions(document, reviewer=select_opinion))
    document = asyncio.run(read_reporter_citation_propositions(document, reviewer=read_proposition))
    document = prepare_reporter_citation_pinpoint_evidence(document)
    document = asyncio.run(review_reporter_citation_pinpoint_pages(document, reviewer=review_pages))
    document = asyncio.run(review_reporter_citation_full_opinions(document, reviewer=review_full_opinion))
    document = judge_reporter_citation_pinpoints(document)
    root, leaf = document.roots[0], document.short_reporters[0]

    assert calls == [(substage, leaf.id) for substage in ("opinion", "proposition", "pages", "full")]
    assert root.identity_judgments == admitted.roots[0].identity_judgments
    assert (
        root.reporter_opinion_reviews
        == root.reporter_support_reviews
        == root.reporter_pinpoint_judgments
        == ()
    )
    assert root.reporter_opinion_evidence == ()
    assert root.reporter_pinpoint_evidence[-1].outcome is PinpointEvidenceOutcome.NO_PROPOSITION
    if root_has_pin:
        assert root.reporter_propositions[-1].passages == root.reporter_propositions[-1].decision.quotes == ()
        assert root.reporter_propositions[-1].ivr is None
    else:
        assert root.reporter_propositions == ()
    assert leaf.reporter_propositions[-1].passages[0].quote == proposition
    assert [review.scope for review in leaf.reporter_support_reviews] == [
        OpinionReviewScope.CITED_PAGES,
        OpinionReviewScope.FULL_OPINION,
    ]
    assert leaf.reporter_pinpoint_judgments[-1].verdict is ReporterPinpointVerdict.CORRECT_PINCITE
    restored = Document.model_validate_json(document.model_dump_json())
    assert restored == document
    assert restored.get_substage("test_identity") == admitted
    assert restored.get_substage("validate_pincite.citation_preparation.page_resolution") == resolved
