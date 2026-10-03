"""Deterministic pagination indexing for saved reporter opinions."""

from types import SimpleNamespace

from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind
from mellea_lrc.model.citations.reporter_opinion import (
    OpinionRetrievalOutcome,
    ReporterRootOpinionRetrieval,
    ReporterRootOpinionSource,
    RetrievedReporterOpinion,
)
from mellea_lrc.model.citations.reporter_pages import (
    IndexedReporterOpinion,
    OpinionPage,
    ReporterRootOpinionPageIndex,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.reporter_root_opinion_page_index import (
    SOURCE_STAGE,
    STAGE,
    _index_opinion,
    _infer_page_namespaces,
    _render_html,
    _ReporterCandidate,
    _selected_cluster_candidates,
    index_reporter_root_opinion_pages,
)


def _opinion(identifier: str, cluster: str, **fields):
    return RetrievedReporterOpinion(
        opinion_id=identifier,
        cluster_id=cluster,
        outcome=OpinionRetrievalOutcome.RETRIEVED,
        response={
            "id": int(identifier),
            "cluster": f"https://www.courtlistener.com/api/rest/v4/clusters/{cluster}/",
            **fields,
        },
    )


def test_render_html_indexes_distinct_page_namespaces_and_exact_spans():
    text, pages = _render_html(
        '<p><page-number citation-index="1" label="101">*101</page-number>'
        "Alpha <em>bold</em>.</p>"
        '<p><span class="star-pagination" data-kind="star">*7</span>Beta &amp; gamma.</p>'
        '<p><span class="paragraph-number">¶12</span>Paragraph body.</p>'
        '<div class="footnote" id="fn1"><a class="footnote">1</a> '
        "Footnote text.</div>"
    )

    assert text == "Alpha bold. Beta & gamma. Paragraph body. 1 Footnote text."
    assert [page.label for page in pages] == ["101", "*7", "¶12"]
    assert [page.kind for page in pages] == [
        PinCiteKind.PAGE,
        PinCiteKind.STAR,
        PinCiteKind.PARAGRAPH,
    ]
    assert pages[0].citation_index == "1"
    assert [text[page.span.start : page.span.end] for page in pages] == [
        "Alpha bold. Beta & gamma. Paragraph body. 1 Footnote text.",
        "Beta & gamma. Paragraph body. 1 Footnote text.",
        "Paragraph body. 1 Footnote text.",
    ]
    # CourtListener's browser asterisk is not treated as a Westlaw star page.
    assert pages[0].volume is None and pages[0].edition is None
    assert pages[1].volume is None and pages[1].edition is None


def test_star_pagination_class_alone_does_not_assert_westlaw_star_semantics():
    text, pages = _render_html('<p><span class="star-pagination">*660</span>Opinion text.</p>')

    assert text == "Opinion text."
    assert len(pages) == 1
    assert pages[0].label == "660"
    assert pages[0].kind is None
    assert text[pages[0].span.start : pages[0].span.end] == "Opinion text."


def test_star_marker_keeps_unknown_namespace_until_cluster_citations_disambiguate_it():
    opinion = _index_opinion(
        _opinion(
            "31",
            "1",
            html_with_citations=(
                '<span class="star-pagination" citation-index="1" label="*1940">*1940</span>Opinion text.'
            ),
        )
    )
    ambiguous_candidates = (
        _ReporterCandidate(volume=556, edition="U.S.", first_page=662),
        _ReporterCandidate(volume=129, edition="S. Ct.", first_page=1937),
    )
    ambiguous = _infer_page_namespaces((opinion,), ambiguous_candidates)[0].pages[0]
    assert ambiguous.kind is None and ambiguous.volume is None and ambiguous.edition is None

    mapped = _infer_page_namespaces((opinion,), ambiguous_candidates[:1])[0].pages[0]
    assert mapped.kind is PinCiteKind.PAGE
    assert (mapped.volume, mapped.edition, mapped.pagination_inferred) == (556, "U.S.", True)


def test_interleaved_pagination_spans_end_at_next_marker_in_the_same_namespace():
    text, pages = _render_html(
        '<page-number label="678" citation-index="1" volume="556" edition="U.S.">678</page-number>'
        "Lead text. "
        '<paragraph-number label="12" citation-index="1">¶12</paragraph-number>Paragraph text. '
        '<page-number label="679" citation-index="1" volume="556" edition="U.S.">679</page-number>'
        "Later lead. "
        '<page-number label="1938" citation-index="2" volume="129" edition="S. Ct.">1938</page-number>'
        "Parallel text."
    )

    assert [page.kind for page in pages] == [
        PinCiteKind.PAGE,
        PinCiteKind.PARAGRAPH,
        PinCiteKind.PAGE,
        PinCiteKind.PAGE,
    ]
    assert [text[page.span.start : page.span.end] for page in pages] == [
        "Lead text. Paragraph text.",
        "Paragraph text. Later lead. Parallel text.",
        "Later lead. Parallel text.",
        "Parallel text.",
    ]


def test_explicit_page_metadata_is_retained_and_unknown_markers_remain_text():
    text, pages = _render_html(
        '<page-number label="9" citation-index="2" volume="12" edition="F.3d">'
        "*9</page-number>Opinion text. "
        '<page-number label="unknown">unparsed label</page-number>'
    )

    assert text == "Opinion text. unparsed label"
    assert len(pages) == 1
    assert pages[0].label == "9"
    assert pages[0].kind is PinCiteKind.PAGE
    assert (pages[0].volume, pages[0].edition, pages[0].citation_index) == (12, "F.3d", "2")
    assert text[pages[0].span.start : pages[0].span.end] == "Opinion text. unparsed label"


def test_plain_text_and_empty_or_missing_opinion_text_do_not_invent_pages():
    plain = _opinion("10", "1", plain_text="  saved plain text\n")
    indexed_plain = _index_opinion(plain)
    assert indexed_plain.text_field == "plain_text"
    assert indexed_plain.text == "  saved plain text\n"
    assert indexed_plain.pages == ()

    empty = RetrievedReporterOpinion(
        opinion_id="11", cluster_id="1", outcome=OpinionRetrievalOutcome.NOT_FOUND, response=None
    )
    indexed_empty = _index_opinion(empty)
    assert indexed_empty.text_field is None and indexed_empty.text == "" and indexed_empty.pages == ()

    empty_html = RetrievedReporterOpinion(
        opinion_id="12",
        cluster_id="1",
        outcome=OpinionRetrievalOutcome.EMPTY_TEXT,
        response={
            "id": 12,
            "cluster": "https://www.courtlistener.com/api/rest/v4/clusters/1/",
            "html_with_citations": "",
        },
    )
    indexed_empty_html = _index_opinion(empty_html)
    assert indexed_empty_html.text_field is None
    assert indexed_empty_html.text == "" and indexed_empty_html.pages == ()


def test_opinion_uses_alternate_saved_xml_when_selected_html_has_no_page_markers():
    indexed = _index_opinion(
        _opinion(
            "13",
            "1",
            html_with_citations="<p>Unpaginated selected HTML.</p>",
            xml_harvard=(
                '<opinion><page-number label="678" citation-index="1">678</page-number>'
                "Paginated XML text.</opinion>"
            ),
        )
    )

    assert indexed.text_field == "xml_harvard"
    assert indexed.text == "Paginated XML text."
    assert indexed.pages[0].label == "678"
    assert indexed.text[indexed.pages[0].span.start : indexed.pages[0].span.end] == "Paginated XML text."


def test_stage_requires_stage_39_and_recovers_completed_checkpoint():
    document = Document.from_source("A filing without reporter citations.")
    try:
        index_reporter_root_opinion_pages(document)
    except ValueError as error:
        assert "Retrieve reporter-root opinions" in str(error)
    else:
        raise AssertionError("Page indexing must require stage 39")

    before = document.model_copy(update={"stage_runs": (SOURCE_STAGE,)})
    after = index_reporter_root_opinion_pages(before)
    restored = Document.model_validate_json(after.model_dump_json())
    assert after.stage_runs == (SOURCE_STAGE, STAGE)
    assert restored == after
    assert restored.get_stage(STAGE) == after


def test_stage_attaches_index_to_reporter_root_and_recovers_it_from_checkpoint():
    from mellea_lrc.model.citations.full_reporter import FullReporterCitation
    from mellea_lrc.providers.courtlistener.models import CourtListenerCluster

    source = "550 U.S. 544"
    citation = FullReporterCitation.from_locator(
        citation_id="citation:1:1",
        stage="1_full_reporter_locators",
        source=source,
        span=Span(0, len(source)),
    )
    document = Document.from_source(source).add_citation(citation).complete("1_full_reporter_locators")
    citation = document.citations[0].record("10_roots").with_root("citation:1:1")
    document = document.replace_citation(citation).complete("10_roots")
    citation = document.roots[0].record("bind_original_source")
    citation = citation.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=citation.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate(
                {
                    "id": 1,
                    "sub_opinions": [20],
                    "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                }
            ),
        )
    )
    source_checkpoint = document.replace_citation(citation).complete("bind_original_source")
    document = source_checkpoint
    citation = document.roots[0].record(SOURCE_STAGE)
    citation = citation.with_reporter_root_opinion_retrieval(
        ReporterRootOpinionRetrieval(
            node_id=citation.nodes[-1].id,
            cluster_id="1",
            sub_opinion_ids=("20",),
            opinions=(
                RetrievedReporterOpinion(
                    opinion_id="20",
                    cluster_id="1",
                    outcome=OpinionRetrievalOutcome.RETRIEVED,
                    response={
                        "id": 20,
                        "cluster": "https://www.courtlistener.com/api/rest/v4/clusters/1/",
                        "html_with_citations": '<page-number label="544">544</page-number>Opinion.',
                    },
                ),
            ),
        )
    )
    document = document.replace_citation(citation).complete(SOURCE_STAGE)
    indexed = index_reporter_root_opinion_pages(document)
    restored = Document.model_validate_json(indexed.model_dump_json())

    result = restored.roots[0].reporter_root_opinion_page_index
    assert indexed.stage_runs[-1] == STAGE
    assert result is not None and result.opinions[0].opinion_id == "20"
    assert result.opinions[0].pages[0].span == Span(0, len("Opinion."))
    assert result.opinions[0].pages[0].pagination_inferred
    assert (result.opinions[0].pages[0].volume, result.opinions[0].pages[0].edition) == (550, "U.S.")
    assert restored.roots[0].reporter_exact_lookup is None
    assert restored.roots[0].identity_judgments == ()
    assert restored.get_stage("bind_original_source") == source_checkpoint
    assert restored.get_stage(STAGE) == indexed


def test_page_namespace_maps_only_to_one_compatible_reporter_citation():
    html_text = (
        '<page-number citation-index="1" label="678">*678</page-number>Lead page. '
        '<page-number citation-index="1" label="700">*700</page-number>Later lead page.'
    )
    first = _index_opinion(_opinion("20", "1", html_with_citations=html_text))
    second = _index_opinion(
        _opinion(
            "21",
            "1",
            html_with_citations='<page-number citation-index="2" label="1940">*1940</page-number>Second namespace.',
        )
    )
    cluster = SimpleNamespace(
        id="1",
        raw_json={
            "citations": [
                {"volume": "556", "reporter": "U.S.", "page": "662"},
                {"volume": "129", "reporter": "S.Ct.", "page": "1937"},
                {"volume": "78", "reporter": "L.Ed.2d", "page": "868"},
                {
                    "volume": "2020",
                    "reporter": "WL",
                    "page": "123",
                    "cite_type": "specialty_west",
                },
            ]
        },
    )
    citation = SimpleNamespace(reporter_root_opinion_source=SimpleNamespace(cluster=cluster))
    candidates = _selected_cluster_candidates(citation)
    indexed = _infer_page_namespaces((first, second), candidates)

    assert {(item.volume, item.edition, item.first_page) for item in candidates} == {
        (556, "U.S.", 662),
        (129, "S. Ct.", 1937),
        (78, "L. Ed. 2d", 868),
    }
    pages = [page for item in indexed for page in item.pages]
    assert [(page.volume, page.edition, page.pagination_inferred) for page in pages] == [
        (556, "U.S.", True),
        (556, "U.S.", True),
        (None, None, False),
    ]
    assert [page.citation_index for page in pages] == ["1", "1", "2"]
    assert (
        indexed[1].text[indexed[1].pages[0].span.start : indexed[1].pages[0].span.end] == "Second namespace."
    )


def test_ambiguous_reporter_candidates_keep_page_namespace_unmapped():
    opinion = _index_opinion(
        _opinion(
            "30",
            "1",
            html_with_citations='<page-number citation-index="1" label="678">*678</page-number>Page.',
        )
    )
    cluster = SimpleNamespace(
        id="1",
        raw_json={
            "citations": [
                {"volume": "556", "reporter": "U.S.", "page": "662"},
                {"volume": "1", "reporter": "F.Supp.3d", "page": "600"},
            ]
        },
    )
    citation = SimpleNamespace(reporter_root_opinion_source=SimpleNamespace(cluster=cluster))
    indexed = _infer_page_namespaces((opinion,), _selected_cluster_candidates(citation))

    page = indexed[0].pages[0]
    assert (page.volume, page.edition, page.pagination_inferred) == (None, None, False)


def test_index_model_round_trips_exact_page_slices():
    index = ReporterRootOpinionPageIndex(
        node_id="citation:1:2:node:40",
        cluster_id="123",
        opinions=(
            IndexedReporterOpinion(
                opinion_id="456",
                text_field="html_with_citations",
                text="Page text.",
                pages=(
                    OpinionPage(
                        label="1",
                        kind=PinCiteKind.PAGE,
                        span=Span(0, 10),
                        volume=12,
                        edition="F.3d",
                    ),
                ),
            ),
        ),
    )

    restored = ReporterRootOpinionPageIndex.model_validate_json(index.model_dump_json())
    assert restored == index
    assert (
        restored.opinions[0].text[
            restored.opinions[0].pages[0].span.start : restored.opinions[0].pages[0].span.end
        ]
        == "Page text."
    )


def test_inferred_namespace_unites_unknown_and_explicit_page_boundaries():
    indexed = _index_opinion(
        _opinion(
            "20",
            "1",
            html_with_citations=(
                '<span class="star-pagination" citation-index="1">*556</span>First page. '
                '<page-number citation-index="1" label="557">557</page-number>Second page.'
            ),
        )
    )
    merged = _infer_page_namespaces((indexed,), (_ReporterCandidate(550, "U.S.", 544),))[0]
    restored = IndexedReporterOpinion.model_validate_json(merged.model_dump_json())
    assert [page.kind for page in restored.pages] == [PinCiteKind.PAGE, PinCiteKind.PAGE]
    assert [restored.text[page.span.start : page.span.end] for page in restored.pages] == [
        "First page.",
        "Second page.",
    ]


def test_nested_head_style_does_not_hide_opinion_body():
    text, pages = _render_html(
        "<head><style>p{display:block}</style><script>untrusted()</script></head>"
        '<page-number label="556">556</page-number><p>Opinion body.</p>'
    )
    assert text == "Opinion body."
    assert pages[0].label == "556"
