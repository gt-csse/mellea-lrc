"""Occurrence pins resolve to shared pages without preferring an opinion type."""

from __future__ import annotations

import asyncio
import copy

import pytest

from mellea_lrc.api import (
    Document,
    grow_roots,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_opinion_retrieval,
)
from mellea_lrc.model.citations import IdCitation, ShortReporterCitation
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_page_resolution import ReporterPageResolutionOutcome
from mellea_lrc.model.citations.reporter_pages import (
    IndexedReporterOpinion,
    OpinionPage,
    ReporterRootOpinionPageIndex,
)
from mellea_lrc.model.span import Span
from mellea_lrc.validation.reporter_citation_page_resolution import (
    STAGE,
    resolve_reporter_citation_pages,
)
from mellea_lrc.validation.reporter_root_opinion_page_index import STAGE as INDEX_STAGE
from tests.test_reporter_root_opinion_retrieval import Client, opinion


def _span(source: str, quote: str, *, after: int = 0) -> Span:
    start = source.index(quote, after)
    return Span(start, start + len(quote))


def _indexed(identifier: str, *pages: tuple[str, PinCiteKind, int | None, str | None]):
    text = ""
    indexed_pages = []
    for label, kind, volume, edition in pages:
        body = f"Opinion {identifier}, page {label}. "
        indexed_pages.append(
            OpinionPage(
                label=label,
                kind=kind,
                volume=volume,
                edition=edition,
                span=Span(len(text), len(text) + len(body)),
            )
        )
        text += body
    return IndexedReporterOpinion(
        opinion_id=identifier,
        text_field="html_with_citations",
        text=text,
        pages=tuple(indexed_pages),
    )


def _page(label: str, *, volume: int | None = 550, edition: str | None = "U.S."):
    return label, PinCiteKind.PAGE, volume, edition


def _retrieved(source: str, *, shorts=(), ids=(), repeat_root_pin: bool = False) -> Document:
    document = asyncio.run(grow_roots(Document.from_source(source)))
    root = document.roots[0]
    for quote, pin in shorts:
        short = (
            ShortReporterCitation.from_short_locator(
                citation_id=f"short:{source.index(quote)}",
                stage="test_occurrences",
                source=source,
                span=_span(source, quote),
                pin_cite_span=_span(source, pin, after=source.index(quote)) if pin is not None else None,
            )
            .record("test_occurrences")
            .with_root(root.id)
        )
        document = document.add_citation(short)
    for quote, pin in ids:
        cite_span = _span(source, quote)
        leaf = (
            IdCitation.from_source(
                source=source,
                span=cite_span,
                stage="test_occurrences",
                pin_span=_span(source, pin, after=cite_span.start) if pin is not None else None,
            )
            .record("test_occurrences")
            .with_root(root.id)
        )
        document = document.add_citation(leaf)
    if repeat_root_pin:
        root = root.record("test_occurrences").with_pin_cite(source, root.pin_cite[-1].span)
        document = document.replace_citation(root)
    if shorts or ids or repeat_root_pin:
        document = document.complete("test_occurrences")
    client = Client(
        [{"id": 1, "sub_opinions": [20, 21]}],
        {"20": opinion("20"), "21": opinion("21", type="040dissent")},
    )
    document = reporter_root_lookup_cluster_retrieval(document, client=client)
    root = document.roots[0].record("test_identity").with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    document = document.replace_citation(root).complete("test_identity")
    return reporter_root_opinion_retrieval(document, client=client)


def _indexed_document(document: Document, *opinions: IndexedReporterOpinion) -> Document:
    root = document.roots[0].record(INDEX_STAGE)
    index = ReporterRootOpinionPageIndex(node_id=root.nodes[-1].id, cluster_id="1", opinions=opinions)
    return document.replace_citation(root.with_reporter_root_opinion_page_index(index)).complete(INDEX_STAGE)


def _resolution(citation):
    assert len(citation.reporter_page_resolutions) == 1
    return citation.reporter_page_resolutions[0]


@pytest.mark.parametrize("pin, opinion_id", [("556", "20"), ("570", "21")])
def test_root_own_pin_selects_lead_or_dissent_by_page(pin, opinion_id):
    before = _indexed_document(
        _retrieved(f"Twombly, 550 U.S. 544, {pin} (2007)."),
        _indexed("20", _page("556")),
        _indexed("21", _page("570")),
    )
    after = resolve_reporter_citation_pages(before)
    root = after.roots[0]
    result = _resolution(root)

    assert result.outcome is ReporterPageResolutionOutcome.RESOLVED
    assert result.locator_citation_id == result.pin_citation_id == result.root_id == root.id
    assert result.pages[0].candidates[0].opinion_id == opinion_id
    assert result.pages[0].candidates[0].pagination_confirmed is True
    assert root.identity_judgments == before.roots[0].identity_judgments
    assert root.pin_cite == before.roots[0].pin_cite
    assert root.root_id == before.roots[0].root_id
    assert root.reporter_root_opinion_page_index == before.roots[0].reporter_root_opinion_page_index


def test_shared_page_keeps_all_writers_as_candidates():
    before = _indexed_document(
        _retrieved("Twombly, 550 U.S. 544, 556 (2007)."),
        _indexed("20", _page("556")),
        _indexed("21", _page("556")),
    )
    result = _resolution(resolve_reporter_citation_pages(before).roots[0])

    assert result.outcome is ReporterPageResolutionOutcome.AMBIGUOUS
    assert [candidate.opinion_id for candidate in result.pages[0].candidates] == ["20", "21"]
    assert all(candidate.pagination_confirmed for candidate in result.pages[0].candidates)


def test_unknown_page_namespace_requires_review_even_for_a_single_writer():
    before = _indexed_document(
        _retrieved("Twombly, 550 U.S. 544, 556 (2007)."),
        _indexed("20", _page("556", volume=None, edition=None)),
        _indexed("21"),
    )
    result = _resolution(resolve_reporter_citation_pages(before).roots[0])

    assert result.outcome is ReporterPageResolutionOutcome.AMBIGUOUS
    assert len(result.pages[0].candidates) == 1
    assert result.pages[0].candidates[0].pagination_confirmed is False


@pytest.mark.parametrize("volume, edition", [(551, "U.S."), (550, "S. Ct.")])
def test_known_different_pagination_namespace_is_unlocated(volume, edition):
    before = _indexed_document(
        _retrieved("Twombly, 550 U.S. 544, 556 (2007)."),
        _indexed("20", _page("556", volume=volume, edition=edition)),
        _indexed("21"),
    )
    result = _resolution(resolve_reporter_citation_pages(before).roots[0])

    assert result.outcome is ReporterPageResolutionOutcome.UNLOCATED
    assert result.pages[0].candidates == ()


def test_page_and_star_targets_do_not_share_numbered_candidates():
    source = "Twombly, 550 U.S. 544, 556 (2007). Id. at *556."
    before = _indexed_document(
        _retrieved(source, ids=(("Id. at *556", "*556"),)),
        _indexed("20", _page("556")),
        _indexed("21", ("*556", PinCiteKind.STAR, 550, "U.S.")),
    )
    after = resolve_reporter_citation_pages(before)

    assert _resolution(after.roots[0]).pages[0].candidates[0].opinion_id == "20"
    result = _resolution(after.short_citations[0])
    assert result.outcome is ReporterPageResolutionOutcome.RESOLVED
    assert result.pages[0].kind is PinCiteKind.STAR
    assert result.pages[0].candidates[0].opinion_id == "21"


def test_short_reporter_uses_its_own_edition_and_pin():
    source = "Twombly, 550 U.S. 544, 556 (2007). Twombly, 127 S. Ct. at 1965."
    before = _indexed_document(
        _retrieved(source, shorts=(("127 S. Ct. at 1965", "1965"),)),
        _indexed("20", _page("556")),
        _indexed("21", _page("1965", volume=127, edition="S. Ct.")),
    )
    after = resolve_reporter_citation_pages(before)
    short = after.short_reporters[0]
    result = _resolution(short)

    assert result.outcome is ReporterPageResolutionOutcome.RESOLVED
    assert result.locator_citation_id == result.pin_citation_id == short.id
    assert result.pages[0].label == 1965
    assert result.pages[0].candidates[0].opinion_id == "21"
    assert short.short_locator == before.short_reporters[0].short_locator
    assert short.pin_cite == before.short_reporters[0].pin_cite
    assert short.root_id == before.short_reporters[0].root_id
    assert not hasattr(short, "reporter_root_opinion_page_index")


def test_unpinned_id_retains_the_immediate_short_reporter_sources():
    source = "Twombly, 550 U.S. 544, 556 (2007). Twombly, 127 S. Ct. at 1965. Id."
    before = _indexed_document(
        _retrieved(source, shorts=(("127 S. Ct. at 1965", "1965"),), ids=(("Id.", None),)),
        _indexed("20", _page("556")),
        _indexed("21", _page("1965", volume=127, edition="S. Ct.")),
    )
    after = resolve_reporter_citation_pages(before)
    short = after.short_reporters[0]
    leaf = next(citation for citation in after.short_citations if isinstance(citation, IdCitation))
    result = _resolution(leaf)

    assert result.outcome is ReporterPageResolutionOutcome.RESOLVED
    assert (result.locator_citation_id, result.locator_reading_index) == (short.id, 0)
    assert (result.pin_citation_id, result.pin_reading_index) == (short.id, 0)
    assert result.pages[0].candidates[0].opinion_id == "21"
    assert leaf.pin_cite is None
    assert leaf.root_id == before.short_citations[-1].root_id


def test_inherited_pin_pointer_is_an_absolute_history_index():
    source = "Twombly, 550 U.S. 544, 556 (2007). Id."
    before = _indexed_document(
        _retrieved(source, ids=(("Id.", None),), repeat_root_pin=True),
        _indexed("20", _page("556")),
        _indexed("21"),
    )
    after = resolve_reporter_citation_pages(before)
    result = _resolution(after.short_citations[0])

    assert result.outcome is ReporterPageResolutionOutcome.RESOLVED
    assert result.pin_citation_id == after.roots[0].id
    assert result.pin_reading_index == 1
    assert after.short_citations[0].pin_cite is None


def test_unknown_statute_interrupts_pin_inheritance_even_when_id_is_attached():
    source = "Twombly, 550 U.S. 544, 556 (2007). 42 U.S.C. § 1983. Id."
    before = _indexed_document(
        _retrieved(source, ids=(("Id.", None),)),
        _indexed("20", _page("556")),
        _indexed("21"),
    )
    after = resolve_reporter_citation_pages(before)
    leaf = after.short_citations[0]
    result = _resolution(leaf)

    assert result.outcome is ReporterPageResolutionOutcome.NO_PIN
    assert result.pages == ()
    assert result.pin_citation_id is result.pin_reading_index is None
    assert leaf.root_id == before.short_citations[0].root_id


def test_an_immediately_preceding_unpinned_occurrence_does_not_scan_back_for_a_pin():
    source = "Twombly, 550 U.S. 544, 556 (2007). Twombly, 550 U.S. at 557. Id."
    before = _indexed_document(
        _retrieved(source, shorts=(("550 U.S. at 557", None),), ids=(("Id.", None),)),
        _indexed("20", _page("556"), _page("557")),
        _indexed("21"),
    )
    after = resolve_reporter_citation_pages(before)
    leaf = next(citation for citation in after.short_citations if isinstance(citation, IdCitation))
    result = _resolution(leaf)

    assert result.outcome is ReporterPageResolutionOutcome.NO_PIN
    assert result.pin_citation_id is None


def test_absent_pin_has_its_own_outcome():
    before = _indexed_document(
        _retrieved("Twombly, 550 U.S. 544 (2007)."),
        _indexed("20", _page("556")),
        _indexed("21"),
    )
    result = _resolution(resolve_reporter_citation_pages(before).roots[0])

    assert result.outcome is ReporterPageResolutionOutcome.NO_PIN
    assert result.pin_citation_id is result.pin_reading_index is None
    assert result.pages == ()


@pytest.mark.parametrize("field", ["locator", "pin_cite"])
def test_failed_source_reading_is_unnormalizable_instead_of_absent(field):
    before = _retrieved("Twombly, 550 U.S. 544, 556 (2007).")
    saved = before.model_dump(mode="json")
    reading = saved["citations"][0][field][-1]
    reading.update(normalizable=False, normalized=None, normalization_error="Historical failed reading")
    before = _indexed_document(Document.model_validate(saved), _indexed("20", _page("556")), _indexed("21"))
    result = _resolution(resolve_reporter_citation_pages(before).roots[0])

    assert result.outcome is ReporterPageResolutionOutcome.UNNORMALIZABLE
    assert result.pin_citation_id == result.locator_citation_id == before.roots[0].id
    assert result.pages == ()


def test_range_can_cross_writings_and_expands_every_requested_page():
    before = _indexed_document(
        _retrieved("Twombly, 550 U.S. 544, 556-58, 560 (2007)."),
        _indexed("20", _page("556"), _page("557")),
        _indexed("21", _page("558"), _page("560")),
    )
    result = _resolution(resolve_reporter_citation_pages(before).roots[0])

    assert result.outcome is ReporterPageResolutionOutcome.RESOLVED
    assert [(page.target_index, page.label) for page in result.pages] == [
        (0, 556),
        (0, 557),
        (0, 558),
        (1, 560),
    ]
    assert [page.candidates[0].opinion_id for page in result.pages] == ["20", "20", "21", "21"]
    assert [page.candidates[0].page_index for page in result.pages] == [0, 1, 0, 1]


def test_whitespace_in_printed_labels_and_reporter_names_keeps_literal_matching():
    before = _indexed_document(
        _retrieved("Twombly, 550 U.S. 544, 556 (2007)."),
        _indexed("20", _page(" 556\n", edition="U. S.")),
        _indexed("21"),
    )
    result = _resolution(resolve_reporter_citation_pages(before).roots[0])

    assert result.outcome is ReporterPageResolutionOutcome.RESOLVED
    assert result.pages[0].candidates[0].opinion_id == "20"


def test_native_json_preserves_shared_index_and_recovers_exact_checkpoints():
    at39 = _retrieved("Twombly, 550 U.S. 544, 556 (2007). Id.", ids=(("Id.", None),))
    at40 = _indexed_document(at39, _indexed("20", _page("556")), _indexed("21"))
    at41 = resolve_reporter_citation_pages(at40)
    saved = Document.model_validate_json(at41.model_dump_json())

    assert saved == at41
    assert saved.get_stage("39_reporter_root_opinion_retrieval") == at39
    assert saved.get_stage(INDEX_STAGE) == at40
    assert saved.get_stage(STAGE) == at41
    assert all(
        citation.reporter_page_resolutions == () for citation in saved.get_stage(INDEX_STAGE).citations
    )
    assert saved.roots[0].reporter_root_opinion_retrieval == at39.roots[0].reporter_root_opinion_retrieval
    with pytest.raises(ValueError, match="already completed"):
        resolve_reporter_citation_pages(saved)


def test_resolution_requires_completed_page_index_stage():
    before = _retrieved("Twombly, 550 U.S. 544, 556 (2007).")
    with pytest.raises(ValueError, match="Index reporter-root opinion pages"):
        resolve_reporter_citation_pages(before)


@pytest.mark.parametrize(
    "corruption",
    ["root", "locator_citation", "locator_reading", "pin_citation", "pin_reading", "opinion", "page"],
)
def test_document_loading_rejects_resolution_pointers_outside_saved_histories(corruption):
    at40 = _indexed_document(
        _retrieved("Twombly, 550 U.S. 544, 556 (2007). Id.", ids=(("Id.", None),)),
        _indexed("20", _page("556")),
        _indexed("21"),
    )
    saved = copy.deepcopy(resolve_reporter_citation_pages(at40).model_dump(mode="json"))
    result = saved["citations"][-1]["reporter_page_resolutions"][0]
    if corruption == "root":
        result["root_id"] = "unknown_root"
    elif corruption == "locator_citation":
        result["locator_citation_id"] = saved["citations"][-1]["id"]
    elif corruption == "locator_reading":
        result["locator_reading_index"] = 100
    elif corruption == "pin_citation":
        result["pin_citation_id"] = saved["citations"][-1]["id"]
    elif corruption == "pin_reading":
        result["pin_reading_index"] = 100
    elif corruption == "opinion":
        result["pages"][0]["candidates"][0]["opinion_id"] = "999"
    else:
        result["pages"][0]["candidates"][0]["page_index"] = 100

    with pytest.raises(ValueError):
        Document.model_validate(saved)
