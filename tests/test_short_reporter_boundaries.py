"""Short pin completion respects independently recognized citation boundaries."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import (
    Document,
    attribute_id_citations,
    attribute_reference_citations,
    attribute_short_reporter_citations,
    find_id_citations,
    find_reference_citations,
    find_short_reporter_citations,
    find_supra_citations,
    grow_leaves,
    grow_roots,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
)
from mellea_lrc.extraction.short_reporter_locator import STAGE
from mellea_lrc.parsing.reporters import short_reporter_readings
from mellea_lrc.model.citations import IdCitation, ReferenceCitation, SupraCitation

ROOTS = (
    "Smith v. Jones, 347 U.S. 483 (1954). "
    "Green v. Brown, 100 F.3d 1 (2000). "
    "White v. Black, 200 F.2d 2 (1952). "
)


def _read_short_names(document: Document) -> Document:
    return resolve_short_reporter_case_names(resolve_short_reporter_colocations(document))


def _roots(source: str) -> Document:
    return asyncio.run(grow_roots(Document.from_source(source)))


@pytest.mark.parametrize(
    "quotes",
    [
        ("347 U.S. at 495", "100 F.3d at 20"),
        ("347 U.S. at 495", "100 F.3d at 20", "200 F.2d at 30"),
        ("347 U.S. at 495", "100F.3d at 20"),
    ],
)
def test_parallel_short_reporters_keep_separate_spans_and_pins(quotes: tuple[str, ...]) -> None:
    source = ROOTS + ", ".join(quotes) + "."
    roots = _roots(source)
    created = find_short_reporter_citations(roots)

    assert tuple(source[slice(*reading.span)] for reading in short_reporter_readings(source)) == quotes
    assert tuple(citation.short_locator[-1].quote for citation in created.short_reporters) == quotes
    assert tuple(citation.site_span.start for citation in created.short_reporters) == tuple(
        sorted(citation.site_span.start for citation in created.short_reporters)
    )
    assert [citation.get_pin_cite()[0].first for citation in created.short_reporters] == [495, 20, 30][
        : len(quotes)
    ]
    assert all(len(citation.get_pin_cite()) == 1 for citation in created.short_reporters)
    assert created.full_locators == roots.full_locators

    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), review=False))
    assert [citation.root_id[-1].value for citation in attributed.short_reporters] == [
        root.id for root in roots.roots[: len(quotes)]
    ]
    restored = Document.model_validate_json(attributed.model_dump_json())
    assert restored.get_stage(STAGE) == created
    assert restored.get_stage("10_roots") == roots


@pytest.mark.parametrize(
    "next_citation",
    [
        "100 F.3d 1 (2000).",
        "42 U.S.C. § 1983.",
        "Id. at 496.",
        "Smith, supra, at 496.",
    ],
)
def test_short_pin_stops_before_the_next_recognized_citation(next_citation: str) -> None:
    source = ROOTS + "See Smith, 347 U.S. at 495, " + next_citation
    roots = _roots(source)
    created = find_short_reporter_citations(roots)

    assert len(created.short_reporters) == 1
    short = created.short_reporters[0]
    assert short.short_locator[-1].quote == "347 U.S. at 495"
    assert short.pin_cite[-1].quote == "495"
    assert short.site_span.end <= source.rindex(next_citation)
    assert created.full_locators == roots.full_locators


@pytest.mark.parametrize("separator", [", ", ",\n ", "; "])
def test_boundary_preserves_page_lists_ranges_and_footnotes(separator: str) -> None:
    quote = "347 U.S. at\n 495 - 97, 501 n.2"
    source = ROOTS + "See Smith, " + quote + separator + "100 F.3d at 20."
    created = find_short_reporter_citations(_roots(source))

    assert [citation.short_locator[-1].quote for citation in created.short_reporters] == [
        quote,
        "100 F.3d at 20",
    ]
    first, second = created.short_reporters
    assert first.pin_cite[-1].quote == "495 - 97, 501 n.2"
    assert [(pin.first, pin.last, pin.footnote) for pin in first.get_pin_cite()] == [
        (495, 497, None),
        (501, 501, "2"),
    ]
    assert [(pin.first, pin.last) for pin in second.get_pin_cite()] == [(20, 20)]
    assert first.short_locator[-1].normalizable is second.short_locator[-1].normalizable is True
    for citation in created.short_reporters:
        for field in (citation.short_locator[-1], citation.pin_cite[-1]):
            assert source[field.span.start : field.span.end] == field.quote


def test_later_leaf_stages_preserve_roots_and_short_reporter_checkpoints() -> None:
    source = (
        ROOTS + "See Smith, 347 U.S. at 495, 100 F.3d at 20, 200 F.2d at 30. "
        "Smith at 498. Id. at 499. Smith, supra, at 500."
    )
    roots = _roots(source)
    created = find_short_reporter_citations(roots)
    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), review=False))
    references = find_reference_citations(attributed)
    references = asyncio.run(attribute_reference_citations(references, review=False))
    ids = find_id_citations(references)
    ids = asyncio.run(attribute_id_citations(ids, review=False))
    supra = find_supra_citations(ids)
    final = asyncio.run(grow_leaves(roots, review_leaves=False))

    assert len(final.short_reporters) == 3
    assert (
        len([citation for citation in final.short_citations if isinstance(citation, ReferenceCitation)]) == 1
    )
    assert len([citation for citation in final.short_citations if isinstance(citation, IdCitation)]) == 1
    assert len([citation for citation in final.short_citations if isinstance(citation, SupraCitation)]) == 1
    assert final.text == source
    assert final.full_locators == roots.full_locators
    assert final.short_reporters == attributed.short_reporters
    final_citations = {citation.id: citation for citation in final.citations}
    for citation in references.short_citations:
        assert final_citations[citation.id] == citation
    for citation in ids.short_citations:
        assert final_citations[citation.id] == citation
    restored = Document.model_validate_json(final.model_dump_json())
    assert restored == final
    assert restored.get_stage("10_roots") == roots
    assert restored.get_stage(STAGE) == created
    assert restored.get_stage("29_short_reporter_attribution") == attributed
    assert restored.get_stage("31_reference_attribution") == references
    assert restored.get_stage("33_id_attribution") == ids
    assert restored.get_stage("34_supra_citations") == supra


def test_finding_shorts_preserves_previously_created_leaf_objects() -> None:
    source = ROOTS + "Smith at 498. Id. at 499. Smith, supra, at 500. 347 U.S. at 495, 100 F.3d at 20."
    roots = _roots(source)
    earlier = find_supra_citations(find_id_citations(find_reference_citations(roots)))
    created = find_short_reporter_citations(earlier)

    assert len(created.short_reporters) == 2
    assert created.full_locators == earlier.full_locators
    created_citations = {citation.id: citation for citation in created.citations}
    assert tuple(created_citations[citation.id] for citation in earlier.citations) == earlier.citations
    restored = Document.model_validate_json(created.model_dump_json())
    assert restored.get_stage("34_supra_citations") == earlier
    assert restored.get_stage("10_roots") == roots


def test_id_follows_the_last_parallel_short_without_merging_existing_roots() -> None:
    source = (
        "Smith v. Jones, 347 U.S. 483, 100 F.3d 1 (1954). "
        "See Smith, 347 U.S. at 495, 100 F.3d at 20. Id. at 21."
    )
    roots = _roots(source)
    assert len(roots.roots) == 2
    first_root, second_root = roots.roots
    assert first_root.get_case_name() == second_root.get_case_name()

    created = find_short_reporter_citations(roots)
    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), review=False))
    assert len(attributed.short_reporters) == 2
    first_short, second_short = attributed.short_reporters
    assert first_short.root_id[-1].value == first_root.id
    assert second_short.root_id[-1].value == second_root.id

    ids = find_id_citations(attributed)
    final = asyncio.run(attribute_id_citations(ids, review=False))
    identity = next(citation for citation in final.short_citations if isinstance(citation, IdCitation))
    assert identity.site_span.start > second_short.site_span.end
    assert identity.pin_cite[-1].quote == "21"
    assert identity.root_id[-1].value == second_root.id
    assert final.roots == roots.roots
    assert final.short_reporters == attributed.short_reporters
    restored = Document.model_validate_json(final.model_dump_json())
    assert restored == final
    assert restored.get_stage("10_roots") == roots
    assert restored.get_stage(STAGE) == created
    assert restored.get_stage("29_short_reporter_attribution") == attributed
    assert restored.get_stage("32_id_citations") == ids
