"""Short reporter occurrences stay separate from full locator roots."""

import asyncio

import pytest
from eyecite.models import FullCaseCitation, ShortCaseCitation

from mellea_lrc.api import Document, find_short_reporter_citations, grow_roots
from mellea_lrc.model import FullReporterCitation, ShortReporterCitation, Span
from mellea_lrc.extraction.full_reporter_locator import full_reporter_readings
from mellea_lrc.extraction.short_reporter_locator import short_reporter_readings


def test_stage_readers_share_eyecite_matching_for_full_and_short_kinds() -> None:
    source = "Smith v. Jones, 347 U.S. 483. See Smith, 347 U.S. at 495."
    full = full_reporter_readings(source)
    short = short_reporter_readings(source)

    assert [type(reading.citation) for reading in full + short] == [FullCaseCitation, ShortCaseCitation]
    assert [source[slice(*reading.span)] for reading in full + short] == ["347 U.S. 483", "347 U.S. at 495"]


def test_short_reporter_is_a_distinct_checkpointed_citation() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495 n.4."
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    document = find_short_reporter_citations(roots)

    assert len(document.full_locators) == len(document.roots) == 1
    assert isinstance(document.full_locators[0], FullReporterCitation)
    assert len(document.short_reporters) == 1
    short = document.short_reporters[0]
    assert isinstance(short, ShortReporterCitation)
    assert not hasattr(short, "locator_span")
    assert short.short_locator_span == Span(
        source.index("347 U.S. at 495"), source.index("347 U.S. at 495") + 15
    )
    assert short.short_locator[-1].quote == "347 U.S. at 495"
    assert short.short_locator[-1].get_normalized().pin_page == "495"
    assert short.root_id == ()
    assert document.get_stage("roots") == roots
    assert document.get_stage("short_reporter_citations") == document
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_short_reporter_discovery_is_optional_and_rejects_a_repeat_run() -> None:
    source = "See Smith, 347 U.S. at 495."
    with pytest.raises(ValueError, match="Form full roots"):
        find_short_reporter_citations(Document.from_source(source))
    before = asyncio.run(grow_roots(Document.from_source(source)))
    after = find_short_reporter_citations(before)

    assert before.citations == ()
    assert after.full_locators == after.roots == ()
    assert len(after.short_reporters) == 1
    with pytest.raises(ValueError):
        find_short_reporter_citations(after)


def test_short_reporter_can_later_attach_without_changing_its_checkpoint() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495."
    found = find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))
    short = found.short_reporters[0]
    attached = found.replace_citation(short.record("attach_short").with_root(found.roots[0].id))
    attached = attached.complete("attach_short")

    assert attached.short_reporters[0].root_id[-1].value == found.roots[0].id
    assert attached.get_stage("short_reporter_citations") == found
    assert Document.model_validate_json(attached.model_dump_json()) == attached
