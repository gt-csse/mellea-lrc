"""Tests for offset-preserving masking and masked-text site hunting."""

from mellea_lrc.experimental import (
    mask_full_spans,
    mask_locator_spans,
    suspected_locators,
)
from mellea_lrc.extraction import extract_citations

_WELL_FORMED = "Norton v. Shelby County, 118 U.S. 425, 442 (1886), an unconstitutional act."
_DAMAGED = "Doe v. Colgate Univ. , 2016 WL1448829, at *2 (N.D.N.Y. Apr. 12, 2016)"


def test_masking_preserves_every_offset() -> None:
    """Downstream spans stay valid only if the text length never changes."""
    document = extract_citations(_WELL_FORMED)
    assert len(mask_locator_spans(document)) == len(document.text)
    assert len(mask_full_spans(document)) == len(document.text)


def test_locator_masking_hides_the_locator_but_keeps_its_context() -> None:
    document = extract_citations(_WELL_FORMED)
    masked = mask_locator_spans(document)
    assert "118 U.S. 425" not in masked
    assert "Norton v. Shelby County" in masked


def test_full_span_masking_also_removes_the_parenthetical_court() -> None:
    """Court abbreviations are gazetteer reporters; full-span masking removes them."""
    document = extract_citations("Doe v. Roe, 2016 WL 1448829, at *2 (N.D.N.Y. Apr. 12, 2016)")
    locator_masked = mask_locator_spans(document)
    full_masked = mask_full_spans(document)
    assert "N.D.N.Y." in locator_masked
    assert "N.D.N.Y." not in full_masked


def test_hunting_reports_a_reporter_that_produced_no_citation() -> None:
    """The damaged WL locator surfaces, alongside expected noise.

    Masking only removes citations that were *extracted*. This one was not, so
    the court abbreviation in its own parenthetical stays exposed and also
    qualifies. That is the intended trade: the filter is recall-oriented and a
    judge is expected to reject freely.
    """
    document = extract_citations(_DAMAGED)
    sites = suspected_locators(document)
    reporters = [site.reporter for site in sites]
    assert "WL" in reporters
    wl = next(site for site in sites if site.reporter == "WL")
    assert document.text[wl.span_start : wl.span_end] == "WL"


def test_hunting_ignores_what_was_already_extracted() -> None:
    document = extract_citations(_WELL_FORMED)
    assert suspected_locators(document) == ()


def test_hunting_requires_digits_on_both_sides() -> None:
    """A reporter string in prose is not a locator candidate."""
    document = extract_citations("The U.S. Supreme Court declined to hear the matter.")
    assert suspected_locators(document) == ()


def test_a_reported_span_indexes_the_original_document() -> None:
    """Offsets survive masking, so a judge can read the real text at that position."""
    document = extract_citations(_DAMAGED)
    site = next(s for s in suspected_locators(document) if s.reporter == "WL")
    assert document.text[site.span_start : site.span_end] == site.reporter
    assert "WL1448829" in site.window
