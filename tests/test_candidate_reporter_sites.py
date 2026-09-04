"""Tests for offset-preserving masking and masked-text site hunting."""

from mellea_lrc.extraction.adjudication import (
    mask_full_spans,
    mask_locator_spans,
    suspected_locators,
)
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

_WELL_FORMED = "Norton v. Shelby County, 118 U.S. 425, 442 (1886), an unconstitutional act."
_DAMAGED = "Doe v. Colgate Univ. , 2016 WL1448829, at *2 (N.D.N.Y. Apr. 12, 2016)"


def _unrelaxed(text: str):
    """Extract without relaxing the separators.

    Hunting exists for locators no pattern reached, so these tests need a
    citation the extractor misses. The shipped `BOUNDED` setting now finds the
    damaged one, which is the point of it -- so the miss has to be produced
    deliberately rather than found.
    """
    return extract_from_plain_text(text, relaxation=Relaxation.NONE)


def test_masking_preserves_every_offset() -> None:
    """Downstream spans stay valid only if the text length never changes."""
    document = extract_from_plain_text(_WELL_FORMED)
    assert len(mask_locator_spans(document)) == len(document.text)
    assert len(mask_full_spans(document)) == len(document.text)


def test_locator_masking_hides_the_locator_but_keeps_its_context() -> None:
    document = extract_from_plain_text(_WELL_FORMED)
    masked = mask_locator_spans(document)
    assert "118 U.S. 425" not in masked
    assert "Norton v. Shelby County" in masked


def test_full_span_masking_also_removes_the_parenthetical_court() -> None:
    """Court abbreviations are gazetteer reporters; full-span masking removes them."""
    document = extract_from_plain_text("Doe v. Roe, 2016 WL 1448829, at *2 (N.D.N.Y. Apr. 12, 2016)")
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
    document = _unrelaxed(_DAMAGED)
    sites = suspected_locators(document)
    reporters = [site.reporter for site in sites]
    assert "WL" in reporters
    wl = next(site for site in sites if site.reporter == "WL")
    assert document.text[wl.span_start : wl.span_end] == "WL"


def test_hunting_ignores_what_was_already_extracted() -> None:
    document = extract_from_plain_text(_WELL_FORMED)
    assert suspected_locators(document) == ()


def test_hunting_requires_digits_on_both_sides() -> None:
    """A reporter string in prose is not a locator candidate."""
    document = extract_from_plain_text("The U.S. Supreme Court declined to hear the matter.")
    assert suspected_locators(document) == ()


def test_a_reported_span_indexes_the_original_document() -> None:
    """Offsets survive masking, so a judge can read the real text at that position."""
    document = _unrelaxed(_DAMAGED)
    site = next(s for s in suspected_locators(document) if s.reporter == "WL")
    assert document.text[site.span_start : site.span_end] == site.reporter
    assert "WL1448829" in site.window


def test_hunting_declines_a_section_rather_than_offering_it_as_a_page() -> None:
    """A section sign between the reporter and the number means it is not a page.

    `100 U.S. § 45` has the volume-and-page shape a cheap filter looks for, and
    a site sent to a judge as a suspected case locator is a statute offered up
    for a case-law verdict. On false-citation-bench this drops 16 of 65 sites,
    every one of them a statute eyecite had failed to parse and so had not
    masked.
    """
    document = _unrelaxed("The report cites 100 U.S. § 45 in passing.")

    assert suspected_locators(document) == ()


def test_the_section_sign_is_read_before_masking_can_erase_it() -> None:
    """eyecite emits a token for a bare `§`, and masking blanks whatever it emits.

    Read from the masked copy, the one character that says "section, not page"
    is gone by the time the filter looks for it.
    """
    document = _unrelaxed("The report cites 100 U.S. § 45 in passing.")

    assert "§" not in mask_full_spans(document)
    assert "§" in document.text


def test_a_sites_window_hides_the_citations_already_read() -> None:
    """The question is about this site, so a neighbour must not be quotable.

    Left visible, a reviewer reads the citation beside the candidate and returns
    a locator the record already holds -- which is what happened when the whole
    layer was run against the bench.
    """
    text = "Doe v. Roe, 550 U.S. 544, 570 (2007). Offices at 1301 McKinney, Suite 5100 Houston, Texas 77010."
    document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    site = next(s for s in suspected_locators(document) if s.reporter == "Houston")

    # The locator that was read is gone, so it cannot be quoted back.
    assert "550 U.S. 544" not in site.window
    # Everything unread is untouched, including the letterhead this site is.
    assert "Houston, Texas 77010" in site.window


def test_the_site_itself_is_never_hidden() -> None:
    """Masking removes what was extracted, and a candidate is what was not."""
    document = _unrelaxed(_DAMAGED)
    site = next(s for s in suspected_locators(document) if s.reporter == "WL")

    assert site.reporter in site.window
    assert document.text[site.span_start : site.span_end] == site.reporter
