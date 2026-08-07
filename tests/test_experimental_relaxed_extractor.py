"""Tests for the experimental layout-tolerant eyecite extractor."""

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.experimental import extract_relaxed_citations
from mellea_lrc.extraction import ExtractedDocument, extract_citations


def _locators(document: ExtractedDocument) -> set[str]:
    return {
        f"{c.citation.volume} {c.citation.reporter} {c.citation.page}"
        for c in document.citations
        if isinstance(c.citation, FullCaseCitation)
    }


def test_recovers_a_space_lost_between_reporter_and_page() -> None:
    """PDF table extraction drops the space; the baseline extractor sees nothing."""
    text = "Doe v. Colgate Univ. , 2016 WL1448829, at *2 (N.D.N.Y. Apr. 12, 2016)"
    assert _locators(extract_citations(text)) == set()
    assert "2016 WL 1448829" in _locators(extract_relaxed_citations(text))


def test_recovers_a_volume_split_from_its_reporter_by_a_page_break() -> None:
    text = "See also White v. McBride , 937\n\nS.W.2d  796,  800  (Tenn.  1996)"
    assert _locators(extract_citations(text)) == set()
    assert "937 S.W.2d 796" in _locators(extract_relaxed_citations(text))


def test_recovers_doubled_whitespace_without_rewriting_the_text() -> None:
    text = "Cracker Barrel Old  Country  Store,  Inc.  v.  Epperson ,  284  S.W.3d  303,  312"
    assert "284 S.W.3d 303" in _locators(extract_relaxed_citations(text))


def test_leaves_well_formed_citations_unchanged() -> None:
    text = "Norton v. Shelby County, 118 U.S. 425, 442 (1886)"
    assert _locators(extract_relaxed_citations(text)) == _locators(extract_citations(text))


def test_reporter_groups_are_not_left_with_absorbed_whitespace() -> None:
    """Relaxed separators let alternation branches ending in \\s* keep a space."""
    text = "United States v. Rucker , 188 Fed. Appx. 772, 778 (10th Cir. 2006)"
    reporters = {
        c.citation.reporter
        for c in extract_relaxed_citations(text).citations
        if isinstance(c.citation, FullCaseCitation)
    }
    assert reporters
    assert all(r == r.strip() for r in reporters if r)


def test_returns_a_plain_extracted_document_with_usable_spans() -> None:
    """No text is rewritten, so spans index directly into document.text."""
    text = "Doe v. Colgate Univ. , 2016 WL1448829, at *2 (N.D.N.Y. Apr. 12, 2016)"
    document = extract_relaxed_citations(text)
    assert isinstance(document, ExtractedDocument)
    citation = next(c for c in document.citations if isinstance(c.citation, FullCaseCitation))
    assert document.text[citation.locator_span.start : citation.locator_span.end] == "2016 WL1448829"
    assert document.text == extract_citations(text).text


def test_a_page_break_before_margin_line_numbers_yields_a_wrong_page() -> None:
    """Known limitation, asserted so it cannot regress silently.

    Relaxing the separator to \\s* also matches newlines, which is what
    recovers page-break splits. The same behaviour makes PDF margin line
    numbers look like a page: the real citation is 214 F.3d 1058.
    """
    text = "Advanced Textile , 214 F.3d\n\n1\n\n2\n\n3\n\n4"
    assert "214 F.3d 1" in _locators(extract_relaxed_citations(text))
