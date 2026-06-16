"""Tests for citation extraction."""

from mellea_lrc.core.citations import CitationKind, FullCaseCitation, FullLawCitation
from mellea_lrc.extraction import extract, extract_citations
from mellea_lrc.preprocessing import preprocess_plain_text_from_string

SAMPLE_TEXT = (
    "Under Norton v. Shelby County, 118 U.S. 425, 442 (1886), an unconstitutional "
    "act confers no rights. See also Fed. R. Civ. P. 72(a) and 28 U.S.C. § 636(b)(1)(A)."
)


def test_extract_accepts_preprocessed_document() -> None:
    preprocessed = preprocess_plain_text_from_string(SAMPLE_TEXT)
    result = extract(preprocessed)
    assert result.preprocessed is preprocessed
    assert result.text == SAMPLE_TEXT
    assert result.citations


def test_extract_citations_returns_canonical_types() -> None:
    result = extract_citations(SAMPLE_TEXT)
    kinds = {item.citation.kind for item in result.citations}

    assert CitationKind.FULL_CASE in kinds
    assert CitationKind.FULL_LAW in kinds
    assert len(result.full_citations) >= 2

    full_case = next(item for item in result.citations if isinstance(item.citation, FullCaseCitation))
    assert full_case.citation.defendant == "Shelby County"
    assert full_case.citation.volume == "118"
    assert full_case.citation.reporter == "U.S."
    assert full_case.resolves_to is None

    full_law = next(item for item in result.citations if isinstance(item.citation, FullLawCitation))
    assert full_law.citation.volume == "28"
    assert full_law.citation.reporter == "U.S.C."
