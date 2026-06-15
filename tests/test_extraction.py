"""Tests for citation extraction."""

from pathlib import Path

import pytest

from mellea_lrc.core.citations import CitationKind, FullCaseCitation, FullLawCitation
from mellea_lrc.extraction import extract, extract_citations, extract_documents
from mellea_lrc.preprocessing import preprocess_plain_text_from_string

TEST_DOCUMENTS_DIR = Path(__file__).parent / "test_documents"

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

    full_case = next(
        item for item in result.citations if isinstance(item.citation, FullCaseCitation)
    )
    assert full_case.citation.defendant == "Shelby County"
    assert full_case.citation.volume == "118"
    assert full_case.citation.reporter == "U.S."
    assert full_case.resolves_to is None

    full_law = next(
        item for item in result.citations if isinstance(item.citation, FullLawCitation)
    )
    assert full_law.citation.volume == "28"
    assert full_law.citation.reporter == "U.S.C."


@pytest.mark.skipif(not TEST_DOCUMENTS_DIR.exists(), reason="test_documents not present locally")
def test_extract_all_test_documents_finds_expected_full_citations() -> None:
    results = extract_documents(TEST_DOCUMENTS_DIR)
    assert len(results) == 5

    total_full = sum(len(result.full_citations) for result in results)
    assert total_full == 89

    by_type: dict[str, int] = {}
    for result in results:
        for item in result.citations:
            kind = item.citation.kind.value
            by_type[kind] = by_type.get(kind, 0) + 1

    assert by_type[CitationKind.FULL_CASE.value] == 88
    assert by_type[CitationKind.FULL_LAW.value] == 1
    assert by_type.get(CitationKind.FULL_JOURNAL.value, 0) == 0
