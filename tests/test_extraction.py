"""Tests for citation extraction."""

from mellea_lrc.core.citations import CitationKind, FullCaseCitation, FullLawCitation
import pytest

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import (
    ExtractedCitation,
    ExtractedDocument,
    ExtractionMetadata,
    extract,
    extract_citations,
)
from mellea_lrc.preprocessing import PreprocessedDocument, preprocess_plain_text_from_string

SAMPLE_TEXT = (
    "Under Norton v. Shelby County, 118 U.S. 425, 442 (1886), an unconstitutional "
    "act confers no rights. See also Fed. R. Civ. P. 72(a) and 28 U.S.C. § 636(b)(1)(A)."
)


def test_extract_accepts_preprocessed_document() -> None:
    preprocessed = preprocess_plain_text_from_string(SAMPLE_TEXT)
    result = extract(preprocessed)
    assert isinstance(result, PreprocessedDocument)
    assert result.source_metadata is preprocessed.source_metadata
    assert result.preprocessing_metadata is preprocessed.preprocessing_metadata
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


def test_extracted_document_rejects_duplicate_citation_ids() -> None:
    preprocessed = preprocess_plain_text_from_string("347 U.S. 483")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, len(preprocessed.text)),
        matched_text=preprocessed.text,
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )

    with pytest.raises(ValueError, match="must be unique"):
        ExtractedDocument(
            source_metadata=preprocessed.source_metadata,
            text=preprocessed.text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(citation, citation),
            extraction_metadata=ExtractionMetadata(),
        )


def test_extracted_document_rejects_span_outside_text() -> None:
    preprocessed = preprocess_plain_text_from_string("347 U.S. 483")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, len(preprocessed.text) + 1),
        matched_text=preprocessed.text,
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )

    with pytest.raises(ValueError, match="span exceeds"):
        ExtractedDocument(
            source_metadata=preprocessed.source_metadata,
            text=preprocessed.text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(citation,),
            extraction_metadata=ExtractionMetadata(),
        )
