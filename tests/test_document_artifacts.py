"""Contract tests for the immutable document artifact hierarchy."""

import pytest

from mellea_lrc.assessment import initialize_assessment
from mellea_lrc.assessment.types import AssessedDocument
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import extract_citations
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument
from mellea_lrc.preprocessing import DocumentBase, PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.validation.types import (
    CitationValidation,
    ValidatedDocument,
    ValidationStatus,
)


def test_document_stages_are_substitutable() -> None:
    extracted = extract_citations("Brown v. Board, 347 U.S. 483.")
    validated = ValidatedDocument(
        metadata=extracted.metadata,
        text=extracted.text,
        citations=extracted.citations,
        validations=tuple(_validation(item.citation_id) for item in extracted.citations),
    )
    assessed = initialize_assessment(validated)

    assert isinstance(assessed, ValidatedDocument)
    assert isinstance(assessed, ExtractedDocument)
    assert isinstance(assessed, PreprocessedDocument)
    assert isinstance(assessed, DocumentBase)


def test_extraction_assigns_deterministic_document_local_ids() -> None:
    text = "Brown v. Board, 347 U.S. 483. See 28 U.S.C. § 636."

    first = extract_citations(text)
    second = extract_citations(text)

    assert [item.citation_id for item in first.citations] == [
        item.citation_id for item in second.citations
    ]


def test_extracted_document_rejects_duplicate_ids() -> None:
    preprocessed = preprocess_plain_text_from_string("347 U.S. 483")
    citation = _citation("cite-1")

    with pytest.raises(ValueError, match="must be unique"):
        ExtractedDocument(
            metadata=preprocessed.metadata,
            text=preprocessed.text,
            citations=(citation, citation),
        )


def test_extracted_document_rejects_span_outside_text() -> None:
    preprocessed = preprocess_plain_text_from_string("347 U.S. 483")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, len(preprocessed.text) + 1),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )

    with pytest.raises(ValueError, match="span exceeds"):
        ExtractedDocument(
            metadata=preprocessed.metadata,
            text=preprocessed.text,
            citations=(citation,),
        )


def test_validated_document_requires_one_result_per_citation() -> None:
    preprocessed = preprocess_plain_text_from_string("347 U.S. 483")

    with pytest.raises(ValueError, match="exactly match"):
        ValidatedDocument(
            metadata=preprocessed.metadata,
            text=preprocessed.text,
            citations=(_citation("cite-1"),),
            validations=(),
        )


def _citation(citation_id: str) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id=citation_id,
        span=Span(0, 12),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )


def _validation(citation_id: str) -> CitationValidation:
    return CitationValidation(
        citation_id=citation_id,
        locator="347 U.S. 483",
        status=ValidationStatus.FOUND,
        source="test",
        message="found",
    )
