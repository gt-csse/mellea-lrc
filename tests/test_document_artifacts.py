"""Contract tests for the immutable document artifact hierarchy."""

import pytest

from mellea_lrc.assessment import initialize_assessment
from mellea_lrc.assessment.types import AssessedDocument
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import extract_citations
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import DocumentBase, PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.validation.types import (
    CitationValidation,
    ValidatedDocument,
    ValidationMetadata,
    ValidationStatus,
)


def test_document_stages_are_substitutable() -> None:
    extracted = extract_citations("Brown v. Board, 347 U.S. 483.")
    validated = ValidatedDocument(
        source_metadata=extracted.source_metadata,
        text=extracted.text,
        preprocessing_metadata=extracted.preprocessing_metadata,
        citations=extracted.citations,
        extraction_metadata=extracted.extraction_metadata,
        validations=tuple(_validation(item.citation_id) for item in extracted.citations),
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
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
        matched_text="347 U.S. 483",
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


def test_validated_document_requires_one_result_per_citation() -> None:
    preprocessed = preprocess_plain_text_from_string("347 U.S. 483")

    with pytest.raises(ValueError, match="exactly match"):
        ValidatedDocument(
            source_metadata=preprocessed.source_metadata,
            text=preprocessed.text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(_citation("cite-1"),),
            extraction_metadata=ExtractionMetadata(),
            validations=(),
            validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
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
