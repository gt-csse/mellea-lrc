"""Contract tests for the immutable document artifact hierarchy."""

from collections.abc import Mapping

import pytest

from mellea_lrc.assessment import initialize_assessment
from mellea_lrc.assessment.types import CaseNameAssessment, CaseNameAssessmentStatus
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.documents import SourceMetadata
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
    assert assessed.source_metadata is extracted.source_metadata
    assert assessed.preprocessing_metadata is extracted.preprocessing_metadata
    assert assessed.extraction_metadata is extracted.extraction_metadata


def test_source_metadata_copies_and_freezes_extras() -> None:
    extras = {"docket": "original"}
    metadata = SourceMetadata(extras=extras)

    extras["docket"] = "changed"

    assert metadata.extras == {"docket": "original"}
    with pytest.raises(TypeError):
        metadata.extras["docket"] = "changed"  # type: ignore[index]


def test_validation_copies_and_deeply_freezes_service_payloads() -> None:
    cluster = {
        "case_name": "Brown v. Board",
        "judges": ["Warren"],
        "court": {"slug": "scotus"},
    }
    validation = CitationValidation(
        citation_id="cite-1",
        locator="347 U.S. 483",
        status=ValidationStatus.FOUND,
        source="test",
        message="found",
        clusters=(cluster,),
    )

    cluster["case_name"] = "Changed"
    cluster["judges"].append("Changed")

    assert validation.clusters[0]["case_name"] == "Brown v. Board"
    assert validation.clusters[0]["judges"] == ("Warren",)
    with pytest.raises(TypeError):
        validation.clusters[0]["case_name"] = "Changed"  # type: ignore[index]
    court = validation.clusters[0]["court"]
    assert isinstance(court, Mapping)
    with pytest.raises(TypeError):
        court["slug"] = "changed"  # type: ignore[index]


def test_assessment_copies_and_freezes_chat_history() -> None:
    history = [{"role": "assistant", "content": "original"}]
    assessment = CaseNameAssessment(
        citation_id="cite-1",
        status=CaseNameAssessmentStatus.EXACT_MATCH,
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board",
        message="match",
        chat_history=history,
    )

    history[0]["content"] = "changed"

    assert assessment.chat_history == (
        {"role": "assistant", "content": "original"},
    )
    assert assessment.chat_history is not None
    with pytest.raises(TypeError):
        assessment.chat_history[0]["content"] = "changed"  # type: ignore[index]


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
