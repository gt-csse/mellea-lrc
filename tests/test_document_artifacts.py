"""Contract tests for the immutable document artifact hierarchy."""

from dataclasses import FrozenInstanceError

import pytest

from mellea_lrc.assessment import initialize_assessment
from mellea_lrc.assessment.types import CaseNameAssessment, CaseNameAssessmentStatus, ChatTurn
from mellea_lrc.core.citations import FullCaseCitation, Reporter
from mellea_lrc.core.documents import SourceMetadata
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.types import CourtListenerCitationRecord
from mellea_lrc.extraction import extract_citations
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import DocumentBase, PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.jurisdiction_inference import Jurisdiction
from mellea_lrc.jurisdiction_inference.leads import evaluate_court_inference, evaluate_reporter_inference
from mellea_lrc.jurisdiction_inference.types import (
    CourtInference,
    CourtInferenceStatus,
    ReporterInference,
    ReporterInferenceStatus,
)
from mellea_lrc.retrieval.types import (
    CitationRetrieval,
    CourtListenerRequestTrace,
    CourtResolutionSource,
    CourtResolutionTrace,
    FoundCitationRetrieval,
    RetrievedCandidate,
    RetrievedDocument,
    RetrievalMetadata,
    RetrievalStatus,
)


def test_document_stages_are_substitutable() -> None:
    extracted = extract_citations("Brown v. Board, 347 U.S. 483.")
    retrieved = RetrievedDocument(
        source_metadata=extracted.source_metadata,
        text=extracted.text,
        preprocessing_metadata=extracted.preprocessing_metadata,
        citations=extracted.citations,
        extraction_metadata=extracted.extraction_metadata,
        retrievals=tuple(_retrieval(item.citation_id) for item in extracted.citations),
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        jurisdictions=tuple(
            Jurisdiction(
                reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, 'reporter') else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, 'court') else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
            )
            for item in extracted.citations
        ),
    )
    assessed = initialize_assessment(retrieved)

    assert isinstance(assessed, RetrievedDocument)
    assert isinstance(assessed, ExtractedDocument)
    assert isinstance(assessed, PreprocessedDocument)
    assert isinstance(assessed, DocumentBase)
    assert assessed.source_metadata is extracted.source_metadata
    assert assessed.preprocessing_metadata is extracted.preprocessing_metadata
    assert assessed.extraction_metadata is extracted.extraction_metadata


def test_source_metadata_copies_and_freezes_extra_data() -> None:
    extra_data = {"docket": "original"}
    metadata = SourceMetadata(extra_data=ExtraData(extra_data))

    extra_data["docket"] = "changed"

    assert metadata.extra_data.to_dict() == {"docket": "original"}
    with pytest.raises(TypeError):
        metadata.extra_data.values["docket"] = "changed"  # type: ignore[index]


def test_retrieval_copies_and_deeply_freezes_service_payloads() -> None:
    extra_data = {
        "judges": ["Warren"],
        "court": {"slug": "scotus"},
    }
    record = CourtListenerCitationRecord(
        case_name="Brown v. Board",
        extra_data=ExtraData(extra_data),
    )
    retrieval = FoundCitationRetrieval(
        citation_id="cite-1",
        locator="347 U.S. 483",
        source="test",
        request_trace=CourtListenerRequestTrace(http_status=200),
        candidate=RetrievedCandidate(
            candidate_id="cite-1:candidate:0",
            record=record,
            court_resolution=CourtResolutionTrace(
                courtlistener_court_id=None,
                resolved_via=CourtResolutionSource.NOT_ATTEMPTED,
                docket_id=None,
                docket_url=None,
                request_trace=None,
            ),
        ),
        extra_data=ExtraData(),
    )

    extra_data["judges"].append("Changed")

    assert retrieval.candidate.record.case_name == "Brown v. Board"
    assert retrieval.candidate.record.extra_data.values["judges"] == ("Warren",)
    with pytest.raises(TypeError):
        retrieval.candidate.record.extra_data.values["judges"] = ()  # type: ignore[index]


def test_assessment_copies_and_freezes_chat_history() -> None:
    turn = ChatTurn(role="assistant", content="original")
    assessment = CaseNameAssessment(
        status=CaseNameAssessmentStatus.EXACT_MATCH,
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board",
        message="match",
        chat_history=(turn,),
    )

    assert assessment.chat_history == (turn,)
    assert assessment.chat_history is not None
    with pytest.raises(FrozenInstanceError):
        assessment.chat_history[0].content = "changed"  # type: ignore[misc]


def test_extraction_assigns_deterministic_document_local_ids() -> None:
    text = "Brown v. Board, 347 U.S. 483. See 28 U.S.C. § 636."

    first = extract_citations(text)
    second = extract_citations(text)

    assert [item.citation_id for item in first.citations] == [item.citation_id for item in second.citations]


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
        citation=FullCaseCitation(volume="347", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="483"),
    )

    with pytest.raises(ValueError, match="span exceeds"):
        ExtractedDocument(
            source_metadata=preprocessed.source_metadata,
            text=preprocessed.text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(citation,),
            extraction_metadata=ExtractionMetadata(),
        )


def test_retrieved_document_requires_one_result_per_citation() -> None:
    preprocessed = preprocess_plain_text_from_string("347 U.S. 483")

    with pytest.raises(ValueError, match="exactly match"):
        RetrievedDocument(
            source_metadata=preprocessed.source_metadata,
            text=preprocessed.text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(_citation("cite-1"),),
            extraction_metadata=ExtractionMetadata(),
            retrievals=(),
            retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
            jurisdictions=tuple(
                Jurisdiction(
                    reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, 'reporter') else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                    court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, 'court') else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
                )
                for item in (_citation("cite-1"),)
            ),
        )


def _citation(citation_id: str) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id=citation_id,
        span=Span(0, 12),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="483"),
    )


def _retrieval(citation_id: str) -> CitationRetrieval:
    return FoundCitationRetrieval(
        citation_id=citation_id,
        locator="347 U.S. 483",
        source="test",
        request_trace=CourtListenerRequestTrace(http_status=200),
        candidate=RetrievedCandidate(
            candidate_id=f"{citation_id}:candidate:0",
            record=CourtListenerCitationRecord(),
            court_resolution=CourtResolutionTrace(
                courtlistener_court_id=None,
                resolved_via=CourtResolutionSource.NOT_ATTEMPTED,
                docket_id=None,
                docket_url=None,
                request_trace=None,
            ),
        ),
        extra_data=ExtraData(),
    )
