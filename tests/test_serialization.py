"""Tests for neutral and Label Studio-specific serialization."""

import json

import pytest
from pydantic import ValidationError

from mellea_lrc.assessment import (
    AmbiguousCitationAssessment,
    AssessmentMetadata,
    AssessmentSkipReason,
    AssessedCitationAssessment,
    AssessedDocument,
    CandidateAssessment,
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    CaseNameReassessed,
    CaseNameReassessmentFailed,
    CaseNameReassessmentNotRequired,
    CaseNameReextractionFailed,
    CitationAssessmentResult,
    CourtAssessment,
    CourtAssessmentStatus,
    FailedCitationAssessment,
    ReextractedCaseName,
    SkippedCitationAssessment,
    WaitingCitationAssessment,
    YearAssessment,
    YearAssessmentStatus,
)
from mellea_lrc.core.spans import Span
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.courtlistener.types import CourtListenerCitationRecord
from mellea_lrc.extraction import extract_citations
from mellea_lrc.jurisdiction_inference.leads import evaluate_court_inference, evaluate_reporter_inference
from mellea_lrc.jurisdiction_inference.types import (
    Jurisdiction,
    ReporterInferenceStatus,
    ReporterInference,
    CourtInference,
    CourtInferenceStatus,
)
from mellea_lrc.core.citations import Reporter
from mellea_lrc.serialization import (
    deserialize_assessed_document,
    deserialize_citation_retrieval,
    deserialize_extracted_document,
    deserialize_citation_assessment,
    deserialize_preprocessed_document,
    deserialize_retrieved_document,
    serialize_assessed_document,
    serialize_citation_retrieval,
    serialize_extracted_document,
    serialize_citation_assessment,
    serialize_preprocessed_document,
    serialize_retrieved_document,
)
from mellea_lrc.retrieval.types import (
    AmbiguousCitationRetrieval,
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
from scripts.label_studio.label_studio import to_label_studio_prediction
from scripts.label_studio.pre_annotate import build_task_payload

SAMPLE_TEXT = "Under Norton v. Shelby County, 118 U.S. 425, 442 (1886), an act confers no rights."
RECAP_TEXT = (
    "Case: Example\n"
    "Recovered file description: sample\n\n"
    "--- Plain text ---\n"
    "The Court cites Oconner v. Agilant Sols., Inc., 444 F. Supp. 3d 593.\n"
)


def _retrieved_candidate(
    citation_id: str,
    record: CourtListenerCitationRecord,
    index: int = 0,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=f"{citation_id}:candidate:{index}",
        record=record,
        court_resolution=CourtResolutionTrace(
            courtlistener_court_id=record.court_id,
            resolved_via=(
                CourtResolutionSource.CLUSTER_PROVIDED
                if record.court_id
                else CourtResolutionSource.NOT_ATTEMPTED
            ),
            docket_id=record.docket_id,
            docket_url=None,
            request_trace=None,
        ),
    )


def test_document_extraction_serializes_without_ui_assumptions() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    artifact = serialize_extracted_document(extraction)

    assert artifact["schema_version"] == 18
    assert artifact["artifact_type"] == "extracted_document"
    assert artifact["source_metadata"]["path"] is None
    assert artifact["text"] == SAMPLE_TEXT
    assert artifact["citations"]
    assert "counts" not in artifact

    citation = artifact["citations"][0]
    assert citation["citation_id"] == extraction.citations[0].citation_id
    assert citation["span"]["start"] == extraction.citations[0].span.start
    assert citation["citation"]["type"] == "FullCaseCitation"
    assert "from_name" not in citation


def test_preprocessed_document_round_trips() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    artifact = serialize_preprocessed_document(extraction)

    restored = deserialize_preprocessed_document(artifact)

    assert restored.text == extraction.text
    assert restored.source_metadata == extraction.source_metadata
    assert restored.preprocessing_metadata == extraction.preprocessing_metadata


def test_unversioned_preprocessed_document_is_rejected() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    artifact = serialize_preprocessed_document(extraction)
    artifact.pop("schema_version")

    with pytest.raises(ValueError, match="schema_version"):
        deserialize_preprocessed_document(artifact)


def test_previous_schema_version_is_rejected() -> None:
    artifact = serialize_preprocessed_document(extract_citations(SAMPLE_TEXT))
    artifact["schema_version"] = 17

    with pytest.raises(ValueError, match="schema_version"):
        deserialize_preprocessed_document(artifact)


def test_missing_stage_metadata_is_rejected() -> None:
    artifact = serialize_extracted_document(extract_citations(SAMPLE_TEXT))
    artifact.pop("extraction_metadata")

    with pytest.raises(ValidationError, match="extraction_metadata"):
        deserialize_extracted_document(artifact)


def test_artifact_transport_rejects_type_coercion() -> None:
    artifact = serialize_preprocessed_document(extract_citations(SAMPLE_TEXT))
    artifact["text"] = 123

    with pytest.raises(ValidationError, match="text"):
        deserialize_preprocessed_document(artifact)


def test_artifact_transport_rejects_unknown_fields() -> None:
    artifact = serialize_preprocessed_document(extract_citations(SAMPLE_TEXT))
    artifact["legacy_metadata"] = {}

    with pytest.raises(ValidationError, match="legacy_metadata"):
        deserialize_preprocessed_document(artifact)


def test_citation_transport_rejects_fields_from_another_citation_kind() -> None:
    artifact = serialize_extracted_document(extract_citations(SAMPLE_TEXT))
    citation = artifact["citations"][0]["citation"]
    citation["publisher"] = "Not valid for a case citation"

    with pytest.raises(ValidationError, match="publisher"):
        deserialize_extracted_document(artifact)


def test_assessment_transport_rejects_state_inappropriate_fields() -> None:
    payload = serialize_citation_assessment(WaitingCitationAssessment(citation_id="cite-1"))
    payload["error"] = "waiting records cannot contain an error"

    with pytest.raises(ValidationError, match="error"):
        deserialize_citation_assessment(payload)


def test_document_extraction_round_trips() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    artifact = serialize_extracted_document(extraction)

    restored = deserialize_extracted_document(artifact)

    assert restored == extraction


def test_document_retrieval_round_trips() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    retrieval = RetrievedDocument(
        source_metadata=extraction.source_metadata,
        text=extraction.text,
        preprocessing_metadata=extraction.preprocessing_metadata,
        citations=extraction.citations,
        extraction_metadata=extraction.extraction_metadata,
        retrievals=(
            FoundCitationRetrieval(
                citation_id=extraction.citations[0].citation_id,
                locator="118 U.S. 425",
                source="test",
                request_trace=CourtListenerRequestTrace(
                    http_status=200, cache="miss", key="118 U.S. 425"
                ),
                candidate=_retrieved_candidate(
                    extraction.citations[0].citation_id,
                    CourtListenerCitationRecord(
                        case_name="Norton v. Shelby County",
                        date_filed="1886-01-01",
                        extra_data=ExtraData({"absolute_url": "/opinion/1/"}),
                    ),
                ),
                extra_data=ExtraData(),
            ),
        ),
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        jurisdictions=tuple(
            Jurisdiction(
                reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, 'reporter') else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, 'court') else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
            )
            for item in extraction.citations
        ),
    )

    artifact = serialize_retrieved_document(retrieval)
    restored = deserialize_retrieved_document(artifact)

    assert artifact["artifact_type"] == "retrieved_document"
    json.dumps(artifact)
    assert restored == retrieval


def test_current_schema_validation_names_deserialize_into_retrieval_domain() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    citation_id = extraction.citations[0].citation_id
    retrieval = RetrievedDocument(
        source_metadata=extraction.source_metadata,
        text=extraction.text,
        preprocessing_metadata=extraction.preprocessing_metadata,
        citations=extraction.citations,
        extraction_metadata=extraction.extraction_metadata,
        retrievals=(
            FoundCitationRetrieval(
                citation_id=citation_id,
                locator="118 U.S. 425",
                source="test",
                request_trace=CourtListenerRequestTrace(
                    http_status=200, cache="miss", key="118 U.S. 425"
                ),
                candidate=_retrieved_candidate(
                    citation_id,
                    CourtListenerCitationRecord(
                        case_name="Norton v. Shelby County",
                        date_filed="1886-01-01",
                        extra_data=ExtraData({"absolute_url": "/opinion/1/"}),
                    ),
                ),
                extra_data=ExtraData(),
            ),
        ),
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        jurisdictions=tuple(
            Jurisdiction(
                reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, 'reporter') else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, 'court') else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
            )
            for item in extraction.citations
        ),
    )
    artifact = serialize_retrieved_document(retrieval)

    restored = deserialize_retrieved_document(artifact)

    assert restored == retrieval
def test_ambiguous_retrieval_candidates_round_trip_with_court_traces() -> None:
    record = AmbiguousCitationRetrieval(
        citation_id="cite-1",
        locator="1 F.3d 2",
        source="test",
        request_trace=CourtListenerRequestTrace(http_status=300, cache="miss", key="key"),
        candidates=(
            RetrievedCandidate(
                candidate_id="cite-1:candidate:0",
                record=CourtListenerCitationRecord(case_name="Example A", docket_id="11"),
                court_resolution=CourtResolutionTrace(
                    courtlistener_court_id="ca1",
                    resolved_via=CourtResolutionSource.DOCKET_LOOKUP,
                    docket_id="11",
                    docket_url="/dockets/11",
                    request_trace=CourtListenerRequestTrace(http_status=200, cache="miss"),
                ),
            ),
            RetrievedCandidate(
                candidate_id="cite-1:candidate:1",
                record=CourtListenerCitationRecord(case_name="Example B", court_id="ca2"),
                court_resolution=CourtResolutionTrace(
                    courtlistener_court_id="ca2",
                    resolved_via=CourtResolutionSource.CLUSTER_PROVIDED,
                    docket_id=None,
                    docket_url=None,
                    request_trace=None,
                ),
            ),
        ),
    )

    payload = serialize_citation_retrieval(record)

    assert "record" not in payload
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["record"]["case_name"] == "Example A"
    assert payload["candidates"][0]["court_resolution"]["courtlistener_court_id"] == "ca1"
    assert deserialize_citation_retrieval(payload) == record


def test_document_assessment_round_trips() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    retrieval = FoundCitationRetrieval(
        citation_id=extraction.citations[0].citation_id,
        locator="118 U.S. 425",
        source="test",
        request_trace=CourtListenerRequestTrace(http_status=200),
        candidate=_retrieved_candidate(
            extraction.citations[0].citation_id,
            CourtListenerCitationRecord(
                case_name="Norton v. Shelby County",
                date_filed="1886-01-01",
            ),
        ),
        extra_data=ExtraData(),
    )
    citation_id = extraction.citations[0].citation_id
    assessment_result = CitationAssessmentResult(
        case_name=CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
                extracted_case_name="Norton v. Shelby County",
                courtlistener_case_name="Norton v. Shelby County",
                message="re-extraction attempted",
            ),
            followup=CaseNameReassessed(
                reextracted_case_name=ReextractedCaseName(
                    case_name="Norton v. Shelby County",
                    case_name_span=Span(start=6, end=29),
                ),
                result=CaseNameAssessment(
                    status=CaseNameAssessmentStatus.EXACT_MATCH,
                    extracted_case_name="Norton v. Shelby County",
                    courtlistener_case_name="Norton v. Shelby County",
                    message="match after re-extraction",
                ),
            ),
        ),
        court=CourtAssessment(
            status=CourtAssessmentStatus.EXACT_MATCH,
            extracted_court="scotus",
            courtlistener_court_id="scotus",
            message="match",
            source="direct",
        ),
        year=YearAssessment(
            status=YearAssessmentStatus.EXACT_MATCH,
            extracted_year="1886",
            courtlistener_year="1886",
            message="match",
        ),
    )
    document_assessment = AssessedDocument(
        source_metadata=extraction.source_metadata,
        text=extraction.text,
        preprocessing_metadata=extraction.preprocessing_metadata,
        citations=extraction.citations,
        extraction_metadata=extraction.extraction_metadata,
        jurisdictions=tuple(
            Jurisdiction(
                reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, 'reporter') and isinstance(item.citation.reporter, Reporter) else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, 'court') else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
            )
            for item in extraction.citations
        ),
        retrievals=(retrieval,),
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        assessments=(
            AssessedCitationAssessment(
                citation_id=citation_id,
                candidate_id=retrieval.candidate.candidate_id,
                result=assessment_result,
            ),
        ),
        assessment_metadata=AssessmentMetadata(),
    )

    artifact = serialize_assessed_document(document_assessment)
    restored = deserialize_assessed_document(artifact)

    assert artifact["artifact_type"] == "assessed_document"
    assert "assessment_complete" not in artifact
    assert "assessment_status_counts" not in artifact
    assert "reassessments" not in artifact
    reextracted = artifact["assessments"][0]["result"]["case_name"]["followup"]["reextracted_case_name"]
    assert reextracted["case_name"] == "Norton v. Shelby County"
    assert reextracted["case_name_span"] == {"start": 6, "end": 29}
    assert "matched_text" not in reextracted
    assert "case_name_counts" not in artifact
    assert "year_counts" not in artifact
    assert "mellea_calls" not in artifact["assessment_metadata"]
    assert restored == document_assessment

    artifact["assessment_metadata"]["mellea_calls"] = 1
    with pytest.raises(ValidationError, match="mellea_calls"):
        deserialize_assessed_document(artifact)


@pytest.mark.parametrize(
    "record",
    [
        WaitingCitationAssessment(citation_id="cite-1"),
        SkippedCitationAssessment(
            citation_id="cite-1",
            reason=AssessmentSkipReason.RETRIEVAL_NOT_ELIGIBLE,
            message="not found",
        ),
        FailedCitationAssessment(citation_id="cite-1", error="RuntimeError: unavailable"),
    ],
)
def test_non_assessed_execution_states_round_trip(record) -> None:
    payload = serialize_citation_assessment(record)

    assert deserialize_citation_assessment(payload) == record


def _minimal_result(case_name: str) -> CitationAssessmentResult:
    return CitationAssessmentResult(
        case_name=CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.EXACT_MATCH,
                extracted_case_name=case_name,
                courtlistener_case_name=case_name,
                message="match",
            ),
            followup=CaseNameReassessmentNotRequired(),
        ),
        court=CourtAssessment(
            status=CourtAssessmentStatus.MISSING,
            extracted_court=None,
            courtlistener_court_id=None,
            message="missing",
            source="direct",
        ),
        year=YearAssessment(
            status=YearAssessmentStatus.MISSING,
            extracted_year=None,
            courtlistener_year=None,
            message="missing",
        ),
    )


def test_ambiguous_citation_assessment_round_trips() -> None:
    record = AmbiguousCitationAssessment(
        citation_id="cite-1",
        candidates=(
            CandidateAssessment(
                candidate_id="cite-1:candidate:0",
                result=_minimal_result("Doe v. Roe"),
            ),
            CandidateAssessment(
                candidate_id="cite-1:candidate:1",
                result=_minimal_result("Doe v. Roe"),
            ),
        ),
    )

    payload = serialize_citation_assessment(record)

    assert payload["status"] == "ambiguous"
    assert deserialize_citation_assessment(payload) == record


def test_gated_ambiguous_citation_assessment_round_trips() -> None:
    record = AmbiguousCitationAssessment(
        citation_id="cite-1",
        candidates=(),
        gated=True,
        message="6 candidates exceed the 5-candidate enumeration limit.",
    )

    payload = serialize_citation_assessment(record)

    assert deserialize_citation_assessment(payload) == record


def test_case_name_followup_round_trips_inside_citation_assessment() -> None:
    record = AssessedCitationAssessment(
        citation_id="cite-1",
        candidate_id="cite-1:candidate:0",
        result=CitationAssessmentResult(
            case_name=CaseNameAssessmentRun(
                initial=CaseNameAssessment(
                    status=CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
                    extracted_case_name="Norton",
                    courtlistener_case_name="Norton v. Shelby County",
                    message="re-extraction attempted",
                ),
                followup=CaseNameReassessmentFailed(
                    reextracted_case_name=ReextractedCaseName(
                        case_name="Norton v. Shelby County",
                        case_name_span=Span(6, 29),
                    ),
                    error="RuntimeError: unavailable",
                ),
            ),
            court=CourtAssessment(
                status=CourtAssessmentStatus.MISSING,
                extracted_court=None,
                courtlistener_court_id="scotus",
                message="missing",
                source="direct",
            ),
            year=YearAssessment(
                status=YearAssessmentStatus.EXACT_MATCH,
                extracted_year="1886",
                courtlistener_year="1886",
                message="match",
            ),
        ),
    )

    payload = serialize_citation_assessment(record)

    assert deserialize_citation_assessment(payload) == record


def test_label_studio_prediction_shape() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    prediction = to_label_studio_prediction(extraction)

    assert prediction["model_version"] == "eyecite-pre-annotation"
    assert prediction["score"] == 1.0
    assert prediction["result"]

    label_results = [item for item in prediction["result"] if item["type"] == "labels"]
    assert label_results
    assert label_results[0]["value"]["labels"] == ["FullCaseCitation"]

    textarea_results = [item for item in prediction["result"] if item["type"] == "textarea"]
    field_names = {item["from_name"] for item in textarea_results}
    assert "plaintiff" in field_names
    assert "volume" in field_names


def test_label_studio_task_text_matches_prediction_spans() -> None:
    task = build_task_payload(RECAP_TEXT, source_path="sample.txt")
    text = task["data"]["text"]
    prediction = task["predictions"][0]

    assert text.startswith("The Court cites")
    assert task["data"]["source_path"] == "sample.txt"
    assert "source_header" in task["data"]

    label_results = [item for item in prediction["result"] if item["type"] == "labels"]
    assert label_results
    for result in label_results:
        value = result["value"]
        assert text[value["start"] : value["end"]] == value["text"]


def test_label_studio_uses_full_span_text_not_matched_text() -> None:
    extraction = extract_citations(RECAP_TEXT)
    prediction = to_label_studio_prediction(extraction)

    label_result = next(item for item in prediction["result"] if item["type"] == "labels")
    span_text = extraction.text[extraction.citations[0].span.start : extraction.citations[0].span.end]

    assert extraction.citations[0].matched_text == "444 F. Supp. 3d 593"
    assert label_result["value"]["text"] == span_text
    assert label_result["value"]["text"] != extraction.citations[0].matched_text
