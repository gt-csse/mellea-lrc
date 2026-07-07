"""Tests for the standalone E2E backend pipeline helpers."""

import asyncio
from mellea_lrc.assessment import (
    AssessmentMetadata,
    AssessedCitationAssessment,
    AssessedDocument,
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    CaseNameReassessed,
    CaseNameReassessmentNotRequired,
    CitationAssessment,
    CitationAssessmentResult,
    CourtAssessment,
    CourtAssessmentStatus,
    ReextractedCaseName,
    WaitingCitationAssessment,
    YearAssessment,
    YearAssessmentStatus,
)
from mellea_lrc.jurisdiction_inference.leads import evaluate_court_inference, evaluate_reporter_inference
from mellea_lrc.jurisdiction_inference.types import (
    Jurisdiction,
    ReporterInferenceStatus,
    ReporterInference,
    CourtInference,
    CourtInferenceStatus,
)
from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation, Reporter
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.types import CourtListenerCitationLookup, CourtListenerCitationRecord
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.serialization import (
    serialize_assessed_document,
    serialize_extracted_document,
    serialize_retrieved_document,
    serialize_preprocessed_document,
)
from mellea_lrc.llm import llm_api_config_from_env
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
from scripts.e2e_backend.api import _review_snapshot_payload
from scripts.e2e_backend.pipeline import (
    E2EBackend,
    assess_review_payload,
    review_document_assessment,
    review_preprocessed,
    retrieve_review_citation_payload,
    retrieve_review_payload,
)


def _retrieved_candidate(
    citation_id: str,
    record: CourtListenerCitationRecord | None = None,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=f"{citation_id}:candidate:0",
        record=record or CourtListenerCitationRecord(),
        court_resolution=CourtResolutionTrace(
            courtlistener_court_id=record.court_id if record else None,
            resolved_via=(
                CourtResolutionSource.CLUSTER_PROVIDED
                if record and record.court_id
                else CourtResolutionSource.NOT_ATTEMPTED
            ),
            docket_id=record.docket_id if record else None,
            docket_url=None,
            request_trace=None,
        ),
    )


class FakeClient:
    def lookup_citation(self, volume: str, reporter: str, page: str):
        assert (volume, reporter, page) == ("347", "U.S.", "483")
        return CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            records=(
                CourtListenerCitationRecord(
                    case_name="Brown v. Board of Education",
                    date_filed="1954-05-17",
                    court="scotus",
                ),
            ),
            cache="miss",
            key="lookup-key",
        )


def _court_assessment() -> CourtAssessment:
    return CourtAssessment(
        status=CourtAssessmentStatus.EXACT_MATCH,
        extracted_court="scotus",
        courtlistener_court_id="scotus",
        message="match",
        source="direct",
    )



def _extracted_document(
    *,
    preprocessed: PreprocessedDocument,
    citations: tuple[ExtractedCitation, ...],
) -> ExtractedDocument:
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
    )


def _retrieved_document(
    *,
    preprocessed: PreprocessedDocument,
    citations: tuple[ExtractedCitation, ...],
    retrievals: tuple[CitationRetrieval, ...],
) -> RetrievedDocument:
    return RetrievedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
        retrievals=retrievals,
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        jurisdictions=tuple(
            Jurisdiction(
                reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, 'reporter') else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, 'court') else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
            )
            for item in citations
        ),
    )


def _assessed_document(
    *,
    preprocessed: PreprocessedDocument,
    citations: tuple[ExtractedCitation, ...],
    retrievals: tuple[CitationRetrieval, ...],
    assessments: tuple[CitationAssessment, ...],
) -> AssessedDocument:
    return AssessedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
        jurisdictions=tuple(
            Jurisdiction(
                reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, 'reporter') and isinstance(item.citation.reporter, Reporter) else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, 'court') else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
            )
            for item in citations
        ),
        retrievals=retrievals,
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        assessments=assessments,
        assessment_metadata=AssessmentMetadata(),
    )


def test_review_preprocessed_returns_frontend_span_payload() -> None:
    output = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        client=FakeClient(),
    )

    citation = output["citations"][0]
    assert output["document"]["text"] == "Brown v. Board, 347 U.S. 483."
    assert citation["start"] == 0
    assert citation["end"] == 28
    assert citation["matched_text"] == "347 U.S. 483"
    assert citation["kind"] == "FullCaseCitation"
    assert citation["fields"]["volume"] == "347"
    assert citation["fields"]["reporter"]["edition_short_name"] == "U.S."
    assert citation["fields"]["reporter"]["root_short_name"] == "U.S."
    assert citation["fields"]["reporter"]["cite_type"] == "federal"
    assert citation["fields"]["page"] == "483"
    assert citation["fields"]["plaintiff"] == "Brown"
    assert citation["retrieval"]["status"] == "found"
    assert citation["retrieval"]["case_names"] == ["Brown v. Board of Education"]
    assert citation["retrieval"]["request_trace"] == {
        "http_status": 200,
        "cache": "miss",
        "key": "lookup-key",
        "error_message": None,
    }
    assert citation["retrieval"]["candidate"]["record"]["date_filed"] == "1954-05-17"
    assert output["stats"]["found"] == 1


def test_retrieve_review_payload_reuses_existing_extraction_payload() -> None:
    extracted = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        retrieve=False,
    )

    output = retrieve_review_payload(extracted, client=FakeClient())

    citation = output["citations"][0]
    assert citation["start"] == extracted["citations"][0]["start"]
    assert citation["end"] == extracted["citations"][0]["end"]
    assert citation["retrieval"]["status"] == "found"
    assert output["retrieval"]["counts"]["found"] == 1


def test_retrieve_review_citation_payload_returns_single_retrieval() -> None:
    extracted = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        retrieve=False,
    )

    retrieval = retrieve_review_citation_payload(
        {"citation": extracted["citations"][0]},
        client=FakeClient(),
    )

    assert retrieval["citation_id"] == extracted["citations"][0]["id"]
    assert retrieval["status"] == "found"
    assert retrieval["request_trace"]["key"] == "lookup-key"


def test_assess_review_payload_adds_exact_case_name_assessment_without_llm() -> None:
    extracted = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954)."),
        retrieve=False,
    )
    extracted["citations"][0]["retrieval"] = {
        "citation_id": extracted["citations"][0]["id"],
        "locator": "347 U.S. 483",
        "status": "found",
        "source": "test",
        "case_names": ["Brown v. Board"],
        "request_trace": {
            "http_status": 200,
            "cache": "miss",
            "key": "key",
            "error_message": None,
        },
        "candidate": {
            "candidate_id": f"{extracted['citations'][0]['id']}:candidate:0",
            "record": {
                "case_name": "Brown v. Board",
                "date_filed": "1954-05-17",
                "court": None,
                "court_id": None,
                "docket_id": None,
                "extra_data": {},
            },
            "court_resolution": {
                "courtlistener_court_id": None,
                "resolved_via": "not_attempted",
                "docket_id": None,
                "docket_url": None,
                "request_trace": None,
            },
        },
        "extra_data": {},
    }

    output = asyncio.run(assess_review_payload(extracted))

    assessment = output["citations"][0]["assessment"]
    assert assessment["status"] == "assessed"
    assert assessment["result"]["case_name"]["initial"]["status"] == "exact_match"
    assert assessment["result"]["case_name"]["followup"]["status"] == "not_required"
    assert assessment["result"]["year"]["status"] == "exact_match"
    assert assessment["result"]["year"]["extracted_year"] == "1954"
    assert assessment["result"]["year"]["courtlistener_year"] == "1954"
    assert output["assessment"]["case_name_counts"]["exact_match"] == 1
    # Extracted court "scotus" but CourtListener court unresolved: reporter
    # inference reruns and still lands on a missing terminal verdict.
    assert output["assessment"]["court_counts"]["missing"] == 1
    assert output["assessment"]["year_counts"]["exact_match"] == 1


def test_review_document_assessment_renders_cached_assessment_payload() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"),
            page="483",
            year="1954",
        ),
    )
    retrieval = FoundCitationRetrieval(
        citation_id="cite-1",
        locator="347 U.S. 483",
        source="test",
        request_trace=CourtListenerRequestTrace(http_status=200),
        candidate=_retrieved_candidate(
            "cite-1",
            CourtListenerCitationRecord(
                case_name="Brown v. Board",
                date_filed="1954-05-17",
            ),
        ),
        extra_data=ExtraData(),
    )
    assessment_result = CitationAssessmentResult(
        case_name=CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.EXACT_MATCH,
                extracted_case_name="Brown v. Board",
                courtlistener_case_name="Brown v. Board",
                message="match",
            ),
            followup=CaseNameReassessmentNotRequired(),
        ),
        court=_court_assessment(),
        year=YearAssessment(
            status=YearAssessmentStatus.EXACT_MATCH,
            extracted_year="1954",
            courtlistener_year="1954",
            message="match",
        ),
    )

    output = review_document_assessment(
        _assessed_document(
            preprocessed=preprocessed,
            citations=(citation,),
            retrievals=(retrieval,),
            assessments=(
                AssessedCitationAssessment(
                    citation_id="cite-1",
                    candidate_id=retrieval.candidate.candidate_id,
                    result=assessment_result,
                ),
            ),
        )
    )

    assert output["document"]["text"] == preprocessed.text
    assert output["citations"][0]["retrieval"]["status"] == "found"
    assert output["citations"][0]["assessment"]["result"]["case_name"]["initial"]["status"] == "exact_match"
    assert output["assessment"]["case_name_counts"]["exact_match"] == 1
    assert output["assessment"]["court_counts"]["exact_match"] == 1
    assert output["assessment"]["year_counts"]["exact_match"] == 1
    assert output["stats"]["assessed"] == 1


def test_review_document_assessment_preserves_waiting_citation() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="483"),
    )

    output = review_document_assessment(
        _assessed_document(
            preprocessed=preprocessed,
            citations=(citation,),
            retrievals=(
                FoundCitationRetrieval(
                    citation_id="cite-1",
                    locator="347 U.S. 483",
                    source="test",
                    request_trace=CourtListenerRequestTrace(http_status=200),
                    candidate=_retrieved_candidate("cite-1"),
                    extra_data=ExtraData(),
                ),
            ),
            assessments=(WaitingCitationAssessment(citation_id="cite-1"),),
        )
    )

    assert output["assessment"]["assessment_complete"] is False
    assert output["assessment"]["assessments"][0]["status"] == "waiting"


def test_review_document_assessment_allows_resolved_reextraction_handoff() -> None:
    """A non-semantic primary conclusion may have a reassessment verdict."""
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="483"),
    )
    year = YearAssessment(
        status=YearAssessmentStatus.EXACT_MATCH,
        extracted_year="1954",
        courtlistener_year="1954",
        message="match",
    )
    primary = CitationAssessmentResult(
        case_name=CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
                extracted_case_name="Brown v. Board",
                courtlistener_case_name="Brown v. Board of Education",
                message="re-extraction attempted",
            ),
            followup=CaseNameReassessed(
                reextracted_case_name=ReextractedCaseName(
                    case_name="Brown v. Board",
                    case_name_span=Span(0, 14),
                ),
                result=CaseNameAssessment(
                    status=CaseNameAssessmentStatus.SEMANTIC_MATCH,
                    extracted_case_name="Brown v. Board of Education",
                    courtlistener_case_name="Brown v. Board of Education",
                    message="semantic match after re-extraction",
                ),
            ),
        ),
        court=_court_assessment(),
        year=year,
    )
    output = review_document_assessment(
        _assessed_document(
            preprocessed=preprocessed,
            citations=(citation,),
            retrievals=(
                FoundCitationRetrieval(
                    citation_id="cite-1",
                    locator="347 U.S. 483",
                    source="test",
                    request_trace=CourtListenerRequestTrace(http_status=200),
                    candidate=_retrieved_candidate("cite-1"),
                    extra_data=ExtraData(),
                ),
            ),
            assessments=(
                AssessedCitationAssessment(
                    citation_id="cite-1",
                    candidate_id="cite-1:candidate:0",
                    result=primary,
                ),
            ),
        )
    )
    followup = output["assessment"]["assessments"][0]["result"]["case_name"]["followup"]
    assert followup["result"]["status"] == "semantic_match"


def test_review_snapshot_payload_detects_serialized_interface_boundaries() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"),
            page="483",
            year="1954",
        ),
    )
    extraction = _extracted_document(preprocessed=preprocessed, citations=(citation,))
    retrieval = _retrieved_document(
        preprocessed=preprocessed,
        citations=(citation,),
        retrievals=(
            FoundCitationRetrieval(
                citation_id="cite-1",
                locator="347 U.S. 483",
                source="test",
                request_trace=CourtListenerRequestTrace(http_status=200),
                candidate=_retrieved_candidate("cite-1"),
                extra_data=ExtraData(),
            ),
        ),
    )
    assessment = _assessed_document(
        preprocessed=preprocessed,
        citations=(citation,),
        retrievals=retrieval.retrievals,
        assessments=(
            AssessedCitationAssessment(
                citation_id="cite-1",
                candidate_id="cite-1:candidate:0",
                result=CitationAssessmentResult(
                    case_name=CaseNameAssessmentRun(
                        initial=CaseNameAssessment(
                            status=CaseNameAssessmentStatus.EXACT_MATCH,
                            extracted_case_name="Brown v. Board",
                            courtlistener_case_name="Brown v. Board",
                            message="match",
                        ),
                        followup=CaseNameReassessmentNotRequired(),
                    ),
                    court=_court_assessment(),
                    year=YearAssessment(
                        status=YearAssessmentStatus.EXACT_MATCH,
                        extracted_year="1954",
                        courtlistener_year="1954",
                        message="match",
                    ),
                ),
            ),
        ),
    )
    backend = E2EBackend()

    assert (
        _review_snapshot_payload(serialize_preprocessed_document(preprocessed), backend)["stage"]
        == "preprocessed"
    )
    assert _review_snapshot_payload(serialize_extracted_document(extraction), backend)["stage"] == "extracted"
    assert _review_snapshot_payload(serialize_retrieved_document(retrieval), backend)["stage"] == "retrieved"
    assert _review_snapshot_payload(serialize_assessed_document(assessment), backend)["stage"] == "assessed"


def test_llm_api_config_binds_an_explicit_openai_compatible_endpoint() -> None:
    config = llm_api_config_from_env(
        {
            "MELLEA_LRC_LLM_MODEL": "model-id",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
            "MELLEA_LRC_LLM_API_BASE": "https://llm.example/v1",
            "MELLEA_LRC_LLM_API_KEY": "api-key",
        }
    )

    assert config.model == "model-id"
    assert config.api_base == "https://llm.example/v1"
    assert config.api_key == "api-key"
