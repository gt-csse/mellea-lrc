"""JSON-ready serialization for reusable mellea-lrc artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, TypeAlias, cast

from pydantic import TypeAdapter

from mellea_lrc.assessment.types import (
    AmbiguousCitationAssessment,
    AssessmentMetadata,
    AssessmentSkipReason,
    AssessmentStatus,
    AssessedCitationAssessment,
    AssessedDocument,
    CandidateAssessment,
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    CaseNameFollowup,
    CaseNameFollowupStatus,
    CaseNameReassessed,
    CaseNameReassessmentFailed,
    CaseNameReassessmentNotRequired,
    CaseNameReextractionFailed,
    ChatTurn,
    CitationAssessment,
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
from mellea_lrc.core.citations import (
    CitationKind,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
    citation_kind,
)
from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import (
    ExtractedCitation,
    ExtractedDocument,
    ExtractionBackend,
    ExtractionMetadata,
)
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
)
from mellea_lrc.retrieval.types import (
    AmbiguousCitationRetrieval,
    CaseNameSearchCorpus,
    CaseNameSearchProbe,
    CaseNameSearchStatus,
    CaseNameSearchTrace,
    CitationRetrieval,
    CourtResolutionSource,
    CourtResolutionTrace,
    FoundCitationRetrieval,
    InvalidCitationRetrieval,
    LookupFailedCitationRetrieval,
    NotFoundCitationRetrieval,
    RetrievedCandidate,
    SkippedCitationRetrieval,
    ThrottledCitationRetrieval,
    RetrievedDocument,
    RetrievalClientMode,
    RetrievalMetadata,
    RetrievalStatus,
)
from mellea_lrc.reporter_jurisdiction.types import (
    ReporterJurisdictionEvidence,
    ReporterJurisdictionInference,
    ReporterJurisdictionStatus,
)
from mellea_lrc.courtlistener.types import CourtListenerCitationRecord, RetrievalFailureDetail
from mellea_lrc.serialization.transport import (
    AssessedDocumentPayload,
    CanonicalCitationPayload,
    CaseNameAssessmentPayload,
    CaseNameFollowupPayload,
    CaseNameSearchTracePayload,
    CitationAssessmentPayload,
    CitationAssessmentResultPayload,
    CLCourtTaxonomyPayload,
    CourtAssessmentPayload,
    CitationRetrievalPayload,
    CourtResolutionTracePayload,
    ReporterJurisdictionInferencePayload,
    ExtractedCitationPayload,
    ExtractedDocumentPayload,
    ReextractedCaseNamePayload,
    PreprocessedDocumentPayload,
    RetrievedDocumentPayload,
    YearAssessmentPayload,
)

if TYPE_CHECKING:
    from mellea_lrc.core.citations import CanonicalCitation

JsonValue: TypeAlias = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
SCHEMA_VERSION = 15

_CITATION_PAYLOAD_ADAPTER = TypeAdapter(CanonicalCitationPayload)
_ASSESSMENT_PAYLOAD_ADAPTER = TypeAdapter(CitationAssessmentPayload)
_CASE_NAME_FOLLOWUP_PAYLOAD_ADAPTER = TypeAdapter(CaseNameFollowupPayload)
_RETRIEVAL_PAYLOAD_ADAPTER = TypeAdapter(CitationRetrievalPayload)

_CITATION_CLASSES = {
    CitationKind.FULL_CASE: FullCaseCitation,
    CitationKind.FULL_LAW: FullLawCitation,
    CitationKind.FULL_JOURNAL: FullJournalCitation,
    CitationKind.SHORT_CASE: ShortCaseCitation,
    CitationKind.SUPRA: SupraCitation,
    CitationKind.ID: IdCitation,
    CitationKind.REFERENCE: ReferenceCitation,
    CitationKind.UNKNOWN: UnknownCitation,
}


def serialize_preprocessed_document(item: PreprocessedDocument) -> dict[str, JsonValue]:
    """Serialize the preprocessing boundary object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "preprocessed_document",
        "text": item.text,
        "source_metadata": _serialize_source_metadata(item.source_metadata),
        "preprocessing_metadata": _serialize_preprocessing_metadata(item.preprocessing_metadata),
    }


def deserialize_preprocessed_document(payload: Mapping[str, object]) -> PreprocessedDocument:
    """Rebuild the preprocessing boundary object from JSON data."""
    validated = PreprocessedDocumentPayload.model_validate(payload)
    return _deserialize_preprocessed_fields(validated.model_dump(mode="python"))


def _deserialize_preprocessed_fields(payload: Mapping[str, object]) -> PreprocessedDocument:
    return PreprocessedDocument(
        source_metadata=_deserialize_source_metadata(_required_mapping_field(payload, "source_metadata")),
        text=_required_str(payload.get("text"), "document text"),
        preprocessing_metadata=_deserialize_preprocessing_metadata(
            _required_mapping_field(payload, "preprocessing_metadata")
        ),
    )


def _serialize_source_metadata(item: SourceMetadata) -> dict[str, JsonValue]:
    return {
        "path": item.path,
        "format": item.format.value,
        "header": item.header,
        "extra_data": _serialize_extra_data(item.extra_data),
    }


def _deserialize_source_metadata(payload: Mapping[str, object]) -> SourceMetadata:
    return SourceMetadata(
        path=_optional_str(payload.get("path")),
        format=SourceFormat(_required_str(payload.get("format"), "source format")),
        header=_optional_str(payload.get("header")),
        extra_data=_deserialize_extra_data(payload.get("extra_data")),
    )


def _serialize_extra_data(item: ExtraData) -> dict[str, JsonValue]:
    return cast("dict[str, JsonValue]", item.to_dict())


def _deserialize_extra_data(value: object) -> ExtraData:
    if not isinstance(value, dict):
        msg = "extra_data must be an object"
        raise TypeError(msg)
    return ExtraData(value)


def _serialize_preprocessing_metadata(
    item: PreprocessingMetadata,
) -> dict[str, JsonValue]:
    return {
        "backend": item.backend.value,
        "backend_version": item.backend_version,
    }


def _deserialize_preprocessing_metadata(
    payload: Mapping[str, object],
) -> PreprocessingMetadata:
    return PreprocessingMetadata(
        backend=PreprocessingBackend(_required_str(payload.get("backend"), "preprocessing backend")),
        backend_version=_optional_str(payload.get("backend_version")),
    )


def _serialize_extraction_metadata(item: ExtractionMetadata) -> dict[str, JsonValue]:
    return {"backend": item.backend.value, "backend_version": item.backend_version}


def _deserialize_extraction_metadata(payload: Mapping[str, object]) -> ExtractionMetadata:
    return ExtractionMetadata(
        backend=ExtractionBackend(_required_str(payload.get("backend"), "extraction backend")),
        backend_version=_optional_str(payload.get("backend_version")),
    )


def _serialize_retrieval_metadata(item: RetrievalMetadata) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"client_mode": item.client_mode, "source": item.source}
    if item.duration_ms is not None:
        payload["duration_ms"] = item.duration_ms
    return payload


def _deserialize_retrieval_metadata(payload: Mapping[str, object]) -> RetrievalMetadata:
    client_mode = _required_str(payload.get("client_mode"), "retrieval client mode")
    if client_mode not in {"deployed", "sdk", "custom"}:
        msg = f"Unknown retrieval client mode: {client_mode!r}"
        raise ValueError(msg)
    raw_duration = payload.get("duration_ms")
    duration_ms = _optional_float(raw_duration) if raw_duration is not None else None
    return RetrievalMetadata(
        client_mode=cast("RetrievalClientMode", client_mode),
        source=_required_str(payload.get("source"), "retrieval metadata source"),
        duration_ms=duration_ms,
    )


def _serialize_assessment_metadata(item: AssessmentMetadata) -> dict[str, JsonValue]:
    return {"mellea_concurrency": item.mellea_concurrency}


def _deserialize_assessment_metadata(payload: Mapping[str, object]) -> AssessmentMetadata:
    return AssessmentMetadata(
        mellea_concurrency=_optional_int(payload.get("mellea_concurrency")),
    )


def _serialize_citation(citation: CanonicalCitation) -> dict[str, JsonValue]:
    payload = cast("dict[str, JsonValue]", asdict(citation))
    payload["type"] = citation_kind(citation).value
    return payload


def _deserialize_citation(payload: Mapping[str, object]) -> CanonicalCitation:
    validated = _CITATION_PAYLOAD_ADAPTER.validate_python(payload)
    normalized = validated.model_dump(mode="python")
    kind = CitationKind(normalized["type"])
    citation_cls = _CITATION_CLASSES[kind]
    kwargs = {
        field.name: _optional_str(normalized.get(field.name))
        for field in fields(citation_cls)
        if field.name in normalized
    }
    return cast("CanonicalCitation", citation_cls(**kwargs))


def serialize_extracted_citation(item: ExtractedCitation) -> dict[str, JsonValue]:
    """Serialize one extracted citation into a UI-agnostic JSON-ready dict."""
    return {
        "citation_id": item.citation_id,
        "span": cast("dict[str, JsonValue]", asdict(item.span)),
        "matched_text": item.matched_text,
        "citation": _serialize_citation(item.citation),
        "resolves_to": item.resolves_to,
    }


def deserialize_extracted_citation(payload: Mapping[str, object]) -> ExtractedCitation:
    """Rebuild one extracted citation from JSON data."""
    validated = ExtractedCitationPayload.model_validate(payload).model_dump(mode="python")
    return ExtractedCitation(
        citation_id=_required_str(validated.get("citation_id"), "citation_id"),
        span=_deserialize_span(_mapping_field(validated.get("span"))),
        matched_text=_required_str(validated.get("matched_text"), "matched_text"),
        citation=_deserialize_citation(_mapping_field(validated.get("citation"))),
        resolves_to=_optional_str(validated.get("resolves_to")),
    )


def serialize_extracted_document(result: ExtractedDocument) -> dict[str, JsonValue]:
    """Serialize a full extraction artifact without annotation-tool assumptions."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "extracted_document",
        "text": result.text,
        "source_metadata": _serialize_source_metadata(result.source_metadata),
        "preprocessing_metadata": _serialize_preprocessing_metadata(result.preprocessing_metadata),
        "extraction_metadata": _serialize_extraction_metadata(result.extraction_metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
    }


def deserialize_extracted_document(payload: Mapping[str, object]) -> ExtractedDocument:
    """Rebuild the extraction boundary object from JSON data."""
    validated = ExtractedDocumentPayload.model_validate(payload).model_dump(mode="python")
    preprocessed = _deserialize_preprocessed_fields(validated)
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=tuple(
            deserialize_extracted_citation(_mapping_field(item))
            for item in _list_field(validated.get("citations"))
        ),
        extraction_metadata=_deserialize_extraction_metadata(
            _required_mapping_field(validated, "extraction_metadata")
        ),
    )


def serialize_citation_retrieval(item: CitationRetrieval) -> dict[str, JsonValue]:
    """Serialize one citation retrieval result (discriminated by ``status``)."""
    base: dict[str, JsonValue] = {
        "citation_id": item.citation_id,
        "status": item.status.value,
        "source": item.source,
    }
    if isinstance(item, FoundCitationRetrieval):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "candidate": _serialize_retrieved_candidate(item.candidate),
                "extra_data": _serialize_extra_data(item.extra_data),
            },
        )
    elif isinstance(item, AmbiguousCitationRetrieval):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "candidates": [_serialize_retrieved_candidate(candidate) for candidate in item.candidates],
                "extra_data": _serialize_extra_data(item.extra_data),
            },
        )
    elif isinstance(item, NotFoundCitationRetrieval):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "candidate_search": _serialize_case_name_search_trace(item.candidate_search),
                "extra_data": _serialize_extra_data(item.extra_data),
            },
        )
    elif isinstance(item, (ThrottledCitationRetrieval, LookupFailedCitationRetrieval)):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "error_message": item.error_message,
                "failure_detail": (
                    _serialize_retrieval_failure_detail(item.failure_detail)
                    if item.failure_detail is not None
                    else None
                ),
                "extra_data": _serialize_extra_data(item.extra_data),
            },
        )
    # Invalid and Skipped carry only the common fields.
    return base


def _serialize_court_resolution_trace(item: CourtResolutionTrace) -> dict[str, JsonValue]:
    """Serialize the court resolution trace for a found citation."""
    return {
        "courtlistener_court_id": item.courtlistener_court_id,
        "resolved_via": item.resolved_via.value,
        "docket_id": item.docket_id,
        "docket_url": item.docket_url,
        "cached": item.cached,
        "error_message": item.error_message,
    }


def _serialize_retrieved_candidate(
    item: RetrievedCandidate,
) -> dict[str, JsonValue]:
    """Serialize one retrieved candidate with identity and provenance."""
    return {
        "candidate_id": item.candidate_id,
        "record": _serialize_citation_record(item.record),
        "court_resolution": _serialize_court_resolution_trace(item.court_resolution),
    }


def _serialize_case_name_search_trace(item: CaseNameSearchTrace) -> dict[str, JsonValue]:
    """Serialize the case-name search trace for a not-found citation."""
    return {
        "status": item.status.value,
        "query": item.query,
        "probes": [
            {
                "corpus": probe.corpus.value,
                "status": probe.status.value,
                "http_status": probe.http_status,
                "case_count": probe.case_count,
                "error_message": probe.error_message,
            }
            for probe in item.probes
        ],
    }


def _deserialize_case_name_search_trace(
    payload: Mapping[str, object] | None,
) -> CaseNameSearchTrace:
    """Rebuild a case-name search trace, tolerating payloads that predate it."""
    if payload is None:
        return CaseNameSearchTrace()
    validated = CaseNameSearchTracePayload.model_validate(payload).model_dump(mode="python")
    status = CaseNameSearchStatus(_required_str(validated.get("status"), "candidate search status"))
    probes = tuple(
        CaseNameSearchProbe(
            corpus=CaseNameSearchCorpus(_required_str(probe.get("corpus"), "search corpus")),
            status=CaseNameSearchStatus(_required_str(probe.get("status"), "probe status")),
            http_status=_optional_int(probe.get("http_status")),
            case_count=_optional_int(probe.get("case_count")),
            error_message=_optional_str(probe.get("error_message")),
        )
        for probe in validated.get("probes", [])
    )
    return CaseNameSearchTrace(
        status=status,
        query=_optional_str(validated.get("query")),
        probes=probes,
    )


def _deserialize_court_resolution_trace(
    payload: Mapping[str, object],
) -> CourtResolutionTrace:
    """Rebuild a court resolution trace from JSON data."""
    validated = CourtResolutionTracePayload.model_validate(payload).model_dump(mode="python")
    return CourtResolutionTrace(
        courtlistener_court_id=_optional_str(validated.get("courtlistener_court_id")),
        resolved_via=CourtResolutionSource(_required_str(validated.get("resolved_via"), "resolved_via")),
        docket_id=_optional_str(validated.get("docket_id")),
        docket_url=_optional_str(validated.get("docket_url")),
        cached=bool(validated.get("cached", False)),
        error_message=_optional_str(validated.get("error_message")),
    )


def _deserialize_retrieved_candidate(
    payload: Mapping[str, object],
) -> RetrievedCandidate:
    """Rebuild one retrieved candidate with identity and provenance."""
    return RetrievedCandidate(
        candidate_id=_required_str(payload.get("candidate_id"), "candidate_id"),
        record=_deserialize_citation_record(_required_mapping_field(payload, "record")),
        court_resolution=_deserialize_court_resolution_trace(
            _required_mapping_field(payload, "court_resolution")
        ),
    )


def _serialize_citation_record(item: CourtListenerCitationRecord) -> dict[str, JsonValue]:
    return {
        "case_name": item.case_name,
        "date_filed": item.date_filed,
        "court": item.court,
        "court_id": item.court_id,
        "docket_id": item.docket_id,
        "extra_data": _serialize_extra_data(item.extra_data),
    }


def _deserialize_citation_record(payload: Mapping[str, object]) -> CourtListenerCitationRecord:
    return CourtListenerCitationRecord(
        case_name=_optional_str(payload.get("case_name")),
        date_filed=_optional_str(payload.get("date_filed")),
        court=_optional_str(payload.get("court")),
        court_id=_optional_str(payload.get("court_id")),
        docket_id=_optional_str(payload.get("docket_id")),
        extra_data=_deserialize_extra_data(payload.get("extra_data")),
    )


def _serialize_retrieval_failure_detail(
    item: RetrievalFailureDetail,
) -> dict[str, JsonValue]:
    return {
        "failure_type": item.failure_type,
        "message": item.message,
        "retryable": item.retryable,
        "upstream_status_code": item.upstream_status_code,
        "key": item.key,
        "url": item.url,
        "retry_after_seconds": item.retry_after_seconds,
        "extra_data": _serialize_extra_data(item.extra_data),
    }


def _deserialize_retrieval_failure_detail(
    payload: Mapping[str, object],
) -> RetrievalFailureDetail:
    retryable = payload.get("retryable")
    if retryable is not None and not isinstance(retryable, bool):
        msg = "failure_detail.retryable must be a boolean"
        raise TypeError(msg)
    return RetrievalFailureDetail(
        failure_type=_optional_str(payload.get("failure_type")),
        message=_optional_str(payload.get("message")),
        retryable=retryable,
        upstream_status_code=_optional_int(payload.get("upstream_status_code")),
        key=_optional_str(payload.get("key")),
        url=_optional_str(payload.get("url")),
        retry_after_seconds=_optional_float(payload.get("retry_after_seconds")),
        extra_data=_deserialize_extra_data(payload.get("extra_data")),
    )


def deserialize_citation_retrieval(payload: Mapping[str, object]) -> CitationRetrieval:
    """Rebuild one citation retrieval result from JSON data (status-discriminated)."""
    validated = _RETRIEVAL_PAYLOAD_ADAPTER.validate_python(payload).model_dump(mode="python")
    status = RetrievalStatus(_required_str(validated.get("status"), "retrieval status"))
    citation_id = _required_str(validated.get("citation_id"), "citation_id")
    source = _required_str(validated.get("source"), "retrieval source")

    if status is RetrievalStatus.FOUND:
        # Pydantic discriminated-union model ensures the "found" branch has these fields.
        return FoundCitationRetrieval(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            candidate=_deserialize_retrieved_candidate(_required_mapping_field(validated, "candidate")),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is RetrievalStatus.AMBIGUOUS:
        return AmbiguousCitationRetrieval(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            candidates=tuple(
                _deserialize_retrieved_candidate(_mapping_field(item))
                for item in _list_field(validated.get("candidates"))
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is RetrievalStatus.NOT_FOUND:
        candidate_search = validated.get("candidate_search")
        return NotFoundCitationRetrieval(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            candidate_search=_deserialize_case_name_search_trace(
                candidate_search if isinstance(candidate_search, Mapping) else None,
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is RetrievalStatus.INVALID:
        return InvalidCitationRetrieval(
            citation_id=citation_id,
            source=source,
        )
    if status is RetrievalStatus.THROTTLED:
        return ThrottledCitationRetrieval(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            error_message=_optional_str(validated.get("error_message")),
            failure_detail=(
                _deserialize_retrieval_failure_detail(_mapping_field(validated.get("failure_detail")))
                if isinstance(validated.get("failure_detail"), dict)
                else None
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is RetrievalStatus.LOOKUP_FAILED:
        return LookupFailedCitationRetrieval(
            citation_id=citation_id,
            locator=_optional_str(validated.get("locator")) or "",
            source=source,
            lookup_status=_optional_int(validated.get("lookup_status")),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            error_message=_optional_str(validated.get("error_message")),
            failure_detail=(
                _deserialize_retrieval_failure_detail(_mapping_field(validated.get("failure_detail")))
                if isinstance(validated.get("failure_detail"), dict)
                else None
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    # Skipped
    return SkippedCitationRetrieval(
        citation_id=citation_id,
        source=source,
    )


def serialize_retrieved_document(result: RetrievedDocument) -> dict[str, JsonValue]:
    """Serialize the retrieval boundary object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "retrieved_document",
        "text": result.text,
        "source_metadata": _serialize_source_metadata(result.source_metadata),
        "preprocessing_metadata": _serialize_preprocessing_metadata(result.preprocessing_metadata),
        "extraction_metadata": _serialize_extraction_metadata(result.extraction_metadata),
        "retrieval_metadata": _serialize_retrieval_metadata(result.retrieval_metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "retrievals": [serialize_citation_retrieval(item) for item in result.retrievals],
    }


def deserialize_retrieved_document(payload: Mapping[str, object]) -> RetrievedDocument:
    """Rebuild the retrieval boundary object from JSON data."""
    validated = RetrievedDocumentPayload.model_validate(payload).model_dump(mode="python")
    preprocessed = _deserialize_preprocessed_fields(validated)
    return RetrievedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=tuple(
            deserialize_extracted_citation(_mapping_field(item))
            for item in _list_field(validated.get("citations"))
        ),
        extraction_metadata=_deserialize_extraction_metadata(
            _required_mapping_field(validated, "extraction_metadata")
        ),
        retrievals=tuple(
            deserialize_citation_retrieval(_mapping_field(item))
            for item in _list_field(validated.get("retrievals"))
        ),
        retrieval_metadata=_deserialize_retrieval_metadata(
            _required_mapping_field(validated, "retrieval_metadata")
        ),
    )


def serialize_case_name_assessment(item: CaseNameAssessment) -> dict[str, JsonValue]:
    """Serialize a case-name assessment."""
    return {
        "status": item.status.value,
        "extracted_case_name": item.extracted_case_name,
        "courtlistener_case_name": item.courtlistener_case_name,
        "message": item.message,
        "chat_history": (
            [_serialize_chat_turn(turn) for turn in item.chat_history]
            if item.chat_history is not None
            else None
        ),
    }


def _serialize_chat_turn(item: ChatTurn) -> dict[str, JsonValue]:
    return {
        "role": item.role,
        "content": item.content,
        "extra_data": _serialize_extra_data(item.extra_data),
    }


def _deserialize_chat_turn(payload: Mapping[str, object]) -> ChatTurn:
    return ChatTurn(
        role=_required_str(payload.get("role"), "chat turn role"),
        content=_required_str(payload.get("content"), "chat turn content"),
        extra_data=_deserialize_extra_data(payload.get("extra_data")),
    )


def deserialize_case_name_assessment(payload: Mapping[str, object]) -> CaseNameAssessment:
    """Rebuild a case-name assessment from JSON data."""
    validated = CaseNameAssessmentPayload.model_validate(payload).model_dump(mode="python")
    raw_history = validated.get("chat_history")
    chat_history = (
        tuple(_deserialize_chat_turn(t) for t in raw_history if isinstance(t, dict))
        if isinstance(raw_history, list)
        else None
    )
    return CaseNameAssessment(
        status=CaseNameAssessmentStatus(
            _required_str(validated.get("status"), "case-name assessment status")
        ),
        extracted_case_name=_optional_str(validated.get("extracted_case_name")),
        courtlistener_case_name=_optional_str(validated.get("courtlistener_case_name")),
        message=_required_str(validated.get("message"), "case-name assessment message"),
        chat_history=chat_history,
    )


def serialize_year_assessment(item: YearAssessment) -> dict[str, JsonValue]:
    """Serialize a year assessment."""
    payload = cast("dict[str, JsonValue]", asdict(item))
    payload["status"] = item.status.value
    return payload


def deserialize_year_assessment(payload: Mapping[str, object]) -> YearAssessment:
    """Rebuild a year assessment from JSON data."""
    validated = YearAssessmentPayload.model_validate(payload).model_dump(mode="python")
    return YearAssessment(
        status=YearAssessmentStatus(_required_str(validated.get("status"), "year assessment status")),
        extracted_year=_optional_str(validated.get("extracted_year")),
        courtlistener_year=_optional_str(validated.get("courtlistener_year")),
        message=_required_str(validated.get("message"), "year assessment message"),
    )


def serialize_court_assessment(item: CourtAssessment) -> dict[str, JsonValue]:
    """Serialize a court assessment."""
    payload = cast("dict[str, JsonValue]", asdict(item))
    payload["status"] = item.status.value
    payload["source"] = item.source
    if item.cl_court_taxonomy:
        payload["cl_court_taxonomy"] = asdict(item.cl_court_taxonomy)
    return payload


def deserialize_court_assessment(payload: Mapping[str, object]) -> CourtAssessment:
    """Rebuild a court assessment from JSON data."""
    validated = CourtAssessmentPayload.model_validate(payload).model_dump(mode="python")
    tax_payload = cast("dict[str, object] | None", validated.get("cl_court_taxonomy"))
    taxonomy = None
    if tax_payload:
        from mellea_lrc.courtlistener.taxonomy import CLCourtTaxonomy
        taxonomy = CLCourtTaxonomy(**tax_payload)  # type: ignore[arg-type]

    return CourtAssessment(
        status=CourtAssessmentStatus(_required_str(validated.get("status"), "court assessment status")),
        extracted_court=_optional_str(validated.get("extracted_court")),
        courtlistener_court_id=_optional_str(validated.get("courtlistener_court_id")),
        message=_required_str(validated.get("message"), "court assessment message"),
        source=validated.get("source", "direct"),  # type: ignore[arg-type]
        cl_court_taxonomy=taxonomy,
    )


def serialize_reporter_inference(
    item: ReporterJurisdictionInference,
) -> dict[str, JsonValue]:
    """Serialize reporter inference context."""
    return {
        "reporter": item.reporter,
        "status": item.status.value,
        "court_ids": list(item.court_ids),
        "evidence": [
            {"source": e.source, "statement": e.statement} for e in item.evidence
        ],
    }


def deserialize_reporter_inference(
    payload: Mapping[str, object],
) -> ReporterJurisdictionInference:
    """Rebuild reporter inference context."""
    validated = ReporterJurisdictionInferencePayload.model_validate(payload).model_dump(
        mode="python"
    )
    evidence = tuple(
        ReporterJurisdictionEvidence(
            source=_required_str(e.get("source"), "evidence source"),
            statement=_required_str(e.get("statement"), "evidence statement"),
        )
        for e in validated.get("evidence", [])
        if isinstance(e, dict)
    )
    return ReporterJurisdictionInference(
        reporter=_optional_str(validated.get("reporter")),
        status=ReporterJurisdictionStatus(
            _required_str(validated.get("status"), "reporter inference status")
        ),
        court_ids=tuple(
            str(c) for c in validated.get("court_ids", []) if isinstance(c, str)
        ),
        evidence=evidence,
    )


def serialize_citation_assessment(item: CitationAssessment) -> dict[str, JsonValue]:
    """Serialize one discriminated citation-assessment execution state."""
    payload: dict[str, JsonValue] = {
        "citation_id": item.citation_id,
        "status": item.status.value,
    }
    if isinstance(item, SkippedCitationAssessment):
        payload["reason"] = item.reason.value
        payload["message"] = item.message
    elif isinstance(item, AssessedCitationAssessment):
        payload["candidate_id"] = item.candidate_id
        payload["result"] = serialize_citation_assessment_result(item.result)
    elif isinstance(item, AmbiguousCitationAssessment):
        payload["candidates"] = [
            {
                "candidate_id": candidate.candidate_id,
                "result": serialize_citation_assessment_result(candidate.result),
            }
            for candidate in item.candidates
        ]
        payload["gated"] = item.gated
        payload["message"] = item.message
    elif isinstance(item, FailedCitationAssessment):
        payload["error"] = item.error
    return payload


def serialize_citation_assessment_result(
    item: CitationAssessmentResult,
) -> dict[str, JsonValue]:
    """Serialize a completed substantive citation assessment."""
    return {
        "case_name": serialize_case_name_assessment_run(item.case_name),
        "reporter_inference": serialize_reporter_inference(item.reporter_inference),
        "court": serialize_court_assessment(item.court),
        "year": serialize_year_assessment(item.year),
    }


def serialize_case_name_assessment_run(item: CaseNameAssessmentRun) -> dict[str, JsonValue]:
    """Serialize the initial case-name assessment and follow-up."""
    return {
        "initial": serialize_case_name_assessment(item.initial),
        "followup": serialize_case_name_followup(item.followup),
    }


def serialize_reextracted_case_name(item: ReextractedCaseName) -> dict[str, JsonValue]:
    """Serialize a grounded case-name extraction."""
    return {
        "case_name": item.case_name,
        "case_name_span": _serialize_span(item.case_name_span),
    }


def serialize_case_name_followup(item: CaseNameFollowup) -> dict[str, JsonValue]:
    """Serialize one field-local case-name follow-up outcome."""
    payload: dict[str, JsonValue] = {"status": item.status.value}
    if isinstance(item, CaseNameReextractionFailed):
        payload["error"] = item.error
    elif isinstance(item, CaseNameReassessed):
        payload["reextracted_case_name"] = serialize_reextracted_case_name(item.reextracted_case_name)
        payload["result"] = serialize_case_name_assessment(item.result)
    elif isinstance(item, CaseNameReassessmentFailed):
        payload["reextracted_case_name"] = serialize_reextracted_case_name(item.reextracted_case_name)
        payload["error"] = item.error
    return payload


def deserialize_citation_assessment(payload: Mapping[str, object]) -> CitationAssessment:
    """Rebuild one discriminated citation-assessment execution state."""
    validated_model = _ASSESSMENT_PAYLOAD_ADAPTER.validate_python(payload)
    validated = validated_model.model_dump(mode="python")
    citation_id = _required_str(validated.get("citation_id"), "citation_id")
    status = AssessmentStatus(_required_str(validated.get("status"), "assessment status"))
    if status == AssessmentStatus.WAITING:
        return WaitingCitationAssessment(citation_id=citation_id)
    if status == AssessmentStatus.SKIPPED:
        return SkippedCitationAssessment(
            citation_id=citation_id,
            reason=AssessmentSkipReason(_required_str(validated.get("reason"), "assessment skip reason")),
            message=_required_str(validated.get("message"), "skipped citation message"),
        )
    if status == AssessmentStatus.ASSESSED:
        result_payload = validated.get("result")
        if not isinstance(result_payload, dict):
            msg = "assessed citation requires a result"
            raise TypeError(msg)
        return AssessedCitationAssessment(
            citation_id=citation_id,
            candidate_id=_required_str(validated.get("candidate_id"), "candidate_id"),
            result=deserialize_citation_assessment_result(result_payload),
        )
    if status == AssessmentStatus.AMBIGUOUS:
        return AmbiguousCitationAssessment(
            citation_id=citation_id,
            candidates=tuple(
                CandidateAssessment(
                    candidate_id=_required_str(candidate.get("candidate_id"), "candidate_id"),
                    result=deserialize_citation_assessment_result(
                        _required_mapping_field(candidate, "result"),
                    ),
                )
                for candidate in _list_field(validated.get("candidates"))
            ),
            gated=bool(validated.get("gated", False)),
            message=_optional_str(validated.get("message")) or "",
        )
    return FailedCitationAssessment(
        citation_id=citation_id,
        error=_required_str(validated.get("error"), "failed citation error"),
    )


def deserialize_citation_assessment_result(
    payload: Mapping[str, object],
) -> CitationAssessmentResult:
    """Rebuild a completed substantive citation assessment."""
    validated = CitationAssessmentResultPayload.model_validate(payload).model_dump(mode="python")
    case_payload = validated.get("case_name")
    reporter_payload = validated.get("reporter_inference")
    court_payload = validated.get("court")
    year_payload = validated.get("year")
    if not all(
        isinstance(item, dict)
        for item in (case_payload, reporter_payload, court_payload, year_payload)
    ):
        msg = "citation assessment requires case_name, reporter_inference, court, and year"
        raise TypeError(msg)
    return CitationAssessmentResult(
        case_name=deserialize_case_name_assessment_run(_mapping_field(case_payload)),
        reporter_inference=deserialize_reporter_inference(
            _mapping_field(reporter_payload)
        ),
        court=deserialize_court_assessment(_mapping_field(court_payload)),
        year=deserialize_year_assessment(_mapping_field(year_payload)),
    )


def deserialize_case_name_assessment_run(payload: Mapping[str, object]) -> CaseNameAssessmentRun:
    """Rebuild an initial case-name assessment and its follow-up."""
    initial = deserialize_case_name_assessment(_required_mapping_field(payload, "initial"))
    followup = deserialize_case_name_followup(_required_mapping_field(payload, "followup"))
    return CaseNameAssessmentRun(initial=initial, followup=followup)


def deserialize_reextracted_case_name(payload: Mapping[str, object]) -> ReextractedCaseName:
    """Rebuild a grounded case-name extraction."""
    validated = ReextractedCaseNamePayload.model_validate(payload).model_dump(mode="python")
    return ReextractedCaseName(
        case_name=_required_str(validated.get("case_name"), "case_name"),
        case_name_span=_deserialize_span(_required_mapping_field(validated, "case_name_span")),
    )


def deserialize_case_name_followup(payload: Mapping[str, object]) -> CaseNameFollowup:
    """Rebuild one field-local case-name follow-up outcome."""
    validated_model = _CASE_NAME_FOLLOWUP_PAYLOAD_ADAPTER.validate_python(payload)
    validated = validated_model.model_dump(mode="python")
    status = CaseNameFollowupStatus(_required_str(validated.get("status"), "case-name follow-up status"))
    if status == CaseNameFollowupStatus.NOT_REQUIRED:
        return CaseNameReassessmentNotRequired()
    if status == CaseNameFollowupStatus.REEXTRACTION_FAILED:
        return CaseNameReextractionFailed(
            error=_required_str(validated.get("error"), "re-extraction failure error"),
        )
    reextracted = deserialize_reextracted_case_name(
        _required_mapping_field(validated, "reextracted_case_name")
    )
    if status == CaseNameFollowupStatus.REASSESSMENT_FAILED:
        return CaseNameReassessmentFailed(
            reextracted_case_name=reextracted,
            error=_required_str(validated.get("error"), "reassessment failure error"),
        )
    return CaseNameReassessed(
        reextracted_case_name=reextracted,
        result=deserialize_case_name_assessment(_required_mapping_field(validated, "result")),
    )


def serialize_assessed_document(result: AssessedDocument) -> dict[str, JsonValue]:
    """Serialize the assessment boundary object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "assessed_document",
        "text": result.text,
        "source_metadata": _serialize_source_metadata(result.source_metadata),
        "preprocessing_metadata": _serialize_preprocessing_metadata(result.preprocessing_metadata),
        "extraction_metadata": _serialize_extraction_metadata(result.extraction_metadata),
        "retrieval_metadata": _serialize_retrieval_metadata(result.retrieval_metadata),
        "assessment_metadata": _serialize_assessment_metadata(result.assessment_metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "retrievals": [serialize_citation_retrieval(item) for item in result.retrievals],
        "assessments": [serialize_citation_assessment(item) for item in result.assessments],
    }


def deserialize_assessed_document(payload: Mapping[str, object]) -> AssessedDocument:
    """Rebuild the assessment boundary object from JSON data."""
    validated = AssessedDocumentPayload.model_validate(payload).model_dump(mode="python")
    preprocessed = _deserialize_preprocessed_fields(validated)
    return AssessedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=tuple(
            deserialize_extracted_citation(_mapping_field(item))
            for item in _list_field(validated.get("citations"))
        ),
        extraction_metadata=_deserialize_extraction_metadata(
            _required_mapping_field(validated, "extraction_metadata")
        ),
        retrievals=tuple(
            deserialize_citation_retrieval(_mapping_field(item))
            for item in _list_field(validated.get("retrievals"))
        ),
        retrieval_metadata=_deserialize_retrieval_metadata(
            _required_mapping_field(validated, "retrieval_metadata")
        ),
        assessments=tuple(
            deserialize_citation_assessment(_mapping_field(item))
            for item in _list_field(validated.get("assessments"))
        ),
        assessment_metadata=_deserialize_assessment_metadata(
            _required_mapping_field(validated, "assessment_metadata")
        ),
    )


def _serialize_span(span: Span) -> dict[str, JsonValue]:
    return {"start": span.start, "end": span.end}


def _deserialize_span(payload: Mapping[str, object]) -> Span:
    return Span(start=_int_field(payload.get("start")), end=_int_field(payload.get("end")))


def _mapping_field(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return value
    msg = "Expected an object"
    raise TypeError(msg)


def _required_mapping_field(
    payload: Mapping[str, object],
    field_name: str,
) -> Mapping[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        msg = f"{field_name} must be an object"
        raise TypeError(msg)
    return value


def _list_field(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    msg = "Expected an array"
    raise TypeError(msg)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = "Expected a string or null"
    raise TypeError(msg)


def _required_str(value: object, field_name: str) -> str:
    text = _optional_str(value)
    if text is None:
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)
    return text


def _required_int(value: object, field_name: str) -> int:
    if value is None:
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)
    parsed = _optional_int(value)
    if parsed is None:
        msg = f"{field_name} must be an integer"
        raise ValueError(msg)
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return _int_field(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_field(value: object) -> int:
    if type(value) is int:
        return value
    msg = "Expected an integer"
    raise TypeError(msg)
