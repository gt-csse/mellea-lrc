"""JSON-ready serialization for reusable mellea-lrc artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, TypeAlias, cast

from pydantic import TypeAdapter

from mellea_lrc.assessment.types import (
    AssessmentMetadata,
    AssessmentSkipReason,
    AssessmentStatus,
    AssessedCitationAssessment,
    AssessedDocument,
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
    CourtAssessmentRun,
    CourtAssessmentStatus,
    CourtFollowup,
    CourtFollowupNotRequired,
    CourtFollowupStatus,
    CourtInferredFromReporter,
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
from mellea_lrc.validation.types import (
    AmbiguousCitationValidation,
    CitationValidation,
    CourtResolutionSource,
    CourtResolutionTrace,
    FoundCitationValidation,
    InvalidCitationValidation,
    LookupFailedCitationValidation,
    NotFoundCitationValidation,
    SkippedCitationValidation,
    ThrottledCitationValidation,
    ValidatedDocument,
    ValidationClientMode,
    ValidationMetadata,
    ValidationStatus,
)
from mellea_lrc.courtlistener.types import CitationMatch, ValidationFailureDetail
from mellea_lrc.serialization.transport import (
    AssessedDocumentPayload,
    CanonicalCitationPayload,
    CaseNameAssessmentPayload,
    CaseNameFollowupPayload,
    CitationAssessmentPayload,
    CitationAssessmentResultPayload,
    CourtAssessmentPayload,
    CourtAssessmentRunPayload,
    CourtFollowupPayload,
    CitationValidationPayload,
    CourtResolutionTracePayload,
    ExtractedCitationPayload,
    ExtractedDocumentPayload,
    ReextractedCaseNamePayload,
    PreprocessedDocumentPayload,
    ValidatedDocumentPayload,
    YearAssessmentPayload,
)

if TYPE_CHECKING:
    from mellea_lrc.core.citations import CanonicalCitation

JsonValue: TypeAlias = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
SCHEMA_VERSION = 11

_CITATION_PAYLOAD_ADAPTER = TypeAdapter(CanonicalCitationPayload)
_ASSESSMENT_PAYLOAD_ADAPTER = TypeAdapter(CitationAssessmentPayload)
_CASE_NAME_FOLLOWUP_PAYLOAD_ADAPTER = TypeAdapter(CaseNameFollowupPayload)
_COURT_FOLLOWUP_PAYLOAD_ADAPTER = TypeAdapter(CourtFollowupPayload)
_VALIDATION_PAYLOAD_ADAPTER = TypeAdapter(CitationValidationPayload)

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


def _serialize_validation_metadata(item: ValidationMetadata) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"client_mode": item.client_mode, "source": item.source}
    if item.duration_ms is not None:
        payload["duration_ms"] = item.duration_ms
    return payload


def _deserialize_validation_metadata(payload: Mapping[str, object]) -> ValidationMetadata:
    client_mode = _required_str(payload.get("client_mode"), "validation client mode")
    if client_mode not in {"deployed", "sdk", "custom"}:
        msg = f"Unknown validation client mode: {client_mode!r}"
        raise ValueError(msg)
    raw_duration = payload.get("duration_ms")
    duration_ms = _optional_float(raw_duration) if raw_duration is not None else None
    return ValidationMetadata(
        client_mode=cast("ValidationClientMode", client_mode),
        source=_required_str(payload.get("source"), "validation metadata source"),
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


def serialize_citation_validation(item: CitationValidation) -> dict[str, JsonValue]:
    """Serialize one citation validation result (discriminated by ``status``)."""
    base: dict[str, JsonValue] = {
        "citation_id": item.citation_id,
        "status": item.status.value,
        "source": item.source,
        "message": item.message,
    }
    if isinstance(item, FoundCitationValidation):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "matches": [_serialize_citation_match(match) for match in item.matches],
                "court_resolution": _serialize_court_resolution_trace(item.court_resolution),
                "extra_data": _serialize_extra_data(item.extra_data),
            },
        )
    elif isinstance(item, AmbiguousCitationValidation):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "matches": [_serialize_citation_match(match) for match in item.matches],
                "extra_data": _serialize_extra_data(item.extra_data),
            },
        )
    elif isinstance(item, NotFoundCitationValidation):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "extra_data": _serialize_extra_data(item.extra_data),
            },
        )
    elif isinstance(item, (ThrottledCitationValidation, LookupFailedCitationValidation)):
        base.update(
            {
                "locator": item.locator,
                "lookup_status": item.lookup_status,
                "lookup_cache": item.lookup_cache,
                "lookup_key": item.lookup_key,
                "error_message": item.error_message,
                "failure_detail": (
                    _serialize_validation_failure_detail(item.failure_detail)
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


def _serialize_citation_match(item: CitationMatch) -> dict[str, JsonValue]:
    return {
        "case_name": item.case_name,
        "date_filed": item.date_filed,
        "court": item.court,
        "court_id": item.court_id,
        "extra_data": _serialize_extra_data(item.extra_data),
    }


def _deserialize_citation_match(payload: Mapping[str, object]) -> CitationMatch:
    return CitationMatch(
        case_name=_optional_str(payload.get("case_name")),
        date_filed=_optional_str(payload.get("date_filed")),
        court=_optional_str(payload.get("court")),
        court_id=_optional_str(payload.get("court_id")),
        extra_data=_deserialize_extra_data(payload.get("extra_data")),
    )


def _serialize_validation_failure_detail(
    item: ValidationFailureDetail,
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


def _deserialize_validation_failure_detail(
    payload: Mapping[str, object],
) -> ValidationFailureDetail:
    retryable = payload.get("retryable")
    if retryable is not None and not isinstance(retryable, bool):
        msg = "failure_detail.retryable must be a boolean"
        raise TypeError(msg)
    return ValidationFailureDetail(
        failure_type=_optional_str(payload.get("failure_type")),
        message=_optional_str(payload.get("message")),
        retryable=retryable,
        upstream_status_code=_optional_int(payload.get("upstream_status_code")),
        key=_optional_str(payload.get("key")),
        url=_optional_str(payload.get("url")),
        retry_after_seconds=_optional_float(payload.get("retry_after_seconds")),
        extra_data=_deserialize_extra_data(payload.get("extra_data")),
    )


def deserialize_citation_validation(payload: Mapping[str, object]) -> CitationValidation:
    """Rebuild one citation validation result from JSON data (status-discriminated)."""
    validated = _VALIDATION_PAYLOAD_ADAPTER.validate_python(payload).model_dump(mode="python")
    status = ValidationStatus(_required_str(validated.get("status"), "validation status"))
    citation_id = _required_str(validated.get("citation_id"), "citation_id")
    source = _required_str(validated.get("source"), "validation source")
    message = _required_str(validated.get("message"), "validation message")

    if status is ValidationStatus.FOUND:
        # Pydantic discriminated-union model ensures the "found" branch has these fields.
        return FoundCitationValidation(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            message=message,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            matches=tuple(
                _deserialize_citation_match(_mapping_field(item))
                for item in _list_field(validated.get("matches"))
            ),
            court_resolution=_deserialize_court_resolution_trace(
                _required_mapping_field(validated, "court_resolution"),
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is ValidationStatus.AMBIGUOUS:
        return AmbiguousCitationValidation(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            message=message,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            matches=tuple(
                _deserialize_citation_match(_mapping_field(item))
                for item in _list_field(validated.get("matches"))
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is ValidationStatus.NOT_FOUND:
        return NotFoundCitationValidation(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            message=message,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is ValidationStatus.INVALID:
        return InvalidCitationValidation(
            citation_id=citation_id,
            source=source,
            message=message,
        )
    if status is ValidationStatus.THROTTLED:
        return ThrottledCitationValidation(
            citation_id=citation_id,
            locator=_required_str(validated.get("locator"), "locator"),
            source=source,
            message=message,
            lookup_status=_required_int(validated.get("lookup_status"), "lookup_status"),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            error_message=_optional_str(validated.get("error_message")),
            failure_detail=(
                _deserialize_validation_failure_detail(_mapping_field(validated.get("failure_detail")))
                if isinstance(validated.get("failure_detail"), dict)
                else None
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    if status is ValidationStatus.LOOKUP_FAILED:
        return LookupFailedCitationValidation(
            citation_id=citation_id,
            locator=_optional_str(validated.get("locator")) or "",
            source=source,
            message=message,
            lookup_status=_optional_int(validated.get("lookup_status")),
            lookup_cache=_optional_str(validated.get("lookup_cache")),
            lookup_key=_optional_str(validated.get("lookup_key")),
            error_message=_optional_str(validated.get("error_message")),
            failure_detail=(
                _deserialize_validation_failure_detail(_mapping_field(validated.get("failure_detail")))
                if isinstance(validated.get("failure_detail"), dict)
                else None
            ),
            extra_data=_deserialize_extra_data(validated.get("extra_data")),
        )
    # Skipped
    return SkippedCitationValidation(
        citation_id=citation_id,
        source=source,
        message=message,
    )


def serialize_validated_document(result: ValidatedDocument) -> dict[str, JsonValue]:
    """Serialize the validation boundary object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "validated_document",
        "text": result.text,
        "source_metadata": _serialize_source_metadata(result.source_metadata),
        "preprocessing_metadata": _serialize_preprocessing_metadata(result.preprocessing_metadata),
        "extraction_metadata": _serialize_extraction_metadata(result.extraction_metadata),
        "validation_metadata": _serialize_validation_metadata(result.validation_metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "validations": [serialize_citation_validation(item) for item in result.validations],
    }


def deserialize_validated_document(payload: Mapping[str, object]) -> ValidatedDocument:
    """Rebuild the validation boundary object from JSON data."""
    validated = ValidatedDocumentPayload.model_validate(payload).model_dump(mode="python")
    preprocessed = _deserialize_preprocessed_fields(validated)
    return ValidatedDocument(
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
        validations=tuple(
            deserialize_citation_validation(_mapping_field(item))
            for item in _list_field(validated.get("validations"))
        ),
        validation_metadata=_deserialize_validation_metadata(
            _required_mapping_field(validated, "validation_metadata")
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
    return payload


def deserialize_court_assessment(payload: Mapping[str, object]) -> CourtAssessment:
    """Rebuild a court assessment from JSON data."""
    validated = CourtAssessmentPayload.model_validate(payload).model_dump(mode="python")
    return CourtAssessment(
        status=CourtAssessmentStatus(_required_str(validated.get("status"), "court assessment status")),
        extracted_court=_optional_str(validated.get("extracted_court")),
        courtlistener_court_id=_optional_str(validated.get("courtlistener_court_id")),
        message=_required_str(validated.get("message"), "court assessment message"),
    )


def serialize_court_assessment_run(item: CourtAssessmentRun) -> dict[str, JsonValue]:
    """Serialize the initial court assessment and follow-up."""
    return {
        "initial": serialize_court_assessment(item.initial),
        "followup": serialize_court_followup(item.followup),
    }


def serialize_court_followup(item: CourtFollowup) -> dict[str, JsonValue]:
    """Serialize one field-local court follow-up outcome."""
    payload: dict[str, JsonValue] = {"status": item.status.value}
    if isinstance(item, CourtInferredFromReporter):
        payload["reporter"] = item.reporter
        payload["citation_court_before"] = item.citation_court_before
        payload["result"] = serialize_court_assessment(item.result)
    return payload


def deserialize_court_assessment_run(payload: Mapping[str, object]) -> CourtAssessmentRun:
    """Rebuild an initial court assessment and its follow-up."""
    initial = deserialize_court_assessment(_required_mapping_field(payload, "initial"))
    followup = deserialize_court_followup(_required_mapping_field(payload, "followup"))
    return CourtAssessmentRun(initial=initial, followup=followup)


def deserialize_court_followup(payload: Mapping[str, object]) -> CourtFollowup:
    """Rebuild one field-local court follow-up outcome."""
    validated_model = _COURT_FOLLOWUP_PAYLOAD_ADAPTER.validate_python(payload)
    validated = validated_model.model_dump(mode="python")
    status = CourtFollowupStatus(_required_str(validated.get("status"), "court followup status"))
    if status == CourtFollowupStatus.NOT_REQUIRED:
        return CourtFollowupNotRequired()
    if status == CourtFollowupStatus.INFERRED_FROM_REPORTER:
        result_payload = validated.get("result")
        if not isinstance(result_payload, dict):
            msg = "inferred_from_reporter follow-up requires a result"
            raise TypeError(msg)
        return CourtInferredFromReporter(
            reporter=_optional_str(validated.get("reporter")),
            citation_court_before=_optional_str(validated.get("citation_court_before")),
            result=deserialize_court_assessment(result_payload),
        )
    msg = f"Unknown court follow-up status: {status.value}"
    raise ValueError(msg)


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
        payload["result"] = serialize_citation_assessment_result(item.result)
    elif isinstance(item, FailedCitationAssessment):
        payload["error"] = item.error
    return payload


def serialize_citation_assessment_result(
    item: CitationAssessmentResult,
) -> dict[str, JsonValue]:
    """Serialize a completed substantive citation assessment."""
    return {
        "case_name": serialize_case_name_assessment_run(item.case_name),
        "court": serialize_court_assessment_run(item.court),
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
            result=deserialize_citation_assessment_result(result_payload),
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
    court_payload = validated.get("court")
    year_payload = validated.get("year")
    if not all(isinstance(item, dict) for item in (case_payload, court_payload, year_payload)):
        msg = "citation assessment requires case_name, court, and year"
        raise TypeError(msg)
    return CitationAssessmentResult(
        case_name=deserialize_case_name_assessment_run(_mapping_field(case_payload)),
        court=deserialize_court_assessment_run(_mapping_field(court_payload)),
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
        "validation_metadata": _serialize_validation_metadata(result.validation_metadata),
        "assessment_metadata": _serialize_assessment_metadata(result.assessment_metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "validations": [serialize_citation_validation(item) for item in result.validations],
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
        validations=tuple(
            deserialize_citation_validation(_mapping_field(item))
            for item in _list_field(validated.get("validations"))
        ),
        validation_metadata=_deserialize_validation_metadata(
            _required_mapping_field(validated, "validation_metadata")
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
