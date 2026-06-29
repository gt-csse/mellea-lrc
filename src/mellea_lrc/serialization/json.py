"""JSON-ready serialization for reusable mellea-lrc artifacts."""

from __future__ import annotations

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
    CaseNameAssessmentStatus,
    ChatTurn,
    CitationAssessment,
    CitationAssessmentResult,
    CitationReassessment,
    FailedCitationAssessment,
    ModifiedExtractedCitation,
    ReassessedCitationReassessment,
    ReassessmentFailedCitationReassessment,
    ReassessmentSkipReason,
    ReassessmentStatus,
    ReextractionFailedCitationReassessment,
    SkippedCitationAssessment,
    SkippedCitationReassessment,
    WaitingCitationAssessment,
    WaitingCitationReassessment,
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
    CitationValidation,
    ValidatedDocument,
    ValidationMetadata,
    ValidationClientMode,
    ValidationStatus,
)
from mellea_lrc.courtlistener.types import CitationMatch, ValidationFailureDetail
from mellea_lrc.serialization.transport import (
    AssessedDocumentPayload,
    CanonicalCitationPayload,
    CaseNameAssessmentPayload,
    CitationAssessmentPayload,
    CitationAssessmentResultPayload,
    CitationReassessmentPayload,
    CitationValidationPayload,
    ExtractedCitationPayload,
    ExtractedDocumentPayload,
    ModifiedExtractedCitationPayload,
    PreprocessedDocumentPayload,
    ValidatedDocumentPayload,
    YearAssessmentPayload,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.core.citations import CanonicalCitation

JsonValue: TypeAlias = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
SCHEMA_VERSION = 6

_CITATION_PAYLOAD_ADAPTER = TypeAdapter(CanonicalCitationPayload)
_ASSESSMENT_PAYLOAD_ADAPTER = TypeAdapter(CitationAssessmentPayload)
_REASSESSMENT_PAYLOAD_ADAPTER = TypeAdapter(CitationReassessmentPayload)

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
    return {"client_mode": item.client_mode, "source": item.source}


def _deserialize_validation_metadata(payload: Mapping[str, object]) -> ValidationMetadata:
    client_mode = _required_str(payload.get("client_mode"), "validation client mode")
    if client_mode not in {"deployed", "sdk", "custom"}:
        msg = f"Unknown validation client mode: {client_mode!r}"
        raise ValueError(msg)
    return ValidationMetadata(
        client_mode=cast("ValidationClientMode", client_mode),
        source=_required_str(payload.get("source"), "validation metadata source"),
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
    """Serialize one citation validation result."""
    return {
        "citation_id": item.citation_id,
        "locator": item.locator,
        "status": item.status.value,
        "source": item.source,
        "message": item.message,
        "lookup_status": item.lookup_status,
        "lookup_cache": item.lookup_cache,
        "lookup_key": item.lookup_key,
        "error_message": item.error_message,
        "failure_detail": (
            _serialize_validation_failure_detail(item.failure_detail)
            if item.failure_detail is not None
            else None
        ),
        "matches": [_serialize_citation_match(match) for match in item.matches],
        "extra_data": _serialize_extra_data(item.extra_data),
    }


def _serialize_citation_match(item: CitationMatch) -> dict[str, JsonValue]:
    return {
        "case_name": item.case_name,
        "date_filed": item.date_filed,
        "court": item.court,
        "extra_data": _serialize_extra_data(item.extra_data),
    }


def _deserialize_citation_match(payload: Mapping[str, object]) -> CitationMatch:
    return CitationMatch(
        case_name=_optional_str(payload.get("case_name")),
        date_filed=_optional_str(payload.get("date_filed")),
        court=_optional_str(payload.get("court")),
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
    """Rebuild one citation validation result from JSON data."""
    validated = CitationValidationPayload.model_validate(payload).model_dump(mode="python")
    return CitationValidation(
        citation_id=_required_str(validated.get("citation_id"), "citation_id"),
        locator=_optional_str(validated.get("locator")),
        status=ValidationStatus(_required_str(validated.get("status"), "validation status")),
        source=_required_str(validated.get("source"), "validation source"),
        message=_required_str(validated.get("message"), "validation message"),
        lookup_status=_optional_int(validated.get("lookup_status")),
        lookup_cache=_optional_str(validated.get("lookup_cache")),
        lookup_key=_optional_str(validated.get("lookup_key")),
        error_message=_optional_str(validated.get("error_message")),
        failure_detail=(
            _deserialize_validation_failure_detail(_mapping_field(validated.get("failure_detail")))
            if isinstance(validated.get("failure_detail"), dict)
            else None
        ),
        matches=tuple(
            _deserialize_citation_match(_mapping_field(item))
            for item in _list_field(validated.get("matches"))
        ),
        extra_data=_deserialize_extra_data(validated.get("extra_data")),
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
        "citation_id": item.citation_id,
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
        citation_id=_required_str(validated.get("citation_id"), "citation_id"),
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
        citation_id=_required_str(validated.get("citation_id"), "citation_id"),
        status=YearAssessmentStatus(_required_str(validated.get("status"), "year assessment status")),
        extracted_year=_optional_str(validated.get("extracted_year")),
        courtlistener_year=_optional_str(validated.get("courtlistener_year")),
        message=_required_str(validated.get("message"), "year assessment message"),
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
        payload["result"] = serialize_citation_assessment_result(item.result)
    elif isinstance(item, FailedCitationAssessment):
        payload["error"] = item.error
    return payload


def serialize_citation_assessment_result(
    item: CitationAssessmentResult,
) -> dict[str, JsonValue]:
    """Serialize a completed substantive citation assessment."""
    return {
        "citation_id": item.citation_id,
        "case_assess": serialize_case_name_assessment(item.case_assess),
        "year_assess": serialize_year_assessment(item.year_assess),
    }


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
    case_payload = validated.get("case_assess")
    year_payload = validated.get("year_assess")
    if not isinstance(case_payload, dict) or not isinstance(year_payload, dict):
        msg = "citation assessment requires case_assess and year_assess"
        raise TypeError(msg)
    return CitationAssessmentResult(
        citation_id=_required_str(validated.get("citation_id"), "citation_id"),
        case_assess=deserialize_case_name_assessment(_mapping_field(case_payload)),
        year_assess=deserialize_year_assessment(_mapping_field(year_payload)),
    )


def serialize_modified_extracted_citation(
    item: ModifiedExtractedCitation,
) -> dict[str, JsonValue]:
    """Serialize one modified extraction bound to a document citation."""
    payload = cast("dict[str, JsonValue]", asdict(item))
    payload["span"] = _serialize_span(item.span) if item.span is not None else None
    return payload


def deserialize_modified_extracted_citation(
    payload: Mapping[str, object],
) -> ModifiedExtractedCitation:
    """Rebuild one modified extraction from JSON data."""
    validated = ModifiedExtractedCitationPayload.model_validate(payload).model_dump(mode="python")
    span_payload = validated.get("span")
    case_name = _optional_str(validated.get("case_name"))
    return ModifiedExtractedCitation(
        citation_id=_required_str(validated.get("citation_id"), "citation_id"),
        span=_deserialize_span(_mapping_field(span_payload)) if isinstance(span_payload, dict) else None,
        matched_text=_optional_str(validated.get("matched_text")),
        case_name=case_name,
    )


def serialize_citation_reassessment(item: CitationReassessment) -> dict[str, JsonValue]:
    """Serialize one discriminated citation-reassessment execution state."""
    payload: dict[str, JsonValue] = {
        "citation_id": item.citation_id,
        "status": item.status.value,
    }
    if isinstance(item, SkippedCitationReassessment):
        payload["reason"] = item.reason.value
        payload["message"] = item.message
    elif isinstance(item, ReassessedCitationReassessment):
        payload["modified_citation"] = serialize_modified_extracted_citation(item.modified_citation)
        payload["result"] = serialize_case_name_assessment(item.result)
    elif isinstance(item, ReextractionFailedCitationReassessment):
        payload["error"] = item.error
    elif isinstance(item, ReassessmentFailedCitationReassessment):
        payload["modified_citation"] = serialize_modified_extracted_citation(item.modified_citation)
        payload["error"] = item.error
    return payload


def deserialize_citation_reassessment(payload: Mapping[str, object]) -> CitationReassessment:
    """Rebuild one discriminated citation-reassessment execution state."""
    validated_model = _REASSESSMENT_PAYLOAD_ADAPTER.validate_python(payload)
    validated = validated_model.model_dump(mode="python")
    citation_id = _required_str(validated.get("citation_id"), "citation_id")
    status = ReassessmentStatus(_required_str(validated.get("status"), "reassessment status"))
    if status == ReassessmentStatus.WAITING:
        return WaitingCitationReassessment(citation_id=citation_id)
    if status == ReassessmentStatus.SKIPPED:
        return SkippedCitationReassessment(
            citation_id=citation_id,
            reason=ReassessmentSkipReason(_required_str(validated.get("reason"), "reassessment skip reason")),
            message=_required_str(validated.get("message"), "skipped reassessment message"),
        )
    if status == ReassessmentStatus.REEXTRACTION_FAILED:
        return ReextractionFailedCitationReassessment(
            citation_id=citation_id,
            error=_required_str(validated.get("error"), "re-extraction failure error"),
        )

    modified_payload = _required_mapping_field(validated, "modified_citation")
    modified_citation = deserialize_modified_extracted_citation(modified_payload)
    if status == ReassessmentStatus.REASSESSMENT_FAILED:
        return ReassessmentFailedCitationReassessment(
            citation_id=citation_id,
            modified_citation=modified_citation,
            error=_required_str(validated.get("error"), "reassessment failure error"),
        )
    return ReassessedCitationReassessment(
        citation_id=citation_id,
        modified_citation=modified_citation,
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
        "reassessments": [serialize_citation_reassessment(item) for item in result.reassessments],
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
        reassessments=tuple(
            deserialize_citation_reassessment(_mapping_field(item))
            for item in _list_field(validated.get("reassessments"))
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
