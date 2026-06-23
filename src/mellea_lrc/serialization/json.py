"""JSON-ready serialization for reusable mellea-lrc artifacts."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, TYPE_CHECKING, TypeAlias, TypeVar, cast

from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentStatus,
    CitationAssessment,
    DocumentAssessment,
    ModifiedExtractedCitation,
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
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import DocumentExtraction, ExtractedCitation
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)
from mellea_lrc.validation.types import CitationValidation, DocumentValidation, ValidationStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.core.citations import CanonicalCitation

JsonValue: TypeAlias = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
T = TypeVar("T")

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
        "text": item.text,
        "metadata": _serialize_preprocessed_metadata(item.metadata),
    }


def deserialize_preprocessed_document(payload: Mapping[str, object]) -> PreprocessedDocument:
    """Rebuild the preprocessing boundary object from JSON data."""
    metadata_payload = _mapping_field(payload.get("metadata") or payload.get("preprocessing"))
    return PreprocessedDocument(
        text=str(payload.get("text") or ""),
        metadata=_deserialize_preprocessed_metadata(metadata_payload),
    )


def _serialize_preprocessed_metadata(
    item: PreprocessedDocumentMetadata,
) -> dict[str, JsonValue]:
    return {
        "source_path": item.source_path,
        "source_format": item.source_format.value,
        "backend": item.backend.value,
        "backend_version": item.backend_version,
        "header": item.header,
        "extras": dict(item.extras),
    }


def _deserialize_preprocessed_metadata(
    payload: Mapping[str, object],
) -> PreprocessedDocumentMetadata:
    return PreprocessedDocumentMetadata(
        source_path=_optional_str(payload.get("source_path")),
        source_format=_enum_field(SourceFormat, payload.get("source_format"), SourceFormat.UNKNOWN),
        backend=_enum_field(
            PreprocessingBackend,
            payload.get("backend"),
            PreprocessingBackend.PLAIN_TEXT,
        ),
        backend_version=_optional_str(payload.get("backend_version")),
        header=_optional_str(payload.get("header")),
        extras={str(key): str(value) for key, value in _mapping_field(payload.get("extras")).items()},
    )


def _serialize_citation(citation: CanonicalCitation) -> dict[str, JsonValue]:
    payload = cast("dict[str, JsonValue]", asdict(citation))
    payload["type"] = citation_kind(citation).value
    return payload


def _deserialize_citation(payload: Mapping[str, object]) -> CanonicalCitation:
    try:
        kind = CitationKind(str(payload.get("type") or CitationKind.UNKNOWN.value))
    except ValueError:
        kind = CitationKind.UNKNOWN
    citation_cls = _CITATION_CLASSES[kind]
    if not is_dataclass(citation_cls):
        return UnknownCitation()
    kwargs = {
        field.name: _optional_str(payload.get(field.name))
        for field in fields(citation_cls)
        if field.name in payload
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
    return ExtractedCitation(
        citation_id=str(payload.get("citation_id") or payload.get("id") or ""),
        span=_deserialize_span(_mapping_field(payload.get("span"))),
        matched_text=str(payload.get("matched_text") or ""),
        citation=_deserialize_citation(_mapping_field(payload.get("citation"))),
        resolves_to=_optional_str(payload.get("resolves_to")),
    )


def serialize_document_extraction(result: DocumentExtraction) -> dict[str, JsonValue]:
    """Serialize a full extraction artifact without annotation-tool assumptions."""
    return {
        "source_path": result.source_path,
        "text": result.text,
        "preprocessing": _serialize_preprocessed_metadata(result.preprocessed.metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "counts": {
            "total": len(result.citations),
            "full": len(result.full_citations),
            "by_type": _count_by_type(result),
        },
    }


def deserialize_document_extraction(payload: Mapping[str, object]) -> DocumentExtraction:
    """Rebuild the extraction boundary object from JSON data."""
    preprocessed = deserialize_preprocessed_document(
        {
            "text": payload.get("text"),
            "metadata": payload.get("preprocessing"),
        }
    )
    return DocumentExtraction(
        preprocessed=preprocessed,
        citations=tuple(
            deserialize_extracted_citation(_mapping_field(item))
            for item in _list_field(payload.get("citations"))
        ),
    )


def serialize_citation_validation(item: CitationValidation) -> dict[str, JsonValue]:
    """Serialize one citation validation result."""
    payload = cast("dict[str, JsonValue]", asdict(item))
    payload["status"] = item.status.value
    payload["case_names"] = list(item.case_names)
    payload["clusters"] = cast("list[JsonValue]", list(item.clusters))
    return payload


def deserialize_citation_validation(payload: Mapping[str, object]) -> CitationValidation:
    """Rebuild one citation validation result from JSON data."""
    return CitationValidation(
        citation_id=str(payload.get("citation_id") or ""),
        locator=_optional_str(payload.get("locator")),
        status=_enum_field(
            ValidationStatus,
            payload.get("status"),
            ValidationStatus.LOOKUP_FAILED,
        ),
        source=str(payload.get("source") or ""),
        message=str(payload.get("message") or ""),
        case_names=tuple(str(value) for value in _list_field(payload.get("case_names"))),
        lookup_status=_optional_int(payload.get("lookup_status")),
        lookup_cache=_optional_str(payload.get("lookup_cache")),
        lookup_key=_optional_str(payload.get("lookup_key")),
        error_message=_optional_str(payload.get("error_message")),
        limit_detail=_optional_json_object(payload.get("limit_detail")),
        clusters=tuple(_json_objects(payload.get("clusters"))),
    )


def serialize_document_validation(result: DocumentValidation) -> dict[str, JsonValue]:
    """Serialize the validation boundary object."""
    return {
        "source_path": result.source_path,
        "text": result.text,
        "preprocessing": _serialize_preprocessed_metadata(result.preprocessed.metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "validations": [serialize_citation_validation(item) for item in result.validations],
        "counts": {
            "total": len(result.validations),
            "found": len(result.found),
            "by_status": _count_validation_by_status(result),
        },
    }


def deserialize_document_validation(payload: Mapping[str, object]) -> DocumentValidation:
    """Rebuild the validation boundary object from JSON data."""
    preprocessed = deserialize_preprocessed_document(
        {
            "text": payload.get("text"),
            "metadata": payload.get("preprocessing"),
        }
    )
    return DocumentValidation(
        preprocessed=preprocessed,
        citations=tuple(
            deserialize_extracted_citation(_mapping_field(item))
            for item in _list_field(payload.get("citations"))
        ),
        validations=tuple(
            deserialize_citation_validation(_mapping_field(item))
            for item in _list_field(payload.get("validations"))
        ),
    )


def serialize_case_name_assessment(item: CaseNameAssessment) -> dict[str, JsonValue]:
    """Serialize a case-name assessment."""
    payload = cast("dict[str, JsonValue]", asdict(item))
    payload["status"] = item.status.value
    return payload


def deserialize_case_name_assessment(payload: Mapping[str, object]) -> CaseNameAssessment:
    """Rebuild a case-name assessment from JSON data."""
    return CaseNameAssessment(
        citation_id=str(payload.get("citation_id") or ""),
        status=_enum_field(
            CaseNameAssessmentStatus,
            payload.get("status"),
            CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
        ),
        extracted_case_name=_optional_str(payload.get("extracted_case_name")),
        courtlistener_case_name=_optional_str(payload.get("courtlistener_case_name")),
        message=str(payload.get("message") or ""),
    )


def serialize_year_assessment(item: YearAssessment) -> dict[str, JsonValue]:
    """Serialize a year assessment."""
    payload = cast("dict[str, JsonValue]", asdict(item))
    payload["status"] = item.status.value
    return payload


def deserialize_year_assessment(payload: Mapping[str, object]) -> YearAssessment:
    """Rebuild a year assessment from JSON data."""
    return YearAssessment(
        citation_id=str(payload.get("citation_id") or ""),
        status=_enum_field(
            YearAssessmentStatus,
            payload.get("status"),
            YearAssessmentStatus.MISSING,
        ),
        extracted_year=_optional_str(payload.get("extracted_year")),
        courtlistener_year=_optional_str(payload.get("courtlistener_year")),
        message=str(payload.get("message") or ""),
    )


def serialize_citation_assessment(item: CitationAssessment) -> dict[str, JsonValue]:
    """Serialize one citation assessment."""
    return {
        "citation_id": item.citation_id,
        "status": item.status.value,
        "message": item.message,
        "case_assess": serialize_case_name_assessment(item.case_assess),
        "year_assess": serialize_year_assessment(item.year_assess),
    }


def deserialize_citation_assessment(payload: Mapping[str, object]) -> CitationAssessment:
    """Rebuild one citation assessment from JSON data."""
    case_payload = payload.get("case_assess")
    year_payload = payload.get("year_assess")
    if not isinstance(case_payload, dict) or not isinstance(year_payload, dict):
        msg = "citation assessment requires case_assess and year_assess"
        raise ValueError(msg)
    return CitationAssessment(
        citation_id=str(payload.get("citation_id") or ""),
        case_assess=deserialize_case_name_assessment(_mapping_field(case_payload)),
        year_assess=deserialize_year_assessment(_mapping_field(year_payload)),
    )


def serialize_modified_extracted_citation(
    item: ModifiedExtractedCitation,
) -> dict[str, JsonValue]:
    """Serialize one modified extraction bound to a document citation."""
    payload = cast("dict[str, JsonValue]", asdict(item))
    payload["span"] = _serialize_span(item.span) if item.span is not None else None
    payload["extracted_case_name"] = item.extracted_case_name
    return payload


def deserialize_modified_extracted_citation(
    payload: Mapping[str, object],
) -> ModifiedExtractedCitation:
    """Rebuild one modified extraction from JSON data."""
    span_payload = payload.get("span")
    return ModifiedExtractedCitation(
        citation_id=str(payload.get("citation_id") or ""),
        span=_deserialize_span(_mapping_field(span_payload)) if isinstance(span_payload, dict) else None,
        matched_text=_optional_str(payload.get("matched_text")),
        plaintiff=_optional_str(payload.get("plaintiff")),
        defendant=_optional_str(payload.get("defendant")),
        case_name=_optional_str(payload.get("case_name")),
    )


def serialize_document_assessment(result: DocumentAssessment) -> dict[str, JsonValue]:
    """Serialize the assessment boundary object."""
    return {
        "source_path": result.source_path,
        "text": result.text,
        "preprocessing": _serialize_preprocessed_metadata(result.preprocessed.metadata),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "validations": [serialize_citation_validation(item) for item in result.validations],
        "assessments": [serialize_citation_assessment(item) for item in result.assessments],
        "modified_citations": [
            serialize_modified_extracted_citation(item) for item in result.modified_citations
        ],
        "reassessments": [serialize_citation_assessment(item) for item in result.reassessments],
        "counts": _count_assessment_by_status(result),
        "case_name_counts": _count_assessment_case_names_by_status(result),
        "year_counts": _count_assessment_years_by_status(result),
    }


def deserialize_document_assessment(payload: Mapping[str, object]) -> DocumentAssessment:
    """Rebuild the assessment boundary object from JSON data."""
    preprocessed = deserialize_preprocessed_document(
        {
            "text": payload.get("text"),
            "metadata": payload.get("preprocessing"),
        }
    )
    return DocumentAssessment(
        preprocessed=preprocessed,
        citations=tuple(
            deserialize_extracted_citation(_mapping_field(item))
            for item in _list_field(payload.get("citations"))
        ),
        validations=tuple(
            deserialize_citation_validation(_mapping_field(item))
            for item in _list_field(payload.get("validations"))
        ),
        assessments=tuple(
            deserialize_citation_assessment(_mapping_field(item))
            for item in _list_field(payload.get("assessments"))
        ),
        modified_citations=tuple(
            deserialize_modified_extracted_citation(_mapping_field(item))
            for item in _list_field(payload.get("modified_citations"))
        ),
        reassessments=tuple(
            deserialize_citation_assessment(_mapping_field(item))
            for item in _list_field(payload.get("reassessments"))
        ),
    )


def _count_by_type(result: DocumentExtraction) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.citations:
        kind = citation_kind(item.citation).value
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _count_validation_by_status(result: DocumentValidation) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.validations:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    return counts


def _count_assessment_by_status(result: DocumentAssessment) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.assessments:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    return counts


def _count_assessment_case_names_by_status(result: DocumentAssessment) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.assessments:
        counts[item.case_assess.status.value] = counts.get(item.case_assess.status.value, 0) + 1
    return counts


def _count_assessment_years_by_status(result: DocumentAssessment) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.assessments:
        counts[item.year_assess.status.value] = counts.get(item.year_assess.status.value, 0) + 1
    return counts


def _serialize_span(span: Span) -> dict[str, JsonValue]:
    return {"start": span.start, "end": span.end}


def _deserialize_span(payload: Mapping[str, object]) -> Span:
    return Span(start=_int_field(payload.get("start")), end=_int_field(payload.get("end")))


def _enum_field(enum_cls: type[T], value: object, default: T) -> T:
    try:
        return enum_cls(str(value)) if value is not None else default
    except ValueError:
        return default


def _mapping_field(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _list_field(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return _int_field(value)
    except (TypeError, ValueError):
        return None


def _int_field(value: object) -> int:
    if isinstance(value, int):
        return value
    return int(str(value or 0))


def _optional_json_object(value: object) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None
    return _json_object(value)


def _json_objects(value: object) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        return []
    return [_json_object(item) for item in value if isinstance(item, dict)]


def _json_object(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    return cast("dict[str, JsonValue]", dict(value))
