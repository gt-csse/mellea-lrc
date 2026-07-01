"""Assembled extraction and validation API for the Modal E2E backend."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mellea_lrc.assessment import (
    AssessedCitationAssessment,
    AssessedDocument,
    CitationAssessment,
    run_assessment_async,
)
from mellea_lrc.core.citations import FullCaseCitation, UnknownCitation, citation_kind
from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import run_extraction
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing.docling import build_docling_converter, is_docling_supported_format
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
)
from mellea_lrc.validation import run_validation
from mellea_lrc.validation.types import (
    CitationValidation,
    ValidatedDocument,
    ValidationMetadata,
    ValidationStatus,
)
from mellea_lrc.serialization import (
    deserialize_citation_validation,
    serialize_citation_assessment,
    serialize_citation_validation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.validation import CourtListenerAccessClient

JsonDict = dict[str, Any]


class DoclingDocument(Protocol):
    """Protocol for the Docling document export surface used here."""

    def export_to_text(self) -> str:
        """Export extracted document content as plain text."""


class DoclingResult(Protocol):
    """Protocol for a Docling conversion result."""

    document: DoclingDocument


class DoclingConverter(Protocol):
    """Protocol for the Docling converter surface used here."""

    def convert(self, source: object) -> DoclingResult:
        """Convert a document source."""


class E2EBackend:
    """Well-defined API for text/PDF extraction, validation, and assessment."""

    def __init__(
        self,
        *,
        converter_factory: Callable[[], DoclingConverter] | None = None,
    ) -> None:
        self._converter_factory = converter_factory or _build_converter
        self._converter: DoclingConverter | None = None

    def review_text(self, text: str, *, validate: bool = True) -> dict[str, Any]:
        """Run the frontend review API for plain text input."""
        return review_preprocessed(_text_to_preprocessed(text), validate=validate)

    def review_preprocessed_document(self, preprocessed: PreprocessedDocument) -> dict[str, Any]:
        """Serialize a preprocessed document snapshot for the frontend review UI."""
        return review_preprocessed_document(preprocessed)

    def review_document_extraction(self, extraction: ExtractedDocument) -> dict[str, Any]:
        """Serialize an extraction snapshot for the frontend review UI."""
        return review_document_extraction(extraction)

    def review_document_validation(self, validation: ValidatedDocument) -> dict[str, Any]:
        """Serialize a validation snapshot for the frontend review UI."""
        return review_document_validation(validation)

    def extract_text(self, text: str) -> dict[str, Any]:
        """Extract frontend review citations from plain text without validation."""
        return review_preprocessed(_text_to_preprocessed(text), validate=False)

    def review_pdf_bytes(
        self,
        content: bytes,
        filename: str,
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Run the frontend review API for PDF bytes."""
        if content[:4] != b"%PDF":
            msg = f"{filename} is not a PDF"
            raise ValueError(msg)
        return review_preprocessed(
            _pdf_to_preprocessed(self._get_converter(), content, filename),
            validate=validate,
        )

    def review_document_bytes(
        self,
        content: bytes,
        filename: str,
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Run the frontend review API for an uploaded document."""
        if Path(filename).suffix.lower() == ".txt":
            return review_preprocessed(
                _text_file_to_preprocessed(content, filename),
                validate=validate,
            )
        return review_preprocessed(
            _document_to_preprocessed(self._get_converter(), content, filename),
            validate=validate,
        )

    def extract_document_bytes(
        self,
        content: bytes,
        filename: str,
    ) -> dict[str, Any]:
        """Extract frontend review citations from an uploaded document without validation."""
        if Path(filename).suffix.lower() == ".txt":
            return review_preprocessed(_text_file_to_preprocessed(content, filename), validate=False)
        return review_preprocessed(
            _document_to_preprocessed(self._get_converter(), content, filename),
            validate=False,
        )

    def validate_review_payload(
        self,
        payload: dict[str, object],
        *,
        client: CourtListenerAccessClient | None = None,
    ) -> dict[str, Any]:
        """Attach CourtListener validation to an existing frontend review payload."""
        return validate_review_payload(payload, client=client)

    def validate_review_citation_payload(
        self,
        payload: dict[str, object],
        *,
        client: CourtListenerAccessClient | None = None,
    ) -> dict[str, Any]:
        """Validate one frontend review citation."""
        return validate_review_citation_payload(payload, client=client)

    async def assess_review_payload(self, payload: dict[str, object]) -> dict[str, Any]:
        """Attach Mellea-assisted assessment to an existing validated review payload."""
        return await assess_review_payload(payload)

    def review_document_assessment(self, assessment: AssessedDocument) -> dict[str, Any]:
        """Serialize a cached assessment artifact for the frontend review UI."""
        return review_document_assessment(assessment)

    def _get_converter(self) -> DoclingConverter:
        if self._converter is None:
            self._converter = self._converter_factory()
        return self._converter


def review_preprocessed(
    preprocessed: PreprocessedDocument,
    *,
    validate: bool = True,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Run extraction, optional validation, and frontend review serialization."""
    extraction = run_extraction(preprocessed)
    validation = _run_validation(extraction, validate=validate, client=client)

    return {
        "document": {
            "text": extraction.text,
            "source_path": extraction.source_path,
            "source_format": extraction.source_metadata.format.value,
            "backend": extraction.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(extraction, validation),
        "validation": _validation_payload(validation),
        "stats": _stats(extraction, validation),
    }


def review_preprocessed_document(preprocessed: PreprocessedDocument) -> dict[str, Any]:
    """Serialize a preprocessed document snapshot without running extraction."""
    return {
        "document": {
            "text": preprocessed.text,
            "source_path": preprocessed.source_metadata.path,
            "source_format": preprocessed.source_metadata.format.value,
            "backend": preprocessed.preprocessing_metadata.backend.value,
        },
        "citations": [],
        "validation": None,
        "assessment": None,
        "stats": {
            "chars": len(preprocessed.text),
            "citation_spans": 0,
            "full_citations": 0,
        },
    }


def review_document_extraction(extraction: ExtractedDocument) -> dict[str, Any]:
    """Serialize a typed extraction artifact as a frontend review payload."""
    return {
        "document": {
            "text": extraction.text,
            "source_path": extraction.source_path,
            "source_format": extraction.source_metadata.format.value,
            "backend": extraction.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(extraction, None),
        "validation": None,
        "assessment": None,
        "stats": _stats(extraction, None),
    }


def review_document_validation(validation: ValidatedDocument) -> dict[str, Any]:
    """Serialize a typed validation artifact as a frontend review payload."""
    extraction = validation
    return {
        "document": {
            "text": extraction.text,
            "source_path": extraction.source_path,
            "source_format": extraction.source_metadata.format.value,
            "backend": extraction.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(extraction, validation),
        "validation": _validation_payload(validation),
        "assessment": None,
        "stats": _stats(extraction, validation),
    }


def validate_review_payload(
    payload: dict[str, object],
    *,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Attach validation to an existing frontend review payload without re-extracting."""
    extraction = _extraction_from_review_payload(payload)
    validation = _run_validation(extraction, validate=True, client=client)
    output = _copy_review_payload(payload)
    validation_by_id = {item.citation_id: _validation_item_payload(item) for item in validation.validations}
    citations = _review_citations(output)
    for citation in citations:
        citation["validation"] = validation_by_id.get(str(citation.get("id")))
    output["citations"] = citations
    output["validation"] = _validation_payload(validation)
    output["stats"] = _merge_stats(output.get("stats"), _stats(extraction, validation))
    return output


def validate_review_citation_payload(
    payload: dict[str, object],
    *,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Validate a single frontend review citation payload."""
    citation = payload.get("citation")
    if not isinstance(citation, dict):
        msg = "citation is required"
        raise TypeError(msg)

    extracted_citation = _extracted_citation_from_review_item(citation)
    validation_text = extracted_citation.matched_text or "citation"
    validation_citation = ExtractedCitation(
        citation_id=extracted_citation.citation_id,
        span=Span(start=0, end=len(validation_text)),
        matched_text=extracted_citation.matched_text,
        citation=extracted_citation.citation,
        resolves_to=None,
    )
    extraction = ExtractedDocument(
        source_metadata=SourceMetadata(),
        text=validation_text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
        citations=(validation_citation,),
        extraction_metadata=ExtractionMetadata(),
    )
    validation = _run_validation(extraction, validate=True, client=client)
    if validation is None or not validation.validations:
        msg = "citation validation did not produce a result"
        raise ValueError(msg)
    return _validation_item_payload(validation.validations[0])


async def assess_review_payload(payload: dict[str, object]) -> dict[str, Any]:
    """Attach Mellea-assisted case-name assessment to an existing review payload."""
    output = _copy_review_payload(payload)
    citations = _review_citations(output)
    extraction = _extraction_from_review_payload(output)
    validation = _validation_from_review_payload(output, extraction)
    document_assessment = await run_assessment_async(validation)
    assessment_by_id = {item.citation_id: item for item in document_assessment.assessments}
    for citation in citations:
        citation["assessment"] = _assessment_payload(assessment_by_id[str(citation.get("id"))])

    output["citations"] = citations
    output["assessment"] = _assessment_payload_document(document_assessment)
    output["stats"] = _merge_stats(
        output.get("stats"),
        {
            "assessed": _assessment_status_counts(document_assessment.assessments).get("assessed", 0),
            "case_name_counts": _assessment_case_name_counts(document_assessment.assessments),
            "court_counts": _assessment_court_counts(document_assessment.assessments),
            "year_counts": _assessment_year_counts(document_assessment.assessments),
        },
    )
    return output


def review_document_assessment(assessment: AssessedDocument) -> dict[str, Any]:
    """Serialize a typed assessment artifact as a frontend review payload."""
    output = review_document_validation(assessment)
    output["assessment"] = _assessment_payload_document(assessment)
    output["stats"] = _merge_stats(
        output.get("stats"),
        {
            "assessed": _assessment_status_counts(assessment.assessments).get("assessed", 0),
            "case_name_counts": _assessment_case_name_counts(assessment.assessments),
            "court_counts": _assessment_court_counts(assessment.assessments),
            "year_counts": _assessment_year_counts(assessment.assessments),
        },
    )
    assessment_by_id = {item.citation_id: _assessment_payload(item) for item in assessment.assessments}
    citations = _review_citations(output)
    for citation in citations:
        citation["assessment"] = assessment_by_id.get(str(citation.get("id")))
    output["citations"] = citations
    return output


def _build_converter() -> DoclingConverter:
    return build_docling_converter(enable_pdf_ocr=True)  # type: ignore[return-value]


def _pdf_to_preprocessed(
    converter: DoclingConverter,
    content: bytes,
    filename: str,
) -> PreprocessedDocument:
    return _docling_to_preprocessed(converter, content, filename)


def _document_to_preprocessed(
    converter: DoclingConverter,
    content: bytes,
    filename: str,
) -> PreprocessedDocument:
    path = Path(filename)
    if not is_docling_supported_format(path):
        msg = f"Unsupported document format: {path.suffix or '<none>'}"
        raise ValueError(msg)
    return _docling_to_preprocessed(converter, content, filename)


def _docling_to_preprocessed(
    converter: DoclingConverter,
    content: bytes,
    filename: str,
) -> PreprocessedDocument:
    from docling.datamodel.base_models import DocumentStream  # noqa: PLC0415

    source = DocumentStream(name=filename, stream=BytesIO(content))
    result = converter.convert(source)
    return PreprocessedDocument(
        source_metadata=SourceMetadata(
            path=filename,
            format=_source_format(filename),
        ),
        text=result.document.export_to_text(),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.DOCLING,
        ),
    )


def _text_to_preprocessed(text: str) -> PreprocessedDocument:
    from mellea_lrc.preprocessing import preprocess_plain_text_from_string  # noqa: PLC0415

    return preprocess_plain_text_from_string(text)


def _text_file_to_preprocessed(content: bytes, filename: str) -> PreprocessedDocument:
    return PreprocessedDocument(
        source_metadata=SourceMetadata(path=filename, format=SourceFormat.TEXT),
        text=content.decode("utf-8"),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
    )


def _source_format(filename: str) -> SourceFormat:
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf": SourceFormat.PDF,
        ".docx": SourceFormat.DOCX,
        ".pptx": SourceFormat.PPTX,
        ".xlsx": SourceFormat.XLSX,
        ".html": SourceFormat.HTML,
        ".htm": SourceFormat.HTML,
        ".md": SourceFormat.MARKDOWN,
    }.get(suffix, SourceFormat.UNKNOWN)


def _run_validation(
    extraction: ExtractedDocument,
    *,
    validate: bool,
    client: CourtListenerAccessClient | None,
) -> ValidatedDocument | None:
    if not validate:
        return None
    return run_validation(
        extraction,
        client_mode="custom" if client is not None else "deployed",
        client=client,
    )


def _citation_payloads(
    extraction: ExtractedDocument,
    validation: ValidatedDocument | None,
) -> list[dict[str, Any]]:
    validation_by_id = {item.citation_id: item for item in validation.validations} if validation else {}
    return [
        {
            "id": item.citation_id,
            "start": item.span.start,
            "end": item.span.end,
            "matched_text": item.matched_text,
            "kind": citation_kind(item.citation).value,
            "fields": _citation_fields(item.citation),
            "resolves_to": item.resolves_to,
            "validation": _single_validation_payload(validation_by_id.get(item.citation_id)),
        }
        for item in extraction.citations
    ]


def _citation_fields(citation: object) -> dict[str, object]:
    return {key: value for key, value in asdict(citation).items() if _has_citation_field_value(value)}


def _has_citation_field_value(value: object) -> bool:
    return value is not None and value not in ("", ())


def _single_validation_payload(item: CitationValidation | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return _validation_item_payload(item)


def _validation_payload(validation: ValidatedDocument | None) -> dict[str, Any] | None:
    if validation is None:
        return None
    return {
        "validations": [_validation_item_payload(item) for item in validation.validations],
        "counts": {
            "total": len(validation.validations),
            "found": len(validation.found),
        },
    }


def _validation_from_review_payload(
    payload: dict[str, object],
    extraction: ExtractedDocument,
) -> ValidatedDocument:
    validation_by_id = {
        item.citation_id: item
        for item in (
            _citation_validation_from_review_item(citation) for citation in _review_citations(payload)
        )
        if item is not None
    }
    return ValidatedDocument(
        source_metadata=extraction.source_metadata,
        text=extraction.text,
        preprocessing_metadata=extraction.preprocessing_metadata,
        citations=extraction.citations,
        extraction_metadata=extraction.extraction_metadata,
        validations=tuple(
            validation_by_id[item.citation_id]
            for item in extraction.citations
            if item.citation_id in validation_by_id
        ),
        validation_metadata=ValidationMetadata(client_mode="custom", source="review_payload"),
    )


def _validation_item_payload(item: CitationValidation) -> dict[str, Any]:
    payload = dict(serialize_citation_validation(item))
    payload["case_names"] = list(item.case_names)
    return payload


def _citation_validation_from_review_item(item: JsonDict) -> CitationValidation | None:
    validation = item.get("validation")
    if not isinstance(validation, dict):
        return None
    validation_payload = dict(validation)
    validation_payload.pop("case_names", None)
    validation_payload["citation_id"] = str(validation.get("citation_id") or item.get("id") or "")
    return deserialize_citation_validation(validation_payload)


def _extraction_from_review_payload(payload: dict[str, object]) -> ExtractedDocument:
    text = _review_document_text(payload)
    citations = tuple(_extracted_citation_from_review_item(item) for item in _review_citations(payload))
    return ExtractedDocument(
        source_metadata=SourceMetadata(
            path=_review_source_path(payload),
            format=_review_source_format(payload),
        ),
        text=text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=_review_preprocessing_backend(payload),
        ),
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
    )


def _extracted_citation_from_review_item(item: JsonDict) -> ExtractedCitation:
    fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
    citation = (
        FullCaseCitation(
            plaintiff=_optional_str(fields.get("plaintiff")),
            defendant=_optional_str(fields.get("defendant")),
            volume=_optional_str(fields.get("volume")),
            reporter=_optional_str(fields.get("reporter")),
            page=_optional_str(fields.get("page")),
            pin_cite=_optional_str(fields.get("pin_cite")),
            extra=_optional_str(fields.get("extra")),
            year=_optional_str(fields.get("year")),
            court=_optional_str(fields.get("court")),
            parenthetical=_optional_str(fields.get("parenthetical")),
        )
        if item.get("kind") == "FullCaseCitation"
        else UnknownCitation()
    )
    return ExtractedCitation(
        citation_id=str(item.get("id") or ""),
        span=Span(start=_int_field(item.get("start")), end=_int_field(item.get("end"))),
        matched_text=str(item.get("matched_text") or ""),
        citation=citation,
        resolves_to=_optional_str(item.get("resolves_to")),
    )


def _assessment_payload(item: CitationAssessment) -> JsonDict:
    return dict(serialize_citation_assessment(item))


def _assessment_payload_document(assessment: AssessedDocument) -> JsonDict:
    return {
        "assessments": [_assessment_payload(item) for item in assessment.assessments],
        "assessment_complete": assessment.assessment_complete,
        "status_counts": _assessment_status_counts(assessment.assessments),
        "case_name_followup_status_counts": _case_name_followup_status_counts(assessment.assessments),
        "case_name_counts": _assessment_case_name_counts(assessment.assessments),
        "court_counts": _assessment_court_counts(assessment.assessments),
        "year_counts": _assessment_year_counts(assessment.assessments),
    }


def _assessment_case_name_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.case_name.initial.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _assessment_court_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.court.final.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _assessment_year_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.year.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _assessment_status_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    return counts


def _case_name_followup_status_counts(
    assessments: tuple[CitationAssessment, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.case_name.followup.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _copy_review_payload(payload: dict[str, object]) -> JsonDict:
    return {
        "document": dict(payload.get("document")) if isinstance(payload.get("document"), dict) else {},
        "citations": _review_citations(payload),
        "validation": payload.get("validation"),
        "assessment": payload.get("assessment"),
        "stats": dict(payload.get("stats")) if isinstance(payload.get("stats"), dict) else {},
    }


def _review_citations(payload: dict[str, object]) -> list[JsonDict]:
    citations = payload.get("citations")
    if not isinstance(citations, list):
        return []
    return [dict(item) for item in citations if isinstance(item, dict)]


def _review_document_text(payload: dict[str, object]) -> str:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    text = document.get("text")
    if not isinstance(text, str) or not text:
        msg = "review payload is missing document.text"
        raise ValueError(msg)
    return text


def _review_source_path(payload: dict[str, object]) -> str | None:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    return _optional_str(document.get("source_path"))


def _review_source_format(payload: dict[str, object]) -> SourceFormat:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    try:
        return SourceFormat(str(document.get("source_format") or SourceFormat.UNKNOWN.value))
    except ValueError:
        return SourceFormat.UNKNOWN


def _review_preprocessing_backend(payload: dict[str, object]) -> PreprocessingBackend:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    try:
        return PreprocessingBackend(str(document.get("backend") or PreprocessingBackend.PLAIN_TEXT.value))
    except ValueError:
        return PreprocessingBackend.PLAIN_TEXT


def _merge_stats(existing: object, updates: dict[str, int]) -> dict[str, int]:
    stats = dict(existing) if isinstance(existing, dict) else {}
    for key, value in updates.items():
        stats[key] = value
    return stats


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


def _list_field(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _int_field(value: object) -> int:
    if isinstance(value, int):
        return value
    return int(str(value))


def _stats(
    extraction: ExtractedDocument,
    validation: ValidatedDocument | None,
) -> dict[str, int]:
    label_count = len(extraction.citations)
    stats = {
        "chars": len(extraction.text),
        "citation_spans": label_count,
        "full_citations": len(extraction.full_citations),
    }
    if validation is not None:
        stats["validated"] = sum(
            1 for item in validation.validations if item.status != ValidationStatus.SKIPPED
        )
        stats["found"] = len(validation.found)
    return stats
