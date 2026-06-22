"""Assembled extraction and validation API for the Modal E2E backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mellea_lrc.assessment import (
    CaseNameAssessmentStatus,
    CitationAssessment,
    DocumentAssessment,
    ModifiedExtractedCitation,
    assess_case_name_exact_match,
    assess_year_exact_match,
    build_extracted_case_name,
    find_text_span_near_full_span,
    get_extended_span_text,
)
from mellea_lrc.core.citations import FullCaseCitation, UnknownCitation, citation_kind
from mellea_lrc.core.spans import Span
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.extraction import run_extraction
from mellea_lrc.extraction.types import DocumentExtraction, ExtractedCitation
from mellea_lrc.preprocessing.docling import is_docling_supported_format
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)
from mellea_lrc.validation import run_validation
from mellea_lrc.validation.types import CitationValidation, DocumentValidation, ValidationStatus
from scripts.label_studio.label_studio import to_label_studio_prediction

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.assessment.types import (
        CaseNameAssessment,
        ModifiedExtractedCitationProposal,
        YearAssessment,
    )
    from mellea_lrc.validation import CourtListenerAccessClient

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReviewCitationAssessmentResult:
    """Assessment plus optional modified extraction history for one citation."""

    assessment: CitationAssessment | None
    modified_citation: ModifiedExtractedCitation | None = None
    reassessment: CitationAssessment | None = None


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
    """Well-defined API for text/PDF extraction, validation, and prediction."""

    def __init__(
        self,
        *,
        converter_factory: Callable[[], DoclingConverter] | None = None,
    ) -> None:
        self._converter_factory = converter_factory or _build_converter
        self._converter: DoclingConverter | None = None

    def predict_text(self, text: str, *, validate: bool = True) -> dict[str, Any]:
        """Run the assembled pipeline for plain text input."""
        return predict_preprocessed(_text_to_preprocessed(text), validate=validate)

    def review_text(self, text: str, *, validate: bool = True) -> dict[str, Any]:
        """Run the frontend review API for plain text input."""
        return review_preprocessed(_text_to_preprocessed(text), validate=validate)

    def review_preprocessed_document(self, preprocessed: PreprocessedDocument) -> dict[str, Any]:
        """Serialize a preprocessed document snapshot for the frontend review UI."""
        return review_preprocessed_document(preprocessed)

    def review_document_extraction(self, extraction: DocumentExtraction) -> dict[str, Any]:
        """Serialize an extraction snapshot for the frontend review UI."""
        return review_document_extraction(extraction)

    def review_document_validation(self, validation: DocumentValidation) -> dict[str, Any]:
        """Serialize a validation snapshot for the frontend review UI."""
        return review_document_validation(validation)

    def extract_text(self, text: str) -> dict[str, Any]:
        """Extract frontend review citations from plain text without validation."""
        return review_preprocessed(_text_to_preprocessed(text), validate=False)

    def predict_pdf_bytes(
        self,
        content: bytes,
        filename: str,
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Run the assembled pipeline for PDF bytes."""
        if content[:4] != b"%PDF":
            msg = f"{filename} is not a PDF"
            raise ValueError(msg)
        return predict_preprocessed(
            _pdf_to_preprocessed(self._get_converter(), content, filename),
            validate=validate,
        )

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

    def assess_review_payload(self, payload: dict[str, object]) -> dict[str, Any]:
        """Attach Mellea-assisted assessment to an existing validated review payload."""
        return assess_review_payload(payload)

    def review_document_assessment(self, assessment: DocumentAssessment) -> dict[str, Any]:
        """Serialize a cached assessment artifact for the frontend review UI."""
        return review_document_assessment(assessment)

    def _get_converter(self) -> DoclingConverter:
        if self._converter is None:
            self._converter = self._converter_factory()
        return self._converter


def predict_preprocessed(
    preprocessed: PreprocessedDocument,
    *,
    validate: bool = True,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Run extraction, optional validation, and Label Studio prediction serialization."""
    extraction = run_extraction(preprocessed)
    validation = _run_validation(extraction, validate=validate, client=client)
    prediction = to_label_studio_prediction(extraction)
    if validation is not None:
        prediction = add_validation_notes(prediction, validation)

    return {
        "text": extraction.text,
        "prediction": prediction,
        "validation": _validation_payload(validation),
        "stats": _stats(extraction, validation),
    }


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
            "source_format": extraction.preprocessed.metadata.source_format.value,
            "backend": extraction.preprocessed.metadata.backend.value,
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
            "source_path": preprocessed.metadata.source_path,
            "source_format": preprocessed.metadata.source_format.value,
            "backend": preprocessed.metadata.backend.value,
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


def review_document_extraction(extraction: DocumentExtraction) -> dict[str, Any]:
    """Serialize a typed extraction artifact as a frontend review payload."""
    return {
        "document": {
            "text": extraction.text,
            "source_path": extraction.source_path,
            "source_format": extraction.preprocessed.metadata.source_format.value,
            "backend": extraction.preprocessed.metadata.backend.value,
        },
        "citations": _citation_payloads(extraction, None),
        "validation": None,
        "assessment": None,
        "stats": _stats(extraction, None),
    }


def review_document_validation(validation: DocumentValidation) -> dict[str, Any]:
    """Serialize a typed validation artifact as a frontend review payload."""
    extraction = DocumentExtraction(
        preprocessed=validation.preprocessed,
        citations=validation.citations,
    )
    return {
        "document": {
            "text": extraction.text,
            "source_path": extraction.source_path,
            "source_format": extraction.preprocessed.metadata.source_format.value,
            "backend": extraction.preprocessed.metadata.backend.value,
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
    extraction = DocumentExtraction(
        preprocessed=PreprocessedDocument(
            text=extracted_citation.matched_text or "citation",
            metadata=PreprocessedDocumentMetadata(
                source_path=None,
                source_format=SourceFormat.UNKNOWN,
                backend=PreprocessingBackend.PLAIN_TEXT,
            ),
        ),
        citations=(extracted_citation,),
    )
    validation = _run_validation(extraction, validate=True, client=client)
    if validation is None or not validation.validations:
        msg = "citation validation did not produce a result"
        raise ValueError(msg)
    return _validation_item_payload(validation.validations[0])


def assess_review_payload(payload: dict[str, object]) -> dict[str, Any]:
    """Attach Mellea-assisted case-name assessment to an existing review payload."""
    output = _copy_review_payload(payload)
    document_text = _review_document_text(output)
    citations = _review_citations(output)
    extraction = _extraction_from_review_payload(output)
    validation = _validation_from_review_payload(output, extraction)
    session = None
    assessments: list[CitationAssessment] = []
    modified_citations = []
    reassessments: list[CitationAssessment] = []

    for citation in citations:
        result = _assess_review_citation(citation, document_text, session)
        if result.assessment is None:
            citation["assessment"] = None
            continue
        assessment = result.assessment
        if (
            assessment.case_assess is not None
            and assessment.case_assess.status == CaseNameAssessmentStatus.NEEDS_ASSESSMENT
        ):
            session = session or start_mellea_session_from_env()
            result = _assess_review_citation(citation, document_text, session)
            if result.assessment is None:
                citation["assessment"] = None
                continue
            assessment = result.assessment
        citation["assessment"] = _assessment_payload(assessment)
        assessments.append(assessment)
        if result.modified_citation is not None:
            modified_citations.append(result.modified_citation)
        if result.reassessment is not None:
            reassessments.append(result.reassessment)

    document_assessment = DocumentAssessment(
        preprocessed=extraction.preprocessed,
        citations=extraction.citations,
        validations=validation.validations,
        assessments=tuple(assessments),
        modified_citations=tuple(modified_citations),
        reassessments=tuple(reassessments),
    )
    output["citations"] = citations
    output["assessment"] = _assessment_payload_document(document_assessment)
    output["stats"] = _merge_stats(
        output.get("stats"),
        {"assessed": len(assessments), **_assessment_counts(assessments)},
    )
    return output


def review_document_assessment(assessment: DocumentAssessment) -> dict[str, Any]:
    """Serialize a typed assessment artifact as a frontend review payload."""
    _ensure_assessment_is_resolved(assessment)
    validation = DocumentValidation(
        preprocessed=assessment.preprocessed,
        citations=assessment.citations,
        validations=assessment.validations,
    )
    output = review_document_validation(validation)
    output["assessment"] = _assessment_payload_document(assessment)
    output["stats"] = _merge_stats(
        output.get("stats"),
        {"assessed": len(assessment.assessments), **_assessment_counts(list(assessment.assessments))},
    )
    assessment_by_id = {item.citation_id: _assessment_payload(item) for item in assessment.assessments}
    citations = _review_citations(output)
    for citation in citations:
        citation["assessment"] = assessment_by_id.get(str(citation.get("id")))
    output["citations"] = citations
    return output


def _ensure_assessment_is_resolved(assessment: DocumentAssessment) -> None:
    # NEEDS_ASSESSMENT is a transient handoff only when both names are present and
    # comparable; a missing extracted or CourtListener name is legitimately
    # unassessable and is allowed to remain NEEDS_ASSESSMENT.
    pending = [
        item.citation_id
        for item in assessment.assessments
        if item.case_assess is not None
        and item.case_assess.status == CaseNameAssessmentStatus.NEEDS_ASSESSMENT
        and item.case_assess.extracted_case_name
        and item.case_assess.courtlistener_case_name
    ]
    if pending:
        msg = (
            "DocumentAssessment contains unresolved case-name assessment handoff states. "
            f"Rerun assessment without a debug cap. Pending citation ids: {', '.join(pending[:5])}"
        )
        raise ValueError(msg)


def add_validation_notes(
    prediction: dict[str, Any],
    validation: DocumentValidation,
) -> dict[str, Any]:
    """Add CourtListener validation messages as Label Studio per-region notes."""
    result = list(prediction.get("result", []))
    for item in validation.validations:
        if item.status == ValidationStatus.SKIPPED:
            continue
        result.append(
            {
                "id": item.citation_id,
                "from_name": "notes",
                "to_name": "text",
                "type": "textarea",
                "value": {"text": [item.message]},
            }
        )
    return {**prediction, "result": result}


def _build_converter() -> DoclingConverter:
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


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
        text=result.document.export_to_text(),
        metadata=PreprocessedDocumentMetadata(
            source_path=filename,
            source_format=_source_format(filename),
            backend=PreprocessingBackend.DOCLING,
        ),
    )


def _text_to_preprocessed(text: str) -> PreprocessedDocument:
    from mellea_lrc.preprocessing import preprocess_plain_text_from_string  # noqa: PLC0415

    return preprocess_plain_text_from_string(text)


def _text_file_to_preprocessed(content: bytes, filename: str) -> PreprocessedDocument:
    return PreprocessedDocument(
        text=content.decode("utf-8"),
        metadata=PreprocessedDocumentMetadata(
            source_path=filename,
            source_format=SourceFormat.TEXT,
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
    extraction: DocumentExtraction,
    *,
    validate: bool,
    client: CourtListenerAccessClient | None,
) -> DocumentValidation | None:
    if not validate:
        return None
    return run_validation(
        extraction,
        client_mode="custom" if client is not None else "deployed",
        client=client,
    )


def _citation_payloads(
    extraction: DocumentExtraction,
    validation: DocumentValidation | None,
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


def _validation_payload(validation: DocumentValidation | None) -> dict[str, Any] | None:
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
    extraction: DocumentExtraction,
) -> DocumentValidation:
    validation_by_id = {
        item.citation_id: item
        for item in (
            _citation_validation_from_review_item(citation) for citation in _review_citations(payload)
        )
        if item is not None
    }
    return DocumentValidation(
        preprocessed=extraction.preprocessed,
        citations=extraction.citations,
        validations=tuple(
            validation_by_id[item.citation_id]
            for item in extraction.citations
            if item.citation_id in validation_by_id
        ),
    )


def _validation_item_payload(item: CitationValidation) -> dict[str, Any]:
    payload = asdict(item)
    payload["status"] = item.status.value
    payload["case_names"] = list(item.case_names)
    payload["clusters"] = list(item.clusters)
    return payload


def _citation_validation_from_review_item(item: JsonDict) -> CitationValidation | None:
    validation = item.get("validation")
    if not isinstance(validation, dict):
        return None
    try:
        status = ValidationStatus(str(validation.get("status")))
    except ValueError:
        status = ValidationStatus.LOOKUP_FAILED
    return CitationValidation(
        citation_id=str(validation.get("citation_id") or item.get("id") or ""),
        locator=_optional_str(validation.get("locator")),
        status=status,
        source=str(validation.get("source") or ""),
        message=str(validation.get("message") or ""),
        case_names=tuple(str(value) for value in _list_field(validation.get("case_names"))),
        lookup_status=_optional_int(validation.get("lookup_status")),
        lookup_cache=_optional_str(validation.get("lookup_cache")),
        lookup_key=_optional_str(validation.get("lookup_key")),
        error_message=_optional_str(validation.get("error_message")),
        limit_detail=_dict_field(validation.get("limit_detail")),
        clusters=tuple(_dict_items(validation.get("clusters"))),
    )


def _extraction_from_review_payload(payload: dict[str, object]) -> DocumentExtraction:
    text = _review_document_text(payload)
    citations = tuple(_extracted_citation_from_review_item(item) for item in _review_citations(payload))
    return DocumentExtraction(
        preprocessed=PreprocessedDocument(
            text=text,
            metadata=PreprocessedDocumentMetadata(
                source_path=_review_source_path(payload),
                source_format=_review_source_format(payload),
                backend=_review_preprocessing_backend(payload),
            ),
        ),
        citations=citations,
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


def _assess_review_citation(
    citation: JsonDict,
    document_text: str,
    session: object | None,
) -> ReviewCitationAssessmentResult:
    validation = citation.get("validation") if isinstance(citation.get("validation"), dict) else None
    if (
        citation.get("kind") != "FullCaseCitation"
        or validation is None
        or validation.get("status") != "found"
    ):
        return ReviewCitationAssessmentResult(assessment=None)

    citation_id = str(citation.get("id") or "")
    fields = citation.get("fields") if isinstance(citation.get("fields"), dict) else {}
    year_assess = _assess_review_year(citation_id, fields, validation)
    extracted_case_name = build_extracted_case_name(
        FullCaseCitation(
            plaintiff=_optional_str(fields.get("plaintiff")),
            defendant=_optional_str(fields.get("defendant")),
        )
    )
    courtlistener_case_name = _first_cluster_case_name(validation)
    exact = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact.status != CaseNameAssessmentStatus.NEEDS_ASSESSMENT or session is None:
        return ReviewCitationAssessmentResult(
            assessment=CitationAssessment(
                citation_id=citation_id,
                case_assess=exact,
                year_assess=year_assess,
            ),
        )

    from mellea_lrc.assessment.mellea import assess_case_name_with_mellea  # noqa: PLC0415

    span = Span(start=_int_field(citation.get("start")), end=_int_field(citation.get("end")))
    case_name_run = assess_case_name_with_mellea(
        session,
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        document_context=get_extended_span_text(document_text, span),
    )
    reassessment = (
        CitationAssessment(
            citation_id=citation_id,
            case_assess=case_name_run.reassessment,
            year_assess=year_assess,
        )
        if case_name_run.reassessment is not None
        else None
    )
    return ReviewCitationAssessmentResult(
        assessment=CitationAssessment(
            citation_id=citation_id,
            case_assess=case_name_run.assessment,
            year_assess=year_assess,
        ),
        modified_citation=_bind_modified_citation(
            case_name_run.modified_citation,
            document_text,
            span,
            citation_id,
        ),
        reassessment=reassessment,
    )


def _bind_modified_citation(
    modified_citation: ModifiedExtractedCitationProposal | None,
    document_text: str,
    full_span: Span,
    citation_id: str,
) -> ModifiedExtractedCitation | None:
    if modified_citation is None or not modified_citation.extracted_case_name:
        return None
    modified_span = find_text_span_near_full_span(
        document_text,
        modified_citation.extracted_case_name,
        full_span,
    )
    return ModifiedExtractedCitation.from_proposal(
        modified_citation,
        citation_id=citation_id,
        span=modified_span,
    )


def _assess_review_year(
    citation_id: str,
    fields: JsonDict,
    validation: JsonDict,
) -> YearAssessment:
    return assess_year_exact_match(
        citation_id=citation_id,
        extracted_year=_optional_str(fields.get("year")),
        courtlistener_year=_first_cluster_year(validation),
    )


def _first_cluster_case_name(validation: JsonDict) -> str | None:
    clusters = validation.get("clusters")
    if not isinstance(clusters, list) or not clusters or not isinstance(clusters[0], dict):
        return None
    case_name = clusters[0].get("case_name") or clusters[0].get("caseName")
    return _optional_str(case_name)


def _first_cluster_year(validation: JsonDict) -> str | None:
    clusters = validation.get("clusters")
    if not isinstance(clusters, list) or not clusters or not isinstance(clusters[0], dict):
        return None
    date_filed = _optional_str(clusters[0].get("date_filed") or clusters[0].get("dateFiled"))
    return date_filed[:4] if date_filed else None


def _assessment_payload(item: CitationAssessment) -> JsonDict:
    return {
        "citation_id": item.citation_id,
        "status": item.status.value,
        "message": item.message,
        "case_assess": _case_assessment_payload(item.case_assess),
        "year_assess": _year_assessment_payload(item.year_assess),
    }


def _case_assessment_payload(item: CaseNameAssessment | None) -> JsonDict | None:
    if item is None:
        return None
    payload = asdict(item)
    payload["status"] = item.status.value
    return payload


def _year_assessment_payload(item: YearAssessment | None) -> JsonDict | None:
    if item is None:
        return None
    payload = asdict(item)
    payload["status"] = item.status.value
    return payload


def _assessment_payload_document(assessment: DocumentAssessment) -> JsonDict:
    return {
        "assessments": [_assessment_payload(item) for item in assessment.assessments],
        "modified_citations": [_modified_citation_payload(item) for item in assessment.modified_citations],
        "reassessments": [_assessment_payload(item) for item in assessment.reassessments],
        "counts": _assessment_counts(list(assessment.assessments)),
    }


def _modified_citation_payload(item: ModifiedExtractedCitation) -> JsonDict:
    payload = asdict(item)
    payload["extracted_case_name"] = item.extracted_case_name
    return payload


def _assessment_counts(assessments: list[CitationAssessment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
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


def _dict_field(value: object) -> JsonDict | None:
    return dict(value) if isinstance(value, dict) else None


def _dict_items(value: object) -> list[JsonDict]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _int_field(value: object) -> int:
    if isinstance(value, int):
        return value
    return int(str(value))


def _stats(
    extraction: DocumentExtraction,
    validation: DocumentValidation | None,
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
