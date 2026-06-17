"""Assembled extraction and validation API for the Modal E2E backend."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mellea_lrc.core.citations import citation_kind
from mellea_lrc.extraction import run_extraction
from mellea_lrc.preprocessing.docling import is_docling_supported_format
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)
from mellea_lrc.validation import validate_extraction
from mellea_lrc.validation.types import ValidationStatus
from scripts.label_studio.label_studio import to_label_studio_prediction

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.extraction.types import DocumentExtraction
    from mellea_lrc.validation import CitationValidation, CourtListenerAccessClient, DocumentValidation


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
    return validate_extraction(
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


def _validation_item_payload(item: CitationValidation) -> dict[str, Any]:
    payload = asdict(item)
    payload["status"] = item.status.value
    payload["case_names"] = list(item.case_names)
    payload["clusters"] = list(item.clusters)
    return payload


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
