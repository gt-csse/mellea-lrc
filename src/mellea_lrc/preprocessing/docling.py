"""Docling-backed preprocessing from raw Layer 3 documents."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)

_SOURCE_FORMAT_BY_SUFFIX = {
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
    ".pptx": SourceFormat.PPTX,
    ".xlsx": SourceFormat.XLSX,
    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,
    ".md": SourceFormat.MARKDOWN,
}


def is_docling_supported_format(path: Path | str) -> bool:
    """Return True when Docling supports the path suffix."""
    return Path(path).suffix.lower() in _SOURCE_FORMAT_BY_SUFFIX


def _docling_version() -> str | None:
    try:
        return version("docling")
    except PackageNotFoundError:
        return None


def _source_format(path: Path) -> SourceFormat:
    return _SOURCE_FORMAT_BY_SUFFIX.get(path.suffix.lower(), SourceFormat.UNKNOWN)


def preprocess_with_docling(path: Path | str) -> PreprocessedDocument:
    """Convert a raw document to plain text using Docling."""
    try:
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions  # noqa: PLC0415
        from docling.document_converter import DocumentConverter  # noqa: PLC0415
        from docling.document_converter import PdfFormatOption  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "Docling is required for raw document preprocessing. Install with: uv sync --group preprocessing"
        )
        raise ImportError(msg) from exc

    source_path = Path(path)
    converter = DocumentConverter()
    if source_path.suffix.lower() == ".pdf":
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = TesseractCliOcrOptions(lang=["eng"])
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
    result = converter.convert(str(source_path))
    text = result.document.export_to_text()

    return PreprocessedDocument(
        text=text,
        metadata=PreprocessedDocumentMetadata(
            source_path=str(source_path),
            source_format=_source_format(source_path),
            backend=PreprocessingBackend.DOCLING,
            backend_version=_docling_version(),
        ),
    )
