"""Docling-backed preprocessing from raw Layer 3 documents."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
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


def build_docling_converter(*, enable_pdf_ocr: bool = True) -> object:
    """Build a Docling ``DocumentConverter`` with project PDF defaults."""
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

    if not enable_pdf_ocr:
        return DocumentConverter()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(lang=["eng"])
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def preprocess_with_docling(path: Path | str, *, converter: object | None = None) -> PreprocessedDocument:
    """Convert a raw document to plain text using Docling."""
    source_path = Path(path)
    docling_converter = converter or build_docling_converter()
    result = docling_converter.convert(str(source_path))  # type: ignore[attr-defined]
    text = result.document.export_to_text()

    return PreprocessedDocument(
        source_metadata=SourceMetadata(
            path=str(source_path),
            format=_source_format(source_path),
        ),
        text=text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.DOCLING,
            backend_version=_docling_version(),
        ),
    )
