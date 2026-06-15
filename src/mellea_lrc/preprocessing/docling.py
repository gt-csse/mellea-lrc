"""Docling-backed preprocessing from raw Layer 3 documents."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mellea_lrc.preprocessing.document import (
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
        from docling.document_converter import DocumentConverter  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "Docling is required for raw document preprocessing. "
            "Install with: uv sync --group preprocessing"
        )
        raise ImportError(msg) from exc

    source_path = Path(path)
    converter = DocumentConverter()
    result = converter.convert(str(source_path))
    text = result.document.export_to_markdown(strict_text=True)

    return PreprocessedDocument(
        text=text,
        metadata=PreprocessedDocumentMetadata(
            source_path=str(source_path),
            source_format=_source_format(source_path),
            backend=PreprocessingBackend.DOCLING,
            backend_version=_docling_version(),
        ),
    )
