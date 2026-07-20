"""Preprocessing layer public API."""

from mellea_lrc.core.documents import DocumentBase, SourceFormat, SourceMetadata
from mellea_lrc.preprocessing.docling import (
    build_docling_converter,
    extract_pdf_pages_with_docling,
    is_docling_supported_format,
    preprocess_with_docling,
)
from mellea_lrc.preprocessing.pipeline import run_preprocessing, run_preprocessing_directory
from mellea_lrc.preprocessing.plain_text import (
    preprocess_plain_text,
    preprocess_plain_text_from_string,
    split_plain_text_file,
)
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
)

__all__ = [
    "DocumentBase",
    "PreprocessedDocument",
    "PreprocessingBackend",
    "PreprocessingMetadata",
    "SourceFormat",
    "SourceMetadata",
    "build_docling_converter",
    "extract_pdf_pages_with_docling",
    "is_docling_supported_format",
    "preprocess_plain_text",
    "preprocess_plain_text_from_string",
    "preprocess_with_docling",
    "run_preprocessing",
    "run_preprocessing_directory",
    "split_plain_text_file",
]
