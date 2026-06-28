"""Preprocessing layer public API."""

from mellea_lrc.preprocessing.docling import (
    build_docling_converter,
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
    DocumentBase,
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)

__all__ = [
    "DocumentBase",
    "PreprocessedDocument",
    "PreprocessedDocumentMetadata",
    "PreprocessingBackend",
    "SourceFormat",
    "build_docling_converter",
    "is_docling_supported_format",
    "preprocess_plain_text",
    "preprocess_plain_text_from_string",
    "preprocess_with_docling",
    "run_preprocessing",
    "run_preprocessing_directory",
    "split_plain_text_file",
]
