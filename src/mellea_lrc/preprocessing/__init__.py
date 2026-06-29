"""Preprocessing layer public API."""

from mellea_lrc.core.documents import DocumentBase, SourceFormat, SourceMetadata
from mellea_lrc.preprocessing.docling import is_docling_supported_format, preprocess_with_docling
from mellea_lrc.preprocessing.pipeline import preprocess, preprocess_directory
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
    "is_docling_supported_format",
    "preprocess",
    "preprocess_directory",
    "preprocess_plain_text",
    "preprocess_plain_text_from_string",
    "preprocess_with_docling",
    "split_plain_text_file",
]
