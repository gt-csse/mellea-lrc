"""Preprocessing layer public API."""

from mellea_lrc.preprocessing.pipeline import preprocess, preprocess_directory
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)
from mellea_lrc.preprocessing.docling import preprocess_with_docling
from mellea_lrc.preprocessing.plain_text import (
    preprocess_plain_text,
    preprocess_plain_text_from_string,
    split_plain_text_file,
)

__all__ = [
    "PreprocessedDocument",
    "PreprocessedDocumentMetadata",
    "PreprocessingBackend",
    "SourceFormat",
    "preprocess",
    "preprocess_directory",
    "preprocess_plain_text",
    "preprocess_plain_text_from_string",
    "preprocess_with_docling",
    "split_plain_text_file",
]
