"""Preprocessing pipeline from Layer 3 raw documents to Layer 2 text."""

from pathlib import Path

from mellea_lrc.preprocessing.document import (
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


def preprocess(path: Path | str) -> PreprocessedDocument:
    """Preprocess a document using the backend appropriate for its format."""
    source_path = Path(path)
    if source_path.suffix.lower() == ".txt":
        return preprocess_plain_text(source_path)
    return preprocess_with_docling(source_path)


def preprocess_directory(directory: Path | str) -> list[PreprocessedDocument]:
    """Preprocess every file in a directory."""
    source_dir = Path(directory)
    paths = sorted(p for p in source_dir.iterdir() if p.is_file())
    return [preprocess(path) for path in paths]
