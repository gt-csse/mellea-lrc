"""Formal types for Layer 2 preprocessed documents."""

from dataclasses import dataclass, field
from enum import StrEnum


class PreprocessingBackend(StrEnum):
    """Engine that produced the preprocessed text."""

    DOCLING = "docling"
    PLAIN_TEXT = "plain_text"


class SourceFormat(StrEnum):
    """Original document format before preprocessing."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PreprocessedDocumentMetadata:
    """Provenance and pipeline metadata for a preprocessed document."""

    source_path: str | None = None
    source_format: SourceFormat = SourceFormat.UNKNOWN
    backend: PreprocessingBackend = PreprocessingBackend.PLAIN_TEXT
    backend_version: str | None = None
    header: str | None = None
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreprocessedDocument:
    """Layer 2 text output consumed by citation extraction."""

    text: str
    metadata: PreprocessedDocumentMetadata

    def __post_init__(self) -> None:
        if not self.text:
            msg = "PreprocessedDocument.text must not be empty"
            raise ValueError(msg)
