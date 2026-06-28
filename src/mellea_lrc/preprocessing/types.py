"""Formal types for Layer 2 preprocessed documents."""

from dataclasses import dataclass, field
from enum import Enum


class PreprocessingBackend(str, Enum):
    """Engine that produced the preprocessed text."""

    DOCLING = "docling"
    PLAIN_TEXT = "plain_text"


class SourceFormat(str, Enum):
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


# Extraction consumes text plus provenance through this shared wrapper.
@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentBase:
    """Immutable provenance shared by every document-stage artifact."""

    metadata: PreprocessedDocumentMetadata

    @property
    def source_path(self) -> str | None:
        """Original source path, when known."""
        return self.metadata.source_path


@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessedDocument(DocumentBase):
    """Layer 2 text output consumed by citation extraction."""

    text: str

    def __post_init__(self) -> None:
        if not self.text:
            msg = "PreprocessedDocument.text must not be empty"
            raise ValueError(msg)
