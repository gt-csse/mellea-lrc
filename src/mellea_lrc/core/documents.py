"""Stage-neutral document identity and source provenance."""

from dataclasses import dataclass
from enum import Enum


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
class SourceMetadata:
    """Identity and provenance of the source supplied to the pipeline."""

    path: str | None = None
    format: SourceFormat = SourceFormat.UNKNOWN
    header: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentBase:
    """Stage-neutral base for immutable document artifacts."""

    source_metadata: SourceMetadata

    @property
    def source_path(self) -> str | None:
        """Return the original source path, when known."""
        return self.source_metadata.path
