"""Stage-neutral document identity and source provenance."""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict


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


class DocumentBase(BaseModel):
    """Stage-neutral base for natively serializable document artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    source_metadata: SourceMetadata

    @property
    def source_path(self) -> str | None:
        """Return the original source path, when known."""
        return self.source_metadata.path
