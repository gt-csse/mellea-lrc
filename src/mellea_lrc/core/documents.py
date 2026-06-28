"""Stage-neutral document identity and source provenance."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from mellea_lrc.core.immutable import freeze_string_map


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
    extras: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", freeze_string_map(self.extras))


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentBase:
    """Stage-neutral base for immutable document artifacts."""

    source_metadata: SourceMetadata

    @property
    def source_path(self) -> str | None:
        """Original source path, when known."""
        return self.source_metadata.path
