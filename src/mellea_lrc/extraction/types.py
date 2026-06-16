"""Extraction result types."""

from dataclasses import dataclass

from mellea_lrc.core.citations import CanonicalCitation, is_full_citation
from mellea_lrc.core.spans import Span
from mellea_lrc.preprocessing.types import PreprocessedDocument


@dataclass(frozen=True, slots=True)
class ExtractedCitation:
    """A canonical citation located in document text."""

    citation_id: str
    span: Span
    matched_text: str
    citation: CanonicalCitation
    resolves_to: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    """All citations extracted from one Layer 2 document."""

    preprocessed: PreprocessedDocument
    citations: tuple[ExtractedCitation, ...]

    @property
    def text(self) -> str:
        """Layer 2 text that was extracted from."""
        return self.preprocessed.text

    @property
    def source_path(self) -> str | None:
        """Original source path, when known."""
        return self.preprocessed.metadata.source_path

    @property
    def full_citations(self) -> tuple[ExtractedCitation, ...]:
        """Return only self-contained bibliographic citations."""
        return tuple(item for item in self.citations if is_full_citation(item.citation))
