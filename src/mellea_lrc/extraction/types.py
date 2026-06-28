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


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractedDocument(PreprocessedDocument):
    """A preprocessed document with canonical extracted citations."""

    citations: tuple[ExtractedCitation, ...]

    @property
    def full_citations(self) -> tuple[ExtractedCitation, ...]:
        """Return only self-contained bibliographic citations."""
        return tuple(item for item in self.citations if is_full_citation(item.citation))

    def __post_init__(self) -> None:
        PreprocessedDocument.__post_init__(self)
        citation_ids = [item.citation_id for item in self.citations]
        if any(not citation_id for citation_id in citation_ids):
            msg = "Extracted citation identifiers must not be empty"
            raise ValueError(msg)
        if len(citation_ids) != len(set(citation_ids)):
            msg = "Extracted citation identifiers must be unique within a document"
            raise ValueError(msg)

        known_ids = set(citation_ids)
        for item in self.citations:
            if item.span.end > len(self.text):
                msg = f"Citation {item.citation_id!r} span exceeds document text"
                raise ValueError(msg)
            if item.resolves_to is not None and (
                item.resolves_to not in known_ids or item.resolves_to == item.citation_id
            ):
                msg = f"Citation {item.citation_id!r} has invalid resolves_to={item.resolves_to!r}"
                raise ValueError(msg)
