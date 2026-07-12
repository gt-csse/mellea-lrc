"""Extraction result types."""

from dataclasses import dataclass
from enum import Enum

from mellea_lrc.core.citations import CanonicalCitation, is_full_citation
from mellea_lrc.core.spans import Span
from mellea_lrc.preprocessing.types import PreprocessedDocument


class ExtractionBackend(str, Enum):
    """Engine that produced the extracted citations."""

    EYECITE = "eyecite"
    MELLEA = "mellea"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ExtractionMetadata:
    """Provenance for the extraction stage."""

    backend: ExtractionBackend = ExtractionBackend.EYECITE
    backend_version: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedCitation:
    """A canonical citation located in document text.

    ``citation_span`` is the eyecite full-citation span. For full case
    citations this can include party names, locator, court/date parenthetical,
    and pin cites.

    eyecite's ``matched_text()`` is the locator string for full citations, so
    we persist it as ``matched_locator_text``. ``matched_citation_text`` is the
    exact document slice covered by ``citation_span``.
    """

    citation_id: str
    citation_span: Span
    matched_locator_text: str
    matched_citation_text: str
    citation: CanonicalCitation
    # Transitional extraction-side projection of eyecite's structured
    # year/month/day. Date components should eventually move into the
    # canonical citation model rather than remain beside it.
    asserted_decision_date: str | None = None
    resolves_to: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractedDocument(PreprocessedDocument):
    """A preprocessed document with canonical extracted citations."""

    citations: tuple[ExtractedCitation, ...]
    extraction_metadata: ExtractionMetadata

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
            if item.citation_span.end > len(self.text):
                msg = f"Citation {item.citation_id!r} citation_span exceeds document text"
                raise ValueError(msg)
            source_text = self.text[item.citation_span.start : item.citation_span.end]
            if source_text != item.matched_citation_text:
                msg = f"Citation {item.citation_id!r} matched_citation_text does not match citation_span"
                raise ValueError(msg)
            if item.resolves_to is not None and (
                item.resolves_to not in known_ids or item.resolves_to == item.citation_id
            ):
                msg = f"Citation {item.citation_id!r} has invalid resolves_to={item.resolves_to!r}"
                raise ValueError(msg)
