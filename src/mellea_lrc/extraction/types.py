"""Extraction result types."""

from dataclasses import dataclass
from enum import Enum

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import is_full_citation
from mellea_lrc.core.findings import Finding
from mellea_lrc.core.pin_cites import PinCitePages
from mellea_lrc.core.record import CitationRecord
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.relaxation import Relaxation
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
    relaxation: Relaxation = Relaxation.FULL
    """Which tokenizer read the text.

    Two levels disagree about whether a given citation is there at all, so a
    document that does not say which one ran cannot be compared with another.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class Document(PreprocessedDocument):
    """One document and everything the pipeline has read in it.

    Not the output of a stage. Every stage appends to this object -- extraction
    adds the roots and then the leaves, validation adds nodes to them and
    withdraws what reaches nothing -- and each hands on the same document with
    more written on it rather than an artifact of its own kind wrapping the one
    before. See `docs/Document.md`.

    What is here and not on a citation is here because it cannot be on one.
    `text` is the coordinate space every span indexes, and per-citation copies
    of it could disagree. `findings` are true of the document and of no citation
    in it. `passes` is what has run, which is the only way a consumer can tell a
    document whose leaves have not grown from one that writes no short forms.
    """

    citations: tuple[CitationRecord, ...]
    unread_case_names: tuple[Span, ...] = ()
    """Text naming a case that no citation covers.

    Read last, from the document with every citation blanked, so it holds what
    nothing else read: a case cited with no locator, which a reporter-driven
    tokenizer cannot see at all, and a case whose citation *was* read but whose
    name was not reached. See
    :mod:`mellea_lrc.extraction.reading.unread_names`.
    """
    findings: tuple[Finding, ...] = ()
    """What a pass learned that belongs to no citation.

    A leaf it could not grow, a site it hunted and rejected. Beside the
    citations rather than among them, because a record is a citation and one
    standing for a citation the document does not hold would be a hole in the
    invariant. See :mod:`mellea_lrc.core.findings`.
    """
    passes: tuple[str, ...] = ()
    """Which passes have run over this document, in order.

    `("roots",)` is the first growth and is **not a reading of the document's
    citations** -- it holds what states a complete identifier and no leaf of any
    kind. `("roots", "leaves")` is the whole of what the filing writes. A
    consumer that does not check this cannot tell the two apart.
    """
    extraction_metadata: ExtractionMetadata

    @property
    def full_citations(self) -> tuple[CitationRecord, ...]:
        """Return only self-contained bibliographic citations."""
        return tuple(item for item in self.citations if is_full_citation(item.stated))

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
            if item.full_span.end > len(self.text):
                msg = f"Citation {item.citation_id!r} span exceeds document text"
                raise ValueError(msg)
            if item.locator_span.end > len(self.text):
                msg = f"Citation {item.citation_id!r} locator span exceeds document text"
                raise ValueError(msg)
            if item.locator_span.start < item.full_span.start or item.locator_span.end > item.full_span.end:
                msg = f"Citation {item.citation_id!r} locator span must be within its full span"
                raise ValueError(msg)
            if item.resolves_to is not None and (
                item.resolves_to not in known_ids or item.resolves_to == item.citation_id
            ):
                msg = f"Citation {item.citation_id!r} has invalid resolves_to={item.resolves_to!r}"
                raise ValueError(msg)


#: The name the document carried while it was only extraction's output. Kept so
#: the name a caller wrote still imports; it is the same class.
ExtractedDocument = Document
