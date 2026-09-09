"""Extraction result types."""

from dataclasses import dataclass
from enum import Enum

from mellea_lrc.core.citations import CanonicalCitation, is_full_citation
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
    relaxation: Relaxation = Relaxation.BOUNDED
    """Which tokenizer read the text.

    Two levels disagree about whether a given citation is there at all, so a
    document that does not say which one ran cannot be compared with another.
    """


@dataclass(frozen=True, slots=True)
class ExtractedCitation:
    """A canonical citation with full and matched-locator spans in document text."""

    citation_id: str
    full_span: Span
    """The citation's whole extent: party names, locator, pin cite, parenthetical."""
    locator_span: Span
    """The minimum sufficient identifier -- volume, reporter and page.

    Named apart from `full_span` because the two answer different questions: this
    is what a lookup resolves, that is what a reader is shown.
    """
    matched_text: str
    citation: CanonicalCitation
    pin_cite_span: Span | None = None
    """Where the pin cite was read from, or `None` when the citation states none.

    The page a filing argues from is not part of the case's identity -- a
    retrieval settles the case name and the court, and cannot settle the page --
    so it is scored on its own, and scoring it needs somewhere to point.

    eyecite supplies this for full case citations only; every other kind is
    located by :mod:`mellea_lrc.extraction.reading.pin_cite_spans`.
    """
    resolves_to: str | None = None
    root_id: str | None = None
    """The citation that stated the identifier this one refers to.

    A **root** is an identifier the filing states -- a claim about which case it
    means -- given once in full and returned to as `Id. at 570`,
    `550 U.S. at 563` or by party name, each return being its own claim about
    its own page. This carries which root, so a consumer does not have to
    rebuild the chain -- and so a corrected attribution survives, which a chain
    of `resolves_to` cannot express: document 022's `Id. at 1072-73` belongs to
    Advanced Textile because a reader said so, not because anything points there.

    A root is not an **authority**. An authority is the identity a root is
    established to have, which takes a lookup: two roots written at one position
    can reach one authority, and a root can reach none. Extraction states the
    claim; validation settles it.

    `None` means **not attributed**, which is a real answer and usually the right
    one. Why it is absent -- the reference is to the record rather than to a
    case, or the chain reached nothing -- is
    :mod:`~mellea_lrc.extraction.structure.citation_tree`'s to explain, exactly
    as `colocation.py` explains why an id is shared.
    """

    colocation_id: str | None = None
    """Shared by citations occupying the same place in the text.

    A filing citing an authority in parallel writes several identifiers for one
    citation, and eyecite extracts each separately. Citations carrying the same
    `colocation_id` are candidates for reaching one authority -- **candidates,
    not a finding**: whether they name the same case is settled by resolving
    them, not by where they sit. `None` means the citation stands alone, which is the
    common case. See :mod:`mellea_lrc.extraction.structure.colocation`.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractedDocument(PreprocessedDocument):
    """A preprocessed document with canonical extracted citations."""

    citations: tuple[ExtractedCitation, ...]
    unread_case_names: tuple[Span, ...] = ()
    """Text naming a case that no citation covers.

    Read last, from the document with every citation blanked, so it holds what
    nothing else read: a case cited with no locator, which a reporter-driven
    tokenizer cannot see at all, and a case whose citation *was* read but whose
    name was not reached. See
    :mod:`mellea_lrc.extraction.reading.unread_names`.
    """
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
