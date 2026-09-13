"""Extraction result types."""

from dataclasses import dataclass
from enum import Enum

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import CanonicalCitation, is_full_citation
from mellea_lrc.core.pin_cites import PinCitePages
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


@dataclass(frozen=True, slots=True)
class ExtractedCitation:
    """One citation the rules read, and where the document holds it.

    **The citation carries its own position.** `span`, `locator_span`, `text`,
    `case_name` and `pin_cite` are fields of
    :data:`~mellea_lrc.core.citations.CanonicalCitation`, because content and
    position go out of step the moment they are stored apart -- which is what
    happened when a reader repaired a case name and the span it repaired stayed
    behind. The accessors below read them, so nothing has to reach inside.

    What is left here is what belongs to the citation's place in a *document*
    rather than to the citation: the identifier this pass assigned it, and which
    other citation it returns to.
    """

    citation_id: str
    citation: CanonicalCitation
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
    one.
    """

    colocation_id: str | None = None
    """Shared by citations occupying the same place in the text.

    A filing citing an authority in parallel writes several identifiers for one
    citation, and eyecite extracts each separately. Citations carrying the same
    `colocation_id` are candidates for reaching one authority -- **candidates,
    not a finding**: whether they name the same case is settled by resolving
    them, not by where they sit. `None` means the citation stands alone, which
    is the common case. See :mod:`mellea_lrc.extraction.structure.colocation`.
    """

    @property
    def full_span(self) -> Span:
        """The citation's whole extent: name, locator, pin cite, parenthetical."""
        span = self.citation.span
        if span is None:
            msg = f"Citation {self.citation_id!r} was read without a position"
            raise ValueError(msg)
        return span

    @property
    def locator_span(self) -> Span:
        """The minimum sufficient identifier -- volume, reporter and page.

        Named apart from `full_span` because the two answer different questions:
        this is what a lookup resolves, that is what a reader is shown.
        """
        span = self.citation.locator_span
        if span is None:
            msg = f"Citation {self.citation_id!r} was read without a locator position"
            raise ValueError(msg)
        return span

    @property
    def matched_text(self) -> str:
        """The characters the parse matched: the locator, as eyecite read them."""
        return self.citation.matched_text or ""

    @property
    def case_name(self) -> CaseName | None:
        """The name this citation is written under, as the rules read it."""
        return self.citation.case_name

    @property
    def case_name_span(self) -> Span | None:
        """Where that name is written, for a reader that wants only the position."""
        name = self.citation.case_name
        return name.span if name is not None else None

    @property
    def pin_cite_span(self) -> Span | None:
        """Where the pin cite was read from, or `None` when it states none."""
        pin_cite = self.citation.pin_cite
        return pin_cite.span if pin_cite is not None else None

    @property
    def pin_cite_pages(self) -> tuple[PinCitePages, ...]:
        """Which pages the pin cite claims, or `()` when it states none.

        Read once, when the pin cite is read, so the two cannot disagree:
        `pin_cite` keeps the filing's own spelling, damage included, and this is
        what that spelling means -- `247-48` and `247 - 248` are one claim, and
        a string equality cannot see it.
        """
        pin_cite = self.citation.pin_cite
        return pin_cite.pages if pin_cite is not None else ()


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
