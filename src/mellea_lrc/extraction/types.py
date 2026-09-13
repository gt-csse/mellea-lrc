"""Extraction result types."""

from dataclasses import InitVar, dataclass, field
from enum import Enum

from mellea_lrc.core.citations import CanonicalCitation, is_full_citation
from mellea_lrc.core.field_log import RULES, FieldLog
from mellea_lrc.core.pin_cites import PinCitePages, read_pin_cite
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.relaxation import Relaxation
from mellea_lrc.preprocessing.types import PreprocessedDocument

#: The one field logged today. See :mod:`mellea_lrc.core.field_log`.
CASE_NAME = "case_name"


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

    case_name_read: InitVar[Span | None] = None
    """Where whatever built this citation read its case name, if it read one.

    Init-only: it opens `field_log`, which is where the name lives from then on.
    Read it back as `case_name_span`.
    """

    read_by: InitVar[str] = RULES
    """What built this citation, which is the first entry in its log.

    `extraction` for the deterministic pass. A reader that proposes a citation
    the rules never read passes its own name, so the record says a citation was
    recovered rather than parsed.
    """

    field_log: FieldLog = field(default_factory=FieldLog, repr=False, compare=False)
    """Every touch on every logged field, in order. See
    :mod:`mellea_lrc.core.field_log`.

    Mutable and shared: `dataclasses.replace` copies the reference, so a
    citation that gains a `root_id` keeps the history it already had.
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

    def __post_init__(self, case_name_read: Span | None, read_by: str) -> None:
        # A log that already has a history belongs to this citation already:
        # `replace` passes the same object, and re-seeding it would record a
        # read that never happened.
        if not self.field_log.history(CASE_NAME):
            self.field_log.touch(CASE_NAME, case_name_read, by=read_by)

    @property
    def case_name_span(self) -> Span | None:
        """Where this citation's case name is written, as last read.

        Located rather than rebuilt: see
        :mod:`mellea_lrc.extraction.reading.case_names`.

        A span rather than a parse, because a case name is not always two
        parties: `In re Flint Water Cases` is a whole name, and eyecite files it
        under `defendant` with the opening words stripped. The parsed fields are
        left exactly as they were read; this says where to find the name on the
        page.

        It may sit outside `full_span`. A filing writes `In Boeser v. Sharp ,
        the court recognized …` and then the citation a sentence later, and the
        name in that sentence is the same case name -- neither position is the
        wrong one, and the fuller of the two is what a reader wants.

        The value is the last touch in `field_log`, so a name a reader wrote
        here reads back the same way a parsed one does, and what it replaced is
        still on the record.
        """
        return self.field_log.value(CASE_NAME)

    def record_case_name(self, span: Span | None, *, by: str, reason: str | None = None) -> None:
        """Write a case name over whatever the field holds, and say who did.

        Writing `None` over a span, and a span over `None`, are both overwrites
        and both are recorded. Nothing is checked here: whether the name is the
        right one is the caller's finding, and the log is what makes it
        reviewable.
        """
        self.field_log.touch(CASE_NAME, span, by=by, reason=reason)

    @property
    def pin_cite_pages(self) -> tuple[PinCitePages, ...]:
        """Which pages the pin cite claims, or `()` when it states none.

        Derived rather than stored, so it cannot disagree with the text it was
        read from. `pin_cite` keeps the filing's own spelling, damage included;
        this is what that spelling means, and it is what a page claim is
        compared on -- `247-48` and `247 - 248` are one claim, and a string
        equality cannot see it.
        """
        return read_pin_cite(getattr(self.citation, "pin_cite", None))


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
