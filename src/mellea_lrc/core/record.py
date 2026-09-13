"""One citation as it is being read, and the evidence for every change to it.

Extraction produces frozen citations. That is the right shape for what the rules
read and the wrong shape for the thing being read *about*, which changes as the
pipeline proceeds: a reader looks at a name standing outside every citation and
says it belongs to one of them, a lookup says two roots name one case, and by
the end the citation the pipeline holds is not the one extraction produced.

So the record is mutable, and four rules keep it auditable.

**The original is never touched.** `source` is the extracted citation as it
arrived. `stated` is the pipeline's current reading of what the filing states.
A reader comparing the two sees exactly what was changed.

**Every change is inside the evidence for it.** A `Correction` lives on the
`Node` that justified it, so a change with no evidence cannot be constructed --
the invariant is the shape rather than a check. `record.corrections` reads them
all back in order, and the state after any point is a fold over a prefix of the
trace, which is what an evaluation slices on.

**What the filing states and what an archive holds are kept apart.** `stated`
is only ever the filing's reading; the archive's answer goes on `found`. A
filing citing the right case under the wrong year keeps its wrong year on
`stated` and gets the right one on `found`, and the disagreement between them is
the finding. Overwriting one with the other erases the defect the pipeline
exists to report.

**A node is named for what it read, not for the stage that ran it.** A model
re-reading a name from the filing's own text produces document evidence whether
it runs in the case-name layer or inside identity; those are the same operation
and only `stage` differs. `reads` is what decides where a node may write:
`DOCUMENT` evidence corrects `stated`, `RECORD` evidence settles `found`.

`found`, `authority_id` and `Resolution` are named here and filled by
validation. They are written down now so both sides use one word for one thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mellea_lrc.core.case_names import CaseName
    from mellea_lrc.core.citations import CanonicalCitation
    from mellea_lrc.core.pin_cites import PinCitePages
    from mellea_lrc.core.spans import Span


class Reads(str, Enum):
    """Where a node's evidence came from, which is what it may write."""

    DOCUMENT = "document"
    """The filing's own text. May correct `stated`."""

    RECORD = "record"
    """An archive. May settle `found`, and never touches `stated`."""


@dataclass(frozen=True, slots=True)
class Correction:
    """One change to the filing's reading, on the node that justified it."""

    field: str
    """Which field of the citation changed: `case_name`, `court`, `pin_cite`."""

    before: Any
    after: Any
    reason: str
    """One line, in the words of whatever made the change."""

    def __post_init__(self) -> None:
        if self.before == self.after:
            msg = f"A correction to {self.field!r} must change the value"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Node:
    """One thing that was asked, what came back, and what it changed."""

    node_id: str
    reads: Reads
    stage: str
    """Which stage ran it: `extraction`, `identity`, `pinpoint`."""

    made_by: str
    """A rule, by module name, or a model, by the name the session reports."""

    outcome: str
    """What it found, in the vocabulary of whatever produced it."""

    message: str | None = None
    depends_on: tuple[str, ...] = ()
    corrections: tuple[Correction, ...] = ()
    """The changes this node justified. Empty for most nodes, which only observe."""

    def __post_init__(self) -> None:
        if not self.node_id:
            msg = "A node must have an identifier"
            raise ValueError(msg)
        if self.corrections and self.reads is not Reads.DOCUMENT:
            msg = f"Node {self.node_id!r} read a record, so it cannot correct what the filing states"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Resolution:
    """What an archive holds at the identity the filing cited. Validation fills it."""

    cluster_id: str | None
    case_name: str | None
    date_filed: str | None
    court_id: str | None
    node_id: str
    """The node that established it."""


@dataclass(slots=True)
class CitationRecord:
    """One citation: what the rules read, what it is now, and the evidence between.

    This is the unit a document holds. There is no separate "extracted
    citation": a citation and its history are one object from the moment the
    rules produce it, which is what lets a reader's answer reach the next reader
    instead of being thrown away.
    """

    citation_id: str
    source: CanonicalCitation
    """What the rules read, frozen. Never touched, so the diff is readable."""

    stated: CanonicalCitation = None  # type: ignore[assignment]
    """The same citation as currently read.

    The same type as `source`, so they compare field by field:
    `source.case_name` against `stated.case_name` is the whole of what a reader
    changed. Defaults to `source`, which is what a citation nobody has read yet
    states.
    """

    resolves_to: str | None = None
    root_id: str | None = None
    """The citation that stated the identifier this one refers to.

    A **root** is an identifier the filing states -- a claim about which case it
    means -- given once in full and returned to as `Id. at 570`,
    `550 U.S. at 563` or by party name, each return being its own claim about
    its own page. This carries which root, so a consumer does not have to
    rebuild the chain, and so a corrected attribution survives, which a chain of
    `resolves_to` cannot express.

    `None` means **not attributed**, which is a real answer and usually the
    right one.
    """

    colocation_id: str | None = None
    """Shared by citations occupying the same place in the text.

    A filing citing an authority in parallel writes several identifiers for one
    citation, and eyecite extracts each separately. Citations carrying the same
    `colocation_id` are candidates for reaching one authority -- **candidates,
    not a finding**. See :mod:`mellea_lrc.extraction.structure.colocation`.
    """

    authority_id: str | None = None
    """The authority the root was established to reach. `None` until a lookup settles it."""

    found: Resolution | None = None
    trace: tuple[Node, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.stated is None:
            self.stated = self.source

    @property
    def corrections(self) -> tuple[Correction, ...]:
        """Every change to `stated`, in the order they were made."""
        return tuple(correction for node in self.trace for correction in node.corrections)

    @property
    def authority(self) -> str | None:
        """The authority this citation belongs to: the one a lookup found, else its root."""
        return self.authority_id or self.root_id

    @property
    def full_span(self) -> Span:
        """The citation's whole extent: name, locator, pin cite, parenthetical."""
        return self._span("span")

    @property
    def locator_span(self) -> Span:
        """The minimum sufficient identifier -- volume, reporter and page.

        Named apart from `full_span` because the two answer different questions:
        this is what a lookup resolves, that is what a reader is shown.
        """
        return self._span("locator_span")

    @property
    def matched_text(self) -> str:
        """The characters the parse matched, as eyecite read them."""
        return self.stated.matched_text or ""

    @property
    def case_name(self) -> CaseName | None:
        """The name this citation is currently read under."""
        return self.stated.case_name

    @property
    def case_name_span(self) -> Span | None:
        """Where that name is written, for a reader that wants only the position."""
        name = self.stated.case_name
        return name.span if name is not None else None

    @property
    def pin_cite_span(self) -> Span | None:
        """Where the pin cite was read from, or `None` when it states none."""
        pin_cite = getattr(self.stated, "pin_cite", None)
        return pin_cite.span if pin_cite is not None else None

    @property
    def pin_cite_pages(self) -> tuple[PinCitePages, ...]:
        """Which pages the pin cite claims, or `()` when it states none."""
        pin_cite = getattr(self.stated, "pin_cite", None)
        return pin_cite.pages if pin_cite is not None else ()

    def observe(self, node: Node) -> Node:
        """Add one node to the trace, returning it so a caller can depend on it.

        Corrections travel on the node, so a node that changes something arrives
        already carrying what it changed, and `stated` is brought up to it here.
        There is no way to correct without leaving the evidence.
        """
        self.trace = (*self.trace, node)
        for correction in node.corrections:
            self.stated = replace(self.stated, **{correction.field: correction.after})
        return node

    def correcting(self, node: Node, field_name: str, value: Any, *, reason: str) -> Node:
        """The same node, carrying a correction to one field of `stated`.

        `None` over a value and a value over `None` are both changes and both
        are recorded.
        """
        before = getattr(self.stated, field_name)
        correction = Correction(field=field_name, before=before, after=value, reason=reason)
        return replace(node, corrections=(*node.corrections, correction))

    def _span(self, name: str) -> Span:
        span = getattr(self.stated, name)
        if span is None:
            msg = f"Citation {self.citation_id!r} was read without a {name}"
            raise ValueError(msg)
        return span
