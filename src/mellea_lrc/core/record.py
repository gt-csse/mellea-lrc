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

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from mellea_lrc.core.citations import is_leaf

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

    details: Mapping[str, Any] = field(default_factory=dict)
    """Whatever the node that made this wants to keep, carried and not read.

    Serialization writes it back verbatim and nothing in `core` looks inside it,
    so a stage with typed nodes of its own -- a locator lookup with its cluster,
    a search with its candidates -- keeps its own fields without `core` knowing
    any of its types. It is the node's own record of what it did, not a place to
    put a citation's state: anything the pipeline reads later belongs on the
    record, where the next reader will look for it.
    """

    def __post_init__(self) -> None:
        if not self.node_id:
            msg = "A node must have an identifier"
            raise ValueError(msg)
        if self.corrections and self.reads is not Reads.DOCUMENT:
            msg = f"Node {self.node_id!r} read a record, so it cannot correct what the filing states"
            raise ValueError(msg)


#: The outcome that marks a citation as one the document does not hold. It is a
#: node like any other, because withdrawing is a reading and a reading needs its
#: evidence -- what was asked, what came back, and why this is not a citation to
#: a case.
WITHDRAWN = "withdrawn"


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
        # A leaf cannot exist without a root. `556 U.S. at 678` claims page 678
        # of a case those characters do not name, so a leaf with no `root_id` is
        # not an incomplete citation -- it is a citation of nothing. The rules
        # cannot attach one reliably either, because attaching it means matching
        # a case name and the names they read are parses; the leaves are grown
        # after validation has settled the roots. See `docs/Extraction.md`,
        # "Roots first, leaves after validation".
        if self.root_id is None and is_leaf(self.stated):
            msg = (
                f"Citation {self.citation_id!r} is a {self.stated.kind.value} and states no root. "
                "A leaf is built from an admitted root or not at all."
            )
            raise ValueError(msg)

    @property
    def corrections(self) -> tuple[Correction, ...]:
        """Every change to `stated`, in the order they were made."""
        return tuple(correction for node in self.trace for correction in node.corrections)

    @property
    def withdrawn(self) -> bool:
        """Whether a reading has taken this citation out of the document.

        Read off the trace rather than stored beside it. A statute read as a
        case, a docket number that is a record entry, a root that reaches no
        authority: the span stays, the record stays addressable -- `root_id` and
        `authority_id` name citation ids, and deleting a record would break
        every reference to it -- and the node says who took it out and why.
        """
        return any(node.outcome == WITHDRAWN for node in self.trace)

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
