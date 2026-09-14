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

**Every change names the evidence for it.** A `Correction` carries the
`node_id` of the node that justified it, and `record.correct` is the only way to
change `stated`: it writes the node and the correction together, so a change
with no evidence cannot be made. The corrections are an ordered list on the
record rather than something held inside the trace, because the trace is a
**graph** -- a node names what it depends on -- and a history held inside a
graph can only be read by walking it. The same goes for the judgement and for
the withdrawal: what the pipeline currently says is a field, and the node that
said it is a pointer. Nothing about a citation's current state is recovered by
traversal.

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
    """One change to the filing's reading, and the node that justified it."""

    field: str
    """Which field of the citation changed: `case_name`, `court`, `pin_cite`."""

    before: Any
    after: Any
    reason: str
    """One line, in the words of whatever made the change."""

    node_id: str
    """The node this rests on. A pointer, not a copy.

    The corrections are kept in order on the record and the node is named from
    here, rather than the corrections being kept inside the nodes. A trace is a
    graph -- a node names what it depends on -- so a correction history held
    inside it can only be recovered by walking the graph, and the history is
    read far more often than the graph is.
    """

    def __post_init__(self) -> None:
        if self.before == self.after:
            msg = f"A correction to {self.field!r} must change the value"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Node:
    """One thing that was asked, and what came back.

    What it *changed* is not here. A node is evidence, and the state it
    justified -- a correction, a judgement, a withdrawal -- is written on the
    record with this node's id beside it. See `docs/Document.md`.
    """

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


#: What a citation's judgement says before anything has judged it. Written at
#: initialization rather than left absent, so a reader never has to tell "no
#: judgement was made" apart from "no judgement was found".
UNJUDGED = "unjudged"


class Question(str, Enum):
    """A question the pipeline asks of a citation, and keeps an answer to.

    Named for **what is asked**, not for the stage that asks it -- the same rule
    a node follows. A citation is judged once per question, so a later answer to
    a different question never overwrites an earlier one: whether a citation
    reaches the authority it names and whether the page it claims says what it
    is cited for are two findings, and a filing can fail either alone.

    Adding a question here is how a new stage gets somewhere to put its verdict.
    The outcomes are the asking stage's own vocabulary; only the questions are
    shared.
    """

    IDENTITY = "identity"
    """Does this citation reach the authority it names?"""

    PINPOINT = "pinpoint"
    """Does the page it claims say what it is cited for?"""


@dataclass(frozen=True, slots=True)
class Judgement:
    """What the pipeline concludes about one citation, and what concluded it."""

    outcome: str
    """In the vocabulary of whatever judged it. `unjudged` until something does."""

    node_id: str | None = None
    """The node that reached it. `None` while the outcome is `unjudged`."""

    message: str | None = None


#: The judgement a citation carries from the moment it is read.
UNJUDGED_YET = Judgement(outcome=UNJUDGED)


def unjudged() -> dict[Question, Judgement]:
    """Every question, unanswered. What a citation is born carrying."""
    return dict.fromkeys(Question, UNJUDGED_YET)


#: The outcome that marks a citation as one the document does not hold. It is a
#: node like any other, because withdrawing is a reading and a reading needs its
#: evidence -- what was asked, what came back, and why this is not a citation to
#: a case.
WITHDRAWN = "withdrawn"


@dataclass(frozen=True, slots=True)
class DateExploration:
    """Everything read about a record's dates while resolving, kept for a later step.

    Written whether or not the date reconciled: the date the filing stated and
    at what precision, the record's filing date, the cluster's free-text other
    dates, the dated phrases found in each opinion header read, and the phrase
    that matched when one did. A wrong identity whose only disagreeing field is
    the date is a strong sign the case is real and the dates differ for a
    reason -- an amendment the archive does not hold, a term-year convention, a
    rehearing -- and that reason has to be found from what was read rather than
    from the verdict alone.
    """

    stated: str
    """The date the filing states, `YYYY` or `YYYY-MM-DD`."""
    stated_precision: str
    """`year` or `day`."""
    record_date_filed: str | None
    other_dates: str | None
    """The cluster's free-text dates, verbatim, when fetched."""
    phrases_by_opinion: tuple[tuple[str, tuple[str, ...]], ...]
    """For each opinion header read, in order: its id and the dated events in it."""
    matched_phrase: str | None
    matched_opinion_id: str | None
    """The opinion whose header stated the filing's year, or None."""


@dataclass(frozen=True, slots=True)
class Resolution:
    """What an archive holds at the identity the filing cited. Validation fills it."""

    cluster_id: str | None
    case_name: str | None
    date_filed: str | None
    court_id: str | None
    node_id: str
    """The node that established it."""
    opinion_ids: tuple[str, ...] = ()
    """The cluster's opinions: the text a later stage reads the pages from."""
    citations: tuple[str, ...] = ()
    """The reporter citations the archive lists for the cluster, `volume reporter page`."""
    dates: DateExploration | None = None
    """What was read about the record's dates, when the plain comparison disagreed."""
    duplicates: tuple[Resolution, ...] = ()
    """Other records that agreed with the filing on every field: one decision the
    archive holds more than once. The words a filing quotes may be in a copy the
    page was not cut from, so a later stage reads these too."""
    docket_id: str | None = None
    """CourtListener's docket, when the case was identified by docket number."""
    govinfo_package_id: str | None = None
    """The Publishing Office's package, when identified by docket number."""


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

    corrections: tuple[Correction, ...] = field(default_factory=tuple)
    """Every change to `stated`, in the order they were made.

    Kept here rather than inside the nodes. A trace is a graph and a history is
    a sequence; holding the sequence inside the graph means walking the graph to
    read it, and this is read on every comparison of what the filing wrote
    against what the pipeline now says it wrote.
    """

    judgements: dict[Question, Judgement] = field(default_factory=unjudged)
    """What the pipeline concludes about this citation, one answer per question.

    Every question is present from the moment the citation is read, saying
    `unjudged`, so a reader never has to tell an absent judgement from an unmade
    one -- and never has to search the trace for whichever node happened to be
    the aggregation.

    **One slot per question, not per stage.** Identity and pinpoint both reach a
    verdict on a root that states a page, and they are answering different
    questions: a citation can reach the right case and misstate the page, or
    reach nothing at all. One field would make the second stage overwrite the
    first. There is deliberately no combined verdict -- how a wrong page and a
    right case add up is the reader's finding to make, not the record's.
    """

    withdrawn_by: str | None = None
    """The node that took this citation out of the document, if one has.

    Written, not derived. A statute read as a case, a docket number that is a
    record entry, a root that reaches no authority: the record stays and stays
    addressable -- `root_id` and `authority_id` name citation ids, and deleting
    a record would break every reference to it -- and this says who took it out.
    """

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
    def withdrawn(self) -> bool:
        """Whether a reading has taken this citation out of the document."""
        return self.withdrawn_by is not None

    @property
    def is_root(self) -> bool:
        """Whether this citation states the identifier rather than referring to one."""
        return self.root_id == self.citation_id

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
        """Add one node to the trace, returning it so a caller can depend on it."""
        if node.node_id not in {seen.node_id for seen in self.trace}:
            self.trace = (*self.trace, node)
        return node

    def correct(self, node: Node, field_name: str, value: Any, *, reason: str) -> Node:
        """Change one field of `stated`, on the evidence of `node`.

        The node is added to the trace and the correction is appended with its
        id, so a correction cannot exist without the evidence for it: this is
        the only way to change `stated`, and it writes both or neither.

        `None` over a value and a value over `None` are both changes and both
        are recorded.
        """
        if node.reads is not Reads.DOCUMENT:
            msg = f"Node {node.node_id!r} read a record, so it cannot correct what the filing states"
            raise ValueError(msg)
        before = getattr(self.stated, field_name)
        self.observe(node)
        self.corrections = (
            *self.corrections,
            Correction(
                field=field_name, before=before, after=value, reason=reason, node_id=node.node_id
            ),
        )
        self.stated = replace(self.stated, **{field_name: value})
        return node

    def judge(
        self, node: Node, question: Question, outcome: str, *, message: str | None = None
    ) -> Node:
        """Answer one question about this citation, on the evidence of `node`.

        Answering one leaves every other question as it was, so a later stage
        adds a finding rather than replacing one.
        """
        self.observe(node)
        self.judgements[question] = Judgement(
            outcome=outcome, node_id=node.node_id, message=message
        )
        return node

    def judgement(self, question: Question) -> Judgement:
        """This citation's answer to one question, `unjudged` until something answers it."""
        return self.judgements.get(question, UNJUDGED_YET)

    def resolve(self, node: Node, resolution: Resolution) -> Node:
        """Settle what an archive holds at this citation's identity, on the evidence of `node`."""
        if node.reads is not Reads.RECORD:
            msg = f"Node {node.node_id!r} read the filing, so it cannot settle what an archive holds"
            raise ValueError(msg)
        if resolution.node_id != node.node_id:
            msg = f"Resolution names node {resolution.node_id!r}, not {node.node_id!r}"
            raise ValueError(msg)
        self.observe(node)
        self.found = resolution
        return node

    def reattribute(self, node: Node, authority_id: str) -> Node:
        """Settle which authority this root reaches, on the evidence of `node`.

        `root_id` is what the filing stated and stays; this writes what a lookup
        found, which is `authority_id`.
        """
        self.observe(node)
        self.authority_id = authority_id
        return node

    def withdraw(self, node: Node) -> Node:
        """Take this citation out of the document, on the evidence of `node`.

        The record stays. Withdrawing is a reading like any other and the node
        carries what was asked and what came back.
        """
        self.observe(node)
        self.withdrawn_by = node.node_id
        return node

    def _span(self, name: str) -> Span:
        span = getattr(self.stated, name)
        if span is None:
            msg = f"Citation {self.citation_id!r} was read without a {name}"
            raise ValueError(msg)
        return span
