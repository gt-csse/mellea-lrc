"""Citation state and the evidence supporting each durable change.

Creation selects a citation kind and starts with empty fields. Document readers
then update the one current field set, including its first locator reading.
Every durable operation keeps its resulting value and the node that justified it.
A node is an observation; it may support zero, one, or several state changes.
Retrieved records and judgements remain separate from what the filing states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from pydantic import Field, JsonValue

from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import (
    CanonicalCitation,
    CitationDate,
    CitationField,
    CitationKind,
    DocketEntry,
    Reporter,
    citation_kind,
    empty_citation,
    is_leaf,
)
from mellea_lrc.model.pin_cites import PinCite
from mellea_lrc.model.spans import Span

if TYPE_CHECKING:
    from mellea_lrc.model.pin_cites import PinCitePages


class Reads(str, Enum):
    """Where a node's evidence came from, which is what it may write."""

    DOCUMENT = "document"
    """The filing's own text. May update citation fields."""

    RECORD = "record"
    """An archive. May settle `found`, and never changes filing fields."""


@dataclass(frozen=True, slots=True)
class FieldUpdate:
    """One change to the filing's reading and the node that justified it."""

    field: CitationField

    before: Any
    after: Any
    reason: str
    """One line, in the words of whatever made the change."""

    node_id: str
    """A pointer to evidence rather than a copy of the node."""

    def __post_init__(self) -> None:
        if self.before == self.after:
            msg = f"An update to {self.field.value!r} must change the value"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Node:
    """One thing that was asked, and what came back.

    What it *changed* is not here. A node is evidence; the durable effects it
    justified are ordered CitationOperation events on the record. See
    `docs/Document.md`.
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
    details: dict[str, JsonValue] = field(default_factory=dict)
    """Whatever the node that made this wants to keep, carried and not read.

    Pydantic writes its JSON value tree back and nothing in `model` looks inside it,
    so a stage with typed nodes of its own -- a locator lookup with its cluster,
    a search with its candidates -- keeps its own fields without `model` knowing
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

    LOCATOR_LOOKUP = "locator_lookup"
    """What did the exact full-reporter-locator lookup return?"""

    DOCKET_LOOKUP = "docket_lookup"
    """What did the current docket-root retrieval route return?"""

    EXTRACTION_REVIEW = "extraction_review"
    """What did a source-grounded model review conclude about this extraction?"""

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

    type: str | None = None
    """How the conclusion was reached, when its evidence path needs a label."""


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

# A virtual, document-owned root. It has no source span and is never a citation
# record, so it cannot appear in locator or root-identity counts. A citation
# attached here has been taken out of the active citation graph. The head's
# own root pointer is itself by definition.
WITHDRAWN_HEAD_ID = "__withdrawn__"


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


class OperationKind(str, Enum):
    """The durable effect of a reading, independent of its pipeline stage."""

    CREATE = "create"
    FIELD_UPDATE = "field_update"
    ROOT_LINK = "root_link"
    ANTECEDENT_LINK = "antecedent_link"
    COLOCATION_LINK = "colocation_link"
    JUDGEMENT = "judgement"
    RESOLUTION = "resolution"
    AUTHORITY = "authority"
    EXTRACTION_REVIEW = "extraction_review"


@dataclass(frozen=True, slots=True)
class CreateOperation:
    """The first event, selecting the empty citation schema."""

    kind: Literal[OperationKind.CREATE]
    node_id: str
    after: CitationKind


FieldValue: TypeAlias = Span | CaseName | Reporter | PinCite | CitationDate | DocketEntry | str | None


@dataclass(frozen=True, slots=True)
class FieldOperation:
    """One field value read from the filing."""

    kind: Literal[OperationKind.FIELD_UPDATE]
    node_id: str
    after: FieldValue
    field: CitationField
    reason: str


@dataclass(frozen=True, slots=True)
class LinkOperation:
    """A current link or evidence pointer, including an explicit removal."""

    kind: Literal[
        OperationKind.ROOT_LINK,
        OperationKind.ANTECEDENT_LINK,
        OperationKind.COLOCATION_LINK,
        OperationKind.AUTHORITY,
        OperationKind.EXTRACTION_REVIEW,
    ]
    node_id: str
    after: str | None


@dataclass(frozen=True, slots=True)
class JudgementOperation:
    """One answer to a named question."""

    kind: Literal[OperationKind.JUDGEMENT]
    node_id: str
    after: Judgement
    question: Question


@dataclass(frozen=True, slots=True)
class ResolutionOperation:
    """The authority found by reading an external record."""

    kind: Literal[OperationKind.RESOLUTION]
    node_id: str
    after: Resolution


CitationOperation: TypeAlias = Annotated[
    CreateOperation | FieldOperation | LinkOperation | JudgementOperation | ResolutionOperation,
    Field(discriminator="kind"),
]


@dataclass(slots=True)
class CitationRecord:
    """One typed citation with its current fields and evidence history."""

    citation_id: str
    fields: CanonicalCitation
    """The one current reading of what the filing says. Its class fixes kind."""

    created_by: str | None = None
    """The node that classified and created this citation."""

    resolves_to: str | None = None
    root_id: str | None = None
    root_link_node_id: str | None = None
    """The evidence node for the latest root attachment, including the virtual head."""
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

    operations: tuple[CitationOperation, ...] = field(default_factory=tuple)
    """Append-only, ordered history of every durable citation-state change."""

    extraction_reviewed_by_llm: bool = False
    """Whether a grounded model has re-read this citation's stated identity fields.

    This is provenance, not a correction: ``True`` says a model completed a
    local reparse of the filing's case-name, court, date, or docket-number
    fields. The serialized validation checkpoint retains the exact model node;
    this field makes the fact directly available to later stages without trace
    traversal. A stage-specific judgement still controls whether a particular
    review may run again.
    """
    extraction_review_node_id: str | None = None
    """The successful model re-reading that set the review flag."""

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

    trace: tuple[Node, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        nodes_by_id = {node.node_id: node for node in self.trace}
        if len(nodes_by_id) != len(self.trace):
            raise ValueError(f"Citation {self.citation_id!r} has duplicate trace node identifiers")
        node_ids = set(nodes_by_id)
        if self.created_by is not None and self.created_by not in node_ids:
            msg = f"Citation {self.citation_id!r} has no creation node {self.created_by!r}"
            raise ValueError(msg)
        for operation in self.operations:
            if operation.node_id not in node_ids:
                msg = f"Citation {self.citation_id!r} has no node for operation {operation.node_id!r}"
                raise ValueError(msg)
            self._verify_operation_evidence(operation, nodes_by_id[operation.node_id])
        self._verify_operation_history()
        if self.extraction_review_node_id is not None and self.extraction_review_node_id not in node_ids:
            msg = f"Citation {self.citation_id!r} has no extraction review node"
            raise ValueError(msg)
        if self.extraction_reviewed_by_llm != (self.extraction_review_node_id is not None):
            msg = f"Citation {self.citation_id!r} has inconsistent extraction review state"
            raise ValueError(msg)
        # A kind-only leaf draft may exist between Create and its first field
        # update. Once it has a span it must have a root. The Document boundary
        # enforces that same rule before admitting any citation to a checkpoint.
        if self.root_id is None and is_leaf(self.fields) and self.fields.span is not None:
            msg = (
                f"Citation {self.citation_id!r} is a {self.fields.kind.value} and states no root. "
                "A leaf is built from an admitted root or not at all."
            )
            raise ValueError(msg)

    @property
    def kind(self) -> CitationKind:
        """The classification established at creation by the field schema."""
        return citation_kind(self.fields)

    @property
    def field_updates(self) -> tuple[FieldUpdate, ...]:
        """Field-only view, deriving previous values from the event sequence."""
        previous = empty_citation(self.kind)
        updates: list[FieldUpdate] = []
        for operation in self.operations:
            if not isinstance(operation, FieldOperation):
                continue
            before = getattr(previous, operation.field.value)
            updates.append(
                FieldUpdate(
                    field=operation.field,
                    before=before,
                    after=operation.after,
                    reason=operation.reason,
                    node_id=operation.node_id,
                )
            )
            previous = replace(previous, **{operation.field.value: operation.after})
        return tuple(updates)

    @property
    def has_complete_history(self) -> bool:
        """Whether the record can be rebuilt from its own CREATE event."""
        return (
            self.created_by is not None
            and bool(self.operations)
            and self.operations[0] == CreateOperation(OperationKind.CREATE, self.created_by, self.kind)
        )

    def get_field(self, name: CitationField) -> FieldValue:
        """Read a field's latest durable value from its operation history.

        ``fields`` is the convenient current-state projection. This method is
        the explicit history-backed read for evaluators and stages that need
        to be clear that they want the newest recorded value, rather than an
        earlier extraction artifact. A transient record without CREATE
        history falls back to its in-memory projection.
        """
        if not isinstance(name, CitationField):
            raise TypeError(f"Citation field must be a CitationField, got {name!r}")
        if not hasattr(self.fields, name.value):
            raise ValueError(f"{self.kind.value} has no field {name.value!r}")
        if not self.has_complete_history:
            return getattr(self.fields, name.value)
        for operation in reversed(self.operations):
            if isinstance(operation, FieldOperation) and operation.field is name:
                return operation.after
        return getattr(empty_citation(self.kind), name.value)

    def _verify_operation_evidence(self, operation: CitationOperation, node: Node) -> None:
        """Apply the same evidence rules to a reloaded event as to a new write."""
        if (
            operation.kind
            in {
                OperationKind.CREATE,
                OperationKind.FIELD_UPDATE,
                OperationKind.EXTRACTION_REVIEW,
            }
            and node.reads is not Reads.DOCUMENT
        ):
            raise ValueError(
                f"Citation {self.citation_id!r} {operation.kind.value} requires document evidence"
            )
        if (
            operation.kind
            in {
                OperationKind.COLOCATION_LINK,
                OperationKind.RESOLUTION,
                OperationKind.AUTHORITY,
            }
            and node.reads is not Reads.RECORD
        ):
            raise ValueError(f"Citation {self.citation_id!r} {operation.kind.value} requires record evidence")
        if isinstance(operation, ResolutionOperation) and operation.after.node_id != node.node_id:
            raise ValueError(f"Citation {self.citation_id!r} resolution names another evidence node")
        if isinstance(operation, JudgementOperation) and operation.after.node_id != node.node_id:
            raise ValueError(f"Citation {self.citation_id!r} judgement names another evidence node")
        if isinstance(operation, LinkOperation):
            if operation.kind is OperationKind.EXTRACTION_REVIEW:
                if operation.after != node.node_id:
                    raise ValueError(
                        f"Citation {self.citation_id!r} {operation.kind.value} names another node"
                    )
            if operation.kind in {OperationKind.ROOT_LINK, OperationKind.AUTHORITY} and not operation.after:
                raise ValueError(f"Citation {self.citation_id!r} {operation.kind.value} needs an identifier")
            if operation.kind is OperationKind.ANTECEDENT_LINK and operation.after == self.citation_id:
                raise ValueError(f"Citation {self.citation_id!r} cannot resolve to itself")

    def _verify_operation_history(self) -> None:
        """Ensure the saved projection is exactly the result of its events.

        Records starting with CREATE replay from empty defaults. Directly
        constructed records are transient and cannot claim complete history.
        """
        if self.created_by is None:
            if any(isinstance(event, CreateOperation) for event in self.operations):
                raise ValueError(f"Citation {self.citation_id!r} has creation event without created_by")
            return
        if not self.has_complete_history:
            raise ValueError(f"Citation {self.citation_id!r} has no valid creation operation")

        state: dict[tuple[OperationKind, CitationField | Question | None], Any] = {
            (OperationKind.ROOT_LINK, None): None,
            (OperationKind.ANTECEDENT_LINK, None): None,
            (OperationKind.COLOCATION_LINK, None): None,
            (OperationKind.AUTHORITY, None): None,
            (OperationKind.RESOLUTION, None): None,
            (OperationKind.EXTRACTION_REVIEW, None): None,
        }
        for question in Question:
            state[(OperationKind.JUDGEMENT, question)] = UNJUDGED_YET
        for field_name in CitationField:
            if hasattr(self.fields, field_name.value):
                state[(OperationKind.FIELD_UPDATE, field_name)] = getattr(
                    empty_citation(self.kind), field_name.value
                )

        for index, operation in enumerate(self.operations):
            if isinstance(operation, CreateOperation):
                if index != 0 or operation.node_id != self.created_by or operation.after != self.kind:
                    raise ValueError(f"Citation {self.citation_id!r} has invalid creation history")
                continue
            target = (
                operation.field
                if isinstance(operation, FieldOperation)
                else (operation.question if isinstance(operation, JudgementOperation) else None)
            )
            slot = (operation.kind, target)
            repeated_withdrawal = (
                operation.kind is OperationKind.ROOT_LINK
                and operation.after == WITHDRAWN_HEAD_ID
                and state.get(slot) == WITHDRAWN_HEAD_ID
            )
            if slot not in state or (state[slot] == operation.after and not repeated_withdrawal):
                raise ValueError(f"Citation {self.citation_id!r} has inconsistent operation history")
            state[slot] = operation.after

        latest_root_link_node_id = next(
            (
                operation.node_id
                for operation in reversed(self.operations)
                if isinstance(operation, LinkOperation) and operation.kind is OperationKind.ROOT_LINK
            ),
            None,
        )
        if self.root_link_node_id != latest_root_link_node_id:
            raise ValueError(
                f"Citation {self.citation_id!r} root_link_node_id state disagrees with operation history"
            )

        for (kind, target), after in state.items():
            if kind is OperationKind.FIELD_UPDATE:
                assert isinstance(target, CitationField)
                actual = getattr(self.fields, target.value)
                path = f"fields.{target.value}"
            elif kind is OperationKind.JUDGEMENT:
                assert isinstance(target, Question)
                actual = self.judgement(target)
                path = f"judgements.{target.value}.node_id"
            else:
                actual = {
                    OperationKind.ROOT_LINK: self.root_id,
                    OperationKind.ANTECEDENT_LINK: self.resolves_to,
                    OperationKind.COLOCATION_LINK: self.colocation_id,
                    OperationKind.AUTHORITY: self.authority_id,
                    OperationKind.RESOLUTION: self.found,
                    OperationKind.EXTRACTION_REVIEW: self.extraction_review_node_id,
                }[kind]
                path = {
                    OperationKind.ROOT_LINK: "root_id",
                    OperationKind.ANTECEDENT_LINK: "resolves_to",
                    OperationKind.COLOCATION_LINK: "colocation_id",
                    OperationKind.AUTHORITY: "authority_id",
                    OperationKind.RESOLUTION: "found",
                    OperationKind.EXTRACTION_REVIEW: "extraction_review_node_id",
                }[kind]
            if actual != after:
                raise ValueError(
                    f"Citation {self.citation_id!r} {path} state disagrees with operation history"
                )

    @property
    def withdrawn(self) -> bool:
        """Whether this citation is currently attached to the virtual head."""
        return self.root_id == WITHDRAWN_HEAD_ID

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
        return self.get_field(CitationField.MATCHED_TEXT) or ""

    @property
    def case_name(self) -> CaseName | None:
        """The name this citation is currently read under."""
        value = self.get_field(CitationField.CASE_NAME)
        if value is not None and not isinstance(value, CaseName):
            raise TypeError(f"{self.kind.value} case_name must be a CaseName, got {type(value).__name__}")
        return value

    @property
    def case_name_span(self) -> Span | None:
        """Where that name is written, for a reader that wants only the position."""
        name = self.case_name
        return name.span if name is not None else None

    @property
    def pin_cite_span(self) -> Span | None:
        """Where the pin cite was read from, or `None` when it states none."""
        pin_cite = getattr(self.fields, "pin_cite", None)
        return pin_cite.span if pin_cite is not None and not isinstance(pin_cite, str) else None

    @property
    def pin_cite_pages(self) -> tuple[PinCitePages, ...]:
        """Which pages the pin cite claims, or `()` when it states none."""
        pin_cite = getattr(self.fields, "pin_cite", None)
        return pin_cite.pages if pin_cite is not None and not isinstance(pin_cite, str) else ()

    def _observe(self, node: Node) -> Node:
        """Add one node to the trace, returning it so a caller can depend on it."""
        if node.node_id not in {seen.node_id for seen in self.trace}:
            self.trace = (*self.trace, node)
        return node

    def _mark_extraction_reviewed_by_llm(self, node: Node) -> None:
        """Record an admitted model reparse with a direct evidence pointer."""
        if node.reads is not Reads.DOCUMENT:
            msg = "Extraction review must read the document"
            raise ValueError(msg)
        self._observe(node)
        if self.extraction_review_node_id != node.node_id:
            self.operations = (
                *self.operations,
                LinkOperation(OperationKind.EXTRACTION_REVIEW, node.node_id, node.node_id),
            )
        self.extraction_reviewed_by_llm = True
        self.extraction_review_node_id = node.node_id

    def _update_fields(
        self,
        node: Node,
        changes: Mapping[CitationField, Any],
        *,
        reason: str,
    ) -> Node:
        """Apply zero or more document-grounded field updates atomically.

        One reader or model node may justify several changes. A no-change
        reading still stays in the trace. The citation kind cannot change.
        """
        if node.reads is not Reads.DOCUMENT:
            msg = f"Node {node.node_id!r} read a record, so it cannot update filing fields"
            raise ValueError(msg)
        updates: dict[str, Any] = {}
        history: list[CitationOperation] = []
        for name, value in changes.items():
            if not isinstance(name, CitationField):
                msg = f"Citation field must be a CitationField, got {name!r}"
                raise TypeError(msg)
            if not hasattr(self.fields, name.value):
                msg = f"{self.kind.value} has no field {name.value!r}"
                raise ValueError(msg)
            before = getattr(self.fields, name.value)
            if before == value:
                continue
            updates[name.value] = value
            history.append(
                FieldOperation(
                    kind=OperationKind.FIELD_UPDATE,
                    node_id=node.node_id,
                    after=value,
                    field=name,
                    reason=reason,
                )
            )
        revised = replace(self.fields, **updates)
        span = revised.span
        locator_span = revised.locator_span
        if (
            span is not None
            and locator_span is not None
            and (locator_span.start < span.start or locator_span.end > span.end)
        ):
            msg = f"Citation {self.citation_id!r} locator span must be within its full span"
            raise ValueError(msg)
        self.fields = revised
        self.operations = (*self.operations, *history)
        self._observe(node)
        return node

    def _update_field(self, node: Node, name: CitationField, value: Any, *, reason: str) -> Node:
        """Update one field through the same transaction as a batch reading."""
        return self._update_fields(node, {name: value}, reason=reason)

    def _judge(
        self,
        node: Node,
        question: Question,
        outcome: str,
        *,
        message: str | None = None,
        type: str | None = None,
    ) -> Node:
        """Answer one question about this citation, on the evidence of `node`.

        Answering one leaves every other question as it was, so a later stage
        adds a finding rather than replacing one.
        """
        self._observe(node)
        previous = self.judgement(question)
        revised = Judgement(outcome=outcome, node_id=node.node_id, message=message, type=type)
        if previous != revised:
            self.operations = (
                *self.operations,
                JudgementOperation(OperationKind.JUDGEMENT, node.node_id, revised, question),
            )
        self.judgements[question] = revised
        return node

    def judgement(self, question: Question) -> Judgement:
        """This citation's answer to one question, `unjudged` until something answers it."""
        return self.judgements.get(question, UNJUDGED_YET)

    def _resolve(self, node: Node, resolution: Resolution) -> Node:
        """Settle what an archive holds at this citation's identity, on the evidence of `node`."""
        if node.reads is not Reads.RECORD:
            msg = f"Node {node.node_id!r} read the filing, so it cannot settle what an archive holds"
            raise ValueError(msg)
        if resolution.node_id != node.node_id:
            msg = f"Resolution names node {resolution.node_id!r}, not {node.node_id!r}"
            raise ValueError(msg)
        self._observe(node)
        if self.found != resolution:
            self.operations = (
                *self.operations,
                ResolutionOperation(OperationKind.RESOLUTION, node.node_id, resolution),
            )
        self.found = resolution
        return node

    def _reattribute(self, node: Node, authority_id: str) -> Node:
        """Settle which authority this root reaches, on the evidence of `node`.

        `root_id` is what the filing stated and stays; this writes what a lookup
        found, which is `authority_id`.
        """
        self._observe(node)
        if self.authority_id != authority_id:
            self.operations = (
                *self.operations,
                LinkOperation(OperationKind.AUTHORITY, node.node_id, authority_id),
            )
        self.authority_id = authority_id
        return node

    def _withdraw(self, node: Node) -> Node:
        """Reattach this citation to the virtual head, retaining prior links."""
        self._observe(node)
        if not (self.withdrawn and self.root_link_node_id == node.node_id):
            self.operations = (
                *self.operations,
                LinkOperation(OperationKind.ROOT_LINK, node.node_id, WITHDRAWN_HEAD_ID),
            )
        self.root_id = WITHDRAWN_HEAD_ID
        self.root_link_node_id = node.node_id
        return node

    def _span(self, name: str) -> Span:
        span = getattr(self.fields, name)
        if span is None:
            msg = f"Citation {self.citation_id!r} was read without a {name}"
            raise ValueError(msg)
        return span
