"""One way in to the model layer: a document, the reviews to run, the record back.

    from mellea_lrc.extraction.adjudication import Review, adjudicate

    document = await adjudicate(document, session=session)                     # none
    document = await adjudicate(document, [Review.PIN_CITE], session=session)

`reviews` is the whole of the configuration, the way `rules` is for
:func:`mellea_lrc.preprocessing.preprocess`. The one difference is the default:
preprocessing runs every rule because none of them costs anything, and **this
defaults to none**, because every review is a model call. Nothing here happens
unless a caller asks for it by name.

## A review is a record change

Each one answers a question the rules leave and writes the answer onto the
record it concerns: a :class:`~mellea_lrc.core.record.Node` in the trace, and a
:class:`~mellea_lrc.core.record.Correction` inside that node where a field
moved. `Reads.DOCUMENT`, because a reviewer is shown the characters and may
therefore correct `stated`; a stage that reads only the record may not, and
`core.record` refuses it.

A review may **create** a record as well as correct one. A bare name states no
identifier, so no pattern could have reached it, and a citation eyecite refused
to build is a citation the filing wrote. What a review may not do is invent a
span: everything it records points at characters the document holds.

## Relaxation is not one of these

`Relaxation` is deterministic, free, and a property of the extraction that
already ran. It is not a review and does not belong in this list -- offering
both through one switch would let a caller ask for a model call by naming a
whitespace rule.

Where a relaxation is the *reason* a review is needed, the review owns it:
`Review.requires` names the level the review assumes, and `adjudicate` refuses a
document read at a lower one rather than answering about text the rules never
saw. Relaxing the pin-cite stop is what produces the pages `PIN_CITE` is asked
about, so asking for that review is asking for that relaxation.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import ReferenceCitation
from mellea_lrc.core.record import CitationRecord, Node, Reads
from mellea_lrc.extraction.adjudication.candidates.case_name_sites import case_name_sites
from mellea_lrc.extraction.adjudication.candidates.pin_cite_sites import pin_cite_sites
from mellea_lrc.extraction.adjudication.review.case_name import Reading, adjudicate_case_name
from mellea_lrc.extraction.adjudication.review.pin_cite import adjudicate_pin_cite
from mellea_lrc.extraction.reading.relaxation import Relaxation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea import MelleaSession

    from mellea_lrc.extraction.types import Document


class Review(str, Enum):
    """One question the rules leave, put to a reader, answered onto the record.

    Each names a question that has no deterministic answer -- not a rule that
    could have been written and was not. A rule that a corpus can show
    generalises belongs in :mod:`mellea_lrc.extraction.reading`; this is for the
    ones where the evidence is two citations and the judgement is a reading.
    """

    PIN_CITE = "pin_cite"
    """What page a citation actually claims.

    Covers both halves of the same question. A pin cite the rules read but cut
    short -- a stop word after the page, a range the converter broke -- and a
    citation they gave no pin cite at all, including a reference citation
    eyecite declined to build. The first is a correction to an existing record,
    the second creates one.
    """

    CASE_NAME = "case_name"
    """What a case name standing outside every citation is.

    A name the rules did not reach, on a citation they did read, or a bare name
    that is a citation of its own under Bluebook Rule 10.9.

    **This runs after validation, not before it.** Both halves of it turn on a
    name: whether the name beside a citation is that citation's, and which root
    a bare name reads back to. The names in the record at this point are
    whatever eyecite's parser made of them -- half a name where an apostrophe
    was spaced out, a suffix where the party was lost -- so every one of those
    questions is asked against a guess.

    Validation resolves each root against the archives and the record then holds
    the authority's real name. The plan is that the record comes back to
    extraction at that point and this review runs on it, where both the sweep
    for bare names and eyecite's own attribution have something true to match
    against. Declared here so the interface is whole; not in any arm.
    """

    @property
    def requires(self) -> Relaxation:
        """The relaxation this review assumes was already applied."""
        return _REQUIRES[self]


_REQUIRES: dict[Review, Relaxation] = {
    Review.PIN_CITE: Relaxation.FULL,
    Review.CASE_NAME: Relaxation.BOUNDED,
}

DEFAULT_REVIEWS: tuple[Review, ...] = ()
"""None of them. Every review is a model call, so the deterministic reading is
what a caller gets for saying nothing."""

_ORDER: tuple[Relaxation, ...] = (Relaxation.NONE, Relaxation.BOUNDED, Relaxation.FULL)

#: What made a correction, on the node that carries it.
ADJUDICATE_CASE_NAME = "adjudicate_case_name"
ADJUDICATE_PIN_CITE = "adjudicate_pin_cite"
#: A reader that could not answer. Recorded, because a site that was looked at
#: and left alone is not a site nobody looked at.
DECLINED = "declined"


async def adjudicate(
    document: Document,
    reviews: Sequence[Review] | None = None,
    *,
    session: MelleaSession,
) -> Document:
    """Run the named reviews over a document, in the order they are given.

    **One at a time, and deliberately.** The model underneath is asynchronous
    and several sites could be in flight together, but a review writes to the
    record: it corrects a field, and the next site is shown the corrected one.
    `adjudicate_case_name` is passed the records so it can say which citation a
    name belongs to, and two sites answering about the same citation at once
    would each decide against a record the other is changing. Nothing here is
    fast, and the cost of getting it wrong is a record that cannot be replayed,
    so the loop stays sequential until a review declares itself independent.
    """
    asked = DEFAULT_REVIEWS if reviews is None else tuple(reviews)
    read_at = document.extraction_metadata.relaxation
    for review in asked:
        if _ORDER.index(read_at) < _ORDER.index(review.requires):
            msg = (
                f"{review.value} assumes {review.requires.value} relaxation and this document "
                f"was read at {read_at.value}; extract it again before asking for the review"
            )
            raise ValueError(msg)
    for review in asked:
        await _RUNNERS[review](document, session)
    return document


async def _case_name(document: Document, session: MelleaSession) -> None:
    """Apply what a reader makes of each case name standing outside every citation.

    `short_form` is a citation the rules did not read at all, so it becomes a
    record of its own. `names_a_citation` corrects the case name of the record
    it names, carrying the node that justified it, and the next site is shown
    the corrected name. The other two readings are findings about the filing
    rather than citations.
    """
    records = {record.citation_id: record for record in document.citations}
    at = {record.citation_id: record.locator_span for record in document.citations}
    recovered: list[CitationRecord] = []
    for site in case_name_sites(document):
        answer = await adjudicate_case_name(document, site, session=session, records=records)
        if answer is None:
            continue
        node = Node(
            node_id=f"case_name:{site.span.start}-{site.span.end}",
            reads=Reads.DOCUMENT,
            stage=Review.CASE_NAME.value,
            made_by=ADJUDICATE_CASE_NAME,
            outcome=answer.reading.value,
            message=answer.reason or None,
        )
        name = CaseName(
            span=answer.span,
            text=answer.name,
            plaintiff=answer.plaintiff,
            defendant=answer.defendant,
        )
        if answer.reading is Reading.NAMES_A_CITATION:
            named = records.get(answer.citation_id or "")
            # The review sweeps what the rules left over, so it can land on a
            # citation that already holds the name it read. Nothing to correct.
            if named is not None and named.case_name != name:
                named.correct(node, "case_name", name, reason=node.message or "")
            continue
        if answer.reading is not Reading.SHORT_FORM or answer.root_id not in at:
            continue
        # A bare name is a citation the record does not hold: it states no
        # identifier, so nothing the rules read could have reached it.
        reference = CitationRecord(
            citation_id=f"case_name:{site.span.start}",
            source=ReferenceCitation(
                span=answer.span,
                locator_span=answer.span,
                matched_text=answer.name,
                case_name=name,
                plaintiff=answer.plaintiff,
                defendant=answer.defendant,
            ),
            root_id=answer.root_id,
        )
        reference.observe(node)
        recovered.append(reference)
    if recovered:
        object.__setattr__(document, "citations", (*document.citations, *recovered))


async def _pin_cite(document: Document, session: MelleaSession) -> None:
    """Apply what a reader makes of each page claim the rules are not sure of.

    Every site names a citation already in the record, so every answer is a
    correction to one field of it. A reading of `states_no_page` is a
    correction too, where the rules read a page and the reader says the
    characters are not one -- `None` over a value is a change and is recorded.

    A site the reader declines, or answers the same way the rules already did,
    leaves a node with no correction on it: the record then says that the page
    was looked at and stands, which is not the same as never having been asked.
    """
    records = {record.citation_id: record for record in document.citations}
    for site in pin_cite_sites(document):
        record = records.get(site.about or "")
        if record is None:
            continue
        answer = await adjudicate_pin_cite(document.text, site, record, session=session)
        node = Node(
            node_id=f"pin_cite:{site.span.start}-{site.span.end}",
            reads=Reads.DOCUMENT,
            stage=Review.PIN_CITE.value,
            made_by=ADJUDICATE_PIN_CITE,
            outcome=answer.reading.value if answer is not None else DECLINED,
            message=(answer.reason if answer is not None else site.note) or None,
        )
        if answer is None or answer.pin_cite == record.stated.pin_cite:
            record.observe(node)
            continue
        record.correct(node, "pin_cite", answer.pin_cite, reason=node.message or "")


_RUNNERS = {
    Review.CASE_NAME: _case_name,
    Review.PIN_CITE: _pin_cite,
}
