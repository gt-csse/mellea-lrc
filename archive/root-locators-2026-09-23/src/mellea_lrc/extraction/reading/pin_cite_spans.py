r"""Where in the document a citation's pin cite was read from.

eyecite returns the pin cite as a **string** and, for one citation kind out of
six, an end offset. Measured over the 26 documents of `false-citation-bench`:
386 full case citations carry ``pin_cite_span_end``, and the other 120 pin cites
-- every short case, `Id.`, reference, statute and journal one -- carry nothing.
``pin_cite_span_start`` is never set at all; the one place `helpers.py` assigns
it is the *pre*-citation path, and the value it puts there is the start of the
whole pre-citation match, antecedent name included, rather than the start of the
pin cite.

So a pin cite that can be pointed at in the document has to be located here.

## Located, not re-read

This finds the string the parse produced, inside the span the parse produced.
It cannot find a pin cite eyecite did not find, and it cannot find one outside
the citation's own extent -- which is what separates it from the ground truth's
own reader, which scans raw text with a pattern of its own and *can* see pin
cites the parse missed. That independence is why the two must not share an
implementation: a ground truth that agreed with extraction by construction would
measure nothing.

## Two geometries

Which end of the citation the pin cite sits at is a property of the kind, and
eyecite is consistent about it:

``556 U.S. 662, 678``
    A full case, statute or journal citation ends at its page, and the pin cite
    follows -- inside ``full_span``, outside the locator.

``695 F.Supp.2d at 1154``, ``Id. at 547``, ``Bell at 546``, ``supra, at 15``
    A short form's page is *part of its identifier*: eyecite's `short_cite_re`
    ends on the page group, and the `Id.`, reference and supra patterns end on
    the pin cite. So the pin cite is the tail of the locator span.

Both are found by exact search. The strings eyecite returns keep the document's
own spacing -- `clean_pin_cite` strips only leading and trailing commas and
spaces -- so no whitespace tolerance is needed and none is used: all 506 pin
cites on the bench are located exactly, and a pin cite that cannot be found is
reported as ``None`` rather than approximated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.model.citations import CitationField, CitationKind
from mellea_lrc.model.operations import update_field
from mellea_lrc.model.pin_cites import PinCite
from mellea_lrc.model.record import Node, Reads
from mellea_lrc.model.spans import Span

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.model.citations import CanonicalCitation
    from mellea_lrc.model.record import CitationRecord

_TAIL_OF_LOCATOR = frozenset(
    {
        CitationKind.SHORT_CASE,
        CitationKind.ID,
        CitationKind.SUPRA,
        CitationKind.REFERENCE,
    }
)


def locate_pin_cite(
    text: str,
    citation: CanonicalCitation,
    *,
    locator_span: Span,
    full_span: Span,
) -> Span | None:
    """The span of the pin cite ``citation`` states, in ``text``'s coordinates.

    ``None`` when the citation states no pin cite, or when the string eyecite
    returned is not present where its kind puts it.

    Both spans must already index ``text``. Callers that parsed a window pass
    the spans they have offset back to the document, so this never has to know
    about windows -- and callers that parsed *repaired* text pass the mapped-back
    spans, so nothing here indexes coordinates that no longer exist.
    """
    pin_cite = getattr(citation, "pin_cite", None)
    if not pin_cite:
        return None

    if citation.kind in _TAIL_OF_LOCATOR:
        locator = text[locator_span.start : locator_span.end]
        if not locator.endswith(pin_cite):
            return None
        start = locator_span.end - len(pin_cite)
        return Span(start=start, end=locator_span.end)

    base = locator_span.end
    after = text[base : max(full_span.end, base)]
    offset = after.find(pin_cite)
    if offset < 0:
        return None
    return Span(start=base + offset, end=base + offset + len(pin_cite))


def read_pin_cites(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Parse pin-cite strings after citation spans have been bounded.

    Eyecite supplies the pin-cite string during ``get_citations``. This pass
    locates that exact string against the finalized locator/full spans and
    builds the structured ``PinCite`` value used by extraction and validation.
    """

    rebuilt: list[CitationRecord] = []
    for item in citations:
        if not hasattr(item.fields, "pin_cite"):
            rebuilt.append(item)
            continue
        parsed = read_pin_cite(text, item.fields)
        if parsed.pin_cite == item.fields.pin_cite:
            rebuilt.append(item)
            continue
        revised = replace(item)
        update_field(
            revised,
            Node(
                node_id=f"pin_cite_read:{item.citation_id}",
                reads=Reads.DOCUMENT,
                stage="pin_cite_read",
                made_by=__name__,
                outcome="structured",
            ),
            CitationField.PIN_CITE,
            parsed.pin_cite,
            reason="Located and structured the pin cite in document text.",
        )
        rebuilt.append(revised)
    return tuple(rebuilt)


def read_pin_cite(text: str, citation: CanonicalCitation) -> CanonicalCitation:
    """Structure one parsed pin string against its locator and full spans."""
    written = getattr(citation, "pin_cite", None)
    if not isinstance(written, str) or not written:
        return citation
    locator_span = getattr(citation, "locator_span", None)
    full_span = getattr(citation, "span", None)
    pin_span = (
        locate_pin_cite(text, citation, locator_span=locator_span, full_span=full_span)
        if locator_span is not None and full_span is not None
        else None
    )
    return replace(citation, pin_cite=PinCite.read(written, pin_span))
