"""Find quoted case-name references with an explicit adjacent pinpoint.

Full citations supply names; relaxed source matching and the shared pinpoint
reader augment eyecite's name-plus-pin shape. A bare name is outside this
stage's scope. Creation retains the case name and pinpoint, while attribution
remains an independent stage.
"""

import re

from eyecite.utils import is_valid_name

from mellea_lrc.extraction.context.leaves import (
    aliases,
    preceding_name,
    reference_pin_after,
    require_leaves,
)
from mellea_lrc.matching.literal import fuzzy_literal
from mellea_lrc.model.citations import ReferenceCitation
from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "30_reference_citations"


def find_reference_citations(document: Document) -> Document:
    require_leaves(document, STAGE)
    blocked = [c.site_span for c in document.citations]
    # A name, court, date, entry or pin already read as part of a full citation
    # is not a separate leaf. Include every source-grounded field history.
    for citation in document.full_locators:
        for name in type(citation).model_fields:
            value = getattr(citation, name)
            if isinstance(value, tuple):
                blocked.extend(
                    field.span
                    for field in value
                    if isinstance(field, CitationField) and field.span is not None
                )
        # A full-citation name can be incomplete in the saved field reading;
        # the same adjacent source-name reader still prevents constituent
        # party prefixes from being proposed as independent references.
        if span := preceding_name(document, citation.site_span):
            blocked.append(span)
    blocked.extend(span for c in document.short_reporters if (span := preceding_name(document, c.site_span)))
    candidates: set[tuple[int, int]] = set()
    for alias in aliases(document):
        # Use eyecite's general name eligibility guard, not a local stopword list.
        # Upstream's guard rejects every name ending in a period (including
        # complete corporate names). It is a bare-name guard, not a ban on
        # punctuation inside our already escaped, source-derived aliases.
        if not is_valid_name(alias.rstrip(".")):
            continue
        pattern = rf"(?<!\w){fuzzy_literal(alias, whitespace=True, newline=True)}(?!\w)"
        for match in re.finditer(pattern, document.text):
            span = Span(*match.span())
            pin = reference_pin_after(document.text, span.end, min(len(document.text), span.end + 100))
            if pin is None:
                continue
            # An apparent name+pin inside a full/short citation is a component
            # of that citation, not an additional reference leaf.
            complete_span = Span(span.start, pin.end)
            if not any(complete_span.overlaps(part) for part in blocked):
                candidates.add(match.span())
    accepted: list[Span] = []
    for start, end in sorted(candidates, key=lambda pair: (-(pair[1] - pair[0]), pair[0])):
        span = Span(start, end)
        if not any(span.overlaps(part) for part in accepted):
            accepted.append(span)
    for span in sorted(accepted, key=lambda part: part.start):
        pin = reference_pin_after(document.text, span.end, min(len(document.text), span.end + 100))
        if pin is None:
            raise ValueError(f"Reference citation has no adjacent pinpoint: {span}")
        document = document.add_citation(
            ReferenceCitation.from_source(source=document.text, span=span, pin_span=pin, stage=STAGE)
        )
    return document.complete(STAGE)
