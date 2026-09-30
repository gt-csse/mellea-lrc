"""Propose quoted references from previously introduced case names.

Name occurrence alone does not prove citation use. These deliberately lenient
sites all require the leaf model review, even with a unique name candidate.
"""

import re

from eyecite.utils import is_valid_name

from mellea_lrc.extraction.leaf_reading import aliases, preceding_name, require_leaves
from mellea_lrc.matching.literal import fuzzy_literal
from mellea_lrc.model.citations import ReferenceCitation
from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "31_reference_citations"


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
            if not any(span.overlaps(part) for part in blocked):
                candidates.add(match.span())
    accepted: list[Span] = []
    for start, end in sorted(candidates, key=lambda pair: (-(pair[1] - pair[0]), pair[0])):
        span = Span(start, end)
        if not any(span.overlaps(part) for part in accepted):
            accepted.append(span)
    for span in sorted(accepted, key=lambda part: part.start):
        document = document.add_citation(
            ReferenceCitation.from_source(source=document.text, span=span, stage=STAGE)
        )
    return document.complete(STAGE)
