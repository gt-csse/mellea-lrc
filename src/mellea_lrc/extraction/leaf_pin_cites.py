"""Read leaf pinpoint targets without making an identity or attribution decision."""

import re

from mellea_lrc.extraction.id_citations import id_pin_span
from mellea_lrc.extraction.leaf_reading import pin_after, require_leaves
from mellea_lrc.model.citations import IdCitation, ReferenceCitation
from mellea_lrc.model.document import Document

STAGE = "33_leaf_pin_cites"


def resolve_leaf_pin_cites(document: Document) -> Document:
    require_leaves(document, STAGE)
    for citation in document.short_citations:
        site = citation.site_span
        end = min(len(document.text), site.end + 100)
        end = min(
            end,
            min(
                (c.site_span.start for c in document.citations if c.site_span.start >= site.end), default=end
            ),
        )
        text = document.text[site.start : end]
        if isinstance(citation, IdCitation):
            keyword = re.match(r"(?:id\.|ibid\.)", text, re.I)
            span = id_pin_span(document.text, site.start + keyword.end(), end) if keyword else None
            if span:
                document = document.replace_citation(
                    citation.record(STAGE).with_pin_cite(document.text, span)
                )
            continue
        elif isinstance(citation, ReferenceCitation):
            # A name reference's pin must directly follow its name; scanning
            # arbitrary prose for an 'at' would borrow unrelated numbers.
            start = site.end - site.start
        else:
            marker = re.search(r"\bat\s+", text, re.I)
            start = marker.end() if marker else None
        if start is not None and (span := pin_after(document.text, site.start + start, end)):
            document = document.replace_citation(citation.record(STAGE).with_pin_cite(document.text, span))
    return document.complete(STAGE)
