"""Read only the written names of short reporter, supra and reference sites."""

import re

from mellea_lrc.extraction.leaf_reading import preceding_name, require_leaves
from mellea_lrc.model.citations import ReferenceCitation, ShortReporterCitation, SupraCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "32_leaf_case_names"


def resolve_leaf_case_names(document: Document) -> Document:
    require_leaves(document, STAGE)
    for citation in document.short_citations:
        span = None
        if isinstance(citation, ReferenceCitation):
            span = citation.site_span
        elif isinstance(citation, ShortReporterCitation):
            span = preceding_name(document, citation.site_span)
        elif isinstance(citation, SupraCitation):
            match = re.search(
                r"\bsupra\b", document.text[citation.site_span.start : citation.site_span.end], re.I
            )
            if match and match.start():
                quote = document.text[
                    citation.site_span.start : citation.site_span.start + match.start()
                ].rstrip(" ,")
                if quote:
                    span = Span(citation.site_span.start, citation.site_span.start + len(quote))
        if span is not None:
            document = document.replace_citation(citation.record(STAGE).with_case_name(document.text, span))
    return document.complete(STAGE)
