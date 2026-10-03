"""Source-ordered citation view with noncase interruption boundaries."""

from mellea_lrc.model.citations import CitationVariant, ReferenceCitation
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.events import events


def citation_chronology(document: Document) -> tuple[tuple[int, int, CitationVariant | None], ...]:
    """Retain every authority boundary used for Id. attachment and inheritance.

    Rejected name references do not become antecedents. Recognized authorities
    without a citation object interrupt the stream; their None entry prevents
    attribution or page inheritance from jumping across a noncase authority.
    """
    stream = [
        (citation.site_span.start, citation.site_span.end, citation)
        for citation in document.citations
        if not (
            isinstance(citation, ReferenceCitation)
            and citation.reviews
            and citation.reviews[-1].decision
            and not citation.reviews[-1].decision.is_citation
        )
    ]
    for event in events(document.text):
        start, end = event.span()
        if not any(
            citation.site_span.start <= start < citation.site_span.end for citation in document.citations
        ):
            stream.append((start, end, None))
    return tuple(sorted(stream, key=lambda item: (item[0], item[1])))
