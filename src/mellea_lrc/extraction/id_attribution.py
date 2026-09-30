"""Resolve Id chains in source order, retaining intervening noncase barriers."""

from mellea_lrc.extraction.leaf_reading import events, require_leaves
from mellea_lrc.model.citations import AttributionResult, IdCitation, ReferenceCitation, latest
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.document import Document

STAGE = "36_id_attribution"


def attribute_id_citations(document: Document) -> Document:
    require_leaves(document, STAGE)
    # Do not filter this stream to known cases: statute/journal/unknown events
    # break a case antecedent. Root identity verdicts and pin page ranges do
    # not change which authority a subsequent Id. refers to.
    stream = [
        (c.site_span.start, c.site_span.end, c)
        for c in document.citations
        if not (
            isinstance(c, ReferenceCitation)
            and c.reviews
            and c.reviews[-1].decision
            and not c.reviews[-1].decision.is_citation
        )
    ]
    for event in events(document.text):
        span = event.span()
        if any(c.site_span.start <= span[0] < c.site_span.end for c in document.citations):
            continue
        stream.append((span[0], span[1], None))
    prior: str | None = None
    roots = {root.id for root in document.roots}
    for _, _, citation in sorted(stream, key=lambda event: (event[0], event[1])):
        if citation is None:
            prior = None
            continue
        if isinstance(citation, IdCitation):
            updated = citation.record(STAGE)
            candidates = (prior,) if prior in roots else ()
            if candidates:
                updated = updated.with_attribution(
                    candidates, AttributionResult.ATTACHED, "Immediately preceding resolved case citation"
                ).with_root(prior)
            else:
                updated = updated.with_attribution(
                    (), AttributionResult.UNRESOLVED, "No resolved case antecedent in citation chronology"
                )
            document = document.replace_citation(updated)
            citation = updated
        root = latest(citation.root_id)
        prior = root if root in roots and root != WITHDRAWN_ROOT_ID else None
    return document.complete(STAGE)
