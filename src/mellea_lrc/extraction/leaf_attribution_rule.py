"""Rule-only short-form attribution; ambiguous choices route to model review."""

from mellea_lrc.extraction.leaf_reading import name_candidates, reporter_candidates, require_leaves
from mellea_lrc.model.citations import AttributionResult, IdCitation, ReferenceCitation, ShortReporterCitation
from mellea_lrc.model.document import Document

STAGE = "34_leaf_attribution_rule"
REVIEW_STAGE = "35_leaf_attribution_llm"


def attribute_leaves_rule(document: Document) -> Document:
    require_leaves(document, STAGE)
    for citation in document.short_citations:
        if isinstance(citation, IdCitation):
            continue
        candidates: tuple[str, ...] = ()
        if isinstance(citation, ShortReporterCitation) and citation.short_locator[-1].normalizable:
            locator = citation.short_locator[-1].get_normalized()
            candidates = reporter_candidates(
                document, locator.volume, locator.edition, citation.site_span.start
            )
        named: tuple[str, ...] = ()
        if citation.case_name:
            # A name-only reference may introduce a nearby full citation that
            # follows it. All such attachments still need semantic review.
            before = None if isinstance(citation, ReferenceCitation) else citation.site_span.start
            named = name_candidates(document, citation.case_name[-1].quote, before)
        if not isinstance(citation, ShortReporterCitation):
            candidates = named
        passing = tuple(root for root in candidates if not citation.case_name or root in named)
        # Name-only mentions need semantic review even when their name matches
        # exactly: a party mentioned in prose is not necessarily a case citation.
        attach = len(passing) == 1 and not isinstance(citation, ReferenceCitation)
        result = AttributionResult.ATTACHED if attach else AttributionResult.UNRESOLVED
        updated = citation.record(STAGE).with_attribution(
            candidates,
            result,
            "Unique preceding locator/name agreement" if attach else "Semantic review required",
        )
        if attach:
            updated = updated.with_root(passing[0])
        else:
            updated = updated.with_route(REVIEW_STAGE)
        document = document.replace_citation(updated)
    return document.complete(STAGE)
