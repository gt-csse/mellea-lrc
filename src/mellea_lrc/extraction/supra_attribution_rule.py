"""Attribute supra citations by source name; route unresolved sites to review."""

from mellea_lrc.extraction.context.leaves import name_candidates, require_leaves
from mellea_lrc.model.citations import AttributionResult, SupraCitation
from mellea_lrc.model.document import Document

STAGE = "37_supra_attribution_rule"
REVIEW_STAGE = "38_supra_attribution_llm"


def attribute_supra_citations_rule(document: Document) -> Document:
    require_leaves(document, STAGE)
    for citation in document.short_citations:
        if not isinstance(citation, SupraCitation):
            continue
        named: tuple[str, ...] = ()
        if citation.case_name:
            named = name_candidates(document, citation.case_name[-1].quote, citation.site_span.start)
        candidates = named
        attach = len(candidates) == 1
        result = AttributionResult.ATTACHED if attach else AttributionResult.UNRESOLVED
        updated = citation.record(STAGE).with_attribution(
            candidates,
            result,
            "Unique preceding locator/name agreement" if attach else "Semantic review required",
        )
        if attach:
            updated = updated.with_root(candidates[0])
        else:
            updated = updated.withdraw().with_route(REVIEW_STAGE)
        document = document.replace_citation(updated)
    return document.complete(STAGE)
