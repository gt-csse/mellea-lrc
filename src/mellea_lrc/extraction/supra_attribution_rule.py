"""Attribute supra citations by source name; route unresolved sites to review."""

from mellea_lrc.extraction.context.leaves import name_candidates, require_leaves
from mellea_lrc.model.citations import AttributionResult, SupraCitation
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.supra_citations.rule_attribution"
REVIEW_SUBSTAGE = "grow_leaves.supra_citations.llm_attribution"


def attribute_supra_citations_rule(document: Document) -> Document:
    require_leaves(document, SUBSTAGE)
    for citation in document.short_citations:
        if not isinstance(citation, SupraCitation):
            continue
        named: tuple[str, ...] = ()
        if citation.case_name:
            named = name_candidates(document, citation.case_name[-1].quote, citation.site_span.start)
        candidates = named
        attach = len(candidates) == 1
        result = AttributionResult.ATTACHED if attach else AttributionResult.UNRESOLVED
        updated = citation.record(SUBSTAGE).with_attribution(
            candidates,
            result,
            "Unique preceding locator/name agreement" if attach else "Semantic review required",
        )
        if attach:
            updated = updated.with_root(candidates[0])
        else:
            updated = updated.withdraw().with_route(REVIEW_SUBSTAGE)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
