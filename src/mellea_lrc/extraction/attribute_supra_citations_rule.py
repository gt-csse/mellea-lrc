"""Attribute supra citations by source name; route unresolved sites to review."""

from mellea_lrc.extraction.context.leaves import name_candidates, require_leaves
from mellea_lrc.model.citations import AttributionResult, SupraCitation
from mellea_lrc.model.citations.fields.base import current_reading
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.grow_supra_leaves.attribute_citations_rule"
REVIEW_SUBSTAGE = "grow_leaves.grow_supra_leaves.review_attributions_llm"


def attribute_supra_citations_rule(document: Document) -> Document:
    require_leaves(document, SUBSTAGE)
    for citation in document.short_citations:
        if not isinstance(citation, SupraCitation):
            continue
        named: tuple[str, ...] = ()
        if (name := current_reading(citation.case_name)) is not None:
            named = name_candidates(document, name.quote, citation.site_span.start)
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
