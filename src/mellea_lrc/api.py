"""Outer compositional API for preprocessing, extraction, and validation.

Callers may compose each `Document -> Document` stage explicitly; docket
hunting and docket-root equivalence review are awaitable. `grow_roots` is the
async convenience composition, with both model stages opt-in.
Validation stages remain independently callable. `grow_leaves` attaches short
forms and repeated occurrences to formed roots, with optional model review.
The workflows are `grow_roots`, `validate_roots`, `grow_leaves`, and `validate_pincite`.
Locator-body and intended-case review are parts of root validation.
"""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction import (
    attribute_reference_citations,
    attribute_short_reporter_citations,
    docket_root_llm_reassignment,
    find_docket_locators,
    find_full_reporter_locators,
    find_short_reporter_citations,
    form_roots,
    hunt_docket_locators,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_docket_entries,
    resolve_pin_cites,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
)
from mellea_lrc.extraction.id_attribution import attribute_id_citations
from mellea_lrc.extraction.id_citations import find_id_citations
from mellea_lrc.extraction.reference_citations import find_reference_citations
from mellea_lrc.extraction.supra_attribution_llm import review_supra_attributions
from mellea_lrc.extraction.supra_attribution_rule import attribute_supra_citations_rule
from mellea_lrc.extraction.supra_case_names import resolve_supra_case_names
from mellea_lrc.extraction.supra_citations import find_supra_citations
from mellea_lrc.extraction.supra_pin_cites import resolve_supra_pin_cites
from mellea_lrc.model.document import Document
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation import (
    docket_root_lookup_courtlistener_llm_review,
    docket_root_lookup_courtlistener_retrieval,
    docket_root_lookup_govinfo_llm_review,
    docket_root_lookup_govinfo_retrieval,
    index_reporter_root_opinion_pages,
    intended_case_courtlistener_opinion_retrieval,
    intended_case_courtlistener_recap_retrieval,
    intended_case_govinfo_opinion_retrieval,
    intended_case_llm_selection,
    judge_reporter_citation_pinpoints,
    locator_body_courtlistener_opinion_retrieval,
    locator_body_courtlistener_recap_retrieval,
    locator_body_govinfo_opinion_retrieval,
    locator_body_llm_judgment,
    prepare_reporter_citation_pinpoint_evidence,
    read_reporter_citation_propositions,
    reporter_root_lookup_ambiguous_llm_judgment,
    reporter_root_lookup_ambiguous_rule_judgment,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_lookup_docket_retrieval,
    reporter_root_lookup_unique_llm_judgment,
    reporter_root_lookup_unique_rule_judgment,
    reporter_root_opinion_retrieval,
    resolve_reporter_citation_pages,
    review_reporter_citation_full_opinions,
    review_reporter_citation_opinions,
    review_reporter_citation_pinpoint_pages,
)
from mellea_lrc.workflows import grow_leaves, grow_roots, validate_pincite, validate_roots

__all__ = [
    "Document",
    "ExtractionRules",
    "attribute_id_citations",
    "attribute_reference_citations",
    "attribute_short_reporter_citations",
    "attribute_supra_citations_rule",
    "docket_root_llm_reassignment",
    "docket_root_lookup_courtlistener_llm_review",
    "docket_root_lookup_courtlistener_retrieval",
    "docket_root_lookup_govinfo_llm_review",
    "docket_root_lookup_govinfo_retrieval",
    "find_docket_locators",
    "find_full_reporter_locators",
    "find_id_citations",
    "find_reference_citations",
    "find_short_reporter_citations",
    "find_supra_citations",
    "form_roots",
    "grow_leaves",
    "grow_roots",
    "hunt_docket_locators",
    "index_reporter_root_opinion_pages",
    "intended_case_courtlistener_opinion_retrieval",
    "intended_case_courtlistener_recap_retrieval",
    "intended_case_govinfo_opinion_retrieval",
    "intended_case_llm_selection",
    "judge_reporter_citation_pinpoints",
    "locator_body_courtlistener_opinion_retrieval",
    "locator_body_courtlistener_recap_retrieval",
    "locator_body_govinfo_opinion_retrieval",
    "locator_body_llm_judgment",
    "prepare_reporter_citation_pinpoint_evidence",
    "preprocess",
    "read_reporter_citation_propositions",
    "reporter_root_lookup_ambiguous_llm_judgment",
    "reporter_root_lookup_ambiguous_rule_judgment",
    "reporter_root_lookup_cluster_retrieval",
    "reporter_root_lookup_docket_retrieval",
    "reporter_root_lookup_unique_llm_judgment",
    "reporter_root_lookup_unique_rule_judgment",
    "reporter_root_opinion_retrieval",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_docket_entries",
    "resolve_pin_cites",
    "resolve_reporter_citation_pages",
    "resolve_short_reporter_case_names",
    "resolve_short_reporter_colocations",
    "resolve_supra_case_names",
    "resolve_supra_pin_cites",
    "review_reporter_citation_full_opinions",
    "review_reporter_citation_opinions",
    "review_reporter_citation_pinpoint_pages",
    "review_supra_attributions",
    "stable",
    "validate_pincite",
    "validate_roots",
]
