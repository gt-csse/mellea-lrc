"""Outer compositional API for preprocessing, extraction, and validation.

Callers may compose each `Document -> Document` stage explicitly; docket
hunting and docket-root equivalence review are awaitable. `grow_roots` is the
async convenience composition, with both model stages opt-in.
Validation stages remain independently callable. `grow_leaves` attaches short
forms and repeated occurrences to formed roots, with optional model review.
The only workflows are `grow_roots`, `validate_roots`, and `grow_leaves`.
Locator-body and intended-case review are parts of root validation.
"""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction import (
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
)
from mellea_lrc.extraction.id_attribution import attribute_id_citations
from mellea_lrc.extraction.id_attribution_llm import review_id_attributions
from mellea_lrc.extraction.id_citations import find_id_citations
from mellea_lrc.extraction.leaf_attribution_llm import review_leaf_attributions
from mellea_lrc.extraction.leaf_attribution_rule import attribute_leaves_rule
from mellea_lrc.extraction.leaf_case_names import resolve_leaf_case_names
from mellea_lrc.extraction.leaf_pin_cites import resolve_leaf_pin_cites
from mellea_lrc.extraction.reference_citations import find_reference_citations
from mellea_lrc.extraction.supra_citations import find_supra_citations
from mellea_lrc.model.document import Document
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation import (
    docket_root_lookup_courtlistener_llm_review,
    docket_root_lookup_courtlistener_retrieval,
    docket_root_lookup_govinfo_llm_review,
    docket_root_lookup_govinfo_retrieval,
    intended_case_courtlistener_opinion_retrieval,
    intended_case_courtlistener_recap_retrieval,
    intended_case_govinfo_opinion_retrieval,
    intended_case_llm_selection,
    locator_body_courtlistener_opinion_retrieval,
    locator_body_courtlistener_recap_retrieval,
    locator_body_govinfo_opinion_retrieval,
    locator_body_llm_judgment,
    reporter_root_lookup_ambiguous_llm_judgment,
    reporter_root_lookup_ambiguous_rule_judgment,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_lookup_docket_retrieval,
    reporter_root_lookup_unique_llm_judgment,
    reporter_root_lookup_unique_rule_judgment,
)
from mellea_lrc.workflows import grow_leaves, grow_roots, validate_roots

__all__ = [
    "Document",
    "ExtractionRules",
    "attribute_id_citations",
    "attribute_leaves_rule",
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
    "intended_case_courtlistener_opinion_retrieval",
    "intended_case_courtlistener_recap_retrieval",
    "intended_case_govinfo_opinion_retrieval",
    "intended_case_llm_selection",
    "locator_body_courtlistener_opinion_retrieval",
    "locator_body_courtlistener_recap_retrieval",
    "locator_body_govinfo_opinion_retrieval",
    "locator_body_llm_judgment",
    "preprocess",
    "reporter_root_lookup_ambiguous_llm_judgment",
    "reporter_root_lookup_ambiguous_rule_judgment",
    "reporter_root_lookup_cluster_retrieval",
    "reporter_root_lookup_docket_retrieval",
    "reporter_root_lookup_unique_llm_judgment",
    "reporter_root_lookup_unique_rule_judgment",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_docket_entries",
    "resolve_leaf_case_names",
    "resolve_leaf_pin_cites",
    "resolve_pin_cites",
    "review_id_attributions",
    "review_leaf_attributions",
    "stable",
    "validate_roots",
]
