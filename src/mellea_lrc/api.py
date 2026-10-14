"""Outer compositional API for preprocessing, extraction, and validation.

Callers may compose workflows, their semantic stages, or individual
`Document -> Document` substages. Docket hunting and docket-root equivalence
review are awaitable and opt-in in `grow_roots`. All three levels have explicit
checkpoint boundaries; substages remain independently callable. `grow_leaves` attaches short
forms and repeated occurrences to formed roots, with optional model review.
The workflows are `grow_roots`, `validate_roots_citation_identity`, `grow_leaves`, and `validate_all_citation_content`.
Locator-body and intended-case review are parts of root validation.
"""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction import (
    attribute_reference_citations,
    attribute_short_reporter_citations,
    find_docket_locators,
    find_full_reporter_locators,
    find_short_reporter_citations,
    form_roots,
    hunt_docket_locators,
    reassign_docket_roots_llm,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_docket_entries,
    resolve_pin_cites,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
)
from mellea_lrc.extraction.attribute_id_citations import attribute_id_citations
from mellea_lrc.extraction.attribute_supra_citations_rule import attribute_supra_citations_rule
from mellea_lrc.extraction.correct_leaf_fields import correct_leaf_fields
from mellea_lrc.extraction.find_id_citations import find_id_citations
from mellea_lrc.extraction.find_reference_citations import find_reference_citations
from mellea_lrc.extraction.find_supra_citations import find_supra_citations
from mellea_lrc.extraction.resolve_supra_case_names import resolve_supra_case_names
from mellea_lrc.extraction.resolve_supra_pin_cites import resolve_supra_pin_cites
from mellea_lrc.extraction.review_supra_attributions_llm import review_supra_attributions_llm
from mellea_lrc.model.document import Document
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation import (
    index_reporter_root_opinion_pages,
    judge_ambiguous_reporter_roots_rule,
    judge_reporter_citation_content,
    judge_root_identity_from_fields,
    judge_unique_reporter_roots_rule,
    prepare_reporter_citation_evidence,
    read_reporter_citation_propositions,
    resolve_reporter_citation_pages,
    retrieve_courtlistener_docket_roots,
    retrieve_courtlistener_intended_case_filings,
    retrieve_courtlistener_intended_case_opinions,
    retrieve_courtlistener_locator_filings,
    retrieve_courtlistener_locator_opinions,
    retrieve_govinfo_docket_roots,
    retrieve_govinfo_intended_case_opinions,
    retrieve_govinfo_locator_opinions,
    retrieve_reporter_root_clusters,
    retrieve_reporter_root_dockets,
    retrieve_reporter_root_opinions,
    review_ambiguous_reporter_roots_llm,
    review_courtlistener_docket_roots_llm,
    review_govinfo_docket_roots_llm,
    review_intended_cases_llm,
    review_locator_bodies_llm,
    review_reporter_citation_full_opinions_llm,
    review_reporter_citation_pages_llm,
    review_unique_reporter_roots_llm,
    select_reporter_citation_opinions,
)
from mellea_lrc.workflows import (
    grow_leaves,
    grow_roots,
    validate_all_citation_content,
    validate_roots_citation_identity,
)
from mellea_lrc.workflows.grow_leaves import (
    correct_leaf_readings,
    grow_id_leaves,
    grow_reference_leaves,
    grow_short_reporter_leaves,
    grow_supra_leaves,
)
from mellea_lrc.workflows.grow_roots import discover_root_locators, form_root_groups, read_root_fields
from mellea_lrc.workflows.validate_all_citation_content import (
    prepare_citation_evidence,
    prepare_root_opinions,
    review_citation_content,
)
from mellea_lrc.workflows.validate_roots_citation_identity import (
    corroborate_locator_bodies,
    discover_intended_cases,
    lookup_docket_roots,
    lookup_reporter_roots,
)

__all__ = [
    "Document",
    "ExtractionRules",
    "attribute_id_citations",
    "attribute_reference_citations",
    "attribute_short_reporter_citations",
    "attribute_supra_citations_rule",
    "correct_leaf_fields",
    "correct_leaf_readings",
    "corroborate_locator_bodies",
    "discover_intended_cases",
    "discover_root_locators",
    "find_docket_locators",
    "find_full_reporter_locators",
    "find_id_citations",
    "find_reference_citations",
    "find_short_reporter_citations",
    "find_supra_citations",
    "form_root_groups",
    "form_roots",
    "grow_id_leaves",
    "grow_leaves",
    "grow_reference_leaves",
    "grow_roots",
    "grow_short_reporter_leaves",
    "grow_supra_leaves",
    "hunt_docket_locators",
    "index_reporter_root_opinion_pages",
    "judge_ambiguous_reporter_roots_rule",
    "judge_reporter_citation_content",
    "judge_root_identity_from_fields",
    "judge_unique_reporter_roots_rule",
    "lookup_docket_roots",
    "lookup_reporter_roots",
    "prepare_citation_evidence",
    "prepare_reporter_citation_evidence",
    "prepare_root_opinions",
    "preprocess",
    "read_reporter_citation_propositions",
    "read_root_fields",
    "reassign_docket_roots_llm",
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
    "retrieve_courtlistener_docket_roots",
    "retrieve_courtlistener_intended_case_filings",
    "retrieve_courtlistener_intended_case_opinions",
    "retrieve_courtlistener_locator_filings",
    "retrieve_courtlistener_locator_opinions",
    "retrieve_govinfo_docket_roots",
    "retrieve_govinfo_intended_case_opinions",
    "retrieve_govinfo_locator_opinions",
    "retrieve_reporter_root_clusters",
    "retrieve_reporter_root_dockets",
    "retrieve_reporter_root_opinions",
    "review_ambiguous_reporter_roots_llm",
    "review_citation_content",
    "review_courtlistener_docket_roots_llm",
    "review_govinfo_docket_roots_llm",
    "review_intended_cases_llm",
    "review_locator_bodies_llm",
    "review_reporter_citation_full_opinions_llm",
    "review_reporter_citation_pages_llm",
    "review_supra_attributions_llm",
    "review_unique_reporter_roots_llm",
    "select_reporter_citation_opinions",
    "stable",
    "validate_all_citation_content",
    "validate_roots_citation_identity",
]
