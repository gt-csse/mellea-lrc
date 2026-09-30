"""Outer compositional API for preprocessing, extraction, and validation.

Callers may compose each `Document -> Document` stage explicitly; docket
hunting and docket-root equivalence review are awaitable. `grow_roots` is the
async convenience composition, with both model stages opt-in.
Validation stages remain independently callable; leaf growth is a later layer.
After locator-body validation, `discover_intended_cases` searches other fields
without issuing a positive judgment about the cited locator.
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
from mellea_lrc.workflows.corroborate_root_locator_bodies import corroborate_root_locator_bodies
from mellea_lrc.workflows.discover_intended_cases import discover_intended_cases
from mellea_lrc.workflows.grow_roots import grow_roots
from mellea_lrc.workflows.validate_roots import validate_roots

__all__ = [
    "Document",
    "ExtractionRules",
    "corroborate_root_locator_bodies",
    "discover_intended_cases",
    "docket_root_llm_reassignment",
    "docket_root_lookup_courtlistener_llm_review",
    "docket_root_lookup_courtlistener_retrieval",
    "docket_root_lookup_govinfo_llm_review",
    "docket_root_lookup_govinfo_retrieval",
    "find_docket_locators",
    "find_full_reporter_locators",
    "find_short_reporter_citations",
    "form_roots",
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
    "resolve_pin_cites",
    "stable",
    "validate_roots",
]
