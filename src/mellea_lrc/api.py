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
    review_docket_root_equivalence,
)
from mellea_lrc.model.document import Document
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation import (
    courtlistener_opinion_field_body_search,
    courtlistener_opinion_locator_body_search,
    courtlistener_recap_field_body_search,
    courtlistener_recap_locator_body_search,
    docket_root_lookup,
    docket_root_lookup_review,
    govinfo_docket_lookup,
    govinfo_docket_lookup_review,
    govinfo_opinion_field_body_search,
    govinfo_opinion_locator_body_search,
    reporter_root_lookup,
    reporter_root_lookup_ambiguous,
    reporter_root_lookup_ambiguous_llm,
    reporter_root_lookup_unique_llm,
    review_intended_case_body_evidence,
    review_locator_body_evidence,
)
from mellea_lrc.workflows.corroborate_root_locator_bodies import corroborate_root_locator_bodies
from mellea_lrc.workflows.discover_intended_cases import discover_intended_cases
from mellea_lrc.workflows.grow_roots import grow_roots
from mellea_lrc.workflows.validate_roots import validate_roots

__all__ = [
    "Document",
    "ExtractionRules",
    "corroborate_root_locator_bodies",
    "courtlistener_opinion_field_body_search",
    "courtlistener_opinion_locator_body_search",
    "courtlistener_recap_field_body_search",
    "courtlistener_recap_locator_body_search",
    "discover_intended_cases",
    "docket_root_lookup",
    "docket_root_lookup_review",
    "find_docket_locators",
    "find_full_reporter_locators",
    "find_short_reporter_citations",
    "form_roots",
    "govinfo_docket_lookup",
    "govinfo_docket_lookup_review",
    "govinfo_opinion_field_body_search",
    "govinfo_opinion_locator_body_search",
    "grow_roots",
    "hunt_docket_locators",
    "preprocess",
    "reporter_root_lookup",
    "reporter_root_lookup_ambiguous",
    "reporter_root_lookup_ambiguous_llm",
    "reporter_root_lookup_unique_llm",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_docket_entries",
    "resolve_pin_cites",
    "review_docket_root_equivalence",
    "review_intended_case_body_evidence",
    "review_locator_body_evidence",
    "stable",
    "validate_roots",
]
