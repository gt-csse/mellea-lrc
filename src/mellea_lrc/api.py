"""Outer compositional API for preprocessing, extraction, and validation.

Callers may compose each `Document -> Document` stage explicitly; the docket
hunting stage is awaitable. `grow_roots` is the async convenience composition.
Validation stages remain independently callable; leaf growth is a later layer.
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
)
from mellea_lrc.model.document import Document
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation import reporter_root_exact_ambiguity, reporter_root_exact_lookup
from mellea_lrc.workflows.grow_roots import grow_roots

__all__ = [
    "Document",
    "ExtractionRules",
    "find_docket_locators",
    "find_full_reporter_locators",
    "find_short_reporter_citations",
    "form_roots",
    "grow_roots",
    "hunt_docket_locators",
    "preprocess",
    "reporter_root_exact_ambiguity",
    "reporter_root_exact_lookup",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_docket_entries",
    "resolve_pin_cites",
    "stable",
]
