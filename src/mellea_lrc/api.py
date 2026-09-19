"""Outer compositional API for Mellea-LRC stages.

This module is the stable boundary for callers composing individual stages.
It intentionally does not choose an end-to-end pipeline: callers begin with a
``Document`` or a serialized document, then explicitly invoke the
``Document -> Document`` stage they want. The command-line interface is the
separate end-to-end entrypoint.
"""

from __future__ import annotations

from mellea_lrc.extraction.adjudication import hunt_docket_locators
from mellea_lrc.extraction.locator_stages import (
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_colocations,
)
from mellea_lrc.extraction.root_stages import form_roots
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.stages import (
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.extraction.types import Document
from mellea_lrc.validation.docket_roots import (
    resolve_docket_root_ambiguities,
    search_docket_roots,
    validate_unique_docket_root_identities,
)
from mellea_lrc.validation.roots import (
    lookup_full_reporter_locators_exact,
    resolve_full_reporter_locator_ambiguities,
    validate_unique_full_reporter_locator_identities,
)

__all__ = [
    "Document",
    "ExtractionRules",
    "find_docket_locators",
    "find_full_reporter_locators",
    "form_roots",
    "hunt_docket_locators",
    "lookup_full_reporter_locators_exact",
    "mark_full_reporter_locator_hunting_skipped",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_docket_root_ambiguities",
    "resolve_full_reporter_locator_ambiguities",
    "resolve_pin_cites",
    "search_docket_roots",
    "stable",
    "validate_unique_docket_root_identities",
    "validate_unique_full_reporter_locator_identities",
]
