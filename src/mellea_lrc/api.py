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
from mellea_lrc.validation.roots import full_reporter_locator_identity

__all__ = [
    "Document",
    "ExtractionRules",
    "find_docket_locators",
    "find_full_reporter_locators",
    "form_roots",
    "full_reporter_locator_identity",
    "hunt_docket_locators",
    "mark_full_reporter_locator_hunting_skipped",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_pin_cites",
    "stable",
]
