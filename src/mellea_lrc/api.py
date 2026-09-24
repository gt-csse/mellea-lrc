"""Outer compositional API for preprocessing and extraction.

Callers may compose each `Document -> Document` stage explicitly. `grow_roots`
is the convenience composition; validation and leaf growth are later layers.
"""

from mellea_lrc.extraction import (
    ExtractionRules,
    find_docket_locators,
    find_full_reporter_locators,
    find_short_reporter_citations,
    form_roots,
    grow_roots,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
    stable,
)
from mellea_lrc.model.document import Document
from mellea_lrc.preprocessing import preprocess

__all__ = [
    "Document",
    "ExtractionRules",
    "find_docket_locators",
    "find_full_reporter_locators",
    "find_short_reporter_citations",
    "form_roots",
    "grow_roots",
    "preprocess",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_pin_cites",
    "stable",
]
