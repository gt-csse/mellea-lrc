"""Citation extraction from Layer 2 preprocessed legal text.

The existing extraction entrypoints differ in where the text comes from:

``extract``
    The front door. A :class:`str` is content, a :class:`~pathlib.Path` is a
    location, and it dispatches to one of the two below.

``extract_from_plain_text``
    Layer 2 text already in hand.

``extract_from_raw_document``
    A file on disk, preprocessed first by the backend its format calls for.

The native ``find_full_reporter_locators`` and ``find_docket_locators``
substages take an immutable :class:`mellea_lrc.model.Document` and return a new
document with source-grounded locator occurrences and a replayable checkpoint.

The existing extraction entrypoints take the same ``relaxation`` keyword,
which decides how much whitespace damage a citation may carry and still be
found. See :class:`Relaxation` for the available levels.
"""

from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text
from mellea_lrc.extraction.find_docket_locators import find_docket_locators
from mellea_lrc.extraction.find_full_reporter_locators import find_full_reporter_locators
from mellea_lrc.extraction.pipeline import extract, extract_from_raw_document
from mellea_lrc.extraction.relaxation import Relaxation
from mellea_lrc.extraction.resolve_colocations import resolve_colocations
from mellea_lrc.extraction.types import (
    ExtractedCitation,
    ExtractedDocument,
    ExtractionBackend,
    ExtractionMetadata,
)

__all__ = [
    "ExtractedCitation",
    "ExtractedDocument",
    "ExtractionBackend",
    "ExtractionMetadata",
    "Relaxation",
    "extract",
    "extract_from_plain_text",
    "extract_from_raw_document",
    "find_docket_locators",
    "find_full_reporter_locators",
    "resolve_colocations",
]
