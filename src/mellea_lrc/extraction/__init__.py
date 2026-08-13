"""Citation extraction from Layer 2 preprocessed legal text.

Three entrypoints, differing only in where the text comes from:

``extract``
    The front door. A :class:`str` is content, a :class:`~pathlib.Path` is a
    location, and it dispatches to one of the two below.

``extract_from_plain_text``
    Layer 2 text already in hand.

``extract_from_raw_document``
    A file on disk, preprocessed first by the backend its format calls for.

There is deliberately no entrypoint taking a ``PreprocessedDocument``. Nothing
serializes one, so it cannot cross a process boundary, and a caller holding one
is already inside the library.
"""

from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text
from mellea_lrc.extraction.pipeline import extract, extract_from_raw_document
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
    "extract",
    "extract_from_plain_text",
    "extract_from_raw_document",
]
