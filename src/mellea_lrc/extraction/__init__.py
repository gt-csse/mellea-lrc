"""Citation extraction from Layer 2 preprocessed legal text."""

from mellea_lrc.extraction.eyecite import extract, extract_baseline, extract_citations
from mellea_lrc.extraction.pipeline import (
    extract_document_file,
    extract_documents,
    run_extraction,
    run_extraction_from_text,
)
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument

__all__ = [
    "ExtractedCitation",
    "ExtractedDocument",

    "extract",
    "extract_baseline",
    "extract_citations",
    "extract_document_file",
    "extract_documents",
    "run_extraction",
    "run_extraction_from_text",
]
