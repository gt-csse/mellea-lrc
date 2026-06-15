"""Citation extraction from Layer 2 preprocessed legal text."""

from pathlib import Path

from mellea_lrc.extraction.eyecite import extract, extract_baseline, extract_citations
from mellea_lrc.extraction.pipeline import run_extraction, run_extraction_from_text
from mellea_lrc.extraction.protocols import ExtractionAugmenter
from mellea_lrc.extraction.result import DocumentExtraction, ExtractedCitation
from mellea_lrc.preprocessing import preprocess

__all__ = [
    "DocumentExtraction",
    "ExtractedCitation",
    "ExtractionAugmenter",
    "extract",
    "extract_baseline",
    "extract_citations",
    "extract_document_file",
    "extract_documents",
    "run_extraction",
    "run_extraction_from_text",
]


def extract_document_file(path: Path | str) -> DocumentExtraction:
    """Preprocess and extract citations from a document file."""
    preprocessed = preprocess(path)
    return run_extraction(preprocessed)


def extract_documents(directory: Path | str) -> list[DocumentExtraction]:
    """Preprocess and extract citations from every file in a directory."""
    source_dir = Path(directory)
    paths = sorted(source_dir.glob("*.txt"))
    return [extract_document_file(path) for path in paths]
