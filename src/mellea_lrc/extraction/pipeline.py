"""Orchestrate citation extraction."""

from pathlib import Path

from mellea_lrc.extraction.eyecite import extract_baseline
from mellea_lrc.extraction.types import ExtractedDocument
from mellea_lrc.preprocessing import run_preprocessing
from mellea_lrc.preprocessing.plain_text import preprocess_plain_text_from_string
from mellea_lrc.preprocessing.types import PreprocessedDocument


def run_extraction(
    preprocessed: PreprocessedDocument,
) -> ExtractedDocument:
    """Run extraction on a preprocessed document."""
    return extract_baseline(preprocessed)


def run_extraction_from_text(
    text: str,
    *,
    source_path: str | None = None,
) -> ExtractedDocument:
    """Run the extraction pipeline on raw Layer 2 text."""
    preprocessed = preprocess_plain_text_from_string(text, source_path=source_path)
    return run_extraction(preprocessed)


def extract_document_file(path: Path | str) -> ExtractedDocument:
    """Preprocess and extract citations from a document file."""
    preprocessed = run_preprocessing(path)
    return run_extraction(preprocessed)


def extract_documents(directory: Path | str) -> list[ExtractedDocument]:
    """Preprocess and extract citations from every text file in a directory."""
    source_dir = Path(directory)
    paths = sorted(source_dir.glob("*.txt"))
    return [extract_document_file(path) for path in paths]
