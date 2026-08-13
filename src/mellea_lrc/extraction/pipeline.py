"""Orchestrate citation extraction."""

from pathlib import Path

from mellea_lrc.extraction.eyecite_extractor import _extract_from_text, extract_from_plain_text
from mellea_lrc.extraction.types import ExtractedDocument
from mellea_lrc.preprocessing import preprocess


def extract_from_raw_document(path: Path) -> ExtractedDocument:
    """Preprocess a document off disk, then extract its citations.

    The backend follows the file's format: plain text is read directly, and
    everything else goes through Docling. Spans index into the *preprocessed*
    text, which for anything but ``.txt`` is not the bytes on disk.
    """
    preprocessed = preprocess(path)
    return _extract_from_text(preprocessed)


def extract(source: str | Path) -> ExtractedDocument:
    """Extract citations from plain text or from a document on disk.

    The argument's type chooses the route: a :class:`str` is content, a
    :class:`~pathlib.Path` is a location. Passing a filename as a string
    extracts from the filename, so reach for :func:`extract_from_raw_document`
    when the path arrives as text.
    """
    if isinstance(source, Path):
        return extract_from_raw_document(source)
    return extract_from_plain_text(source)
