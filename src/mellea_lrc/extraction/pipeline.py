"""Orchestrate citation extraction."""

from pathlib import Path

from mellea_lrc.extraction.eyecite_extractor import _extract_from_text, extract_from_plain_text
from mellea_lrc.extraction.reading.relaxation import Relaxation
from mellea_lrc.extraction.types import ExtractedDocument
from mellea_lrc.preprocessing import preprocess


def extract_from_raw_document(
    path: Path,
    *,
    relaxation: Relaxation = Relaxation.BOUNDED,
) -> ExtractedDocument:
    """Preprocess a document off disk, then extract its citations.

    The backend follows the file's format: plain text is read directly, and
    everything else goes through Docling. Spans index into the *preprocessed*
    text, which for anything but ``.txt`` is not the bytes on disk.
    """
    preprocessed = preprocess(path)
    return _extract_from_text(preprocessed, relaxation=relaxation)


def extract(
    source: str | Path,
    *,
    relaxation: Relaxation = Relaxation.BOUNDED,
) -> ExtractedDocument:
    """Extract citations from plain text or from a document on disk.

    The argument's type chooses the route: a :class:`str` is content, a
    :class:`~pathlib.Path` is a location. Passing a filename as a string
    extracts from the filename, so reach for :func:`extract_from_raw_document`
    when the path arrives as text.

    ``relaxation`` chooses how much separator damage a citation may carry and
    still be found; see :class:`~mellea_lrc.extraction.reading.relaxation.Relaxation`.
    """
    if isinstance(source, Path):
        return extract_from_raw_document(source, relaxation=relaxation)
    return extract_from_plain_text(source, relaxation=relaxation)
