"""Load pre-exported plain text files into canonical preprocessing types."""

from pathlib import Path

from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
)

_PLAIN_TEXT_MARKER = "--- Plain text ---"


def split_plain_text_file(text: str) -> tuple[str, str]:
    """Split a RECAP-style export into metadata header and body text."""
    if _PLAIN_TEXT_MARKER not in text:
        return "", text

    header, body = text.split(f"{_PLAIN_TEXT_MARKER}\n", maxsplit=1)
    return header.strip(), body


def preprocess_plain_text(path: Path | str) -> PreprocessedDocument:
    """Load a `.txt` file as a preprocessed document."""
    source_path = Path(path)
    raw = source_path.read_text(encoding="utf-8")
    header, body = split_plain_text_file(raw)

    return PreprocessedDocument(
        text=body,
        source_metadata=SourceMetadata(
            path=str(source_path),
            format=SourceFormat.TEXT,
            header=header or None,
        ),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
    )


def preprocess_plain_text_from_string(
    text: str,
    *,
    source_path: str | None = None,
) -> PreprocessedDocument:
    """Wrap raw text in a preprocessed document without reading a file."""
    header, body = split_plain_text_file(text)
    return PreprocessedDocument(
        text=body,
        source_metadata=SourceMetadata(
            path=source_path,
            format=SourceFormat.TEXT,
            header=header or None,
        ),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
    )
